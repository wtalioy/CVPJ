import os
import math
from typing import Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FerretV4", "build_ferret_v4"]

# ============================================================
# FerretV4 模型配置参数 (可根据超参搜索结果修改)
# ============================================================
DIM = 64               # 主干维度基数: 32, 64, 128
DROPOUT = 0.5          # 分类器 Dropout 比例: 0.3, 0.5, 0.7
# ============================================================


class ConstrainedConv(nn.Module):
    """
    受限卷积层：通过强制学习残差特征（高通行为）
    软约束实现：
      - kernel 中心权重为负：center = -softplus(center_raw)
      - 四周权重为非负：surround = softplus(surround_raw)
      - 归一化使 kernel 总和为 0：sum(surround_scaled) + center = 0
    这样形成稳定的高通滤波核，有利于去除内容语义、强化痕迹特征。
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, dilation=1):
        super().__init__()
        assert kernel_size % 2 == 1, "建议奇数 kernel 以明确中心位置"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # 原始可学习参数（未受限）
        self.weight_raw = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        # 可选偏置（严格高通通常不需要）
        self.bias = None

        self.eps = 1e-6
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming 初始化，随后会被软约束重投影
        nn.init.kaiming_normal_(self.weight_raw, mode="fan_out", nonlinearity="relu")

    def _constrain_kernel(self, w: torch.Tensor) -> torch.Tensor:
        """
        将原始权重 w 投影为受限权重：
          - 中心负、四周非负、总和为 0（典型高通核）
        """
        B, C, H, W = w.shape  # (out_c, in_c, k, k)
        ch, cw = H // 2, W // 2

        # 拆分中心与四周
        center_raw = w[..., ch, cw]                          # (out_c, in_c)
        surround_raw = w.clone()
        surround_raw[..., ch, cw] = 0.0

        # 软约束：中心负、四周非负
        center_neg = -F.softplus(center_raw)                 # <= 0
        surround_pos = F.softplus(surround_raw)              # >= 0

        # 将四周缩放，使得 sum(surround_scaled) = |center_neg|
        s = surround_pos.sum(dim=(2, 3)) + self.eps          # (out_c, in_c)
        scale = center_neg.abs() / s                         # (out_c, in_c)
        surround_scaled = surround_pos * scale.unsqueeze(-1).unsqueeze(-1)

        # 组装受限核：总和恰为 0
        w_constrained = surround_scaled
        w_constrained[..., ch, cw] = center_neg

        return w_constrained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._constrain_kernel(self.weight_raw)
        return F.conv2d(
            x, w, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation,
            groups=1
        )


class SpectralBottleneck(nn.Module):
    """
    频谱瓶颈模块：通过极低维度的频谱统计避免语义泄露
    输出维度 = C + 4
      - 低频：对 |FFT| 取 [0:2, 0:2] 的能量，先在通道上取均值 -> 4 个标量（若空间不足 2 则按实际长度）
      - 高频：去除上述低频块后，对剩余频率在空间维度上求均值 -> 每通道 1 个标量，共 C 个
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(in_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 轻量归一，抑制幅度尺度差异
        x = self.bn(x)

        # 保存原始 dtype，FFT 需要 float32（cuFFT 半精度只支持 2 的幂次维度）
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()

        # rfft2: 最后一维长度为 floor(W/2)+1
        fft = torch.fft.rfft2(x, norm="forward")
        mag = torch.abs(fft)

        # 低频 2x2（若尺寸不足 2 则取可用范围）
        h_lim = min(2, mag.size(2))
        w_lim = min(2, mag.size(3))
        low = mag[:, :, :h_lim, :w_lim]                      # (B, C, h_lim, w_lim)

        # 低频跨通道聚合，得到极少量全局统计
        low_feat = low.mean(dim=1).reshape(x.size(0), -1)    # (B, h_lim*w_lim) ~ (B, 4)

        # 高频：去除低频小块后，对空间维度求均值，保留每通道统计
        high = mag.clone()
        high[:, :, :h_lim, :w_lim] = 0.0
        high_feat = high.mean(dim=(2, 3))                    # (B, C)

        result = torch.cat([low_feat, high_feat], dim=1)     # (B, C + (h_lim*w_lim))
        
        # 恢复原始 dtype
        if input_dtype == torch.float16:
            result = result.half()
        
        return result


def _make_stage(in_c: int, out_c: int, depth: int) -> nn.Sequential:
    layers = []
    for i in range(depth):
        layers.append(nn.Conv2d(in_c if i == 0 else out_c, out_c, 3, padding=1, stride=2 if i == 0 else 1, bias=False))
        layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class FerretV4(nn.Module):
    """
    FerretV4
      - 残差痕迹提取 -> 窄而深的局部痕迹 backbone -> 空间 GAP/频谱瓶颈融合 -> 分类
    """
    def __init__(self, num_classes: int = 2, dim: int = None, in_channels: int = 3, dropout: float = None):
        super().__init__()
        
        # 使用配置常量作为默认值
        if dim is None:
            dim = DIM
        if dropout is None:
            dropout = DROPOUT

        # 1) 残差提取层（去除内容语义，聚焦痕迹）
        self.res_extractor = nn.Sequential(
            ConstrainedConv(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 2) 局部痕迹 Backbone
        self.stage1 = _make_stage(32, dim,   depth=2)   # 下采样 /2
        self.stage2 = _make_stage(dim, dim*2, depth=2)  # 下采样 /4
        self.stage3 = _make_stage(dim*2, dim*4, depth=2)# 下采样 /8

        # 3) 空间与频谱分支
        self.spec_gap = nn.AdaptiveAvgPool2d((1, 1))
        c3 = dim * 4
        self.spec_stat = SpectralBottleneck(c3)

        # 4) 分类头
        # 空间 feat 维度 = c3
        # 频谱 feat 维度 = c3 + 4（当特征图尺寸>=2x2时；通常成立）
        # 融合维度 = 2*c3 + 4
        in_features = 2 * c3 + 4
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        # 统一初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差痕迹
        res = self.res_extractor(x)

        # 局部痕迹 backbone
        x1 = self.stage1(res)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        # 空间特征（每通道全局平均）
        spat_feat = self.spec_gap(x3).view(x3.size(0), -1)   # (B, c3)

        # 频谱特征（C + 4）
        spec_feat = self.spec_stat(x3)
        # 保守处理：当特征图空间过小导致低频块 < 2x2 时，补零到 4 维，确保 (C + 4)
        c3 = x3.size(1)
        low_needed = 4
        low_current = spec_feat.size(1) - c3
        if low_current < low_needed:
            pad = torch.zeros(spec_feat.size(0), low_needed - low_current, device=spec_feat.device, dtype=spec_feat.dtype)
            spec_feat = torch.cat([pad, spec_feat], dim=1)

        # 融合
        feat = torch.cat([spat_feat, spec_feat], dim=1)      # 维度 = 2*c3 + 4
        return self.classifier(feat)


# 构建函数，供 utils.build_model 使用

def build_ferret_v4(
    num_classes: int = 2,
    *,
    in_channels: int = 3,
    dim: int = None,
    dropout: float = None,
    checkpoint: Optional[str] = None,
    map_location: str = "cpu",
) -> nn.Module:
    """
    构建 FerretV4 模型

    Args:
        num_classes: 分类数
        in_channels: 输入通道数
        dim: 主干维度基数 (默认使用文件顶部 DIM 常量)
        dropout: 分类器 Dropout (默认使用文件顶部 DROPOUT 常量)
        checkpoint: 可选权重路径（支持包含 'model_state' 的 dict 或纯 state_dict）
        map_location: 加载设备
    """
    model = FerretV4(num_classes=num_classes, dim=dim, in_channels=in_channels, dropout=dropout)

    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=map_location)
        state_dict = state.get("model_state", state)
        model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == "__main__":
    # 简单自测
    model = build_ferret_v4(num_classes=2, dim=64, in_channels=3)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)

