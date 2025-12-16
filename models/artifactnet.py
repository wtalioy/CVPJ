import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_artifactnet", "ArtifactNet"]


class MultiScaleResidualBlock(nn.Module):
    """多尺度残差块，捕捉不同尺度的伪影特征"""
    def __init__(self, in_channels, out_channels, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        
        # 1x1卷积降维
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # 多尺度卷积组
        self.conv3x3 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(mid_channels, mid_channels, 5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(mid_channels, mid_channels, 7, padding=3, bias=False)
        
        # 深度可分离卷积
        self.depthwise = nn.Conv2d(mid_channels * 3, mid_channels * 3, 3, padding=1, 
                                  groups=mid_channels * 3, bias=False)
        self.pointwise = nn.Conv2d(mid_channels * 3, mid_channels, 1, bias=False)
        
        # 输出层
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv_out = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        identity = x
        
        # 降维
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 多尺度特征提取
        x1 = self.relu(self.conv3x3(x))
        x2 = self.relu(self.conv5x5(x))
        x3 = self.relu(self.conv7x7(x))
        
        # 特征融合
        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_fused = self.depthwise(x_cat)
        x_fused = self.pointwise(x_fused)
        x_fused = self.bn2(x_fused)
        x_fused = self.relu(x_fused)
        x_fused = self.dropout(x_fused)
        
        # 输出
        out = self.conv_out(x_fused)
        out = self.bn_out(out)
        
        # 残差连接
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class CrossChannelAttention(nn.Module):
    """修正后的跨通道注意力机制"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        
        # 应用通道注意力
        x_attended = x * channel_att
        
        return x_attended


class FrequencyAwareBlock(nn.Module):
    """频率感知模块，检测频域异常"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 频域分析分支
        self.freq_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )
        
        # 空间分支
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 空间特征
        spatial_feat = self.spatial_branch(x)
        
        # 计算梯度作为高频特征近似
        # 使用Sobel算子检测边缘（高频）
        sobel_kernel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
            dtype=x.dtype, device=x.device
        ).repeat(self.channels, 1, 1, 1)
        
        sobel_kernel_y = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
            dtype=x.dtype, device=x.device
        ).repeat(self.channels, 1, 1, 1)
        
        # 计算梯度
        grad_x = F.conv2d(x, sobel_kernel_x, padding=1, groups=self.channels)
        grad_y = F.conv2d(x, sobel_kernel_y, padding=1, groups=self.channels)
        
        # 梯度幅度（高频特征）
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
        # 处理高频特征
        freq_feat = self.freq_branch(grad_magnitude)
        
        # 特征融合
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        output = self.fusion(combined)
        
        # 残差连接
        return output + x


class ArtifactNet(nn.Module):
    """专门检测AI生成图像伪影的网络"""
    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 32,
        blocks: list = [1, 2, 2, 1],  # 简化各阶段块数，避免过深
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        # 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 四个阶段
        self.stage1 = self._make_stage(base_channels * 2, base_channels * 4, blocks[0])
        self.stage2 = self._make_stage(base_channels * 4, base_channels * 8, blocks[1])
        self.stage3 = self._make_stage(base_channels * 8, base_channels * 16, blocks[2])
        self.stage4 = self._make_stage(base_channels * 16, base_channels * 32, blocks[3])
        
        # 下采样
        self.downsample1 = nn.MaxPool2d(2, 2)
        self.downsample2 = nn.MaxPool2d(2, 2)
        self.downsample3 = nn.MaxPool2d(2, 2)
        
        # 全局特征融合 - 修正通道数计算
        # 阶段输出通道数: 64, 128, 256, 512, 1024
        # 但经过上采样融合后，总通道数应该是: 64+128+256+512+1024 = 1984
        self.global_fusion = nn.Sequential(
            nn.Conv2d(1984, base_channels * 32, 1, bias=False),  # 1984 -> 1024
            nn.BatchNorm2d(base_channels * 32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 32, base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 16, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks):
        """创建网络阶段"""
        blocks = []
        
        # 第一个块处理通道数变化
        blocks.append(MultiScaleResidualBlock(in_channels, out_channels))
        
        # 后续块保持通道数不变
        for i in range(1, num_blocks):
            blocks.append(MultiScaleResidualBlock(out_channels, out_channels))
            # 每隔一个块添加频率感知模块
            if i % 2 == 0:
                blocks.append(FrequencyAwareBlock(out_channels))
            # 每个块后都添加通道注意力
            blocks.append(CrossChannelAttention(out_channels))
        
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始特征提取
        x0 = self.stem(x)  # [B, 64, H/4, W/4]
        
        # 多阶段特征提取
        x1 = self.stage1(x0)  # [B, 128, H/4, W/4]
        x1_pool = self.downsample1(x1)  # [B, 128, H/8, W/8]
        
        x2 = self.stage2(x1_pool)  # [B, 256, H/8, W/8]
        x2_pool = self.downsample2(x2)  # [B, 256, H/16, W/16]
        
        x3 = self.stage3(x2_pool)  # [B, 512, H/16, W/16]
        x3_pool = self.downsample3(x3)  # [B, 512, H/32, W/32]
        
        x4 = self.stage4(x3_pool)  # [B, 1024, H/32, W/32]
        
        # 多尺度特征融合
        # 上采样并融合不同尺度的特征
        target_size = x4.shape[2:]  # (H/32, W/32)
        
        x0_up = F.interpolate(x0, size=target_size, mode='bilinear', align_corners=False)
        x1_up = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=False)
        x2_up = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        x3_up = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)
        
        # 特征拼接（通道维度）
        features = torch.cat([x0_up, x1_up, x2_up, x3_up, x4], dim=1)
        
        # 全局融合
        fused = self.global_fusion(features)
        
        # 分类
        output = self.classifier(fused)
        
        return output


def build_artifactnet(
    num_classes: int = 2,
    base_channels: int = 32,
    blocks: list = None,
    checkpoint: Optional[str] = None,
    map_location: str = "cpu",
) -> nn.Module:
    """
    创建ArtifactNet模型并可选加载检查点
    """
    if blocks is None:
        blocks = [1, 2, 2, 1]  # 简化结构，确保可以训练
    
    model = ArtifactNet(
        num_classes=num_classes,
        base_channels=base_channels,
        blocks=blocks,
    )

    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=map_location)
        state_dict = state.get("model_state", state)
        model.load_state_dict(state_dict, strict=False)

    return model