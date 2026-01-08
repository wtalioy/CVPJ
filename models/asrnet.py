import os
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ASRNet", "build_asrnet"]


class ConstrainedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, dilation=1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight_raw = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = None
        self.eps = 1e-6
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.weight_raw, mode="fan_out", nonlinearity="relu")

    def _constrain_kernel(self, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = w.shape  # (out_c, in_c, k, k)
        ch, cw = H // 2, W // 2

        center_raw = w[..., ch, cw]                          # (out_c, in_c)
        surround_raw = w.clone()
        surround_raw[..., ch, cw] = 0.0

        center_neg = -F.softplus(center_raw)                 # <= 0
        surround_pos = F.softplus(surround_raw)              # >= 0

        s = surround_pos.sum(dim=(2, 3)) + self.eps          # (out_c, in_c)
        scale = center_neg.abs() / s                         # (out_c, in_c)
        surround_scaled = surround_pos * scale.unsqueeze(-1).unsqueeze(-1)

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
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(in_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()

        fft = torch.fft.rfft2(x, norm="forward")
        mag = torch.abs(fft)

        h_lim = min(2, mag.size(2))
        w_lim = min(2, mag.size(3))
        low = mag[:, :, :h_lim, :w_lim]                      # (B, C, h_lim, w_lim)

        low_feat = low.mean(dim=1).reshape(x.size(0), -1)    # (B, h_lim*w_lim) ~ (B, 4)

        high = mag.clone()
        high[:, :, :h_lim, :w_lim] = 0.0
        high_feat = high.mean(dim=(2, 3))                    # (B, C)

        result = torch.cat([low_feat, high_feat], dim=1)     # (B, C + (h_lim*w_lim))
        
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


class PrototypeDistanceHead(nn.Module):
    def __init__(self, in_features, num_sub_centers=3):
        super().__init__()
        self.real_center = nn.Parameter(torch.randn(1, in_features))
        self.fake_centers = nn.Parameter(torch.randn(num_sub_centers, in_features))
        
        self.softplus = nn.Softplus()

    def forward(self, x):
        dist_real = torch.cdist(x.unsqueeze(1), self.real_center.unsqueeze(0)).squeeze(1)
        dists_fake = torch.cdist(x.unsqueeze(1), self.fake_centers.unsqueeze(0)).squeeze(1)
        min_dist_fake, _ = torch.min(dists_fake, dim=1, keepdim=True)

        logits = torch.cat([-dist_real, -min_dist_fake], dim=1)
        return logits
        

class ASRNet(nn.Module):
    def __init__(self, num_classes: int = 2, dim: int = 64, in_channels: int = 3, dropout: float = 0.5):
        super().__init__()
        
        self.res_extractor = nn.Sequential(
            ConstrainedConv(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.stage1 = _make_stage(32, dim, depth=2)
        self.stage2 = _make_stage(dim, dim*2, depth=2)
        self.stage3 = _make_stage(dim*2, dim*4, depth=2)

        self.spec_gap = nn.AdaptiveAvgPool2d((1, 1))
        c3 = dim * 4
        self.spec_stat = SpectralBottleneck(c3)

        in_features = 2 * c3 + 4
        self.classifier = PrototypeDistanceHead(in_features, num_sub_centers=5)

        self._init_weights()

    def _init_weights(self):
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
        res = self.res_extractor(x)
        x1 = self.stage1(res)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        spat_feat = self.spec_gap(x3).view(x3.size(0), -1)   # (B, c3)
        spec_feat = self.spec_stat(x3)
        c3 = x3.size(1)
        low_needed = 4
        low_current = spec_feat.size(1) - c3
        if low_current < low_needed:
            pad = torch.zeros(spec_feat.size(0), low_needed - low_current, device=spec_feat.device, dtype=spec_feat.dtype)
            spec_feat = torch.cat([pad, spec_feat], dim=1)

        feat = torch.cat([spat_feat, spec_feat], dim=1)
        return self.classifier(feat)


def build_asrnet(
    num_classes: int = 2,
    *,
    in_channels: int = 3,
    dim: int = 64,
    dropout: float = 0.5,
    checkpoint: Optional[str] = None,
    map_location: str = "cpu",
) -> nn.Module:
    model = ASRNet(num_classes=num_classes, dim=dim, in_channels=in_channels, dropout=dropout)

    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=map_location)
        state_dict = state.get("model_state", state)
        model.load_state_dict(state_dict, strict=False)

    return model

