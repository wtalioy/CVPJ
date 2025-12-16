"""
FerretNet with DDFC (Dual-Domain Feature Coupler)
Deepfake detection network combining spatial and frequency domain features
"""
import os
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_ferretnet_ddfc", "FerretDDFC"]


# ==========================================
# 1. Basic modules (from original FerretNet)
# ==========================================
class ExclusionMedianValues(nn.Module):
    """Median filtering excluding center point, used for extracting local texture residuals"""
    def __init__(self, kernel_size: int, stride: int = 1, padding: Optional[int] = None):
        super().__init__()
        self.k = kernel_size
        self.s = stride
        self.p = padding if padding is not None else kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        center_idx = (self.k * self.k) // 2
        padded = F.pad(x, (self.p, self.p, self.p, self.p), mode="constant", value=0.0)
        unfolded = F.unfold(padded, kernel_size=(self.k, self.k), stride=self.s)
        unfolded = unfolded.view(n, c, self.k * self.k, -1)
        mask = torch.arange(self.k * self.k, device=x.device) != center_idx
        unfolded = unfolded[:, :, mask, :]
        medians = unfolded.median(dim=2).values
        h_out = (h + 2 * self.p - self.k) // self.s + 1
        w_out = (w + 2 * self.p - self.k) // self.s + 1
        return medians.view(n, c, h_out, w_out)


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.depth_wise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, 
            padding=padding, groups=in_channels, bias=False
        )
        self.point_wise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


class DilatedConv2d(nn.Module):
    """Multi-scale dilated convolution"""
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, 
                               groups=in_channels, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride, 2, 
                               groups=in_channels, dilation=2, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.conv1(x) + self.conv2(x))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        x = x * channel_out
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_out = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_out

class DSBlock(nn.Module):
    """Original dilated + separable convolution block"""
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            DilatedConv2d(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.layers(x) + self.shortcut(x))


# ==========================================
# 2. Frequency domain modules 
# ==========================================
class FourierUnit(nn.Module):
    """
    Frequency domain feature extraction unit
    Performs convolution operations in frequency domain to capture global spectral patterns
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2,
                              kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        
        B, C, H_pad, W_pad = x.size()
        
        ffted = torch.fft.rfft2(x, norm="ortho")
        ffted = torch.view_as_real(ffted)  # [B, C, H, W//2+1, 2]
        
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(B, C * 2, H_pad, W_pad // 2 + 1)
        
        ffted = self.relu(self.bn(self.conv(ffted)))
        
        out_channels = ffted.size(1) // 2
        ffted = ffted.view(B, out_channels, 2, H_pad, W_pad // 2 + 1)
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)
        
        output = torch.fft.irfft2(ffted, s=(H_pad, W_pad), norm="ortho")
        
        output = output[:, :, :H, :W]
        return output


class SpectralBranch(nn.Module):
    """
    Spectral branch: includes local and global frequency domain feature extraction
    Inspired by SpecXNet's LFU (Local Fourier Unit) design
    """
    def __init__(self, in_channels: int, out_channels: int, enable_lfu: bool = True):
        super().__init__()
        self.enable_lfu = enable_lfu
        
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.gfu = FourierUnit(out_channels, out_channels)
        
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels, out_channels)
        
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)
        target_size = x.shape[2:]
        
        gfu_out = self.gfu(x)
        
        if self.enable_lfu:
            pooled = F.adaptive_avg_pool2d(x, (max(2, x.shape[2] // 4), max(2, x.shape[3] // 4)))
            lfu_out = self.lfu(pooled)
            lfu_out = F.interpolate(lfu_out, size=target_size, mode="bilinear", align_corners=False)
        else:
            lfu_out = 0
        
        out = self.post_conv(x + gfu_out + lfu_out)
        return out


# ==========================================
# 3. DDFC fusion modules
# ==========================================
class CrossDomainAttention(nn.Module):
    """
    Cross-domain attention: enables spatial and frequency domain features to enhance each other
    Spatial features guide the focus areas of frequency features, frequency features enhance global perception of spatial features
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        mid_channels = max(channels // reduction, 8)
        
        self.spat_to_spec = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.spec_to_spat = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_spat: torch.Tensor, x_spec: torch.Tensor):
        attn_for_spec = self.spat_to_spec(x_spat)
        attn_for_spat = self.spec_to_spat(x_spec)
        
        x_spat_enhanced = x_spat + x_spat * attn_for_spat
        x_spec_enhanced = x_spec + x_spec * attn_for_spec
        
        return x_spat_enhanced, x_spec_enhanced


class GatedFusion(nn.Module):
    """
    Gated fusion: adaptively fuses spatial and frequency domain features
    Learns which domain features to trust more at each position
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_spat: torch.Tensor, x_spec: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_spat, x_spec], dim=1)
        gate = self.gate(combined)
        return gate * x_spat + (1 - gate) * x_spec


class DDFC_DSBlock(nn.Module):
    """
    DSBlock integrated with DDFC mechanism
    Dual-branch parallel processing: spatial domain uses original DSBlock, frequency domain uses SpectralBranch
    Achieves feature complementarity through cross-domain attention and gated fusion
    """
    def __init__(self, in_channels: int, out_channels: int, enable_lfu: bool = True):
        super().__init__()
        
        self.spatial_branch = nn.Sequential(
            DilatedConv2d(in_channels, out_channels, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        
        self.spectral_branch = SpectralBranch(in_channels, out_channels, enable_lfu=enable_lfu)
        
        self.cross_attn = CrossDomainAttention(out_channels)
        
        self.fusion = GatedFusion(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_spat = self.spatial_branch(x)
        out_spec = self.spectral_branch(x)
        
        out_spat, out_spec = self.cross_attn(out_spat, out_spec)
        
        out = self.fusion(out_spat, out_spec)
        
        out = self.relu(out + self.shortcut(x))
        return out


# ==========================================
# 4. Main network: FerretDDFC
# ==========================================
class FerretDDFC(nn.Module):
    """
    FerretNet with DDFC (Dual-Domain Feature Coupler)

    Dual-domain feature coupling network combining spatial and frequency domains.
    - Downsampling layers use original DSBlock (for stability)
    - Feature extraction layers use DDFC_DSBlock (dual-domain features)
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dim: int,
        depths: Sequence[int],
        window_size: int,
        enable_lfu: bool = True,
        use_ddfc: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_ddfc = use_ddfc

        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, self.dim // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim // 2),
            nn.ReLU(inplace=True),
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d(self.dim // 2, self.dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )

        self.feature = nn.Sequential()
        current_dim = self.dim

        for depth in depths:
            layers = []
            
            layers.append(DSBlock(current_dim, current_dim * 2, stride=2))
            current_dim *= 2
            
            for _ in range(max(depth - 1, 0)):
                if self.use_ddfc:
                    layers.append(DDFC_DSBlock(current_dim, current_dim, enable_lfu=enable_lfu))
                else:
                    layers.append(DSBlock(current_dim, current_dim, stride=1))
            
            self.feature.append(nn.Sequential(*layers))

        self.feature.append(nn.Conv2d(current_dim, current_dim, 1, 1, bias=False))

        self.cbam = CBAM(channels=current_dim, reduction=16)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.logit = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(current_dim, num_classes),
        )

        self.lpd = ExclusionMedianValues(window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.lpd(x)
        
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.feature(x)
        x = self.cbam(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.logit(x)
        return x


# ==========================================
# 5. Build functions
# ==========================================
def build_ferretnet_ddfc(
    num_classes: int = 2,
    *,
    in_channels: int = 3,
    dim: int = 96,
    depths: Iterable[int] = (2, 2),
    window_size: int = 3,
    enable_lfu: bool = True,
    use_ddfc: bool = True,
    checkpoint: Optional[str] = None,
    map_location: str = "cpu",
) -> nn.Module:
    """
    Build FerretNet with DDFC model
    
    Args:
        num_classes: number of classes
        in_channels: input channels
        dim: initial feature dimension
        depths: depth of each stage
        window_size: median filter window size
        enable_lfu: whether to enable local Fourier unit
        use_ddfc: whether to use DDFC module (False will revert to original FerretNet)
        checkpoint: path to checkpoint
        map_location: device to load weights
    
    Returns:
        FerretDDFC model instance
    """
    model = FerretDDFC(
        in_channels=in_channels,
        num_classes=num_classes,
        dim=dim,
        depths=tuple(depths),
        window_size=window_size,
        enable_lfu=enable_lfu,
        use_ddfc=use_ddfc,
    )

    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=map_location)
        state_dict = state.get("model_state", state)
        model.load_state_dict(state_dict, strict=False)

    return model
