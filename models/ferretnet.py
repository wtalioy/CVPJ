import os
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_ferretnet", "Ferret"]


class ExclusionMedianValues(nn.Module):
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 0):
        super().__init__()
        self.depth_wise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding=padding, groups=in_channels, bias=False
        )
        self.point_wise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


class DilatedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            dilation=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=2,
            groups=in_channels,
            dilation=2,
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            dilation=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) + self.conv2(x)
        x = self.conv3(x)
        return x


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
        x = x * spatial_out
        
        return x


class DSBlock(nn.Module):
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
        x = self.layers(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class Ferret(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dim: int,
        depths: Sequence[int],
        window_size: int,
    ):
        super().__init__()
        self.dim = dim

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
        for depth in depths:
            blocks = nn.Sequential(
                DSBlock(self.dim, self.dim * 2, stride=2),
                *[DSBlock(self.dim * 2, self.dim * 2, stride=1) for _ in range(max(depth - 1, 0))],
            )
            self.feature.append(blocks)
            self.dim *= 2
        self.feature.append(nn.Conv2d(self.dim, self.dim, 1, 1, bias=False))

        self.cbam = CBAM(channels=self.dim, reduction=16)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.logit = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(self.dim, num_classes),
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


def build_ferretnet(
    num_classes: int = 2,
    *,
    in_channels: int = 3,
    dim: int = 96,
    depths: Iterable[int] = (2, 2),
    window_size: int = 3,
    checkpoint: Optional[str] = None,
    map_location: str = "cpu",
) -> nn.Module:
    model = Ferret(
        in_channels=in_channels,
        num_classes=num_classes,
        dim=dim,
        depths=depths,
        window_size=window_size,
    )

    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=map_location)
        state_dict = state.get("model_state", state)
        model.load_state_dict(state_dict, strict=False)

    return model

