import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_specxnet", "ffc_xception", "Xception"]


# ---------- spectral helpers (ported) ----------
def center_crop(t, target_h, target_w):
    _, _, H, W = t.shape
    start_h = (H - target_h) // 2
    start_w = (W - target_w) // 2
    return t[:, :, start_h:start_h + target_h, start_w:start_w + target_w]


def match_size(a, b):
    _, _, H_a, W_a = a.shape
    _, _, H_b, W_b = b.shape
    target_H = min(H_a, H_b)
    target_W = min(W_a, W_b)
    a_cropped = a[:, :, :target_H, :target_W]
    b_cropped = b[:, :, :target_H, :target_W]
    return a_cropped, b_cropped


class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super().__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if isinstance(x, tuple) else (x, 0)
        id_l, id_g = x
        x_cat = id_l if isinstance(id_g, int) else torch.cat([id_l, id_g], dim=1)
        x_pool = self.avgpool(x_cat)
        x_conv = self.relu1(self.conv1(x_pool))
        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x_conv))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x_conv))
        return x_l, x_g


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2,
                                    kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        B, C, H, W = x.size()
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        B, C, H_pad, W_pad = x.size()
        ffted = torch.fft.rfft2(x, norm="ortho")
        ffted = torch.view_as_real(ffted)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(B, C * 2, H_pad, W_pad // 2 + 1)
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        out_channels = ffted.size(1) // 2
        ffted = ffted.view(B, out_channels, 2, H_pad, W_pad // 2 + 1)
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(H_pad, W_pad), norm="ortho")
        output = output[:, :, :H, :W]
        return output


class DualFourierAttention(nn.Module):
    def __init__(self, in_channels_local, in_channels_global):
        super().__init__()
        self.norm_l = nn.BatchNorm2d(in_channels_local)
        self.norm_g = nn.BatchNorm2d(in_channels_global)

        self.attn_local = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels_local, in_channels_local // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_local // 2, in_channels_local, 1),
            nn.Sigmoid(),
        )
        self.attn_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels_global, in_channels_global // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_global // 2, in_channels_global, 1),
            nn.Sigmoid(),
        )

    def forward(self, Y_l, Y_g):
        Y_l = self.norm_l(Y_l)
        Y_g = self.norm_g(Y_g)

        A_g = self.attn_global(Y_g)
        A_l = self.attn_local(Y_l)

        if A_g.shape[-2:] != Y_l.shape[-2:]:
            A_g = F.interpolate(A_g, size=Y_l.shape[-2:], mode="bilinear", align_corners=False)
        if A_l.shape[-2:] != Y_g.shape[-2:]:
            A_l = F.interpolate(A_l, size=Y_g.shape[-2:], mode="bilinear", align_corners=False)

        Y_l_mod = Y_l * A_g + Y_l
        Y_g_mod = Y_g * A_l + Y_g
        return Y_l_mod, Y_g_mod


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, enable_lfu=True):
        super().__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.Identity()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=False),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        target_size = x.shape[2:]
        fu_out = self.fu(x)
        if self.enable_lfu:
            pooled = F.adaptive_avg_pool2d(x, (2, 2))
            lfu_out = self.lfu(pooled)
            lfu_out = F.interpolate(lfu_out, size=target_size, mode="nearest")
            if lfu_out.shape[2:] != target_size:
                lfu_out = center_crop(lfu_out, target_size[0], target_size[1])
        else:
            lfu_out = 0
        out = self.conv2(x + fu_out + lfu_out)
        return out


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super().__init__()
        assert stride in [1, 2]
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gout = ratio_gout
        self.convl2l = nn.Identity() if (in_cl == 0 or out_cl == 0) else nn.Conv2d(in_cl, out_cl,
                                                                                   kernel_size, stride,
                                                                                   padding, dilation,
                                                                                   groups, bias)
        self.convl2g = nn.Identity() if (in_cl == 0 or out_cg == 0) else nn.Conv2d(in_cl, out_cg,
                                                                                   kernel_size, stride,
                                                                                   padding, dilation,
                                                                                   groups, bias)
        self.convg2l = nn.Identity() if (in_cg == 0 or out_cl == 0) else nn.Conv2d(in_cg, out_cl,
                                                                                   kernel_size, stride,
                                                                                   padding, dilation,
                                                                                   groups, bias)
        self.convg2g = nn.Identity() if (in_cg == 0 or out_cg == 0) else SpectralTransform(in_cg, out_cg,
                                                                                           stride, enable_lfu)

    def forward(self, x):
        if not isinstance(x, tuple):
            x = (x, torch.zeros_like(x))
        x_l, x_g = x
        out_l_local = self.convl2l(x_l)
        out_l_global = self.convg2l(x_g)
        out_g_local = self.convl2g(x_l)
        out_g_global = self.convg2g(x_g)
        if not isinstance(out_g_local, int) and not isinstance(out_g_global, int):
            out_g_local, out_g_global = match_size(out_g_local, out_g_global)
            out_g = out_g_local + out_g_global
        else:
            out_g = 0
        if not isinstance(out_l_local, int) and not isinstance(out_l_global, int):
            out_l_local, out_l_global = match_size(out_l_local, out_l_global)
            out_l = out_l_local + out_l_global
        else:
            out_l = 0
        return out_l, out_g


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                 enable_lfu=True):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        self.dual_attn = DualFourierAttention(
            in_channels_local=int(out_channels * (1 - ratio_gout)),
            in_channels_global=int(out_channels * ratio_gout),
        )
        self.bn_l = nn.Identity() if ratio_gout == 1 else norm_layer(int(out_channels * (1 - ratio_gout)))
        self.bn_g = nn.Identity() if ratio_gout == 0 else norm_layer(int(out_channels * ratio_gout))
        self.act_l = nn.Identity() if ratio_gout == 1 else activation_layer(inplace=False)
        self.act_g = nn.Identity() if ratio_gout == 0 else activation_layer(inplace=False)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        x_l, x_g = self.dual_attn(x_l, x_g)
        return x_l, x_g


# ---------- SpecXNet FFC Xception (ported) ----------
class TupleReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        if isinstance(x, tuple):
            return self.relu(x[0]), self.relu(x[1])
        return self.relu(x)


class TupleMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        if isinstance(x, tuple):
            return self.pool(x[0]), self.pool(x[1])
        return self.pool(x)


class FFCSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                 ratio_gin=0.5, ratio_gout=0.5, lfu=True):
        super().__init__()
        local_in_channels = int(in_channels * (1 - ratio_gin))
        if local_in_channels < 1:
            local_in_channels = in_channels
        self.dw = FFC_BN_ACT(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding, groups=local_in_channels,
                             ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=lfu)
        self.pw = FFC_BN_ACT(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                             ratio_gin=ratio_gout, ratio_gout=ratio_gout, enable_lfu=lfu)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class FFCBlock(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True,
                 grow_first=True, ratio_gin=0.5, ratio_gout=0.5, lfu=True):
        super().__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = FFC_BN_ACT(in_filters, out_filters, kernel_size=1, stride=strides,
                                   padding=0, ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=lfu)
        else:
            self.skip = None

        layers = []
        if grow_first:
            layers.append(TupleReLU(inplace=False))
            layers.append(FFCSeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1,
                                             bias=False, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu))
            filters = out_filters
        else:
            filters = in_filters

        for _ in range(reps - 1):
            layers.append(TupleReLU(inplace=False))
            layers.append(FFCSeparableConv2d(filters, filters, kernel_size=3, stride=1, padding=1,
                                             bias=False, ratio_gin=ratio_gout, ratio_gout=ratio_gout, lfu=lfu))

        if not grow_first:
            layers.append(TupleReLU(inplace=False))
            layers.append(FFCSeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1,
                                             bias=False, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu))

        if strides != 1:
            layers.append(TupleMaxPool2d(3, stride=strides, padding=1))
        self.rep = nn.Sequential(*layers)

    def forward(self, inp):
        if not isinstance(inp, tuple):
            inp = (inp, torch.zeros_like(inp))
        out = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
        else:
            skip = inp
        out_local, skip_local = match_size(out[0], skip[0])
        out_global, skip_global = match_size(out[1], skip[1])
        out_local = F.relu(out_local + skip_local, inplace=False)
        out_global = F.relu(out_global + skip_global, inplace=False)
        return out_local, out_global


class Xception(nn.Module):
    def __init__(self, num_classes=2, ratio_gin=0.5, ratio_gout=0.5, lfu=True):
        super().__init__()
        self.num_classes = num_classes
        self.ratio = ratio_gin

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = FFCBlock(64, 128, reps=2, strides=2, start_with_relu=False, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block2 = FFCBlock(128, 256, reps=2, strides=2, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block3 = FFCBlock(256, 728, reps=2, strides=2, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)

        self.block4 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block5 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block6 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block7 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)

        self.block8 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block9 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block10 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block11 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)

        self.block12 = FFCBlock(728, 1024, reps=2, strides=2, start_with_relu=True, grow_first=False,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)

        self.conv3 = FFCSeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1,
                                        bias=False, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.bn3 = nn.BatchNorm2d(int(1536 * (1 - ratio_gout)))

        self.conv4 = FFCSeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1,
                                        bias=False, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.bn4 = nn.BatchNorm2d(int(2048 * (1 - ratio_gout)))

        self.fc = nn.Linear(int(2048 * (1 - ratio_gout)), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        _, _, H, W = x.shape
        if H % 2 == 1 or W % 2 == 1:
            new_H = H - (H % 2)
            new_W = W - (W % 2)
            x = x[:, :, :new_H, :new_W]

        local_channels = int(x.size(1) * (1 - self.ratio))
        if local_channels == 0 or local_channels == x.size(1):
            x = (x, torch.zeros_like(x))
        else:
            x = (x[:, :local_channels, ...], x[:, local_channels:, ...])

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x_local = x[0]
        x_local = self.conv3((x_local, torch.zeros_like(x_local)))[0]
        x_local = self.bn3(x_local)
        x_local = self.relu(x_local)

        x_local = self.conv4((x_local, torch.zeros_like(x_local)))[0]
        x_local = self.bn4(x_local)
        x_local = self.relu(x_local)

        x_local = F.adaptive_avg_pool2d(x_local, (1, 1))
        x_local = x_local.view(x_local.size(0), -1)
        x_local = self.fc(x_local)
        return x_local


def ffc_xception(pretrained=False, **kwargs):
    ratio = kwargs.pop("ratio", 0.5)
    kwargs.pop("use_se", None)
    model = Xception(ratio_gin=ratio, ratio_gout=ratio, **kwargs)
    if pretrained:
        # No pretrained weights bundled.
        pass
    return model


def build_specxnet(
    num_classes: int = 2,
    ratio: float = 0.5,
    checkpoint: Optional[str] = None,
    map_location: str = "cpu",
) -> nn.Module:
    """
    Create the SpecXNet FFC Xception model and optionally load a checkpoint.
    """
    model = ffc_xception(num_classes=num_classes, ratio=ratio)

    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=map_location)
        state_dict = state.get("model_state", state)
        model.load_state_dict(state_dict, strict=False)

    return model

