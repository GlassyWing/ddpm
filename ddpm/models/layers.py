import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter, init

from ddpm.models.pos_embs.rotary import Rotary2D, apply_rotary_position_embeddings
from ddpm.utils import default

import math

__all__ = ["GroupNormCustom", "ResidualBlock", "Downsample", "Upsample",
           "AttentionBlock", "QKVAttention", "ScaleActConv"]


class SELayer(nn.Module):
    def __init__(self, channel, reduction=3, offset=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.offset = offset

    def forward(self, x):
        b, c, _, _ = x.size()
        y_mu = self.avg_pool(x)
        y = y_mu.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return (x + self.offset) * y.expand_as(x)


class GroupNormCustom(nn.Module):

    def __init__(self, n_groups, num_channels, eps=1e-6, affine=True):
        super().__init__()

        self.gn = nn.GroupNorm(n_groups, num_channels)

    def forward(self, x):
        return self.gn(x)


def calculate_out_padding(iH, iW, kernel_size, stride, padding, dilation=1):
    H, W = iH * 2, iW * 2

    OPH = H - int((iH - 1) * stride + dilation * (kernel_size - 1) - 2 * padding) - 1
    OPW = W - int((iW - 1) * stride + dilation * (kernel_size - 1) - 2 * padding) - 1
    return OPH, OPW


def create_divisor(iH, iW, kernel_size, stride, padding, dilation=1):
    H, W = iH * 2, iW * 2

    OPH = H - int((iH - 1) * stride + dilation * (kernel_size - 1) - 2 * padding) - 1
    OPW = W - int((iW - 1) * stride + dilation * (kernel_size - 1) - 2 * padding) - 1

    divisor = F.fold(F.unfold(torch.ones(1, 1, H, W), kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation, ),
                     output_size=(H, W),
                     kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    divisor[divisor == 0] = 1

    return divisor, (OPH, OPW)


class Upsample(nn.Module):

    def __init__(self, dim, dim_out=None, image_size=None, dilation=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = SELayer(dim)
        self.act = nn.SiLU(inplace=True)

        self.kernel_size = 3
        self.stride = 2
        self.ratio = dilation

        self.dilation = 1
        self.padding = int((self.kernel_size - 1) * self.dilation / 2)

        if image_size is not None:
            iH, iW = image_size, image_size
            divisor, (OPH, OPW) = create_divisor(iH, iW, self.kernel_size, self.stride, self.padding, self.dilation)
            upsample = nn.ConvTranspose2d(dim, dim,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          dilation=self.dilation,
                                          output_padding=(OPH, OPW),
                                          bias=False)
            self.register_buffer("divisor", divisor)
            self._conv_t = True
        else:
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(dim, dim, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation)
            )
            self._conv_t = False
        self.upsample = upsample

    def forward(self, x):
        x = self.scale(x)
        x = self.act(x)
        if self.ratio != 1:
            x = F.interpolate(x, scale_factor=1 / self.ratio, mode='nearest')
        x = self.upsample(x)
        if self._conv_t:
            if self.divisor.shape[-2:] != x.shape[-2:]:
                self.divisor = create_divisor(x.shape[2] // 2, x.shape[3] // 2,
                                              self.kernel_size, self.stride,
                                              self.padding, self.dilation)[0].to(x.device)
            x = x / self.divisor
        if self.ratio != 1:
            x = F.interpolate(x, scale_factor=self.ratio, mode='nearest')
        return x


class ScaleActConv(nn.Module):

    def __init__(self, dim, dim_out=None, kernel_size=3,
                 padding=1,
                 stride=1, bias=False, reduction=3,
                 dilation=1,
                 offset=1,
                 act='silu',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = SELayer(dim, reduction, offset)
        if act == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(dim, default(dim_out, dim), kernel_size,
                              padding=int(kernel_size - 1) * dilation // 2,
                              stride=stride,
                              bias=bias,
                              dilation=dilation,
                              **kwargs)

    def forward(self, x):
        x = self.scale(x)
        x = self.act(x)
        x = self.conv(x)
        return x


def Downsample(dim, dim_out=None, image_size=None, dilation=1):
    return ScaleActConv(dim, dim_out, stride=2, dilation=dilation)


class ResidualBlock(nn.Module):

    def __init__(self, in_c, out_c, emb_dim, n_groups=8, with_time_emb=True, dilation=1):
        super().__init__()
        self.with_time_emb = with_time_emb

        self.map = nn.Conv2d(in_c, out_c, kernel_size=(1, 1), bias=False)
        self.block1 = ScaleActConv(out_c, out_c, dilation=dilation, bias=True)
        self.block2 = ScaleActConv(out_c, out_c, dilation=dilation, bias=True)

        if with_time_emb:
            self.dense = nn.Conv2d(emb_dim, out_c, kernel_size=(1, 1), bias=True)

        self.out_c = out_c

    def forward(self, x, t):
        if x.shape[1] == self.out_c:
            xi = x
        else:
            xi = x = self.map(x)

        x = self.block1(x)
        if self.with_time_emb:
            x = x + self.dense(t.unsqueeze(-1).unsqueeze(-1))
        x = self.block2(x)
        return xi + x


class QKVAttention(nn.Module):

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, 1)  # 3 x (B, C, -1)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            out = F.scaled_dot_product_attention(q.reshape(bs, self.n_heads, ch, length).permute(0, 1, 3, 2),
                                                 k.reshape(bs, self.n_heads, ch, length).permute(0, 1, 3, 2),
                                                 v.reshape(bs, self.n_heads, ch, length).permute(0, 1, 3, 2))

        return out.permute(0, 1, 3, 2).reshape(bs, -1, length)


class AttentionBlock(nn.Module):

    def __init__(self, channels: int, num_heads: int, num_head_channels: int, resolution_multiplier: int = 1):
        super().__init__()
        self.rotary = Rotary2D(channels, resolution_multiplier=resolution_multiplier, )
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = GroupNormCustom(min(4, channels // 4), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.attn = QKVAttention(num_heads)
        self.proj_out = nn.Conv1d(channels, channels, 1)

        self.zero_init_weights()

    @torch.no_grad()
    def zero_init_weights(self):
        for p in self.proj_out.parameters():
            p.zero_()

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        x = self.norm(x)
        xi = x.view(B, C, -1)
        qkv = self.qkv(xi)  # (B, 3 x C, -1)
        # ---------------- Apply rotary 2d position embedding -----------------
        q_, k_, v_ = qkv.permute(0, 2, 1).chunk(3, 2)  # (B, -1, C)
        q_, k_ = apply_rotary_position_embeddings(self.rotary.forward(x), q_, k_)
        qkv = torch.cat((q_, k_, v_), 2).permute(0, 2, 1)  # (B, 3 x C, -1)
        # ----------------------------------------------------------------------
        h = self.attn(qkv)
        h = self.proj_out(h)
        return (xi + h).reshape(B, C, H, W)
