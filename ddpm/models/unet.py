from collections import OrderedDict

from torch import nn
from ..utils.hand import default

__all__ = ["Unet"]


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), (5, 5), padding=2),
        nn.SiLU(inplace=True),
        nn.GroupNorm(32, default(dim_out, dim))
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Conv2d(dim, default(dim_out, dim), (5, 5), padding=2),
        nn.SiLU(inplace=True),
        nn.GroupNorm(32, default(dim_out, dim)),
        nn.AvgPool2d((2, 2), (2, 2))
    )


class Residual(nn.Module):

    def __init__(self, fn, emb_dim=None, dim=None):
        super().__init__()
        self.alpha_fn = nn.Sequential(
            nn.Linear(emb_dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.fn = fn

    def forward(self, a, t_emb, b=None):
        alpha = self.alpha_fn(t_emb)
        return a + alpha[:, :, None, None] * self.fn(b if b is not None else a)


class ResidualBlock(nn.Module):

    def __init__(self, in_c, out_c, emb_dim, n_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=(1, 1), bias=False)
        self.dense = nn.Linear(emb_dim, out_c, bias=False)
        self.res = Residual(nn.Sequential(
            nn.Conv2d(out_c, out_c, (5, 5), padding=(2, 2)),
            nn.SiLU(inplace=True),
        ), emb_dim, out_c)
        self.out_c = out_c
        self.norm = nn.GroupNorm(n_groups, out_c)

    def forward(self, x, t):
        if x.shape[1] == self.out_c:
            xi = x
        else:
            xi = x = self.conv(x)
        ti = t
        t = self.dense(t)
        x = x + t[:, :, None, None]
        x = self.res(xi, ti, x)
        x = self.norm(x)
        return x


class Unet(nn.Module):

    def __init__(self, img_c, img_size, scales, emb_dim, min_pixel=4, n_block=2, n_groups=32):
        super().__init__()
        self.n_block = 2
        self.img_c = img_c
        self.scales = scales
        self.img_size = img_size
        self.n_groups = n_groups

        self.stem = nn.Conv2d(img_c, emb_dim, (5, 5), padding=2)

        skip_pooling = 0
        cur_c = emb_dim
        encoder_blocks = OrderedDict()
        chs = []
        for i, scale in enumerate(scales):
            for j in range(n_block):
                chs.append([cur_c, scale * emb_dim])
                block = ResidualBlock(cur_c, scale * emb_dim, emb_dim, n_groups)
                cur_c = scale * emb_dim
                encoder_blocks[f'enc_block_{i * n_block + j}'] = block
            if img_size > min_pixel:
                encoder_blocks[f'down_block_{i}'] = Downsample(cur_c)
                img_size = img_size // 2
            else:
                skip_pooling += 1
        chs_inv = chs[::-1]
        encoder_blocks[f'enc_block_{(len(scales)) * n_block}'] = ResidualBlock(cur_c, cur_c, emb_dim, n_groups)
        self.encoder_blocks = nn.ModuleDict(encoder_blocks)
        decoder_blocks = OrderedDict()
        for i, scale in enumerate(scales[::-1]):
            if i >= skip_pooling:
                decoder_blocks[f'up_block_{i}'] = Upsample(cur_c)
            for j in range(n_block):
                decoder_blocks[f'dec_block_{i * n_block + j}'] = ResidualBlock(*chs_inv[i * n_block + j][::-1],
                                                                               emb_dim,
                                                                               n_groups)
                cur_c = chs_inv[i * n_block + j][0]
        decoder_blocks[f'to_rgb'] = nn.Sequential(nn.Conv2d(cur_c, img_c, (5, 5), padding=2))
        self.decoder_blocks = nn.ModuleDict(decoder_blocks)

    def forward(self, x, t):
        x = self.stem(x)

        inners = [x]
        for name, module in self.encoder_blocks.items():
            if name.startswith("enc"):
                x = module(x, t)
                inners.append(x)
            else:
                x = module(x)
                inners.append(x)

        inners = inners[:-2]
        for name, module in self.decoder_blocks.items():

            if name.startswith("up"):
                x = module(x)
                xi = inners.pop()
                x = x + xi

            elif name.startswith("dec"):
                xi = inners.pop()
                x = module(x, t)
                x = x + xi
            else:
                x = module(x)
        return x
