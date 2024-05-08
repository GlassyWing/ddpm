import math

import torch
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm, trange
import torch.nn.functional as F


class RectifiedFlow(nn.Module):

    def __init__(self, model, img_size, T=1000, embedding_size=2, stride=2, eta=1, resolution_multiplier=1):
        super().__init__()
        self.embedding_size = embedding_size
        self.model = model
        self.img_size = img_size
        self.resolution_multiplier = resolution_multiplier
        self.N = T

    def p_x(self, x, ema):
        z0 = torch.randn_like(x)
        z1 = x
        ti = torch.rand(z0.shape[0], 1, 1, 1, device=x.device)
        zt = ti * z1 + (1. - ti) * z0
        target = z1 - z0
        return zt, ti.squeeze(-1).squeeze(-1), target

    def t(self, ti):
        return ti.expand(ti.shape[0], self.embedding_size)

    def forward(self, x, t_in):
        te = self.t(t_in)
        x_r = self.model(x, te)
        return x_r

    def sample_ode(self, z0, N, verbose=False):
        if N == -1:
            N = self.N
        dt = 1. / N
        z = z0
        batch_size = z.size(0)

        if verbose:
            rg = trange(N)
        else:
            rg = range(N)
        for i in rg:
            ti = torch.ones(batch_size, 1, device=z0.device, dtype=torch.float) * i / N

            low_z = z[..., ::2, ::2]
            low_v = self.forward(low_z, ti)
            low_v = F.interpolate(low_v, scale_factor=2, mode='nearest')

            low_z = z + low_v * dt
            high_v = self.forward(low_z, ti)
            high_v_s = high_v[..., ::2, ::2]
            high_v_s = F.interpolate(high_v_s, scale_factor=2, mode='nearest')
            tol = high_v - high_v_s

            v = low_v + tol

            z = z + v * dt

        return z

    @torch.no_grad()
    def sample(self, path=None, n=4, z_samples=None, t0=0, device="cuda", r=1):
        if z_samples is None:
            z_samples = torch.randn(n ** 2, 3, self.img_size, self.img_size, device=device)
        else:
            z_samples = z_samples.copy()
        if r != 1:
            z_samples_r = torch.randn(n ** 2, 3, self.img_size * r, self.img_size * r, device=device)
            z_samples_r[..., ::r, ::r] = z_samples
            z_samples = z_samples_r

        z_samples = self.sample_ode(z_samples, self.N, True)
        x_samples = torch.clip(z_samples, -1, 1)
        if path is None:
            return x_samples
        save_image(x_samples, path, nrow=n, normalize=True)
