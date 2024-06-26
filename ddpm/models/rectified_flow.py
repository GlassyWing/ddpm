import math

import torch
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm, trange


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

    # def p_x(self, x, ema):
    #     z0 = torch.randn_like(x)
    #     z1 = x
    #     v = z1 - z0
    #
    #     # N = 8
    #     # eps = 1e-4
    #     # ts = (torch.arange(eps, N) / (N - 1))
    #     # idx = torch.randint(0, N - 1, size=(len(v),))
    #     # t1 = ts[idx].to(x.device)
    #     # t2 = ts[idx + 1].to(x.device)
    #     t2 = torch.rand(z0.size(0), 1, device=x.device)
    #     t1 = torch.rand_like(t2) * t2
    #
    #     z1 = v * t1.view(-1, 1, 1, 1) + z0
    #
    #     with torch.no_grad():
    #         z2 = v * t2.view(-1, 1, 1, 1) + z0
    #         v2 = self(z2, t2.view(-1, 1))
    #         target = v2 + t1.view(-1, 1, 1, 1) * (v - v2)
    #
    #     return z1, t1.view(-1, 1), target

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
            pred = self.forward(z, ti)
            z = z + pred * dt

        return z

    @torch.no_grad()
    def sample(self, path=None, n=4, z_samples=None, t0=0, device="cuda"):
        if z_samples is None:
            z_samples = torch.randn(n ** 2, 3, self.img_size, self.img_size, device=device)
        else:
            z_samples = z_samples.copy()
        if self.resolution_multiplier != 1:
            z_samples = torch.nn.functional.interpolate(z_samples, scale_factor=self.resolution_multiplier, mode='nearest')
        z_samples = self.sample_ode(z_samples, self.N, True)
        x_samples = torch.clip(z_samples, -1, 1)
        if path is None:
            return x_samples
        save_image(x_samples, path, nrow=n, normalize=True)
