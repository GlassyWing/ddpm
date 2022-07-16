import torch
from torch import nn
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

__all__ = ["DDPM"]


class FourierMapping(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        assert out_c % 2 == 0
        self.register_buffer("B", torch.randn(in_c, out_c // 2))

    def forward(self, x):
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        x_proj = (2. * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class DDPM(nn.Module):

    def __init__(self, model, img_size, T=1000, embedding_size=2):
        super().__init__()
        self.img_size = img_size
        self.T = T
        self.alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
        self.beta = np.sqrt(1 - self.alpha ** 2)
        self.bar_alpha = np.cumprod(self.alpha)
        self.bar_beta = np.sqrt(1 - self.bar_alpha ** 2)
        self.sigma = self.beta
        self.model = model

        # self.t = nn.Embedding(T, embedding_dim=embedding_size)
        self.t = nn.Sequential(
            FourierMapping(1, embedding_size),
            nn.Linear(embedding_size, embedding_size)
        )

    def forward(self, x_in, t_in):
        t_e = self.t(t_in)  # (n, z)
        x_r = self.model(x_in, t_e)
        return x_r

    @torch.no_grad()
    def sample(self, path=None, n=4, z_samples=None, t0=0, device="cuda"):
        if z_samples is None:
            z_samples = torch.randn(n ** 2, 3, self.img_size, self.img_size, device=device)
        else:
            z_samples = z_samples.copy()
        for t in tqdm(range(t0, self.T), ncols=0):
            t = self.T - t - 1
            bt = torch.tensor([t] * z_samples.shape[0], dtype=torch.long, device=device)
            z_samples -= self.beta[t] ** 2 / self.bar_beta[t] * self.forward(z_samples, bt)
            z_samples /= self.alpha[t]
            z_samples += torch.randn_like(z_samples) * self.sigma[t]
        x_samples = torch.clip(z_samples, -1, 1)
        if path is None:
            return x_samples
        save_image(x_samples, path, nrow=n, normalize=True)
