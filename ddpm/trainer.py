from copy import deepcopy

import torch
import os
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import *
from .models import DDPM
from .utils import update_average
import torch_optimizer as to
from warmup_scheduler import GradualWarmupScheduler


class Trainer:

    def __init__(self, ddpm: DDPM,
                 dataset_path,
                 img_size,
                 exp_name="",
                 transform=None,
                 train_batch_size=32,
                 train_lr=1e-3,
                 train_num_epochs=10000,
                 ema_decay=0.995,
                 num_workers=2,
                 save_and_sample_every=100,
                 ):
        self.ddpm = ddpm
        self.img_size = img_size
        self.train_batch_size = train_batch_size
        self.train_lr = train_lr
        self.train_num_epochs = train_num_epochs
        self.ema_decay = ema_decay
        self.num_workers = num_workers
        self.save_and_sample_every = save_and_sample_every

        self.sample_path = os.path.join("experiments", exp_name, "outputs")
        self.checkpoint_path = os.path.join("experiments", exp_name, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        if os.path.isdir(dataset_path):
            self.dataset = ImageFolderDataset(dataset_path, img_size, transform)
        else:
            self.dataset = dataset_path

        self.ddpm_shadow = deepcopy(ddpm)
        self.ema_updater = update_average
        update_average(self.ddpm_shadow, ddpm, beta=0.)

        self.optimizer = to.Lamb(self.ddpm.parameters(), lr=train_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                         milestones=[15, 100, 500],
                                                         gamma=0.5)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, 5, after_scheduler=scheduler)
        self.device = next(self.ddpm.parameters()).device

    def prepare_data(self, x_real):
        batch_images = x_real.to(self.device)
        batch_size = len(batch_images)
        batch_steps = np.random.choice(self.ddpm.T, batch_size)

        batch_beta = torch.from_numpy(self.ddpm.beta[batch_steps]).to(self.device).float()
        batch_bar_alpha = torch.from_numpy(self.ddpm.bar_alpha[batch_steps]).to(self.device).float()
        batch_bar_beta = torch.from_numpy(self.ddpm.bar_beta[batch_steps]).to(self.device).float()
        batch_steps = torch.from_numpy(batch_steps).to(self.device)

        w2 = batch_bar_beta / batch_beta

        batch_bar_alpha = batch_bar_alpha.reshape(-1, 1, 1, 1)
        batch_bar_beta = batch_bar_beta.reshape(-1, 1, 1, 1)
        w2 = w2.reshape(-1, 1, 1, 1)
        batch_steps = batch_steps.long()

        batch_noise = torch.randn_like(batch_images)
        batch_noise_images = batch_images * batch_bar_alpha + batch_noise * batch_bar_beta
        return batch_noise_images, batch_steps, batch_noise, w2

    def train(self):
        train_dataloader = DataLoader(self.dataset, self.train_batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=True,
                                      drop_last=True)

        step = 0
        for epoch in range(self.train_num_epochs):
            with tqdm(train_dataloader) as train_bar:
                for idx, x_real in enumerate(train_bar):
                    noise_images, steps, noise, w2 = self.prepare_data(x_real)
                    self.optimizer.zero_grad()
                    denoise = self.ddpm(noise_images, steps)
                    loss = torch.sum((denoise - noise) ** 2, dim=[1, 2, 3], keepdim=True).sum()
                    loss.backward()
                    self.optimizer.step()

                    self.ema_updater(self.ddpm_shadow, self.ddpm, beta=self.ema_decay)

                    train_bar.set_description(f"[{epoch}/{self.train_num_epochs}] loss: {loss.item():.4f}")
                    step += 1

                    if step != 0 and step % self.save_and_sample_every == 0:
                        self.ddpm.sample(f"{self.sample_path}/ddpm_ckpt_{epoch}_{step}.png",
                                         4, device=self.device)

                        self.ddpm_shadow.sample(f"{self.sample_path}/ddpm_ckpt_{epoch}_{step}_ema.png",
                                                4, device=self.device)
                        torch.save(self.ddpm.state_dict(),
                                   f"{self.checkpoint_path}/ddpm_ckpt_{epoch}_{step}.pth")
                self.scheduler.step()
