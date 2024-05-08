import time
from copy import deepcopy

import torch
import os

from tqdm import tqdm
from .utils import update_average


class Trainer:

    def __init__(self, sampler,
                 exp_name="",
                 train_batch_size=32,
                 train_lr=2e-4,
                 train_num_epochs=10000,
                 ema_decay=0.995,
                 num_workers=2,
                 save_and_sample_every=100,
                 accumulation_steps=2,
                 accu_scheduler_epochs=5,
                 accu_min_steps=8,
                 accu_max_steps=64,
                 ):
        self.sampler = sampler
        self.train_batch_size = train_batch_size
        self.train_init_lr = train_lr
        self.train_lr = self.train_init_lr / accumulation_steps
        self.train_num_epochs = train_num_epochs
        self.ema_decay = ema_decay
        self.num_workers = num_workers
        self.save_and_sample_every = save_and_sample_every

        self.accumulation_steps = accu_min_steps
        self.accu_scheduler_epochs = accu_scheduler_epochs
        self.accu_max_steps = accu_max_steps
        self.accu_min_steps = accu_min_steps

        self.sample_path = os.path.join("experiments", exp_name, "outputs")
        self.checkpoint_path = os.path.join("experiments", exp_name, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        self.sampler_shadow = deepcopy(sampler)
        self.ema_updater = update_average
        update_average(self.sampler_shadow, sampler, beta=0.)

        self.device = next(self.sampler.parameters()).device

    def train(self, dataloader):
        train_dataloader = dataloader

        optimizer = torch.optim.AdamW(self.sampler.parameters(), lr=self.train_lr)

        step = 0
        last_log_steps = step
        for epoch in range(self.train_num_epochs):

            self.sampler.zero_grad(set_to_none=True)

            with tqdm(train_dataloader) as train_bar:
                for idx, x_real in enumerate(train_bar):

                    # time.sleep(0.35)
                    x_real = x_real.to(self.device)

                    # with torch.cuda.amp.autocast():
                    noise_images, steps, noise = self.sampler.p_x(x_real, self.sampler_shadow)
                    denoise = self.sampler(noise_images, steps)
                    loss = torch.sum((denoise - noise) ** 2, dim=[1, 2, 3], keepdim=True).sum()

                    # scaler.scale(loss).backward()
                    loss.backward()

                    if (step + 1 - last_log_steps) % self.accumulation_steps == 0:
                        optimizer.step()
                        self.sampler.zero_grad(set_to_none=True)

                        # self.ema_updater(self.sampler_shadow, self.sampler, beta=self.ema_decay)
                        # scheduler.step()

                    train_bar.set_description(f"[{epoch}/{self.train_num_epochs}] "
                                              f"loss: {loss.item():.4f} ")
                    step += 1

                    if step % self.save_and_sample_every == 0:
                        self.sampler.sample(
                            f"{self.sample_path}/sample_ckpt_{epoch}_{step}_a{self.accumulation_steps}.png",
                            4, device=self.device)
                        # self.sampler_shadow.sample(
                        #     f"{self.sample_path}/sample_ckpt_{epoch}_{step}_a{self.accumulation_steps}_ema.png",
                        #     4, device=self.device)
                        torch.save(self.sampler.model.state_dict(),
                                   f"{self.checkpoint_path}/sample_ckpt_{epoch}_{step}_a{self.accumulation_steps}.pth")
                        # torch.save(self.sampler_shadow.model.state_dict(),
                        #            f"{self.checkpoint_path}/sample_ckpt_{epoch}_{step}_a{self.accumulation_steps}_ema.pth")

            if (epoch + 1) % self.accu_scheduler_epochs == 0:
                last_accu_steps = self.accumulation_steps

                if self.accumulation_steps < self.accu_max_steps:
                    self.accumulation_steps *= 2
                    self.train_lr = self.train_init_lr / self.accumulation_steps
                    last_log_steps = step

                    print(f"Update accumulation_steps: {self.accumulation_steps}, train_lr: {self.train_lr}")
                elif self.accumulation_steps >= self.accu_max_steps:
                    self.accumulation_steps = self.accu_min_steps
                    self.train_lr = self.train_init_lr / self.accumulation_steps
                    last_log_steps = step
                    print(f"Update accumulation_steps: {self.accumulation_steps}, train_lr: {self.train_lr}")

                if last_accu_steps != self.accumulation_steps:
                    del optimizer
                    optimizer = torch.optim.AdamW(self.sampler.parameters(), lr=self.train_lr)
