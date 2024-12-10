import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ddpm.datasets import ImageFolderDataset
from ddpm.models import DDPM, Unet, DDIM
from ddpm.models.rectified_flow import RectifiedFlow
from ddpm.trainer import Trainer
from ddpm.utils import weights_init

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser(description="Trainer for Diffusion model.")
    parser.add_argument("--dataset_path", "-d", type=str, required=True, help="dataset path")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs.")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="size of each sample batch")
    parser.add_argument("--pretrained_weights", "-p", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--name", type=str, default="", help="experiment name")
    parser.add_argument("--type", "-t", type=str, default="rflow",
                        help="Sampler type [ddpm/ddim/rflow], Default, `rflow`")
    parser.add_argument("--stride", "-s", type=int, default=1, help="sample stride for ddim")
    parser.add_argument("--num_steps", "-n", type=int, default=1000, help="sample times. Default, 1000")
    parser.add_argument("--accum_min", "-i", type=int, default=1, help="accumulation min steps, Default, 1.")
    parser.add_argument("--accum_max", "-a", type=int, default=128, help="accumulation max steps, Default, 128.")

    opt = parser.parse_args()
    img_size = 128
    scales = [1, 1,
              2, 2,
              4, 4]
    emb_dim = 128
    attn_apply_level = 2
    T = opt.num_steps
    sampler = opt.type
    stride = opt.stride
    accum_min = opt.accum_min
    accum_max = opt.accum_max

    tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    ds = ImageFolderDataset(opt.dataset_path, img_size, transform=tfms)
    dataloader = DataLoader(ds, batch_size=opt.batch_size, num_workers=opt.n_cpu,
                            persistent_workers=True,
                            )

    model = Unet(3, img_size, scales, emb_dim=emb_dim, attn_apply_level=attn_apply_level)
    print(model)

    if sampler == "ddpm":
        diffusion = DDPM(model, img_size, T, emb_dim)
    elif sampler == "ddim":
        diffusion = DDIM(model, img_size, T, emb_dim, stride=stride)  # Sample steps = T // stride. E.g, 1000 / 50 = 20
    elif sampler == "rflow":
        diffusion = RectifiedFlow(model, img_size, T, emb_dim)
    else:
        raise ValueError("Unsupported sampler type")

    if opt.pretrained_weights is None:
        diffusion = diffusion.apply(weights_init())

    diffusion.cuda(0)

    trainer = Trainer(
        diffusion,
        train_batch_size=opt.batch_size,
        train_lr=2e-4,
        train_num_epochs=opt.epochs,
        ema_decay=0.995,
        save_and_sample_every=500,
        num_workers=opt.n_cpu,
        accu_min_steps=accum_min,
        accu_max_steps=accum_max,
        accu_scheduler_epochs=8,
    )
    
    if opt.pretrained_weights is not None:
        trainer.load(opt.pretrained_weights)
    
    trainer.train(dataloader)
