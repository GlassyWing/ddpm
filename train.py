import argparse
import logging

import torch
from torchvision import transforms

from ddpm.models import DDPM, Unet
from ddpm.trainer import Trainer
from ddpm.utils import weights_init

if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser(description="Trainer for Diffusion model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each sample batch")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--name", type=str, default="", help="experiment name")

    opt = parser.parse_args()
    img_size = 128
    scales = [1, 1, 2, 2, 4, 4]
    num_layers = len(scales) * 2 + 1
    emb_dim = 64
    T = 1000

    tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    model = Unet(3, img_size, scales, emb_dim=emb_dim)
    diffusion = DDPM(model, img_size, T, emb_dim)

    if opt.pretrained_weights is not None:
        pretrained_dict = torch.load(opt.pretrained_weights, map_location="cpu")
        model_dict = diffusion.state_dict()

        # Fiter out unneccessary keys
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        diffusion.load_state_dict(model_dict)
        print("load pretrained weights!")

    diffusion = diffusion.cuda(2)
    diffusion.apply(weights_init())

    trainer = Trainer(
        diffusion,
        opt.dataset_path,
        img_size,
        transform=tfms,
        train_batch_size=opt.batch_size,
        train_lr=1e-3,
        train_num_epochs=opt.epochs,
        ema_decay=0.995,
        save_and_sample_every=500,
        num_workers=opt.n_cpu,
    )
    trainer.train()
