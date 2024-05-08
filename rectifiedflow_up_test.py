import torch

from ddpm.models import Unet, DDIM, DDPM
from ddpm.models.rectified_flow_up import RectifiedFlow

if __name__ == '__main__':
    img_size = 128
    resolution_multiplier = 1
    generator_resolution = img_size * resolution_multiplier
    scales = [1, 1, 2, 2, 4, 4]
    emb_dim = 64
    T = 100
    stride = 4
    attn_apply_level = 3  # attn resolution will be: (attn_apply_level + 1) ** 2

    # torch.manual_seed(49)
    model = Unet(3, img_size, scales,
                 emb_dim=emb_dim,
                 attn_apply_level=attn_apply_level,
                 resolution_multiplier=resolution_multiplier)
    # model.load_state_dict(torch.load("RectifiedFlow_ckpt_1_0.pth", map_location="cpu"))
    # print(model.state_dict())
    model.load_state_dict(torch.load("experiments/checkpoints/sample_ckpt_109_957000_a64.pth", map_location="cpu"))
    diffusion = RectifiedFlow(model, img_size, T, emb_dim, stride=stride,  resolution_multiplier=resolution_multiplier)
    diffusion.to("cuda:0")
    diffusion = diffusion.eval()

    diffusion.sample(device="cuda:0", n=4, path=f"./demo_{T}_no_consistent.png", r=2)
