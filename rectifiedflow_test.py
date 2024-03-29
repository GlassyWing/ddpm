import torch

from ddpm.models import Unet, DDIM, DDPM, RectifiedFlow

if __name__ == '__main__':
    img_size = 128
    resolution_multiplier = 2
    generator_resolution = img_size * resolution_multiplier
    scales = [1, 1, 2, 2, 4, 4]
    emb_dim = 64
    T = 35
    stride = 4
    attn_apply_level = 3  # attn resolution will be: (attn_apply_level + 1) ** 2

    # torch.manual_seed(0)
    model = Unet(3, img_size, scales,
                 emb_dim=emb_dim,
                 attn_apply_level=attn_apply_level,
                 resolution_multiplier=resolution_multiplier)
    # model.load_state_dict(torch.load("RectifiedFlow_ckpt_1_0.pth", map_location="cpu"))
    # print(model.state_dict())
    model.load_state_dict(torch.load("experiments/checkpoints/sample_ckpt_28_124500.pth", map_location="cpu"))
    diffusion = RectifiedFlow(model, generator_resolution, T, emb_dim, stride=stride)
    diffusion.to("cuda:1")
    diffusion = diffusion.eval()

    diffusion.sample(device="cuda:1", path=f"./demo_2_{T}.png")
