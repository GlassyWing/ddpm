import torch
import glob
import os

from ddpm.models import Unet, DDIM, DDPM, RectifiedFlow


def get_most_recent_file(folder_path):
    # 获取文件夹中所有文件的路径
    files = glob.glob(os.path.join(folder_path, '*'))

    # 过滤掉非文件项（例如子文件夹）
    files = [f for f in files if os.path.isfile(f)]

    if not files:
        return None  # 如果文件夹为空或没有文件，返回 None

    # 获取每个文件的最后修改时间，并找到修改时间最新的文件
    most_recent_file = max(files, key=os.path.getmtime)

    return most_recent_file


if __name__ == '__main__':
    img_size = 128
    resolution_multiplier = 1
    generator_resolution = img_size * resolution_multiplier
    scales = [1, 1, 2, 2, 4, 4]
    emb_dim = 128
    T = 55
    stride = 4
    attn_apply_level = 2  # attn resolution will be: (attn_apply_level + 1) ** 2

    model = Unet(3, img_size, scales,
                 emb_dim=emb_dim,
                 attn_apply_level=attn_apply_level,
                 resolution_multiplier=resolution_multiplier)
    # model.load_state_dict(torch.load("RectifiedFlow_ckpt_1_0.pth", map_location="cpu"))
    # print(model.state_dict())
    last_ckpt = get_most_recent_file("experiments/checkpoints")
    # last_ckpt = "experiments\checkpoints\cat_sample_ckpt_21_186000_a4.pth"
    print(f"load ckpt: {last_ckpt}")
    model.load_state_dict(torch.load(last_ckpt, map_location="cpu")["model"])
    diffusion = RectifiedFlow(model, img_size, T, emb_dim, stride=stride, resolution_multiplier=resolution_multiplier)
    diffusion.to("cuda:0")
    diffusion = diffusion.eval()

    diffusion.sample(device="cuda:0", n=4, path=f"./demo_{T}_no_consistent_ori.png")
