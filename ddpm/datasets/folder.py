import os
from glob import glob

import cv2
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

__all__ = ["ImageFolderDataset"]


class ImageFolderDataset(Dataset):

    def __init__(self, image_dir, img_dim, transform=None):
        self.img_paths = glob(os.path.join(image_dir, "*"))
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim
        self.transform = transform
        self.images = {}
        
    def __getitem__(self, idx):
        if idx in self.images:
            image = self.images[idx]
        else:
            image = cv2.imread(self.img_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w, c = image.shape
            if h > w:
                top_h = int((h - w) / 2)
                image = image[top_h:top_h + w]
            else:
                left_w = int((w - h) / 2)
                image = image[:, left_w:left_w + h]
            image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray(image, mode="RGB")
            self.images[idx] = image
            
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.img_paths)
