import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

def _build_transforms(is_train: bool):
    # 兼容旧 torchvision 没有 GaussianBlur 的情况
    try:
        blur = T.GaussianBlur(kernel_size=3)
    except Exception:
        blur = T.Lambda(lambda x: x)

    if is_train:
        return T.Compose([
            T.ToPILImage(),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.RandomPerspective(distortion_scale=0.25, p=0.3),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.ToTensor(),
            blur,
            T.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

class PairNPZDataset(Dataset):
    def __init__(self, path_npz, is_train=False):
        arr = np.load(path_npz, allow_pickle=False)
        self.x = arr["x"]  # (N,28,56), uint8
        self.y = arr["y"].astype(np.int64) if "y" in arr.files else None
        self.ids = arr["id"] if "id" in arr.files else None
        self.is_train = is_train
        self.tf = _build_transforms(is_train)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]  # (28,56), uint8
        # 先对整张 28x56 做一致变换
        x = self.tf(img)   # (1,28,56) tensor
        xa = x[:, :, :28]  # (1,28,28)
        xb = x[:, :, 28:]  # (1,28,28)

        if self.y is None:
            return xa, xb, self.ids[idx]
        else:
            return xa, xb, int(self.y[idx])
