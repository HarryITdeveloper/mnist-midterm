import numpy as np
import torch
from torch.utils.data import Dataset

class PairNPZDataset(Dataset):
    def __init__(self, path_npz, is_train=False):
        arr = np.load(path_npz, allow_pickle=False)
        self.x = arr["x"]  # (N, 28, 56), uint8
        self.y = arr["y"].astype(np.int64) if "y" in arr.files else None
        self.ids = arr["id"] if "id" in arr.files else None
        self.is_train = is_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]  # (28,56)
        # split into two (28,28)
        xa = img[:, :28]
        xb = img[:, 28:]
        # normalize to [0,1], add channel dim
        xa = torch.from_numpy(xa).float().unsqueeze(0) / 255.0
        xb = torch.from_numpy(xb).float().unsqueeze(0) / 255.0
        if self.y is None:
            return xa, xb, self.ids[idx]
        else:
            y = int(self.y[idx])
            return xa, xb, y
