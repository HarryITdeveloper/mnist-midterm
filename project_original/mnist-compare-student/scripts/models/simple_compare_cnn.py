import torch
import torch.nn as nn
import torch.nn.functional as F

class Tower(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class CompareNet(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.tower = Tower(out_dim=feat_dim)
        in_dim = feat_dim*2  
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, xa, xb):
        fa = self.tower(xa)
        fb = self.tower(xb)
        fuse = torch.cat([fa, fb], dim=-1)
        logit = self.head(fuse).squeeze(1)
        return logit

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
