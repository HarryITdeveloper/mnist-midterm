# mnist-compare-student/scripts/models/siamese_compare.py
import torch
import torch.nn as nn

__all__ = ["Model", "count_params"]

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class FeatExtractor(nn.Module):
    def __init__(self, ch=(32, 64, 128), p_drop=0.1):
        super().__init__()
        layers, in_c = [], 1
        for c in ch:
            layers += [
                nn.Conv2d(in_c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(p_drop),
            ]
            in_c = c
        self.backbone = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):  # x: (B,1,28,28)
        h = self.backbone(x)
        f = torch.cat([self.gap(h), self.gmp(h)], dim=1)  # (B,2*C,1,1)
        return f.flatten(1)  # (B, 2*C)

class CompareHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        in_dim = feat_dim * 4  # fL, fR, |fL-fR|, fL*fR
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, fL, fR):
        z = torch.cat([fL, fR, torch.abs(fL - fR), fL * fR], dim=1)
        return self.net(z)  # (B,1)

class Model(nn.Module):
    """
    前向签名保持与你现有训练/推理一致：forward(xa, xb) → logits
    """
    def __init__(self, ch=(32, 64, 128), p_drop=0.1):
        super().__init__()
        self.feat = FeatExtractor(ch=ch, p_drop=p_drop)
        feat_dim = ch[-1] * 2  # GAP+GMP
        self.head = CompareHead(feat_dim)

    def forward(self, xa, xb):
        fL = self.feat(xa)
        fR = self.feat(xb)
        logit = self.head(fL, fR).squeeze(1)
        return logit
