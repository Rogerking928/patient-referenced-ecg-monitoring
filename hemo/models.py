"""Model zoo: raw-waveform ResNet1D, handcrafted gradient boosting, and the hybrid."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

N_STAGES = 5


class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(),
                                nn.Linear(max(c // r, 4), c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(-1))
        return x * w.unsqueeze(-1)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, k, stride, p):
        super().__init__()
        pad = k // 2
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=pad, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=pad, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = SEBlock(cout)
        self.do = nn.Dropout(p)
        self.short = (nn.Identity() if (cin == cout and stride == 1)
                      else nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                         nn.BatchNorm1d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.b1(self.c1(x)))
        y = self.do(y)
        y = self.b2(self.c2(y))
        y = self.se(y)
        return self.act(y + self.short(x))


class Encoder1D(nn.Module):
    """Waveform -> embedding."""

    def __init__(self, in_ch, base=32, n_blocks=5, k=9, p=0.15, embed=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True), nn.MaxPool1d(2),
        )
        blocks, c = [], base
        for i in range(n_blocks):
            cout = min(base * 2 ** ((i + 1) // 2), 256)
            blocks.append(ResBlock1D(c, cout, k, stride=2, p=p))
            c = cout
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(2 * c, embed), nn.ReLU(inplace=True), nn.Dropout(p))
        self.embed_dim = embed

    def forward(self, x):
        h = self.blocks(self.stem(x))
        h = torch.cat([h.mean(-1), h.max(-1).values], dim=1)
        return self.head(h)


class MultiTaskHead(nn.Module):
    """Four heads: stage, volume lost, imminent decompensation, reserve."""

    def __init__(self, dim, p=0.15):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, 96), nn.ReLU(inplace=True), nn.Dropout(p))
        self.stage = nn.Linear(96, N_STAGES)
        self.volume = nn.Linear(96, 1)
        self.decomp = nn.Linear(96, 1)
        self.reserve = nn.Linear(96, 1)

    def forward(self, h):
        z = self.trunk(h)
        return dict(stage=self.stage(z), volume=self.volume(z).squeeze(-1),
                    decomp=self.decomp(z).squeeze(-1), reserve=self.reserve(z).squeeze(-1))


class WaveNet(nn.Module):
    def __init__(self, in_ch, cfg):
        super().__init__()
        c = cfg.models.cnn
        self.enc = Encoder1D(in_ch, c.base_channels, c.n_blocks, c.kernel, c.dropout, c.embed_dim)
        self.head = MultiTaskHead(c.embed_dim, c.dropout)

    def forward(self, x, f=None):
        return self.head(self.enc(x))

    def embed(self, x, f=None):
        return self.enc(x)


class HybridNet(nn.Module):
    """Waveform embedding concatenated with the handcrafted physiological vector."""

    def __init__(self, in_ch, n_feat, cfg):
        super().__init__()
        c = cfg.models.cnn
        self.enc = Encoder1D(in_ch, c.base_channels, c.n_blocks, c.kernel, c.dropout, c.embed_dim)
        h = cfg.models.hybrid.feat_hidden
        self.fnet = nn.Sequential(nn.Linear(n_feat, h), nn.ReLU(inplace=True),
                                  nn.Dropout(c.dropout), nn.Linear(h, h), nn.ReLU(inplace=True))
        self.head = MultiTaskHead(c.embed_dim + h, c.dropout)

    def forward(self, x, f):
        return self.head(torch.cat([self.enc(x), self.fnet(f)], dim=1))

    def embed(self, x, f):
        return torch.cat([self.enc(x), self.fnet(f)], dim=1)


class FeatMLP(nn.Module):
    """Handcrafted features only, same multi-task head -- the neural counterpart
    of the boosted model, used when a like-for-like architecture is needed."""

    def __init__(self, n_feat, cfg):
        super().__init__()
        p = cfg.models.cnn.dropout
        h = cfg.models.hybrid.feat_hidden
        self.fnet = nn.Sequential(nn.Linear(n_feat, 2 * h), nn.ReLU(inplace=True), nn.Dropout(p),
                                  nn.Linear(2 * h, h), nn.ReLU(inplace=True))
        self.head = MultiTaskHead(h, p)

    def forward(self, x, f):
        return self.head(self.fnet(f))

    def embed(self, x, f):
        return self.fnet(f)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
