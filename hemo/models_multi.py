"""Heads for the multi-aetiology task.  Kept separate from `models.py` so the
haemorrhage arm's modules are never touched."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .models import Encoder1D
from .multi import AETIOLOGIES

N_AET = len(AETIOLOGIES)


class AetHead(nn.Module):
    """Primary aetiology, per-aetiology presence (multi-label), severity, compromise."""

    def __init__(self, dim, p=0.15):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(inplace=True), nn.Dropout(p))
        self.aet = nn.Linear(128, N_AET)
        self.present = nn.Linear(128, N_AET)
        self.severity = nn.Linear(128, 1)
        self.compromise = nn.Linear(128, 1)

    def forward(self, h):
        z = self.trunk(h)
        return dict(aet=self.aet(z), present=self.present(z),
                    severity=self.severity(z).squeeze(-1),
                    compromise=self.compromise(z).squeeze(-1))


class AetNet(nn.Module):
    def __init__(self, in_ch, cfg):
        super().__init__()
        c = cfg.models.cnn
        self.enc = Encoder1D(in_ch, c.base_channels, c.n_blocks, c.kernel, c.dropout, c.embed_dim)
        self.head = AetHead(c.embed_dim, c.dropout)

    def forward(self, x, f=None):
        return self.head(self.enc(x))

    def embed(self, x, f=None):
        return self.enc(x)


class HybridAetNet(nn.Module):
    def __init__(self, in_ch, n_feat, cfg):
        super().__init__()
        c = cfg.models.cnn
        h = cfg.models.hybrid.feat_hidden
        self.enc = Encoder1D(in_ch, c.base_channels, c.n_blocks, c.kernel, c.dropout, c.embed_dim)
        self.fnet = nn.Sequential(nn.Linear(n_feat, h), nn.ReLU(inplace=True),
                                  nn.Dropout(c.dropout), nn.Linear(h, h), nn.ReLU(inplace=True))
        self.head = AetHead(c.embed_dim + h, c.dropout)

    def forward(self, x, f):
        return self.head(torch.cat([self.enc(x), self.fnet(f)], dim=1))

    def embed(self, x, f):
        return torch.cat([self.enc(x), self.fnet(f)], dim=1)


class FeatAetNet(nn.Module):
    def __init__(self, n_feat, cfg):
        super().__init__()
        p = cfg.models.cnn.dropout
        h = cfg.models.hybrid.feat_hidden
        self.fnet = nn.Sequential(nn.Linear(n_feat, 2 * h), nn.ReLU(inplace=True), nn.Dropout(p),
                                  nn.Linear(2 * h, h), nn.ReLU(inplace=True))
        self.head = AetHead(h, p)

    def forward(self, x, f):
        return self.head(self.fnet(f))

    def embed(self, x, f):
        return self.fnet(f)
