"""NetVLAD++ action spotter (Giancola & Ghanem, CVPRW 2021) with an extra team head.

Temporally-aware pooling: frames before and after the window centre are pooled by
separate NetVLAD layers, so the model can tell "just before a goal" from "just after".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    def __init__(self, num_clusters: int, dim: int):
        super().__init__()
        self.k = num_clusters
        self.dim = dim
        self.centroids = nn.Parameter(torch.randn(num_clusters, dim) / math.sqrt(dim))
        self.assign = nn.Linear(dim, num_clusters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        x = F.normalize(x, dim=-1)
        a = F.softmax(self.assign(x), dim=-1)  # (B, T, K)
        vlad = torch.einsum("btk,btd->bkd", a, x) - a.sum(1).unsqueeze(-1) * self.centroids
        vlad = F.normalize(vlad, dim=-1)  # intra-normalisation
        return F.normalize(vlad.flatten(1), dim=-1)  # (B, K*D)


class NetVLADPlusPlus(nn.Module):
    def __init__(self, input_dim: int = 512, num_classes: int = 17, clusters: int = 64,
                 proj_dim: int = 512, dropout: float = 0.4, num_teams: int = 3):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim) if input_dim != proj_dim else nn.Identity()
        self.before = NetVLAD(clusters // 2, proj_dim)
        self.after = NetVLAD(clusters // 2, proj_dim)
        out = clusters * proj_dim
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(out, num_classes)
        self.team = nn.Linear(out, num_teams)
        self.config = dict(input_dim=input_dim, num_classes=num_classes, clusters=clusters,
                           proj_dim=proj_dim, dropout=dropout, num_teams=num_teams)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        mid = x.shape[1] // 2
        z = torch.cat([self.before(x[:, :mid]), self.after(x[:, mid:])], dim=1)
        z = self.drop(z)
        return self.cls(z), self.team(z)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model: NetVLADPlusPlus, path: str, extra: dict | None = None) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> tuple[NetVLADPlusPlus, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = NetVLADPlusPlus(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    return m, ck
