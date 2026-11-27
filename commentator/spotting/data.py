"""SoccerNet-v2 features + labels -> training windows for a NetVLAD++-style spotter.

Features are 2 fps, PCA-512 ResNet (SoccerNet's `*_ResNET_TF2_PCA512.npy`).
A training sample is a window of `window` seconds centred on a frame; its target
is a soft (Gaussian) score per class that peaks when an event sits at the centre,
which is what gives the spotter tight temporal precision.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..schema import CLASS_TO_IDX, SOCCERNET_CLASSES

FPS = 2
NUM_CLASSES = len(SOCCERNET_CLASSES)
TEAMS = {"home": 0, "away": 1}  # anything else -> 2 ("not applicable")


@dataclass
class Half:
    game: str
    half: int  # 1 or 2
    feats: np.ndarray  # (T, D) float16
    events: list[tuple[int, int, int]]  # (frame, class_idx, team_idx)


def load_half_labels(label_path: str, half: int, n_frames: int) -> list[tuple[int, int, int]]:
    with open(label_path) as f:
        ann = json.load(f)["annotations"]
    out = []
    for a in ann:
        h = int(a["gameTime"].split(" - ")[0])
        if h != half or a["label"] not in CLASS_TO_IDX:
            continue
        frame = min(int(int(a["position"]) / 1000 * FPS), n_frames - 1)
        out.append((frame, CLASS_TO_IDX[a["label"]], TEAMS.get(a.get("team", ""), 2)))
    return out


def load_split(root: str, split: str, feature: str = "ResNET_TF2_PCA512",
               max_games: int = 0) -> list[Half]:
    from SoccerNet.utils import getListGames

    halves = []
    games = getListGames(split)
    if max_games:
        games = games[:max_games]
    for g in games:
        d = os.path.join(root, g)
        lab = os.path.join(d, "Labels-v2.json")
        for h in (1, 2):
            fp = os.path.join(d, f"{h}_{feature}.npy")
            if not (os.path.exists(fp) and os.path.exists(lab)):
                continue
            feats = np.load(fp).astype(np.float16)
            halves.append(Half(g, h, feats, load_half_labels(lab, h, len(feats))))
    return halves


def window_at(feats: np.ndarray, center: int, size: int) -> np.ndarray:
    """(size, D) window centred on `center`, edge-padded by replication."""
    half = size // 2
    idx = np.clip(np.arange(center - half, center - half + size), 0, len(feats) - 1)
    return feats[idx]


class SpottingWindows(Dataset):
    """Each epoch: every event (jittered) + `neg_ratio` x as many random frames."""

    def __init__(self, halves: list[Half], window_s: float = 15, sigma_s: float = 1.5,
                 neg_ratio: float = 2.0, jitter_s: float = 2.0, seed: int = 0):
        self.halves = halves
        self.size = int(window_s * FPS)
        self.sigma = sigma_s * FPS
        self.neg_ratio = neg_ratio
        self.jitter = int(jitter_s * FPS)
        self.rng = np.random.default_rng(seed)
        # per-half dense lookup: event frames per class for fast soft targets
        self._ev = [np.array(h.events, dtype=np.int64).reshape(-1, 3) for h in halves]
        self.resample()

    def resample(self) -> None:
        items = []
        for hi, ev in enumerate(self._ev):
            for fr, _, _ in ev:
                j = self.rng.integers(-self.jitter, self.jitter + 1)
                items.append((hi, int(fr + j)))
        n_neg = int(len(items) * self.neg_ratio)
        hs = self.rng.integers(0, len(self.halves), n_neg)
        for hi in hs:
            items.append((int(hi), int(self.rng.integers(0, len(self.halves[hi].feats)))))
        self.items = [(hi, int(np.clip(c, 0, len(self.halves[hi].feats) - 1))) for hi, c in items]

    def __len__(self) -> int:
        return len(self.items)

    def targets(self, hi: int, center: int) -> tuple[np.ndarray, int]:
        ev = self._ev[hi]
        y = np.zeros(NUM_CLASSES, dtype=np.float32)
        team = -100  # ignore index
        if len(ev):
            d = ev[:, 0] - center
            near = np.abs(d) <= self.size // 2
            if near.any():
                w = np.exp(-(d[near] ** 2) / (2 * self.sigma ** 2))
                for wi, c in zip(w, ev[near, 1]):
                    y[c] = max(y[c], wi)
                k = np.argmin(np.abs(d))
                if abs(d[k]) <= 2 * self.sigma:
                    team = int(ev[k, 2])
        return y, team

    def __getitem__(self, i):
        hi, c = self.items[i]
        x = window_at(self.halves[hi].feats, c, self.size).astype(np.float32)
        y, team = self.targets(hi, c)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(team)


def class_pos_weight(halves: list[Half], cap: float = 20.0) -> torch.Tensor:
    """Inverse-frequency weights for BCE — counters goal/card rarity."""
    counts = np.zeros(NUM_CLASSES)
    for h in halves:
        for _, c, _ in h.events:
            counts[c] += 1
    w = counts.max() / np.maximum(counts, 1)
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)
