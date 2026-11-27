"""Extract 2 fps ResNet-152 frame features from a raw clip (for clips without
SoccerNet precomputed features).

SoccerNet's features were made with a TF2/Keras ResNet-152 + PCA-512. If you place
SoccerNet's `pca_512_TF2.pkl` and `average_512_TF2.pkl` (from the sn-spotting repo)
in models/, the same PCA is applied; torchvision weights still differ slightly, so
treat spotting on raw uploads as experimental and prefer fine-tuning on your own
extracted features.
"""
from __future__ import annotations

import os
import pickle

import cv2
import numpy as np
import torch

from .data import FPS


PCA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def has_pca(pca_dir: str = PCA_DIR) -> bool:
    return all(os.path.exists(os.path.join(pca_dir, f)) for f in ("pca_512_TF2.pkl", "average_512_TF2.pkl"))


def extract(video: str, device=None, batch: int = 32, pca_dir: str = PCA_DIR,
            t0: float = 0.0, t1: float | None = None) -> np.ndarray:
    import torchvision

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    net = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(n / fps, t1) if t1 is not None else n / fps
    want = [int(round(i * fps / FPS)) for i in range(int(t0 * FPS), int(end * FPS))]
    fi = want[0] if want else 0
    if fi:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    frames, feats, wi = [], [], 0
    while wi < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if fi == want[wi]:
            frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (224, 224)))
            wi += 1
            if len(frames) == batch:
                feats.append(_run(net, frames, mean, std, device))
                frames = []
        fi += 1
    if frames:
        feats.append(_run(net, frames, mean, std, device))
    cap.release()
    X = np.concatenate(feats) if feats else np.zeros((0, 2048), np.float32)

    pca_p, avg_p = os.path.join(pca_dir, "pca_512_TF2.pkl"), os.path.join(pca_dir, "average_512_TF2.pkl")
    if os.path.exists(pca_p) and os.path.exists(avg_p):
        with open(pca_p, "rb") as f:
            pca = pickle.load(f)
        with open(avg_p, "rb") as f:
            avg = pickle.load(f)
        X = pca.transform(X - avg)
    return X.astype(np.float32)


@torch.no_grad()
def _run(net, frames, mean, std, device):
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255
    x = ((x - mean) / std).to(device)
    return net(x).cpu().numpy()
