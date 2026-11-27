"""Dense sliding-window inference + temporal NMS -> spotted events."""
from __future__ import annotations

import numpy as np
import torch

from ..schema import SOCCERNET_CLASSES, SOCCERNET_TO_EVENT, Event
from .data import FPS, window_at

TEAM_NAMES = ["home", "away", None]

# Per-class NMS windows (seconds). Rare, isolated events get wide windows; frequent
# ones (throw-ins, ball out) narrow so neighbours survive.
NMS_WINDOW_S = {c: 20.0 for c in SOCCERNET_CLASSES}
NMS_WINDOW_S.update({"Ball out of play": 8.0, "Throw-in": 8.0, "Clearance": 8.0, "Foul": 10.0})


@torch.no_grad()
def frame_scores(model, feats: np.ndarray, window: int, device, batch: int = 512, stride: int = 1):
    """Score every `stride`-th frame. Returns (scores (T,C), team_probs (T,3))."""
    model.eval()
    centers = np.arange(0, len(feats), stride)
    cls_out, team_out = [], []
    for i in range(0, len(centers), batch):
        xb = np.stack([window_at(feats, c, window) for c in centers[i:i + batch]]).astype(np.float32)
        logits, team = model(torch.from_numpy(xb).to(device))
        cls_out.append(torch.sigmoid(logits).float().cpu().numpy())
        team_out.append(torch.softmax(team, -1).float().cpu().numpy())
    s, t = np.concatenate(cls_out), np.concatenate(team_out)
    if stride > 1:  # upsample back to every frame
        idx = np.minimum(np.arange(len(feats)) // stride, len(s) - 1)
        s, t = s[idx], t[idx]
    return s, t


def temporal_nms(scores: np.ndarray, window_frames: int, threshold: float) -> list[tuple[int, float]]:
    """Greedy 1-D NMS on one class's score curve -> [(frame, score)]."""
    s = scores.copy()
    keep = []
    while True:
        i = int(np.argmax(s))
        v = float(s[i])
        if v < threshold:
            break
        keep.append((i, v))
        lo, hi = max(0, i - window_frames // 2), min(len(s), i + window_frames // 2 + 1)
        s[lo:hi] = -1
    return keep


def spot_half(model, feats: np.ndarray, device, window: int, threshold: float = 0.05,
              stride: int = 1) -> list[dict]:
    """Raw SoccerNet-style predictions for one half (low threshold, for mAP)."""
    scores, team = frame_scores(model, feats, window, device, stride=stride)
    preds = []
    for c, name in enumerate(SOCCERNET_CLASSES):
        for fr, conf in temporal_nms(scores[:, c], int(NMS_WINDOW_S[name] * FPS), threshold):
            k = int(np.argmax(team[fr]))
            preds.append(dict(frame=fr, label=name, confidence=conf,
                              team=TEAM_NAMES[k], team_conf=round(float(team[fr, k]), 3)))
    return sorted(preds, key=lambda p: p["frame"])


def to_soccernet_json(game: str, preds_by_half: dict[int, list[dict]]) -> dict:
    out = []
    for half, preds in preds_by_half.items():
        for p in preds:
            ms = int(p["frame"] / FPS * 1000)
            sec = ms // 1000
            out.append({
                "gameTime": f"{half} - {sec // 60:02d}:{sec % 60:02d}",
                "label": p["label"], "position": str(ms), "half": str(half),
                "confidence": str(p["confidence"]), "team": p.get("team"), "team_conf": p.get("team_conf"),
            })
    return {"UrlLocal": game, "predictions": out}


def to_events(preds: list[dict], offset_s: float = 0.0, half: int | None = None,
              thresholds: dict[str, float] | None = None, default_threshold: float = 0.5,
              team_min_conf: float = 0.0) -> list[Event]:
    """Filter raw predictions by (per-class) confidence and map to commentary events."""
    thresholds = thresholds or {}
    evs = []
    for p in preds:
        if p["confidence"] < thresholds.get(p["label"], default_threshold):
            continue
        t = p["frame"] / FPS - offset_s
        if t < 0:
            continue
        team = p.get("team") if (p.get("team_conf") or 1.0) >= team_min_conf else None
        evs.append(Event(event=SOCCERNET_TO_EVENT[p["label"]], t=t, team=team,
                         confidence=round(min(1.0, p["confidence"]), 3),
                         source_label=p["label"], half=half))
    return evs
