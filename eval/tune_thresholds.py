"""Week 7 robustness: per-class narration thresholds tuned on the valid split.

For each SoccerNet class pick the confidence threshold maximising F-beta (beta<1
favours precision: a false "GOAL!" is worse than a missed throw-in) with a ±5 s
match tolerance. Writes models/thresholds.json, read by the pipeline.

    python eval/tune_thresholds.py --ckpt models/spotter.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commentator.schema import SOCCERNET_CLASSES  # noqa: E402
from commentator.spotting.data import FPS, load_split  # noqa: E402
from commentator.spotting.infer import spot_half  # noqa: E402
from commentator.spotting.model import load_checkpoint, pick_device  # noqa: E402


def match(preds, gts, tol):
    """Greedy one-to-one matching by confidence. Returns (tp flags per pred, n_gt)."""
    used, flags = set(), []
    for p in sorted(preds, key=lambda x: -x[1]):
        hit = None
        for j, g in enumerate(gts):
            if j not in used and abs(g - p[0]) <= tol:
                hit = j
                break
        if hit is not None:
            used.add(hit)
        flags.append((p[1], hit is not None))
    return flags, len(gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/spotter.pt")
    ap.add_argument("--root", default="data/SoccerNet")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--out", default="models/thresholds.json")
    a = ap.parse_args()

    device = pick_device()
    model, ck = load_checkpoint(a.ckpt, device)
    window = int(ck.get("window_s", 15) * FPS)
    halves = load_split(a.root, a.split)
    if not halves:  # fall back to the held-out tail of train (as in train.py)
        halves = load_split(a.root, "train")
        halves = halves[-max(2, int(len(halves) * 0.15)) // 2 * 2:]
    per_class = {c: [] for c in SOCCERNET_CLASSES}
    n_gt = {c: 0 for c in SOCCERNET_CLASSES}
    for h in halves:
        preds = spot_half(model, h.feats.astype(np.float32), device, window, threshold=0.1)
        for ci, c in enumerate(SOCCERNET_CLASSES):
            ps = [(p["frame"] / FPS, p["confidence"]) for p in preds if p["label"] == c]
            gs = [fr / FPS for fr, k, _ in h.events if k == ci]
            f, n = match(ps, gs, a.tol)
            per_class[c] += f
            n_gt[c] += n

    th, report = {}, {}
    b2 = a.beta ** 2
    for c in SOCCERNET_CLASSES:
        flags = sorted(per_class[c], key=lambda x: -x[0])
        best = (0.0, 0.9, 0.0, 0.0)
        tp = 0
        for k, (conf, hit) in enumerate(flags, 1):
            tp += hit
            p, r = tp / k, tp / max(n_gt[c], 1)
            f = (1 + b2) * p * r / max(b2 * p + r, 1e-9)
            if f > best[0]:
                best = (f, conf, p, r)
        # rare classes (a handful of valid examples) can't support a tuned threshold
        th[c] = round(max(0.6 if n_gt[c] < 10 else 0.15, float(best[1])), 3)
        report[c] = {"threshold": th[c], "F": round(best[0], 3), "precision": round(best[2], 3),
                     "recall": round(best[3], 3), "n_gt": n_gt[c]}
        print(f"{c:20s} thr {th[c]:.2f}  P {best[2]:.2f}  R {best[3]:.2f}  (n={n_gt[c]})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(th, f, indent=1)
    with open("eval/thresholds_report.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
