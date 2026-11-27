"""Run the spotter over a split and score it with SoccerNet's official average-mAP.

    python -m commentator.spotting.evaluate --ckpt models/spotter.pt --split test
Writes predictions to outputs/predictions/<game>/results_spotting.json and a
metrics JSON (tight + loose, per class) to eval/spotting_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..schema import SOCCERNET_CLASSES
from .data import FPS
from .infer import spot_half, to_soccernet_json
from .model import load_checkpoint, pick_device


def predict_split(ckpt: str, root: str, split: str, out_dir: str, max_games: int = 0) -> list[str]:
    from SoccerNet.utils import getListGames

    device = pick_device()
    model, ck = load_checkpoint(ckpt, device)
    window = int(ck.get("window_s", 15) * FPS)
    feature = ck.get("feature", "ResNET_TF2_PCA512")
    games = getListGames(split)[: max_games or None]
    done = []
    for i, g in enumerate(games):
        by_half = {}
        for h in (1, 2):
            fp = os.path.join(root, g, f"{h}_{feature}.npy")
            if not os.path.exists(fp):
                break
            by_half[h] = spot_half(model, np.load(fp), device, window)
        if len(by_half) < 2:
            continue
        od = os.path.join(out_dir, g)
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "results_spotting.json"), "w") as f:
            json.dump(to_soccernet_json(g, by_half), f)
        done.append(g)
        print(f"[{i+1}/{len(games)}] {g}", flush=True)
    return done


def official_map(root: str, pred_dir: str, games: list[str]) -> dict:
    """SoccerNet's official average-mAP over `games` (a subset of a split is fine)."""
    import tempfile

    from SoccerNet.Evaluation.ActionSpotting import evaluate
    from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2

    lab_dir = tempfile.mkdtemp(prefix="sn_labels_")
    for g in games:  # labels-only mirror so the evaluator infers exactly these games
        os.makedirs(os.path.join(lab_dir, g), exist_ok=True)
        os.symlink(os.path.abspath(os.path.join(root, g, "Labels-v2.json")),
                   os.path.join(lab_dir, g, "Labels-v2.json"))
    out = {"n_games": len(games)}
    for metric in ("tight", "loose"):
        r = evaluate(SoccerNet_path=lab_dir, Predictions_path=pred_dir, dataset=None,
                     EVENT_DICTIONARY=EVENT_DICTIONARY_V2, version=2,
                     prediction_file="results_spotting.json", metric=metric)
        out[metric] = {
            "a_mAP": float(r["a_mAP"]),
            "a_mAP_visible": float(r["a_mAP_visible"]),
            "a_mAP_unshown": float(r["a_mAP_unshown"]),
            "per_class": {c: float(v) for c, v in zip(SOCCERNET_CLASSES, r["a_mAP_per_class"])},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/spotter.pt")
    ap.add_argument("--root", default="data/SoccerNet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-dir", default="outputs/predictions")
    ap.add_argument("--out", default="eval/spotting_metrics.json")
    ap.add_argument("--max-games", type=int, default=0)
    args = ap.parse_args()

    games = predict_split(args.ckpt, args.root, args.split, args.pred_dir, args.max_games)
    res = official_map(args.root, args.pred_dir, games)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"evaluated on {res['n_games']} {args.split} games")
    for m in ("tight", "loose"):
        print(f"{m:5s} average-mAP: {100*res[m]['a_mAP']:.2f}")
    print("per-class (loose):")
    for c, v in sorted(res["loose"]["per_class"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:20s} {100*v:6.2f}")


if __name__ == "__main__":
    main()
