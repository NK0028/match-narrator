"""Train the NetVLAD++ spotter on SoccerNet precomputed features.

    python -m commentator.spotting.train --epochs 30 --out models/spotter.pt
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import NUM_CLASSES, SpottingWindows, class_pos_weight, load_split
from .model import NetVLADPlusPlus, pick_device, save_checkpoint


def run_epoch(model, loader, device, pos_weight, opt=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for x, y, team in loader:
        x, y, team = x.to(device), y.to(device), team.to(device)
        with torch.set_grad_enabled(train):
            logits, tl = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            if (team >= 0).any():
                loss = loss + 0.3 * F.cross_entropy(tl, team, ignore_index=-100)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot += loss.item() * len(x)
        n += len(x)
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/SoccerNet")
    ap.add_argument("--feature", default="ResNET_TF2_PCA512")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=float, default=15.0)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--out", default="models/spotter.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    t0 = time.time()
    train_h = load_split(args.root, "train", args.feature, args.max_games)
    valid_h = load_split(args.root, "valid", args.feature, args.max_games)
    if not valid_h:  # valid split not downloaded yet: hold out 15% of train games
        cut = max(2, int(len(train_h) * 0.15)) // 2 * 2
        train_h, valid_h = train_h[:-cut], train_h[-cut:]
    print(f"loaded {len(train_h)} train / {len(valid_h)} valid halves in {time.time()-t0:.0f}s on {device}")
    dim = train_h[0].feats.shape[1]

    train_ds = SpottingWindows(train_h, window_s=args.window, seed=0)
    valid_ds = SpottingWindows(valid_h, window_s=args.window, seed=1, jitter_s=0)
    pw = class_pos_weight(train_h).to(device)
    print("pos_weight", np.round(pw.cpu().numpy(), 1).tolist())

    model = NetVLADPlusPlus(input_dim=dim, num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=2)
    vl_loader = DataLoader(valid_ds, batch_size=args.batch, num_workers=0)

    best, bad, history = float("inf"), 0, []
    for ep in range(1, args.epochs + 1):
        train_ds.resample()
        tr_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
        tr = run_epoch(model, tr_loader, device, pw, opt)
        vl = run_epoch(model, vl_loader, device, pw)
        sched.step(vl)
        history.append({"epoch": ep, "train_loss": tr, "valid_loss": vl, "lr": opt.param_groups[0]["lr"]})
        print(f"ep {ep:02d} train {tr:.4f} valid {vl:.4f} lr {opt.param_groups[0]['lr']:.1e} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            save_checkpoint(model, args.out, {"window_s": args.window, "feature": args.feature,
                                              "epoch": ep, "valid_loss": vl})
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break
    with open(args.out.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f, indent=1)
    print(f"best valid loss {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
