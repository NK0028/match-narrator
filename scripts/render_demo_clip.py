"""Render a licensing-safe 2-D "match view" video for a window of a SoccerNet game.

Broadcast footage is copyrighted (and SoccerNet videos are NDA-only), so the hosted
demo plays this animated top-down visualisation instead. Player motion is
synthetic; the ball is scripted toward the ground-truth event locations
(corners to the flag, goals into the net, ...), and the scoreboard clock is the
real match clock. Commentary comes from the *spotter's predictions* on SoccerNet
features for the same window — the video does not leak them.

    python scripts/render_demo_clip.py "<game>" --half 2 --start 1200 --end 1500 --out web/public/demo/clip.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commentator.tts.engine import ffmpeg_exe  # noqa: E402

W, H, FPS = 960, 600, 15
PX0, PY0, PX1, PY1 = 60, 80, 900, 560  # pitch rectangle
HOME_C, AWAY_C = (235, 235, 235), (40, 40, 200)  # BGR: white vs red


def pitch(img):
    img[:] = (40, 110, 40)
    for i in range(10):
        x0 = PX0 + i * (PX1 - PX0) // 10
        if i % 2:
            img[PY0:PY1, x0:x0 + (PX1 - PX0) // 10] = (46, 122, 46)
    c, t = (230, 230, 230), 2
    cv2.rectangle(img, (PX0, PY0), (PX1, PY1), c, t)
    mx, my = (PX0 + PX1) // 2, (PY0 + PY1) // 2
    cv2.line(img, (mx, PY0), (mx, PY1), c, t)
    cv2.circle(img, (mx, my), 60, c, t)
    for side in (0, 1):
        x = PX0 if side == 0 else PX1
        d = 1 if side == 0 else -1
        cv2.rectangle(img, (x, my - 110), (x + d * 120, my + 110), c, t)
        cv2.rectangle(img, (x, my - 45), (x + d * 40, my + 45), c, t)
        cv2.rectangle(img, (x - d * 12, my - 25), (x, my + 25), (255, 255, 255), -1)


def to_px(xy):
    return int(PX0 + xy[0] * (PX1 - PX0)), int(PY0 + xy[1] * (PY1 - PY0))


def ball_targets(events, start, home_attacks_right):
    """Keyframes (t, x, y) in pitch coords [0,1] for the ball."""
    keys = [(0.0, 0.5, 0.5)]
    for e in events:
        t = e["t"]
        right = (e.get("team") == "home") == home_attacks_right
        gx = 1.0 if right else 0.0
        lab = e["label"]
        if lab == "Goal":
            keys += [(t - 2.0, abs(gx - 0.15), 0.45), (t, gx + (0.01 if right else -0.01), 0.5), (t + 6, 0.5, 0.5)]
        elif lab.startswith("Shots"):
            keys += [(t - 1.5, abs(gx - 0.2), 0.4), (t, abs(gx - 0.03), 0.55 if "off" in lab else 0.5)]
        elif lab == "Corner":
            keys += [(t, gx, 0.0 if int(t) % 2 else 1.0), (t + 3, abs(gx - 0.08), 0.5)]
        elif lab == "Kick-off":
            keys += [(t, 0.5, 0.5)]
        elif lab in ("Throw-in", "Ball out of play"):
            keys += [(t, 0.3 + 0.4 * ((int(t) * 7) % 10) / 10, 0.0 if int(t) % 2 else 1.0)]
        elif lab in ("Foul", "Direct free-kick", "Indirect free-kick", "Offside"):
            keys += [(t, 0.35 + 0.3 * ((int(t) * 3) % 10) / 10, 0.3 + 0.4 * ((int(t) * 5) % 10) / 10)]
    keys.sort()
    return np.array(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--root", default="data/SoccerNet")
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=300)
    ap.add_argument("--out", default="outputs/demo/clip.mp4")
    a = ap.parse_args()

    with open(os.path.join(a.root, a.game, "Labels-v2.json")) as f:
        lab = json.load(f)
    home, away = lab["gameHomeTeam"], lab["gameAwayTeam"]
    evs, score = [], [0, 0]
    for x in lab["annotations"]:
        h = int(x["gameTime"].split(" - ")[0])
        t = int(x["position"]) / 1000
        if x["label"] == "Goal" and (h < a.half or (h == a.half and t < a.start)):
            score[0 if x["team"] == "home" else 1] += 1
        if h == a.half and a.start <= t < a.end:
            evs.append({"t": t - a.start, "label": x["label"], "team": x.get("team")})
    keys = ball_targets(evs, a.start, home_attacks_right=a.half == 1)

    rng = np.random.default_rng(0)
    base = np.concatenate([
        np.column_stack([rng.uniform(0.1, 0.5, 11), rng.uniform(0.1, 0.9, 11)]),
        np.column_stack([rng.uniform(0.5, 0.9, 11), rng.uniform(0.1, 0.9, 11)]),
    ])
    base[0], base[11] = (0.04, 0.5), (0.96, 0.5)  # keepers
    if a.half == 2:
        base[:, 0] = 1 - base[:, 0]
    phase = rng.uniform(0, 2 * np.pi, (22, 2))
    freq = rng.uniform(0.05, 0.15, (22, 2))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    canvas = np.zeros((H, W, 3), np.uint8)
    pitch(canvas)
    n = int((a.end - a.start) * FPS)
    goals_seen = [e for e in evs if e["label"] == "Goal"]
    for i in range(n):
        t = i / FPS
        img = canvas.copy()
        bx = np.interp(t, keys[:, 0], keys[:, 1])
        by = np.interp(t, keys[:, 0], keys[:, 2])
        drift = 0.06 * np.sin(2 * np.pi * freq * t + phase)
        pos = base + drift
        pos[1:11] += 0.35 * (np.array([bx, by]) - pos[1:11]) * 0.5  # outfielders lean to the ball
        pos[12:22] += 0.35 * (np.array([bx, by]) - pos[12:22]) * 0.5
        for k, p in enumerate(np.clip(pos, 0.01, 0.99)):
            col = HOME_C if k < 11 else AWAY_C
            if k in (0, 11):
                col = (0, 200, 230) if k == 0 else (200, 120, 0)
            cv2.circle(img, to_px(p), 9, col, -1)
            cv2.circle(img, to_px(p), 9, (20, 20, 20), 1)
        cv2.circle(img, to_px((bx, by)), 5, (255, 255, 255), -1)
        cv2.circle(img, to_px((bx, by)), 5, (0, 0, 0), 1)
        s = list(score)
        for g in goals_seen:
            if t >= g["t"]:
                s[0 if g["team"] == "home" else 1] += 1
        clock = a.start + t + (45 * 60 if a.half == 2 else 0)
        cv2.rectangle(img, (0, 0), (W, 56), (25, 25, 25), -1)
        txt = f"{home}  {s[0]} - {s[1]}  {away}"
        cv2.putText(img, txt, (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{int(clock // 60):02d}:{int(clock % 60):02d}", (W - 110, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "synthetic match view - not broadcast footage", (PX0, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 200), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", "-crf", "28", "-movflags", "+faststart", a.out], check=True)
    os.remove(tmp)
    print(a.out, f"{n} frames")


if __name__ == "__main__":
    main()
