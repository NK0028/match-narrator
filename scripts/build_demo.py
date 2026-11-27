"""Build the demo clips shown by the web app (web/public/demo/<id>/ + index.json).

Each clip = a window of a SoccerNet *test* game (never seen in training):
  - commentary from the trained spotter's predictions on SoccerNet features
  - ground-truth events stored alongside for the timeline comparison
  - a synthetic top-down match view video (licensing-safe stand-in for footage)

    python scripts/build_demo.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from commentator.pipeline import ground_truth_events, run_soccernet  # noqa: E402

DEMOS = [
    # (game, half, start_s, end_s, title)
    ("england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea", 1, 1080, 1560,
     "West Brom v Chelsea · 18'–26'"),
    ("england_epl/2015-2016/2015-09-20 - 18-00 Southampton 2 - 3 Manchester United", 1, 1800, 2160,
     "Southampton v Man Utd · 30'–36'"),
    ("england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester", 1, 2100, 2520,
     "Man Utd v Leicester · 35'–42'"),
]
OUT = os.path.join("web", "public", "demo")


def main():
    os.makedirs(OUT, exist_ok=True)
    index = []
    for game, half, start, end, title in DEMOS:
        tr = run_soccernet(game, half, start, end, out_root="outputs/demo_build", voice=True)
        src = os.path.join("outputs/demo_build", tr.clip_id)
        dst = os.path.join(OUT, tr.clip_id)
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        subprocess.run([sys.executable, "scripts/render_demo_clip.py", game, "--half", str(half),
                        "--start", str(start), "--end", str(end), "--out", os.path.join(dst, "clip.mp4")], check=True)
        gt = ground_truth_events(os.path.join("data/SoccerNet", game, "Labels-v2.json"), half, start, end)
        tr.video = "clip.mp4"
        tr.stats["title"] = title
        tr.stats["source"] = {"game": game, "half": half, "start_s": start, "end_s": end}
        tr.stats["ground_truth"] = [json.loads(e.model_dump_json()) for e in gt]
        for ln in tr.lines:
            ln.audio = None  # per-line wavs aren't shipped; the mixed track is
        for f in ("commentary.m4a", "captions.vtt"):
            shutil.copy(os.path.join(src, f), dst)
        with open(os.path.join(dst, "track.json"), "w") as f:
            f.write(tr.model_dump_json())
        index.append({"id": tr.clip_id, "title": title, "duration": tr.duration,
                      "n_events": len(tr.events), "n_lines": len(tr.lines)})
        print(f"built {tr.clip_id}: {len(tr.lines)} lines")
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
