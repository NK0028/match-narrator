"""Measure live mode on a real clip: speed, how continuous the commentary is, Claude latency.

    python eval/live_eval.py path/to/match.mp4 [--no-llm] [--imgsz 1280]

Writes eval/live_metrics.json and eval/live_spotcheck.csv (30 random lines with their
facts and empty "accurate (y/n)" / "fluent (1-5)" columns to fill in while watching).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from commentator.live.session import LiveSession  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--imgsz", type=int, default=None, help="YOLO input size (live default 1280; 640 misses most balls)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", default="outputs/clips/live-eval")
    ap.add_argument("--label", help="how to name the clip in the metrics (default: its file name)")
    a = ap.parse_args()
    if a.imgsz:
        os.environ["YOLO_IMGSZ"] = str(a.imgsz)

    s = LiveSession("live-eval", a.video, a.out, language=a.lang, use_llm=not a.no_llm, voice=False)
    t0 = time.time()
    s.start()
    while s.thread.is_alive():
        time.sleep(1)
        print(f"\ranalysed {s.processed_t:7.1f}/{s.duration:.0f}s  lines {len(s.lines):4d}  "
              f"{s.processed_t / max(1e-6, time.time() - t0):5.1f}x real time", end="", flush=True)
    print()
    if s.status != "done":
        sys.exit(f"session failed: {s.error}")

    L, dur = s.lines, s.duration
    gaps = [b.t - a_.end for a_, b in zip(L, L[1:])] or [0.0]
    spoken = sum(ln.end - ln.t for ln in L)
    per_min = len(L) / (dur / 60) if dur else 0
    by_type: dict[str, int] = {}
    for ln in L:
        by_type[ln.event] = by_type.get(ln.event, 0) + 1
    metrics = {
        "video": a.label or os.path.basename(a.video), "duration_s": round(dur, 1), "lines": len(L),
        "lines_per_minute": round(per_min, 1),
        "speech_coverage_pct": round(100 * spoken / dur, 1) if dur else 0,
        "gap_p50_s": round(float(np.percentile(gaps, 50)), 2), "gap_p95_s": round(float(np.percentile(gaps, 95)), 2),
        "gap_max_s": round(max(gaps), 2), "overlaps": sum(g < -1e-6 for g in gaps),
        "llm_lines_pct": round(100 * sum(ln.source == "llm" for ln in L) / max(1, len(L)), 1),
        "by_type": by_type, "kits": s.kits, "match_stats": s.stats, **s.perf,
    }
    os.makedirs("eval", exist_ok=True)
    with open("eval/live_metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    rng = random.Random(0)
    with open("eval/live_spotcheck.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "line", "source", "facts", "accurate_y_n", "fluent_1_5"])
        for ln in sorted(rng.sample(L, min(30, len(L))), key=lambda x: x.t):
            w.writerow([f"{int(ln.t // 60):02d}:{ln.t % 60:04.1f}", ln.text, ln.source,
                        json.dumps(s.events[ln.event_index].facts()), "", ""])
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
