"""Download SoccerNet-v2 action-spotting labels + precomputed features.

Labels and features are public; raw videos need the SoccerNet NDA password and
are NOT downloaded here (the spotter trains on features only).

    python scripts/download_soccernet.py --splits train valid test --workers 8
    python scripts/download_soccernet.py --splits test --max-games 5   # quick start
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

DEFAULT_FILES = ["Labels-v2.json", "1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/SoccerNet")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--max-games", type=int, default=0, help="per split; 0 = all")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = []
    for split in args.splits:
        games = getListGames(split)
        if args.max_games:
            games = games[: args.max_games]
        for g in games:
            missing = [f for f in args.files if not os.path.exists(os.path.join(args.root, g, f))]
            if missing:
                jobs.append((split, g, missing))
    print(f"{len(jobs)} games need files")

    def fetch(job):
        _, game, files = job
        d = SoccerNetDownloader(LocalDirectory=args.root)
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            d.downloadGame(game=game, files=files)
        return game

    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, j) for j in jobs]
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:  # keep going; rerun the script to retry
                print("FAILED", e)
            if done % 10 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
