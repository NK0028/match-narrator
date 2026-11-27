"""End-to-end: match video/features -> events -> context -> commentary -> voice.

    # SoccerNet game (features) narrated over a window of a half
    python -m commentator.pipeline soccernet "england_epl/2014-2015/..." --half 1 --start 0 --end 600

    # Your own clip with an events sidecar (or let the spotter run on extracted features)
    python -m commentator.pipeline video clip.mp4 --events clip.events.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time

import numpy as np

from .generation.generator import CommentaryGenerator, dedupe
from .schema import CommentaryTrack, Event, MatchInfo
from .spotting.data import FPS
from .tts.engine import TTS, encode_m4a, mix_track, mux, to_vtt, voice_lines

log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPOTTER = os.path.join(ROOT, "models", "spotter.pt")

# Confidence thresholds per SoccerNet class for narration (tuned on valid; see eval/).
NARRATION_THRESHOLDS = {"Goal": 0.45, "Penalty": 0.5, "Red card": 0.5, "Yellow->red card": 0.5}
DEFAULT_THRESHOLD = 0.5
# The spotter's team head is at chance on test (see eval/REPORT.md), so by default we
# don't let it name a side; team comes from the jersey-colour context layer instead.
SPOTTER_TEAM_MIN_CONF = float(os.environ.get("SPOTTER_TEAM_MIN_CONF", "1.01"))


def load_thresholds(path: str = os.path.join(ROOT, "models", "thresholds.json")) -> dict[str, float]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return NARRATION_THRESHOLDS


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80]


def spot_features(feats: np.ndarray, ckpt: str = SPOTTER, half: int | None = None,
                  start: float = 0.0, end: float | None = None, thresholds: dict | None = None) -> list[Event]:
    from .spotting.infer import spot_half, to_events
    from .spotting.model import load_checkpoint, pick_device

    device = pick_device()
    model, ck = load_checkpoint(ckpt, device)
    window = int(ck.get("window_s", 15) * FPS)
    lo, hi = int(start * FPS), int((end if end is not None else len(feats) / FPS) * FPS)
    preds = spot_half(model, feats[lo:hi], device, window, threshold=0.2)
    return to_events(preds, half=half, thresholds=thresholds or load_thresholds(),
                     default_threshold=DEFAULT_THRESHOLD, team_min_conf=SPOTTER_TEAM_MIN_CONF)


def ground_truth_events(label_path: str, half: int, start: float = 0, end: float = 1e9) -> list[Event]:
    from .schema import SOCCERNET_TO_EVENT

    with open(label_path) as f:
        ann = json.load(f)["annotations"]
    out = []
    for a in ann:
        h = int(a["gameTime"].split(" - ")[0])
        t = int(a["position"]) / 1000
        if h == half and start <= t < end and a["label"] in SOCCERNET_TO_EVENT:
            team = a.get("team") if a.get("team") in ("home", "away") else None
            out.append(Event(event=SOCCERNET_TO_EVENT[a["label"]], t=t - start, team=team,
                             source_label=a["label"], half=h, meta={"visibility": a.get("visibility")}))
    return sorted(out, key=lambda e: e.t)


def add_score(events: list[Event], match: MatchInfo) -> None:
    """Running score from spotted goals (only when the scorer's team is known)."""
    score = {"home": 0, "away": 0}
    for e in events:
        if e.event == "goal" and e.team in score:
            score[e.team] += 1
            e.meta["score"] = f"{match.home} {score['home']}-{score['away']} {match.away}"


def build_track(clip_id: str, events: list[Event], duration: float, out_dir: str,
                match: MatchInfo | None = None, video: str | None = None, voice: bool = True,
                use_llm: bool | None = None, language: str = "en") -> CommentaryTrack:
    t0 = time.time()
    match = match or MatchInfo(language=language)
    os.makedirs(out_dir, exist_ok=True)
    events = dedupe(events)
    add_score(events, match)
    gen = CommentaryGenerator(use_llm=use_llm, language=language)
    lines = gen.generate(events, match, duration)
    t_gen = time.time() - t0
    track = CommentaryTrack(clip_id=clip_id, duration=duration, match=match, events=events, lines=lines)
    if voice:
        tts = TTS(language=language)
        if tts.backend != "none":
            audio_dir = os.path.join(out_dir, "audio")
            voice_lines(lines, audio_dir, tts)
            wav = mix_track(lines, audio_dir, duration, os.path.join(out_dir, "commentary.wav"))
            track.audio_track = os.path.basename(encode_m4a(wav, os.path.join(out_dir, "commentary.m4a")))
            if video:
                mux(video, os.path.join(out_dir, "commentary.wav"), os.path.join(out_dir, "narrated.mp4"))
    if video:
        track.video = os.path.basename(video) if os.path.dirname(os.path.abspath(video)) == os.path.abspath(out_dir) else video
    track.stats = {
        "n_events": len(events), "n_lines": len(lines),
        "llm_lines": sum(ln.source == "llm" for ln in lines),
        "llm_rejected": len(gen.rejected),
        "generation_s": round(t_gen, 3), "total_s": round(time.time() - t0, 3),
    }
    with open(os.path.join(out_dir, "track.json"), "w") as f:
        f.write(track.model_dump_json(indent=1))
    with open(os.path.join(out_dir, "captions.vtt"), "w") as f:
        f.write(to_vtt(lines))
    with open(os.path.join(out_dir, "events.json"), "w") as f:
        json.dump([e.model_dump() for e in events], f, indent=1)
    return track


def run_soccernet(game: str, half: int, start: float, end: float, root: str = "data/SoccerNet",
                  ckpt: str = SPOTTER, out_root: str = "outputs/clips", use_gt: bool = False,
                  **kw) -> CommentaryTrack:
    gdir = os.path.join(root, game)
    with open(os.path.join(gdir, "Labels-v2.json")) as f:
        lab = json.load(f)
    match = MatchInfo(home=lab.get("gameHomeTeam", "home"), away=lab.get("gameAwayTeam", "away"))
    feats = np.load(os.path.join(gdir, f"{half}_ResNET_TF2_PCA512.npy"))
    end = min(end, len(feats) / FPS)
    if use_gt:
        events = ground_truth_events(os.path.join(gdir, "Labels-v2.json"), half, start, end)
    else:
        events = spot_features(feats, ckpt, half, start, end)
    clip_id = slug(f"{game.split('/')[-1]}-h{half}-{int(start)}-{int(end)}")
    return build_track(clip_id, events, end - start, os.path.join(out_root, clip_id), match, **kw)


def run_video(video: str, events_path: str | None = None, ckpt: str = SPOTTER,
              out_root: str = "outputs/clips", context: bool = True, match: MatchInfo | None = None,
              **kw) -> CommentaryTrack:
    import cv2

    cap = cv2.VideoCapture(video)
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 25)
    cap.release()
    if events_path:
        with open(events_path) as f:
            events = [Event(**e) for e in json.load(f)]
    else:
        from .spotting.features import extract, has_pca

        if has_pca():
            events = spot_features(extract(video), ckpt)
        else:  # torchvision features are 2048-d; the spotter needs SoccerNet's PCA-512
            log.warning("spotting skipped: models/pca_512_TF2.pkl missing (supply an events JSON)")
            events = []
    if context and events:
        try:
            from .context.tagging import ContextExtractor

            events = ContextExtractor().enrich(video, events, match)
        except Exception as e:
            log.warning("context layer skipped: %s", e)
    clip_id = slug(os.path.splitext(os.path.basename(video))[0])
    return build_track(clip_id, events, duration, os.path.join(out_root, clip_id), match, video=video, **kw)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("soccernet")
    s.add_argument("game")
    s.add_argument("--half", type=int, default=1)
    s.add_argument("--start", type=float, default=0)
    s.add_argument("--end", type=float, default=600)
    s.add_argument("--gt", action="store_true", help="use ground-truth labels instead of the spotter")
    v = sub.add_parser("video")
    v.add_argument("video")
    v.add_argument("--events")
    v.add_argument("--no-context", action="store_true")
    for p in (s, v):
        p.add_argument("--no-voice", action="store_true")
        p.add_argument("--llm", choices=["auto", "on", "off"], default="auto")
        p.add_argument("--lang", default="en")
    a = ap.parse_args()
    use_llm = {"auto": None, "on": True, "off": False}[a.llm]
    if a.cmd == "soccernet":
        tr = run_soccernet(a.game, a.half, a.start, a.end, use_gt=a.gt, voice=not a.no_voice,
                           use_llm=use_llm, language=a.lang)
    else:
        tr = run_video(a.video, a.events, context=not a.no_context, voice=not a.no_voice,
                       use_llm=use_llm, language=a.lang)
    for ln in tr.lines:
        print(f"[{ln.t:7.1f}s] ({ln.source}) {ln.text}")
    print(json.dumps(tr.stats))


if __name__ == "__main__":
    main()
