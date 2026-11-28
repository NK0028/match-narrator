"""A live commentary session: analyse a video in a worker thread, publish lines as they're ready.

Readers (the SSE endpoint) wait on `cond` and read `lines`, `ready_t` and `stats`.
`ready_t` is the media time up to which commentary is final — the player must not run
past it, which is what gives the broadcast-style delay.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

from ..generation import llm as llm_mod
from ..schema import CommentaryTrack, Event, MatchInfo
from .analyzer import PlayAnalyzer
from .audio import Excitement
from .director import Director
from .moments import MomentDetector

log = logging.getLogger(__name__)

# Big events from the raw-video spotter must be confirmed by a crowd roar (z) within ±8 s.
ROAR_CONFIRM = {"goal": 2.0, "penalty": 1.5, "red_card": 1.0}


def video_duration(video: str) -> float:
    import cv2

    cap = cv2.VideoCapture(video)
    d = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 25)
    cap.release()
    return float(d)


class LiveSession:
    def __init__(self, sid: str, video: str, out_dir: str, match: MatchInfo | None = None,
                 language: str = "en", tracker=None, use_llm: bool | None = None,
                 excitement: Excitement | None = None, voice: bool = True):
        self.sid, self.video, self.out_dir = sid, video, out_dir
        self.match = match or MatchInfo(language=language)
        self.language, self.tracker, self.voice = language, tracker, voice
        self.use_llm = llm_mod.available() if use_llm is None else use_llm
        self._excitement = excitement
        self.status = "queued"
        self.error: str | None = None
        self.duration = 0.0
        self.processed_t = 0.0
        self.ready_t = 0.0
        self.lines: list = []
        self.events: list[Event] = []
        self.stats: dict = {}
        self.kits: list[str] = []
        self.perf: dict = {}
        self.cond = threading.Condition()
        self.thread: threading.Thread | None = None

    # ---- public --------------------------------------------------------------------
    def start(self) -> "LiveSession":
        self.thread = threading.Thread(target=self._run, name=f"live-{self.sid}", daemon=True)
        self.thread.start()
        return self

    def snapshot(self, since: int = 0) -> dict:
        with self.cond:
            return {"session_id": self.sid, "status": self.status, "error": self.error, "language": self.language,
                    "duration": self.duration, "processed_t": round(self.processed_t, 2),
                    "ready_t": round(self.ready_t, 2), "kits": self.kits, "stats": self.stats,
                    "n_lines": len(self.lines), "lines": [ln.model_dump() for ln in self.lines[since:]],
                    "perf": self.perf}

    # ---- worker --------------------------------------------------------------------
    def _publish(self, lines, processed_t: float, director: Director, detector: MomentDetector) -> None:
        with self.cond:
            self.lines.extend(lines)
            self.processed_t = processed_t
            # commentary is final up to the first line still waiting for Claude
            pend = [p.line.t for p in director.pending]
            self.ready_t = min([processed_t] + pend)
            self.stats = detector.stats.as_dict()
            self.kits = sorted(detector.kits)
            self.cond.notify_all()

    def _run(self) -> None:
        t_wall = time.time()
        try:
            self.status = "running"
            self.duration = video_duration(self.video)
            exc = self._excitement or Excitement.from_video(self.video)
            roars = exc.peaks()
            detector = MomentDetector(roars=roars)
            llm = None
            if self.use_llm:
                try:
                    llm = llm_mod.LiveColour()
                except Exception as e:
                    log.warning("live LLM disabled: %s", e)
            director = Director(self.language, llm=llm, stats_fn=detector.filler_event)
            spotter = _ChunkSpotter(self.video, exc) if _ChunkSpotter.enabled() else None
            analyzer = PlayAnalyzer(tracker=self.tracker)
            last_pub = 0.0
            counts = {"frames": 0, "wide": 0, "ball_wide": 0, "holder": 0}
            for st in analyzer.run(self.video):
                counts["frames"] += 1
                if st.view == "wide":
                    counts["wide"] += 1
                    counts["ball_wide"] += st.ball_xy is not None
                    counts["holder"] += st.holder is not None
                moments = detector.feed(st)
                if spotter:
                    moments += spotter.until(st.t)
                director.push(moments)
                director.advance(st.t)
                lines = director.collect()
                if lines or st.t - last_pub >= 0.5:
                    self._publish(lines, st.t, director, detector)
                    last_pub = st.t
            # let in-flight Claude lines land, then finalise everything
            deadline = time.monotonic() + director.deadline_s
            while director.pending and time.monotonic() < deadline:
                self._publish(director.collect(), self.duration, director, detector)
                time.sleep(0.1)
            self._publish(director.collect(force=True), self.duration, director, detector)
            director.close()
            self.events = director.events
            wall = time.time() - t_wall
            self.perf = {"wall_s": round(wall, 1), "x_realtime": round(self.duration / wall, 2) if wall else None,
                         "llm_calls": llm.calls if llm else 0,
                         "llm_latency_p50": _pct(llm.latencies, 50) if llm else None,
                         "llm_latency_p95": _pct(llm.latencies, 95) if llm else None,
                         "llm_late": director.llm_late, "llm_rejected": len(director.rejected),
                         "roars": len(roars), **counts,
                         "ball_rate_wide": round(counts["ball_wide"] / counts["wide"], 3) if counts["wide"] else None,
                         "holder_rate_wide": round(counts["holder"] / counts["wide"], 3) if counts["wide"] else None}
            with self.cond:
                self.status = "finalizing"
                self.cond.notify_all()
            self._write_track()
            with self.cond:
                self.status = "done"
                self.cond.notify_all()
        except Exception as e:
            log.exception("live session failed")
            with self.cond:
                self.status, self.error = "error", str(e)
                self.cond.notify_all()

    def _write_track(self) -> None:
        from ..tts.engine import to_vtt

        os.makedirs(self.out_dir, exist_ok=True)
        tr = CommentaryTrack(clip_id=self.sid, duration=self.duration, match=self.match, events=self.events,
                             lines=self.lines, video=os.path.basename(self.video),
                             stats={"title": f"Live: {os.path.basename(self.video)}", "live": True,
                                    "n_events": len(self.events), "n_lines": len(self.lines),
                                    "llm_lines": sum(ln.source == "llm" for ln in self.lines),
                                    "match_stats": self.stats, **self.perf})
        with open(os.path.join(self.out_dir, "captions.vtt"), "w") as f:
            f.write(to_vtt(self.lines))
        with open(os.path.join(self.out_dir, "events.json"), "w") as f:
            json.dump([e.model_dump() for e in self.events], f, indent=1)
        if self.voice and self.lines and self.duration <= float(os.environ.get("LIVE_TTS_MAX_S", "1800")):
            try:  # downloadable voiced track (the live player speaks in the browser instead)
                from ..tts.engine import TTS, encode_m4a, mix_track, voice_lines

                tts = TTS(language=self.language)
                if tts.backend != "none":
                    adir = os.path.join(self.out_dir, "audio")
                    voice_lines(self.lines, adir, tts)
                    wav = mix_track(self.lines, adir, self.duration, os.path.join(self.out_dir, "commentary.wav"))
                    tr.audio_track = os.path.basename(encode_m4a(wav, os.path.join(self.out_dir, "commentary.m4a")))
            except Exception as e:
                log.warning("server-side voice skipped: %s", e)
        with open(os.path.join(self.out_dir, "track.json"), "w") as f:
            f.write(tr.model_dump_json(indent=1))


def _pct(xs: list[float], q: int):
    import numpy as np

    return round(float(np.percentile(xs, q)), 2) if xs else None


class _ChunkSpotter:
    """Optional: the SoccerNet spotter on raw-video features, 30 s at a time.

    Off unless SoccerNet's PCA files are in models/ and LIVE_SPOTTER=1, because
    torchvision features differ from SoccerNet's. Big events need a crowd roar.
    """

    CHUNK_S = 30.0

    @staticmethod
    def enabled() -> bool:
        from ..spotting.features import has_pca

        return os.environ.get("LIVE_SPOTTER") == "1" and has_pca()

    def __init__(self, video: str, exc: Excitement):
        self.video, self.exc, self.done_t, self.buf = video, exc, 0.0, []

    def until(self, t: float) -> list[Event]:
        from ..pipeline import spot_features
        from ..spotting.features import extract

        while t >= self.done_t + self.CHUNK_S:
            t0, t1 = self.done_t, self.done_t + self.CHUNK_S
            for e in spot_features(extract(self.video, t0=t0, t1=t1)):
                e = e.model_copy(update={"t": e.t + t0, "timestamp": ""})
                need = ROAR_CONFIRM.get(e.event)
                if need is None or self.exc.max_between(e.t - 8, e.t + 8) >= need:
                    self.buf.append(Event(**e.model_dump()))
            self.done_t = t1
        out = [e for e in self.buf if e.t <= t]
        self.buf = [e for e in self.buf if e.t > t]
        return out
