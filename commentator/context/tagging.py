"""Attach team colour + player tag to each spotted event (context extraction).

Tier 1 (always): per-shot generic tags player_1..N from ByteTrack IDs.
Tier 2 (optional): jersey-number OCR on the event frame -> "#7", then roster -> name.
"""
from __future__ import annotations

import logging
import re

import numpy as np

from ..schema import Event, MatchInfo
from .detect import ClipTracker, FrameDets
from .teams import TeamClassifier, jersey_colour

log = logging.getLogger(__name__)

# Events where "the player nearest the ball" is meaningful.
BALL_EVENTS = {"goal", "shot", "corner", "free_kick", "penalty", "kickoff", "clearance", "throw_in"}


class JerseyOCR:
    """Thin wrapper over EasyOCR (optional dependency: pip install easyocr)."""

    def __init__(self):
        import easyocr  # noqa: F401  (raise ImportError early if missing)

        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, frame: np.ndarray, box) -> str | None:
        x1, y1, x2, y2 = [int(v) for v in box]
        if y2 - y1 < 80:  # too small to read a number reliably
            return None
        crop = frame[y1 + (y2 - y1) // 8: y1 + (y2 - y1) // 2, x1:x2]
        for _, text, conf in self.reader.readtext(crop, allowlist="0123456789"):
            if conf > 0.5 and re.fullmatch(r"\d{1,2}", text):
                return text
        return None


def _nearest(players, point):
    if not players:
        return None
    d = [np.hypot(p.center[0] - point[0], p.feet[1] - point[1]) for p in players]
    return players[int(np.argmin(d))]


class ContextExtractor:
    def __init__(self, tracker: ClipTracker | None = None, use_ocr: bool = False,
                 before_s: float = 1.5, after_s: float = 0.5):
        self.tracker = tracker or ClipTracker()
        self.before, self.after = before_s, after_s
        self.ocr = None
        if use_ocr:
            try:
                self.ocr = JerseyOCR()
            except Exception as e:  # pragma: no cover - optional
                log.warning("jersey OCR disabled: %s", e)

    def enrich(self, video: str, events: list[Event], match: MatchInfo | None = None) -> list[Event]:
        match = match or MatchInfo()
        windows: list[list[FrameDets]] = [
            self.tracker.track_window(video, max(0, e.t - self.before), e.t + self.after) for e in events
        ]
        colours = [c for w in windows for fd in w for p in fd.players
                   if (c := jersey_colour(fd.frame, p.box)) is not None]
        teams = TeamClassifier().fit(colours) if len(colours) >= 6 else None
        if teams:
            match.home_colour = match.home_colour or teams.names[0]
            match.away_colour = match.away_colour or teams.names[1]

        tags: dict[tuple, str] = {}
        out = []
        for ei, (ev, win) in enumerate(zip(events, windows)):
            ev = ev.model_copy(deep=True)
            if not win:
                out.append(ev)
                continue
            fd = min(win, key=lambda f: abs(f.t - ev.t))
            ev.meta["n_players"] = len(fd.players)
            subject = None
            if ev.event in BALL_EVENTS and fd.ball is not None:
                subject = _nearest(fd.players, fd.ball.center)
                ev.meta["ball_xy"] = [round(v, 1) for v in fd.ball.center]
            elif fd.players:  # fall back to the most prominent (largest) player in view
                subject = max(fd.players, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
            if subject is not None:
                if subject.track_id is not None:
                    key = (ei, fd.shot_id, subject.track_id)
                    tags.setdefault(key, f"player_{len(tags) + 1}")
                    tag = tags[key]
                else:
                    tag = None
                team_name = None
                if teams and (c := jersey_colour(fd.frame, subject.box)) is not None:
                    _, team_name, outlier = teams.predict(c)
                    if outlier:
                        team_name = None  # referee / keeper — don't guess
                number = self.ocr.read(fd.frame, subject.box) if self.ocr else None
                if number:
                    name = self._roster_name(match, team_name, number)
                    ev.player = f"{name} (#{number})" if name else f"#{number} ({tag})" if tag else f"#{number}"
                elif tag:
                    ev.player = tag
                # keep the spotter's home/away when we can map it; otherwise use colour
                if team_name:
                    ev.meta["team_colour"] = team_name
                    if not ev.team or ev.team in ("home", "away"):
                        ev.team = team_name
            out.append(ev)
        return out

    @staticmethod
    def _roster_name(match: MatchInfo, team_colour: str | None, number: str) -> str | None:
        for team, roster in match.roster.items():
            colour = match.home_colour if team == "home" else match.away_colour if team == "away" else team
            if team_colour is None or colour == team_colour or team == team_colour:
                if number in roster:
                    return roster[number]
        return None
