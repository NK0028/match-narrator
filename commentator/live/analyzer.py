"""Per-frame play state from a broadcast video: who has the ball, which team, what view.

Runs YOLO + ByteTrack on every sampled frame (5 fps by default), names the two kits by
k-means on jersey colours, and picks the ball holder as the player whose feet are nearest
the ball. No pitch homography: positions are screen-relative, motion comes from camera pan.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np

from ..context.detect import FrameDets
from ..context.teams import TeamClassifier, jersey_colour

log = logging.getLogger(__name__)


@dataclass
class PlayState:
    t: float
    shot_id: int
    view: str  # "wide" | "close_up" | "empty"
    n_players: int
    ball_xy: tuple[float, float] | None = None  # normalised 0..1 screen coords
    holder: int | None = None  # ByteTrack id of the player with the ball
    holder_team: str | None = None  # kit colour name
    cam_dx: float = 0.0  # camera pan since last sample, fraction of frame width (+ = right)
    team_counts: dict[str, int] = field(default_factory=dict)


def classify_view(fd: FrameDets, frame_h: int) -> str:
    if not fd.players:
        return "empty"
    hs = sorted((p.box[3] - p.box[1]) / frame_h for p in fd.players)
    med = hs[len(hs) // 2]
    if len(fd.players) >= 4 and med < 0.22:
        return "wide"
    return "close_up"


def ball_holder(fd: FrameDets, max_dist_heights: float = 0.9):
    """Player whose feet are nearest the ball, within ~one body height; else None."""
    if fd.ball is None or not fd.players:
        return None
    bx, by = fd.ball.center
    best, bd = None, 1e9
    for p in fd.players:
        fx, fy = p.feet
        h = max(1.0, p.box[3] - p.box[1])
        d = np.hypot(bx - fx, by - fy) / h
        if d < bd:
            best, bd = p, d
    return best if bd <= max_dist_heights else None


def camera_pan(prev_small: np.ndarray | None, cur_small: np.ndarray) -> float:
    if prev_small is None:
        return 0.0
    (dx, _), resp = cv2.phaseCorrelate(prev_small, cur_small)
    if resp < 0.05:
        return 0.0
    return float(-dx / cur_small.shape[1])  # content moving left = camera panning right


class PlayAnalyzer:
    def __init__(self, tracker=None, sample_fps: float = 5.0, fit_after: int = 60):
        if tracker is None:
            from ..context.detect import ClipTracker

            tracker = ClipTracker()
        self.tracker = tracker
        self.sample_fps = sample_fps
        self.fit_after = fit_after  # jersey colours needed before naming kits
        self.teams: TeamClassifier | None = None
        self._colours: list[np.ndarray] = []
        self._recent_outlier: list[bool] = []

    @property
    def kit_names(self) -> list[str]:
        return list(self.teams.names) if self.teams else []

    def _team_of(self, frame, det) -> str | None:
        c = jersey_colour(frame, det.box)
        if c is None:
            return None
        if self.teams is None:
            self._colours.append(c)
            if len(self._colours) >= self.fit_after:
                self.teams = TeamClassifier().fit(self._colours)
                log.info("kits: %s", self.teams.names)
            return None
        _, name, outlier = self.teams.predict(c)
        self._recent_outlier = (self._recent_outlier + [outlier])[-200:]
        if len(self._recent_outlier) == 200 and np.mean(self._recent_outlier) > 0.4:
            log.info("kit colours drifted; refitting")  # lighting change, new camera, ...
            self.teams, self._colours, self._recent_outlier = None, [], []
        return None if outlier else name

    def run(self, video: str, t0: float = 0.0, t1: float | None = None) -> Iterator[PlayState]:
        prev_small, prev_shot = None, None
        for fd in self.tracker.iter_frames(video, t0, t1, sample_fps=self.sample_fps):
            frame = fd.frame
            fh, fw = frame.shape[:2]
            small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            dx = camera_pan(prev_small if fd.shot_id == prev_shot else None, small)
            prev_small, prev_shot = small, fd.shot_id
            view = classify_view(fd, fh)
            st = PlayState(t=fd.t, shot_id=fd.shot_id, view=view, n_players=len(fd.players), cam_dx=dx)
            if view == "wide":
                counts: dict[str, int] = {}
                for p in fd.players:
                    if (name := self._team_of(frame, p)) is not None:
                        counts[name] = counts.get(name, 0) + 1
                st.team_counts = counts
            if fd.ball is not None:
                bx, by = fd.ball.center
                st.ball_xy = (round(bx / fw, 3), round(by / fh, 3))
            h = ball_holder(fd) if view == "wide" else None
            if h is not None:
                st.holder = h.track_id
                st.holder_team = self._team_of(frame, h) if self.teams else None
            yield st
