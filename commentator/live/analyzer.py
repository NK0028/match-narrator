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
    kits: tuple[str, ...] = ()  # current kit names (they can change after a refit)


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
    def __init__(self, tracker=None, sample_fps: float | None = None, fit_after: int = 60):
        if tracker is None:
            import os

            from ..context.detect import ClipTracker

            # the ball is ~10 px in a wide 1080p shot: at 640 YOLO finds it in ~5% of wide frames,
            # at 1280 in ~70% (measured on a broadcast highlights clip), so live mode defaults to 1280
            tracker = ClipTracker(imgsz=int(os.environ.get("YOLO_IMGSZ", "1280")))
        self.tracker = tracker
        import os

        # 4 fps keeps 1280-px detection ahead of real time on a laptop GPU; holder hysteresis still sees 3 frames
        self.sample_fps = sample_fps or float(os.environ.get("LIVE_SAMPLE_FPS", "4"))
        self.fit_after = fit_after  # jersey colours needed before naming kits
        self.teams: TeamClassifier | None = None
        self._colours: list[np.ndarray] = []
        self._recent_outlier: list[bool] = []
        self._ball: tuple[float, float, float] | None = None  # last accepted (t, x, y), normalised
        self._old_teams: TeamClassifier | None = None

    def _plausible_ball(self, t: float, xy: tuple[float, float], conf: float, shot_changed: bool) -> bool:
        """Reject one-off 'balls' (heads, boots, logos) that teleport across the frame."""
        prev = self._ball
        ok = (conf >= 0.5 or prev is None or shot_changed or t - prev[0] > 1.0
              or np.hypot(xy[0] - prev[1], xy[1] - prev[2]) <= 0.25 + 0.6 * (t - prev[0]))
        if ok:
            self._ball = (t, xy[0], xy[1])
        return ok

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
                old = self._old_teams
                self.teams = TeamClassifier().fit(self._colours)
                if old is not None:  # keep the old names for clusters that barely moved
                    for k, c in enumerate(self.teams.centers_lab):
                        d = np.linalg.norm(old.centers_lab - c, axis=1)
                        j = int(np.argmin(d))
                        if d[j] < 30 and old.names[j] not in self.teams.names:
                            self.teams.names[k] = old.names[j]
                log.info("kits: %s", self.teams.names)
            return None
        _, name, outlier = self.teams.predict(c)
        self._recent_outlier = (self._recent_outlier + [outlier])[-200:]
        if len(self._recent_outlier) == 200 and np.mean(self._recent_outlier) > 0.4:
            log.info("kit colours drifted; refitting")  # lighting change, new camera, ...
            self._old_teams = self.teams
            self.teams, self._colours, self._recent_outlier = None, [], []
        return None if outlier else name

    def run(self, video: str, t0: float = 0.0, t1: float | None = None) -> Iterator[PlayState]:
        prev_small, prev_shot = None, None
        for fd in self.tracker.iter_frames(video, t0, t1, sample_fps=self.sample_fps):
            frame = fd.frame
            fh, fw = frame.shape[:2]
            small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            dx = camera_pan(prev_small if fd.shot_id == prev_shot else None, small)
            prev_shot_before = prev_shot
            prev_small, prev_shot = small, fd.shot_id
            view = classify_view(fd, fh)
            st = PlayState(t=fd.t, shot_id=fd.shot_id, view=view, n_players=len(fd.players), cam_dx=dx,
                           kits=tuple(self.kit_names))
            if view == "wide":
                counts: dict[str, int] = {}
                for p in fd.players:
                    if (name := self._team_of(frame, p)) is not None:
                        counts[name] = counts.get(name, 0) + 1
                st.team_counts = counts
            if fd.ball is not None:
                bx, by = fd.ball.center
                xy = (round(bx / fw, 3), round(by / fh, 3))
                if self._plausible_ball(fd.t, xy, fd.ball.conf, fd.shot_id != prev_shot_before):
                    st.ball_xy = xy
                else:
                    fd.ball = None
            st.kits = tuple(self.kit_names)  # may have just been fitted
            h = ball_holder(fd) if view == "wide" else None
            if h is not None:
                st.holder = h.track_id
                st.holder_team = self._team_of(frame, h) if self.teams else None
            yield st
