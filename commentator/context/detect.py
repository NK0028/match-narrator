"""Player/ball detection + per-shot tracking (YOLO + ByteTrack) on short clips.

Only run on a few seconds around each spotted event — never a full match.
Track IDs reset at camera cuts (detected with a histogram jump), which is why tags
are "per-shot": player_3 is only guaranteed to be the same person within one shot.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

PERSON, BALL = 0, 32  # COCO ids (sports ball = 32)


@dataclass
class Detection:
    track_id: int | None
    cls: int
    box: tuple[float, float, float, float]
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def feet(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2, y2


@dataclass
class FrameDets:
    t: float
    frame: np.ndarray | None
    players: list[Detection] = field(default_factory=list)
    ball: Detection | None = None
    shot_id: int = 0


def is_cut(prev: np.ndarray | None, cur: np.ndarray, thresh: float = 0.5) -> bool:
    if prev is None:
        return False
    hs = []
    for img in (prev, cur):
        hsv = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hs.append(cv2.normalize(h, h).flatten())
    return cv2.compareHist(hs[0], hs[1], cv2.HISTCMP_BHATTACHARYYA) > thresh


def _default_weights() -> str:
    here = os.path.join(os.path.dirname(__file__), "..", "..", "models", "yolo11n.pt")
    return os.path.abspath(here) if os.path.exists(here) else "yolo11n.pt"


class ClipTracker:
    def __init__(self, weights: str | None = None, conf: float = 0.25, device: str | None = None,
                 imgsz: int | None = None):
        from ultralytics import YOLO

        self.weights = weights or os.environ.get("YOLO_WEIGHTS") or _default_weights()
        self.model = YOLO(self.weights)
        self.conf = conf
        self.device = device
        # the ball is tiny in wide broadcast shots; a larger input size helps ball recall
        self.imgsz = imgsz or int(os.environ.get("YOLO_IMGSZ", "640"))
        self.ball_conf = 0.2

    @staticmethod
    def _new_tracker():
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        return BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))

    def track_window(self, video: str, t0: float, t1: float, sample_fps: float = 5.0,
                     keep_frames: bool = True) -> list[FrameDets]:
        return list(self.iter_frames(video, t0, t1, sample_fps, keep_frames))

    def iter_frames(self, video: str, t0: float = 0.0, t1: float | None = None, sample_fps: float = 5.0,
                    keep_frames: bool = True):
        """Yield FrameDets one sampled frame at a time (streaming; used by live mode)."""
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start = max(0, int(t0 * fps))
        end_f = int(t1 * fps) if t1 is not None else (n - 1 if n else 10**9)
        stop = min(n - 1, end_f) if n else end_f
        step = max(1, int(round(fps / sample_fps)))
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        prev, shot, persist, tracker = None, 0, False, None
        try:
            for fi in range(start, stop + 1):
                ok = cap.grab()
                if not ok:
                    break
                if (fi - start) % step:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if is_cut(prev, frame):
                    shot += 1
                    persist = False  # new shot -> new IDs (reset tracker)
                prev = frame
                # Detect once, then track players only: ByteTrack drops the ball (tiny, fast,
                # rarely IoU-matched between samples), so the ball comes from raw detections.
                res = self.model.predict(frame, conf=0.1, classes=[PERSON, BALL], verbose=False,
                                         device=self.device, imgsz=self.imgsz)[0]
                if not persist or tracker is None:
                    tracker = self._new_tracker()
                persist = True
                fd = FrameDets(t=fi / fps, frame=frame if keep_frames else None, shot_id=shot)
                boxes = res.boxes.cpu().numpy() if res.boxes is not None else None
                if boxes is not None and len(boxes):
                    cls = boxes.cls.astype(int)
                    people = boxes[cls == PERSON]
                    if len(people):
                        for row in tracker.update(people, frame):
                            x1, y1, x2, y2, tid, score = row[:6]
                            fd.players.append(Detection(int(tid), PERSON, (x1, y1, x2, y2), float(score)))
                    for i in np.where(cls == BALL)[0]:
                        cf = float(boxes.conf[i])
                        if cf >= self.ball_conf and (fd.ball is None or cf > fd.ball.conf):
                            fd.ball = Detection(None, BALL, tuple(boxes.xyxy[i].tolist()), cf)
                yield fd
        finally:
            cap.release()
