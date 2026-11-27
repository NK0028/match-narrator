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


class ClipTracker:
    def __init__(self, weights: str | None = None, conf: float = 0.25, device: str | None = None):
        from ultralytics import YOLO

        self.weights = weights or os.environ.get("YOLO_WEIGHTS", "yolo11n.pt")
        self.model = YOLO(self.weights)
        self.conf = conf
        self.device = device

    def track_window(self, video: str, t0: float, t1: float, sample_fps: float = 5.0,
                     keep_frames: bool = True) -> list[FrameDets]:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start = max(0, int(t0 * fps))
        stop = min(n - 1 if n else int(t1 * fps), int(t1 * fps))
        step = max(1, int(round(fps / sample_fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        out: list[FrameDets] = []
        prev, shot, persist = None, 0, False
        for fi in range(start, stop + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if (fi - start) % step:
                continue
            if is_cut(prev, frame):
                shot += 1
                persist = False  # new shot -> new IDs (reset tracker)
            prev = frame
            res = self.model.track(frame, persist=persist, tracker="bytetrack.yaml", conf=self.conf,
                                   classes=[PERSON, BALL], verbose=False, device=self.device)[0]
            persist = True
            fd = FrameDets(t=fi / fps, frame=frame if keep_frames else None, shot_id=shot)
            if res.boxes is not None and len(res.boxes):
                ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
                for box, c, cf, tid in zip(res.boxes.xyxy.tolist(), res.boxes.cls.int().tolist(),
                                           res.boxes.conf.tolist(), ids):
                    d = Detection(tid, c, tuple(box), cf)
                    if c == PERSON:
                        fd.players.append(d)
                    elif c == BALL and (fd.ball is None or cf > fd.ball.conf):
                        fd.ball = d
            out.append(fd)
        cap.release()
        return out
