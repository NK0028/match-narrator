"""Team assignment by k-means on jersey colours.

For each player box we take the torso region, drop grass-green pixels, and
summarise the rest as a mean Lab colour. k-means (k=2, plus an outlier test for
referees/keepers) splits players into two teams; each cluster gets a readable
colour name ("white", "red", ...) for the commentary.
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Named reference colours (RGB) for describing clusters.
NAMED = {
    "white": (235, 235, 235), "black": (25, 25, 25), "grey": (128, 128, 128),
    "red": (200, 30, 40), "dark red": (120, 20, 30), "blue": (30, 60, 200),
    "sky blue": (110, 170, 230), "navy": (20, 30, 80), "yellow": (240, 210, 40),
    "orange": (240, 130, 30), "green": (30, 140, 60), "purple": (110, 40, 140),
    "pink": (230, 120, 170), "claret": (110, 30, 60),
}


def colour_name(rgb: np.ndarray) -> str:
    lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    best, bd = "unknown", 1e9
    for name, ref in NAMED.items():
        rl = cv2.cvtColor(np.uint8([[ref]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
        d = np.linalg.norm(lab - rl)
        if d < bd:
            best, bd = name, d
    return best


def jersey_colour(frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """Mean RGB of the non-grass torso pixels of a player box (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    if h < 8 or w < 4:
        return None
    crop = frame_bgr[max(0, y1 + int(0.15 * h)): y1 + int(0.55 * h), max(0, x1 + int(0.2 * w)): x2 - int(0.2 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60)
    px = crop[~grass]
    if len(px) < 10:
        return None
    return px[:, ::-1].mean(0)  # BGR -> RGB


class TeamClassifier:
    def __init__(self):
        self.km: KMeans | None = None
        self.names: list[str] = []
        self.centers_lab: np.ndarray | None = None
        self.spread: float = 0.0

    @staticmethod
    def _lab(rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(np.uint8(np.clip(rgb, 0, 255)).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)

    def fit(self, colours: list[np.ndarray]) -> "TeamClassifier":
        X = self._lab(np.array(colours))
        self.km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        self.centers_lab = self.km.cluster_centers_
        d = np.linalg.norm(X - self.centers_lab[self.km.labels_], axis=1)
        self.spread = float(np.percentile(d, 90)) if len(d) else 0.0
        rgb = cv2.cvtColor(np.uint8(self.centers_lab).reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
        self.names = [colour_name(c) for c in rgb]
        if self.names[0] == self.names[1]:
            self.names = [f"{self.names[0]} (A)", f"{self.names[1]} (B)"]
        return self

    def predict(self, colour: np.ndarray) -> tuple[int, str, bool]:
        """-> (cluster, colour name, is_outlier e.g. referee/keeper)."""
        x = self._lab(np.array([colour]))[0]
        d = np.linalg.norm(self.centers_lab - x, axis=1)
        k = int(np.argmin(d))
        return k, self.names[k], bool(d[k] > 2.0 * max(self.spread, 1.0))
