"""Crowd excitement from the match's own audio track.

Loudness (RMS, dB) per 0.5 s window, z-scored against a rolling 60 s median/MAD. A crowd
roar is a strong, cheap signal for goals and big chances, and it lets us demand
confirmation before narrating a goal that the (raw-video) spotter thinks it saw.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import wave

import numpy as np

HOP_S = 0.5
SR = 16000


def load_mono(video: str, sr: int = SR) -> np.ndarray | None:
    from ..tts.engine import ffmpeg_exe

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "1",
                            "-ar", str(sr), "-f", "wav", wav], capture_output=True)
        if r.returncode != 0 or not os.path.exists(wav):
            return None  # no audio stream
        with wave.open(wav) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return x if len(x) else None


class Excitement:
    def __init__(self, z: np.ndarray, hop: float = HOP_S):
        self.z, self.hop = z, hop

    @classmethod
    def from_samples(cls, x: np.ndarray, sr: int = SR, hop: float = HOP_S, window_s: float = 60.0):
        n = int(sr * hop)
        frames = len(x) // n
        if frames == 0:
            return cls(np.zeros(0), hop)
        rms = np.sqrt((x[: frames * n].reshape(frames, n) ** 2).mean(1) + 1e-10)
        db = 20 * np.log10(rms)
        w = max(3, int(window_s / hop))
        z = np.zeros(frames)
        for i in range(frames):
            seg = db[max(0, i - w // 2): i + w // 2 + 1]
            med = np.median(seg)
            mad = np.median(np.abs(seg - med)) * 1.4826 + 0.5  # +0.5 dB floor: quiet clips stay calm
            z[i] = (db[i] - med) / mad
        return cls(z, hop)

    @classmethod
    def from_video(cls, video: str) -> "Excitement":
        x = load_mono(video)
        return cls.from_samples(x) if x is not None else cls(np.zeros(0))

    def at(self, t: float) -> float:
        i = int(t / self.hop)
        return float(self.z[i]) if 0 <= i < len(self.z) else 0.0

    def max_between(self, t0: float, t1: float) -> float:
        a, b = max(0, int(t0 / self.hop)), min(len(self.z), int(t1 / self.hop) + 1)
        return float(self.z[a:b].max()) if b > a else 0.0

    def peaks(self, z_thresh: float = 2.5, min_gap_s: float = 10.0) -> list[tuple[float, float]]:
        """(time, z) of distinct roars, strongest first within each min_gap window."""
        out: list[tuple[float, float]] = []
        for i in np.argsort(-self.z):
            if self.z[i] < z_thresh:
                break
            t = i * self.hop
            if all(abs(t - o) >= min_gap_s for o, _ in out):
                out.append((t, float(self.z[i])))
        return sorted(out)
