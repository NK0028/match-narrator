"""Text-to-speech + a single commentary audio track anchored to event times.

Backends (auto-picked in this order): macOS `say`, `espeak-ng`/`espeak` (Linux /
Docker), pyttsx3, Coqui TTS (if installed and TTS_BACKEND=coqui). Excitement in
[0, 1] raises the speaking rate (and pitch where supported) for big moments.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

from ..schema import CommentaryLine

log = logging.getLogger(__name__)
SR = 22050


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _to_wav(src: str, dst: str) -> None:
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(SR), dst],
                   check=True)


def pick_backend() -> str:
    want = os.environ.get("TTS_BACKEND", "auto")
    if want != "auto":
        return want
    if shutil.which("say"):
        return "say"
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return "espeak"
    try:
        import pyttsx3  # noqa: F401

        return "pyttsx3"
    except Exception:
        return "none"


class TTS:
    def __init__(self, backend: str | None = None, voice: str | None = None, language: str = "en"):
        self.backend = backend or pick_backend()
        self.voice = voice or os.environ.get("TTS_VOICE")
        self.language = language
        self._coqui = None
        log.info("TTS backend: %s", self.backend)

    def synth(self, text: str, out_wav: str, excitement: float = 0.5) -> bool:
        if self.backend == "none":
            return False
        with tempfile.TemporaryDirectory() as td:
            try:
                if self.backend == "say":
                    raw = os.path.join(td, "x.aiff")
                    rate = int(175 + 70 * excitement)
                    cmd = ["say", "-r", str(rate), "-o", raw]
                    if self.voice:
                        cmd += ["-v", self.voice]
                    subprocess.run(cmd + [text], check=True)
                elif self.backend == "espeak":
                    raw = os.path.join(td, "x.wav")
                    exe = shutil.which("espeak-ng") or shutil.which("espeak")
                    voice = self.voice or ("ur" if self.language == "ur" else "en-gb")
                    subprocess.run([exe, "-v", voice, "-s", str(int(160 + 60 * excitement)),
                                    "-p", str(int(45 + 30 * excitement)), "-w", raw, text], check=True)
                elif self.backend == "pyttsx3":
                    import pyttsx3

                    raw = os.path.join(td, "x.wav")
                    eng = pyttsx3.init()
                    eng.setProperty("rate", int(170 + 60 * excitement))
                    eng.save_to_file(text, raw)
                    eng.runAndWait()
                elif self.backend == "coqui":
                    raw = os.path.join(td, "x.wav")
                    if self._coqui is None:
                        from TTS.api import TTS as Coqui

                        self._coqui = Coqui(os.environ.get("COQUI_MODEL", "tts_models/en/ljspeech/vits"))
                    self._coqui.tts_to_file(text=text, file_path=raw, speed=1.0 + 0.3 * excitement)
                else:
                    raise ValueError(f"unknown TTS backend {self.backend}")
                _to_wav(raw, out_wav)
                return True
            except Exception as e:
                log.warning("TTS failed (%s): %s", self.backend, e)
                return False


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return a


def write_wav(path: str, audio: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def voice_lines(lines: list[CommentaryLine], out_dir: str, tts: TTS) -> list[CommentaryLine]:
    """Voice each line to out_dir/line_XX.wav; set line.audio and true end time."""
    os.makedirs(out_dir, exist_ok=True)
    prev_end = 0.0
    for i, ln in enumerate(lines):
        p = os.path.join(out_dir, f"line_{i:03d}.wav")
        if tts.synth(ln.text, p, ln.excitement):
            ln.audio = os.path.basename(p)
            dur = len(read_wav(p)) / SR
            ln.t = round(max(ln.t, prev_end + 0.15), 2)  # never overlap speech
            ln.end = round(ln.t + dur, 2)
        prev_end = ln.end
    return lines


def mix_track(lines: list[CommentaryLine], audio_dir: str, duration: float, out_wav: str) -> str:
    """One WAV the length of the clip with each voiced line placed at its start time."""
    track = np.zeros(int((duration + 5) * SR), dtype=np.float32)
    for ln in lines:
        if not ln.audio:
            continue
        a = read_wav(os.path.join(audio_dir, ln.audio))
        s = int(ln.t * SR)
        e = min(len(track), s + len(a))
        if s < len(track):
            track[s:e] += a[: e - s] * 0.9
    write_wav(out_wav, track[: int(duration * SR) + 1])
    return out_wav


def encode_m4a(wav: str, out: str, bitrate: str = "64k") -> str:
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", wav, "-c:a", "aac", "-b:a", bitrate, out],
                   check=True)
    return out


def mux(video: str, audio_wav: str, out_mp4: str, keep_original: float = 0.35) -> str:
    """Video with commentary mixed over the (ducked) original audio, if any."""
    has_audio = subprocess.run([ffmpeg_exe(), "-i", video], capture_output=True, text=True).stderr.count("Audio:") > 0
    if has_audio:
        filt = f"[0:a]volume={keep_original}[o];[o][1:a]amix=inputs=2:duration=first:normalize=0[a]"
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", video, "-i", audio_wav, "-filter_complex", filt,
               "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", out_mp4]
    else:
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", video, "-i", audio_wav, "-map", "0:v", "-map", "1:a",
               "-c:v", "copy", "-c:a", "aac", "-shortest", out_mp4]
    subprocess.run(cmd, check=True)
    return out_mp4


def to_vtt(lines: list[CommentaryLine]) -> str:
    def ts(s: float) -> str:
        h, rem = divmod(max(0.0, s), 3600)
        m, sec = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"

    out = ["WEBVTT", ""]
    for i, ln in enumerate(lines):
        out += [str(i + 1), f"{ts(ln.t)} --> {ts(ln.end)}", ln.text, ""]
    return "\n".join(out)
