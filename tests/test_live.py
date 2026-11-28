"""Live mode: analyzer -> moments -> director -> session/SSE, with a scripted fake tracker."""
from __future__ import annotations

import threading
import time

import numpy as np
from fastapi.testclient import TestClient

from commentator.context.detect import Detection, FrameDets
from commentator.generation.guard import violations
from commentator.live.analyzer import PlayAnalyzer
from commentator.live.audio import Excitement
from commentator.live.director import Director
from commentator.live.moments import MomentDetector
from commentator.schema import CommentaryLine, Event

RED, WHITE = (40, 30, 200), (235, 235, 235)  # BGR


class ScriptedTracker:
    """6 players (ids 1,3,5 red; 2,4,6 white) on grass; the ball sits at a scripted holder's feet."""

    def __init__(self, holders: list[tuple[float, float, int]], duration: float = 10.0, pan: float = 0.0):
        self.holders, self.duration, self.pan = holders, duration, pan

    def iter_frames(self, video, t0=0.0, t1=None, sample_fps=5.0, keep_frames=True):
        n = int(self.duration * sample_fps)
        for i in range(n):
            t = i / sample_fps
            frame = np.zeros((360, 640, 3), np.uint8)
            frame[:] = (40, 140, 40)
            players = []
            for pid in range(1, 7):
                x = 40 + (pid - 1) * 100
                frame[150:200, x:x + 22] = RED if pid % 2 else WHITE
                players.append(Detection(pid, 0, (x, 150, x + 22, 200), 0.9))
            holder = next((h for a, b, h in self.holders if a <= t < b), None)
            ball = None
            if holder is not None:
                bx = 40 + (holder - 1) * 100 + 11
                ball = Detection(None, 32, (bx - 3, 196, bx + 3, 202), 0.8)
            yield FrameDets(t=t, frame=frame, players=players, ball=ball, shot_id=0)


def run_moments(tracker, **kw):
    an = PlayAnalyzer(tracker=tracker, fit_after=12)
    det = MomentDetector(**kw)
    evs = []
    for st in an.run("x.mp4"):
        evs += det.feed(st)
    return an, det, evs


def test_analyzer_names_kits_and_finds_holder():
    an = PlayAnalyzer(tracker=ScriptedTracker([(0, 10, 3)]), fit_after=12)
    states = list(an.run("x.mp4"))
    assert set(an.kit_names) == {"red", "white"}
    assert all(s.view == "wide" for s in states)
    late = [s for s in states if s.t > 2]
    assert all(s.holder == 3 and s.holder_team == "red" for s in late)


def test_moments_pass_turnover_and_flicker_suppressed():
    # red #1 -> red #3 (pass) -> white #2 (turnover); a one-frame touch by red #5 is flicker
    holders = [(0, 3, 1), (3, 5, 3), (5, 6.0, 2), (6.0, 6.2, 5), (6.2, 9, 2)]
    _, det, evs = run_moments(ScriptedTracker(holders))
    kinds = [e.event for e in evs]
    assert "pass" in kinds and kinds.count("turnover") == 1, kinds
    p = next(e for e in evs if e.event == "pass")
    t = next(e for e in evs if e.event == "turnover")
    assert p.team == "red" and 3.0 <= p.t < 4.0
    assert t.team == "white" and t.meta["opponent"] == "red" and 5.0 <= t.t < 6.0
    assert det.stats.passes == {"red": 1} and det.stats.turnovers == 1
    pct = det.stats.possession_pct()
    assert pct["red"] > 0 and pct["white"] > 0 and sum(pct.values()) in (99, 100, 101)


def test_moments_attack_needs_sustained_pan():
    class Pan(ScriptedTracker):
        pass

    an = PlayAnalyzer(tracker=ScriptedTracker([(0, 20, 1)], duration=20), fit_after=12)
    det = MomentDetector()
    evs = []
    for st in an.run("x.mp4"):
        st.cam_dx = 0.03  # camera panning right steadily (~0.6 frame widths per 4 s)
        evs += det.feed(st)
    att = [e for e in evs if e.event == "attack"]
    assert att and att[0].meta["direction"] == "right" and att[0].team == "red"
    assert all(b.t - a.t >= 12 for a, b in zip(att, att[1:]))  # rate-limited


def moment_stream(seconds=120, every=1.0):
    evs = []
    t = 1.0
    while t < seconds:
        kind = ["pass", "pass", "carry", "possession", "turnover", "attack"][int(t) % 6]
        evs.append(Event(event=kind, t=t, team="red" if int(t) % 2 else "white",
                         meta={"direction": "left", "pass_chain": int(t) % 7, "opponent": "white"}))
        t += every
    return evs


def drive(director, events, until, step=0.2):
    lines, i = [], 0
    t = 0.0
    while t <= until:
        new = [e for e in events if t - step < e.t <= t]
        director.push(new)
        director.advance(t)
        lines += director.collect()
        t += step
    return lines + director.collect(force=True)


def test_director_paces_continuously_without_overlap():
    det = MomentDetector()
    d = Director(stats_fn=det.stats_event)
    lines = drive(d, moment_stream(120), 120)
    assert len(lines) >= 120 / 6  # at least a line every ~6 s
    for a, b in zip(lines, lines[1:]):
        assert b.t >= a.end - 1e-6, (a, b)  # never talk over ourselves
    gaps = [b.t - a.end for a, b in zip(lines, lines[1:])]
    assert max(gaps) < 8.0
    for ln in lines:  # never narrate the past
        ev = d.events[ln.event_index]
        assert ln.t - ev.t <= d.stale_s + 0.6
        assert not violations(ln.text, ev.facts())


def test_director_fills_silence_with_stats():
    det = MomentDetector()
    det.stats.possession_s = {"red": 60.0, "white": 40.0}
    det.kits = {"red", "white"}
    d = Director(stats_fn=det.stats_event)
    lines = drive(d, [Event(event="pass", t=1.0, team="red")], 40)
    assert any(ln.event == "stats" for ln in lines)
    st = next(ln for ln in lines if ln.event == "stats")
    assert not violations(st.text, d.events[st.event_index].facts())


def test_big_moment_preempts_queue():
    evs = moment_stream(30, every=0.5) + [Event(event="roar", t=20.0, meta={"z": 4.0})]
    d = Director()
    lines = drive(d, evs, 30)
    roar = [ln for ln in lines if ln.event == "roar"]
    assert roar and abs(roar[0].t - 20.3) < 0.05 and roar[0].excitement >= 0.7
    for a, b in zip(lines, lines[1:]):
        assert b.t >= a.end - 1e-6  # the interrupted line is cut short, not talked over


def test_filler_never_claims_unmeasured_stats():
    det = MomentDetector()
    det.kits = {"red", "white"}
    det.stats.possession_s = {"red": 5.0}  # 5 s of data is not "100 percent of the ball"
    ev = det.stats_event(10.0)
    assert ev.team is None and "possession_pct" not in ev.meta
    d = Director(stats_fn=det.filler_event)
    lines = drive(d, [], 30)
    assert lines and all("percent" not in ln.text for ln in lines)
    assert max(b.t - a.end for a, b in zip(lines, lines[1:])) < 7.0


class FakeLLM:
    def __init__(self, text, delay=0.0):
        self.text, self.delay, self.calls, self.latencies = text, delay, 0, []

    def live_line(self, facts, recent, language):
        time.sleep(self.delay)
        self.calls += 1
        return self.text(facts) if callable(self.text) else self.text


def test_llm_line_used_when_grounded_and_on_time():
    d = Director(llm=FakeLLM(lambda f: f"{f['team'].title()} turn it over and go."), deadline_s=2.0)
    lines = drive(d, [Event(event="turnover", t=1.0, team="red")], 3)
    assert lines[0].source == "llm" and "Red" in lines[0].text


def test_llm_hallucination_and_lateness_fall_back_to_template():
    d = Director(llm=FakeLLM("GOAL! What a strike!"), deadline_s=2.0)
    lines = drive(d, [Event(event="turnover", t=1.0, team="red")], 3)
    assert lines[0].source == "template" and d.rejected
    slow = Director(llm=FakeLLM("Red win it back.", delay=1.0), deadline_s=0.05)
    slow.push([Event(event="turnover", t=1.0, team="red")])
    slow.advance(1.5)
    time.sleep(0.1)
    out = slow.collect()
    assert out and out[0].source == "template" and slow.llm_late == 1
    slow.close()


def test_guard_on_live_types():
    assert violations("White win it back!", {"event": "pass", "team": "white"})
    assert violations("Play swings left.", {"event": "attack", "direction": "right"})
    assert not violations("Play swings right.", {"event": "attack", "direction": "right"})
    assert violations("Listen to that crowd!", {"event": "pass"})
    assert violations("Red have 70 percent of the ball.", {"event": "stats", "team": "red", "possession_pct": 61})


def test_excitement_finds_roar():
    sr = 16000
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.02, sr * 120).astype(np.float32)
    x[sr * 60: sr * 63] *= 12  # 3 s roar at 60 s
    exc = Excitement.from_samples(x, sr)
    peaks = exc.peaks()
    assert len(peaks) == 1 and 59.5 <= peaks[0][0] <= 63.0
    assert exc.max_between(55, 65) > 2.5 and exc.max_between(0, 50) < 2.5


class FakeSession:
    def __init__(self, sid, *a, **k):
        self.sid, self.cond = sid, threading.Condition()
        self.lines = [CommentaryLine(t=1.0, end=2.0, text="Red have it now.", event_index=0, event="possession")]
        self.status, self.ready_t, self.stats, self.duration = "done", 5.0, {"turnovers": 0}, 5.0
        self.kits, self.error, self.perf = ["red", "white"], None, {"x_realtime": 9.0}

    def start(self):
        return self

    def snapshot(self, since=0):
        return {"status": self.status, "lines": [ln.model_dump() for ln in self.lines[since:]]}


def test_live_api_streams_and_replays(monkeypatch, tmp_path):
    import server.app as appmod

    monkeypatch.setattr(appmod, "CLIPS", str(tmp_path))
    monkeypatch.setattr(appmod, "_new_live_session", lambda sid, *a, **k: FakeSession(sid))
    appmod.LIVE.clear()
    c = TestClient(appmod.app)
    r = c.post("/api/live", files={"video": ("m.mp4", b"\x00" * 100, "video/mp4")})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    body = c.get(f"/api/live/{sid}/stream").text
    assert "event: line" in body and "Red have it now." in body and "event: done" in body
    assert "event: line" not in c.get(f"/api/live/{sid}/stream?since=1").text  # resume skips sent lines
    assert c.get(f"/api/live/{sid}").json()["status"] == "done"
    assert c.post("/api/live", files={"video": ("m.txt", b"x", "text/plain")}).status_code == 400
