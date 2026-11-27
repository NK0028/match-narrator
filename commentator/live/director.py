"""Live commentary director: decides what to say, and when, from a stream of moments.

Rules (in priority order) each time media time advances:
1. a big event (goal, card, crowd roar, ...) is said at once and pre-empts the queue;
2. otherwise, when the commentator is free, the freshest moment younger than `stale_s`
   (stale moments are dropped — live commentary never narrates the past);
3. otherwise, after `fill_s` of silence, a filler line from the running match stats.

Text comes from Claude when available (asynchronously, with a wall-clock deadline) and
from templates otherwise; every Claude line must pass the grounding guard and fit the
slot reserved for it, or the template is used instead.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from ..generation.guard import is_grounded
from ..generation.templates import render, render_live
from ..schema import IMPORTANCE, LIVE_TYPES, CommentaryLine, Event

log = logging.getLogger(__name__)

WORDS_PER_S = 2.8
LANG_NAMES = {"en": "English", "ur": "Urdu"}
BIG = 6  # importance at/above which a moment pre-empts


def speech_s(text: str) -> float:
    return max(1.4, len(text.split()) / WORDS_PER_S + 0.3)


class _Pending:
    def __init__(self, line: CommentaryLine, ev: Event, future: Future | None, slot_s: float):
        self.line, self.ev, self.future, self.slot_s = line, ev, future, slot_s
        self.submitted = time.monotonic()


class Director:
    def __init__(self, language: str = "en", llm=None, deadline_s: float = 3.0, stale_s: float = 3.0,
                 fill_s: float = 5.0, gap_s: float = 0.4, seed: int = 0,
                 stats_fn: Callable[[float], Event] | None = None):
        self.language, self.llm, self.deadline_s = language, llm, deadline_s
        self.stale_s, self.fill_s, self.gap_s, self.seed = stale_s, fill_s, gap_s, seed
        self.stats_fn = stats_fn
        self.queue: list[Event] = []
        self.events: list[Event] = []  # everything narrated (grounding links)
        self.pending: list[_Pending] = []
        self.free_at = 0.0  # media time the commentator stops talking
        self.last_end = 0.0
        self.n = 0
        self.recent: list[str] = []
        self.rejected: list[dict] = []
        self.llm_late = 0
        self._pool = ThreadPoolExecutor(max_workers=4) if llm else None
        self._last_filler = -1e9

    # -------------------------------------------------------------------------------
    def push(self, events: list[Event]) -> None:
        self.queue.extend(events)
        self.queue.sort(key=lambda e: e.t)

    def _template(self, ev: Event) -> str:
        avoid = set(self.recent[-6:])
        if ev.event in LIVE_TYPES:
            return render_live(ev, self.n, self.language, self.seed, avoid)
        return render(ev, self.n, self.language, self.seed)

    def _say(self, ev: Event, start: float, interrupt: bool = False) -> None:
        text = self._template(ev)
        slot = max(speech_s(text), 3.2 if ev.event != "pass" else speech_s(text))
        idx = len(self.events)
        self.events.append(ev)
        line = CommentaryLine(t=round(start, 2), end=round(start + speech_s(text), 2), text=text,
                              event_index=idx, event=ev.event, source="template",
                              excitement=min(1.0, IMPORTANCE[ev.event] / 10))
        fut = None
        worth_llm = ev.event != "pass" or (ev.meta.get("pass_chain") or 0) >= 4  # routine passes: template
        if self._pool is not None and worth_llm:
            fut = self._pool.submit(self.llm.live_line, ev.facts(), list(self.recent),
                                    LANG_NAMES.get(self.language, "English"))
        self.pending.append(_Pending(line, ev, fut, slot))
        if interrupt:
            line.excitement = max(line.excitement, 0.9)
        self.n += 1
        self.recent.append(text)
        self.free_at = start + slot + self.gap_s
        self.last_end = start + slot

    def advance(self, now: float) -> None:
        """Media time `now` has been analysed; schedule whatever should be said by then."""
        while True:
            ready = [e for e in self.queue if e.t <= now]
            if not ready:
                break
            big = [e for e in ready if IMPORTANCE[e.event] >= BIG]
            if big:
                ev = max(big, key=lambda e: (IMPORTANCE[e.event], e.t))
                self.queue = [e for e in self.queue if e.t > ev.t]  # everything before it is moot
                start = ev.t + 0.3
                self._say(ev, start, interrupt=self.free_at > start)
                continue
            start = max(self.free_at, ready[0].t + 0.3)
            if start > now:
                break  # still talking; keep the queue for later (staleness decided then)
            fresh = [e for e in ready if start - e.t <= self.stale_s]
            self.queue = [e for e in self.queue if e.t > now]
            if not fresh:
                break
            ev = max(fresh, key=lambda e: (IMPORTANCE[e.event], e.t))
            self._say(ev, max(start, ev.t + 0.3))
        # filler after a silence
        if (self.stats_fn and now - self.free_at >= self.fill_s and now - self._last_filler >= 20.0):
            ev = self.stats_fn(self.free_at + 0.5)
            self._last_filler = now
            self._say(ev, max(self.free_at, now - 0.5))

    def collect(self, force: bool = False) -> list[CommentaryLine]:
        """Finalised lines in order: Claude's text if it arrived in time and is grounded."""
        out: list[CommentaryLine] = []
        while self.pending:
            p = self.pending[0]
            if p.future is not None and not p.future.done():
                if not force and time.monotonic() - p.submitted < self.deadline_s:
                    break
                p.future.cancel()
                self.llm_late += 1
            elif p.future is not None:
                cand = None
                try:
                    cand = p.future.result()
                except Exception as e:  # pragma: no cover - network
                    log.warning("live LLM failed: %s", e)
                if cand:
                    if not is_grounded(cand, p.ev.facts()):
                        self.rejected.append({"event": p.ev.facts(), "text": cand})
                    elif speech_s(cand) > p.slot_s + 0.8:
                        self.rejected.append({"event": p.ev.facts(), "text": cand, "why": "too long"})
                    else:
                        p.line.text, p.line.source = cand, "llm"
                        p.line.end = round(p.line.t + speech_s(cand), 2)
                        self.recent.append(cand)
            self.pending.pop(0)
            out.append(p.line)
        return out

    def close(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
