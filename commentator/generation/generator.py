"""Events -> timed commentary track (template + LLM hybrid).

Design rule: templates carry routine events (speed, zero hallucination); the LLM
only adds variety on the big moments, and every LLM line must pass the grounding
guard or it's replaced by the template.
"""
from __future__ import annotations

import logging

from ..schema import DEFAULT_NARRATED, IMPORTANCE, CommentaryLine, Event, MatchInfo
from . import llm as llm_mod
from .guard import is_grounded
from .templates import render

log = logging.getLogger(__name__)

# Lines start slightly *after* the event (viewer sees it, then hears it); goals get
# a beat longer so the roar lands after the ball hits the net.
LAG_S = {"goal": 0.6, "penalty": 0.4, "red_card": 0.5}
DEFAULT_LAG_S = 0.3
WORDS_PER_S = 2.8
LANG_NAMES = {"en": "English", "ur": "Urdu"}

# Pairs where the second is implied by the first and would be redundant to narrate
# (e.g. a goal is always followed by a kick-off; a shot right before a goal).
REDUNDANT_AFTER = {("goal", "kickoff"): 90.0, ("foul", "free_kick"): 8.0, ("card", "free_kick"): 8.0}
REDUNDANT_BEFORE = {("shot", "goal"): 6.0, ("foul", "card"): 6.0, ("foul", "penalty"): 6.0}


def dedupe(events: list[Event], same_window_s: float = 6.0) -> list[Event]:
    """Merge duplicates of the same type close in time; keep the most confident."""
    events = sorted(events, key=lambda e: e.t)
    out: list[Event] = []
    for e in events:
        if out and out[-1].event == e.event and e.t - out[-1].t < same_window_s:
            if e.confidence > out[-1].confidence:
                out[-1] = e
            continue
        out.append(e)
    return out


ROUTINE = {"throw_in", "clearance", "ball_out"}


def select(events: list[Event], narrated: set[str] | None = None, min_gap_s: float = 2.5,
           fill_gap_s: float | None = 40.0) -> list[int]:
    """Indices of events worth a line: narratable, non-redundant, not crowding a bigger one.

    With `fill_gap_s`, a routine event (throw-in, clearance, ball out) is also narrated
    when nothing has been said for that long, so the track doesn't go silent.
    """
    narrated = narrated or DEFAULT_NARRATED
    keep: list[int] = []
    last_t = -1e9
    for i, e in enumerate(events):
        if e.event not in narrated:
            if not (fill_gap_s and e.event in ROUTINE and e.t - last_t >= fill_gap_s
                    and not any(o.event in narrated and 0 <= o.t - e.t < 10 for o in events)):
                continue
        redundant = False
        for j in range(len(events)):
            if j == i:
                continue
            o, dt = events[j], e.t - events[j].t
            if 0 < dt <= REDUNDANT_AFTER.get((o.event, e.event), -1):
                redundant = True  # e follows o and is implied by it
            if 0 < -dt <= REDUNDANT_BEFORE.get((e.event, o.event), -1):
                redundant = True  # e precedes o and o subsumes it
        if redundant:
            continue
        if keep and e.t - events[keep[-1]].t < min_gap_s:
            if IMPORTANCE[e.event] > IMPORTANCE[events[keep[-1]].event]:
                keep[-1] = i
                last_t = e.t
            continue
        keep.append(i)
        last_t = e.t
    return keep


class CommentaryGenerator:
    def __init__(self, use_llm: bool | None = None, llm_min_importance: int = 6,
                 language: str = "en", seed: int = 0):
        self.language = language
        self.seed = seed
        self.llm_min_importance = llm_min_importance
        self.llm = None
        if use_llm or (use_llm is None and llm_mod.available()):
            try:
                self.llm = llm_mod.ClaudeColour()
            except Exception as e:  # missing SDK / credentials
                log.warning("LLM disabled: %s", e)
        self.rejected: list[dict] = []

    def generate(self, events: list[Event], match: MatchInfo | None = None,
                 duration: float | None = None, narrated: set[str] | None = None) -> list[CommentaryLine]:
        match = match or MatchInfo(language=self.language)
        lines: list[CommentaryLine] = []
        history: list[dict] = []
        names = {n for r in match.roster.values() for n in r.values()} | {match.home, match.away}
        for i in select(events, narrated):
            ev = events[i]
            facts = ev.facts()
            text, source = render(ev, i, self.language, self.seed, match), "template"
            if self.llm and IMPORTANCE[ev.event] >= self.llm_min_importance:
                cand = self.llm.line(facts, history, LANG_NAMES.get(self.language, "English"))
                if cand and (self.language != "en" or is_grounded(cand, facts, names)):
                    text, source = cand, "llm"
                elif cand:
                    self.rejected.append({"event": facts, "text": cand})
            history.append(facts)
            start = ev.t + LAG_S.get(ev.event, DEFAULT_LAG_S)
            dur = max(1.5, len(text.split()) / WORDS_PER_S + 0.4)
            if lines and start < lines[-1].end:  # don't talk over the previous line
                start = lines[-1].end + 0.1
            end = start + dur
            if duration is not None and start >= duration:
                break
            lines.append(CommentaryLine(
                t=round(start, 2), end=round(end, 2), text=text, event_index=i, event=ev.event,
                source=source, excitement=min(1.0, IMPORTANCE[ev.event] / 10),
            ))
        return lines
