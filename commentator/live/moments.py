"""PlayState stream -> play-by-play moments (as `Event`s) + running match stats.

A small state machine with hysteresis: a ball holder must persist for `hold_s` before
it counts, so one-frame detector flicker never becomes a "pass". Moments carry only
facts the tracker actually measured (team kit colour, direction of play, pass chain,
possession share) — the commentary layer may not say anything else.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..schema import Event
from .analyzer import PlayState


@dataclass
class MatchStats:
    possession_s: dict[str, float] = field(default_factory=dict)
    passes: dict[str, int] = field(default_factory=dict)
    turnovers: int = 0
    roars: int = 0
    longest_chain: int = 0

    def possession_pct(self) -> dict[str, int]:
        tot = sum(self.possession_s.values())
        return {k: round(100 * v / tot) for k, v in self.possession_s.items()} if tot > 0 else {}

    def as_dict(self) -> dict:
        return {"possession_pct": self.possession_pct(), "passes": dict(self.passes),
                "turnovers": self.turnovers, "roars": self.roars, "longest_chain": self.longest_chain}


class MomentDetector:
    def __init__(self, hold_s: float = 0.6, lose_s: float = 1.5, carry_s: float = 2.5,
                 attack_window_s: float = 4.0, attack_pan: float = 0.35, attack_gap_s: float = 12.0,
                 roars: list[tuple[float, float]] | None = None):
        self.hold_s, self.lose_s, self.carry_s = hold_s, lose_s, carry_s
        self.attack_window_s, self.attack_pan, self.attack_gap_s = attack_window_s, attack_pan, attack_gap_s
        self.roars = sorted(roars or [])
        self.stats = MatchStats()
        self.kits: set[str] = set()
        # holder state
        self._cand: tuple | None = None  # (holder id, team, shot)
        self._cand_since = 0.0
        self._holder: tuple | None = None  # confirmed (id, team, shot)
        self._holder_since = 0.0
        self._last_seen = -1e9
        self._team: str | None = None  # team in possession
        self._chain = 0
        self._carried = False
        self._carry_pan = 0.0
        self._pans: deque = deque()
        self._last_attack = -1e9
        self._view: str | None = None
        self._view_since = 0.0
        self._closeup_said_shot: int | None = None
        self._last_t: float | None = None

    # ---- helpers -------------------------------------------------------------------
    def _ev(self, kind: str, t: float, team: str | None = None, conf: float = 0.7, **meta) -> Event:
        if team and len(self.kits) == 2:
            meta.setdefault("opponent", next(k for k in self.kits if k != team))
        return Event(event=kind, t=round(t, 2), team=team, confidence=conf,
                     meta={k: v for k, v in meta.items() if v is not None})

    def _direction(self) -> str | None:
        s = sum(dx for _, dx in self._pans)
        if abs(s) < self.attack_pan:
            return None
        return "right" if s > 0 else "left"

    # ---- main ----------------------------------------------------------------------
    def feed(self, st: PlayState) -> list[Event]:
        out: list[Event] = []
        dt = 0.0 if self._last_t is None else max(0.0, st.t - self._last_t)
        prev_t = self._last_t if self._last_t is not None else st.t
        self._last_t = st.t
        self.kits |= set(st.team_counts)

        # crowd roars (precomputed from the audio) as we pass them
        while self.roars and self.roars[0][0] <= st.t:
            rt, z = self.roars.pop(0)
            if rt >= prev_t - 1.0:
                self.stats.roars += 1
                out.append(self._ev("roar", rt, self._team, conf=min(1.0, z / 5), z=round(z, 1)))

        # view changes: one "close-up" line per shot after a spell of wide play
        if st.view != self._view:
            self._view, self._view_since = st.view, st.t
        elif (st.view == "close_up" and st.t - self._view_since >= 2.0
              and self._closeup_said_shot != st.shot_id):
            self._closeup_said_shot = st.shot_id
            out.append(self._ev("close_up", self._view_since, self._team, conf=0.6))

        # camera pan window -> direction of play
        self._pans.append((st.t, st.cam_dx))
        while self._pans and st.t - self._pans[0][0] > self.attack_window_s:
            self._pans.popleft()

        if self._team:
            self.stats.possession_s[self._team] = self.stats.possession_s.get(self._team, 0.0) + dt

        # holder hysteresis
        cand = (st.holder, st.holder_team, st.shot_id) if st.holder is not None else None
        if cand is not None:
            self._last_seen = st.t
            if cand != self._cand:
                self._cand, self._cand_since = cand, st.t
            if st.t - self._cand_since >= self.hold_s - 1e-6 and cand != self._holder:
                out += self._new_holder(cand, self._cand_since)
        elif self._holder and st.t - self._last_seen > self.lose_s:
            self._holder, self._cand = None, None  # ball loose / out of view

        # carry: one player keeps it while the camera follows
        if self._holder and not self._carried:
            self._carry_pan += abs(st.cam_dx)
            if st.t - self._holder_since >= self.carry_s and self._carry_pan >= 0.12:
                self._carried = True
                out.append(self._ev("carry", self._holder_since, self._holder[1], conf=0.6,
                                    direction=self._direction()))

        # attack: sustained pan one way while a team has the ball
        d = self._direction()
        if d and self._team and st.view == "wide" and st.t - self._last_attack >= self.attack_gap_s:
            self._last_attack = st.t
            out.append(self._ev("attack", st.t, self._team, conf=0.6, direction=d,
                                pass_chain=self._chain or None))
        return out

    def _new_holder(self, cand: tuple, t: float) -> list[Event]:
        out: list[Event] = []
        prev = self._holder
        hid, team, shot = cand
        self._holder, self._holder_since, self._carried, self._carry_pan = cand, t, False, 0.0
        if team is None:
            return out
        if prev is not None and prev[2] == shot and prev[1] == team and prev[0] != hid:
            self._chain += 1
            self.stats.passes[team] = self.stats.passes.get(team, 0) + 1
            self.stats.longest_chain = max(self.stats.longest_chain, self._chain)
            out.append(self._ev("pass", t, team, conf=0.6, pass_chain=self._chain))
        elif self._team is not None and team != self._team:
            self.stats.turnovers += 1
            self._chain = 0
            out.append(self._ev("turnover", t, team, conf=0.6))
        elif self._team is None:
            out.append(self._ev("possession", t, team, conf=0.6))
        self._team = team
        return out

    def stats_event(self, t: float) -> Event:
        pct = self.stats.possession_pct()
        lead = max(pct, key=pct.get) if pct else None
        return self._ev("stats", t, lead, conf=1.0, possession_pct=pct[lead] if lead else None,
                        passes=self.stats.passes.get(lead) if lead else None)
