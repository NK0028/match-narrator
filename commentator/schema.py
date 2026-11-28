"""The structured event interface — the strict boundary between vision and language.

The commentary side only ever sees a list of `Event` objects, never pixels.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# SoccerNet-v2 action-spotting classes (order matters: it is the model's output order
# and matches SoccerNet.Evaluation's EVENT_DICTIONARY_V2).
SOCCERNET_CLASSES: list[str] = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside", "Shots on target",
    "Shots off target", "Clearance", "Ball out of play", "Throw-in", "Foul",
    "Indirect free-kick", "Direct free-kick", "Corner", "Yellow card", "Red card",
    "Yellow->red card",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(SOCCERNET_CLASSES)}

# Commentary-level event types. In-scope v1 types first; the rest are
# "robustness" extras (week 7) that the generator also handles.
EventType = Literal[
    "goal", "shot", "corner", "card", "kickoff", "substitution",
    "penalty", "offside", "foul", "free_kick", "red_card", "clearance", "throw_in",
    "ball_out",
    # live play-by-play moments (from tracking + crowd audio, see commentator/live/)
    "pass", "possession", "turnover", "carry", "attack", "roar", "close_up", "stats", "play",
]

SOCCERNET_TO_EVENT: dict[str, EventType] = {
    "Goal": "goal",
    "Shots on target": "shot",
    "Shots off target": "shot",
    "Corner": "corner",
    "Yellow card": "card",
    "Red card": "red_card",
    "Yellow->red card": "red_card",
    "Kick-off": "kickoff",
    "Substitution": "substitution",
    "Penalty": "penalty",
    "Offside": "offside",
    "Foul": "foul",
    "Direct free-kick": "free_kick",
    "Indirect free-kick": "free_kick",
    "Clearance": "clearance",
    "Throw-in": "throw_in",
    "Ball out of play": "ball_out",
}

# Event types narrated by default. Low-value, high-frequency classes are skipped
# so the commentary track stays a few lines per minute.
# free_kick is excluded: its spotting precision is ~0.4 on test and foul lines already
# cover it ("Free kick awarded after a foul") — see eval/REPORT.md.
DEFAULT_NARRATED: set[str] = {
    "goal", "shot", "corner", "card", "red_card", "kickoff", "substitution",
    "penalty", "offside", "foul",
}

# Higher = more important. Used for de-duplication tie-breaks, LLM "colour" choice,
# highlight ranking and excitement of the voice.
IMPORTANCE: dict[str, int] = {
    "goal": 10, "penalty": 9, "red_card": 9, "card": 6, "shot": 6, "kickoff": 4,
    "corner": 4, "free_kick": 4, "substitution": 3, "offside": 3, "foul": 2,
    "clearance": 1, "throw_in": 1, "ball_out": 0,
    "roar": 7, "attack": 5, "turnover": 3, "possession": 2, "carry": 2, "pass": 1,
    "close_up": 1, "stats": 0, "play": 0,
}

# Live moment types: narrated continuously in live mode, never by the batch generator.
LIVE_TYPES: set[str] = {"pass", "possession", "turnover", "carry", "attack", "roar", "close_up", "stats", "play"}


def fmt_clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def parse_clock(s: str) -> float:
    """'67:12' -> 4032.0 ; also accepts SoccerNet '2 - 21:12' (half - mm:ss)."""
    if " - " in s:
        s = s.split(" - ", 1)[1]
    parts = [float(p) for p in s.split(":")]
    out = 0.0
    for p in parts:
        out = out * 60 + p
    return out


class Event(BaseModel):
    """One spotted event. `t` is seconds into the clip/half being narrated."""

    event: EventType
    t: float = Field(ge=0, description="seconds from start of the video")
    timestamp: str = ""  # mm:ss, derived from t when missing
    team: Optional[str] = None  # "home"/"away" or a jersey colour such as "white"
    player: Optional[str] = None  # "#7 (player_3)", "player_3", or a roster name
    confidence: float = Field(default=1.0, ge=0, le=1)
    source_label: Optional[str] = None  # original SoccerNet class, for audit
    half: Optional[int] = None
    meta: dict = Field(default_factory=dict)  # pitch zone, ball position, ...

    @model_validator(mode="after")
    def _ts(self):
        if not self.timestamp:
            self.timestamp = fmt_clock(self.t)
        return self

    def facts(self) -> dict:
        """The only fields the language model is allowed to see."""
        d = {"event": self.event, "timestamp": self.timestamp}
        if self.team:
            d["team"] = self.team
        if self.player:
            d["player"] = self.player
        if self.meta.get("zone"):
            d["zone"] = self.meta["zone"]
        if self.meta.get("score"):
            d["score_after"] = self.meta["score"]
        for k in ("direction", "pass_chain", "possession_pct", "opponent", "passes"):
            if self.meta.get(k) is not None:
                d[k] = self.meta[k]
        return d


class CommentaryLine(BaseModel):
    t: float  # when the line should start (event time + lead/lag offset)
    end: float
    text: str
    event_index: int  # index into the events list it narrates (grounding link)
    event: EventType
    source: Literal["template", "llm"] = "template"
    excitement: float = 0.5  # 0..1, drives TTS rate/pitch
    audio: Optional[str] = None  # relative path to the voiced line


class MatchInfo(BaseModel):
    home: str = "home"
    away: str = "away"
    home_colour: Optional[str] = None
    away_colour: Optional[str] = None
    roster: dict[str, dict[str, str]] = Field(default_factory=dict)  # team -> {number: name}
    language: str = "en"


class CommentaryTrack(BaseModel):
    clip_id: str
    duration: float
    match: MatchInfo = Field(default_factory=MatchInfo)
    events: list[Event]
    lines: list[CommentaryLine]
    stats: dict = Field(default_factory=dict)
    audio_track: Optional[str] = None
    video: Optional[str] = None
