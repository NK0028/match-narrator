"""Template fast-path: no model call, near-zero latency, cannot hallucinate.

Several variants per event type; choice is deterministic per (event index, seed)
so a clip always renders the same track.
"""
from __future__ import annotations

import random

from ..schema import Event

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "en": {
        "goal": [
            "GOAL! {player} finds the net for the {team} side!",
            "It's in! {player} scores for {team}!",
            "GOAL! What a moment for {team} — {player} with the finish!",
        ],
        "shot": [
            "{player} has a go — shot by the {team} team.",
            "{player} lets fly for {team}!",
            "A strike from {player}, {team} pushing forward.",
        ],
        "corner": [
            "Corner kick for the {team} side.",
            "{team} win a corner.",
            "It's behind — corner to {team}.",
        ],
        "card": [
            "The referee reaches for a yellow — {player} is booked.",
            "Yellow card, {player} of {team} goes into the book.",
        ],
        "red_card": [
            "Red card! {player} is sent off — {team} down to ten.",
            "Off he goes! A red card for {player}.",
        ],
        "kickoff": [
            "And we're under way — {team} get us started.",
            "Kick-off, {team} in possession.",
        ],
        "substitution": [
            "A change for {team}.",
            "Substitution for the {team} side.",
        ],
        "penalty": [
            "Penalty! The referee points to the spot for {team}!",
            "It's a penalty to {team}!",
        ],
        "offside": [
            "The flag is up — offside against {team}.",
            "Offside, {team} caught too far forward.",
        ],
        "foul": [
            "Foul given, the referee blows his whistle.",
            "Free kick awarded after a foul.",
        ],
        "free_kick": [
            "Free kick to {team}.",
            "{team} with a set piece here.",
        ],
        "clearance": ["Cleared away by {team}."],
        "throw_in": ["Throw-in to {team}."],
        "ball_out": ["The ball goes out of play."],
    },
    # Stretch goal: Urdu commentary (romanised Urdu would also work with English TTS).
    "ur": {
        "goal": ["گول! {player} نے {team} کے لیے گیند جال میں ڈال دی!", "یہ گول ہے! {team} کے لیے {player}!"],
        "shot": ["{player} کا زوردار شاٹ، {team} کی جانب سے!"],
        "corner": ["{team} کو کارنر مل گیا۔"],
        "card": ["ریفری نے پیلا کارڈ دکھا دیا — {player}۔"],
        "red_card": ["سرخ کارڈ! {player} میدان سے باہر۔"],
        "kickoff": ["کھیل شروع ہو گیا — {team} کی کک آف۔"],
        "substitution": ["{team} کی طرف سے تبدیلی۔"],
        "penalty": ["پنالٹی! {team} کو پنالٹی مل گئی!"],
        "offside": ["آف سائیڈ! {team} کے خلاف جھنڈا بلند۔"],
        "foul": ["فاؤل، ریفری نے سیٹی بجا دی۔"],
        "free_kick": ["{team} کو فری کک۔"],
        "clearance": ["{team} نے گیند کلیئر کر دی۔"],
        "throw_in": ["{team} کو تھرو اِن۔"],
        "ball_out": ["گیند میدان سے باہر۔"],
    },
}

# Team-neutral variants, used when the team is unknown (rather than guessing a side).
NO_TEAM: dict[str, dict[str, list[str]]] = {
    "en": {
        "goal": ["GOAL! {player} finds the net!", "It's in! What a moment!", "GOAL! The net bulges!"],
        "shot": ["{player} has a go!", "A strike from distance!", "He lets fly!"],
        "corner": ["Corner kick.", "It's behind — that's a corner.", "A corner to come."],
        "card": ["The referee reaches for a yellow card.", "Yellow card — {player} goes into the book."],
        "red_card": ["Red card! {player} is sent off!", "Off he goes — a red card!"],
        "kickoff": ["And we're under way.", "Kick-off, and play is under way."],
        "substitution": ["A substitution is being made.", "Change of personnel here."],
        "penalty": ["Penalty! The referee points to the spot!"],
        "offside": ["The flag is up — offside.", "Offside, caught too far forward."],
        "foul": ["Foul given, the referee blows his whistle.", "Free kick awarded after a foul."],
        "free_kick": ["A free kick here.", "A set piece opportunity."],
        "clearance": ["Cleared away.", "Hacked clear by the defence.", "Headed away, danger averted."],
        "throw_in": ["Throw-in, play restarts from the touchline.", "A throw-in down the line.",
                     "Quick throw taken."],
        "ball_out": ["The ball runs out of play.", "That one's drifted out.", "Overhit, and it's out."],
    },
}

# Template words that must never be whitelisted as "common" capitalised words.
NO_TEMPLATE_WORDS = {"player", "team"}

# What we say when a field is unknown — neutral wording rather than a guess.
UNKNOWN = {
    "en": {"player": "a player", "team": "attacking"},
    "ur": {"player": "ایک کھلاڑی", "team": "حملہ آور ٹیم"},
}


def describe_team(ev: Event, match=None) -> str | None:
    if not ev.team:
        return None
    if match is not None:
        if ev.team == "home" and match.home != "home":
            return match.home
        if ev.team == "away" and match.away != "away":
            return match.away
    return ev.team


def render(ev: Event, idx: int = 0, lang: str = "en", seed: int = 0, match=None) -> str:
    lang = lang if lang in TEMPLATES else "en"
    team = describe_team(ev, match)
    if team is None and lang in NO_TEAM:
        variants = NO_TEAM[lang][ev.event]
    else:
        variants = TEMPLATES[lang].get(ev.event) or TEMPLATES["en"][ev.event]
    rng = random.Random(f"{seed}-{idx}-{ev.event}")
    tpl = rng.choice(variants)
    team = team or UNKNOWN[lang]["team"]
    player = ev.player or UNKNOWN[lang]["player"]
    text = tpl.format(player=player, team=team)
    # Tidy the "a player" / "attacking side" phrasings at sentence start.
    return text[0].upper() + text[1:]
