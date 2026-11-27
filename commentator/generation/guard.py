"""Hallucination guard: does a commentary line mention an event that didn't happen?

Used twice: online, to reject an LLM line (falling back to the template), and
offline, to measure the system's hallucination rate.
"""
from __future__ import annotations

import re

# Keyword -> event types that license the keyword.
KEYWORDS: dict[str, set[str]] = {
    r"\bgoal\b|\bscores?\b|\bscored\b|\bnet\b|\bit'?s in\b|\bequali[sz]": {"goal"},
    r"\bpenalty\b|\bspot\b": {"penalty"},
    r"\bred card\b|\bsent off\b|\bdown to ten\b|\bdismiss": {"red_card"},
    r"\byellow\b|\bbooked\b|\bbooking\b|\bcaution": {"card", "red_card"},
    r"\bcorner\b": {"corner"},
    r"\boffside\b|\bflag\b": {"offside"},
    r"\bsubstitut|\bcomes on\b|\bcomes off\b|\bchange for\b": {"substitution"},
    r"\bkick-?off\b|\bunder ?way\b": {"kickoff", "goal"},
    r"\bshot\b|\bstrike\b|\bshoots?\b|\blets fly\b|\bsave[sd]?\b|\bkeeper\b": {"shot", "goal", "penalty"},
    r"\bfoul\b|\bwhistle\b": {"foul", "free_kick", "card", "red_card", "penalty", "offside"},
    r"\bfree[- ]kick\b|\bset piece\b": {"free_kick", "foul", "card", "red_card"},
}

NUMBER = re.compile(r"#?\b(\d{1,2})\b")


def violations(text: str, event: dict, allowed_names: set[str] | None = None) -> list[str]:
    """Reasons this line is not grounded in `event` (empty list = grounded)."""
    t = text.lower()
    ev = event["event"]
    out = []
    for pat, licensed in KEYWORDS.items():
        if re.search(pat, t) and ev not in licensed:
            out.append(f"mentions '{re.search(pat, t).group(0)}' but event is {ev}")
    # Jersey numbers / scores must come from the facts
    fact_text = " ".join(str(v) for v in event.values()).lower()
    for m in NUMBER.finditer(t):
        if m.group(1) not in fact_text:
            out.append(f"invented number {m.group(0)}")
    # Proper names: any capitalised word not in the facts / allowed set
    allowed = {w for n in (allowed_names or set()) for w in re.findall(r"[a-z]+", str(n).lower())}
    allowed |= set(re.findall(r"[a-z]+", fact_text))
    for name in re.findall(r"\b([A-Z][a-z]{2,})\b", text):
        if name.lower() not in allowed and name.lower() not in COMMON_CAPS:
            out.append(f"unlicensed name {name}")
    return out


COMMON_CAPS = {
    "what", "and", "the", "it", "its", "goal", "off", "red", "yellow", "corner", "penalty", "free",
    "a", "an", "oh", "there", "that", "this", "now", "here", "superb", "brilliant", "great",
    "card", "referee", "player", "team", "side", "kick", "shot", "offside", "foul", "wow",
    "home", "away", "cleared", "throw", "substitution", "change", "strike", "booked", "kick-off",
    "lets", "has", "goes", "what", "moment", "unlucky", "close", "chance", "attack", "they", "he", "his", "into", "from", "for", "with", "unbelievable", "magnificent",
}


def is_grounded(text: str, event: dict, allowed_names: set[str] | None = None) -> bool:
    return not violations(text, event, allowed_names)


def _template_vocab() -> set[str]:
    from .templates import NO_TEMPLATE_WORDS, NO_TEAM, TEMPLATES

    words = set()
    for table in (TEMPLATES, NO_TEAM):
        for lang in table.values():
            for variants in lang.values():
                for v in variants:
                    words |= set(re.findall(r"[a-z]+", v.lower()))
    return words - NO_TEMPLATE_WORDS


COMMON_CAPS |= _template_vocab()
