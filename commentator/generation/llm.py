"""LLM "colour" commentary via Claude. Only ever sees the structured event facts.

Enabled when Anthropic credentials are available (ANTHROPIC_API_KEY, or an
`ant auth login` profile). Model is configurable with COMMENTARY_MODEL.
"""
from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("COMMENTARY_MODEL", "claude-opus-5-5")

SYSTEM = """You are a live football TV commentator. You receive ONE event as JSON facts \
plus the few events just before it. Write ONE short spoken commentary line (max 20 words) \
about the current event.

Hard rules:
- Use only the facts given. Never mention any event type other than the current one \
(e.g. don't say "goal" for a shot, don't invent saves, fouls, scores or minutes).
- Refer to the player and team exactly as given; if a field is missing, stay generic \
("a player", "they"). Never invent names, jersey numbers or scores.
- Plain text only, no quotes, no emojis, no hashtags.
Language: {language}."""


class ClaudeColour:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low"):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=1, timeout=30.0)
        self.model = model
        self.effort = effort
        self.calls = 0
        self.latencies: list[float] = []

    def line(self, facts: dict, history: list[dict], language: str = "English") -> str | None:
        return self._cached(json.dumps(facts, sort_keys=True), json.dumps(history[-3:], sort_keys=True), language)

    @lru_cache(maxsize=512)  # cache repeated phrasings / identical requests
    def _cached(self, facts_json: str, history_json: str, language: str) -> str | None:
        user = f"Previous events: {history_json}\nCurrent event: {facts_json}\nCommentary line:"
        t0 = time.time()
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=2048,  # includes the model's (low-effort) thinking
                system=SYSTEM.format(language=language),
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except self.anthropic.APIConnectionError as e:
            log.warning("LLM connection failed: %s", e)
            return None
        except self.anthropic.RateLimitError as e:
            log.warning("LLM rate limited: %s", e)
            return None
        except self.anthropic.APIStatusError as e:
            log.warning("LLM error %s: %s", e.status_code, e.message)
            return None
        self.calls += 1
        self.latencies.append(time.time() - t0)
        if resp.stop_reason == "refusal":
            return None
        text = " ".join(b.text for b in resp.content if b.type == "text").strip().strip('"')
        return text or None


def available() -> bool:
    if os.environ.get("COMMENTARY_LLM", "auto") == "off":
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return os.path.isdir(os.path.expanduser("~/.config/anthropic")) and os.environ.get("COMMENTARY_LLM") == "on"


LIVE_MODEL = os.environ.get("COMMENTARY_LIVE_MODEL", "claude-haiku-5-5")

LIVE_SYSTEM = """You are a live football TV commentator calling the match as it happens. \
Each request gives you the CURRENT moment as JSON facts measured by a video tracker, plus \
your last few spoken lines. Reply with ONE short spoken line (max {max_words} words).

Hard rules:
- Describe only the current moment's facts. Never mention goals, shots, saves, fouls, \
cards, corners, penalties or scores unless the current event type is that event.
- Teams are named by kit colour (e.g. "white"); use them exactly as given, or stay generic. \
Never invent player names, shirt numbers, minutes, scores or pitch positions.
- Only say "left"/"right" if the facts include that direction.
- Vary your phrasing: do not repeat or closely paraphrase your recent lines.
- Plain text only, no quotes or emojis.
Language: {language}."""


class LiveColour(ClaudeColour):
    """Low-latency play-by-play lines for live mode (one line per moment)."""

    def __init__(self, model: str = LIVE_MODEL, effort: str = "low", max_words: int = 12):
        super().__init__(model=model, effort=effort)
        self.max_words = max_words

    def live_line(self, facts: dict, recent_lines: list[str], language: str = "English") -> str | None:
        user = (f"Your recent lines: {json.dumps(recent_lines[-4:])}\n"
                f"Current moment: {json.dumps(facts, sort_keys=True)}\nLine:")
        t0 = time.time()
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=1024,
                system=LIVE_SYSTEM.format(language=language, max_words=self.max_words),
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except self.anthropic.APIError as e:
            log.warning("live LLM error: %s", e)
            return None
        self.calls += 1
        self.latencies.append(time.time() - t0)
        if resp.stop_reason in ("refusal", "max_tokens"):
            return None
        text = " ".join(b.text for b in resp.content if b.type == "text").strip().strip('"')
        return text or None
