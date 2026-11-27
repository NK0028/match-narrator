import json

import numpy as np
import pytest
import torch

from commentator.context.detect import Detection, FrameDets
from commentator.context.tagging import ContextExtractor
from commentator.context.teams import TeamClassifier
from commentator.generation.generator import CommentaryGenerator, dedupe, select
from commentator.generation.guard import is_grounded, violations
from commentator.generation.templates import TEMPLATES, render
from commentator.schema import Event, MatchInfo, parse_clock
from commentator.spotting.infer import temporal_nms, to_events, to_soccernet_json
from commentator.spotting.model import NetVLADPlusPlus
from commentator.textmetrics import all_metrics
from commentator.tts.engine import to_vtt


def ev(kind, t, **kw):
    return Event(event=kind, t=t, **kw)


def test_event_schema_and_facts():
    e = ev("goal", 4032, team="white", player="#7 (player_3)", confidence=0.88)
    assert e.timestamp == "67:12"
    assert e.facts() == {"event": "goal", "timestamp": "67:12", "team": "white", "player": "#7 (player_3)"}
    assert parse_clock("2 - 21:12") == 1272


def test_all_templates_are_grounded():
    for lang in ("en",):
        for kind, variants in TEMPLATES[lang].items():
            for i in range(len(variants) * 3):
                text = render(ev(kind, 10, team="white", player="player_3"), i, lang)
                assert is_grounded(text, {"event": kind, "team": "white", "player": "player_3"}), text


def test_guard_catches_hallucinations():
    assert violations("GOAL! What a finish!", {"event": "shot", "team": "red"})
    assert violations("Messi scores!", {"event": "goal", "team": "red"})
    assert violations("Number 10 takes the corner", {"event": "corner", "team": "red"})
    assert not violations("Corner to red.", {"event": "corner", "team": "red"})


def test_dedupe_and_redundancy():
    evs = [ev("shot", 30), ev("goal", 33, confidence=0.9), ev("goal", 35, confidence=0.5), ev("kickoff", 80)]
    d = dedupe(evs)
    assert [e.event for e in d] == ["shot", "goal", "kickoff"]
    assert [d[i].event for i in select(d)] == ["goal"]  # shot subsumed, post-goal kick-off implied


def test_generator_timing_never_overlaps():
    evs = [ev("corner", 10.0, team="red"), ev("shot", 12.0, team="red"), ev("card", 13.0, team="blue")]
    lines = CommentaryGenerator(use_llm=False).generate(evs)
    for a, b in zip(lines, lines[1:]):
        assert b.t >= a.end
    assert lines[0].t >= 10.0
    assert "WEBVTT" in to_vtt(lines)


def test_empty_events():
    assert CommentaryGenerator(use_llm=False).generate([]) == []


def test_nms_and_events():
    s = np.zeros(200)
    s[50], s[52], s[150] = 0.9, 0.8, 0.6
    keep = temporal_nms(s, 20, 0.5)
    assert [k[0] for k in keep] == [50, 150]
    preds = [{"frame": 50, "label": "Goal", "confidence": 0.9, "team": "home"},
             {"frame": 150, "label": "Corner", "confidence": 0.3, "team": "away"}]
    evs = to_events(preds, default_threshold=0.5)
    assert len(evs) == 1 and evs[0].event == "goal" and evs[0].t == 25.0
    js = to_soccernet_json("g", {1: preds})
    assert js["predictions"][0]["gameTime"] == "1 - 00:25"


def test_model_shapes():
    m = NetVLADPlusPlus(input_dim=512)
    logits, team = m(torch.randn(4, 30, 512))
    assert logits.shape == (4, 17) and team.shape == (4, 3)


def test_team_classifier_separates_kits():
    rng = np.random.default_rng(0)
    white = [np.array([235, 235, 235]) + rng.normal(0, 8, 3) for _ in range(10)]
    red = [np.array([200, 30, 40]) + rng.normal(0, 8, 3) for _ in range(10)]
    tc = TeamClassifier().fit(white + red)
    assert set(tc.names) == {"white", "red"}
    k_w, name_w, out_w = tc.predict(np.array([230, 230, 230]))
    assert name_w == "white" and not out_w
    assert tc.predict(np.array([240, 220, 30]))[2]  # yellow referee is an outlier


class FakeTracker:
    """Two teams standing on grass; ball next to a red player."""

    def track_window(self, video, t0, t1, **kw):
        frame = np.zeros((400, 600, 3), np.uint8)
        frame[:] = (40, 140, 40)  # BGR grass
        players = []
        for i in range(6):
            x = 50 + i * 90
            colour = (235, 235, 235) if i % 2 else (40, 30, 200)  # white / red in BGR
            frame[100:200, x:x + 40] = colour
            players.append(Detection(i + 1, 0, (x, 100, x + 40, 200), 0.9))
        return [FrameDets(t=(t0 + t1) / 2, frame=frame, players=players,
                          ball=Detection(None, 32, (60, 195, 70, 205), 0.8))]


def test_context_extractor_tags_player_and_team():
    ce = ContextExtractor(tracker=FakeTracker())
    match = MatchInfo()
    out = ce.enrich("x.mp4", [ev("shot", 5.0)], match)
    assert out[0].player == "player_1"
    assert out[0].team == "red"
    assert {match.home_colour, match.away_colour} == {"white", "red"}


def test_text_metrics_identity():
    m = all_metrics(["corner kick for the red side"], [["corner kick for the red side"]])
    assert m["BLEU-4"] > 0.99 and m["ROUGE-L"] > 0.99


def test_api_commentary_endpoint():
    from fastapi.testclient import TestClient

    from server.app import app

    c = TestClient(app)
    r = c.post("/api/commentary", json={"events": [json.loads(ev("goal", 12, team="home").model_dump_json())],
                                        "match": {"home": "Leeds", "away": "York"}})
    assert r.status_code == 200
    assert "Leeds" in r.json()[0]["text"]
    assert c.get("/api/health").json()["ok"]


@pytest.mark.parametrize("lang", ["ur"])
def test_urdu_templates_render(lang):
    assert render(ev("goal", 1, team="ٹیم", player="کھلاڑی"), 0, lang)


class FakeLLM:
    latencies: list = []

    def __init__(self, reply):
        self.reply = reply

    def line(self, facts, history, language="English"):
        return self.reply


def test_llm_lines_are_guarded():
    evs = [ev("shot", 10, team="red", player="player_2")]
    gen = CommentaryGenerator(use_llm=False)
    gen.llm = FakeLLM("GOAL! Messi wins it at the death!")  # hallucinated
    lines = gen.generate(evs)
    assert lines[0].source == "template" and len(gen.rejected) == 1
    gen = CommentaryGenerator(use_llm=False)
    gen.llm = FakeLLM("Player_2 lets fly from distance for red!")  # grounded
    assert gen.generate(evs)[0].source == "llm"
