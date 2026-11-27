"""End-to-end evaluation of the commentary system on SoccerNet test games.

Needs spotter predictions from `python -m commentator.spotting.evaluate` (written to
outputs/predictions/<game>/results_spotting.json). Measures:
  * event precision / recall *into the commentary* (did real events get narrated?)
  * system hallucination rate (lines narrating an event that didn't happen)
  * linguistic hallucination rate (line text not grounded in its event JSON)
  * team attribution accuracy, per-type P/R, failure cases
  * text metrics vs SoccerNet-Caption references (if Labels-caption.json present)
  * latency per clip + model size
and writes eval/system_metrics.json, eval/failure_cases.json, eval/human_rating_sheet.csv.

    python eval/run_eval.py [--llm-games 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commentator.generation.generator import CommentaryGenerator, dedupe, select  # noqa: E402
from commentator.generation.guard import violations  # noqa: E402
from commentator.pipeline import DEFAULT_THRESHOLD, SPOTTER_TEAM_MIN_CONF, ground_truth_events, load_thresholds  # noqa: E402
from commentator.schema import MatchInfo  # noqa: E402
from commentator.spotting.infer import to_events  # noqa: E402
from commentator.textmetrics import all_metrics  # noqa: E402

KEY_TYPES = ["goal", "shot", "corner", "card", "kickoff", "substitution", "penalty", "red_card", "offside",
             "free_kick", "foul"]


def load_preds(path: str, half: int) -> list[dict]:
    with open(path) as f:
        P = json.load(f)["predictions"]
    return [dict(frame=int(int(p["position"]) / 1000 * 2), label=p["label"], confidence=float(p["confidence"]),
                 team=p.get("team"), team_conf=p.get("team_conf")) for p in P if int(p["half"]) == half]


def captions(gdir: str, half: int) -> list[tuple[float, str]]:
    p = os.path.join(gdir, "Labels-caption.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        ann = json.load(f)["annotations"]
    return [(int(a["position"]) / 1000, a["anonymized"]) for a in ann
            if int(a["gameTime"].split(" - ")[0]) == half]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/SoccerNet")
    ap.add_argument("--pred-dir", default="outputs/predictions")
    ap.add_argument("--tol", type=float, default=10.0, help="seconds for a narrated event to count as real")
    ap.add_argument("--llm-games", type=int, default=0, help="also run the LLM path on N games")
    ap.add_argument("--ckpt", default="models/spotter.pt")
    a = ap.parse_args()

    thresholds = load_thresholds()
    games = sorted(os.path.relpath(os.path.dirname(os.path.join(dp, f)), a.pred_dir)
                   for dp, _, fs in os.walk(a.pred_dir) for f in fs if f == "results_spotting.json")
    per_type = {t: {"tp": 0, "fp": 0, "fn": 0} for t in KEY_TYPES}
    team_ok = team_n = 0
    lines_total = lines_hall = ling_viol = 0
    hyps, refs = [], []
    failures, sheet_rows = [], []
    llm_stats = {"lines": 0, "llm_lines": 0, "violations": 0, "rejected": 0, "latency": []}

    for gi, g in enumerate(games):
        gdir = os.path.join(a.root, g)
        lab = os.path.join(gdir, "Labels-v2.json")
        with open(lab) as f:
            L = json.load(f)
        match = MatchInfo(home=L["gameHomeTeam"], away=L["gameAwayTeam"])
        for half in (1, 2):
            preds = load_preds(os.path.join(a.pred_dir, g, "results_spotting.json"), half)
            pev = dedupe(to_events(preds, half=half, thresholds=thresholds, default_threshold=DEFAULT_THRESHOLD,
                                   team_min_conf=SPOTTER_TEAM_MIN_CONF))
            raw_team = {round(p["frame"] / 2, 1): p["team"] for p in preds}
            gev = dedupe(ground_truth_events(lab, half))
            g_sel = [gev[i] for i in select(gev)]
            use_llm = gi < a.llm_games
            gen = CommentaryGenerator(use_llm=use_llm)
            lines = gen.generate(pev, match)
            matched_gt = set()
            for ln in lines:
                ev = pev[ln.event_index]
                lines_total += 1
                v = violations(ln.text, ev.facts(), {match.home, match.away})
                ling_viol += bool(v)
                cand = [j for j, ge in enumerate(gev) if ge.event == ev.event and abs(ge.t - ev.t) <= a.tol]
                hit = min(cand, key=lambda j: abs(gev[j].t - ev.t)) if cand else None
                if ev.event in per_type:
                    per_type[ev.event]["tp" if hit is not None else "fp"] += 1
                if hit is None:
                    lines_hall += 1
                    if ev.event in ("goal", "penalty", "red_card"):
                        failures.append({"game": g, "half": half, "type": "false_positive", "event": ev.event,
                                         "t": ev.timestamp, "confidence": ev.confidence, "line": ln.text})
                else:
                    matched_gt.add(hit)
                    head = raw_team.get(round(ev.t, 1))
                    if gev[hit].team in ("home", "away") and head in ("home", "away"):
                        team_n += 1
                        team_ok += gev[hit].team == head
                        if gev[hit].team != head and ev.event == "goal":
                            failures.append({"game": g, "half": half, "type": "wrong_team", "event": "goal",
                                             "t": ev.timestamp, "line": ln.text})
                if use_llm:
                    llm_stats["lines"] += 1
                    llm_stats["llm_lines"] += ln.source == "llm"
                    llm_stats["violations"] += bool(v)
                caps = [c for t, c in captions(gdir, half) if abs(t - ev.t) <= 15]
                if caps:
                    hyps.append(ln.text)
                    refs.append(caps)
                sheet_rows.append({"game": g, "half": half, "time": ev.timestamp, "event_json": json.dumps(ev.facts()),
                                   "line": ln.text, "source": ln.source, "fluency_1_5": "", "factual_1_5": "",
                                   "relevance_1_5": ""})
            if use_llm and gen.llm:
                llm_stats["rejected"] += len(gen.rejected)
                llm_stats["latency"] += gen.llm.latencies
            # recall: GT narratable events that got a line
            for ge in g_sel:
                if ge.event not in per_type:
                    continue
                if not any(gev[j] is ge for j in matched_gt):
                    per_type[ge.event]["fn"] += 1
                    if ge.event == "goal":
                        failures.append({"game": g, "half": half, "type": "missed", "event": "goal",
                                         "t": ge.timestamp})

    def pr(d):
        p = d["tp"] / max(d["tp"] + d["fp"], 1)
        r = d["tp"] / max(d["tp"] + d["fn"], 1)
        return {"precision": round(p, 3), "recall": round(r, 3), **d}

    tot = {k: sum(per_type[t][k] for t in KEY_TYPES) for k in ("tp", "fp", "fn")}
    out = {
        "n_games": len(games), "tolerance_s": a.tol,
        "narrated_events": pr(tot),
        "per_type": {t: pr(d) for t, d in per_type.items()},
        "system_hallucination_rate": round(lines_hall / max(lines_total, 1), 4),
        "linguistic_hallucination_rate": round(ling_viol / max(lines_total, 1), 4),
        "team_head_accuracy": round(team_ok / max(team_n, 1), 3), "team_n": team_n,
        "team_named_in_commentary": SPOTTER_TEAM_MIN_CONF <= 1.0,
        "lines_total": lines_total,
        "lines_per_minute": round(lines_total / max(len(games) * 90, 1), 2),
    }
    if hyps:
        out["text_metrics_vs_soccernet_caption"] = {k: round(v, 4) if isinstance(v, float) else v
                                                    for k, v in all_metrics(hyps, refs).items()}
    if a.llm_games:
        lat = llm_stats.pop("latency")
        llm_stats["latency_p50_s"] = round(float(np.median(lat)), 2) if lat else None
        llm_stats["hallucination_rate_after_guard"] = round(llm_stats["violations"] / max(llm_stats["lines"], 1), 4)
        out["llm"] = llm_stats

    # latency + size
    lat = {}
    if os.path.exists(a.ckpt) and games:
        from commentator.spotting.infer import spot_half
        from commentator.spotting.model import load_checkpoint, pick_device

        dev = pick_device()
        m, ck = load_checkpoint(a.ckpt, dev)
        feats = np.load(os.path.join(a.root, games[0], "1_ResNET_TF2_PCA512.npy"))
        t0 = time.time()
        spot_half(m, feats, dev, int(ck.get("window_s", 15) * 2))
        lat["spotting_full_half_s"] = round(time.time() - t0, 2)
        lat["spotting_per_10min_clip_s"] = round(lat["spotting_full_half_s"] * 600 / (len(feats) / 2), 2)
        lat["spotter_params"] = int(sum(p.numel() for p in m.parameters()))
        lat["spotter_size_mb"] = round(os.path.getsize(a.ckpt) / 2 ** 20, 2)
        t0 = time.time()
        CommentaryGenerator(use_llm=False).generate(pev, match)
        lat["template_generation_per_half_ms"] = round(1000 * (time.time() - t0), 2)
    out["latency"] = lat

    os.makedirs("eval", exist_ok=True)
    with open("eval/system_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    with open("eval/failure_cases.json", "w") as f:
        json.dump(failures, f, indent=1)
    random.Random(0).shuffle(sheet_rows)
    with open("eval/human_rating_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet_rows[0].keys()) if sheet_rows else ["line"])
        w.writeheader()
        w.writerows(sheet_rows[:50])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
