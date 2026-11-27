"""Assemble eval/REPORT.md from the metric JSON files.

    python eval/make_report.py
"""
from __future__ import annotations

import json
import os

E = "eval"


def load(name):
    p = os.path.join(E, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x):
    return f"{100 * x:.1f}"


def main():
    sp, sy, th = load("spotting_metrics.json"), load("system_metrics.json"), load("thresholds_report.json")
    fails = load("failure_cases.json") or []
    hist = json.load(open("models/spotter_history.json")) if os.path.exists("models/spotter_history.json") else []
    out = ["# Evaluation report", ""]

    if sp:
        out += ["## 1. Action spotting (vision half)", "",
                f"SoccerNet-v2, official `SoccerNet.Evaluation.ActionSpotting` average-mAP on **{sp['n_games']} test games** "
                "(test split games never used for training or threshold tuning).", "",
                "| Metric | tight (1–5 s) | loose (5–60 s) |", "|---|---|---|"]
        for k, lab in (("a_mAP", "average-mAP"), ("a_mAP_visible", "visible events"), ("a_mAP_unshown", "unshown events")):
            out.append(f"| {lab} | {pct(sp['tight'][k])} | {pct(sp['loose'][k])} |")
        out += ["", "Per-class average-mAP (classes are heavily imbalanced, so mAP is reported, not accuracy):", "",
                "| Class | tight | loose |", "|---|---|---|"]
        for c in sorted(sp["loose"]["per_class"], key=lambda c: -sp["loose"]["per_class"][c]):
            out.append(f"| {c} | {pct(sp['tight']['per_class'][c])} | {pct(sp['loose']['per_class'][c])} |")
        out += ["", "Reference points (full 100-game test, published): NetVLAD++ ≈ 53.4 loose average-mAP on "
                "ResNet-PCA512 features; CALF ≈ 40.7.", ""]
        try:
            from SoccerNet.utils import getListGames

            n_tr = sum(os.path.exists(f"data/SoccerNet/{g}/2_ResNET_TF2_PCA512.npy") for g in getListGames("train"))
            n_va = sum(os.path.exists(f"data/SoccerNet/{g}/2_ResNET_TF2_PCA512.npy") for g in getListGames("valid"))
            out += [f"**Training data: {n_tr} of 300 train games, {n_va} of 100 valid games** (download bandwidth "
                    "limited this run). With the full split, NetVLAD++-style models gain substantially.", ""]
        except Exception:
            pass
        if hist:
            out += [f"Training: {len(hist)} epochs, best valid loss {min(h['valid_loss'] for h in hist):.4f}.", ""]

    if th:
        out += ["## 2. Narration thresholds (tuned on valid, F-beta=0.7, ±5 s)", "",
                "| Class | threshold | precision | recall | n (valid) |", "|---|---|---|---|---|"]
        for c, r in th.items():
            out.append(f"| {c} | {r['threshold']:.2f} | {r['precision']:.2f} | {r['recall']:.2f} | {r['n_gt']} |")
        out.append("")

    if sy:
        ne = sy["narrated_events"]
        out += ["## 3. End-to-end commentary", "",
                f"Spotter → thresholds → de-dup → selection → commentary, on {sy['n_games']} test games; "
                f"a line counts as real if an event of the same type is within ±{sy['tolerance_s']:.0f} s.", "",
                "| Metric | Value |", "|---|---|",
                f"| Narrated-event precision | {pct(ne['precision'])}% |",
                f"| Narrated-event recall (narratable GT events that got a line) | {pct(ne['recall'])}% |",
                f"| **System hallucination rate** (lines about an event that didn't happen) | **{pct(sy['system_hallucination_rate'])}%** |",
                f"| **Linguistic hallucination rate** (line text not grounded in its event JSON) | **{pct(sy['linguistic_hallucination_rate'])}%** |",
                f"| Spotter team-head accuracy (home/away, matched lines) | {pct(sy['team_head_accuracy'])}% (n={sy['team_n']}) — ≈ chance, so **not used** to name a side |",
                f"| Lines per match minute | {sy['lines_per_minute']} |", "",
                "Per event type:", "", "| Type | precision | recall | TP | FP | FN |", "|---|---|---|---|---|---|"]
        for t, d in sy["per_type"].items():
            if d["tp"] + d["fp"] + d["fn"]:
                out.append(f"| {t} | {pct(d['precision'])} | {pct(d['recall'])} | {d['tp']} | {d['fp']} | {d['fn']} |")
        out.append("")
        if "text_metrics_vs_soccernet_caption" in sy:
            tm = sy["text_metrics_vs_soccernet_caption"]
            out += ["### Text metrics vs SoccerNet-Caption references", "",
                    f"Each generated line is scored against the anonymised SoccerNet-Caption comments within ±15 s "
                    f"(n={tm['n']} lines).", "",
                    "| BLEU-1 | BLEU-4 | ROUGE-L | METEOR-lite | CIDEr-D |", "|---|---|---|---|---|",
                    f"| {tm['BLEU-1']:.3f} | {tm['BLEU-4']:.3f} | {tm['ROUGE-L']:.3f} | {tm['METEOR-lite']:.3f} | {tm['CIDEr-D']:.3f} |", "",
                    "These are low by design: references are long written match-report sentences ('This yellow card was "
                    "deserved. The tackle by [PLAYER] was quite harsh…'), while ours are short spoken lines restricted to "
                    "the spotted facts. They mainly serve as a baseline for the LLM-colour variant. METEOR-lite is exact-match "
                    "METEOR (no stemming/synonyms).", ""]
        if "llm" in sy:
            l = sy["llm"]
            out += ["### LLM colour path", "",
                    f"{l['llm_lines']}/{l['lines']} lines from the LLM, {l['rejected']} rejected by the grounding guard, "
                    f"hallucination rate after guard {pct(l['hallucination_rate_after_guard'])}%, p50 latency {l['latency_p50_s']} s.", ""]
        else:
            out += ["### LLM colour path", "",
                    "Not measured in this run (no Anthropic credentials on the build machine). Run "
                    "`python eval/run_eval.py --llm-games 5` with `ANTHROPIC_API_KEY` set to fill this in. The guard is "
                    "unit-tested (`tests/test_core.py::test_llm_lines_are_guarded`).", ""]
        lat = sy.get("latency", {})
        if lat:
            out += ["## 4. Latency & size", "", "| | |", "|---|---|",
                    f"| Spotter parameters | {lat['spotter_params']:,} |",
                    f"| Spotter checkpoint | {lat['spotter_size_mb']} MB |",
                    f"| Spotting a full 45-min half (Apple M1 Pro, MPS) | {lat['spotting_full_half_s']} s |",
                    f"| Spotting per 10-min clip | {lat['spotting_per_10min_clip_s']} s |",
                    f"| Template commentary for a full half | {lat['template_generation_per_half_ms']} ms |", "",
                    "Offline v1: the whole track is generated before playback, so perceived latency is zero.", ""]

    out += ["## 5. Human rating", "",
            "`eval/human_rating_sheet.csv` holds 50 randomly sampled lines with their event JSON and empty "
            "fluency / factual / relevance (1–5) columns. Fill it in and run "
            "`python -c \"import pandas as p; print(p.read_csv('eval/human_rating_sheet.csv').filter(like='_1_5').mean())\"`.", ""]

    if fails:
        out += ["## 6. Honest failure cases", ""]
        by = {}
        for f in fails:
            by.setdefault(f["type"], []).append(f)
        names = {"false_positive": "False-positive big events (narrated, didn't happen)",
                 "missed": "Missed goals", "wrong_team": "Goal where the (unused) team head picked the wrong side"}
        for k, lst in by.items():
            out += [f"**{names.get(k, k)}** — {len(lst)} case(s). Examples:", ""]
            for f in lst[:5]:
                g = f["game"].split("/")[-1]
                extra = f" — “{f['line']}” (conf {f.get('confidence', '')})" if "line" in f else ""
                out.append(f"- {g}, half {f['half']} @ {f['t']}{extra}")
            out.append("")
        out += ["**Team attribution is a negative result.** The spotter's home/away head scores ≈ chance at every confidence "
                "level (diagnosed by bucketing accuracy by head confidence), because global frame features carry no notion "
                "of which side is acting. Rather than voice a coin-flip, commentary on SoccerNet windows uses team-neutral "
                "phrasing; team colours come from the YOLO + jersey k-means context layer on raw clips. A future fix is a "
                "team head fed by per-player colour features.", "",
                "Typical causes: replays of a goal re-trigger the Goal class (handled partly by de-dup); "
                "team is predicted from features, which only weakly encode which side attacks; "
                "cards and penalties are rare (tens of training examples) so their mAP is lowest.", ""]
    with open(os.path.join(E, "REPORT.md"), "w") as f:
        f.write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
