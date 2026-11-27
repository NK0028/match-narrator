# Evaluation report

## 1. Action spotting (vision half)

SoccerNet-v2, official `SoccerNet.Evaluation.ActionSpotting` average-mAP on **50 test games** (test split games never used for training or threshold tuning).

| Metric | tight (1–5 s) | loose (5–60 s) |
|---|---|---|
| average-mAP | 23.1 | 41.8 |
| visible events | 26.4 | 47.7 |
| unshown events | 14.6 | 32.2 |

Per-class average-mAP (classes are heavily imbalanced, so mAP is reported, not accuracy):

| Class | tight | loose |
|---|---|---|
| Corner | 40.3 | 73.9 |
| Substitution | 25.5 | 64.6 |
| Ball out of play | 39.9 | 63.9 |
| Foul | 25.8 | 54.6 |
| Direct free-kick | 31.9 | 54.3 |
| Kick-off | 21.4 | 53.6 |
| Goal | 36.8 | 49.8 |
| Throw-in | 21.1 | 47.7 |
| Clearance | 26.6 | 45.6 |
| Shots off target | 30.0 | 38.3 |
| Penalty | 30.7 | 35.5 |
| Indirect free-kick | 10.9 | 34.6 |
| Yellow card | 14.6 | 33.5 |
| Offside | 19.0 | 28.2 |
| Shots on target | 17.7 | 28.1 |
| Red card | 0.3 | 3.8 |
| Yellow->red card | 0.4 | 1.2 |

Reference points (full 100-game test, published): NetVLAD++ ≈ 53.4 loose average-mAP on ResNet-PCA512 features; CALF ≈ 40.7.

**Training data: 61 of 300 train games, 15 of 100 valid games** (download bandwidth limited this run). With the full split, NetVLAD++-style models gain substantially.

Training: 37 epochs, best valid loss 0.4208.

## 2. Narration thresholds (tuned on valid, F-beta=0.7, ±5 s)

| Class | threshold | precision | recall | n (valid) |
|---|---|---|---|---|
| Penalty | 0.60 | 0.75 | 0.60 | 5 |
| Kick-off | 0.87 | 0.85 | 0.35 | 81 |
| Goal | 0.89 | 0.84 | 0.49 | 53 |
| Substitution | 0.86 | 0.64 | 0.48 | 77 |
| Offside | 0.83 | 0.70 | 0.18 | 38 |
| Shots on target | 0.75 | 0.48 | 0.28 | 165 |
| Shots off target | 0.85 | 0.49 | 0.28 | 140 |
| Clearance | 0.76 | 0.64 | 0.25 | 230 |
| Ball out of play | 0.51 | 0.72 | 0.55 | 940 |
| Throw-in | 0.59 | 0.59 | 0.37 | 529 |
| Foul | 0.60 | 0.63 | 0.36 | 248 |
| Indirect free-kick | 0.42 | 0.16 | 0.31 | 163 |
| Direct free-kick | 0.83 | 0.89 | 0.50 | 64 |
| Corner | 0.86 | 0.86 | 0.54 | 140 |
| Yellow card | 0.81 | 0.49 | 0.55 | 44 |
| Red card | 0.60 | 0.25 | 1.00 | 1 |
| Yellow->red card | 0.90 | 0.00 | 0.00 | 0 |

## 3. End-to-end commentary

Spotter → thresholds → de-dup → selection → commentary, on 50 test games; a line counts as real if an event of the same type is within ±10 s.

| Metric | Value |
|---|---|
| Narrated-event precision | 70.6% |
| Narrated-event recall (narratable GT events that got a line) | 33.8% |
| **System hallucination rate** (lines about an event that didn't happen) | **27.2%** |
| **Linguistic hallucination rate** (line text not grounded in its event JSON) | **0.0%** |
| Spotter team-head accuracy (home/away, matched lines) | 54.4% (n=1221) — ≈ chance, so **not used** to name a side |
| Lines per match minute | 0.72 |

Per event type:

| Type | precision | recall | TP | FP | FN |
|---|---|---|---|---|---|
| goal | 64.6 | 31.5 | 51 | 28 | 111 |
| shot | 63.2 | 24.1 | 220 | 128 | 694 |
| corner | 83.0 | 51.9 | 235 | 48 | 218 |
| card | 44.3 | 36.8 | 74 | 93 | 127 |
| kickoff | 74.0 | 62.6 | 77 | 27 | 46 |
| substitution | 78.8 | 58.9 | 152 | 41 | 106 |
| penalty | 50.0 | 33.3 | 2 | 2 | 4 |
| red_card | 0.0 | 0.0 | 0 | 0 | 11 |
| offside | 69.4 | 12.1 | 25 | 11 | 182 |
| foul | 75.4 | 29.8 | 337 | 110 | 794 |

### Text metrics vs SoccerNet-Caption references

Each generated line is scored against the anonymised SoccerNet-Caption comments within ±15 s (n=1238 lines).

| BLEU-1 | BLEU-4 | ROUGE-L | METEOR-lite | CIDEr-D |
|---|---|---|---|---|
| 0.012 | 0.002 | 0.092 | 0.053 | 0.038 |

These are low by design: references are long written match-report sentences ('This yellow card was deserved. The tackle by [PLAYER] was quite harsh…'), while ours are short spoken lines restricted to the spotted facts. They mainly serve as a baseline for the LLM-colour variant. METEOR-lite is exact-match METEOR (no stemming/synonyms).

### LLM colour path

Not measured in this run (no Anthropic credentials on the build machine). Run `python eval/run_eval.py --llm-games 5` with `ANTHROPIC_API_KEY` set to fill this in. The guard is unit-tested (`tests/test_core.py::test_llm_lines_are_guarded`).

## 4. Latency & size

| | |
|---|---|
| Spotter parameters | 720,980 |
| Spotter checkpoint | 2.75 MB |
| Spotting a full 45-min half (Apple M1 Pro, MPS) | 0.65 s |
| Spotting per 10-min clip | 0.14 s |
| Template commentary for a full half | 0.56 ms |

Offline v1: the whole track is generated before playback, so perceived latency is zero.

## 5. Human rating

`eval/human_rating_sheet.csv` holds 50 randomly sampled lines with their event JSON and empty fluency / factual / relevance (1–5) columns. Fill it in and run `python -c "import pandas as p; print(p.read_csv('eval/human_rating_sheet.csv').filter(like='_1_5').mean())"`.

## 6. Honest failure cases

**Missed goals** — 111 case(s). Examples:

- 2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal, half 1 @ 29:19
- 2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal, half 2 @ 36:34
- 2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea, half 1 @ 31:01
- 2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea, half 2 @ 39:13
- 2015-08-23 - 15-30 West Brom 2 - 3 Chelsea, half 1 @ 34:50

**False-positive big events (narrated, didn't happen)** — 30 case(s). Examples:

- 2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea, half 2 @ 04:24 — “GOAL! a player finds the net!” (conf 0.898)
- 2015-08-23 - 15-30 West Brom 2 - 3 Chelsea, half 2 @ 43:32 — “GOAL! The net bulges!” (conf 0.897)
- 2015-09-20 - 18-00 Southampton 2 - 3 Manchester United, half 2 @ 04:36 — “It's in! What a moment!” (conf 0.947)
- 2015-10-24 - 17-00 West Ham 2 - 1 Chelsea, half 2 @ 46:02 — “GOAL! a player finds the net!” (conf 0.934)
- 2015-11-07 - 20-30 Stoke City 1 - 0 Chelsea, half 1 @ 33:41 — “GOAL! a player finds the net!” (conf 0.906)

**Goal where the (unused) team head picked the wrong side** — 12 case(s). Examples:

- 2015-08-29 - 17-00 Liverpool 0 - 3 West Ham, half 1 @ 28:20 — “GOAL! The net bulges!” (conf )
- 2015-09-26 - 19-30 Newcastle Utd 2 - 2 Chelsea, half 2 @ 14:26 — “GOAL! a player finds the net!” (conf )
- 2015-09-26 - 19-30 Newcastle Utd 2 - 2 Chelsea, half 2 @ 40:59 — “It's in! What a moment!” (conf )
- 2016-09-24 - 14-30 Manchester United 4 - 1 Leicester, half 1 @ 39:03 — “GOAL! The net bulges!” (conf )
- 2016-10-15 - 14-30 Chelsea 3 - 0 Leicester, half 1 @ 32:19 — “GOAL! The net bulges!” (conf )

**Team attribution is a negative result.** The spotter's home/away head scores ≈ chance at every confidence level (diagnosed by bucketing accuracy by head confidence), because global frame features carry no notion of which side is acting. Rather than voice a coin-flip, commentary on SoccerNet windows uses team-neutral phrasing; team colours come from the YOLO + jersey k-means context layer on raw clips. A future fix is a team head fed by per-player colour features.

Typical causes: replays of a goal re-trigger the Goal class (handled partly by de-dup); team is predicted from features, which only weakly encode which side attacks; cards and penalties are rare (tens of training examples) so their mAP is lowest.
