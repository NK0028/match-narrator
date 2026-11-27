# Match Narrator — football video → AI commentary

A system that watches football match video, **spots key events with timestamps**, attaches
**team and player context**, and generates **grounded natural-language commentary** that is
**voiced and played in sync** with the clip.

```
match video ─► frames/features ─► ACTION SPOTTING ─► detect+track+tag players ─► STRUCTURED EVENTS ─► commentary ─► TTS ─► web player
              (SoccerNet 2 fps     (NetVLAD++,        (YOLO11 + ByteTrack,         {"event":"goal",       (templates +   (say /     (React, synced
               ResNet-PCA512)       17 classes+team)   k-means jersey colours)      "timestamp":"67:12",   guarded LLM)   espeak)    captions+audio)
                                                                                    "team":"white", ...}
            └──────────────── computer vision ────────────────┘                     └────── language & audio ──────┘
```

**Demo video:** [`docs/demo.mp4`](docs/demo.mp4) — 90 s of a SoccerNet test match narrated from the spotter's predictions.

The JSON event list is the hinge: the language side **never sees pixels**, only facts. That keeps
the system debuggable, the LLM swappable, and — measured below — the hallucination rate near zero.

**Results**: see the [summary](#results-summary) below; full tables in [`eval/REPORT.md`](eval/REPORT.md).

---

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cd web && npm ci && npm run build && cd ..
uvicorn server.app:app --port 8000      # open http://localhost:8000
```

The repo ships a trained spotter (`models/spotter.pt`, 2.8 MB) and three demo clips in
`web/public/demo/`, so the app works without downloading any data.

## Reproduce everything (8-week plan, end to end)

| Week | Step | Command |
|---|---|---|
| 1 | Download SoccerNet labels + precomputed features (no NDA needed) | `python scripts/download_soccernet.py --splits train valid test` |
| 2 | Train the NetVLAD++ spotter on features | `python -m commentator.spotting.train --epochs 40` |
| 2 | Official average-mAP (tight + loose, per class) | `python -m commentator.spotting.evaluate --split test` |
| 3 | Context layer on a raw clip (YOLO + ByteTrack + jersey k-means) | `python -m commentator.pipeline video clip.mp4` |
| 4–5 | Commentary + voice for a SoccerNet window | `python -m commentator.pipeline soccernet "<game>" --half 1 --start 0 --end 600` |
| 6 | Web app | `uvicorn server.app:app` (+ `npm run dev` in `web/` for hot reload) |
| 7 | Tune per-class narration thresholds on valid | `python eval/tune_thresholds.py` |
| 8 | System evaluation + report | `python eval/run_eval.py && python eval/make_report.py` |
| 8 | Rebuild demo clips | `python scripts/build_demo.py` |

Tests: `pytest -q`.

## How it works

### 1. Action spotting — `commentator/spotting/`
- **Input**: SoccerNet's 2 fps ResNet-152 + PCA-512 features (raw video is never decoded for spotting).
- **Model**: NetVLAD++ (Giancola & Ghanem 2021): a 15 s window around each frame is split into
  *before* and *after* halves, each pooled by its own NetVLAD (32 clusters), concatenated, then
  a 17-way sigmoid head (SoccerNet classes) plus a 3-way **team head** (home / away / n.a.).
- **Training**: Gaussian soft targets peaking at the event frame (σ = 1.5 s) for tight temporal
  precision; positives = every event (jittered ±2 s) + 2× random negatives per epoch;
  inverse-frequency `pos_weight` (capped at 20) against class imbalance; early stopping on valid.
- **Inference**: dense sliding window + per-class temporal NMS → SoccerNet-format predictions →
  per-class thresholds (tuned on valid, precision-weighted) → `Event` objects.

### 2. Context — `commentator/context/`
Run **only on short windows around spotted events** (−1.5 s / +0.5 s), never a full match.
- `detect.py`: YOLO11 persons + ball with ByteTrack IDs; a histogram jump detects camera cuts and
  resets the tracker, so `player_N` tags are honest *per-shot* identities.
- `teams.py`: torso crops with grass masked out → Lab colour → k-means (k = 2) → readable kit names
  ("white", "red"); far-from-both-clusters players (referee, keepers) are left untagged rather than guessed.
- `tagging.py`: for ball events, the player nearest the ball is the subject; optional jersey-number OCR
  (`pip install easyocr`) maps `#7` → roster name when the number is readable.

### 3. Commentary — `commentator/generation/`
- `generator.py`: de-duplicates nearby same-type spots, drops redundant lines (shot right before a
  goal, kick-off after a goal), keeps lines from talking over each other, and anchors each line to its
  event time with a small lag (goals land 0.6 s after the ball hits the net).
- `templates.py`: the **fast path** — several variants per event type (English + Urdu), zero latency,
  cannot hallucinate.
- `llm.py`: optional **colour** on big moments only (goals, shots, cards, penalties) via Claude
  (`COMMENTARY_MODEL`, default `claude-opus-5-5`, low effort, server-side refusal fallback). The prompt
  contains only the event JSON and the previous three events.
- `guard.py`: every LLM line is checked against its event facts (event-type keywords, invented numbers,
  unlicensed proper names); a failing line is replaced by the template. The same checker measures the
  hallucination rate offline.

### 4. Voice — `commentator/tts/engine.py`
macOS `say`, `espeak-ng` (Docker/Linux), pyttsx3 or Coqui TTS. Excitement (from event importance)
raises speaking rate/pitch for goals. Lines are voiced individually, re-timed so speech never overlaps,
mixed into one track at their event times, encoded to AAC, and optionally muxed over the ducked original audio.

### 5. App — `server/` + `web/`
- FastAPI: `GET /api/clips`, `GET /api/clips/{id}`, `POST /api/process` (upload a clip, optional events
  JSON), `GET /api/jobs/{id}`, and `POST /api/commentary` (the language half alone: events in, timed lines out).
- React: video with caption overlay and a commentary audio track locked to the video clock, a
  spotted-vs-ground-truth event timeline (click to seek), live stats (score, shots, corners, cards),
  the commentary feed (template vs LLM badges, confidence), and the raw event JSON the LLM sees.
- The build also works as a **static site**: without a backend it reads `demo/index.json`.

## Data & licensing
- SoccerNet labels/features are public; raw broadcast videos need the SoccerNet NDA and are **not used
  or redistributed**. Demo clips are a **synthetic top-down match view** (`scripts/render_demo_clip.py`):
  player motion is synthetic, the ball is scripted to ground-truth event locations, the clock is the real
  match clock. The **commentary comes from the trained spotter's predictions** on that match's real
  features, and the timeline shows both prediction and truth so you can see the misses.
- For your own footage, use clips you recorded or have permission to use.

## Environment variables
| Var | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | enables the LLM colour path |
| `COMMENTARY_MODEL` | `claude-opus-5-5` | Claude model for colour lines |
| `COMMENTARY_LLM` | `auto` | `off` forces templates |
| `TTS_BACKEND` | `auto` | `say` / `espeak` / `pyttsx3` / `coqui` / `none` |
| `YOLO_WEIGHTS` | `yolo11n.pt` | detector weights |
| `ALLOW_UPLOADS`, `MAX_UPLOAD_MB` | `1`, `200` | upload endpoint controls |

## Known limitations (v1 scope)
- Player identity is per-shot only; cross-cut re-identification (SoccerNet Re-ID) is a stretch goal.
- Spotting on raw uploads uses torchvision ResNet-152 features, which differ from SoccerNet's TF2
  features — treat it as experimental, or supply an events JSON.
- Team on SoccerNet windows comes from the spotter's team head (home/away), not jersey colours.
- No fouls/offside judgement, tactics or true real-time streaming.

## Results summary

Measured on **50 SoccerNet-v2 test games** (unseen in training); spotter trained on 61 of 300 train games.
Full tables and failure cases: [`eval/REPORT.md`](eval/REPORT.md).

| | |
|---|---|
| Spotting average-mAP, loose / tight (official SoccerNet eval) | **41.8 / 23.1** |
| Per-class loose mAP: corner / goal / substitution / yellow card | 73.9 / 49.8 / 64.6 / 33.5 |
| Narrated-event precision / recall (±10 s) | 70.6% / 33.8% |
| **Linguistic hallucination rate** (line not grounded in its event JSON) | **0.0%** |
| System hallucination rate (line about an event that didn't happen — spotter false positives) | 27.2% |
| Spotter: params / size / full 45-min half on M1 Pro | 720,980 / 2.75 MB / 0.65 s |

Honest notes: published NetVLAD++ reaches 53.4 loose mAP with all 300 training games — this run used
~20% of them. The spotter's home/away team head scored ≈ chance (54%), so commentary on
SoccerNet windows is deliberately team-neutral rather than guessing a side. The LLM colour path is
implemented and unit-tested but wasn't measured here (no API key on the build machine).

## Deployment

**Live demo (static):** https://match-narrator.netlify.app (Netlify) · mirror: https://huggingface.co/spaces/nk8560/match-narrator — plays the three demo clips; uploads need the Docker backend.

**Netlify** builds `web/` from [`netlify.toml`](netlify.toml) (`npm ci && npm run build`, publish `web/dist`). Connect the repo in Netlify, or deploy by hand with `netlify deploy --prod` from the repo root.

> Hugging Face Docker Spaces now require a PRO plan, so the public demo is a free *static* Space built from `web/dist/`.

**Hugging Face Space (full app, uploads included)** — HF builds the `Dockerfile` remotely:
```bash
scripts/deploy_hf_space.sh <hf-user>/match-narrator public
```
Optionally add an `ANTHROPIC_API_KEY` secret in the Space settings to enable LLM colour lines.

**Any Docker host**: `docker build -t match-narrator . && docker run -p 7860:7860 match-narrator`.

**Static demo only** (no uploads): `cd web && npm run build` and host `web/dist/` anywhere.

## Repository layout
```
commentator/   schema.py · pipeline.py · textmetrics.py
  spotting/    data.py model.py train.py infer.py evaluate.py features.py
  context/     detect.py teams.py tagging.py
  generation/  templates.py llm.py guard.py generator.py
  tts/         engine.py
server/app.py  FastAPI
web/           React + Vite player
eval/          tune_thresholds.py run_eval.py make_report.py REPORT.md + metric JSONs
scripts/       download_soccernet.py render_demo_clip.py build_demo.py
models/        spotter.pt thresholds.json
```
