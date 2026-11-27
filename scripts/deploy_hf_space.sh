#!/usr/bin/env bash
# Deploy the app as a Hugging Face Docker Space (HF builds the Dockerfile remotely).
#   scripts/deploy_hf_space.sh <hf-user>/<space-name> [public|private]
# Optional afterwards, in the Space settings: add secret ANTHROPIC_API_KEY to enable LLM colour lines.
set -euo pipefail
SPACE="${1:?usage: deploy_hf_space.sh user/space [public|private]}"
VIS="${2:-public}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cd "$ROOT"
test -f models/spotter.pt && test -f models/thresholds.json
rsync -a --exclude node_modules --exclude dist web/ "$STAGE/web/"
cp -R commentator server Dockerfile .dockerignore requirements.txt "$STAGE/"
mkdir -p "$STAGE/models" && cp models/spotter.pt models/thresholds.json "$STAGE/models/"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
cat > "$STAGE/README.md" <<'MD'
---
title: Match Narrator
emoji: ⚽
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Football video → spotted events → grounded AI commentary
---
Action spotting (NetVLAD++ on SoccerNet features) → structured events → template + guarded-LLM
commentary → TTS, played in sync. Demo clips use a synthetic match view; commentary comes from
the spotter's predictions on real SoccerNet test-match features.
MD

FLAG=""; [ "$VIS" = "private" ] && FLAG="--private"
hf repo create "$SPACE" --repo-type space --space-sdk docker $FLAG --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy Match Narrator"
echo "https://huggingface.co/spaces/$SPACE"
