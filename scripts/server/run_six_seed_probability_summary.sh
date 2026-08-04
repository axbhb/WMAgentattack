#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0710
HISTORIC="$ARCHIVE/selected_replay_multiseed"
FRESH="$ARCHIVE/replay_probability_fresh_multiseed"
RETROFIT="$ARCHIVE/replay_probability_retrofit_multiseed"
OUTPUT="$ARCHIVE/replay_probability_six_seed"
SELECTION="$ARCHIVE/replay_probability_calibration/fresh_replay_selections.json"
CLEAN_STATS="$ROOT/artifacts/clean_multiseed_llama31_70b_solvability.json"

cd "$ROOT"
mkdir -p "$OUTPUT/reconstructed"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PY" scripts/35_reconstruct_six_seed_probability_eval.py \
  --selection-json "$SELECTION" \
  --historic-replay "$HISTORIC/seed7/replay.json" \
  --historic-replay "$HISTORIC/seed13/replay.json" \
  --historic-replay "$HISTORIC/seed21/replay.json" \
  --retrofit-replay "$RETROFIT/seed7/replay.json" \
  --retrofit-replay "$RETROFIT/seed13/replay.json" \
  --retrofit-replay "$RETROFIT/seed21/replay.json" \
  --output-dir "$OUTPUT/reconstructed" \
  >"$OUTPUT/reconstruction_stdout.json"

"$PY" scripts/30_summarize_selected_replay_multiseed.py \
  --replay "$OUTPUT/reconstructed/seed7_reconstructed.json" \
  --replay "$OUTPUT/reconstructed/seed13_reconstructed.json" \
  --replay "$OUTPUT/reconstructed/seed21_reconstructed.json" \
  --replay "$FRESH/seed31/replay.json" \
  --replay "$FRESH/seed37/replay.json" \
  --replay "$FRESH/seed43/replay.json" \
  --clean-solvability-json "$CLEAN_STATS" \
  --bootstrap-samples 20000 \
  --bootstrap-seed 20260711 \
  --output "$OUTPUT/multiseed_summary.json" \
  >"$OUTPUT/multiseed_summary_stdout.json"

echo "SIX_SEED_PROBABILITY_SUMMARY_DONE $(date -Is)"
