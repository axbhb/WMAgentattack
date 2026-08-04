#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0710
PILOT="$ARCHIVE/validation_probability_pilot"
OUTPUT="$ARCHIVE/replay_probability_calibration"
CANDIDATES="$ARCHIVE/candidate_ranker_robustness/repeated_cv_seed_ensemble"
CLEAN_STATS="$ROOT/artifacts/clean_multiseed_llama31_70b_solvability.json"
BASELINES="$ARCHIVE/selected_replay_multiseed/frozen_replay_selections.json"

cd "$ROOT"
mkdir -p "$OUTPUT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PY" scripts/32_fit_replay_probability_calibrators.py \
  --selection-json "$PILOT/validation_probability_pilot.json" \
  --selection-name validation_probability_pilot \
  --replay "$PILOT/seed7/replay.json" \
  --replay "$PILOT/seed13/replay.json" \
  --replay "$PILOT/seed21/replay.json" \
  --test-candidates "$CANDIDATES/seed7_test_candidates.json" \
  --clean-solvability-json "$CLEAN_STATS" \
  --baseline-selections-json "$BASELINES" \
  --cv-seeds 101,211,307,401,503 \
  --folds 5 \
  --top-k 16 \
  --utility-floors 0.333333,0.5,0.666667 \
  --utility-uncertainty-weight 0.5 \
  --max-per-user-task 2 \
  --output-dir "$OUTPUT" \
  >"$OUTPUT/calibration_stdout.json"

echo "REPLAY_PROBABILITY_CALIBRATION_DONE $(date -Is)"
