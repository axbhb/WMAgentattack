#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0712/within_task_contrast
SCORES="$ARCHIVE/train_candidate_scores"
ROBUST=/share/guozhix/wmagentattack/0710/candidate_ranker_robustness/repeated_cv_seed_ensemble
OUTPUT="$ARCHIVE/within_task_contrast_selection.json"
CLEAN_STATS="$ROOT/artifacts/clean_multiseed_llama31_70b_solvability.json"

cd "$ROOT"
mkdir -p "$ARCHIVE"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PY" scripts/36_select_within_task_contrast.py \
  --source "continuous_seed7=$SCORES/continuous/seed7_train_candidates.json" \
  --source "continuous_seed13=$SCORES/continuous/seed13_train_candidates.json" \
  --source "continuous_seed21=$SCORES/continuous/seed21_train_candidates.json" \
  --source "norank_seed7=$SCORES/norank/seed7_train_candidates.json" \
  --source "norank_seed13=$SCORES/norank/seed13_train_candidates.json" \
  --source "norank_seed21=$SCORES/norank/seed21_train_candidates.json" \
  --source "binary_seed7=$SCORES/binary/seed7_train_candidates.json" \
  --source "binary_seed13=$SCORES/binary/seed13_train_candidates.json" \
  --source "binary_seed21=$SCORES/binary/seed21_train_candidates.json" \
  --primary-source continuous_seed7 \
  --clean-solvability-json "$CLEAN_STATS" \
  --exclude-candidates "$ROBUST/seed7_val_candidates.json" \
  --exclude-candidates "$ROBUST/seed7_test_candidates.json" \
  --tasks-per-suite 6 \
  --pairs-per-task 4 \
  --chunks 4 \
  --output "$OUTPUT" \
  >"$ARCHIVE/selection_stdout.json"

echo "WITHIN_TASK_CONTRAST_SELECTION_DONE $(date -Is)"
