#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0712/within_task_contrast
SELECTION="$ARCHIVE/within_task_confirmation_selection.json"
REPLAYS="$ARCHIVE/confirmation_replays"
IC="$ARCHIVE/injection_conditioned_scores"
DATASET="$ARCHIVE/confirmation_probability_dataset"
MODELS="$ARCHIVE/hierarchical_models"
OUTPUT="$ARCHIVE/confirmation_evaluation"

cd "$ROOT"
mkdir -p "$DATASET" "$OUTPUT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PY" scripts/37_merge_within_task_contrast_replays.py \
  --selection-json "$SELECTION" \
  --selection-prefix within_task_confirmation \
  --output-stem within_task_confirmation \
  --replay-root "$REPLAYS" \
  --base-seeds 51,57,63,69,75 \
  --chunks 4 \
  --injection-source "continuous_seed7=$IC/continuous/seed7_train_candidates.json" \
  --injection-source "continuous_seed13=$IC/continuous/seed13_train_candidates.json" \
  --injection-source "continuous_seed21=$IC/continuous/seed21_train_candidates.json" \
  --injection-source "norank_seed7=$IC/norank/seed7_train_candidates.json" \
  --injection-source "norank_seed13=$IC/norank/seed13_train_candidates.json" \
  --injection-source "norank_seed21=$IC/norank/seed21_train_candidates.json" \
  --injection-source "binary_seed7=$IC/binary/seed7_train_candidates.json" \
  --injection-source "binary_seed13=$IC/binary/seed13_train_candidates.json" \
  --injection-source "binary_seed21=$IC/binary/seed21_train_candidates.json" \
  --output-dir "$DATASET" \
  >"$DATASET/merge_stdout.json"

"$PY" scripts/41_evaluate_within_task_confirmation.py \
  --training-dataset "$ARCHIVE/probability_dataset/within_task_contrast_probability_dataset.json" \
  --confirmation-dataset "$DATASET/within_task_confirmation_probability_dataset.json" \
  --main-summary "$MODELS/hierarchical_contrast_summary.json" \
  --gate-decision "$MODELS/confirmation_gate_decision.json" \
  --analysis-protocol "$ROOT/configs/0712_within_task_confirmation_analysis.json" \
  --bootstrap-samples 20000 \
  --bootstrap-seed 20260713 \
  --output-dir "$OUTPUT" \
  >"$OUTPUT/evaluation_stdout.json"

echo "WITHIN_TASK_CONFIRMATION_EVALUATION_DONE $(date -Is)"
