#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0712/within_task_contrast
SELECTION="$ARCHIVE/within_task_contrast_selection.json"
REPLAYS="$ARCHIVE/replays"
IC="$ARCHIVE/injection_conditioned_scores"
DATASET="$ARCHIVE/probability_dataset"
MODELS="$ARCHIVE/hierarchical_models"
PROTOCOL="$ROOT/configs/0712_within_task_confirmation_gate.json"

cd "$ROOT"
mkdir -p "$DATASET" "$MODELS"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PY" scripts/37_merge_within_task_contrast_replays.py \
  --selection-json "$SELECTION" \
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

"$PY" scripts/38_evaluate_hierarchical_contrast_models.py \
  --dataset "$DATASET/within_task_contrast_probability_dataset.json" \
  --cv-seeds 101,211,307,401,503 \
  --bootstrap-samples 20000 \
  --bootstrap-seed 20260712 \
  --output-dir "$MODELS" \
  >"$MODELS/evaluation_stdout.json"

"$PY" scripts/40_apply_within_task_confirmation_gate.py \
  --summary "$MODELS/hierarchical_contrast_summary.json" \
  --protocol "$PROTOCOL" \
  --output "$MODELS/confirmation_gate_decision.json" \
  >"$MODELS/confirmation_gate_stdout.json"

bash scripts/server/submit_within_task_confirmation_if_go.sh

echo "WITHIN_TASK_CONTRAST_EVALUATION_DONE $(date -Is)"
