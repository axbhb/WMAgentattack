#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0713/grouped_task_confirmation
SELECTION="$ARCHIVE/grouped_task_confirmation_selection.json"
DATASET="$ARCHIVE/probability_dataset"
OUTPUT="$ARCHIVE/evaluation"
GROUPED=/share/guozhix/wmagentattack/0712/grouped_user_task_split_raw
DEVELOPMENT=/share/guozhix/wmagentattack/0712/within_task_contrast/probability_dataset/within_task_contrast_probability_dataset.json

cd "$ROOT"
mkdir -p "$DATASET" "$OUTPUT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PY" scripts/37_merge_within_task_contrast_replays.py \
  --selection-json "$SELECTION" \
  --selection-prefix grouped_task_confirmation \
  --output-stem grouped_task_confirmation \
  --replay-root "$ARCHIVE/replays" \
  --base-seeds 91,97,103,109,115 \
  --chunks 4 \
  --output-dir "$DATASET" \
  >"$DATASET/merge_stdout.json"

"$PY" scripts/47_evaluate_grouped_task_confirmation.py \
  --dataset "$DATASET/grouped_task_confirmation_probability_dataset.json" \
  --selection "$SELECTION" \
  --protocol "$ROOT/configs/0713_grouped_task_confirmation_protocol.json" \
  --development-dataset "$DEVELOPMENT" \
  --train-trajectories "$GROUPED/train_trajectories.jsonl" \
  --test-trajectories "$GROUPED/test_trajectories.jsonl" \
  --bootstrap-samples 20000 \
  --bootstrap-seed 20260716 \
  --output-dir "$OUTPUT" \
  >"$OUTPUT/evaluation_stdout.json"

echo "GROUPED_TASK_CONFIRMATION_EVALUATION_DONE $(date -Is)"

