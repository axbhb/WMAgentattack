#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0713/headwise_remaining_confirmation_v2
SELECTION="$ARCHIVE/headwise_remaining_confirmation_selection.json"
DATASET="$ARCHIVE/probability_dataset"
OUTPUT="$ARCHIVE/evaluation"
GROUPED=/share/guozhix/wmagentattack/0712/grouped_user_task_split_raw

cd "$ROOT"
mkdir -p "$DATASET" "$OUTPUT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PY" scripts/37_merge_within_task_contrast_replays.py \
  --selection-json "$SELECTION" \
  --selection-prefix headwise_remaining_confirmation \
  --output-stem headwise_remaining_confirmation \
  --replay-root "$ARCHIVE/replays" \
  --base-seeds 127,131,137,139,149 \
  --chunks 4 \
  --output-dir "$DATASET" \
  >"$DATASET/merge_stdout.json"

"$PY" scripts/53_evaluate_headwise_remaining_confirmation.py \
  --dataset "$DATASET/headwise_remaining_confirmation_probability_dataset.json" \
  --selection "$SELECTION" \
  --protocol "$ROOT/configs/0713_headwise_remaining_confirmation_protocol.json" \
  --train-trajectories "$GROUPED/train_trajectories.jsonl" \
  --test-trajectories "$GROUPED/test_trajectories.jsonl" \
  --bootstrap-samples 20000 \
  --bootstrap-seed 20260719 \
  --output-dir "$OUTPUT" \
  >"$OUTPUT/evaluation_stdout.json"

echo "HEADWISE_REMAINING_EVALUATION_DONE $(date -Is)"
