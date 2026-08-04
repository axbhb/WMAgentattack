#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0712/within_task_contrast
SELECTION="$ARCHIVE/within_task_external_test_selection.json"
REPLAYS="$ARCHIVE/external_test_replays"
IC="$ARCHIVE/injection_conditioned_scores"
DATASET="$ARCHIVE/external_test_probability_dataset"
OUTPUT="$ARCHIVE/pair_heldout_shrinkage_validation"

cd "$ROOT"
mkdir -p "$DATASET" "$OUTPUT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PY" scripts/37_merge_within_task_contrast_replays.py \
  --selection-json "$SELECTION" \
  --selection-prefix within_task_external_test \
  --output-stem within_task_external_test \
  --replay-root "$REPLAYS" \
  --base-seeds 51,57,63,69,75 \
  --chunks 4 \
  --injection-source "continuous_seed7=$IC/continuous/seed7_test_candidates.json" \
  --injection-source "continuous_seed13=$IC/continuous/seed13_test_candidates.json" \
  --injection-source "continuous_seed21=$IC/continuous/seed21_test_candidates.json" \
  --injection-source "norank_seed7=$IC/norank/seed7_test_candidates.json" \
  --injection-source "norank_seed13=$IC/norank/seed13_test_candidates.json" \
  --injection-source "norank_seed21=$IC/norank/seed21_test_candidates.json" \
  --injection-source "binary_seed7=$IC/binary/seed7_test_candidates.json" \
  --injection-source "binary_seed13=$IC/binary/seed13_test_candidates.json" \
  --injection-source "binary_seed21=$IC/binary/seed21_test_candidates.json" \
  --output-dir "$DATASET" \
  >"$DATASET/merge_stdout.json"

"$PY" scripts/44_evaluate_pair_heldout_shrinkage.py \
  --training-dataset "$ARCHIVE/probability_dataset/within_task_contrast_probability_dataset.json" \
  --validation-dataset "$DATASET/within_task_external_test_probability_dataset.json" \
  --protocol "$ROOT/configs/0712_pair_heldout_shrinkage_protocol.json" \
  --bootstrap-samples 20000 \
  --bootstrap-seed 20260714 \
  --output-dir "$OUTPUT" \
  >"$OUTPUT/evaluation_stdout.json"

echo "PAIR_HELDOUT_SHRINKAGE_EVALUATION_DONE $(date -Is)"
