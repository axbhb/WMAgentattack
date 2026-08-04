#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0712/within_task_contrast
CLEAN_STATS="$ROOT/artifacts/clean_multiseed_llama31_70b_solvability.json"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PY" scripts/42_select_within_task_external_test.py \
  --source "continuous_seed7=$ROOT/artifacts/full_dreamer_v3_llama31_70b/seed7_test_candidates.json" \
  --source "continuous_seed13=$ROOT/artifacts/full_dreamer_v3_llama31_70b/seed13_test_candidates.json" \
  --source "continuous_seed21=$ROOT/artifacts/full_dreamer_v3_llama31_70b/seed21_test_candidates.json" \
  --source "norank_seed7=$ROOT/artifacts/full_dreamer_v3_llama31_70b_hybrid_lite_norank/seed7_test_candidates.json" \
  --source "norank_seed13=$ROOT/artifacts/full_dreamer_v3_llama31_70b_hybrid_lite_norank/seed13_test_candidates.json" \
  --source "norank_seed21=$ROOT/artifacts/full_dreamer_v3_llama31_70b_hybrid_lite_norank/seed21_test_candidates.json" \
  --source "binary_seed7=$ROOT/artifacts/next_round_70b_ablation/full_binary/seed7_test_candidates.json" \
  --source "binary_seed13=$ROOT/artifacts/next_round_70b_ablation/full_binary/seed13_test_candidates.json" \
  --source "binary_seed21=$ROOT/artifacts/next_round_70b_ablation/full_binary/seed21_test_candidates.json" \
  --primary-source continuous_seed7 \
  --clean-solvability-json "$CLEAN_STATS" \
  --exclude-selection-json "$ARCHIVE/within_task_contrast_selection.json" \
  --exclude-selection-json "$ARCHIVE/within_task_confirmation_selection.json" \
  --output "$ARCHIVE/within_task_external_test_selection.json" \
  >"$ARCHIVE/external_test_selection_stdout.json"

echo "WITHIN_TASK_EXTERNAL_TEST_SELECTION_DONE $(date -Is)"
