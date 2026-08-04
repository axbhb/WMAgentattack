#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
SCORES=/share/guozhix/wmagentattack/0712/grouped_user_task_dreamer
GROUPED=/share/guozhix/wmagentattack/0712/grouped_user_task_split_raw
ARCHIVE=/share/guozhix/wmagentattack/0713/grouped_task_confirmation

cd "$ROOT"
mkdir -p "$ARCHIVE"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PY" scripts/46_select_grouped_task_confirmation.py \
  --source "clean_seed7=$SCORES/seed7/test_clean_prefix_rollout_candidates.json" \
  --source "clean_seed13=$SCORES/seed13/test_clean_prefix_rollout_candidates.json" \
  --source "clean_seed21=$SCORES/seed21/test_clean_prefix_rollout_candidates.json" \
  --source "injection_seed7=$SCORES/seed7/test_injection_conditioned_rollout_candidates.json" \
  --source "injection_seed13=$SCORES/seed13/test_injection_conditioned_rollout_candidates.json" \
  --source "injection_seed21=$SCORES/seed21/test_injection_conditioned_rollout_candidates.json" \
  --train-trajectories "$GROUPED/train_trajectories.jsonl" \
  --test-trajectories "$GROUPED/test_trajectories.jsonl" \
  --output "$ARCHIVE/grouped_task_confirmation_selection.json" \
  >"$ARCHIVE/selection_stdout.json"

echo "GROUPED_TASK_CONFIRMATION_SELECTION_DONE $(date -Is)"
