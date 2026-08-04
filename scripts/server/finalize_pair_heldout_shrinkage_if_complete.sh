#!/usr/bin/env bash

set -euo pipefail

ARCHIVE=/share/guozhix/wmagentattack/0712/within_task_contrast
LOCK="$ARCHIVE/pair_heldout_finalize.lock"
SUMMARY="$ARCHIVE/pair_heldout_shrinkage_validation/pair_heldout_shrinkage_summary.json"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "PAIR_HELDOUT_FINALIZE_ALREADY_LOCKED"
  exit 0
fi
if [[ -s "$SUMMARY" ]]; then
  echo "PAIR_HELDOUT_FINALIZE_ALREADY_DONE"
  exit 0
fi

missing=0
for base_seed in 51 57 63 69 75; do
  for chunk in 0 1 2 3; do
    replay="$ARCHIVE/external_test_replays/base_seed${base_seed}/chunk${chunk}/replay.json"
    if [[ ! -s "$replay" ]]; then
      missing=$((missing + 1))
    fi
  done
done
if (( missing > 0 )); then
  echo "PAIR_HELDOUT_FINALIZE_PENDING missing=$missing"
  exit 0
fi

cd /share/guozhix/WMagentattack
bash scripts/server/run_merge_and_evaluate_pair_heldout_shrinkage.sh
