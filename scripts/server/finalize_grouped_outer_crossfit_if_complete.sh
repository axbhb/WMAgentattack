#!/usr/bin/env bash

set -euo pipefail

ARCHIVE=/share/guozhix/wmagentattack/0713/grouped_outer_crossfit
LOCK="$ARCHIVE/finalize.lock"
SUMMARY="$ARCHIVE/evaluation/grouped_train_hybrid_validation.json"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "GROUPED_OUTER_CROSSFIT_FINALIZE_ALREADY_LOCKED"
  exit 0
fi
if [[ -s "$SUMMARY" ]]; then
  echo "GROUPED_OUTER_CROSSFIT_FINALIZE_ALREADY_DONE"
  exit 0
fi

missing=0
for fold in 0 1 2 3; do
  for seed in 7 13 21; do
    if [[ ! -f "$ARCHIVE/scores/fold${fold}/seed${seed}/DONE" ]]; then
      missing=$((missing + 1))
    fi
  done
done
if (( missing > 0 )); then
  echo "GROUPED_OUTER_CROSSFIT_FINALIZE_PENDING missing=$missing"
  exit 0
fi

cd /share/guozhix/WMagentattack
bash scripts/server/run_merge_and_evaluate_grouped_outer_crossfit.sh
