#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0713/grouped_outer_crossfit
OOF="$ARCHIVE/oof"
EVALUATION="$ARCHIVE/evaluation"
DATA=/share/guozhix/wmagentattack/0712/grouped_user_task_split_continuous_probability

cd "$ROOT"
mkdir -p "$OOF" "$EVALUATION"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

"$PY" scripts/57_merge_grouped_outer_crossfit_scores.py \
  --fold-root "$ARCHIVE/scores" \
  --folds 4 \
  --expected-pairs 664 \
  --expected-tasks 68 \
  --output-archive "$OOF" \
  >"$ARCHIVE/merge_stdout.json"

"$PY" scripts/49_evaluate_grouped_train_hybrid.py \
  --train-archive "$OOF" \
  --eval-archive /share/guozhix/wmagentattack/0712/grouped_user_task_dreamer \
  --train-steps "$DATA/train_steps.jsonl" \
  --validation-steps "$DATA/val_steps.jsonl" \
  --bootstrap-samples 20000 \
  --bootstrap-seed 20260723 \
  --output-dir "$EVALUATION" \
  >"$EVALUATION/evaluation_stdout.json"

echo "GROUPED_OUTER_CROSSFIT_EVALUATION_DONE $(date -Is)"
