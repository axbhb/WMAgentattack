#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-gu-seedeval
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

: "${TRAIN_JOB_ID:?TRAIN_JOB_ID is required}"
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
SOURCE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1
RUN_TAG=${RUN_TAG:-seedrep_20260715_group_utility_head_only_v1}
ARCHIVE=/share/guozhix/wmagentattack/0715/grouped_continuous_utility/$RUN_TAG
FOLD=${FOLD:-1}
DATA=$SOURCE/data/fold${FOLD}/full
TRAIN=$DATA/train_steps.jsonl
VAL=$DATA/val_steps.jsonl
TEST=$DATA/test_steps.jsonl
VARIANT_ROOT=$ARCHIVE/models/fold${FOLD}/group_utility_head_only
BASELINE_ROOT=$SOURCE/models/fold${FOLD}/pct100

cd "$ROOT"
mkdir -p "$ARCHIVE/baseline/fold${FOLD}" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

for SEED in 7 13 21; do
  test -s "$VARIANT_ROOT/seed${SEED}/model/model.pt"
  test -s "$VARIANT_ROOT/seed${SEED}/training_stdout.json"
  test -s "$VARIANT_ROOT/seed${SEED}/val_metrics.json"
  test -s "$VARIANT_ROOT/seed${SEED}/test_metrics.json"
  test -s "$BASELINE_ROOT/seed${SEED}/model/model.pt"
  BASELINE_OUT=$ARCHIVE/baseline/fold${FOLD}/seed${SEED}
  mkdir -p "$BASELINE_OUT"
  for SPLIT in val test; do
    EVAL_FILE=$VAL
    if [[ "$SPLIT" == "test" ]]; then
      EVAL_FILE=$TEST
    fi
    PYTHONWARNINGS=ignore "$PY" scripts/24_eval_full_dreamer_v3.py \
      --test "$EVAL_FILE" \
      --model "$BASELINE_ROOT/seed${SEED}/model" \
      --output "$BASELINE_OUT/${SPLIT}_metrics.json" \
      --validation-group-step first \
      > "$BASELINE_OUT/${SPLIT}_metrics_stdout.json"
  done
done

PYTHONWARNINGS=ignore "$PY" scripts/70_evaluate_v2_downstream_selection.py \
  --data-root "$DATA" \
  --model-root "$VARIANT_ROOT" \
  --calibration-mode fit_decision_time \
  --calibration-train "$TRAIN" \
  --decision-step first \
  --seeds 7 13 21 \
  --budgets 1 2 4 \
  --bootstrap-draws 10000 \
  --output "$ARCHIVE/variant_downstream.json" \
  > "$ARCHIVE/variant_downstream_stdout.json"

PYTHONWARNINGS=ignore "$PY" scripts/70_evaluate_v2_downstream_selection.py \
  --data-root "$DATA" \
  --model-root "$BASELINE_ROOT" \
  --calibration-mode fit_decision_time \
  --calibration-train "$TRAIN" \
  --decision-step first \
  --seeds 7 13 21 \
  --budgets 1 2 4 \
  --bootstrap-draws 10000 \
  --output "$ARCHIVE/baseline_downstream.json" \
  > "$ARCHIVE/baseline_downstream_stdout.json"

"$PY" scripts/82_summarize_v2_group_utility_seedrep.py \
  --archive-root "$ARCHIVE" \
  --fold "$FOLD" \
  --seeds 7 13 21 \
  --variant group_utility_head_only \
  --output "$ARCHIVE/summary.json" \
  > "$ARCHIVE/summary_stdout.json"

echo "AGENTDOJO_V2_GROUP_UTILITY_HEAD_ONLY_SEEDREP_EVAL_DONE tag=$RUN_TAG train_job=$TRAIN_JOB_ID fold=$FOLD $(date -Is)"
