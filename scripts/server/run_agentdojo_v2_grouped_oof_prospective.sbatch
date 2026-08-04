#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-oof-select
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

: "${TRAIN_JOB_ID:?TRAIN_JOB_ID is required for provenance}"
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
RUN_TAG=${RUN_TAG:-formal_20260715_oof_v1}
ARCHIVE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/$RUN_TAG
FOLD=$((SLURM_ARRAY_TASK_ID / 2 + 1))
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
  SIZE=pct25
  TRAIN=$ARCHIVE/data/fold${FOLD}/pct25/train_steps.jsonl
else
  SIZE=pct100
  TRAIN=$ARCHIVE/data/fold${FOLD}/full/train_steps.jsonl
fi
DATA=$ARCHIVE/data/fold${FOLD}/full
MODEL_ROOT=$ARCHIVE/models/fold${FOLD}/${SIZE}
OUT=$ARCHIVE/prospective/fold${FOLD}/${SIZE}

cd "$ROOT"
mkdir -p "$OUT" logs
test -s "$TRAIN"
test -s "$DATA/val_steps.jsonl"
test -s "$DATA/test_steps.jsonl"
for SEED in 7 13 21; do
  test -s "$MODEL_ROOT/seed${SEED}/model/model.pt"
done
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PYTHONWARNINGS=ignore "$PY" scripts/70_evaluate_v2_downstream_selection.py \
  --data-root "$DATA" \
  --model-root "$MODEL_ROOT" \
  --calibration-mode fit_decision_time \
  --calibration-train "$TRAIN" \
  --decision-step first \
  --budgets 1 2 4 \
  --bootstrap-draws 10000 \
  --output "$OUT/result.json" \
  > "$OUT/stdout.json"

echo "AGENTDOJO_V2_OOF_SELECTION_DONE tag=$RUN_TAG train_job=$TRAIN_JOB_ID fold=$FOLD size=$SIZE $(date -Is)"
