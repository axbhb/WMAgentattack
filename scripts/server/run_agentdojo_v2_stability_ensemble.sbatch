#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-stability-ensemble
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
SOURCE_RUN_TAG=${SOURCE_RUN_TAG:-formal_20260715_oof_v1}
RUN_TAG=${RUN_TAG:-formal_20260715_stability_v1}
SOURCE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/$SOURCE_RUN_TAG
ARCHIVE=/share/guozhix/wmagentattack/0715/stability_ensemble/$RUN_TAG

FOLD=$((SLURM_ARRAY_TASK_ID / 2))
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
  SIZE=pct25
  TRAIN=$SOURCE/data/fold${FOLD}/pct25/train_steps.jsonl
else
  SIZE=pct100
  TRAIN=$SOURCE/data/fold${FOLD}/full/train_steps.jsonl
fi
DATA=$SOURCE/data/fold${FOLD}/full
MODEL_ROOT=$SOURCE/models/fold${FOLD}/${SIZE}
OUT=$ARCHIVE/fold${FOLD}/${SIZE}

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

PYTHONWARNINGS=ignore "$PY" scripts/77_evaluate_v2_stability_ensemble.py \
  --data-root "$DATA" \
  --model-root "$MODEL_ROOT" \
  --calibration-train "$TRAIN" \
  --budgets 1 2 4 \
  --bootstrap-draws 10000 \
  --output "$OUT/result.json" \
  > "$OUT/stdout.json"

echo "AGENTDOJO_V2_STABILITY_ENSEMBLE_DONE tag=$RUN_TAG fold=$FOLD size=$SIZE $(date -Is)"
