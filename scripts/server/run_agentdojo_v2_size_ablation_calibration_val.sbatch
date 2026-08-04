#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-sizecal-v
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
FULL_DATA=/share/guozhix/wmagentattack/0714/agentdojo_v2/final_dataset
RUN_TAG=${RUN_TAG:-formal_20260715_nested_v1}
ARCHIVE=/share/guozhix/wmagentattack/0715/data_scale_ablation/$RUN_TAG
SIZES=(pct25 pct25 pct25 pct50 pct50 pct50)
SEEDS=(7 13 21 7 13 21)
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL=$ARCHIVE/models/$SIZE/seed${SEED}/model
TRAIN=$ARCHIVE/datasets/$SIZE/train_steps.jsonl
OUT=$ARCHIVE/calibration/$SIZE/seed${SEED}

cd "$ROOT"
mkdir -p "$OUT" logs
test -s "$MODEL/model.pt"
test -s "$TRAIN"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

PYTHONWARNINGS=ignore "$PY" scripts/67_fit_frozen_risk_calibrator.py \
  --model "$MODEL" \
  --train "$TRAIN" \
  --val "$FULL_DATA/val_steps.jsonl" \
  --output-root "$OUT" \
  > "$OUT/fit_stdout.json"

echo "AGENTDOJO_V2_SIZE_CAL_VAL_DONE tag=$RUN_TAG size=$SIZE seed=$SEED $(date -Is)"
