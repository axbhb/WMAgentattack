#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-frcal-val
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
DATA=/share/guozhix/wmagentattack/0714/agentdojo_v2/final_dataset
BASE=/share/guozhix/wmagentattack/0715/dreamer_v2_final/formal_20260715_softprob_v1/binary_risk
RUN_TAG=${RUN_TAG:-formal}
ARCHIVE=/share/guozhix/wmagentattack/0715/frozen_risk_calibration/$RUN_TAG
SEEDS=(7 13 21)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL=$BASE/seed${SEED}/model
OUT=$ARCHIVE/seed${SEED}

cd "$ROOT"
mkdir -p "$OUT" logs
test -s "$MODEL/model.pt"
test -s "$MODEL/metadata.json"
(
  cd "$DATA"
  grep -E '  (train|val)_steps\.jsonl$' checksums.sha256 | sha256sum -c -
)
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
PYTHONWARNINGS=ignore "$PY" scripts/67_fit_frozen_risk_calibrator.py \
  --model "$MODEL" \
  --train "$DATA/train_steps.jsonl" \
  --val "$DATA/val_steps.jsonl" \
  --output-root "$OUT" \
  > "$OUT/fit_stdout.json"

echo "FROZEN_RISK_CAL_VAL_DONE tag=$RUN_TAG seed=$SEED $(date -Is)"
