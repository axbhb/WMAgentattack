#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-groupcal-test
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

: "${RUN_TAG:?RUN_TAG is required}"
: "${SELECTED_VARIANT:?SELECTED_VARIANT is required}"
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
DATA=/share/guozhix/wmagentattack/0714/agentdojo_v2/final_dataset
ARCHIVE=/share/guozhix/wmagentattack/0715/dreamer_v2_group_calibration/$RUN_TAG
VARIANTS=(calib0 "$SELECTED_VARIANT" calib0 "$SELECTED_VARIANT" calib0 "$SELECTED_VARIANT")
SEEDS=(7 7 13 13 21 21)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=$ARCHIVE/$VARIANT/seed${SEED}

if [[ "$SELECTED_VARIANT" == "calib0" && $((SLURM_ARRAY_TASK_ID % 2)) -eq 1 ]]; then
  echo "SKIP_DUPLICATE_CONTROL seed=$SEED"
  exit 0
fi

cd "$ROOT"
test -s "$OUT/model/metadata.json"
test -s "$DATA/test_steps.jsonl"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
PYTHONWARNINGS=ignore "$PY" scripts/24_eval_full_dreamer_v3.py \
  --test "$DATA/test_steps.jsonl" \
  --model "$OUT/model" \
  --output "$OUT/test_metrics.json" \
  > "$OUT/test_metrics_stdout.json"

echo "AGENTDOJO_V2_GROUP_CAL_TEST_DONE tag=$RUN_TAG variant=$VARIANT seed=$SEED $(date -Is)"
