#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-frcal-test
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

: "${RUN_TAG:?RUN_TAG is required}"
: "${SELECTED_CANDIDATE:?SELECTED_CANDIDATE is required}"
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
DATA=/share/guozhix/wmagentattack/0714/agentdojo_v2/final_dataset
BASE=/share/guozhix/wmagentattack/0715/dreamer_v2_final/formal_20260715_softprob_v1/binary_risk
ARCHIVE=/share/guozhix/wmagentattack/0715/frozen_risk_calibration/$RUN_TAG
SEEDS=(7 13 21)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL=$BASE/seed${SEED}/model
OUT=$ARCHIVE/seed${SEED}/$SELECTED_CANDIDATE
CALIBRATOR_ARGS=()
if [[ "$SELECTED_CANDIDATE" != "identity" ]]; then
  CALIBRATOR_ARGS=(--calibrator "$OUT/calibrator.json")
fi

cd "$ROOT"
mkdir -p "$OUT" logs
test -s "$MODEL/model.pt"
test -s "$DATA/test_steps.jsonl"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
PYTHONWARNINGS=ignore "$PY" scripts/69_eval_frozen_risk_calibrator.py \
  --model "$MODEL" \
  --test "$DATA/test_steps.jsonl" \
  "${CALIBRATOR_ARGS[@]}" \
  --output "$OUT/test_result.json" \
  > "$OUT/test_stdout.json"

echo "FROZEN_RISK_CAL_TEST_DONE tag=$RUN_TAG candidate=$SELECTED_CANDIDATE seed=$SEED $(date -Is)"
