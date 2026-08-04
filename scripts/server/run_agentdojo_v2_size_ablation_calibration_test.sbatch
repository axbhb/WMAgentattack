#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-sizecal-t
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
CAL_ROOT=$ARCHIVE/calibration/$SIZE
SELECTION=$CAL_ROOT/selection.json
SELECTED_CANDIDATE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate_by_validation"])' "$SELECTION")
OUT=$CAL_ROOT/seed${SEED}/$SELECTED_CANDIDATE
CALIBRATOR_ARGS=()
if [[ "$SELECTED_CANDIDATE" != "identity" ]]; then
  CALIBRATOR_ARGS=(--calibrator "$OUT/calibrator.json")
fi

cd "$ROOT"
mkdir -p "$OUT" logs
test -s "$MODEL/model.pt"
test -s "$SELECTION"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

PYTHONWARNINGS=ignore "$PY" scripts/69_eval_frozen_risk_calibrator.py \
  --model "$MODEL" \
  --test "$FULL_DATA/test_steps.jsonl" \
  "${CALIBRATOR_ARGS[@]}" \
  --output "$OUT/test_result.json" \
  > "$OUT/test_stdout.json"

echo "AGENTDOJO_V2_SIZE_CAL_TEST_DONE tag=$RUN_TAG size=$SIZE seed=$SEED candidate=$SELECTED_CANDIDATE $(date -Is)"
