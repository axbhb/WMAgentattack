#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-select
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
DATA=${DATA:-/share/guozhix/wmagentattack/0714/agentdojo_v2/final_dataset}
MODEL_ROOT=${MODEL_ROOT:-/share/guozhix/wmagentattack/0715/dreamer_v2_final/formal_20260715_softprob_v1/binary_risk}
CALIBRATION_ROOT=${CALIBRATION_ROOT:-/share/guozhix/wmagentattack/0715/frozen_risk_calibration/formal_20260715_frozen_cal_v1}
CALIBRATION_SELECTION=${CALIBRATION_SELECTION:-$CALIBRATION_ROOT/selection.json}
RUN_TAG=${RUN_TAG:-formal_20260715_full_v1}
ARCHIVE=${ARCHIVE:-/share/guozhix/wmagentattack/0715/downstream_selection/$RUN_TAG}

cd "$ROOT"
mkdir -p "$ARCHIVE" logs
test -s "$DATA/val_steps.jsonl"
test -s "$DATA/test_steps.jsonl"
test -s "$CALIBRATION_SELECTION"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PYTHONWARNINGS=ignore "$PY" scripts/70_evaluate_v2_downstream_selection.py \
  --data-root "$DATA" \
  --model-root "$MODEL_ROOT" \
  --calibration-root "$CALIBRATION_ROOT" \
  --calibration-selection "$CALIBRATION_SELECTION" \
  --budgets 1 2 4 \
  --bootstrap-draws 10000 \
  --output "$ARCHIVE/result.json" \
  > "$ARCHIVE/stdout.json"

echo "AGENTDOJO_V2_DOWNSTREAM_SELECTION_DONE tag=$RUN_TAG $(date -Is)"
