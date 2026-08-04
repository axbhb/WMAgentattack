#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-prospect
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
DATA=${DATA:-/share/guozhix/wmagentattack/0714/agentdojo_v2/final_dataset}
TRAIN=${TRAIN:-$DATA/train_steps.jsonl}
MODEL_ROOT=${MODEL_ROOT:-/share/guozhix/wmagentattack/0715/dreamer_v2_final/formal_20260715_softprob_v1/binary_risk}
RUN_TAG=${RUN_TAG:-formal_20260715_full_prospective_v1}
ARCHIVE=${ARCHIVE:-/share/guozhix/wmagentattack/0715/downstream_selection/$RUN_TAG}

cd "$ROOT"
mkdir -p "$ARCHIVE" logs
test -s "$TRAIN"
test -s "$DATA/val_steps.jsonl"
test -s "$DATA/test_steps.jsonl"
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
  --output "$ARCHIVE/result.json" \
  > "$ARCHIVE/stdout.json"

echo "AGENTDOJO_V2_PROSPECTIVE_SELECTION_DONE tag=$RUN_TAG $(date -Is)"
