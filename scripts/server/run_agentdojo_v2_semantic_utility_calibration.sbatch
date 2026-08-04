#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-sem-util-cal
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
SOURCE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1
SEMANTIC=/share/guozhix/wmagentattack/0715/semantic_dreamer/semantic_dreamer_pilot30_20260716_v2
DATA=$SOURCE/data/fold1/full
MODEL=$SEMANTIC/models/fold1/semantic_head_only/seed7/model

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

PYTHONWARNINGS=ignore "$PY" scripts/90_evaluate_v2_utility_calibration.py \
  --data-root "$DATA" \
  --model "$MODEL" \
  --decision-step first \
  --budgets 1 2 4 \
  --output "$SEMANTIC/utility_calibration.json" \
  > "$SEMANTIC/utility_calibration_stdout.json"

echo "AGENTDOJO_V2_SEMANTIC_UTILITY_CALIBRATION_DONE $(date -Is)"
