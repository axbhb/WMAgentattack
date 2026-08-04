#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-dual-fold
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
SOURCE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1
RUN_TAG=${RUN_TAG:-dual_component_pointwise_frozen_oof_20260716_v1}
ARCHIVE=/share/guozhix/wmagentattack/0715/dual_component_value/$RUN_TAG
FOLD=$SLURM_ARRAY_TASK_ID
DATA=$SOURCE/data/fold${FOLD}/full
CACHE=/share/guozhix/hf_cache
OUT=$ARCHIVE/fold${FOLD}

cd "$ROOT"
mkdir -p "$OUT" "$CACHE" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$CACHE"
export TRANSFORMERS_CACHE="$CACHE"
export TOKENIZERS_PARALLELISM=false

PYTHONWARNINGS=ignore "$PY" scripts/87_probe_v2_dual_component_value.py \
  --data-root "$DATA" \
  --model-name intfloat/e5-base-v2 \
  --cache-dir "$CACHE" \
  --embedding-batch-size 32 \
  --utility-estimator pointwise_ridge \
  --utility-alpha 10.0 \
  --frozen-recipe attack_plus_2utility \
  --output "$OUT/result.json" \
  > "$OUT/result_stdout.json"

echo "AGENTDOJO_V2_DUAL_COMPONENT_FROZEN_FOLD_DONE tag=$RUN_TAG fold=$FOLD $(date -Is)"
