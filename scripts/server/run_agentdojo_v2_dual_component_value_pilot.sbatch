#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-dual-value
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
FOLD=${FOLD:-1}
RUN_TAG=${RUN_TAG:-dual_component_discovery_20260716_v1}
ARCHIVE=/share/guozhix/wmagentattack/0715/dual_component_value/$RUN_TAG
DATA=$SOURCE/data/fold${FOLD}/full
CACHE=/share/guozhix/hf_cache
UTILITY_ESTIMATOR=${UTILITY_ESTIMATOR:-pairwise_ridge}
UTILITY_ALPHA=${UTILITY_ALPHA:-10.0}
UTILITY_SHRINKAGE=${UTILITY_SHRINKAGE:-4.0}

cd "$ROOT"
mkdir -p "$ARCHIVE" "$CACHE" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$CACHE"
export TRANSFORMERS_CACHE="$CACHE"
export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=()
if [[ -n "${FROZEN_RECIPE:-}" ]]; then
  EXTRA_ARGS+=(--frozen-recipe "$FROZEN_RECIPE")
fi
if [[ "${DOMAIN_INTERACTIONS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--domain-interactions)
fi

PYTHONWARNINGS=ignore "$PY" scripts/87_probe_v2_dual_component_value.py \
  --data-root "$DATA" \
  --model-name intfloat/e5-base-v2 \
  --cache-dir "$CACHE" \
  --embedding-batch-size 32 \
  --utility-estimator "$UTILITY_ESTIMATOR" \
  --utility-alpha "$UTILITY_ALPHA" \
  --utility-shrinkage "$UTILITY_SHRINKAGE" \
  --output "$ARCHIVE/fold${FOLD}_result.json" \
  "${EXTRA_ARGS[@]}" \
  > "$ARCHIVE/fold${FOLD}_stdout.json"

echo "AGENTDOJO_V2_DUAL_COMPONENT_VALUE_DONE tag=$RUN_TAG fold=$FOLD $(date -Is)"
