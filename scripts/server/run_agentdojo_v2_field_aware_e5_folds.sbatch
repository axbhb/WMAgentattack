#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-field-e5
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-4%1
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
SOURCE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1
RUN_TAG=${RUN_TAG:-field_aware_e5_20260716_v1}
ARCHIVE=/share/guozhix/wmagentattack/0716/field_aware_e5/$RUN_TAG
FOLD=$SLURM_ARRAY_TASK_ID
DATA=$SOURCE/data/fold${FOLD}/full
CACHE=/share/guozhix/hf_cache
OUT=$ARCHIVE/frozen/fold${FOLD}

cd "$ROOT"
mkdir -p "$OUT" "$CACHE" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$CACHE"
export TRANSFORMERS_CACHE="$CACHE"
export TOKENIZERS_PARALLELISM=false

PYTHONWARNINGS=ignore "$PY" scripts/97_probe_v2_field_aware_e5.py \
  --data-root "$DATA" \
  --model-name intfloat/e5-base-v2 \
  --cache-dir "$CACHE" \
  --embedding-batch-size 32 \
  --output "$OUT/result.json" \
  > "$OUT/result_stdout.json"

echo "AGENTDOJO_V2_FIELD_AWARE_E5_DONE tag=$RUN_TAG fold=$FOLD $(date -Is)"
