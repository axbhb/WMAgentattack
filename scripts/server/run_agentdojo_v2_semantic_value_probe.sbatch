#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-semantic-probe
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
SOURCE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1
RUN_TAG=${RUN_TAG:-e5_base_v2_probe_20260715_v1}
ARCHIVE=/share/guozhix/wmagentattack/0715/semantic_value_probe/$RUN_TAG
FOLD=${FOLD:-1}
DATA=$SOURCE/data/fold${FOLD}/full
CACHE=/share/guozhix/hf_cache

cd "$ROOT"
mkdir -p "$ARCHIVE" "$CACHE" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$CACHE"
export TRANSFORMERS_CACHE="$CACHE"
export TOKENIZERS_PARALLELISM=false

for REQUIRED in \
  "$DATA/train_steps.jsonl" \
  "$DATA/val_steps.jsonl" \
  "$DATA/test_steps.jsonl"; do
  test -s "$REQUIRED"
done

PYTHONWARNINGS=ignore "$PY" scripts/85_probe_v2_semantic_configuration_value.py \
  --data-root "$DATA" \
  --model-name intfloat/e5-base-v2 \
  --cache-dir "$CACHE" \
  --embedding-batch-size 32 \
  --output "$ARCHIVE/result.json" \
  > "$ARCHIVE/result_stdout.json"

echo "AGENTDOJO_V2_SEMANTIC_VALUE_PROBE_DONE tag=$RUN_TAG fold=$FOLD $(date -Is)"
