#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-resid-disc
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
RUN_TAG=${RUN_TAG:-residual_preservation_20260716_v1}
ARCHIVE=/share/guozhix/wmagentattack/0716/residual_preservation/$RUN_TAG
DATA=$SOURCE/data/fold1/full
CACHE=/share/guozhix/hf_cache

cd "$ROOT"
mkdir -p "$ARCHIVE/discovery" "$CACHE" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$CACHE"
export TRANSFORMERS_CACHE="$CACHE"
export TOKENIZERS_PARALLELISM=false

for REQUIRED in "$DATA/train_steps.jsonl" "$DATA/val_steps.jsonl"; do
  test -s "$REQUIRED"
done

PYTHONWARNINGS=ignore "$PY" scripts/91_probe_v2_residual_preservation.py \
  --data-root "$DATA" \
  --model-name intfloat/e5-base-v2 \
  --cache-dir "$CACHE" \
  --embedding-batch-size 32 \
  --mode discover \
  --output "$ARCHIVE/discovery/result.json" \
  > "$ARCHIVE/discovery/result_stdout.json"

echo "AGENTDOJO_V2_RESIDUAL_DISCOVERY_DONE tag=$RUN_TAG $(date -Is)"

