#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-toprank
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --array=0-4%5
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

: "${PAIR_SCHEME:?PAIR_SCHEME is required}"

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
SOURCE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1
RUN_TAG=${RUN_TAG:-top_heavy_ranking_20260716_v1}
ARCHIVE=/share/guozhix/wmagentattack/0716/top_heavy_ranking/$RUN_TAG
FOLD=$SLURM_ARRAY_TASK_ID
DATA=$SOURCE/data/fold${FOLD}/full
CACHE=/share/guozhix/hf_cache
OUT=$ARCHIVE/frozen/$PAIR_SCHEME/fold${FOLD}

cd "$ROOT"
mkdir -p "$OUT" "$CACHE" logs
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

PYTHONWARNINGS=ignore "$PY" scripts/94_probe_v2_top_heavy_ranking.py \
  --data-root "$DATA" \
  --model-name intfloat/e5-base-v2 \
  --cache-dir "$CACHE" \
  --embedding-batch-size 32 \
  --pair-scheme "$PAIR_SCHEME" \
  --output "$OUT/result.json" \
  > "$OUT/result_stdout.json"

echo "AGENTDOJO_V2_TOP_HEAVY_DONE tag=$RUN_TAG scheme=$PAIR_SCHEME fold=$FOLD $(date -Is)"
