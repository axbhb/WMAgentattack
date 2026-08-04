#!/usr/bin/env bash
#SBATCH --job-name=wma-fresh-census-70b
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
RUN_TAG=${RUN_TAG:-fresh_task_screen_20260716_v1}
ARCHIVE=/share/guozhix/wmagentattack/0716/fresh_task_screen/$RUN_TAG
MANIFEST=$ARCHIVE/manifest.json
INDEX=$SLURM_ARRAY_TASK_ID
CHUNKS=4
if (( INDEX < CHUNKS )); then
  RUN_SEED=103
  CHUNK=$INDEX
else
  RUN_SEED=107
  CHUNK=$((INDEX - CHUNKS))
fi
OUTPUT=$ARCHIVE/seed${RUN_SEED}/chunk${CHUNK}.json
LOGDIR=$ARCHIVE/seed${RUN_SEED}/runs/chunk${CHUNK}

cd "$ROOT"
mkdir -p "$(dirname "$OUTPUT")" "$LOGDIR" logs
test -s "$MANIFEST"
test -s "$MODEL/config.json"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PY" scripts/99_run_selected_clean_agentdojo.py \
  --manifest "$MANIFEST" \
  --model-path "$MODEL" \
  --model-label "meta-llama-3.1-70b-fresh-clean-seed${RUN_SEED}" \
  --benchmark-version v1.2.2 \
  --quantization 4bit \
  --prompt-profile base \
  --protocol function_tags \
  --max-input-tokens 8192 \
  --max-new-tokens 256 \
  --max-tool-output-chars 12000 \
  --agentdojo-local-alias \
  --run-seed "$RUN_SEED" \
  --chunk-index "$CHUNK" \
  --num-chunks "$CHUNKS" \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.95 \
  --continue-on-error \
  --logdir "$LOGDIR" \
  --output "$OUTPUT"

echo "AGENTDOJO_FRESH_CENSUS_DONE tag=$RUN_TAG seed=$RUN_SEED chunk=$CHUNK $(date -Is)"
