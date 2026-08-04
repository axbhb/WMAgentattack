#!/usr/bin/env bash
#SBATCH --job-name=wma-parser-confirm
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
ARCHIVE=/share/guozhix/wmagentattack/0716/function_tag_parser_retry_confirmation/function_tag_parser_retry_confirmation_20260716_v1
MANIFEST=$ARCHIVE/travel_manifest.json
SEEDS=(109 113 127)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
CHUNK=$((SLURM_ARRAY_TASK_ID % 2))
RUN_SEED=${SEEDS[$SEED_INDEX]}

cd "$ROOT"
printf '%s  %s\n' \
  4fde15eca5dd21c02a0cd31df6cd43bff9ca27abcfeff6233a0c40f3c80df22a src/wmagentattack/qwen_agentdojo.py \
  167de20e2983c402c6b6ba5fe7375ab36b479e16918a41339ef8af9bd2fa4e73 scripts/99_run_selected_clean_agentdojo.py \
  d8bd1f9591c5b96d95ca1407ef5eca8579afdc1b8f31faa97d831bebeb32e8e3 "$MANIFEST" \
  | sha256sum -c -
mkdir -p "$ARCHIVE/seed${RUN_SEED}/runs/chunk${CHUNK}" logs
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PY" scripts/99_run_selected_clean_agentdojo.py \
  --manifest "$MANIFEST" \
  --model-path "$MODEL" \
  --model-label "meta-llama-3.1-70b-parser-retry-confirm-seed${RUN_SEED}" \
  --benchmark-version v1.2.2 \
  --quantization 4bit \
  --prompt-profile base \
  --protocol function_tags_repair_retry \
  --max-input-tokens 8192 \
  --max-new-tokens 256 \
  --max-tool-output-chars 12000 \
  --agentdojo-local-alias \
  --run-seed "$RUN_SEED" \
  --chunk-index "$CHUNK" \
  --num-chunks 2 \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.95 \
  --continue-on-error \
  --logdir "$ARCHIVE/seed${RUN_SEED}/runs/chunk${CHUNK}" \
  --output "$ARCHIVE/seed${RUN_SEED}/chunk${CHUNK}.json"

echo "AGENTDOJO_PARSER_CONFIRM_DONE seed=$RUN_SEED chunk=$CHUNK $(date -Is)"
