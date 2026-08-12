#!/usr/bin/env bash
#SBATCH --job-name=wma-fresh-v3
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/0813_fresh_v3_%A_%a.out
#SBATCH --error=logs/0813_fresh_v3_%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-fresh-validation-aug13
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
ARCHIVE=/share/guozhix/wmagentattack/0813/fresh_integrated_validation/formal_v1
MANIFEST=$ARCHIVE/frozen_inputs/manifest.json
SEEDS=(401 409 419)
RUN_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUTPUT=$ARCHIVE/clean_runs/seed${RUN_SEED}/result.json
LOGDIR=$ARCHIVE/clean_runs/seed${RUN_SEED}/traces

cd "$ROOT"
mkdir -p "$(dirname "$OUTPUT")" "$LOGDIR" logs
test -s "$MODEL/config.json"
(cd "$ARCHIVE/frozen_inputs" && sha256sum --quiet -c SHA256SUMS)
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PY" scripts/99_run_selected_clean_agentdojo.py \
  --manifest "$MANIFEST" \
  --custom-task-module wmagentattack.custom_agentdojo_confirmation_v3 \
  --model-path "$MODEL" \
  --model-label "meta-llama-3.1-70b-fresh-v3-seed${RUN_SEED}" \
  --benchmark-version v1.2.2 \
  --quantization 4bit \
  --prompt-profile constraint_checklist \
  --protocol function_tags_repair_retry \
  --max-input-tokens 8192 \
  --max-new-tokens 256 \
  --max-tool-output-chars 12000 \
  --agentdojo-local-alias \
  --run-seed "$RUN_SEED" \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.95 \
  --continue-on-error \
  --logdir "$LOGDIR" \
  --output "$OUTPUT"

echo "FRESH_CUSTOM_CONFIRMATION_V3_DONE seed=$RUN_SEED $(date -Is)"
