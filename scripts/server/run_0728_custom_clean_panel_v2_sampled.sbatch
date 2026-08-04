#!/usr/bin/env bash
#SBATCH --job-name=wma-panel-v2-s
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%3
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
ARCHIVE=/share/guozhix/wmagentattack/0728/custom_clean_panel_v2/fixed_v1
FROZEN=$ARCHIVE/frozen_inputs
MANIFEST=$FROZEN/stochastic_manifest.json
SEEDS=(307 311 313 317 331 337)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
CHUNK=$((SLURM_ARRAY_TASK_ID % 2))
RUN_SEED=${SEEDS[$SEED_INDEX]}
OUTPUT=$ARCHIVE/sampled/seed${RUN_SEED}/chunk${CHUNK}.json
LOGDIR=$ARCHIVE/sampled/seed${RUN_SEED}/runs/chunk${CHUNK}

cd "$ROOT"
mkdir -p "$(dirname "$OUTPUT")" "$LOGDIR" logs
test -s "$MODEL/config.json"
test -s "$FROZEN/SHA256SUMS"
test -s "$FROZEN/frozen_repo_SHA256SUMS"
(
  cd "$FROZEN"
  sha256sum --quiet -c SHA256SUMS
)
sha256sum --quiet -c "$FROZEN/frozen_repo_SHA256SUMS"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PY" scripts/99_run_selected_clean_agentdojo.py \
  --manifest "$MANIFEST" \
  --custom-task-module wmagentattack.custom_agentdojo_panel_v2 \
  --model-path "$MODEL" \
  --model-label "meta-llama-3.1-70b-panel-v2-sampled-seed${RUN_SEED}" \
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
  --chunk-index "$CHUNK" \
  --num-chunks 2 \
  --continue-on-error \
  --logdir "$LOGDIR" \
  --output "$OUTPUT"

echo "CUSTOM_CLEAN_PANEL_V2_SAMPLED_DONE seed=$RUN_SEED chunk=$CHUNK $(date -Is)"
