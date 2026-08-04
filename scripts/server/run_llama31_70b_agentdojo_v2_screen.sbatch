#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-screen-70b
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
ARCHIVE=/share/guozhix/wmagentattack/0714/agentdojo_v2
MANIFEST=${V2_MANIFEST:-$ARCHIVE/manifests/screen_manifest.json}
RUN_SEED=${V2_RUN_SEED:-7}
CHUNK=${SLURM_ARRAY_TASK_ID}
CHUNKS=8
OUTPUT=$ARCHIVE/screen/results/seed${RUN_SEED}/chunk${CHUNK}.json
LOGDIR=$ARCHIVE/screen/runs/seed${RUN_SEED}/chunk${CHUNK}

cd "$ROOT"
mkdir -p "$(dirname "$OUTPUT")" "$LOGDIR" logs
test -s "$MANIFEST"
test -s "$MODEL/config.json"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PY" scripts/62_run_agentdojo_v2_manifest.py \
  --manifest "$MANIFEST" \
  --model-path "$MODEL" \
  --model-label "meta-llama-3.1-70b-agentdojo-v2-screen-seed${RUN_SEED}" \
  --dataset-version v2.0-screen \
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
  --logdir "$LOGDIR" \
  --output "$OUTPUT"

echo "AGENTDOJO_V2_SCREEN_CHUNK_DONE seed=$RUN_SEED chunk=$CHUNK $(date -Is)"
bash scripts/server/finalize_agentdojo_v2_screen_if_complete.sh "$RUN_SEED" "$CHUNKS"
