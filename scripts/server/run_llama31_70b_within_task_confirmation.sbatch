#!/usr/bin/env bash
#SBATCH --job-name=wma-confirm-70b
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
ARCHIVE=/share/guozhix/wmagentattack/0712/within_task_contrast
SELECTION="$ARCHIVE/within_task_confirmation_selection.json"
GATE="$ARCHIVE/hierarchical_models/confirmation_gate_decision.json"
CHUNK="$SLURM_ARRAY_TASK_ID"
BASE_SEED="${BASE_SEED_OVERRIDE:?submit with --export=ALL,BASE_SEED_OVERRIDE=<seed>}"
ACTUAL_SEED=$((BASE_SEED + 1000 * CHUNK))
SELECTION_NAME="within_task_confirmation_chunk${CHUNK}"
OUTPUT="$ARCHIVE/confirmation_replays/base_seed${BASE_SEED}/chunk${CHUNK}/replay.json"
LOGDIR="$ROOT/runs/agentdojo_within_task_confirmation_llama31_70b/base_seed${BASE_SEED}/chunk${CHUNK}"

cd "$ROOT"
mkdir -p "$(dirname "$OUTPUT")" "$LOGDIR" logs
test -s "$SELECTION"
test -s "$GATE"
test -s "$MODEL/config.json"
"$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["go"] is True else 2)' "$GATE"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$PY" scripts/12_run_agentdojo_hf_selected.py \
  --selection "$SELECTION" \
  --selection-name "$SELECTION_NAME" \
  --model-path "$MODEL" \
  --model-label "meta-llama-3.1-70b-confirm-b${BASE_SEED}-c${CHUNK}" \
  --benchmark-version v1.2.2 \
  --quantization 4bit \
  --prompt-profile base \
  --protocol function_tags \
  --max-input-tokens 8192 \
  --max-new-tokens 256 \
  --max-tool-output-chars 12000 \
  --agentdojo-local-alias \
  --seed "$ACTUAL_SEED" \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.95 \
  --logdir "$LOGDIR" \
  --output "$OUTPUT" \
  --force-rerun

echo "WITHIN_TASK_CONFIRMATION_DONE base_seed=$BASE_SEED actual_seed=$ACTUAL_SEED chunk=$CHUNK $(date -Is)"
bash scripts/server/finalize_within_task_confirmation_if_complete.sh
