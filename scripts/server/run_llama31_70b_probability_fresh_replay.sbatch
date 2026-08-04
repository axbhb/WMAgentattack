#!/usr/bin/env bash
#SBATCH --job-name=wma-prob-fresh-70b
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
ARCHIVE=/share/guozhix/wmagentattack/0710
CALIBRATION="$ARCHIVE/replay_probability_calibration"
FRESH="$ARCHIVE/replay_probability_fresh_multiseed"
SELECTION="$CALIBRATION/fresh_replay_selections.json"
SEEDS=(31 37 43)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
OUTPUT="$FRESH/seed${SEED}/replay.json"
LOGDIR="$ROOT/runs/agentdojo_probability_fresh_replay_llama31_70b/seed${SEED}"

cd "$ROOT"
mkdir -p "$FRESH/seed${SEED}" "$LOGDIR" logs
test -s "$SELECTION"
test -s "$MODEL/config.json"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$PY" scripts/12_run_agentdojo_hf_selected.py \
  --selection "$SELECTION" \
  --selection-name replay_probability_world_model \
  --selection-name repeated_cv_world_model \
  --selection-name sklearn_baseline \
  --model-path "$MODEL" \
  --model-label "meta-llama-3.1-70b-prob-fresh-seed${SEED}" \
  --benchmark-version v1.2.2 \
  --quantization 4bit \
  --prompt-profile base \
  --protocol function_tags \
  --max-input-tokens 8192 \
  --max-new-tokens 256 \
  --max-tool-output-chars 12000 \
  --agentdojo-local-alias \
  --seed "$SEED" \
  --do-sample \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --logdir "$LOGDIR" \
  --output "$OUTPUT" \
  --force-rerun

echo "PROBABILITY_FRESH_REPLAY_70B_DONE seed=$SEED $(date -Is)"
