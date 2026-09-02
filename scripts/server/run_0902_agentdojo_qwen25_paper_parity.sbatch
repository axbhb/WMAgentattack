#!/usr/bin/env bash
#SBATCH --job-name=WMA_QwenADParity
#SBATCH --partition=6000ada
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/share/guozhix/WMagentattack-qwen-parity-sep2/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack-qwen-parity-sep2/logs/%x-%A_%a.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-qwen-parity-sep2
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
ARCHIVE=/share/guozhix/wmagentattack/0902/agentdojo_qwen25_paper_parity/formal_v1
MODEL_LABEL=qwen2.5-7b-paper-parity-greedy
SUITES=(workspace travel banking slack)
SUITE=${SUITES[$SLURM_ARRAY_TASK_ID]}
MODE=${MODE:-clean}

cd "$WT"
mkdir -p logs "$ARCHIVE/runs/$SUITE"
export PYTHONPATH="$WT/src:$WT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

args=(
  --model-path "$MODEL"
  --model-label "$MODEL_LABEL"
  --benchmark-version v1
  --suite "$SUITE"
  --max-new-tokens 256
  --max-tool-output-chars 12000
  --prompt-profile base
  --max-input-tokens 8192
  --protocol native
  --quantization 4bit
  --device cuda:0
  --logdir "$ARCHIVE/runs/$SUITE"
)
if [[ "$MODE" == clean ]]; then
  args+=(--clean-only --force-rerun)
elif [[ "$MODE" == attack ]]; then
  args+=(--attack important_instructions)
else
  echo "unsupported MODE=$MODE" >&2
  exit 2
fi

"$PY" scripts/10_run_agentdojo_hf_full.py "${args[@]}"
printf 'DONE mode=%s suite=%s job=%s task=%s\n' "$MODE" "$SUITE" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
