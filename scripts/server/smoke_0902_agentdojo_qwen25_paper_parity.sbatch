#!/usr/bin/env bash
#SBATCH --job-name=WMA_QwenADSmoke
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/share/guozhix/WMagentattack-qwen-parity-sep2/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-qwen-parity-sep2/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-qwen-parity-sep2
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
OUT=/share/guozhix/wmagentattack/0902/agentdojo_qwen25_paper_parity/smoke_v1
cd "$WT"
mkdir -p "$OUT" logs
export PYTHONPATH="$WT/src:$WT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" scripts/10_run_agentdojo_hf_full.py \
  --model-path "$MODEL" \
  --model-label qwen2.5-7b-paper-parity-smoke \
  --benchmark-version v1 \
  --suite workspace \
  --user-task user_task_0 \
  --attack important_instructions \
  --injection-task injection_task_0 \
  --max-new-tokens 256 \
  --max-tool-output-chars 12000 \
  --prompt-profile base \
  --max-input-tokens 8192 \
  --protocol native \
  --quantization 4bit \
  --device cuda:0 \
  --logdir "$OUT" \
  --force-rerun
