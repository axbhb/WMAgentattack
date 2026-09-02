#!/usr/bin/env bash
#SBATCH --job-name=WMA_QwenADCleanGate
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/share/guozhix/WMagentattack-qwen-parity-sep2/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-qwen-parity-sep2/logs/%x-%j.err

set -euo pipefail
WT=/share/guozhix/WMagentattack-qwen-parity-sep2
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0902/agentdojo_qwen25_paper_parity/formal_v1
cd "$WT"
export PYTHONPATH="$WT/src:$WT/external/agentdojo/src:${PYTHONPATH:-}"
"$PY" scripts/302_gate_agentdojo_qwen_clean.py \
  --run-root "$ARCHIVE/runs" \
  --model-label qwen2.5-7b-paper-parity-greedy \
  --output "$ARCHIVE/clean_gate.json"
