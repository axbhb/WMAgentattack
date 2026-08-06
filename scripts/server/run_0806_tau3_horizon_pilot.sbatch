#!/usr/bin/env bash
#SBATCH --job-name=wma-tau3-horizon
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-tau3-interactive-aug5
SOURCE_ROOT=/share/guozhix/WMagentattack-multisource-aug5/external/tau3
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0806/tau3_horizon_extension/pilot_v1
PROTOCOL=$ARCHIVE/frozen_protocol.json
MANIFEST=$ARCHIVE/manifest.json

cd "$ROOT"
mkdir -p "$ARCHIVE/outputs" "$ARCHIVE/audits" logs
test -s "$PROTOCOL"
test -s "$MANIFEST"
export PYTHONPATH="$ROOT/src:$SOURCE_ROOT/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PY" scripts/168_run_tau3_interactive_llm.py \
  --protocol "$PROTOCOL" \
  --manifest "$MANIFEST" \
  --source-root "$SOURCE_ROOT" \
  --chunk-index "$SLURM_ARRAY_TASK_ID" \
  --num-chunks 2 \
  --output "$ARCHIVE/outputs/chunk${SLURM_ARRAY_TASK_ID}.json" \
  --audit "$ARCHIVE/audits/chunk${SLURM_ARRAY_TASK_ID}.json"

echo "TAU3_HORIZON_PILOT_DONE chunk=$SLURM_ARRAY_TASK_ID $(date -Is)"
