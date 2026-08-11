#!/usr/bin/env bash
#SBATCH --job-name=wma-tau3-tail
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%A_%a.err

set -euo pipefail
ROOT=/share/guozhix/WMagentattack-tau3-interactive-aug5
SOURCE_ROOT=/share/guozhix/WMagentattack-multisource-aug5/external/tau3
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0807/tau3_tail_horizon/pilot_v1
cd "$ROOT"
mkdir -p "$ARCHIVE/outputs" "$ARCHIVE/audits" logs
export PYTHONPATH="$ROOT/src:$SOURCE_ROOT/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" scripts/175_run_tau3_tail_horizon.py \
  --protocol "$ARCHIVE/frozen_protocol.json" --manifest "$ARCHIVE/manifest.json" \
  --source-root "$SOURCE_ROOT" --chunk-index "$SLURM_ARRAY_TASK_ID" --num-chunks 2 \
  --output "$ARCHIVE/outputs/chunk${SLURM_ARRAY_TASK_ID}.json" \
  --audit "$ARCHIVE/audits/chunk${SLURM_ARRAY_TASK_ID}.json"
echo "TAU3_TAIL_DONE chunk=$SLURM_ARRAY_TASK_ID $(date -Is)"
