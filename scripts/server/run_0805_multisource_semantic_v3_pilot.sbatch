#!/usr/bin/env bash
#SBATCH --job-name=wma-ms-v3-pilot
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/share/guozhix/WMagentattack-multisource-aug5/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack-multisource-aug5/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-multisource-aug5
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
PROTOCOL=$ROOT/configs/0805_multisource_semantic_v3_protocol.json
ARCHIVE=/share/guozhix/wmagentattack/0805/multisource_semantic_v3/pilot_v1
SOURCES=(tool_sandbox injecagent tau3)
SOURCE=${SOURCES[$SLURM_ARRAY_TASK_ID]}

case "$SOURCE" in
  tool_sandbox) SOURCE_ROOT=$ROOT/external/tool_sandbox ;;
  injecagent) SOURCE_ROOT=$ROOT/external/injecagent ;;
  tau3) SOURCE_ROOT=$ROOT/external/tau3 ;;
esac

cd "$ROOT"
mkdir -p "$ARCHIVE/outputs/$SOURCE" "$ARCHIVE/audits/$SOURCE" logs
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:$ROOT/external/tool_sandbox:$ROOT/external/tau3/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PY" scripts/155_run_multisource_llm.py \
  --protocol "$PROTOCOL" \
  --manifest "$ARCHIVE/manifests/${SOURCE}.json" \
  --source-root "$SOURCE_ROOT" \
  --chunk-index 0 \
  --num-chunks 1 \
  --output "$ARCHIVE/outputs/$SOURCE/chunk0.json" \
  --audit "$ARCHIVE/audits/$SOURCE/chunk0.json"

echo "MULTISOURCE_PILOT_DONE source=$SOURCE $(date -Is)"
