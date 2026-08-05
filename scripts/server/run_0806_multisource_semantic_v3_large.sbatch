#!/usr/bin/env bash
#SBATCH --job-name=wma-ms-v3-large
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-20%4
#SBATCH --output=/share/guozhix/WMagentattack-multisource-aug5/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack-multisource-aug5/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-multisource-aug5
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0806/multisource_semantic_v3/large_v1
PROTOCOL=$ARCHIVE/frozen_protocol.json
PILOT_GATE=/share/guozhix/wmagentattack/0805/multisource_semantic_v3/pilot_v1/pilot_gate.json

"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["decision"] == "PILOT_GO__BUILD_FROZEN_LARGE_MULTISOURCE_V1"' "$PILOT_GATE"

if (( SLURM_ARRAY_TASK_ID < 2 )); then
  SOURCE=tool_sandbox
  CHUNK=$SLURM_ARRAY_TASK_ID
  NUM_CHUNKS=2
  SOURCE_ROOT=$ROOT/external/tool_sandbox
elif (( SLURM_ARRAY_TASK_ID < 18 )); then
  SOURCE=injecagent
  CHUNK=$((SLURM_ARRAY_TASK_ID - 2))
  NUM_CHUNKS=16
  SOURCE_ROOT=$ROOT/external/injecagent
else
  SOURCE=tau3
  CHUNK=$((SLURM_ARRAY_TASK_ID - 18))
  NUM_CHUNKS=3
  SOURCE_ROOT=$ROOT/external/tau3
fi

cd "$ROOT"
mkdir -p "$ARCHIVE/outputs/$SOURCE" "$ARCHIVE/audits/$SOURCE" logs
test -s "$PROTOCOL"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:$ROOT/external/tool_sandbox:$ROOT/external/tau3/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PY" scripts/155_run_multisource_llm.py \
  --protocol "$PROTOCOL" \
  --manifest "$ARCHIVE/manifests/${SOURCE}.json" \
  --source-root "$SOURCE_ROOT" \
  --chunk-index "$CHUNK" \
  --num-chunks "$NUM_CHUNKS" \
  --output "$ARCHIVE/outputs/$SOURCE/chunk${CHUNK}.json" \
  --audit "$ARCHIVE/audits/$SOURCE/chunk${CHUNK}.json"

echo "MULTISOURCE_LARGE_DONE source=$SOURCE chunk=$CHUNK/$NUM_CHUNKS $(date -Is)"
