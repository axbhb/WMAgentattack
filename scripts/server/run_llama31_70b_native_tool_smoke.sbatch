#!/usr/bin/env bash
#SBATCH --job-name=wma-native-tool-smoke
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
MODEL=/share/guozhix/.cache/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
SOURCE=/share/guozhix/wmagentattack/0716/fresh_task_screen/fresh_task_screen_20260716_v1/manifest.json
ARCHIVE=/share/guozhix/wmagentattack/0716/native_tool_protocol/native_tool_protocol_20260716_v1
MANIFEST=$ARCHIVE/smoke_manifest.json

cd "$ROOT"
mkdir -p "$ARCHIVE/smoke/runs" logs
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PY" scripts/104_filter_fresh_clean_manifest.py \
  --input "$SOURCE" --output "$MANIFEST" --suite travel \
  --task-id user_task_15

"$PY" scripts/99_run_selected_clean_agentdojo.py \
  --manifest "$MANIFEST" \
  --model-path "$MODEL" \
  --model-label meta-llama-3.1-70b-native-tool-smoke \
  --benchmark-version v1.2.2 \
  --quantization 4bit \
  --prompt-profile base \
  --protocol native \
  --max-input-tokens 8192 \
  --max-new-tokens 256 \
  --max-tool-output-chars 12000 \
  --agentdojo-local-alias \
  --run-seed 101 \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.95 \
  --logdir "$ARCHIVE/smoke/runs" \
  --output "$ARCHIVE/smoke/result.json"

"$PY" - "$ARCHIVE/smoke/result.json" <<'PY'
import json, sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())["results"][0]
trace = json.loads(Path(result["raw_trace"]).read_text())
tool_calls = sum(
    len(message.get("tool_calls") or [])
    for message in trace["messages"]
    if message.get("role") == "assistant"
)
if tool_calls < 1:
    raise RuntimeError("Native smoke emitted no valid tool call")
print(json.dumps({"utility": result["utility"], "tool_calls": tool_calls}, indent=2))
PY

echo "AGENTDOJO_NATIVE_TOOL_SMOKE_DONE $(date -Is)"
