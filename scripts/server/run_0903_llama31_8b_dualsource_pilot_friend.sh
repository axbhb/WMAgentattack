#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: $0 batch0|batch1|gate|launch}
WT=/home/pth/projects/WMagentattack-llama31-8b-dualsource-sep3
PY=/home/pth/venvs/wmagentattack_qwen25/bin/python
MODEL=/home/pth/models/Meta-Llama-3.1-8B-Instruct
ARCHIVE=/home/pth/outputs/wmagentattack/0903/llama31_8b_dualsource_pilot/formal_v1
PROTOCOL=$WT/configs/0903_llama31_8b_dualsource_pilot_protocol.json
SELECTION=$WT/data/0903_llama31_8b_dualsource_pilot/agentdojo_selection.json
INJEC_MANIFEST=$WT/data/0903_llama31_8b_dualsource_pilot/injecagent_manifest.json

cd "$WT"
mkdir -p "$ARCHIVE"/{runs,logs,status,normalized}
export PYTHONPATH="$WT/src:$WT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

run_agentdojo_seed() {
  local seed=$1
  local device=$2
  local logdir="$ARCHIVE/runs/seed${seed}"
  local label="llama31-8b-dualsource-pilot-seed${seed}"
  test -s "$MODEL/config.json"
  CUDA_VISIBLE_DEVICES="$device" "$PY" scripts/10_run_agentdojo_hf_full.py \
    --model-path "$MODEL" \
    --model-label "$label" \
    --benchmark-version v1 \
    --selection-manifest "$SELECTION" \
    --attack direct \
    --attack ignore_previous \
    --attack system_message \
    --attack injecagent \
    --attack important_instructions \
    --attack important_instructions_no_user_name \
    --attack important_instructions_no_model_name \
    --attack tool_knowledge \
    --max-new-tokens 256 \
    --max-tool-output-chars 12000 \
    --prompt-profile base \
    --max-input-tokens 8192 \
    --protocol native \
    --agentdojo-local-alias \
    --quantization 4bit \
    --device cuda:0 \
    --seed "$seed" \
    --do-sample \
    --temperature 0.2 \
    --top-p 0.9 \
    --logdir "$logdir" \
    --force-rerun
  touch "$ARCHIVE/status/agentdojo_seed${seed}.complete"
}

case "$MODE" in
  batch0)
    run_agentdojo_seed 1103 0
    run_agentdojo_seed 1117 0
    touch "$ARCHIVE/status/batch0.complete"
    ;;
  batch1)
    run_agentdojo_seed 1109 1
    CUDA_VISIBLE_DEVICES=1 "$PY" scripts/304_run_llama31_8b_injecagent_pilot.py \
      --protocol "$PROTOCOL" \
      --manifest "$INJEC_MANIFEST" \
      --output "$ARCHIVE/injecagent_output.json" \
      --device cuda:0
    touch "$ARCHIVE/status/injecagent.complete" "$ARCHIVE/status/batch1.complete"
    ;;
  gate)
    while [[ ! -f "$ARCHIVE/status/batch0.complete" || ! -f "$ARCHIVE/status/batch1.complete" ]]; do
      sleep 60
    done
    "$PY" scripts/305_gate_llama31_8b_dualsource_pilot.py \
      --protocol "$PROTOCOL" \
      --agentdojo-root "$ARCHIVE/runs" \
      --injecagent-output "$ARCHIVE/injecagent_output.json" \
      --output-dir "$ARCHIVE/normalized" \
      >"$ARCHIVE/logs/gate.out" 2>"$ARCHIVE/logs/gate.err"
    touch "$ARCHIVE/status/gate.complete"
    ;;
  launch)
    test -s "$MODEL/config.json"
    if [[ -e "$ARCHIVE/status/launched" ]]; then
      echo "Pilot already launched; refusing duplicate."
      exit 3
    fi
    touch "$ARCHIVE/status/launched"
    nohup bash "$0" batch0 >"$ARCHIVE/logs/batch0.out" 2>"$ARCHIVE/logs/batch0.err" &
    echo $! >"$ARCHIVE/status/batch0.pid"
    nohup bash "$0" batch1 >"$ARCHIVE/logs/batch1.out" 2>"$ARCHIVE/logs/batch1.err" &
    echo $! >"$ARCHIVE/status/batch1.pid"
    nohup bash "$0" gate >"$ARCHIVE/logs/gate_watch.out" 2>"$ARCHIVE/logs/gate_watch.err" &
    echo $! >"$ARCHIVE/status/gate.pid"
    echo "DUALSOURCE_PILOT_LAUNCHED"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
