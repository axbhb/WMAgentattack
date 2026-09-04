#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: $0 batch0|batch1|finalize|launch}
WT=/home/pth/projects/WMagentattack-llama31-8b-dualsource-full-sep4
PY=/home/pth/venvs/wmagentattack_qwen25/bin/python
MODEL=/home/pth/models/Meta-Llama-3.1-8B-Instruct
ARCHIVE=/home/pth/outputs/wmagentattack/0904/llama31_8b_dualsource_full/formal_v1
PROTOCOL=$WT/configs/0904_llama31_8b_dualsource_full_protocol.json
SELECTION=$WT/data/0904_llama31_8b_dualsource_full/agentdojo_full_selection.json
INJEC_MANIFEST=$WT/data/0904_llama31_8b_dualsource_full/injecagent_full_manifest.json

cd "$WT"
mkdir -p "$ARCHIVE"/{runs,injecagent,logs,status,normalized}
export PYTHONPATH="$WT/src:$WT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

run_agentdojo_seed() {
  local seed=$1
  local device=$2
  local logdir="$ARCHIVE/runs/seed${seed}"
  CUDA_VISIBLE_DEVICES="$device" "$PY" scripts/10_run_agentdojo_hf_full.py \
    --model-path "$MODEL" \
    --model-label "llama31-8b-dualsource-full-seed${seed}" \
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
    --logdir "$logdir"
  touch "$ARCHIVE/status/agentdojo_seed${seed}.complete"
}

run_injecagent_seed() {
  local seed=$1
  local device=$2
  CUDA_VISIBLE_DEVICES="$device" "$PY" scripts/307_run_llama31_8b_injecagent_full.py \
    --protocol "$PROTOCOL" \
    --manifest "$INJEC_MANIFEST" \
    --output-root "$ARCHIVE/injecagent" \
    --seed "$seed" \
    --device cuda:0
  touch "$ARCHIVE/status/injecagent_seed${seed}.complete"
}

case "$MODE" in
  batch0)
    run_agentdojo_seed 1103 0
    run_agentdojo_seed 1117 0
    touch "$ARCHIVE/status/batch0.complete"
    ;;
  batch1)
    run_agentdojo_seed 1109 1
    run_injecagent_seed 1103 1
    run_injecagent_seed 1109 1
    run_injecagent_seed 1117 1
    touch "$ARCHIVE/status/batch1.complete"
    ;;
  finalize)
    while [[ ! -f "$ARCHIVE/status/batch0.complete" || ! -f "$ARCHIVE/status/batch1.complete" ]]; do
      sleep 120
    done
    if "$PY" scripts/308_finalize_llama31_8b_dualsource_full.py \
      --protocol "$PROTOCOL" \
      --selection "$SELECTION" \
      --injecagent-manifest "$INJEC_MANIFEST" \
      --agentdojo-root "$ARCHIVE/runs" \
      --injecagent-root "$ARCHIVE/injecagent" \
      --output-dir "$ARCHIVE/normalized" \
      >"$ARCHIVE/logs/finalize.out" 2>"$ARCHIVE/logs/finalize.err"; then
      touch "$ARCHIVE/status/full_collection.complete"
    else
      touch "$ARCHIVE/status/full_collection.invalid"
      exit 1
    fi
    ;;
  launch)
    test -s "$MODEL/config.json"
    test -s "$SELECTION"
    test -s "$INJEC_MANIFEST"
    if [[ -e "$ARCHIVE/status/launched" ]]; then
      echo "Full collection already launched; refusing duplicate."
      exit 3
    fi
    touch "$ARCHIVE/status/launched"
    nohup bash "$0" batch0 >"$ARCHIVE/logs/batch0.out" 2>"$ARCHIVE/logs/batch0.err" &
    echo $! >"$ARCHIVE/status/batch0.pid"
    nohup bash "$0" batch1 >"$ARCHIVE/logs/batch1.out" 2>"$ARCHIVE/logs/batch1.err" &
    echo $! >"$ARCHIVE/status/batch1.pid"
    nohup bash "$0" finalize >"$ARCHIVE/logs/finalize_watch.out" 2>"$ARCHIVE/logs/finalize_watch.err" &
    echo $! >"$ARCHIVE/status/finalize.pid"
    echo "DUALSOURCE_FULL_COLLECTION_LAUNCHED"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
