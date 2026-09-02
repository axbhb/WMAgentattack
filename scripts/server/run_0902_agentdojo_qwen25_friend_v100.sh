#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: $0 smoke|clean-worker|clean-gate|attack-worker|summary [worker]}
WORKER=${2:-0}
WT=/home/pth/projects/WMagentattack-qwen-parity-sep2
PY=/home/pth/venvs/wmagentattack_qwen25/bin/python
MODEL=/home/pth/models/Qwen2.5-7B-Instruct
ARCHIVE=/home/pth/outputs/wmagentattack/0902/agentdojo_qwen25_paper_parity/formal_v1
SMOKE=/home/pth/outputs/wmagentattack/0902/agentdojo_qwen25_paper_parity/smoke_v1
MODEL_LABEL=qwen2.5-7b-paper-parity-greedy

cd "$WT"
mkdir -p "$ARCHIVE/runs" "$SMOKE" "$ARCHIVE/logs" "$ARCHIVE/status"
export PYTHONPATH="$WT/src:$WT/external/agentdojo/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

run_suite() {
  local suite=$1
  local mode=$2
  local logdir="$ARCHIVE/runs/$suite"
  mkdir -p "$logdir"
  args=(
    --model-path "$MODEL"
    --model-label "$MODEL_LABEL"
    --benchmark-version v1
    --suite "$suite"
    --max-new-tokens 256
    --max-tool-output-chars 12000
    --prompt-profile base
    --max-input-tokens 8192
    --protocol native
    --quantization 4bit
    --device cuda:0
    --logdir "$logdir"
  )
  if [[ "$mode" == clean ]]; then
    args+=(--clean-only --force-rerun)
  elif [[ "$mode" == attack ]]; then
    args+=(--attack important_instructions)
  else
    echo "unsupported suite mode: $mode" >&2
    exit 2
  fi
  "$PY" scripts/10_run_agentdojo_hf_full.py "${args[@]}"
}

case "$MODE" in
  smoke)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
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
      --logdir "$SMOKE" \
      --force-rerun
    touch "$SMOKE/COMPLETE"
    ;;
  clean-worker|attack-worker)
    if [[ "$WORKER" == 0 ]]; then
      suites=(workspace banking)
    elif [[ "$WORKER" == 1 ]]; then
      suites=(travel slack)
    else
      echo "worker must be 0 or 1" >&2
      exit 2
    fi
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$WORKER}
    phase=${MODE%-worker}
    for suite in "${suites[@]}"; do
      run_suite "$suite" "$phase"
    done
    touch "$ARCHIVE/status/${phase}_worker_${WORKER}.complete"
    ;;
  clean-gate)
    test -f "$ARCHIVE/status/clean_worker_0.complete"
    test -f "$ARCHIVE/status/clean_worker_1.complete"
    "$PY" scripts/302_gate_agentdojo_qwen_clean.py \
      --run-root "$ARCHIVE/runs" \
      --model-label "$MODEL_LABEL" \
      --output "$ARCHIVE/clean_gate.json"
    ;;
  summary)
    test -f "$ARCHIVE/status/attack_worker_0.complete"
    test -f "$ARCHIVE/status/attack_worker_1.complete"
    "$PY" scripts/301_summarize_agentdojo_qwen_paper_parity.py \
      --run-root "$ARCHIVE/runs" \
      --model-label "$MODEL_LABEL" \
      --attack important_instructions \
      --output "$ARCHIVE/paper_parity_summary.json"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
