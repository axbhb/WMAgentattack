#!/usr/bin/env bash
#SBATCH --job-name=wma-tau3-ms-method
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/share/guozhix/WMagentattack-tau3-multistep-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-tau3-multistep-aug5/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-tau3-multistep-aug5
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
PILOT=/share/guozhix/wmagentattack/0806/tau3_multistep_scale_readiness/pilot_v1
METHOD=/share/guozhix/wmagentattack/0806/tau3_multistep_scale_readiness/method_v1
PROTOCOL=$METHOD/frozen_protocol.json

cd "$ROOT"
mkdir -p "$METHOD/training"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PY" scripts/165_train_tau3_multistep_scale_readiness.py \
  --protocol "$PROTOCOL" \
  --dataset "$PILOT/tau3_multistep_dataset.json" \
  --audit "$PILOT/dataset_audit.json" \
  --pilot-gate "$PILOT/pilot_gate.json" \
  --output-dir "$METHOD/training" \
  --device auto

"$PY" scripts/166_summarize_tau3_multistep_scale_readiness.py \
  --protocol "$PROTOCOL" \
  --pilot-gate "$PILOT/pilot_gate.json" \
  --run-metrics "$METHOD/training/run_metrics.json" \
  --action-predictions "$METHOD/training/action_predictions.jsonl" \
  --transition-predictions "$METHOD/training/transition_predictions.jsonl" \
  --output "$METHOD/method_gate.json" \
  --report "$METHOD/method_results.md"

(cd "$METHOD" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "TAU3_MULTISTEP_METHOD_DONE $(date -Is)"
