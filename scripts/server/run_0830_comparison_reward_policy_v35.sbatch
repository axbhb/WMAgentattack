#!/usr/bin/env bash
#SBATCH --job-name=wma-v35-compare-reward
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/share/guozhix/WMagentattack-comparison-reward-aug30/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-comparison-reward-aug30/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-comparison-reward-aug30
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
PROTOCOL=$ROOT/configs/0830_comparison_reward_policy_v35_protocol.json
ARCHIVE=/share/guozhix/wmagentattack/0830/comparison_reward_policy_v35/formal_v1
MANIFEST=/share/guozhix/wmagentattack/0714/agentdojo_v2/manifests/screen_manifest.json
LABELS=/share/guozhix/wmagentattack/0714/agentdojo_v2/final_dataset/label_groups.jsonl
V5=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/formal_v1/results/predictions.jsonl

cd "$ROOT"
mkdir -p "$ARCHIVE/results" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PY" -m pytest -q tests/test_comparison_reward_policy.py tests/test_attack_conditioned_ranker.py
"$PY" scripts/283_train_comparison_reward_policy_v35.py \
  --protocol "$PROTOCOL" \
  --manifest "$MANIFEST" \
  --label-groups "$LABELS" \
  --v5-predictions "$V5" \
  --output-dir "$ARCHIVE/results"
"$PY" scripts/284_gate_comparison_reward_policy_v35.py \
  --protocol "$PROTOCOL" \
  --predictions "$ARCHIVE/results/predictions.jsonl" \
  --support-audit "$ARCHIVE/results/preference_support_audit.json" \
  --run-metrics "$ARCHIVE/results/run_metrics.json" \
  --output "$ARCHIVE/gate.json"

echo "WMA_V35_COMPARISON_REWARD_DONE $(date -Is)"
