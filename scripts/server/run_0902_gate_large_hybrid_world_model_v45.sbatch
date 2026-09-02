#!/bin/bash
#SBATCH --job-name=wm-v45-gate
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/share/guozhix/wmagentattack/0902/large_hybrid_world_model_v45/formal_v1/logs/gate-%j.out
#SBATCH --error=/share/guozhix/wmagentattack/0902/large_hybrid_world_model_v45/formal_v1/logs/gate-%j.err

set -euo pipefail

REPO=${WMAGENTATTACK_REPO:-/share/guozhix/WMagentattack-large-hybrid-sep2}
ARCHIVE=/share/guozhix/wmagentattack/0902/large_hybrid_world_model_v45/formal_v1
DATASET=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/preflight_v1/a/dataset.json
V5=/share/guozhix/wmagentattack/0817/structured_joint_outcome_v5/formal_v1/results/predictions.jsonl
V22=/share/guozhix/wmagentattack/0821/parallel_world_model_gates_v22/formal_v2/long_horizon/results/predictions.jsonl
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

"${PY}" scripts/299_gate_large_hybrid_world_model_v45.py \
  --protocol configs/0902_large_hybrid_world_model_v45_protocol.json \
  --dataset "${DATASET}" \
  --runs-root "${ARCHIVE}/runs" \
  --v5-predictions "${V5}" \
  --v22-predictions "${V22}" \
  --output "${ARCHIVE}/large_hybrid_world_model_v45_summary.json" \
  --markdown "${ARCHIVE}/large_hybrid_world_model_v45_results.md"
