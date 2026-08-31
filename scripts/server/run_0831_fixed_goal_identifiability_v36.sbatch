#!/usr/bin/env bash
#SBATCH --job-name=wma-v36-fixed-goal
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/share/guozhix/WMagentattack-fixed-goal-aug31/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-fixed-goal-aug31/logs/%x-%j.err
set -euo pipefail
ROOT=/share/guozhix/WMagentattack-fixed-goal-aug31
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0831/fixed_goal_identifiability_v36/formal_v1
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" -m pytest -q tests/test_fixed_goal_identifiability.py tests/test_comparison_reward_policy.py tests/test_attack_conditioned_ranker.py
"$PY" scripts/285_audit_fixed_goal_identifiability_v36.py --protocol configs/0831_fixed_goal_identifiability_v36_protocol.json --output "$ARCHIVE/gate.json"
date -Iseconds > "$ARCHIVE/COMPLETE"
