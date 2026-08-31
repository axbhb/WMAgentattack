#!/usr/bin/env bash
#SBATCH --job-name=wma-v37-clean-pairing
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/share/guozhix/WMagentattack-clean-pairing-aug31/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-clean-pairing-aug31/logs/%x-%j.err
set -euo pipefail
ROOT=/share/guozhix/WMagentattack-clean-pairing-aug31
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0831/clean_pairing_v37/formal_v1
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" -m pytest -q tests/test_clean_pairing.py tests/test_fixed_goal_identifiability.py tests/test_comparison_reward_policy.py tests/test_attack_conditioned_ranker.py
"$PY" scripts/286_run_clean_pairing_v37.py --protocol "$ARCHIVE/preregistered_protocol.json"
date -Iseconds > "$ARCHIVE/COMPLETE"
