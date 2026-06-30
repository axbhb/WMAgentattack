#!/bin/bash
#SBATCH --job-name=wma-dreamer-tuned-e5
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python

cd "$ROOT"
mkdir -p logs artifacts

PYTHONWARNINGS=ignore PYTHONPATH=src "$PY" scripts/15_train_dreamer_world_model.py \
  --train data/agentdojo_full_llama31_8b/splits/train_steps.jsonl \
  --model-out artifacts/agentdojo_full_llama31_8b_dreamer_world_model_risk2_util2_e5 \
  --epochs 5 \
  --batch-size 16 \
  --risk-loss-scale 2.0 \
  --utility-loss-scale 2.0 \
  --risk-pos-weight 2.0 \
  --utility-pos-weight 1.0

PYTHONWARNINGS=ignore PYTHONPATH=src "$PY" scripts/16_eval_dreamer_world_model.py \
  --test data/agentdojo_full_llama31_8b/splits/test_steps.jsonl \
  --model artifacts/agentdojo_full_llama31_8b_dreamer_world_model_risk2_util2_e5 \
  --output artifacts/agentdojo_full_llama31_8b_dreamer_world_model_risk2_util2_e5_metrics.json
