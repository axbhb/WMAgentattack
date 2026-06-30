#!/bin/bash
#SBATCH --job-name=wma-dreamer-smoke
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
CONDA=/share/guozhix/software/miniconda3/bin/conda

cd "$ROOT"
mkdir -p logs artifacts

"$CONDA" run -n sheeprl_env python scripts/15_train_dreamer_world_model.py \
  --train data/agentdojo_full_llama31_8b/splits/train_steps.jsonl \
  --model-out artifacts/agentdojo_full_llama31_8b_dreamer_world_model \
  --epochs 3 \
  --batch-size 16

"$CONDA" run -n sheeprl_env python scripts/16_eval_dreamer_world_model.py \
  --test data/agentdojo_full_llama31_8b/splits/test_steps.jsonl \
  --model artifacts/agentdojo_full_llama31_8b_dreamer_world_model \
  --output artifacts/agentdojo_full_llama31_8b_dreamer_world_model_metrics.json
