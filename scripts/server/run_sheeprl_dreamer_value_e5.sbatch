#!/bin/bash
#SBATCH --job-name=wma-dreamer-value-e5
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
WM_PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
RUN_ROOT="$ROOT/runs/agentdojo_full_hf/meta-llama-3.1-8b-instruct-transformers-4bit-compact12000-base-function_tags-ctx8192-local"
MODEL_OUT="$ROOT/artifacts/agentdojo_full_llama31_8b_dreamer_world_model_value_e5"
ARCHIVE=/share/guozhix/wmagentattack/0630/dreamer_sheeprl/utility_value_round

cd "$ROOT"
mkdir -p logs artifacts "$ARCHIVE"

PYTHONWARNINGS=ignore PYTHONPATH=src "$PY" scripts/15_train_dreamer_world_model.py \
  --train data/agentdojo_full_llama31_8b/splits/train_steps.jsonl \
  --model-out "$MODEL_OUT" \
  --epochs 5 \
  --batch-size 16 \
  --risk-loss-scale 2.0 \
  --utility-loss-scale 2.0 \
  --final-utility-loss-scale 2.0 \
  --risk-pos-weight 2.0 \
  --utility-pos-weight 1.0 \
  --kl-dynamic-scale 0.5 \
  --kl-representation-scale 0.1 \
  --kl-free-nats 1.0

PYTHONWARNINGS=ignore PYTHONPATH=src "$PY" scripts/16_eval_dreamer_world_model.py \
  --test data/agentdojo_full_llama31_8b/splits/test_steps.jsonl \
  --model "$MODEL_OUT" \
  --output artifacts/agentdojo_full_llama31_8b_dreamer_world_model_value_e5_metrics.json

for SPLIT in val test; do
  PYTHONWARNINGS=ignore PYTHONPATH=src "$PY" scripts/11_select_world_model_agentdojo_pairs.py \
    --run-root "$RUN_ROOT" \
    --model "$MODEL_OUT" \
    --model-backend dreamer \
    --output "artifacts/dreamer_value_${SPLIT}_candidate_cache.json" \
    --top-k 32 \
    --seed 7 \
    --scoring-mode clean_prefix_rollout \
    --rollout-horizon 3 \
    --allowed-trajectories "data/agentdojo_full_llama31_8b/splits/${SPLIT}_trajectories.jsonl" \
    --standardized-steps data/agentdojo_full_llama31_8b/steps.jsonl \
    --max-per-user-task 2 \
    --exclude-world-model-from-baselines \
    --include-candidates

  PYTHONWARNINGS=ignore PYTHONPATH=src "$WM_PY" scripts/17_tune_selection_weights.py \
    --candidate-json "artifacts/dreamer_value_${SPLIT}_candidate_cache.json" \
    --output "artifacts/dreamer_value_${SPLIT}_utility_value_sweep.json" \
    --top-k 16,24,32 \
    --seeds 7,13,21 \
    --max-per-user-task 2
done

cp artifacts/agentdojo_full_llama31_8b_dreamer_world_model_value_e5_metrics.json "$ARCHIVE"/
cp artifacts/dreamer_value_*_candidate_cache.json "$ARCHIVE"/
cp artifacts/dreamer_value_*_utility_value_sweep.json "$ARCHIVE"/
cp artifacts/dreamer_value_*_utility_value_sweep.csv "$ARCHIVE"/
