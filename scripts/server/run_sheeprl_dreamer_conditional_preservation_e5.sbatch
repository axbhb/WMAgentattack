#!/usr/bin/env bash
#SBATCH --job-name=wma-dreamer-cond-e5
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
SHEEPRL_PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
WM_PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0630/dreamer_sheeprl/conditional_preservation_round
SOFT_ARCHIVE=/share/guozhix/wmagentattack/0630/dreamer_sheeprl/conditional_preservation_soft_round
CLEAN_STATS="${CLEAN_STATS:-$ROOT/artifacts/clean_multiseed_llama31_8b_solvability.json}"
SPLIT_DIR="${SPLIT_DIR:-$ROOT/data/agentdojo_full_llama31_8b/splits}"
COND_SPLIT_DIR="${COND_SPLIT_DIR:-$ROOT/data/agentdojo_full_llama31_8b/splits_conditional_preservation_soft}"
MODEL_OUT="$ROOT/artifacts/agentdojo_full_llama31_8b_dreamer_world_model_conditional_preservation_soft_e5"
THRESHOLD="${THRESHOLD:-0.5}"
WEIGHT_FLOOR="${WEIGHT_FLOOR:-0.05}"
RUN_ROOT="$ROOT/runs/agentdojo_full_hf/meta-llama-3.1-8b-instruct-transformers-4bit-compact12000-base-function_tags-ctx8192-local"

mkdir -p "$ROOT/logs" "$ARCHIVE" "$SOFT_ARCHIVE"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/share/guozhix/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

test -f "$CLEAN_STATS"

PYTHONWARNINGS=ignore PYTHONPATH=src "$WM_PY" scripts/21_apply_conditional_preservation_labels.py \
  --split-dir "$SPLIT_DIR" \
  --clean-solvability-json "$CLEAN_STATS" \
  --out-dir "$COND_SPLIT_DIR" \
  --min-base-success-rate "$THRESHOLD" \
  --preservation-weight-mode soft \
  --preservation-weight-floor "$WEIGHT_FLOOR"

PYTHONWARNINGS=ignore PYTHONPATH=src "$SHEEPRL_PY" scripts/15_train_dreamer_world_model.py \
  --train "$COND_SPLIT_DIR/train_steps.jsonl" \
  --model-out "$MODEL_OUT" \
  --epochs 5 \
  --batch-size 16 \
  --risk-loss-scale 2.0 \
  --utility-loss-scale 2.0 \
  --final-utility-loss-scale 2.0 \
  --utility-ranking-loss-scale 1.0 \
  --utility-ranking-margin 0.2 \
  --kl-dynamic-scale 0.5 \
  --kl-representation-scale 0.1 \
  --kl-free-nats 1.0 \
  --utility-pos-weight 1.0

for SPLIT in val test; do
  PYTHONWARNINGS=ignore PYTHONPATH=src "$SHEEPRL_PY" scripts/16_eval_dreamer_world_model.py \
    --test "$COND_SPLIT_DIR/${SPLIT}_steps.jsonl" \
    --model "$MODEL_OUT" \
    --output "$ROOT/artifacts/agentdojo_full_llama31_8b_dreamer_world_model_conditional_preservation_soft_e5_${SPLIT}_metrics.json"

  PYTHONWARNINGS=ignore PYTHONPATH=src "$SHEEPRL_PY" scripts/11_select_world_model_agentdojo_pairs.py \
    --run-root "$RUN_ROOT" \
    --model "$MODEL_OUT" \
    --model-backend dreamer \
    --output "artifacts/dreamer_conditional_soft_${SPLIT}_candidate_cache.json" \
    --top-k 32 \
    --seed 7 \
    --scoring-mode clean_prefix_rollout \
    --rollout-horizon 3 \
    --allowed-trajectories "data/agentdojo_full_llama31_8b/splits/${SPLIT}_trajectories.jsonl" \
    --standardized-steps data/agentdojo_full_llama31_8b/steps.jsonl \
    --max-per-user-task 2 \
    --exclude-world-model-from-baselines \
    --include-candidates

  PYTHONWARNINGS=ignore PYTHONPATH=src "$WM_PY" scripts/18_pareto_utility_selection.py \
    --candidate-json "artifacts/dreamer_conditional_soft_${SPLIT}_candidate_cache.json" \
    --output "artifacts/dreamer_conditional_soft_${SPLIT}_pareto_utility_selection.json" \
    --top-k 16,24,32 \
    --seeds 7,13,21 \
    --utility-keys utility_score,final_utility_score,min_utility_score \
    --quantiles 0.50,0.60,0.70,0.80,0.90 \
    --max-per-user-task 2 \
    --clean-solvability-json "$CLEAN_STATS" \
    --min-base-success-rate "$THRESHOLD"
done

cp "$CLEAN_STATS" "$SOFT_ARCHIVE"/
cp "$COND_SPLIT_DIR/conditional_summary.json" "$SOFT_ARCHIVE"/
cp "$ROOT"/artifacts/agentdojo_full_llama31_8b_dreamer_world_model_conditional_preservation_soft_e5_*_metrics.json "$SOFT_ARCHIVE"/
cp "$MODEL_OUT/metadata.json" "$SOFT_ARCHIVE"/conditional_preservation_soft_metadata.json
cp artifacts/dreamer_conditional_soft_*_candidate_cache.json "$SOFT_ARCHIVE"/
cp artifacts/dreamer_conditional_soft_*_pareto_utility_selection.json "$SOFT_ARCHIVE"/
cp artifacts/dreamer_conditional_soft_*_pareto_utility_selection.csv "$SOFT_ARCHIVE"/
