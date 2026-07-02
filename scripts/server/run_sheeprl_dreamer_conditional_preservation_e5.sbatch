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
CLEAN_STATS="${CLEAN_STATS:-$ROOT/artifacts/clean_multiseed_llama31_8b_solvability.json}"
SPLIT_DIR="${SPLIT_DIR:-$ROOT/data/agentdojo_full_llama31_8b/splits}"
COND_SPLIT_DIR="${COND_SPLIT_DIR:-$ROOT/data/agentdojo_full_llama31_8b/splits_conditional_preservation}"
MODEL_OUT="$ROOT/artifacts/agentdojo_full_llama31_8b_dreamer_world_model_conditional_preservation_e5"
THRESHOLD="${THRESHOLD:-0.5}"

mkdir -p "$ROOT/logs" "$ARCHIVE"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/share/guozhix/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

test -f "$CLEAN_STATS"

PYTHONWARNINGS=ignore PYTHONPATH=src "$WM_PY" scripts/21_apply_conditional_preservation_labels.py \
  --split-dir "$SPLIT_DIR" \
  --clean-solvability-json "$CLEAN_STATS" \
  --out-dir "$COND_SPLIT_DIR" \
  --min-base-success-rate "$THRESHOLD"

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
  --kl-scale 0.05 \
  --kl-dynamic-scale 0.5 \
  --kl-representation-scale 0.1 \
  --kl-free-nats 1.0 \
  --utility-pos-weight 1.0 \
  --seed 7

for SPLIT in val test; do
  PYTHONWARNINGS=ignore PYTHONPATH=src "$SHEEPRL_PY" scripts/16_eval_dreamer_world_model.py \
    --test "$COND_SPLIT_DIR/${SPLIT}_steps.jsonl" \
    --model "$MODEL_OUT" \
    --output "$ROOT/artifacts/agentdojo_full_llama31_8b_dreamer_world_model_conditional_preservation_e5_${SPLIT}_metrics.json"
done

cp "$CLEAN_STATS" "$ARCHIVE"/
cp "$COND_SPLIT_DIR/conditional_summary.json" "$ARCHIVE"/
cp "$ROOT"/artifacts/agentdojo_full_llama31_8b_dreamer_world_model_conditional_preservation_e5_*_metrics.json "$ARCHIVE"/
cp "$MODEL_OUT/training_history.json" "$ARCHIVE"/conditional_preservation_training_history.json

