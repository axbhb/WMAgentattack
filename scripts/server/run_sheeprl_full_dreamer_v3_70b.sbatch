#!/usr/bin/env bash
#SBATCH --job-name=wma-full-dv3-70b
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
WM_PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
DATA="$ROOT/data/agentdojo_full_llama31_70b/splits_continuous_probability"
RUN_ROOT="$ROOT/runs/agentdojo_full_hf/meta-llama-3.1-70b-instruct-transformers-4bit-compact12000-base-function_tags-ctx8192-local"
CLEAN_STATS="$ROOT/artifacts/clean_multiseed_llama31_70b_solvability.json"
SEEDS=(7 13 21)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
MODEL_OUT="$ROOT/artifacts/full_dreamer_v3_llama31_70b_seed${SEED}"
REPORT_ROOT="$ROOT/artifacts/full_dreamer_v3_llama31_70b"
ARCHIVE=/share/guozhix/wmagentattack/0710/full_dreamer_v3_llama31_70b

cd "$ROOT"
mkdir -p logs "$REPORT_ROOT" "$ARCHIVE/seed${SEED}"
test -s "$DATA/train_steps.jsonl"
test -s "$DATA/val_steps.jsonl"
test -s "$DATA/test_steps.jsonl"
test -s "$CLEAN_STATS"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PYTHONWARNINGS=ignore "$PY" scripts/23_train_full_dreamer_v3.py \
  --train "$DATA/train_steps.jsonl" \
  --val "$DATA/val_steps.jsonl" \
  --model-out "$MODEL_OUT" \
  --epochs 30 \
  --batch-size 16 \
  --obs-dim 768 \
  --dense-units 256 \
  --recurrent-state-size 256 \
  --stochastic-size 32 \
  --discrete-size 32 \
  --encoder-layers 2 \
  --decoder-layers 2 \
  --actor-layers 2 \
  --critic-layers 2 \
  --head-layers 1 \
  --reward-bins 255 \
  --imagination-horizon 5 \
  --imagination-batch-size 256 \
  --behavior-cloning-scale 1.0 \
  --risk-reward-scale 1.0 \
  --utility-reward-scale 1.0 \
  --target-skill-reward-scale 0.25 \
  --seed "$SEED" \
  --device cuda

for SPLIT in val test; do
  OUTPUT="$REPORT_ROOT/seed${SEED}_${SPLIT}_metrics.json"
  PYTHONWARNINGS=ignore "$PY" scripts/24_eval_full_dreamer_v3.py \
    --test "$DATA/${SPLIT}_steps.jsonl" \
    --model "$MODEL_OUT" \
    --output "$OUTPUT"

  CANDIDATES="$REPORT_ROOT/seed${SEED}_${SPLIT}_candidates.json"
  PYTHONWARNINGS=ignore "$PY" scripts/11_select_world_model_agentdojo_pairs.py \
    --run-root "$RUN_ROOT" \
    --model "$MODEL_OUT" \
    --model-backend dreamer_full \
    --output "$CANDIDATES" \
    --top-k 32 \
    --seed "$SEED" \
    --scoring-mode clean_prefix_rollout \
    --rollout-horizon 5 \
    --allowed-trajectories "$DATA/${SPLIT}_trajectories.jsonl" \
    --standardized-steps "$ROOT/data/agentdojo_full_llama31_70b/steps.jsonl" \
    --max-per-user-task 2 \
    --exclude-world-model-from-baselines \
    --utility-score-key preservation_score \
    --include-candidates

  PARETO="$REPORT_ROOT/seed${SEED}_${SPLIT}_pareto.json"
  PYTHONWARNINGS=ignore PYTHONPATH=src "$WM_PY" scripts/18_pareto_utility_selection.py \
    --candidate-json "$CANDIDATES" \
    --output "$PARETO" \
    --top-k 16,24,32 \
    --seeds 7,13,21 \
    --utility-keys preservation_score,utility_score,final_utility_score,value_score \
    --quantiles 0.50,0.60,0.70,0.80,0.90 \
    --max-per-user-task 2 \
    --clean-solvability-json "$CLEAN_STATS" \
    --min-base-success-rate 0.5

  cp "$OUTPUT" "$ARCHIVE/seed${SEED}/"
  cp "$CANDIDATES" "$ARCHIVE/seed${SEED}/"
  cp "$PARETO" "$ARCHIVE/seed${SEED}/"
  cp "${PARETO%.json}.csv" "$ARCHIVE/seed${SEED}/"
done

cp "$MODEL_OUT/metadata.json" "$ARCHIVE/seed${SEED}/"
echo "FULL_DREAMER_V3_70B_DONE seed=$SEED $(date -Is)"
