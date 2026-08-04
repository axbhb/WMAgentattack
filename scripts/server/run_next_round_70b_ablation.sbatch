#!/usr/bin/env bash
#SBATCH --job-name=wma-70b-ablate
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
SHEEP_PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
WM_PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
CONTINUOUS_DATA="$ROOT/data/agentdojo_full_llama31_70b/splits_continuous_probability"
BINARY_DATA="$ROOT/data/agentdojo_full_llama31_70b/splits_conditional_preservation_soft"
RUN_ROOT="$ROOT/runs/agentdojo_full_hf/meta-llama-3.1-70b-instruct-transformers-4bit-compact12000-base-function_tags-ctx8192-local"
CLEAN_STATS="$ROOT/artifacts/clean_multiseed_llama31_70b_solvability.json"
ROUND_ROOT="$ROOT/artifacts/next_round_70b_ablation"
ARCHIVE=/share/guozhix/wmagentattack/0710/next_round_70b_ablation

VARIANTS=(sklearn full_binary full_binary full_binary full_horizon1 full_horizon1 full_horizon1)
SEEDS=(7 7 13 21 7 13 21)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

cd "$ROOT"
mkdir -p logs "$ROUND_ROOT" "$ARCHIVE"
test -s "$CONTINUOUS_DATA/train_steps.jsonl"
test -s "$BINARY_DATA/train_steps.jsonl"
test -s "$CLEAN_STATS"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

run_selection() {
  local py="$1"
  local backend="$2"
  local model="$3"
  local data="$4"
  local report_root="$5"
  local utility_keys="$6"
  mkdir -p "$report_root"
  for split in val test; do
    local candidates="$report_root/seed${SEED}_${split}_candidates.json"
    PYTHONWARNINGS=ignore "$py" scripts/11_select_world_model_agentdojo_pairs.py \
      --run-root "$RUN_ROOT" \
      --model "$model" \
      --model-backend "$backend" \
      --output "$candidates" \
      --top-k 32 \
      --seed "$SEED" \
      --scoring-mode clean_prefix_rollout \
      --rollout-horizon 5 \
      --allowed-trajectories "$data/${split}_trajectories.jsonl" \
      --standardized-steps "$ROOT/data/agentdojo_full_llama31_70b/steps.jsonl" \
      --max-per-user-task 2 \
      --exclude-world-model-from-baselines \
      --utility-score-key utility_score \
      --include-candidates

    local pareto="$report_root/seed${SEED}_${split}_pareto.json"
    PYTHONWARNINGS=ignore PYTHONPATH=src "$WM_PY" scripts/18_pareto_utility_selection.py \
      --candidate-json "$candidates" \
      --output "$pareto" \
      --top-k 16,24,32 \
      --seeds 7,13,21 \
      --utility-keys "$utility_keys" \
      --quantiles 0.50,0.60,0.70,0.80,0.90 \
      --max-per-user-task 2 \
      --clean-solvability-json "$CLEAN_STATS" \
      --min-base-success-rate 0.5
  done
}

run_full_variant() {
  local variant="$1"
  local data="$2"
  local horizon="$3"
  local utility_keys="$4"
  local model="$ROUND_ROOT/models/${variant}_seed${SEED}"
  local reports="$ROUND_ROOT/$variant"
  mkdir -p "$ROUND_ROOT/models" "$reports" "$ARCHIVE/$variant/seed${SEED}"

  PYTHONWARNINGS=ignore "$SHEEP_PY" scripts/23_train_full_dreamer_v3.py \
    --train "$data/train_steps.jsonl" \
    --val "$data/val_steps.jsonl" \
    --model-out "$model" \
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
    --imagination-horizon "$horizon" \
    --imagination-batch-size 256 \
    --behavior-cloning-scale 1.0 \
    --risk-reward-scale 1.0 \
    --utility-reward-scale 1.0 \
    --target-skill-reward-scale 0.25 \
    --seed "$SEED" \
    --device cuda

  for split in val test; do
    PYTHONWARNINGS=ignore "$SHEEP_PY" scripts/24_eval_full_dreamer_v3.py \
      --test "$data/${split}_steps.jsonl" \
      --model "$model" \
      --output "$reports/seed${SEED}_${split}_metrics.json"
  done
  run_selection "$SHEEP_PY" dreamer_full "$model" "$data" "$reports" "$utility_keys"
  cp "$model/metadata.json" "$ARCHIVE/$variant/seed${SEED}/"
  cp "$reports"/seed"${SEED}"_* "$ARCHIVE/$variant/seed${SEED}/"
}

if [[ "$VARIANT" == "sklearn" ]]; then
  REPORTS="$ROUND_ROOT/sklearn"
  MODEL="$ROUND_ROOT/models/sklearn_70b.joblib"
  mkdir -p "$ROUND_ROOT/models" "$REPORTS" "$ARCHIVE/sklearn/seed7"
  PYTHONWARNINGS=ignore "$WM_PY" scripts/03_train_world_model.py \
    --train "$CONTINUOUS_DATA/train_steps.jsonl" \
    --model-out "$MODEL"
  for split in val test; do
    PYTHONWARNINGS=ignore "$WM_PY" scripts/04_eval_world_model.py \
      --test "$CONTINUOUS_DATA/${split}_steps.jsonl" \
      --model "$MODEL" \
      --output "$REPORTS/seed7_${split}_metrics.json"
  done
  run_selection "$WM_PY" sklearn "$MODEL" "$CONTINUOUS_DATA" "$REPORTS" utility_score
  cp "$MODEL" "$ARCHIVE/sklearn/seed7/"
  cp "$REPORTS"/seed7_* "$ARCHIVE/sklearn/seed7/"
elif [[ "$VARIANT" == "full_binary" ]]; then
  run_full_variant full_binary "$BINARY_DATA" 5 utility_score,final_utility_score,value_score
elif [[ "$VARIANT" == "full_horizon1" ]]; then
  run_full_variant full_horizon1 "$CONTINUOUS_DATA" 1 preservation_score,utility_score,final_utility_score,value_score
else
  echo "Unknown variant: $VARIANT" >&2
  exit 2
fi

echo "NEXT_ROUND_70B_ABLATION_DONE variant=$VARIANT seed=$SEED $(date -Is)"
