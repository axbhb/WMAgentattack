#!/usr/bin/env bash
#SBATCH --job-name=wma-score-contrast
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-8%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
DATA="$ROOT/data/agentdojo_full_llama31_70b"
RUN_ROOT="$ROOT/runs/agentdojo_full_hf/meta-llama-3.1-70b-instruct-transformers-4bit-compact12000-base-function_tags-ctx8192-local"
ARCHIVE=/share/guozhix/wmagentattack/0712/within_task_contrast/train_candidate_scores
SOURCES=(continuous norank binary)
SEEDS=(7 13 21)
SOURCE_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SOURCE="${SOURCES[$SOURCE_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

case "$SOURCE" in
  continuous)
    MODEL="$ROOT/artifacts/full_dreamer_v3_llama31_70b_seed${SEED}"
    ;;
  norank)
    MODEL="$ROOT/artifacts/full_dreamer_v3_llama31_70b_hybrid_lite_norank_seed${SEED}"
    ;;
  binary)
    MODEL="$ROOT/artifacts/next_round_70b_ablation/models/full_binary_seed${SEED}"
    ;;
  *)
    echo "Unknown source: $SOURCE" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="$ARCHIVE/$SOURCE"
OUTPUT="$OUTPUT_DIR/seed${SEED}_train_candidates.json"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" logs
test -s "$MODEL/metadata.json"
test -s "$DATA/splits_continuous_probability/train_trajectories.jsonl"
test -s "$DATA/steps.jsonl"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PYTHONWARNINGS=ignore "$PY" scripts/11_select_world_model_agentdojo_pairs.py \
  --run-root "$RUN_ROOT" \
  --model "$MODEL" \
  --model-backend dreamer_full \
  --output "$OUTPUT" \
  --top-k 96 \
  --seed "$SEED" \
  --scoring-mode clean_prefix_rollout \
  --rollout-horizon 5 \
  --allowed-trajectories "$DATA/splits_continuous_probability/train_trajectories.jsonl" \
  --standardized-steps "$DATA/steps.jsonl" \
  --max-per-user-task 0 \
  --utility-score-key preservation_score \
  --include-candidates \
  >"$OUTPUT_DIR/seed${SEED}_stdout.json"

echo "TRAIN_CONTRAST_SCORING_DONE source=$SOURCE seed=$SEED $(date -Is)"
