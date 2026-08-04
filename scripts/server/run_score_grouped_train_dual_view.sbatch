#!/usr/bin/env bash
#SBATCH --job-name=wma-gtrain-score
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
DATA=/share/guozhix/wmagentattack/0712/grouped_user_task_split_continuous_probability
RUN_ROOT="$ROOT/runs/agentdojo_full_hf/meta-llama-3.1-70b-instruct-transformers-4bit-compact12000-base-function_tags-ctx8192-local"
MODEL_ROOT="$ROOT/artifacts/grouped_full_dreamer_v3_llama31_70b"
ARCHIVE=/share/guozhix/wmagentattack/0713/grouped_train_scores
SEEDS=(7 13 21)
MODES=(clean_prefix_rollout injection_conditioned_rollout)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
MODE_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
SEED="${SEEDS[$SEED_INDEX]}"
MODE="${MODES[$MODE_INDEX]}"
MODEL="${MODEL_ROOT}_seed${SEED}"
OUT_DIR="$ARCHIVE/seed${SEED}"
OUTPUT="$OUT_DIR/train_${MODE}_candidates.json"

cd "$ROOT"
mkdir -p "$OUT_DIR" logs
echo 'e994793672512680f370a452b814f23bb0a72ad7bc2d80c959cb44953a53b7a1  /share/guozhix/wmagentattack/0712/grouped_user_task_split_continuous_probability/train_steps.jsonl' | sha256sum -c -
test -s "$MODEL/metadata.json"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PYTHONWARNINGS=ignore "$PY" scripts/11_select_world_model_agentdojo_pairs.py \
  --run-root "$RUN_ROOT" \
  --model "$MODEL" \
  --model-backend dreamer_full \
  --output "$OUTPUT" \
  --top-k 64 \
  --seed "$SEED" \
  --scoring-mode "$MODE" \
  --rollout-horizon 5 \
  --allowed-trajectories "$DATA/train_trajectories.jsonl" \
  --standardized-steps "$ROOT/data/agentdojo_full_llama31_70b/steps.jsonl" \
  --max-per-user-task 0 \
  --exclude-world-model-from-baselines \
  --utility-score-key preservation_score \
  --include-candidates \
  >"$OUT_DIR/train_${MODE}_stdout.json"

sha256sum "$OUTPUT" >"$OUTPUT.sha256"
CANDIDATE_COUNT="$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["candidates"]))' "$OUTPUT")"
echo "GROUPED_TRAIN_SCORE_DONE seed=$SEED mode=$MODE candidates=$CANDIDATE_COUNT $(date -Is)"
