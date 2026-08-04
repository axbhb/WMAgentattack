#!/usr/bin/env bash
#SBATCH --job-name=wma-grouped-dv3
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
DATA=/share/guozhix/wmagentattack/0712/grouped_user_task_split_continuous_probability
RUN_ROOT="$ROOT/runs/agentdojo_full_hf/meta-llama-3.1-70b-instruct-transformers-4bit-compact12000-base-function_tags-ctx8192-local"
MODEL_ROOT="$ROOT/artifacts/grouped_full_dreamer_v3_llama31_70b"
ARCHIVE=/share/guozhix/wmagentattack/0712/grouped_user_task_dreamer
SEEDS=(7 13 21)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
MODEL_OUT="${MODEL_ROOT}_seed${SEED}"
SEED_ARCHIVE="$ARCHIVE/seed${SEED}"

cd "$ROOT"
mkdir -p "$MODEL_ROOT" "$SEED_ARCHIVE" logs
echo 'e994793672512680f370a452b814f23bb0a72ad7bc2d80c959cb44953a53b7a1  /share/guozhix/wmagentattack/0712/grouped_user_task_split_continuous_probability/train_steps.jsonl' | sha256sum -c -
echo 'ef4ee8add6b8a2d5a073ab9b45a14ddb1e04da8919c6a00b44d96ea908f6d3d5  /share/guozhix/wmagentattack/0712/grouped_user_task_split_continuous_probability/val_steps.jsonl' | sha256sum -c -
echo 'ed8f9638c343694249fa17c557dc68841d4f937da9e8428d18feffa61bbda630  /share/guozhix/wmagentattack/0712/grouped_user_task_split_continuous_probability/test_steps.jsonl' | sha256sum -c -
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
  --device cuda \
  >"$SEED_ARCHIVE/training_stdout.json"

for split in val test; do
  PYTHONWARNINGS=ignore "$PY" scripts/24_eval_full_dreamer_v3.py \
    --test "$DATA/${split}_steps.jsonl" \
    --model "$MODEL_OUT" \
    --output "$SEED_ARCHIVE/${split}_metrics.json" \
    >"$SEED_ARCHIVE/${split}_metrics_stdout.json"

  for mode in clean_prefix_rollout injection_conditioned_rollout; do
    output="$SEED_ARCHIVE/${split}_${mode}_candidates.json"
    PYTHONWARNINGS=ignore "$PY" scripts/11_select_world_model_agentdojo_pairs.py \
      --run-root "$RUN_ROOT" \
      --model "$MODEL_OUT" \
      --model-backend dreamer_full \
      --output "$output" \
      --top-k 64 \
      --seed "$SEED" \
      --scoring-mode "$mode" \
      --rollout-horizon 5 \
      --allowed-trajectories "$DATA/${split}_trajectories.jsonl" \
      --standardized-steps "$ROOT/data/agentdojo_full_llama31_70b/steps.jsonl" \
      --max-per-user-task 0 \
      --exclude-world-model-from-baselines \
      --utility-score-key preservation_score \
      --include-candidates \
      >"$SEED_ARCHIVE/${split}_${mode}_stdout.json"
  done
done

cp "$MODEL_OUT/metadata.json" "$SEED_ARCHIVE/model_metadata.json"
echo "GROUPED_DREAMER_DONE seed=$SEED $(date -Is)"
