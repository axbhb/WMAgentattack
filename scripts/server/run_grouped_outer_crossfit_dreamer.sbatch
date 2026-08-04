#!/usr/bin/env bash
#SBATCH --job-name=wma-outer-cf-dv3
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0713/grouped_outer_crossfit
FOLDS="$ARCHIVE/folds"
RUN_ROOT="$ROOT/runs/agentdojo_full_hf/meta-llama-3.1-70b-instruct-transformers-4bit-compact12000-base-function_tags-ctx8192-local"
MODEL_ROOT="$ROOT/artifacts/grouped_outer_crossfit_dreamer"
PROTOCOL="$ROOT/configs/0713_grouped_outer_crossfit_protocol.json"
SEEDS=(7 13 21)
FOLD=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SEED="${SEEDS[$SEED_INDEX]}"
FOLD_DIR="$FOLDS/fold${FOLD}"
OUT_DIR="$ARCHIVE/scores/fold${FOLD}/seed${SEED}"
MODEL_OUT="$MODEL_ROOT/fold${FOLD}_seed${SEED}"

cd "$ROOT"
mkdir -p "$OUT_DIR" "$MODEL_ROOT" logs
test -s "$PROTOCOL"
EXPECTED_SHA=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["folds"]["manifest_sha256"])' "$PROTOCOL")
test "$(sha256sum "$FOLDS/fold_manifest.json" | awk '{print $1}')" = "$EXPECTED_SHA"
for file in train_steps val_steps held_steps held_trajectories; do
  test -s "$FOLD_DIR/${file}.jsonl"
done
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PYTHONWARNINGS=ignore "$PY" scripts/23_train_full_dreamer_v3.py \
  --train "$FOLD_DIR/train_steps.jsonl" \
  --val "$FOLD_DIR/val_steps.jsonl" \
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
  >"$OUT_DIR/training_stdout.json"

PYTHONWARNINGS=ignore "$PY" scripts/24_eval_full_dreamer_v3.py \
  --test "$FOLD_DIR/held_steps.jsonl" \
  --model "$MODEL_OUT" \
  --output "$OUT_DIR/held_metrics.json" \
  >"$OUT_DIR/held_metrics_stdout.json"

for mode in clean_prefix_rollout injection_conditioned_rollout; do
  output="$OUT_DIR/held_${mode}_candidates.json"
  PYTHONWARNINGS=ignore "$PY" scripts/11_select_world_model_agentdojo_pairs.py \
    --run-root "$RUN_ROOT" \
    --model "$MODEL_OUT" \
    --model-backend dreamer_full \
    --output "$output" \
    --top-k 64 \
    --seed "$SEED" \
    --scoring-mode "$mode" \
    --rollout-horizon 5 \
    --allowed-trajectories "$FOLD_DIR/held_trajectories.jsonl" \
    --standardized-steps "$ROOT/data/agentdojo_full_llama31_70b/steps.jsonl" \
    --max-per-user-task 0 \
    --exclude-world-model-from-baselines \
    --utility-score-key preservation_score \
    --include-candidates \
    >"$OUT_DIR/held_${mode}_stdout.json"
  sha256sum "$output" >"$output.sha256"
done

cp "$MODEL_OUT/metadata.json" "$OUT_DIR/model_metadata.json"
touch "$OUT_DIR/DONE"
echo "GROUPED_OUTER_CROSSFIT_DONE fold=$FOLD seed=$SEED $(date -Is)"
bash scripts/server/finalize_grouped_outer_crossfit_if_complete.sh
