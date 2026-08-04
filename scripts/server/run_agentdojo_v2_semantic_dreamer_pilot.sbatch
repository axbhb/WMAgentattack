#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-sem-dreamer
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
SOURCE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1
RUN_TAG=${RUN_TAG:-semantic_dreamer_smoke5_20260716_v1}
ARCHIVE=/share/guozhix/wmagentattack/0715/semantic_dreamer/$RUN_TAG
FOLD=${FOLD:-1}
SEED=${SEED:-7}
EPOCHS=${EPOCHS:-5}
OBS_DIM=${OBS_DIM:-800}
DATA=$SOURCE/data/fold${FOLD}/full
TRAIN=$DATA/train_steps.jsonl
VAL=$DATA/val_steps.jsonl
TEST=$DATA/test_steps.jsonl
FEATURE=${FEATURE_PATH:-$ARCHIVE/features/fold${FOLD}/e5_structured_observations.npz}
MODEL_ROOT=$ARCHIVE/models/fold${FOLD}/semantic_head_only
OUT=$MODEL_ROOT/seed${SEED}
MODEL_OUT=$OUT/model
HF_CACHE=/share/guozhix/hf_cache

cd "$ROOT"
mkdir -p "$(dirname "$FEATURE")" "$OUT" "$HF_CACHE" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

FEATURE_EXTRA_ARGS=()
if [[ "${INCLUDE_HASH:-0}" == "1" ]]; then
  FEATURE_EXTRA_ARGS+=(--include-hash)
fi

for REQUIRED in "$TRAIN" "$VAL" "$TEST"; do
  test -s "$REQUIRED"
done

if [[ ! -s "$FEATURE" ]]; then
  PYTHONWARNINGS=ignore "$PY" scripts/89_build_e5_observation_cache.py \
    --steps "$TRAIN" "$VAL" "$TEST" \
    --model-name intfloat/e5-base-v2 \
    --cache-dir "$HF_CACHE" \
    --embedding-batch-size 32 \
    --structured-dim 32 \
    "${FEATURE_EXTRA_ARGS[@]}" \
    --output "$FEATURE" \
    > "$ARCHIVE/feature_build_stdout.json"
fi

PYTHONWARNINGS=ignore "$PY" scripts/23_train_full_dreamer_v3.py \
  --train "$TRAIN" \
  --val "$VAL" \
  --model-out "$MODEL_OUT" \
  --epochs "$EPOCHS" \
  --batch-size 16 \
  --obs-dim "$OBS_DIM" \
  --observation-feature-mode precomputed \
  --observation-feature-path "$FEATURE" \
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
  --binary-risk-loss-scale 1.0 \
  --soft-risk-loss-scale 0.0 \
  --risk-reward-binary-mix 1.0 \
  --utility-reward-scale 1.0 \
  --utility-reward-binary-mix 0.0 \
  --target-skill-reward-scale 0.25 \
  --risk-loss-scale 1.0 \
  --binary-utility-loss-scale 0.0 \
  --soft-utility-loss-scale 1.0 \
  --utility-ranking-loss-scale 0.0 \
  --ranking-pairs-per-batch 0 \
  --group-utility-calibration-loss-scale 1.0 \
  --group-utility-ranking-loss-scale 0.5 \
  --group-utility-min-target-gap 0.1 \
  --group-utility-pairs-per-task 8 \
  --grouped-utility-batches \
  --group-utility-calibration-detach-latent \
  --group-utility-ranking-detach-latent \
  --group-utility-head-only-updates \
  --preservation-loss-scale 1.0 \
  --validation-risk-mode continuous \
  --validation-utility-mode continuous \
  --validation-aggregation multiseed_group \
  --validation-group-step first \
  --seed "$SEED" \
  --device cuda \
  > "$OUT/training_stdout.json"

for SPLIT in val test; do
  EVAL_FILE=$VAL
  if [[ "$SPLIT" == "test" ]]; then
    EVAL_FILE=$TEST
  fi
  PYTHONWARNINGS=ignore "$PY" scripts/24_eval_full_dreamer_v3.py \
    --test "$EVAL_FILE" \
    --model "$MODEL_OUT" \
    --output "$OUT/${SPLIT}_metrics.json" \
    --validation-group-step first \
    > "$OUT/${SPLIT}_metrics_stdout.json"
done

PYTHONWARNINGS=ignore "$PY" scripts/70_evaluate_v2_downstream_selection.py \
  --data-root "$DATA" \
  --model-root "$MODEL_ROOT" \
  --calibration-mode fit_decision_time \
  --calibration-train "$TRAIN" \
  --decision-step first \
  --seeds "$SEED" \
  --budgets 1 2 4 \
  --bootstrap-draws 10000 \
  --bootstrap-seed 20260716 \
  --output "$ARCHIVE/downstream_selection.json" \
  > "$ARCHIVE/downstream_selection_stdout.json"

echo "AGENTDOJO_V2_SEMANTIC_DREAMER_DONE tag=$RUN_TAG fold=$FOLD seed=$SEED epochs=$EPOCHS $(date -Is)"
