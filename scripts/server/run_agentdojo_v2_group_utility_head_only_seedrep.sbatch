#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-gu-seedrep
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
SOURCE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1
RUN_TAG=${RUN_TAG:-seedrep_20260715_group_utility_head_only_v1}
ARCHIVE=/share/guozhix/wmagentattack/0715/grouped_continuous_utility/$RUN_TAG
FOLD=${FOLD:-1}
EPOCHS=${EPOCHS:-30}
SEEDS=(13 21)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DATA=$SOURCE/data/fold${FOLD}/full
TRAIN=$DATA/train_steps.jsonl
VAL=$DATA/val_steps.jsonl
TEST=$DATA/test_steps.jsonl
OUT=$ARCHIVE/models/fold${FOLD}/group_utility_head_only/seed${SEED}
MODEL_OUT=$OUT/model

cd "$ROOT"
mkdir -p "$OUT" logs
for PATH_TO_CHECK in "$TRAIN" "$VAL" "$TEST"; do
  test -s "$PATH_TO_CHECK"
done
(
  cd "$SOURCE/data/fold${FOLD}"
  sha256sum -c checksums.sha256
)
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PYTHONWARNINGS=ignore "$PY" scripts/23_train_full_dreamer_v3.py \
  --train "$TRAIN" \
  --val "$VAL" \
  --model-out "$MODEL_OUT" \
  --epochs "$EPOCHS" \
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

echo "AGENTDOJO_V2_GROUP_UTILITY_HEAD_ONLY_SEEDREP_DONE tag=$RUN_TAG fold=$FOLD seed=$SEED epochs=$EPOCHS $(date -Is)"
