#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-config-value
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
BASELINE_ARCHIVE=/share/guozhix/wmagentattack/0715/grouped_continuous_utility/seedrep_20260715_group_utility_head_only_v1
RUN_TAG=${RUN_TAG:-pilot5_20260715_configuration_value_v1}
ARCHIVE=/share/guozhix/wmagentattack/0715/configuration_value/$RUN_TAG
FOLD=${FOLD:-1}
SEED=${SEED:-7}
EPOCHS=${EPOCHS:-5}
DATA=$SOURCE/data/fold${FOLD}/full
TRAIN=$DATA/train_steps.jsonl
VAL=$DATA/val_steps.jsonl
TEST=$DATA/test_steps.jsonl
OUT=$ARCHIVE/models/fold${FOLD}/configuration_value/seed${SEED}
MODEL_OUT=$OUT/model

cd "$ROOT"
mkdir -p "$OUT" logs
for REQUIRED in \
  "$TRAIN" \
  "$VAL" \
  "$TEST" \
  "$BASELINE_ARCHIVE/baseline_downstream.json"; do
  test -s "$REQUIRED"
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
  --configuration-value-head \
  --group-value-calibration-loss-scale 1.0 \
  --group-value-ranking-loss-scale 0.5 \
  --group-value-min-target-gap 0.1 \
  --group-value-pairs-per-task 8 \
  --group-value-head-only-updates \
  --preservation-loss-scale 1.0 \
  --validation-risk-mode continuous \
  --validation-utility-mode continuous \
  --validation-aggregation multiseed_group \
  --validation-group-step first \
  --checkpoint-objective grouped_configuration_value_brier \
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

PYTHONWARNINGS=ignore "$PY" scripts/84_evaluate_v2_configuration_value.py \
  --data-root "$DATA" \
  --model-root "$ARCHIVE/models/fold${FOLD}/configuration_value" \
  --baseline-downstream "$BASELINE_ARCHIVE/baseline_downstream.json" \
  --decision-step first \
  --seeds "$SEED" \
  --budgets 1 2 4 \
  --bootstrap-draws 10000 \
  --output "$ARCHIVE/direct_value_selection.json" \
  > "$ARCHIVE/direct_value_selection_stdout.json"

echo "AGENTDOJO_V2_CONFIGURATION_VALUE_DONE tag=$RUN_TAG fold=$FOLD seed=$SEED epochs=$EPOCHS $(date -Is)"
