#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-dv3-risk
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
DATA=/share/guozhix/wmagentattack/0714/agentdojo_v2/final_dataset
RUN_TAG=${RUN_TAG:-formal}
EPOCHS=${EPOCHS:-30}
ARCHIVE=/share/guozhix/wmagentattack/0715/dreamer_v2_final/$RUN_TAG
VARIANTS=(binary_risk soft_risk binary_risk soft_risk binary_risk soft_risk)
SEEDS=(7 7 13 13 21 21)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=$ARCHIVE/$VARIANT/seed${SEED}
MODEL_OUT=$OUT/model

cd "$ROOT"
mkdir -p "$OUT" logs
test -s "$DATA/train_steps.jsonl"
test -s "$DATA/val_steps.jsonl"
test -s "$DATA/test_steps.jsonl"
(
  cd "$DATA"
  grep -E '  (train|val|test)_steps\.jsonl$' checksums.sha256 | sha256sum -c -
)
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

RISK_ARGS=(
  --binary-risk-loss-scale 1.0
  --soft-risk-loss-scale 0.0
  --risk-reward-binary-mix 1.0
)
if [[ "$VARIANT" == "soft_risk" ]]; then
  RISK_ARGS=(
    --binary-risk-loss-scale 0.0
    --soft-risk-loss-scale 1.0
    --risk-reward-binary-mix 0.0
  )
fi

PYTHONWARNINGS=ignore "$PY" scripts/23_train_full_dreamer_v3.py \
  --train "$DATA/train_steps.jsonl" \
  --val "$DATA/val_steps.jsonl" \
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
  --utility-reward-scale 1.0 \
  --utility-reward-binary-mix 0.0 \
  --target-skill-reward-scale 0.25 \
  --risk-loss-scale 1.0 \
  "${RISK_ARGS[@]}" \
  --binary-utility-loss-scale 0.0 \
  --soft-utility-loss-scale 1.0 \
  --utility-ranking-loss-scale 0.1 \
  --utility-ranking-margin 0.2 \
  --utility-ranking-detach-latent \
  --ranking-pairs-per-batch 1 \
  --preservation-loss-scale 1.0 \
  --validation-risk-mode continuous \
  --validation-utility-mode continuous \
  --validation-aggregation multiseed_group \
  --seed "$SEED" \
  --device cuda \
  > "$OUT/training_stdout.json"

for SPLIT in val test; do
  PYTHONWARNINGS=ignore "$PY" scripts/24_eval_full_dreamer_v3.py \
    --test "$DATA/${SPLIT}_steps.jsonl" \
    --model "$MODEL_OUT" \
    --output "$OUT/${SPLIT}_metrics.json" \
    > "$OUT/${SPLIT}_metrics_stdout.json"
done

echo "AGENTDOJO_V2_DREAMER_DONE tag=$RUN_TAG variant=$VARIANT seed=$SEED $(date -Is)"
bash scripts/server/finalize_agentdojo_v2_dreamer_if_complete.sh "$RUN_TAG"
