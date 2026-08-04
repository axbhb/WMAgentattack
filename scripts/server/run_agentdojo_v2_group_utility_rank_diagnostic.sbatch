#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-gu-rankdiag
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
SOURCE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1
RUN_TAG=${RUN_TAG:-seedrep_20260715_group_utility_head_only_v1}
ARCHIVE=/share/guozhix/wmagentattack/0715/grouped_continuous_utility/$RUN_TAG
FOLD=${FOLD:-1}
DATA=$SOURCE/data/fold${FOLD}/full
BASELINE_ROOT=$SOURCE/models/fold${FOLD}/pct100
VARIANT_ROOT=$ARCHIVE/models/fold${FOLD}/group_utility_head_only

cd "$ROOT"
mkdir -p "$ARCHIVE" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

for REQUIRED in \
  "$ARCHIVE/baseline_downstream.json" \
  "$ARCHIVE/variant_downstream.json" \
  "$DATA/val_steps.jsonl" \
  "$DATA/test_steps.jsonl"; do
  test -s "$REQUIRED"
done

PYTHONWARNINGS=ignore "$PY" scripts/83_diagnose_v2_group_utility_rank_stability.py \
  --data-root "$DATA" \
  --baseline-model-root "$BASELINE_ROOT" \
  --variant-model-root "$VARIANT_ROOT" \
  --baseline-downstream "$ARCHIVE/baseline_downstream.json" \
  --variant-downstream "$ARCHIVE/variant_downstream.json" \
  --decision-step first \
  --seeds 7 13 21 \
  --output "$ARCHIVE/rank_stability_diagnostic.json" \
  --candidate-output "$ARCHIVE/rank_stability_candidate_scores.json" \
  > "$ARCHIVE/rank_stability_diagnostic_stdout.json"

echo "AGENTDOJO_V2_GROUP_UTILITY_RANK_DIAGNOSTIC_DONE tag=$RUN_TAG fold=$FOLD $(date -Is)"
