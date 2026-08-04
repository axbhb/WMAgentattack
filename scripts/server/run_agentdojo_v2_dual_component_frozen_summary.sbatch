#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-dual-summary
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

: "${FOLD_JOB_ID:?FOLD_JOB_ID is required}"
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
RUN_TAG=${RUN_TAG:-dual_component_pointwise_frozen_oof_20260716_v1}
ARCHIVE=/share/guozhix/wmagentattack/0715/dual_component_value/$RUN_TAG
BASELINE=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1/final_summary.json

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

RESULTS=()
for FOLD in 0 1 2 3 4; do
  RESULT="$ARCHIVE/fold${FOLD}/result.json"
  test -s "$RESULT"
  RESULTS+=("$RESULT")
done

"$PY" scripts/88_summarize_v2_dual_component_folds.py \
  --fold-results "${RESULTS[@]}" \
  --baseline-summary "$BASELINE" \
  --bootstrap-samples 200000 \
  --bootstrap-seed 20260716 \
  --output "$ARCHIVE/summary.json" \
  > "$ARCHIVE/summary_stdout.json"

echo "AGENTDOJO_V2_DUAL_COMPONENT_FROZEN_SUMMARY_DONE tag=$RUN_TAG fold_job=$FOLD_JOB_ID $(date -Is)"
