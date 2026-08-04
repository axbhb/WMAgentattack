#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-toprank-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

: "${PAIR_SCHEME:?PAIR_SCHEME is required}"

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
RUN_TAG=${RUN_TAG:-top_heavy_ranking_20260716_v1}
ARCHIVE=/share/guozhix/wmagentattack/0716/top_heavy_ranking/$RUN_TAG
DREAMER=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1/final_summary.json
E5=/share/guozhix/wmagentattack/0715/semantic_value_probe/frozen_e5_structured_oof_components_20260716_v2/summary.json

cd "$ROOT"
mkdir -p "$ARCHIVE/frozen/$PAIR_SCHEME" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

FOLDS=()
for FOLD in 0 1 2 3 4; do
  RESULT="$ARCHIVE/frozen/$PAIR_SCHEME/fold${FOLD}/result.json"
  test -s "$RESULT"
  FOLDS+=("$RESULT")
done

"$PY" scripts/95_summarize_v2_top_heavy_ranking.py \
  --fold-results "${FOLDS[@]}" \
  --dreamer-summary "$DREAMER" \
  --e5-summary "$E5" \
  --output "$ARCHIVE/frozen/$PAIR_SCHEME/summary.json" \
  > "$ARCHIVE/frozen/$PAIR_SCHEME/summary_stdout.json"

echo "AGENTDOJO_V2_TOP_HEAVY_SUMMARY_DONE tag=$RUN_TAG scheme=$PAIR_SCHEME $(date -Is)"
