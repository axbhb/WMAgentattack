#!/usr/bin/env bash
#SBATCH --job-name=wma-v2-resid-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

: "${TARGET_KIND:?TARGET_KIND is required}"

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/sheeprl_env/bin/python
RUN_TAG=${RUN_TAG:-residual_preservation_20260716_v1}
ARCHIVE=/share/guozhix/wmagentattack/0716/residual_preservation/$RUN_TAG
DREAMER=/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1/final_summary.json
E5=/share/guozhix/wmagentattack/0715/semantic_value_probe/frozen_e5_structured_oof_components_20260716_v2/summary.json
DUAL=/share/guozhix/wmagentattack/0715/dual_component_value/dual_component_pointwise_frozen_oof_20260716_v1/summary.json

cd "$ROOT"
mkdir -p "$ARCHIVE/frozen/$TARGET_KIND" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

FOLDS=()
for FOLD in 0 1 2 3 4; do
  RESULT="$ARCHIVE/frozen/$TARGET_KIND/fold${FOLD}/result.json"
  test -s "$RESULT"
  FOLDS+=("$RESULT")
done

"$PY" scripts/92_summarize_v2_residual_preservation_folds.py \
  --fold-results "${FOLDS[@]}" \
  --dreamer-summary "$DREAMER" \
  --e5-summary "$E5" \
  --dual-summary "$DUAL" \
  --output "$ARCHIVE/frozen/$TARGET_KIND/summary.json" \
  > "$ARCHIVE/frozen/$TARGET_KIND/summary_stdout.json"

echo "AGENTDOJO_V2_RESIDUAL_SUMMARY_DONE tag=$RUN_TAG target=$TARGET_KIND $(date -Is)"
