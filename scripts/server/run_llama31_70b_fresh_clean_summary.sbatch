#!/usr/bin/env bash
#SBATCH --job-name=wma-fresh-clean-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
RUN_TAG=${RUN_TAG:-fresh_task_screen_20260716_v1}
RUN_SEED=${RUN_SEED:-101}
ARCHIVE=/share/guozhix/wmagentattack/0716/fresh_task_screen/$RUN_TAG

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
CHUNKS=()
for INDEX in 0 1 2 3; do
  CHUNK="$ARCHIVE/seed${RUN_SEED}/chunk${INDEX}.json"
  test -s "$CHUNK"
  CHUNKS+=("$CHUNK")
done
"$PY" scripts/100_summarize_fresh_clean_screen.py \
  --manifest "$ARCHIVE/manifest.json" \
  --chunks "${CHUNKS[@]}" \
  --output "$ARCHIVE/seed${RUN_SEED}/summary.json" \
  > "$ARCHIVE/seed${RUN_SEED}/summary_stdout.json"

echo "AGENTDOJO_FRESH_CLEAN_SUMMARY_DONE tag=$RUN_TAG seed=$RUN_SEED $(date -Is)"
