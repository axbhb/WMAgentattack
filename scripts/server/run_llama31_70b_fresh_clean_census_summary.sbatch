#!/usr/bin/env bash
#SBATCH --job-name=wma-fresh-census-sum
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
ARCHIVE=/share/guozhix/wmagentattack/0716/fresh_task_screen/$RUN_TAG

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
for SEED in 101 103 107; do
  for CHUNK in 0 1 2 3; do
    test -s "$ARCHIVE/seed${SEED}/chunk${CHUNK}.json"
  done
done
"$PY" scripts/102_summarize_fresh_clean_multiseed.py \
  --manifest "$ARCHIVE/manifest.json" \
  --archive-root "$ARCHIVE" \
  --output "$ARCHIVE/clean_multiseed_summary.json" \
  > "$ARCHIVE/clean_multiseed_summary_stdout.json"

echo "AGENTDOJO_FRESH_CENSUS_SUMMARY_DONE tag=$RUN_TAG $(date -Is)"
