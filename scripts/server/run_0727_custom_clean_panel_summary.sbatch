#!/usr/bin/env bash
#SBATCH --job-name=wma-custom-clean-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0727/custom_clean_panel/fixed_v1

cd "$ROOT"
test -s "$ARCHIVE/frozen_repo_SHA256SUMS"
sha256sum --quiet -c "$ARCHIVE/frozen_repo_SHA256SUMS"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
"$PY" scripts/136_summarize_custom_clean_panel.py \
  --protocol "$ARCHIVE/frozen_protocol.json" \
  --manifest "$ARCHIVE/manifest.json" \
  --archive-root "$ARCHIVE" \
  --output "$ARCHIVE/custom_clean_panel_summary.json" \
  > "$ARCHIVE/custom_clean_panel_summary_stdout.json"
sha256sum "$ARCHIVE/custom_clean_panel_summary.json" > "$ARCHIVE/result_SHA256SUMS"

echo "CUSTOM_CLEAN_PANEL_SUMMARY_DONE $(date -Is)"
