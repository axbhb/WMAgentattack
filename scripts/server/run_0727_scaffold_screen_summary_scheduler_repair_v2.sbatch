#!/usr/bin/env bash
#SBATCH --job-name=wma-scaffold-v2-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/share/guozhix/WMagentattack/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0727/scaffold_screen/scheduler_repair_v2

cd "$ROOT"
test -s "$ARCHIVE/frozen_repo_SHA256SUMS"
sha256sum --quiet -c "$ARCHIVE/frozen_repo_SHA256SUMS"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
"$PY" scripts/133_summarize_scaffold_screen.py \
  --protocol "$ARCHIVE/frozen_protocol.json" \
  --manifest "$ARCHIVE/manifest.json" \
  --archive-root "$ARCHIVE" \
  --output "$ARCHIVE/screening_summary.json" \
  > "$ARCHIVE/screening_summary_stdout.json"
sha256sum "$ARCHIVE/screening_summary.json" > "$ARCHIVE/result_SHA256SUMS"

echo "SCAFFOLD_SCREEN_V2_SUMMARY_DONE $(date -Is)"
