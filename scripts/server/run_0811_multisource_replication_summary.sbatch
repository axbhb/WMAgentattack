#!/usr/bin/env bash
#SBATCH --job-name=wma-ms-repl-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.err

set -euo pipefail
ROOT=/share/guozhix/WMagentattack-tau3-interactive-aug5
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0811/multisource_replication/auxiliary_v1
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"$PY" scripts/178_summarize_multisource_replication.py \
  --protocol "$ARCHIVE/frozen_protocol.json" --archive "$ARCHIVE" \
  --output "$ARCHIVE/replication_gate.json"
(cd "$ARCHIVE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "MULTISOURCE_REPLICATION_SUMMARY_DONE $(date -Is)"
