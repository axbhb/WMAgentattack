#!/usr/bin/env bash
#SBATCH --job-name=wma-ms-v3-lsum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/share/guozhix/WMagentattack-multisource-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-multisource-aug5/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-multisource-aug5
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0806/multisource_semantic_v3/large_v1
PROTOCOL=$ARCHIVE/frozen_protocol.json

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
"$PY" scripts/156_summarize_multisource_build.py \
  --protocol "$PROTOCOL" \
  --archive "$ARCHIVE" \
  --scale large \
  --dataset "$ARCHIVE/multisource_large_records.jsonl" \
  --output "$ARCHIVE/large_gate.json"

(cd "$ARCHIVE" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "MULTISOURCE_LARGE_SUMMARY_DONE $(date -Is)"
