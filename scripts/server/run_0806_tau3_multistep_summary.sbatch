#!/usr/bin/env bash
#SBATCH --job-name=wma-tau3-ms-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/share/guozhix/WMagentattack-tau3-multistep-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-tau3-multistep-aug5/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-tau3-multistep-aug5
SOURCE_ROOT=/share/guozhix/WMagentattack-multisource-aug5/external/tau3
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0806/tau3_multistep_scale_readiness/pilot_v1
PROTOCOL=$ARCHIVE/frozen_protocol.json
MANIFEST=$ARCHIVE/manifest.json

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$SOURCE_ROOT/src:${PYTHONPATH:-}"
"$PY" scripts/164_summarize_tau3_multistep_pilot.py \
  --protocol "$PROTOCOL" \
  --manifest "$MANIFEST" \
  --archive "$ARCHIVE" \
  --dataset "$ARCHIVE/tau3_multistep_dataset.json" \
  --audit "$ARCHIVE/dataset_audit.json" \
  --output "$ARCHIVE/pilot_gate.json" \
  --report "$ARCHIVE/pilot_results.md"

(cd "$ARCHIVE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "TAU3_MULTISTEP_SUMMARY_DONE $(date -Is)"
