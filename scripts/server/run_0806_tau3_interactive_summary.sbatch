#!/usr/bin/env bash
#SBATCH --job-name=wma-tau3-int-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-tau3-interactive-aug5
SOURCE_ROOT=/share/guozhix/WMagentattack-multisource-aug5/external/tau3
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0806/tau3_interaction_faithful_repair/pilot_v1
PARENT=/share/guozhix/wmagentattack/0806/tau3_multistep_scale_readiness/pilot_v1/tau3_multistep_dataset.json
PROTOCOL=$ARCHIVE/frozen_protocol.json
MANIFEST=$ARCHIVE/manifest.json

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$SOURCE_ROOT/src:${PYTHONPATH:-}"
"$PY" scripts/169_summarize_tau3_interactive_pilot.py \
  --protocol "$PROTOCOL" \
  --manifest "$MANIFEST" \
  --archive "$ARCHIVE" \
  --parent-dataset "$PARENT" \
  --dataset "$ARCHIVE/tau3_interactive_dataset.json" \
  --audit "$ARCHIVE/dataset_audit.json" \
  --output "$ARCHIVE/interaction_gate.json" \
  --report "$ARCHIVE/interaction_results.md"

(cd "$ARCHIVE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "TAU3_INTERACTIVE_SUMMARY_DONE $(date -Is)"
