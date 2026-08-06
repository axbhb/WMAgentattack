#!/usr/bin/env bash
#SBATCH --job-name=wma-tau3-horizon-conf-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-tau3-interactive-aug5
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0806/tau3_horizon_extension/confirmation_v1
PARENT=/share/guozhix/wmagentattack/0806/tau3_interaction_faithful_repair/pilot_v1/outputs
PILOT=/share/guozhix/wmagentattack/0806/tau3_horizon_extension/pilot_v1/outputs

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"$PY" scripts/173_summarize_tau3_horizon_confirmation.py \
  --protocol "$ARCHIVE/frozen_protocol.json" \
  --manifest "$ARCHIVE/manifest.json" \
  --archive "$ARCHIVE" \
  --parent-output-dir "$PARENT" \
  --pilot-output-dir "$PILOT" \
  --dataset "$ARCHIVE/tau3_horizon_confirmation_dataset.json" \
  --audit "$ARCHIVE/dataset_audit.json" \
  --output "$ARCHIVE/confirmation_gate.json" \
  --report "$ARCHIVE/confirmation_results.md"

(cd "$ARCHIVE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "TAU3_HORIZON_CONFIRMATION_SUMMARY_DONE $(date -Is)"
