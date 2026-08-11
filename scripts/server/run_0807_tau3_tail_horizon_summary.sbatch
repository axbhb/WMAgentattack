#!/usr/bin/env bash
#SBATCH --job-name=wma-tau3-tail-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.err

set -euo pipefail
ROOT=/share/guozhix/WMagentattack-tau3-interactive-aug5
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0807/tau3_tail_horizon/pilot_v1
PARENT=/share/guozhix/wmagentattack/0806/tau3_horizon_extension/confirmation_v1/outputs
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"$PY" scripts/171_summarize_tau3_horizon_pilot.py \
  --protocol "$ARCHIVE/frozen_protocol.json" --manifest "$ARCHIVE/manifest.json" \
  --archive "$ARCHIVE" --parent-output-dir "$PARENT" \
  --dataset "$ARCHIVE/tau3_tail_dataset.json" --audit "$ARCHIVE/dataset_audit.json" \
  --output "$ARCHIVE/base_gate.json" --report "$ARCHIVE/base_results.md"
"$PY" scripts/176_finalize_tau3_tail_horizon.py \
  --protocol "$ARCHIVE/frozen_protocol.json" --base-gate "$ARCHIVE/base_gate.json" \
  --dataset "$ARCHIVE/tau3_tail_dataset.json" --output-dir "$ARCHIVE/outputs" \
  --output "$ARCHIVE/tail_gate.json" --report "$ARCHIVE/tail_results.md"
(cd "$ARCHIVE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "TAU3_TAIL_SUMMARY_DONE $(date -Is)"
