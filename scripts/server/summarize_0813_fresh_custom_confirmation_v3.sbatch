#!/usr/bin/env bash
#SBATCH --job-name=wma-fresh-v3-gate
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/0813_fresh_v3_gate_%j.out
#SBATCH --error=logs/0813_fresh_v3_gate_%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-fresh-validation-aug13
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0813/fresh_integrated_validation/formal_v1
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
(cd "$ARCHIVE/frozen_inputs" && sha256sum --quiet -c SHA256SUMS)
"$PY" scripts/193_summarize_fresh_custom_confirmation_v3.py \
  --protocol "$ARCHIVE/frozen_inputs/protocol.json" \
  --manifest "$ARCHIVE/frozen_inputs/manifest.json" \
  --result-root "$ARCHIVE/clean_runs" \
  --output-dir "$ARCHIVE/fresh_dataset"

date --iso-8601=seconds > "$ARCHIVE/fresh_dataset/finished_at.txt"
(cd "$ARCHIVE/fresh_dataset" && find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256)
echo "FRESH_CUSTOM_CONFIRMATION_V3_GATE_DONE $(date -Is)"
