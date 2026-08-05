#!/usr/bin/env bash
#SBATCH --job-name=wma-ms-v3-lsum-r
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

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
"$PY" - "$ARCHIVE" <<'PY'
import json
import sys
from pathlib import Path

archive = Path(sys.argv[1])
files = sorted((archive / "recovery_audits" / "injecagent").glob("chunk*.json"))
assert len(files) == 16, f"expected 16 recovery audits, found {len(files)}"
assert all(json.loads(path.read_text(encoding="utf-8"))["passed"] for path in files)
PY
"$PY" scripts/156_summarize_multisource_build.py \
  --protocol "$ARCHIVE/frozen_protocol.json" \
  --archive "$ARCHIVE" \
  --scale large \
  --dataset "$ARCHIVE/multisource_large_records.jsonl" \
  --output "$ARCHIVE/large_gate.json"

(cd "$ARCHIVE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "MULTISOURCE_LARGE_RECOVERED_SUMMARY_DONE $(date -Is)"
