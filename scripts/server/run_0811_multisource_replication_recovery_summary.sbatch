#!/usr/bin/env bash
#SBATCH --job-name=wma-ms-recover-sum
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.err

set -euo pipefail
ROOT=/share/guozhix/WMagentattack-tau3-interactive-aug5
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0811/multisource_replication/auxiliary_v1
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"$PY" scripts/180_finalize_multisource_replication_recovery.py \
  --protocol "$ARCHIVE/frozen_protocol.json" \
  --finalize-protocol configs/0811_multisource_replication_recovery_finalize.json \
  --archive "$ARCHIVE" \
  --tool-recovery-audit "$ARCHIVE/recovery/tool_sandbox_semantic_replica_audit.json" \
  --output "$ARCHIVE/replication_gate_recovery.json"
echo "MULTISOURCE_REPLICATION_RECOVERY_SUMMARY_DONE $(date -Is)"
