#!/usr/bin/env bash
#SBATCH --job-name=wma-ts-recover
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-tau3-interactive-aug5/logs/%x-%j.err

set -euo pipefail
ROOT=/share/guozhix/WMagentattack-tau3-interactive-aug5
SOURCE_ROOT=/share/guozhix/WMagentattack-multisource-aug5/external/tool_sandbox
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0811/multisource_replication/auxiliary_v1
cd "$ROOT"
mkdir -p "$ARCHIVE/recovery" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"$PY" scripts/179_reaudit_toolsandbox_runtime_ids.py \
  --recovery-protocol configs/0811_toolsandbox_replica_measurement_recovery.json \
  --parent-protocol "$ARCHIVE/frozen_protocol.json" \
  --manifest "$ARCHIVE/manifests/tool_sandbox.json" \
  --immutable-output "$ARCHIVE/outputs/tool_sandbox.json" \
  --original-audit "$ARCHIVE/audits/tool_sandbox.json" \
  --source-root "$SOURCE_ROOT" \
  --output "$ARCHIVE/recovery/tool_sandbox_semantic_replica_audit.json"
echo "TOOLSANDBOX_REPLICA_RECOVERY_DONE $(date -Is)"
