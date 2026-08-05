#!/usr/bin/env bash
#SBATCH --job-name=wma-ms-v3-reaudit
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=2-17%8
#SBATCH --output=/share/guozhix/WMagentattack-multisource-aug5/logs/%x-%A_%a.out
#SBATCH --error=/share/guozhix/WMagentattack-multisource-aug5/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-multisource-aug5
PY=/share/guozhix/software/venvs/wmagentattack_multisource_aug5/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0806/multisource_semantic_v3/large_v1
CHUNK=$((SLURM_ARRAY_TASK_ID - 2))

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"$PY" scripts/157_reaudit_multisource_shard.py \
  --protocol "$ARCHIVE/frozen_protocol.json" \
  --manifest "$ARCHIVE/manifests/injecagent.json" \
  --output "$ARCHIVE/outputs/injecagent/chunk${CHUNK}.json" \
  --original-audit "$ARCHIVE/audits/injecagent/chunk${CHUNK}.json" \
  --recovery-audit "$ARCHIVE/recovery_audits/injecagent/chunk${CHUNK}.json" \
  --chunk-index "$CHUNK" \
  --num-chunks 16

echo "MULTISOURCE_SHARD_REAUDIT_DONE source=injecagent chunk=$CHUNK/16 $(date -Is)"
