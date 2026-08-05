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
SOURCE_ROOT=$ROOT/external/injecagent
OUTPUT=$ARCHIVE/outputs/injecagent/chunk${CHUNK}.json
ORIGINAL_AUDIT=$ARCHIVE/audits/injecagent/chunk${CHUNK}.json

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# One worker may finish all frozen completions but be interrupted before its
# audit is emitted. Reparse that immutable output without loading the LLM and
# write to a separate recovery path; never overwrite the original shard.
if [[ ! -s "$ORIGINAL_AUDIT" ]]; then
  RECOVERY_OUTPUT=$ARCHIVE/recovery_outputs/injecagent/chunk${CHUNK}.json
  RECOVERY_ORIGINAL_AUDIT=$ARCHIVE/recovery_original_audits/injecagent/chunk${CHUNK}.json
  mkdir -p "$(dirname "$RECOVERY_OUTPUT")" "$(dirname "$RECOVERY_ORIGINAL_AUDIT")"
  set +e
  "$PY" scripts/155_run_multisource_llm.py \
    --protocol "$ARCHIVE/frozen_protocol.json" \
    --manifest "$ARCHIVE/manifests/injecagent.json" \
    --source-root "$SOURCE_ROOT" \
    --chunk-index "$CHUNK" \
    --num-chunks 16 \
    --cached-input "$OUTPUT" \
    --output "$RECOVERY_OUTPUT" \
    --audit "$RECOVERY_ORIGINAL_AUDIT"
  REPLAY_STATUS=$?
  set -e
  test "$REPLAY_STATUS" -eq 1
  OUTPUT=$RECOVERY_OUTPUT
  ORIGINAL_AUDIT=$RECOVERY_ORIGINAL_AUDIT
fi

"$PY" scripts/157_reaudit_multisource_shard.py \
  --protocol "$ARCHIVE/frozen_protocol.json" \
  --manifest "$ARCHIVE/manifests/injecagent.json" \
  --output "$OUTPUT" \
  --original-audit "$ORIGINAL_AUDIT" \
  --recovery-audit "$ARCHIVE/recovery_audits/injecagent/chunk${CHUNK}.json" \
  --chunk-index "$CHUNK" \
  --num-chunks 16

echo "MULTISOURCE_SHARD_REAUDIT_DONE source=injecagent chunk=$CHUNK/16 $(date -Is)"
