#!/usr/bin/env bash
#SBATCH --job-name=wma-ms-v3-recover
#SBATCH --partition=6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/share/guozhix/WMagentattack-multisource-aug5/logs/%x-%j.out
#SBATCH --error=/share/guozhix/WMagentattack-multisource-aug5/logs/%x-%j.err

set -euo pipefail

ROOT=/share/guozhix/WMagentattack-multisource-aug5
cd "$ROOT"

# InjecAgent occupies original array task IDs 2--17. Run only the label-blind
# release reaudit for those immutable shards, sequentially in one CPU job so
# the repair does not consume another 16-entry Slurm array quota.
for TASK_ID in $(seq 2 17); do
  SLURM_ARRAY_TASK_ID=$TASK_ID \
    bash scripts/server/run_0806_multisource_semantic_v3_reaudit_failed.sbatch
done

bash scripts/server/run_0806_multisource_semantic_v3_large_summary_recovered.sbatch
echo "MULTISOURCE_RECOVERY_AND_SUMMARY_DONE $(date -Is)"
