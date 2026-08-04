#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
ARCHIVE=/share/guozhix/wmagentattack/0713/grouped_task_confirmation
PROTOCOL="$ROOT/configs/0713_grouped_task_confirmation_protocol.json"
MANIFEST="$ARCHIVE/job_manifest.tsv"

cd "$ROOT"
test -s "$PROTOCOL"
test -s "$ARCHIVE/grouped_task_confirmation_selection.json"
mkdir -p "$ARCHIVE"
touch "$MANIFEST"

for base_seed in 91 97 103 109 115; do
  if awk -F '\t' -v seed="$base_seed" '$1 == seed {found=1} END {exit !found}' "$MANIFEST"; then
    continue
  fi
  job_id=$(sbatch --parsable \
    --export="ALL,BASE_SEED_OVERRIDE=$base_seed" \
    scripts/server/run_llama31_70b_grouped_task_confirmation.sbatch)
  printf '%s\t%s\n' "$base_seed" "$job_id" >>"$MANIFEST"
  echo "GROUPED_TASK_CONFIRMATION_SUBMITTED base_seed=$base_seed job_id=$job_id"
done

