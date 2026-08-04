#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
ARCHIVE=/share/guozhix/wmagentattack/0713/headwise_remaining_confirmation_v2
PROTOCOL="$ROOT/configs/0713_headwise_remaining_confirmation_protocol.json"
MANIFEST="$ARCHIVE/job_manifest.tsv"

cd "$ROOT"
test -s "$PROTOCOL"
test -s "$ARCHIVE/headwise_remaining_confirmation_selection.json"
mkdir -p "$ARCHIVE"
touch "$MANIFEST"

for base_seed in 127 131 137 139 149; do
  if awk -F '\t' -v seed="$base_seed" '$1 == seed {found=1} END {exit !found}' "$MANIFEST"; then
    continue
  fi
  job_id=$(sbatch --parsable \
    --export="ALL,BASE_SEED_OVERRIDE=$base_seed" \
    scripts/server/run_llama31_70b_headwise_remaining_confirmation.sbatch)
  printf '%s\t%s\n' "$base_seed" "$job_id" >>"$MANIFEST"
  echo "HEADWISE_REMAINING_SUBMITTED base_seed=$base_seed job_id=$job_id"
done
