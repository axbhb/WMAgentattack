#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
ARCHIVE=/share/guozhix/wmagentattack/0712/within_task_contrast
PROTOCOL="$ROOT/configs/0712_pair_heldout_shrinkage_protocol.json"
MANIFEST="$ARCHIVE/pair_heldout_job_manifest.tsv"

cd "$ROOT"
test -s "$PROTOCOL"
touch "$MANIFEST"
target_seed_count=4
if [[ "${PAIR_HELDOUT_CONTINUATION:-0}" == "1" ]]; then
  target_seed_count=5
fi
for base_seed in 51 57 63 69 75; do
  submitted_seed_count=$(awk -F '\t' '$1 ~ /^[0-9]+$/ {count++} END {print count+0}' "$MANIFEST")
  if (( submitted_seed_count >= target_seed_count )); then
    break
  fi
  if awk -F '\t' -v seed="$base_seed" '$1 == seed {found=1} END {exit !found}' "$MANIFEST"; then
    continue
  fi
  job_id=$(sbatch --parsable \
    --export="ALL,BASE_SEED_OVERRIDE=$base_seed" \
    scripts/server/run_llama31_70b_within_task_external_test.sbatch)
  printf '%s\t%s\n' "$base_seed" "$job_id" >>"$MANIFEST"
  echo "PAIR_HELDOUT_SUBMITTED base_seed=$base_seed job_id=$job_id"
done

if [[ "${PAIR_HELDOUT_CONTINUATION:-0}" != "1" ]] \
  && ! awk -F '\t' '$1 == "continuation" {found=1} END {exit !found}' "$MANIFEST"; then
  first_job=$(awk -F '\t' '$1 == 51 {print $2; exit}' "$MANIFEST")
  test -n "$first_job"
  continuation_id=$(sbatch --parsable \
    --dependency="afterok:$first_job" \
    --partition=6000ada \
    --cpus-per-task=1 \
    --mem=1G \
    --time=00:15:00 \
    --job-name=wma-external-submit \
    --output="$ROOT/logs/wma-external-submit-%j.out" \
    --error="$ROOT/logs/wma-external-submit-%j.err" \
    --wrap="PAIR_HELDOUT_CONTINUATION=1 bash scripts/server/submit_pair_heldout_shrinkage_validation.sh")
  printf 'continuation\t%s\n' "$continuation_id" >>"$MANIFEST"
  echo "PAIR_HELDOUT_CONTINUATION_SUBMITTED job_id=$continuation_id dependency=$first_job"
fi
