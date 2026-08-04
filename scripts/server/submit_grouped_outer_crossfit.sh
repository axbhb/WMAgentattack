#!/usr/bin/env bash

set -euo pipefail

ROOT=/share/guozhix/WMagentattack
ARCHIVE=/share/guozhix/wmagentattack/0713/grouped_outer_crossfit
MANIFEST="$ARCHIVE/job_manifest.tsv"

cd "$ROOT"
test -s configs/0713_grouped_outer_crossfit_protocol.json
test -s "$ARCHIVE/folds/fold_manifest.json"
mkdir -p "$ARCHIVE"
if [[ -s "$MANIFEST" ]]; then
  echo "GROUPED_OUTER_CROSSFIT_ALREADY_SUBMITTED"
  cat "$MANIFEST"
  exit 0
fi
job_id=$(sbatch --parsable scripts/server/run_grouped_outer_crossfit_dreamer.sbatch)
printf 'outer_crossfit\t%s\n' "$job_id" >"$MANIFEST"
echo "GROUPED_OUTER_CROSSFIT_SUBMITTED job_id=$job_id"
