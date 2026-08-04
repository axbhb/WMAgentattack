#!/usr/bin/env bash
set -euo pipefail

RUN_TAG=${1:-formal}
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0715/dreamer_v2_final/$RUN_TAG
LOCK=$ARCHIVE/.finalize.lock

for variant in binary_risk soft_risk; do
  for seed in 7 13 21; do
    result=$ARCHIVE/$variant/seed${seed}/test_metrics.json
    if [[ ! -s "$result" ]]; then
      echo "V2_DREAMER_NOT_READY missing=$result"
      exit 0
    fi
  done
done

mkdir -p "$ARCHIVE"
(
  flock -n 9 || exit 0
  if [[ -s "$ARCHIVE/summary.json" ]]; then
    echo "V2_DREAMER_ALREADY_FINALIZED $ARCHIVE"
    exit 0
  fi
  cd "$ROOT"
  export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
  "$PY" scripts/65_summarize_agentdojo_v2_dreamer.py \
    --root "$ARCHIVE" \
    --seeds 7 13 21 \
    --output "$ARCHIVE/summary.json" \
    > "$ARCHIVE/summary_stdout.json"
  echo "AGENTDOJO_V2_DREAMER_FINALIZED tag=$RUN_TAG $(date -Is)"
) 9>"$LOCK"
