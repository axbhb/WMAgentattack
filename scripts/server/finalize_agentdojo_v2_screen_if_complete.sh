#!/usr/bin/env bash
set -euo pipefail

RUN_SEED=${1:-7}
CHUNKS=${2:-8}
ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0714/agentdojo_v2
MANIFEST=$ARCHIVE/manifests/screen_manifest.json
RESULT_ROOT=$ARCHIVE/screen/results/seed${RUN_SEED}
OUT_DIR=$ARCHIVE/screen_dataset/seed${RUN_SEED}
LOCK=$ARCHIVE/screen/.finalize-seed${RUN_SEED}.lock

for ((chunk=0; chunk<CHUNKS; chunk++)); do
  result="$RESULT_ROOT/chunk${chunk}.json"
  if [[ ! -s "$result" ]]; then
    echo "V2_SCREEN_NOT_READY missing=$result"
    exit 0
  fi
  if ! "$PY" - "$result" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
summary = data["summary"]
ready = summary["completed"] == summary["expected"] and summary["failed"] == 0
raise SystemExit(0 if ready else 1)
PY
  then
    echo "V2_SCREEN_NOT_READY incomplete=$result"
    exit 0
  fi
done

mkdir -p "$(dirname "$LOCK")" "$OUT_DIR"
(
  flock -n 9 || exit 0
  if [[ -s "$OUT_DIR/audit.json" ]]; then
    echo "V2_SCREEN_ALREADY_FINALIZED $OUT_DIR"
    exit 0
  fi
  cd "$ROOT"
  export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
  "$PY" scripts/63_prepare_agentdojo_v2_dataset.py \
    --manifest "$MANIFEST" \
    --result-root "$RESULT_ROOT" \
    --clean-run-root "$ROOT/runs/agentdojo_clean_multiseed_llama31_70b" \
    --out-dir "$OUT_DIR" \
    --require-complete
  echo "AGENTDOJO_V2_SCREEN_FINALIZED seed=$RUN_SEED $(date -Is)"
) 9>"$LOCK"
