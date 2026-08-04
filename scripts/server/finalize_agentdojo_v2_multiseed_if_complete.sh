#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0714/agentdojo_v2
MANIFEST=$ARCHIVE/manifests/screen_manifest.json
RESULT_ROOT=$ARCHIVE/final/results
RAW_DIR=$ARCHIVE/final/raw_dataset
OUT_DIR=$ARCHIVE/final_dataset
LOCK=$ARCHIVE/final/.finalize.lock
SEEDS=(7 17 29 43 61)
CHUNKS=8

for seed in "${SEEDS[@]}"; do
  for ((chunk=0; chunk<CHUNKS; chunk++)); do
    result="$RESULT_ROOT/seed${seed}/chunk${chunk}.json"
    if [[ ! -s "$result" ]]; then
      echo "V2_MULTISEED_NOT_READY missing=$result"
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
      echo "V2_MULTISEED_NOT_READY incomplete=$result"
      exit 0
    fi
  done
done

mkdir -p "$(dirname "$LOCK")" "$RAW_DIR" "$OUT_DIR"
(
  flock -n 9 || exit 0
  if [[ -s "$OUT_DIR/audit.json" ]]; then
    echo "V2_MULTISEED_ALREADY_FINALIZED $OUT_DIR"
    exit 0
  fi
  cd "$ROOT"
  export PYTHONPATH="$ROOT/src:$ROOT/external/agentdojo/src:${PYTHONPATH:-}"
  "$PY" scripts/63_prepare_agentdojo_v2_dataset.py \
    --manifest "$MANIFEST" \
    --result-root "$RESULT_ROOT" \
    --clean-run-root "$ROOT/runs/agentdojo_clean_multiseed_llama31_70b" \
    --out-dir "$RAW_DIR" \
    --require-complete
  "$PY" scripts/64_build_agentdojo_v2_final.py \
    --split-dir "$RAW_DIR" \
    --out-dir "$OUT_DIR" \
    --dataset-version v2.1-final-5seed \
    --expected-attack-seed 7 \
    --expected-attack-seed 17 \
    --expected-attack-seed 29 \
    --expected-attack-seed 43 \
    --expected-attack-seed 61 \
    --expected-attack-groups 400 \
    --min-clean-seeds 3 \
    --min-base-success-rate 0.5 \
    --preservation-weight-floor 0.05
  echo "AGENTDOJO_V2_FINALIZED $(date -Is)"
) 9>"$LOCK"
