#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0724/observed_clean_state_replay/fixed_v1
DEVELOPMENT=/share/guozhix/wmagentattack/0716/function_tag_parser_retry/function_tag_parser_retry_20260716_v1
CONFIRMATION=/share/guozhix/wmagentattack/0716/function_tag_parser_retry_confirmation/function_tag_parser_retry_confirmation_20260716_v1

cd "$ROOT"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite frozen archive: $ARCHIVE" >&2
  exit 2
fi
mkdir -p "$ARCHIVE/frozen" logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

failure_record() {
  status=$?
  {
    echo "failed_at=$(date --iso-8601=seconds)"
    echo "exit_status=$status"
  } > "$ARCHIVE/FAILED"
  exit "$status"
}
trap failure_record ERR

{
  echo "execution_mode=remote_login_existing_clean_replay"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
} > "$ARCHIVE/runtime.txt"

cp configs/0724_observed_clean_state_replay_protocol.json \
  "$ARCHIVE/protocol.preregistered.json"
cp src/wmagentattack/clean_state_instrumentation.py \
  scripts/120_replay_observed_clean_state.py \
  tests/test_observed_clean_state_replay.py \
  "$ARCHIVE/frozen/"

sha256sum \
  configs/0724_observed_clean_state_replay_protocol.json \
  src/wmagentattack/clean_state_instrumentation.py \
  scripts/120_replay_observed_clean_state.py \
  tests/test_observed_clean_state_replay.py \
  scripts/server/run_0724_observed_clean_state_replay.sh \
  > "$ARCHIVE/code.prerun.sha256"

find "$DEVELOPMENT" "$CONFIRMATION" -type f \
  \( -name 'chunk*.json' -o -name 'none.json' \) -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$ARCHIVE/source_data.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_clean_state_instrumentation.py \
  tests/test_clean_state_interface_audit.py \
  tests/test_observed_clean_state_replay.py \
  tests/test_exact_simulator.py \
  > "$ARCHIVE/preflight_tests.txt"

"$PY" scripts/120_replay_observed_clean_state.py \
  --protocol configs/0724_observed_clean_state_replay_protocol.json \
  --output "$ARCHIVE/replay_audit.json" \
  > "$ARCHIVE/replay_stdout.json" \
  2> "$ARCHIVE/replay_stderr.log"

"$PY" - "$ARCHIVE/replay_audit.json" "$ARCHIVE/decision.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
Path(sys.argv[2]).write_text(
    json.dumps(report["decision"], indent=2), encoding="utf-8"
)
PY

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "runtime_failures=0"
} >> "$ARCHIVE/runtime.txt"

date --iso-8601=seconds > "$ARCHIVE/COMPLETE"
(
  cd "$ARCHIVE"
  find . -type f -not -name checksums.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > checksums.sha256
  sha256sum -c checksums.sha256 > checksums.verify.txt
)
trap - ERR
