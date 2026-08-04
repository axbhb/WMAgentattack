#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=/share/guozhix/WMagentattack
PY=/share/guozhix/software/miniconda3/envs/wmagentattack/bin/python
ARCHIVE=/share/guozhix/wmagentattack/0724/clean_state_interface_audit/fixed_v1

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
  echo "execution_mode=remote_login_clean_cpu_audit"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_head=$(git rev-parse HEAD)"
  "$PY" --version
} > "$ARCHIVE/runtime.txt"

cp configs/0724_clean_state_interface_audit_protocol.json \
  "$ARCHIVE/protocol.preregistered.json"
cp src/wmagentattack/clean_state_instrumentation.py \
  scripts/119_audit_agentdojo_clean_state_interfaces.py \
  tests/test_clean_state_instrumentation.py \
  tests/test_clean_state_interface_audit.py \
  "$ARCHIVE/frozen/"

sha256sum \
  configs/0724_clean_state_interface_audit_protocol.json \
  src/wmagentattack/clean_state_instrumentation.py \
  scripts/119_audit_agentdojo_clean_state_interfaces.py \
  tests/test_clean_state_instrumentation.py \
  tests/test_clean_state_interface_audit.py \
  scripts/server/run_0724_clean_state_interface_audit.sh \
  > "$ARCHIVE/code.prerun.sha256"

"$PY" -m pytest -q \
  tests/test_clean_state_instrumentation.py \
  tests/test_clean_state_interface_audit.py \
  tests/test_exact_simulator.py \
  > "$ARCHIVE/preflight_tests.txt"

"$PY" scripts/119_audit_agentdojo_clean_state_interfaces.py \
  --benchmark-version v1.2.2 \
  --suites banking slack travel workspace \
  --output "$ARCHIVE/audit.json" \
  > "$ARCHIVE/audit_stdout.json" \
  2> "$ARCHIVE/audit_stderr.log"

"$PY" - "$ARCHIVE/audit.json" "$ARCHIVE/decision.json" <<'PY'
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
decision_path = Path(sys.argv[2])
report = json.loads(audit_path.read_text(encoding="utf-8"))
overall = report["overall"]
contract = report["safety_contract"]
gates = {
    "zero_ground_truth_execution_errors": overall["execution_error_call_count"] == 0,
    "all_ground_truth_tasks_pass_final_official_utility": (
        overall["final_ground_truth_utility_successes"] == overall["task_count"]
    ),
    "all_expert_traces_match_their_target_only_goal_slots": (
        overall["expert_trace_slot_match_failures"] == 0
    ),
    "all_state_snapshots_are_canonical_json": (
        overall["canonical_state_json_roundtrip_failures"] == 0
    ),
    "safety_contract_holds": (
        contract["clean_user_tasks_only"]
        and contract["ground_truth_replay_only"]
        and not contract["llm_loaded"]
        and not contract["attacks_constructed"]
        and not contract["external_endpoints"]
        and not contract["training_examples_created"]
    ),
}
adapter_ready = all(gates.values())
decision = {
    "protocol_id": "0724_clean_state_interface_audit_fixed_v1",
    "gates": gates,
    "state_delta_adapter_ready": adapter_ready,
    "fractional_progress_ready": False,
    "irreversibility_ready": False,
    "dynamic_candidate_preconditions_ready": False,
    "clean_data_gate": "BLOCKED",
    "decision": (
        "STATE_DELTA_ADAPTER_READY_CLEAN_GATE_BLOCKED"
        if adapter_ready
        else "STATE_DELTA_ADAPTER_NOT_READY_CLEAN_GATE_BLOCKED"
    ),
    "next_admissible_step": (
        "fixed-budget clean observed-victim instrumentation pilot"
        if adapter_ready
        else "repair exact state adapter without generating attack data"
    ),
}
decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
print(json.dumps(decision, indent=2))
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
