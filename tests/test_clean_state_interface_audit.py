import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "119_audit_agentdojo_clean_state_interfaces.py"
SPEC = importlib.util.spec_from_file_location("clean_state_interface_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _passing_overall():
    return {
        "execution_error_call_count": 0,
        "final_ground_truth_utility_successes": 10,
        "task_count": 10,
        "expert_trace_slot_match_failures": 0,
        "canonical_state_json_roundtrip_failures": 0,
    }


def test_state_adapter_gate_requires_every_preregistered_condition():
    assert MODULE._state_delta_adapter_ready(_passing_overall())

    failure_cases = {
        "execution_error_call_count": 1,
        "final_ground_truth_utility_successes": 9,
        "expert_trace_slot_match_failures": 1,
        "canonical_state_json_roundtrip_failures": 1,
    }
    for field, value in failure_cases.items():
        overall = _passing_overall()
        overall[field] = value
        assert not MODULE._state_delta_adapter_ready(overall), field


def test_canonical_json_roundtrip_rejects_nonfinite_values():
    assert MODULE._canonical_json_roundtrip_ok({"a": [1, "x", None]})
    assert not MODULE._canonical_json_roundtrip_ok({"a": float("nan")})

