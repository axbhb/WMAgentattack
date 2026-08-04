import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "121_build_clean_evidence_ledger_dataset.py"
SPEC = importlib.util.spec_from_file_location("clean_evidence_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_prefix_feature_contract_accepts_causal_fields():
    MODULE._assert_feature_contract(
        {
            "trusted_goal": "find a hotel",
            "last_event": {"tool_name": "search", "arguments": {}},
            "state_summary": {"last_state_changed": False},
            "canonical_state": {"calendar": {"events": []}},
            "evidence_text": "Hotel A costs 180",
            "new_evidence_text": "Hotel A costs 180",
            "evidence_length": {"item_count": 1},
            "prefix_length": 1,
        }
    )


@pytest.mark.parametrize(
    "bad_field",
    ["utility", "security", "expert_slot_coverage", "attack_family"],
)
def test_prefix_feature_contract_rejects_privileged_fields(bad_field):
    with pytest.raises(ValueError):
        MODULE._assert_feature_contract({bad_field: 1})


def test_delta_roots_are_structural_not_state_fingerprints():
    roots = MODULE._delta_roots(
        (
            {"op": "replace", "path": "/reservation/0/status", "value": "ok"},
            {"op": "add", "path": "/calendar/events/1", "value": {}},
        )
    )
    assert roots == {"/calendar": 1, "/reservation": 1}
