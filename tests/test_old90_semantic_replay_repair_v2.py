import importlib.util
import json
from pathlib import Path

from wmagentattack.runtime_metadata_v2 import load_runtime_metadata_registry
from wmagentattack.state_storage_v2 import (
    ContentAddressedStateStore,
    build_exact_state_transition,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "131_regress_old90_semantic_replay_v2.py"
PROTOCOL = ROOT / "configs" / "0726_old90_extraction_state_regression_repair_v2.json"
METADATA = ROOT / "configs" / "0726_runtime_metadata_normalization_v2.json"
SPEC = importlib.util.spec_from_file_location("semantic_repair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
semantic_repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(semantic_repair)


def _episode(store, timestamp):
    before = {"inbox": {"emails": {}}}
    after = {
        "inbox": {
            "emails": {
                "1": {"id_": "1", "subject": "Hello", "timestamp": timestamp}
            }
        }
    }
    delta = [
        {
            "op": "add",
            "path": "/inbox/emails/1",
            "value": {"id_": "1", "subject": "Hello", "timestamp": timestamp},
        }
    ]
    transition = build_exact_state_transition(
        store,
        episode_id="fixture",
        call_index=0,
        initial_state=before,
        state_before=before,
        state_after=after,
        exact_delta=delta,
        execution_status="success",
        error_type=None,
    )
    return {
        "episode_id": "fixture",
        "panel": "fixture",
        "seed": 1,
        "user_task_id": "task",
        "source_trace_sha256": "a" * 64,
        "pairing": {
            "proposals": 1,
            "executed_calls": 1,
            "terminal_unexecuted": 0,
            "alignment_ok": True,
        },
        "transitions": [
            {
                "call_index": 0,
                "tool_name": "send_email",
                "execution_status": "success",
                "state_transition": transition.model_dump(mode="json"),
                "structured_records": [
                    {
                        "record_id": "r",
                        "attributes": [
                            {
                                "name": "timestamp",
                                "value": timestamp,
                                "kind": "TIME_SCOPED",
                            }
                        ],
                    }
                ],
                "new_conflicts": [],
            }
        ],
        "final_ledger": {"records": [], "conflicts": []},
        "checks": {
            "logged_execution_status_exact": True,
            "state_fingerprint_roundtrip_exact": True,
            "ledger_replay_idempotent": True,
            "terminal_unexecuted_updates_zero": True,
        },
        "observed_tools": ["send_email"],
    }


def test_repair_protocol_preserves_v1_failure_and_raw_state():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["prior_decision"] == (
        "OLD90_EXTRACTION_STATE_REGRESSION_FAIL_ENGINEERING_ONLY"
    )
    assert protocol["status"] == (
        "preregistered_after_v1_diagnosis_before_repair_execution"
    )
    assert set(protocol["frozen_gates"].values()) == {True}
    assert protocol["normalization_contract"]["raw_exact_state_retained"] is True
    assert protocol["data_policy"]["outcome_fields_accessed"] is False


def test_semantic_projection_matches_different_registered_wall_clocks(tmp_path):
    registry = load_runtime_metadata_registry(METADATA)
    left_store = ContentAddressedStateStore(tmp_path / "left")
    right_store = ContentAddressedStateStore(tmp_path / "right")
    left, left_manifest = semantic_repair._semantic_projection(
        _episode(left_store, "2026-01-01T01:02:03.100000"),
        left_store,
        registry,
    )
    right, right_manifest = semantic_repair._semantic_projection(
        _episode(right_store, "2026-02-02T02:03:04.200000"),
        right_store,
        registry,
    )
    assert left == right
    assert left_manifest == right_manifest
    assert left_manifest["binding_count"] == 1
    assert "2026-01-01" not in json.dumps(left)
    assert "VOLATILE::timestamp::CALL_000::ITEM_000" in json.dumps(left)


def test_repair_source_does_not_access_outcomes_or_train_models():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'result["utility"]' not in source
    assert "result['utility']" not in source
    assert ".utility(" not in source
    assert ".security(" not in source
    assert "ground_truth(" not in source
    assert "train(" not in source
    assert "transformers" not in source
