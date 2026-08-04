import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "145_audit_custom_panel_v2_architecture_preflight.py"
SPEC = importlib.util.spec_from_file_location("architecture_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_fails_closed_on_out_of_schema_argument_target(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    audit_path = tmp_path / "audit.json"
    protocol_path = tmp_path / "protocol.json"
    dataset = {
        "schema_version": "wmagentattack.custom_clean_panel_v2_architecture_dataset.v2",
        "argument_key_vocab": ["query"],
        "episodes": [
            {
                "prefixes": [
                    {
                        "targets": {
                            "argument_keys": ["properties"],
                        }
                    }
                ]
            }
        ],
    }
    audit = {
        "schema_version": "wmagentattack.custom_clean_panel_v2_architecture_dataset.v2",
        "passed": True,
        "gates": {"fixture": True},
        "prefixes": 467,
        "evidence_statuses": {"UNOBSERVED": 411},
        "unknown_argument_target_keys": [],
    }
    dataset_sha = _write(dataset_path, dataset)
    audit_sha = _write(audit_path, audit)
    protocol = {
        "status": "preregistered_before_training",
        "implementation_sha256": {},
        "source": {
            "dataset_sha256": dataset_sha,
            "dataset_audit_sha256": audit_sha,
        },
        "frozen_variants": [
            "semantic_markov",
            "observable_execution",
            "observable_execution_ledger_v2",
        ],
        "representation_contract": {"strict_nesting": True},
        "training": {"training_seeds": [7, 17, 29], "hyperparameter_grid": False},
        "fixed_budget": {"training_runs": 9, "new_victim_model_calls": 0},
        "scope": {
            "completion_or_reporting_training": False,
            "attack_data": False,
            "h2_attack_planning": False,
            "dreamer_training": False,
        },
    }
    _write(protocol_path, protocol)

    result = MODULE.audit(protocol_path, dataset_path, audit_path)

    assert result["passed"] is False
    assert result["unknown_argument_target_keys"] == ["properties"]
    assert result["gates"]["all_argument_targets_in_declared_schema_vocab"] is False
