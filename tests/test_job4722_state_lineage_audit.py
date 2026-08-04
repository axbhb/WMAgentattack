import importlib.util
from pathlib import Path

from wmagentattack import clean_evidence_probe


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "125_audit_job4722_state_feature_lineage.py"
SPEC = importlib.util.spec_from_file_location("job4722_state_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _prefix():
    return {
        "features": {
            "trusted_goal": "Find a hotel",
            "last_event": {"tool_name": "search", "arguments": {}},
            "state_summary": {
                "last_state_changed": False,
                "cumulative_state_changes": 0,
                "cumulative_errors": 0,
                "delta_roots": {},
            },
            "canonical_state": {"reservations": {"hotel": "A"}},
            "evidence_text": "hotel A price 100",
            "new_evidence_text": "hotel A price 100",
            "evidence_length": {
                "item_count": 1,
                "character_count": 17,
                "token_count": 4,
            },
            "prefix_length": 1,
        },
        "targets": {"expert_slot_coverage": 0.0, "is_final_prefix": False},
    }


def test_functional_probe_separates_state_and_non_state_variants():
    changed = MODULE._feature_probe(clean_evidence_probe, _prefix(), hash_dimension=16)
    assert all(changed[variant] for variant in MODULE.STATE_VECTOR_VARIANTS)
    assert all(not changed[variant] for variant in MODULE.NON_STATE_VECTOR_VARIANTS)
    assert changed["event_transformer_state_evidence"]


def test_line_number_audit_requires_literal_source_evidence(tmp_path):
    source = tmp_path / "example.py"
    source.write_text("first\ncanonical_state\nlast\n", encoding="utf-8")
    assert MODULE._line_numbers(source, ["canonical_state"]) == {
        "canonical_state": [2]
    }
