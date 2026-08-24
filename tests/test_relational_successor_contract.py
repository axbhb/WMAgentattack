import json
from pathlib import Path

from wmagentattack.relational_successor_contract import (
    relational_successor_delta,
    static_record_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def _record(terms):
    return {
        "record_position": 0, "call_index": 0, "entity_type": "webpage",
        "entity_key": {"url": "x"}, "entity_candidates": [], "link_status": "UNIQUE",
        "attributes": [{"name": "content", "value": "secret", "kind": "SINGLE_VALUED"}],
        "context": {}, "source_tool": "get_webpage", "source_arguments": {"url": "x"},
        "execution_status": "success", "matched_goal_terms": terms,
    }


def test_record_local_goal_links_are_exact_and_private():
    current = {
        "goal": {"fact_terms": ["alpha", "beta", "gamma"]},
        "goal_evidence": {"matched_fact_terms": []}, "evidence_records": [],
    }
    following = {
        "goal": current["goal"], "goal_evidence": {"matched_fact_terms": ["alpha", "beta", "gamma"]},
        "evidence_records": [_record(["alpha", "beta", "gamma"])],
        "execution": {"last_status": "success"},
    }
    target, audit = relational_successor_delta(current, following)
    assert target["added_evidence_records"][0]["newly_matched_goal_term_indices"] == [0, 1, 2]
    assert target["newly_matched_goal_term_indices"] == [0, 1, 2]
    encoded = json.dumps(target)
    assert "secret" not in encoded and "alpha" not in encoded
    assert audit["uncovered_new_goal_terms"] == []


def test_static_candidates_include_webpage_and_error_without_outcomes():
    base = json.loads((ROOT / "configs/0726_travel_tool_output_adapter_registry_v2.json").read_text())
    extension = json.loads((ROOT / "configs/0729_custom_panel_v2_adapter_extension.json").read_text())
    schemas = json.loads((ROOT / "configs/0824_static_tool_output_schema_v29.json").read_text())
    candidates, by_tool = static_record_candidates(base, extension, schemas)
    decoded = [json.loads(value) for value in candidates]
    assert any(value["entity_type"] == "webpage" for value in decoded)
    assert any(value["entity_type"] == "execution_error" for value in decoded)
    assert "get_webpage" in by_tool and "send_email" in by_tool
    assert not schemas["outcome_labels_present"]
