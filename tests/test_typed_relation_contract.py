import json

from wmagentattack.bound_successor_world_model import record_signature
from wmagentattack.typed_relation_contract import (
    has_forbidden_key,
    relation_score,
    schema_vocabulary,
    structural_relation,
    typed_goal_units,
)


def cloud_file_signature():
    return record_signature({
        "entity_type": "cloud_file",
        "link_status": "UNIQUE",
        "attributes": [
            {"name": "owner", "kind": "SINGLE_VALUED"},
            {"name": "shared_with", "kind": "SET_VALUED"},
            {"name": "size", "kind": "SINGLE_VALUED"},
        ],
    })


def test_typed_units_remove_raw_values_but_preserve_schema_roles():
    signature = cloud_file_signature()
    goal = {
        "fact_terms": ["owner", "permission", "165"],
        "operation_terms": ["find", "report"],
        "logic_terms": [],
        "typed_mentions": [{"kind": "number", "value": "165"}],
    }
    action = {"tool_id": "workspace::share_file", "arguments": [{"field": "permission"}]}
    units = typed_goal_units(goal, action, schema_vocabulary([signature]))
    assert "ATTRIBUTE_TOKEN" in units[0]["roles"]
    assert "ACTION_FIELD_TOKEN" in units[1]["roles"]
    assert "VALUE_KIND:number" in units[2]["roles"]
    assert all(len(unit["unit_hash"]) == 64 for unit in units)


def test_structural_relations_distinguish_direct_and_action_bridges():
    signature = cloud_file_signature()
    goal = {"fact_terms": ["owner", "permission"], "operation_terms": ["find"], "logic_terms": [], "typed_mentions": []}
    action = {"tool_id": "workspace::share_file", "arguments": [{"field": "permission"}]}
    units = typed_goal_units(goal, action, schema_vocabulary([signature]))
    direct_types, direct_score = structural_relation(units[0], signature, action)
    bridge_types, bridge_score = structural_relation(units[1], signature, action)
    assert "DIRECT_ATTRIBUTE" in direct_types and direct_score == 1.0
    assert "ACTION_FIELD_BRIDGE" in bridge_types and bridge_score == 0.75


def test_combined_score_is_fixed_and_bounded():
    assert relation_score(1.0, 1.0) == 1.0
    assert relation_score(0.0, -1.0) == 0.0
    assert 0.0 < relation_score(0.5, 0.0) < 1.0


def test_forbidden_raw_goal_and_task_keys_are_detected():
    assert has_forbidden_key({"normalized_goal": "secret"})
    assert has_forbidden_key({"nested": [{"task_id": "x"}]})
    assert not has_forbidden_key({"unit_hash": "a" * 64, "roles": ["ATTRIBUTE_TOKEN"]})
