from wmagentattack.intervention_union import (
    _argument_value_class,
    effect_tokens,
    normalized_action_descriptor,
)


def test_argument_normalization_removes_fixed_values():
    assert _argument_value_class("person@example.com")["category"] == "email"
    assert _argument_value_class("not-a-date-v18")["category"] == "invalid_or_missing_sentinel"
    assert _argument_value_class(2_147_483_647)["range"] == "missing_entity_sentinel"


def test_action_descriptor_uses_link_flag_not_raw_value():
    value = normalized_action_descriptor(
        {"tool_id": "suite::tool", "arguments": {"file_id": "9"}},
        {"evidence_records": [{"entity_key": {"file_id": "9"}}]},
    )
    assert value["arguments"][0]["exact_value_observed_in_state"] is True
    assert "9" not in str(value["arguments"][0]["value_class"])


def test_effect_tokens_are_value_blind():
    tokens = effect_tokens(
        {
            "delta_bits": [1, 0, 0, 0, 0],
            "execution_status": "success",
            "newly_matched_goal_term_count": 0,
            "added_records": [
                {
                    "entity_type": "cloud_file",
                    "link_status": "unique",
                    "source_tool": "get_file_by_id",
                    "attributes": [["permission", "SINGLE_VALUED"]],
                }
            ],
            "added_conflicts": [],
        }
    )
    assert "entity=cloud_file" in tokens
    assert all("secret" not in token for token in tokens)
