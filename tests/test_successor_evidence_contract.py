from wmagentattack.successor_evidence_contract import render_effect_tokens


def test_renderer_preserves_record_bindings_and_action_source():
    target = {
        "execution_status": "success",
        "delta_bits": [1, 1, 0, 0, 0],
        "added_evidence_records": [{
            "entity_type": "webpage",
            "link_status": "UNIQUE",
            "attributes": [{"name": "content", "kind": "SINGLE_VALUED"}],
        }],
        "added_conflicts": [],
        "newly_matched_goal_term_indices": [0, 2, 4],
    }
    tokens = render_effect_tokens(target, {"tool_id": "slack::get_webpage", "arguments": []})
    assert "attribute=webpage::content::SINGLE_VALUED" in tokens
    assert "entity=webpage" in tokens
    assert "source=get_webpage" in tokens
    assert "matched_count=3" in tokens


def test_matched_count_is_clipped_deterministically():
    target = {
        "execution_status": "success",
        "delta_bits": [0, 0, 0, 0, 0],
        "added_evidence_records": [],
        "added_conflicts": [],
        "newly_matched_goal_term_indices": [0, 1, 2, 3, 4],
    }
    tokens = render_effect_tokens(target, {"tool_id": "workspace::search_files", "arguments": []})
    assert "matched_count=3" in tokens
    assert not any(token.startswith("entity=") for token in tokens)


def test_structured_target_has_no_composite_label_field():
    target = {
        "execution_status": "error",
        "delta_bits": [0, 0, 0, 0, 1],
        "added_evidence_records": [],
        "added_conflicts": [],
        "newly_matched_goal_term_indices": [],
    }
    assert "effect_tokens" not in target
    assert "task_id" not in target
    assert "utility" not in target
    assert "security" not in target
