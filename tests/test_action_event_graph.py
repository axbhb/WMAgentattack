from wmagentattack.action_event_graph import (
    audit_action_event_graph_dataset, build_action_event_graph_dataset,
    event_graph_features,
)


def step(step_id=0):
    return {
        "trajectory_id":"t", "step_id":step_id, "selected_skill":"record_read",
        "selected_tool":"search_contacts", "skill_arguments":{"query":"Emma"},
        "skill_output":"{'id': '7', 'email': 'e@example.com'}", "tool_error":None,
        "previous_skills":[], "user_goal":"Find Emma's email",
        "task_success":True, "attack_success":False, "target_skill":"email_external",
    }


def event(step_id=0):
    return {
        "trajectory_id":"t", "step_id":step_id, "event_id":f"e{step_id}",
        "task_name":"suite|task", "current_legal_candidate_ids":["read", "finish"],
    }


def test_features_keep_action_structure_but_drop_raw_values_and_labels():
    features, summary = event_graph_features(
        step(), previous_step=None, current_legal=["read","finish"], previous_legal=None,
    )
    text=" ".join(features)
    assert "action.tool=search_contacts" in features
    assert "receipt.key_group=identifier" in features
    assert "receipt.key_group=communication" in features
    assert "Emma" not in text and "example.com" not in text
    assert "task_success" not in text and "target_skill" not in text
    assert summary["selected_tool_present"]


def test_build_is_deterministic_and_event_aligned():
    steps=[step(0),{**step(1),"previous_skills":["record_read"]}]
    events=[event(0),event(1)]
    left=build_action_event_graph_dataset(steps,events)
    right=build_action_event_graph_dataset(steps,events)
    assert left==right
    assert [row["event_id"] for row in left["rows"]]==["e0","e1"]


def test_audit_rejects_forbidden_feature():
    dataset=build_action_event_graph_dataset([step()],[event()])
    dataset["rows"][0]["features"].append("task_success=true")
    dataset["feature_catalog"]=sorted(set(dataset["feature_catalog"]+["task_success=true"]))
    audit=audit_action_event_graph_dataset(dataset,expected_rows=1,expected_tasks=1)
    assert not audit["checks"]["forbidden_features_absent"]
