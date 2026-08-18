from wmagentattack.parameter_intervention import audit_parameter_interventions


def _outcome(row_ref: str, status: str):
    return {
        "manifest_row_ref": row_ref,
        "model_visible": {
            "events": {"execution_status": status},
            "next_semantic_state": {"goal_evidence": {"matched_fact_terms": []}},
        },
        "simulator_audit_only": {
            "state_changed": status == "success",
            "state_delta_roots": {"/x": 1} if status == "success" else {},
            "tool_error_type": None if status == "success" else "ValueError",
        },
    }


def test_parameter_audit_requires_paired_success_to_error_flip():
    manifest = {
        "pair_audit_only": [
            {
                "pair_ref": "p",
                "task_id": "t",
                "suite": "slack",
                "tool_name": "add_user_to_channel",
                "control_row_ref": "control",
                "corrupted_row_ref": "corrupt",
            }
        ]
    }
    dataset = {
        "counterfactual_outcomes": [
            _outcome("control", "success"),
            _outcome("corrupt", "error"),
        ]
    }
    audit = audit_parameter_interventions(dataset, manifest)
    assert audit["complete_pairs"] == 1
    assert audit["paired_status_flips"] == 1
    assert audit["pairs_with_effect_change"] == 1
    assert audit["error_types"] == {"ValueError": 1}
