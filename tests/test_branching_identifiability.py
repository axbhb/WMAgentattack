from wmagentattack.branching_identifiability import audit_paired_branch_effects


def _outcome(row_ref: str, decision_ref: str, *, error: bool, changed: bool):
    status = "error" if error else "success"
    return {
        "manifest_row_ref": row_ref,
        "model_visible": {
            "events": {
                "execution_status": status,
                "conflict_count_delta": 0,
                "unresolved_entity_count_delta": 0,
            },
            "next_semantic_state": {
                "goal_evidence": {
                    "matched_fact_terms": ["x"] if changed else [],
                    "unmatched_fact_terms": [] if changed else ["x"],
                    "unique_entity_records": int(changed),
                    "ambiguous_entity_records": 0,
                    "unlinked_entity_records": 0,
                    "conflict_count": 0,
                    "observed_entity_types": ["record"] if changed else [],
                },
                "execution": {"cumulative_errors": int(error)},
            },
        },
        "simulator_audit_only": {
            "state_changed": changed,
            "state_delta_roots": {"/x": 1} if changed else {},
        },
        "_decision_ref": decision_ref,
    }


def test_effect_audit_groups_same_prefix_and_excludes_action_identity():
    manifest = {
        "rows": [
            {
                "manifest_row_ref": f"r{i}",
                "query": {"decision_ref": "anchor"},
            }
            for i in range(4)
        ]
    }
    dataset = {
        "counterfactual_outcomes": [
            _outcome("r0", "anchor", error=False, changed=False),
            _outcome("r1", "anchor", error=False, changed=False),
            _outcome("r2", "anchor", error=False, changed=True),
            _outcome("r3", "anchor", error=True, changed=False),
        ]
    }
    audit = audit_paired_branch_effects(dataset, manifest)
    assert audit["anchors"] == 1
    assert audit["complete_four_action_anchors"] == 1
    assert audit["anchors_with_three_effects"] == 1
    assert audit["boundary_events"]["execution_error"] == 1
    assert audit["pairwise_effect_difference_fraction"] == 5 / 6
