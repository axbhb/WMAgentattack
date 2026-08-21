from wmagentattack.paired_generation_schema import audit_paired_generation_tables


def _tables():
    episodes = []
    outcomes = []
    pairs = []
    transitions = []
    classes = [
        "attack0_utility0", "attack0_utility1",
        "attack1_utility0", "attack1_utility1",
    ]
    for index, label in enumerate(classes):
        pair = f"p{index}"
        control = f"e{index}c"
        intervention = f"e{index}i"
        pairs.append({
            "pair_ref": pair,
            "control_episode_ref": control,
            "intervention_episode_ref": intervention,
            "same_initial_state": True,
            "same_task": True,
            "same_template": True,
            "same_seed": True,
            "changed_factor_count": 1,
        })
        for variant, episode in (("c", control), ("i", intervention)):
            outcome = f"o{index}{variant}"
            episodes.append({
                "episode_ref": episode,
                "pair_ref": pair,
                "source": "agentdojo",
                "transition_tier": "observed",
                "outcome_ref": outcome,
                "split": "train" if index < 2 else "confirmation",
                "component_ref": f"component{index}",
            })
            outcomes.append({
                "outcome_ref": outcome,
                "episode_ref": episode,
                "joint_label_valid": True,
                "joint_class": label,
            })
            transitions.append({
                "transition_ref": f"t{index}{variant}",
                "episode_ref": episode,
                "model_input": {"state": {}, "action": {}},
                "transition_target": {"execution_status": "success"},
            })
    return {"episodes": episodes, "transitions": transitions, "outcomes": outcomes, "pairs": pairs}


def test_four_table_contract_passes_without_outcome_leakage() -> None:
    audit = audit_paired_generation_tables(
        _tables(), expected_episodes=8, expected_pairs=4,
        minimum_joint_per_cell=2, maximum_joint_class_fraction=0.25,
    )
    assert audit["passed"] is True


def test_final_outcome_in_transition_is_rejected() -> None:
    tables = _tables()
    tables["transitions"][0]["transition_target"]["attack_success"] = True
    audit = audit_paired_generation_tables(
        tables, expected_episodes=8, expected_pairs=4,
        minimum_joint_per_cell=2, maximum_joint_class_fraction=0.25,
    )
    assert audit["checks"]["transition_fields_are_outcome_blind"] is False

