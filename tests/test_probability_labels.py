from wmagentattack.probability_labels import (
    BetaEvidence,
    build_training_context_evidence,
    build_training_global_evidence,
    context_for_row,
    estimate_probability_label,
    global_for_row,
)


def _trajectory(name: str, success: bool, split_location: str = "inbox"):
    return {
        "trajectory_id": name,
        "domain": "workspace",
        "task_id": "user_task_1",
        "final_task_success": success,
        "steps": [
            {
                "attack_action": "attack",
                "attack_location": split_location,
            }
        ],
    }


def test_training_context_is_leave_one_out():
    rows = [_trajectory("a", True), _trajectory("b", False), _trajectory("c", False)]
    evidence = build_training_context_evidence(rows)
    loo = context_for_row(rows[0], training_context=evidence, split="train")
    assert loo == BetaEvidence(successes=0.0, attempts=2.0)
    held_out = context_for_row(rows[0], training_context=evidence, split="test")
    assert held_out == BetaEvidence(successes=1.0, attempts=3.0)


def test_continuous_label_is_not_a_binary_weight():
    label = estimate_probability_label(
        clean=BetaEvidence(successes=2.0, attempts=3.0),
        global_attack=BetaEvidence(successes=2.0, attempts=10.0),
        attacked=True,
        observed_success=False,
        context=BetaEvidence(successes=1.0, attempts=4.0),
        split="loo",
    )
    assert 0.0 < label.utility_probability < 1.0
    assert 0.0 < label.preservation_probability < 1.0
    assert label.alpha > 0 and label.beta > 0
    assert 0.0 <= label.confidence <= 1.0


def test_clean_label_uses_smoothed_clean_probability():
    label = estimate_probability_label(
        clean=BetaEvidence(successes=0.0, attempts=3.0),
        global_attack=None,
        attacked=False,
        observed_success=False,
        context=None,
        split="train_only",
    )
    assert label.utility_probability == 0.125
    assert label.preservation_probability is None


def test_training_global_evidence_is_leave_one_out():
    rows = [_trajectory("a", True), _trajectory("b", False), _trajectory("c", False)]
    evidence = build_training_global_evidence(rows)
    assert evidence == BetaEvidence(successes=1.0, attempts=3.0)
    loo = global_for_row(rows[0], training_global=evidence, split="train")
    assert loo == BetaEvidence(successes=0.0, attempts=2.0)
