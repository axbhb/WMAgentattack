import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "parser_retry_confirmation",
    ROOT / "scripts" / "108_evaluate_parser_retry_confirmation.py",
)
confirmation_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(confirmation_module)


def _summary(seeds, successes):
    tasks = []
    for task_id, count in successes.items():
        tasks.append(
            {
                "suite": "travel",
                "user_task_id": task_id,
                "successes": count,
                "outcomes": {
                    str(seed): index < count for index, seed in enumerate(seeds)
                },
            }
        )
    return {
        "protocol": {"seeds": list(seeds)},
        "counts": {
            "tasks": len(tasks),
            "clean_episodes": len(tasks) * len(seeds),
        },
        "tasks": tasks,
    }


def _protocol():
    return {
        "development_reference": {"seeds": [101, 103, 107]},
        "confirmation": {"seeds": [109, 113, 127], "tasks": 3, "episodes": 9},
        "decision_gate": {
            "retention_successes_out_of_three": 2,
            "minimum_confirmation_retained_tasks": 2,
            "minimum_tasks_retained_in_both_development_and_confirmation": 2,
        },
        "decision_if_go": "GO",
        "decision_if_no_go": "NO_GO",
        "claim_boundary": "clean-only",
    }


def _audit():
    return {"counts": {"episodes": 9, "failures_without_tool_call": 1}}


def test_confirmation_gate_passes_with_two_durable_tasks():
    development = _summary((101, 103, 107), {"a": 2, "b": 2, "c": 0})
    confirmation = _summary((109, 113, 127), {"a": 2, "b": 3, "c": 1})

    result = confirmation_module.evaluate(
        development, confirmation, _audit(), _protocol()
    )

    assert result["gate"]["all_required_criteria_met"] is True
    assert result["decision"] == "GO"
    assert result["durable_task_ids"] == ["travel::a", "travel::b"]


def test_confirmation_gate_fails_when_retained_tasks_do_not_overlap():
    development = _summary((101, 103, 107), {"a": 2, "b": 2, "c": 0})
    confirmation = _summary((109, 113, 127), {"a": 1, "b": 2, "c": 2})

    result = confirmation_module.evaluate(
        development, confirmation, _audit(), _protocol()
    )

    assert result["gate"]["at_least_two_confirmation_retained_tasks"] is True
    assert (
        result["gate"]["at_least_two_durable_development_confirmation_tasks"]
        is False
    )
    assert result["decision"] == "NO_GO"
