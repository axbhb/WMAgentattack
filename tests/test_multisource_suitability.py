import copy

import numpy as np

from wmagentattack.multisource_suitability import (
    build_suitability_dataset,
    candidate_id,
    representation_vector,
    split_task_fingerprints,
    task_fingerprint,
)


def _schema(name: str = "lookup") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Lookup a visible record",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _record(source: str, goal: str, index: int, variant: str = "clean") -> dict:
    model_input = {"trusted_goal": goal, "tool_schemas": [_schema()]}
    metadata = {}
    execution = {"tier": "not_executed_text_response", "replica_identical": None}
    if source == "injecagent":
        model_input["observation"] = "Completed trusted tool call lookup. Result: visible"
        metadata = {"user_tool": "lookup"}
    if source == "tau3":
        model_input["policy"] = "Visible policy"
        metadata = {"domain": "retail"}
    if source == "tool_sandbox":
        metadata = {"primary_category": "OTHER"}
    return {
        "row_id": f"{source}::{index}::{variant}",
        "source": source,
        "group_id": f"{source}::{index}",
        "variant": variant,
        "model_input": model_input,
        "metadata": metadata,
        "decision": {"kind": "text"},
        "execution": execution,
    }


def _protocol(records: list[dict]) -> dict:
    counts = {}
    for row in records:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    return {
        "protocol_id": "test",
        "source": {"expected_rows": len(records), "expected_source_counts": counts},
        "split": {
            "seed": "fixed",
            "ratios": {"training": 0.6, "calibration": 0.2, "confirmation": 0.2},
        },
        "preflight_gate": {
            "minimum_task_units_per_source": 3,
            "minimum_confirmation_tasks_per_source": 1,
            "minimum_exact_rows_for_error_probe": 50,
            "minimum_exact_errors_for_error_probe": 10,
            "minimum_exact_successes_for_error_probe": 10,
            "minimum_each_error_class_per_training_and_confirmation": 2,
        },
    }


def test_split_is_deterministic_and_task_disjoint():
    tasks = [task_fingerprint("tau3", f"goal {i}") for i in range(10)]
    first = split_task_fingerprints(
        tasks, source="tau3", seed="fixed", train_ratio=0.6, calibration_ratio=0.2
    )
    second = split_task_fingerprints(
        list(reversed(tasks)),
        source="tau3",
        seed="fixed",
        train_ratio=0.6,
        calibration_ratio=0.2,
    )
    assert first == second
    assert list(first.values()).count("training") == 6
    assert list(first.values()).count("calibration") == 2
    assert list(first.values()).count("confirmation") == 2


def test_injecagent_pairs_remain_in_one_goal_disjoint_split():
    records = []
    for index in range(5):
        for variant in ("clean", "poisoned"):
            records.append(_record("injecagent", f"goal {index}", index, variant))
    dataset, audit = build_suitability_dataset(records, _protocol(records))
    assert audit["passed"]
    grouped = {}
    for row in dataset["rows"]:
        grouped.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in grouped.values())


def test_target_and_outcome_do_not_change_causal_representation():
    records = [
        _record("injecagent", f"goal {index}", index, variant)
        for index in range(5)
        for variant in ("clean", "poisoned")
    ]
    dataset, audit = build_suitability_dataset(records, _protocol(records))
    assert audit["passed"]
    row = dataset["rows"][0]
    changed = copy.deepcopy(row)
    changed["target_candidate_id"] = candidate_id("injecagent", "lookup")
    changed["exact_outcome"] = {
        "available": True,
        "execution_error": True,
        "state_changed": True,
        "output_nonempty": False,
    }
    for variant in (
        "semantic_markov",
        "structured_markov_v3",
        "full_history_diagnostic",
    ):
        left = representation_vector(row, variant=variant, hash_dimension=16)
        right = representation_vector(changed, variant=variant, hash_dimension=16)
        assert left.shape == (130,)
        assert np.array_equal(left, right)
