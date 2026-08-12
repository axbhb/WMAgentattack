from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from wmagentattack.three_source_unified import _agentdojo_cohorts


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_three_source_expansion",
    ROOT / "scripts" / "182_train_three_source_expansion.py",
)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def test_source_task_balanced_weights_preserve_frozen_source_mass() -> None:
    rows = [
        {"source": "agentdojo", "task_key": "a"},
        {"source": "agentdojo", "task_key": "a"},
        {"source": "agentdojo", "task_key": "b"},
        {"source": "tool_sandbox", "task_key": "c"},
        {"source": "injecagent", "task_key": "d"},
        {"source": "injecagent", "task_key": "e"},
        {"source": "injecagent", "task_key": "e"},
    ]
    requested = {"agentdojo": 0.5, "tool_sandbox": 0.25, "injecagent": 0.25}
    weights = TRAIN.source_task_balanced_weights(rows, requested)
    total = float(weights.sum())
    realized = {
        source: float(
            sum(weight for row, weight in zip(rows, weights) if row["source"] == source)
            / total
        )
        for source in requested
    }
    assert realized == {source: np.float32(mass) for source, mass in requested.items()}
    task_mass = {
        task: float(
            sum(weight for row, weight in zip(rows, weights) if row["task_key"] == task)
            / total
        )
        for task in {row["task_key"] for row in rows}
    }
    assert np.isclose(task_mass["a"], task_mass["b"])
    assert np.isclose(task_mass["d"], task_mass["e"])


def test_agentdojo_cohorts_partition_one_task_per_domain() -> None:
    metadata = []
    for domain in ("banking", "slack", "travel", "workspace"):
        for split, count in (("train", 3), ("val", 1), ("test", 1)):
            for index in range(count):
                metadata.append(
                    {
                        "suite": domain,
                        "user_task_id": f"{split}_{index}",
                        "task_split": split,
                    }
                )
    cohorts = _agentdojo_cohorts(metadata)
    assert set(cohorts) == {
        "original_test",
        "original_val",
        "train0",
        "train1",
        "train2",
    }
    assert all(len(tasks) == 4 for tasks in cohorts.values())
    assert len(set().union(*(set(tasks) for tasks in cohorts.values()))) == 20
