import importlib.util
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agentdojo_split", ROOT / "scripts" / "09_split_real_agentdojo_dataset.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@dataclass
class _Trajectory:
    domain: str
    task_id: str
    final_attack_success: bool
    final_task_success: bool
    steps: list = field(default_factory=list)


def test_grouped_split_keeps_user_tasks_disjoint_and_is_deterministic():
    rows = []
    for domain in ("banking", "slack", "travel", "workspace"):
        for task in range(12):
            for attempt in range(2 + task % 3):
                rows.append(
                    _Trajectory(
                        domain=domain,
                        task_id=f"task_{task}",
                        final_attack_success=(task + attempt) % 3 == 0,
                        final_task_success=(task + attempt) % 2 == 0,
                    )
                )
    first = MODULE._grouped_user_task_split(
        rows, seed=7, train_ratio=0.7, val_ratio=0.15, search_iterations=100
    )
    second = MODULE._grouped_user_task_split(
        rows, seed=7, train_ratio=0.7, val_ratio=0.15, search_iterations=100
    )
    keys = {
        name: {(row.domain, row.task_id) for row in split}
        for name, split in first.items()
    }
    assert not (keys["train"] & keys["val"])
    assert not (keys["train"] & keys["test"])
    assert not (keys["val"] & keys["test"])
    assert sum(map(len, first.values())) == len(rows)
    for domain in ("banking", "slack", "travel", "workspace"):
        assert all(any(row.domain == domain for row in split) for split in first.values())
    assert {
        name: [(row.domain, row.task_id) for row in split]
        for name, split in first.items()
    } == {
        name: [(row.domain, row.task_id) for row in split]
        for name, split in second.items()
    }
