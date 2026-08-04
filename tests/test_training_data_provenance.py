import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_full_dreamer", ROOT / "scripts" / "23_train_full_dreamer_v3.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@dataclass
class _Step:
    trajectory_id: str
    domain: str
    task_id: str


def test_step_provenance_records_hash_and_task_counts(tmp_path):
    path = tmp_path / "steps.jsonl"
    path.write_bytes(b"one\ntwo\n")
    steps = [
        _Step("trajectory_1", "banking", "task_1"),
        _Step("trajectory_1", "banking", "task_1"),
        _Step("trajectory_2", "slack", "task_2"),
    ]
    result = MODULE._step_provenance(path, steps)
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["trajectory_count"] == 2
    assert result["user_task_count"] == 2
    assert result["user_tasks_by_domain"] == {"banking": 1, "slack": 1}
