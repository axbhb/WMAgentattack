import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "all_fresh",
    ROOT / "scripts" / "54_analyze_all_grouped_fresh_outcomes.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_task_and_pair_keys_are_distinct():
    row = {
        "suite": "banking",
        "user_task_id": "user_task_1",
        "injection_task_id": "injection_task_2",
    }
    assert MODULE._task_key(row) == ("banking", "user_task_1")
    assert MODULE._key(row) == (
        "banking",
        "user_task_1",
        "injection_task_2",
    )
