import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "outer_folds",
    ROOT / "scripts" / "56_make_grouped_outer_crossfit_folds.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_outer_assignments_hold_every_task_once_and_balance_suites():
    tasks = {
        (suite, f"task_{index}")
        for suite in ("banking", "slack", "travel", "workspace")
        for index in range(9)
    }
    assignments = MODULE._outer_assignments(tasks, folds=4, seed=17)
    assert set(assignments) == tasks
    for suite in ("banking", "slack", "travel", "workspace"):
        counts = [
            sum(task[0] == suite and fold == value for task, value in assignments.items())
            for fold in range(4)
        ]
        assert max(counts) - min(counts) <= 1


def test_inner_validation_is_suite_balanced_and_disjoint():
    tasks = {
        (suite, f"task_{index}")
        for suite in ("banking", "slack", "travel", "workspace")
        for index in range(8)
    }
    selected = MODULE._inner_validation_tasks(
        tasks, fold=0, seed=11, per_suite=2
    )
    assert len(selected) == 8
    assert all(sum(task[0] == suite for task in selected) == 2 for suite in {t[0] for t in tasks})
