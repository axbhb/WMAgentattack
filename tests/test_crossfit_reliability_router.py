import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "router",
    ROOT / "scripts" / "55_evaluate_crossfit_reliability_router.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_stump_selects_feature_that_routes_better_tasks():
    tasks = [("banking", str(index)) for index in range(4)]
    statistics = {
        task: {feature: float(index) for feature in MODULE.FEATURES}
        for index, task in enumerate(tasks)
    }
    hybrid = {task: (0.9 if index >= 2 else 0.1) for index, task in enumerate(tasks)}
    text = {task: 0.5 for task in tasks}
    stump = MODULE._select_stump(tasks, statistics, hybrid, text)
    assert stump["direction"] == "high"
    assert stump["threshold"] == 1.5
    assert stump["train_mean_primary"] == 0.7


def test_candidate_thresholds_include_all_and_none_routes():
    thresholds = MODULE._candidate_thresholds([1.0, 2.0, 3.0])
    assert thresholds[0] == -1.0
    assert thresholds[-1] == 5.0
    assert thresholds[1:-1] == [1.5, 2.5]
