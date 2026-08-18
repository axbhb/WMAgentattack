import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v13_summary", ROOT / "scripts" / "228_summarize_predicted_event_graph_v13.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _rows(metric, left, right):
    a = []
    b = []
    for seed in (7, 17, 29):
        for task in ("a", "b"):
            common = {"training_seed": seed, "task_name": task}
            a.append({**common, metric: left})
            b.append({**common, metric: right})
    return a, b


def test_nll_effect_is_left_minus_right():
    left, right = _rows("nll", 2.0, 1.75)
    assert MODULE._effect(left, right, "nll")["mean"] == 0.25


def test_accuracy_effect_is_right_minus_left():
    left, right = _rows("accuracy", 0.4, 0.5)
    assert abs(MODULE._effect(left, right, "accuracy", higher=True)["mean"] - 0.1) < 1e-12
