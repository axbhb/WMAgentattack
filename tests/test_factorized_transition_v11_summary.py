import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summary_v11", ROOT / "scripts" / "222_summarize_factorized_transition_v11.py"
)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


def rows(metric, left, right):
    output_left, output_right = [], []
    for seed in (7, 17, 29):
        for task in ("a", "b"):
            common = {"training_seed": seed, "task_name": task}
            output_left.append({**common, metric: left})
            output_right.append({**common, metric: right})
    return output_left, output_right


def test_nll_effect_is_left_minus_right():
    left, right = rows("nll", 2.0, 1.5)
    effect = SUMMARY._paired_effect(left, right, "nll")
    assert effect["mean"] == 0.5


def test_accuracy_effect_is_right_minus_left():
    left, right = rows("accuracy", 0.4, 0.6)
    effect = SUMMARY._paired_effect(left, right, "accuracy", higher_is_better=True)
    assert abs(effect["mean"] - 0.2) < 1e-12
