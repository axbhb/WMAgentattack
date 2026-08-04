import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "top_heavy_probe_test", ROOT / "scripts" / "94_probe_v2_top_heavy_ranking.py"
)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


def _rows():
    return [
        {"task_key": "d|a", "group_id": "a0", "target": 1.8},
        {"task_key": "d|a", "group_id": "a1", "target": 1.2},
        {"task_key": "d|a", "group_id": "a2", "target": 0.8},
        {"task_key": "d|a", "group_id": "a3", "target": 0.4},
        {"task_key": "d|b", "group_id": "b0", "target": 1.4},
        {"task_key": "d|b", "group_id": "b1", "target": 1.4},
        {"task_key": "d|b", "group_id": "b2", "target": 0.6},
    ]


def test_top_anchor_only_uses_maximum_tier_as_high_item():
    rows = _rows()
    pairs, counts = PROBE._pair_examples(rows, "top_anchor")
    assert counts == {"d|a": 3, "d|b": 2}
    assert all(
        rows[high]["target"] == max(
            row["target"] for row in rows if row["task_key"] == rows[high]["task_key"]
        )
        for high, _, _, _ in pairs
    )


def test_lambda_ndcg3_weights_are_positive_and_task_normalized():
    rows = _rows()
    pairs, _ = PROBE._pair_examples(rows, "lambda_ndcg3")
    for task in {row["task_key"] for row in rows}:
        task_weights = [
            weight for high, _, _, weight in pairs if rows[high]["task_key"] == task
        ]
        assert task_weights
        assert all(weight > 0 for weight in task_weights)
        assert abs(sum(task_weights) / len(task_weights) - 1.0) < 1e-12


def test_largest_gap_control_matches_fixed_pair_cap():
    rows = _rows()
    pairs, counts = PROBE._pair_examples(rows, "largest_gap_control", max_per_task=2)
    assert len(pairs) == 4
    assert counts == {"d|a": 2, "d|b": 2}
