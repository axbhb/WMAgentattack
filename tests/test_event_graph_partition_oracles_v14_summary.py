import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v14_summary", ROOT / "scripts" / "231_summarize_event_graph_partition_oracles_v14.py"
)
MODULE = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(MODULE)


def test_arm_effect_direction():
    left = []; right = []
    for seed in (7, 17, 29):
        for task in ("a", "b"):
            common = {"training_seed": seed, "task_name": task, "horizon": 2}
            left.append({**common, "action_nll": 2.0})
            right.append({**common, "action_nll": 1.75})
    assert MODULE._arm_effect(left, right, "action_nll")["mean"] == 0.25
