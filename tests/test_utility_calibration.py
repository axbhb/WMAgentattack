import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "90_evaluate_v2_utility_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("utility_calibration", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_grouped_utility_data_uses_first_decision_and_complete_attack_groups():
    steps = []
    scores = []
    for trajectory in range(2):
        for step_id, score in ((0, 0.2 + trajectory * 0.2), (1, 0.9)):
            steps.append(
                SimpleNamespace(
                    trajectory_id=f"t{trajectory}",
                    step_id=step_id,
                    multiseed_group_id="attack::g",
                    multiseed_trials=2,
                    utility_probability_target=0.5,
                )
            )
            scores.append(score)
    data = MODULE._grouped_utility_data(steps, np.asarray(scores))
    assert data.group_count == 1
    assert data.trajectory_count == 2
    assert np.allclose(data.score_groups[0], [0.2, 0.4])
    assert np.allclose(data.targets, [0.5])
