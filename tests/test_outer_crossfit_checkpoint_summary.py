import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "60_summarize_outer_crossfit_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("outer_checkpoint_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_checkpoint_summary_requires_complete_epoch_budget(tmp_path):
    for seed, auc in ((7, 0.7), (13, 0.9)):
        directory = tmp_path / "fold0" / f"seed{seed}"
        directory.mkdir(parents=True)
        (directory / "held_metrics.json").write_text(
            json.dumps({"test_steps": 10, "metrics": {"risk_auc": auc}}),
            encoding="utf-8",
        )
        (directory / "model_metadata.json").write_text(
            json.dumps(
                {
                    "config": {"epochs": 30},
                    "training_history": [{"epoch": 30}],
                    "parameter_count": 100,
                }
            ),
            encoding="utf-8",
        )
    result = MODULE.summarize(tmp_path, expected_checkpoints=2)
    assert result["all_checkpoints_reached_epoch_budget"]
    assert result["parameter_count_identical"]
    assert result["aggregate_metrics"]["risk_auc"]["mean"] == 0.8
