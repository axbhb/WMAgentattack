import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v2_dreamer_summary", ROOT / "scripts" / "65_summarize_agentdojo_v2_dreamer.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_summary_loader_reads_metrics_payload(tmp_path):
    path = tmp_path / "binary_risk" / "seed7"
    path.mkdir(parents=True)
    (path / "val_metrics.json").write_text(
        json.dumps({"metrics": {"validation_objective": 0.5}}),
        encoding="utf-8",
    )
    assert MODULE._load(tmp_path, "binary_risk", 7, "val") == {
        "validation_objective": 0.5
    }


def test_mean_std_uses_population_standard_deviation():
    summary = MODULE._mean_std([1.0, 2.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["std"] > 0.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
