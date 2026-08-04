import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "factorized_summary",
    ROOT / "scripts" / "112_summarize_factorized_smoke.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_summary_applies_frozen_architecture_thresholds_and_clean_block(tmp_path, monkeypatch):
    for states, validation_nll in ((4, 1.1), (6, 0.9), (8, 1.0)):
        path = tmp_path / "io_hmm" / f"state{states}"
        path.mkdir(parents=True)
        payload = {
            "config": {"num_states": states},
            "metrics": {
                "validation": {
                    "mean_event_nll": validation_nll,
                    "next_event_accuracy": 0.60,
                },
                "test": {"mean_event_nll": 1.0, "next_event_accuracy": 0.58},
            },
            "counterbaseline": {
                "metrics": {
                    "validation": {"mean_event_nll": 1.2},
                    "test": {"mean_event_nll": 1.1},
                }
            },
        }
        (path / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    for seed, accuracy in ((7, 0.61), (17, 0.62), (29, 0.60)):
        path = tmp_path / "event" / f"seed{seed}"
        path.mkdir(parents=True)
        payload = {
            "metrics": {
                "validation": {
                    "next_tool_accuracy": accuracy,
                    "joint_count_nll": 1.0,
                },
                "test": {
                    "next_tool_accuracy": 0.59,
                    "joint_count_nll": 1.1,
                },
            },
            "joint_constant_baseline": {
                "count_nll": {"validation": 1.2, "test": 1.2}
            },
        }
        (path / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["summary", "--root", str(tmp_path), "--output", str(output)],
    )
    MODULE.main()
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["best_io_hmm_selected_by_validation_only"]["states"] == 6
    assert summary["architecture_signal"] is True
    assert summary["gates"]["clean_eligibility_gate"] is False
    assert summary["decision"] == "ARCHITECTURE_SIGNAL_ONLY_CLEAN_GATE_BLOCKED"

