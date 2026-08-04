import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v2_group_calibration_summary",
    ROOT / "scripts" / "66_summarize_agentdojo_v2_group_calibration.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_metric(root: Path, variant: str, seed: int, split: str, value: float):
    path = root / variant / f"seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{split}_metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "validation_objective": value,
                    "validation_objective_multiseed_group": value,
                    "grouped_risk_probability_brier_score": value / 3,
                    "risk_auc": 1.0 - value,
                    "next_skill_accuracy": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )


def test_selection_reads_validation_only_and_final_keeps_frozen_choice(tmp_path):
    seeds = [7, 13, 21]
    validation = {
        "calib0": [0.40, 0.42, 0.38],
        "calib01": [0.31, 0.30, 0.32],
        "calib025": [0.20, 0.22, 0.21],
        "calib05": [0.27, 0.25, 0.26],
    }
    for variant, values in validation.items():
        for seed, value in zip(seeds, values, strict=True):
            _write_metric(tmp_path, variant, seed, "val", value)
            # Invalid JSON proves selection never opens test payloads.
            test_path = tmp_path / variant / f"seed{seed}" / "test_metrics.json"
            test_path.write_text("not-json", encoding="utf-8")

    selection = MODULE.select(tmp_path, seeds)
    assert selection["selected_variant_by_validation"] == "calib025"
    assert selection["selected_weight"] == 0.25

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    for variant, values in {
        "calib0": [0.3, 0.3, 0.3],
        # Deliberately worse test performance must not change selection.
        "calib025": [0.9, 0.9, 0.9],
    }.items():
        for seed, value in zip(seeds, values, strict=True):
            _write_metric(tmp_path, variant, seed, "test", value)

    final = MODULE.finalize(tmp_path, seeds, selection_path)
    assert final["selected_variant_by_validation"] == "calib025"
    assert final["test_variants_evaluated"] == ["calib0", "calib025"]
    assert final["test_aggregate"]["calib025"]["validation_objective"][
        "mean"
    ] == 0.9
