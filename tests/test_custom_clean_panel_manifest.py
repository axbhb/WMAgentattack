import importlib.util
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "134_build_custom_clean_panel.py"
SPEC = importlib.util.spec_from_file_location("custom_panel_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
custom_panel_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(custom_panel_manifest)


def test_manifest_is_deterministic_balanced_and_label_blind():
    first = custom_panel_manifest.build_manifest()
    second = custom_panel_manifest.build_manifest()
    assert first == second
    assert first["summary"]["tasks"] == 24
    assert Counter(row["split"] for row in first["rows"]) == {
        "training": 8,
        "calibration": 8,
        "confirmation": 8,
    }
    assert set(first["summary"]["by_split_suite"].values()) == {2}
    assert first["independence_contract"][
        "outcome_labels_read_during_manifest_construction"
    ] is False
    forbidden = {"utility", "success", "outcome", "label", "prediction"}
    assert not any(forbidden & set(row) for row in first["rows"])


def test_template_families_never_cross_task_split():
    manifest = custom_panel_manifest.build_manifest()
    owners = {}
    for row in manifest["rows"]:
        family = row["template_family"]
        owners.setdefault(family, row["split"])
        assert owners[family] == row["split"]
