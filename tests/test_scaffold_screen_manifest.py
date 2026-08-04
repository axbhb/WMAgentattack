import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "132_build_scaffold_screen_manifest.py"
SPEC = importlib.util.spec_from_file_location("scaffold_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
scaffold_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scaffold_manifest)


def _existing():
    return {
        "task_selection": {
            suite: {"train": ["user_task_0"]}
            for suite in scaffold_manifest.SUITES
        }
    }


def _tasks():
    return {
        suite: [f"user_task_{index}" for index in range(8)]
        for suite in scaffold_manifest.SUITES
    }


def test_scaffold_manifest_is_deterministic_balanced_and_label_blind():
    first = scaffold_manifest.build_manifest(
        _existing(), _tasks(), benchmark_version="v1.2.2", tasks_per_suite=4
    )
    second = scaffold_manifest.build_manifest(
        _existing(), _tasks(), benchmark_version="v1.2.2", tasks_per_suite=4
    )
    assert first == second
    assert first["summary"]["rows"] == 16
    assert set(first["summary"]["rows_by_suite"].values()) == {4}
    assert first["selection"]["outcome_labels_read"] is False
    assert all(row["user_task_id"] != "user_task_0" for row in first["rows"])


def test_scaffold_manifest_can_never_be_used_as_confirmation():
    result = scaffold_manifest.build_manifest(
        _existing(), _tasks(), benchmark_version="v1.2.2", tasks_per_suite=4
    )
    contract = result["independence_contract"]
    assert contract["screening_only"] is True
    assert contract["eligible_for_model_confirmation"] is False
    assert contract["eligible_for_attack_confirmation"] is False
    assert all(row["eligible_for_future_confirmation"] is False for row in result["rows"])
