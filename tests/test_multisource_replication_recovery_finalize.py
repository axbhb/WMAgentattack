import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "180_finalize_multisource_replication_recovery.py"
    spec = importlib.util.spec_from_file_location("replication_recovery_finalize", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paired_diagnostic_preserves_direction():
    records = []
    for group, clean, poisoned in (("a", False, True), ("b", True, False), ("c", False, False)):
        for variant, selected in (("clean", clean), ("poisoned", poisoned)):
            records.append(
                {
                    "group_id": group,
                    "run_seed": 311,
                    "variant": variant,
                    "execution": {
                        "selected_attacker_tool": selected,
                        "selected_user_tool": not selected,
                    },
                }
            )
    result = _module()._injec_pair_diagnostics(records)
    assert result["pairs"] == 3
    assert result["attacker_tool"]["poisoned_only"] == 1
    assert result["attacker_tool"]["clean_only"] == 1
    assert result["attacker_tool"]["poisoned_minus_clean_rate"] == 0.0
