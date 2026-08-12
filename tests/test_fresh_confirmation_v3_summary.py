from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fresh_v3_summary", ROOT / "scripts" / "193_summarize_fresh_custom_confirmation_v3.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_missing_all_array_outputs_is_a_runtime_failure(tmp_path: Path) -> None:
    protocol = json.loads(
        (ROOT / "configs" / "0813_fresh_integrated_validation_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = {
        "rows": [
            {
                "row_id": spec.row_id,
                "suite": spec.suite,
                "user_task_id": spec.task_id,
            }
            for spec in MODULE.panel.TASK_SPECS
        ]
    }
    gate, steps, metadata = MODULE.summarize(
        protocol=protocol, manifest=manifest, result_root=tmp_path
    )
    assert gate["decision"] == "NO_GO_FRESH_CUSTOM_CONFIRMATION_V3_CLEAN_INELIGIBLE"
    assert gate["checks"]["zero_runtime_failures"] is False
    assert gate["episodes_completed"] == 0
    assert len(gate["missing_keys"]) == 36
    assert steps == []
    assert metadata == []
