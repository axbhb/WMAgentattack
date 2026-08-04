import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "136_summarize_custom_clean_panel.py"
SPEC = importlib.util.spec_from_file_location("custom_panel_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
custom_panel_summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(custom_panel_summary)


def _fixture(tmp_path: Path, successful_ids: set[str]):
    tmp_path.mkdir(parents=True, exist_ok=True)
    suites = ("banking", "slack", "travel", "workspace")
    splits = ("training", "calibration", "confirmation")
    rows = []
    for split in splits:
        for suite in suites:
            for index in range(2):
                task_id = f"user_task_{split}_{suite}_{index}"
                rows.append(
                    {
                        "row_id": f"clean::{suite}::{task_id}",
                        "suite": suite,
                        "user_task_id": task_id,
                        "split": split,
                        "template_family": f"{split}_family_{index}",
                    }
                )
    manifest = {
        "custom_task_module": "wmagentattack.custom_agentdojo_panel_v1",
        "rows": rows,
    }
    protocol = {
        "clean_replay": {
            "development_seeds": [233, 239, 241],
            "confirmation_seeds": [251, 257, 263],
            "chunks_per_seed": 1,
        },
        "frozen_data_sufficiency_gate": {
            "minimum_total_durable_tasks": 8,
            "minimum_durable_tasks_per_split": {"training": 3, "calibration": 3, "confirmation": 3},
            "minimum_durable_tasks_in_each_core_suite": {"banking": 1, "slack": 1, "workspace": 1},
            "minimum_domains_with_two_durable_tasks": 2,
        },
        "completion_head_balance_gate": {
            "minimum_training_durable_success_tasks": 3,
            "minimum_training_all_six_failure_tasks": 2,
            "minimum_confirmation_durable_success_tasks": 3,
            "minimum_confirmation_all_six_failure_tasks": 2,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    protocol_path = tmp_path / "protocol.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    archive = tmp_path / "archive"
    for seed in (233, 239, 241, 251, 257, 263):
        directory = archive / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        results = []
        for index, row in enumerate(rows):
            trace = directory / f"trace_{index}.json"
            trace.write_text(json.dumps({"messages": [{"role": "assistant", "tool_calls": [{"function": "x"}]}]}), encoding="utf-8")
            results.append(
                {
                    **row,
                    "status": "completed",
                    "utility": row["row_id"] in successful_ids,
                    "raw_trace": str(trace),
                }
            )
        (directory / "chunk0.json").write_text(
            json.dumps(
                {
                    "run_seed": seed,
                    "chunk_index": 0,
                    "num_chunks": 1,
                    "custom_task_module": manifest["custom_task_module"],
                    "results": results,
                }
            ),
            encoding="utf-8",
        )
    return protocol_path, manifest_path, archive, rows


def test_balanced_custom_panel_passes_both_gates(tmp_path):
    _, _, _, rows = _fixture(tmp_path / "seed", set())
    successful = {row["row_id"] for row in rows if row["user_task_id"].endswith("_0")}
    protocol, manifest, archive, _ = _fixture(tmp_path / "actual", successful)
    result = custom_panel_summary.summarize(protocol, manifest, archive)
    assert result["coverage"]["complete"] is True
    assert result["data_sufficiency_gate"]["passed"] is True
    assert result["completion_head_balance_gate"]["passed"] is True
    assert result["decision"] == "CUSTOM_PANEL_READY_FOR_FROZEN_LEDGER_AND_COMPLETION_ABLATION"
    assert result["task_counts"]["seed_split_informative"] is False


def test_missing_chunk_blocks_all_downstream_gates(tmp_path):
    protocol, manifest, archive, rows = _fixture(tmp_path, set())
    (archive / "seed263" / "chunk0.json").unlink()
    result = custom_panel_summary.summarize(protocol, manifest, archive)
    assert result["coverage"]["complete"] is False
    assert result["data_sufficiency_gate"]["passed"] is False
    assert result["dynamics_progress_ablation_permitted"] is False
    assert result["attack_data_permitted"] is False
