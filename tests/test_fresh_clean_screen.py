import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILD = _load("fresh_clean_build_test", ROOT / "scripts" / "98_build_fresh_clean_task_manifest.py")
SUMMARY = _load("fresh_clean_summary_test", ROOT / "scripts" / "100_summarize_fresh_clean_screen.py")
MULTISEED = _load(
    "fresh_clean_multiseed_test",
    ROOT / "scripts" / "102_summarize_fresh_clean_multiseed.py",
)
FILTER = _load(
    "fresh_clean_filter_test",
    ROOT / "scripts" / "104_filter_fresh_clean_manifest.py",
)


def test_manifest_excludes_existing_tasks_without_reading_outcomes():
    protocol = {
        "task_selection": {
            "banking": {"train": ["user_task_0"]},
            "slack": {"train": ["user_task_0"]},
            "travel": {"train": ["user_task_0"]},
            "workspace": {"train": ["user_task_0"]},
        }
    }
    tasks = {suite: ["user_task_0", "user_task_1"] for suite in BUILD.SUITES}
    result = BUILD.build_manifest(protocol, tasks, benchmark_version="v1.2.2")
    assert result["summary"]["rows"] == 4
    assert result["selection"]["outcome_labels_read"] is False
    assert all(row["user_task_id"] == "user_task_1" for row in result["rows"])


def test_summary_requires_exact_chunk_coverage(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {"row_id": "clean::banking::a"},
                    {"row_id": "clean::slack::b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    chunk0 = tmp_path / "chunk0.json"
    chunk1 = tmp_path / "chunk1.json"
    chunk0.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "row_id": "clean::banking::a",
                        "status": "completed",
                        "suite": "banking",
                        "user_task_id": "a",
                        "utility": True,
                        "elapsed_seconds": 2.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    chunk1.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "row_id": "clean::slack::b",
                        "status": "completed",
                        "suite": "slack",
                        "user_task_id": "b",
                        "utility": False,
                        "elapsed_seconds": 3.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = SUMMARY.summarize(manifest, [chunk0, chunk1])
    assert result["counts"]["tasks"] == 2
    assert result["counts"]["clean_successes"] == 1
    assert result["timing_seconds"]["aggregate"] == 5.0


def test_multiseed_retains_at_least_two_successes(tmp_path):
    manifest = tmp_path / "manifest.json"
    rows = [
        {"row_id": "clean::banking::a", "suite": "banking", "user_task_id": "a"},
        {"row_id": "clean::banking::b", "suite": "banking", "user_task_id": "b"},
    ]
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    outcomes = {101: [True, False], 103: [True, True], 107: [False, False]}
    for seed, values in outcomes.items():
        directory = tmp_path / f"seed{seed}"
        directory.mkdir()
        (directory / "chunk0.json").write_text(
            json.dumps(
                {
                    "run_seed": seed,
                    "results": [
                        {
                            **row,
                            "status": "completed",
                            "utility": value,
                        }
                        for row, value in zip(rows, values)
                    ],
                }
            ),
            encoding="utf-8",
        )
    result = MULTISEED.summarize(
        manifest, tmp_path, seeds=(101, 103, 107), chunks=1
    )
    assert result["counts"]["retained_tasks"] == 1
    assert result["tasks"][0]["retained"] is True
    assert result["tasks"][1]["retained"] is False


def test_filter_manifest_keeps_clean_safety_contract():
    manifest = {
        "scope": "AgentDojo sandbox only; clean-task solvability screen",
        "safety_contract": {
            "clean_tasks_only": True,
            "allow_real_network_endpoints": False,
        },
        "rows": [
            {"suite": "travel", "user_task_id": "user_task_1"},
            {"suite": "travel", "user_task_id": "user_task_2"},
            {"suite": "slack", "user_task_id": "user_task_1"},
        ],
    }
    result = FILTER.filter_manifest(
        manifest,
        suite="travel",
        task_ids={"user_task_2"},
    )
    assert result["selection"]["attack_outcomes_used"] is False
    assert result["selection"]["selected_rows"] == 1
    assert result["rows"] == [
        {"suite": "travel", "user_task_id": "user_task_2"}
    ]
