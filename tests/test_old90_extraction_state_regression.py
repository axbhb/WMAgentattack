import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "130_regress_old90_extraction_state_v2.py"
PROTOCOL_PATH = ROOT / "configs" / "0726_old90_extraction_state_regression.json"
SPEC = importlib.util.spec_from_file_location("old90_regression", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
old90_regression = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(old90_regression)


def test_old90_protocol_is_engineering_only_and_frozen():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["status"] == "preregistered_before_execution"
    assert set(protocol["frozen_gates"].values()) == {True}
    assert protocol["data_policy"]["engineering_regression_only"] is True
    assert protocol["data_policy"]["outcome_fields_accessed"] is False
    assert protocol["data_policy"]["model_training"] is False
    assert protocol["data_policy"]["model_comparison"] is False


def test_source_descriptor_projection_drops_outcome_fields(tmp_path):
    archive = tmp_path / "archive"
    chunk_dir = archive / "seed1"
    chunk_dir.mkdir(parents=True)
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    (chunk_dir / "chunk0.json").write_text(
        json.dumps(
            {
                "run_seed": 1,
                "results": [
                    {
                        "status": "completed",
                        "user_task_id": "user_task_0",
                        "raw_trace": str(trace),
                        "utility": {"sentinel": "must_not_be_projected"},
                        "security": {"sentinel": "must_not_be_projected"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    protocol = {
        "source_panels": [
            {
                "name": "fixture",
                "archive_root": str(archive),
                "seeds": [1],
                "chunks_per_seed": 1,
                "expected_episodes": 1,
            }
        ]
    }
    descriptors, chunks = old90_regression._load_source_descriptors(protocol)
    assert len(chunks) == 1
    assert len(descriptors) == 1
    serialized = json.dumps(descriptors)
    assert "utility" not in serialized
    assert "security" not in serialized
    assert "sentinel" not in serialized


def test_output_leakage_scanner_catches_nested_outcomes():
    assert old90_regression._forbidden_paths(
        {"episode": [{"targets": {"task_success": True}}], "utility": False}
    ) == ["episode.0.targets.task_success", "utility"]
    assert old90_regression._forbidden_paths(
        {"records": [{"outcome_labels_present": False}]}
    ) == []


def test_regression_source_never_calls_experts_checkers_or_training():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'result["utility"]' not in source
    assert "result['utility']" not in source
    assert ".utility(" not in source
    assert ".security(" not in source
    assert "ground_truth(" not in source
    assert "train(" not in source
    assert "transformers" not in source
