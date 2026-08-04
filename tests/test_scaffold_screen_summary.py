import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "133_summarize_scaffold_screen.py"
SPEC = importlib.util.spec_from_file_location("scaffold_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
scaffold_summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scaffold_summary)


def _write_fixture(tmp_path: Path, candidate_outcomes: dict[str, list[list[bool]]]):
    suites = ["banking", "slack", "travel", "workspace"]
    rows = [
        {
            "row_id": f"clean::{suite}::user_task_1",
            "suite": suite,
            "user_task_id": "user_task_1",
            "screening_only": True,
        }
        for suite in suites
    ]
    manifest = {
        "rows": rows,
        "safety_contract": {"clean_tasks_only": True, "allow_real_network_endpoints": False},
    }
    candidates = [
        {"id": candidate_id, "prompt_profile": "base", "do_sample": True}
        for candidate_id in candidate_outcomes
    ]
    protocol = {
        "screening": {
            "tasks": 4,
            "seeds": [151, 157, 163],
            "chunks_per_seed": 1,
            "candidates": candidates,
            "baseline_candidate": "base_sampled",
            "candidate_order_for_exact_ties": [
                candidate_id for candidate_id in candidate_outcomes if candidate_id != "base_sampled"
            ],
        },
        "frozen_selection_gate": {
            "minimum_retained_task_gain_over_baseline": 2,
            "minimum_clean_success_gain_over_baseline": 3,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    protocol_path = tmp_path / "protocol.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    archive = tmp_path / "archive"
    for candidate_id, per_seed in candidate_outcomes.items():
        for seed_index, seed in enumerate((151, 157, 163)):
            output_dir = archive / candidate_id / f"seed{seed}"
            output_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for row_index, row in enumerate(rows):
                trace = output_dir / f"trace_{row_index}.json"
                trace.write_text(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "assistant", "tool_calls": [{"function": "x"}]}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                results.append(
                    {
                        **row,
                        "status": "completed",
                        "utility": per_seed[seed_index][row_index],
                        "raw_trace": str(trace),
                    }
                )
            (output_dir / "chunk0.json").write_text(
                json.dumps(
                    {
                        "run_seed": seed,
                        "chunk_index": 0,
                        "num_chunks": 1,
                        "results": results,
                    }
                ),
                encoding="utf-8",
            )
    return protocol_path, manifest_path, archive


def test_material_candidate_is_selected(tmp_path):
    baseline = [
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, False],
    ]
    improved = [[True, True, True, True]] * 3
    paths = _write_fixture(
        tmp_path,
        {"base_sampled": baseline, "constraint_checklist_sampled": improved},
    )
    result = scaffold_summary.summarize(*paths)
    assert result["gate"]["all_candidates_complete"] is True
    assert result["gate"]["selected_candidate"] == "constraint_checklist_sampled"
    comparison = result["paired_vs_baseline"]["constraint_checklist_sampled"]
    assert comparison["retained_task_gain"] == 3
    assert comparison["clean_success_delta"] == 9
    assert comparison["eligible_to_replace_baseline"] is True


def test_non_material_candidate_retains_baseline(tmp_path):
    baseline = [[True, False, False, False]] * 3
    small_gain = [[True, True, False, False]] * 3
    paths = _write_fixture(
        tmp_path,
        {"base_sampled": baseline, "base_greedy": small_gain},
    )
    result = scaffold_summary.summarize(*paths)
    assert result["gate"]["selected_candidate"] == "base_sampled"
    assert result["gate"]["decision"] == (
        "SCAFFOLD_SCREEN_RETAIN_BASE_SAMPLED_NO_MATERIAL_IMPROVEMENT"
    )


def test_missing_chunk_yields_no_selection(tmp_path):
    outcomes = [[True, False, False, False]] * 3
    protocol, manifest, archive = _write_fixture(
        tmp_path, {"base_sampled": outcomes, "base_greedy": outcomes}
    )
    (archive / "base_greedy" / "seed163" / "chunk0.json").unlink()
    result = scaffold_summary.summarize(protocol, manifest, archive)
    assert result["gate"]["all_candidates_complete"] is False
    assert result["gate"]["selected_candidate"] is None
    assert result["gate"]["decision"] == "SCAFFOLD_SCREEN_INCOMPLETE_NO_SELECTION"
