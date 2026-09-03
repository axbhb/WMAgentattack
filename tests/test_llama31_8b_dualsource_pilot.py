import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "0903_llama31_8b_dualsource_pilot_protocol.json"
DATA = ROOT / "data" / "0903_llama31_8b_dualsource_pilot"


def test_frozen_budget_and_source_semantics():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "completed_scientific_no_scale"
    assert protocol["victim_model"]["training_data_decoding"]["seeds"] == [
        1103,
        1109,
        1117,
    ]
    assert protocol["victim_model"]["locked_evaluation_decoding"]["do_sample"] is False
    assert protocol["agentdojo"]["pilot_budget"]["total_trajectories"] == 1224
    assert protocol["injecagent"]["pilot_budget"]["decisions"] == 48
    assert "never consecutive time steps" in protocol["injecagent"]["transition_semantics"]
    assert protocol["authorization_boundary"]["real_external_endpoint_calls"] == 0
    assert protocol["pilot_result"]["passed"] is False
    assert protocol["pilot_result"]["full_scale_authorized"] is False
    assert (
        protocol["pilot_result"]["decision"]
        == "NO_SCALE_RETAIN_COUNTEREVIDENCE_AND_REDESIGN_ONE_CAUSE"
    )


def test_task_selection_and_intervention_pairs_are_complete():
    selection = json.loads((DATA / "agentdojo_selection.json").read_text(encoding="utf-8"))
    assert sum(len(row["user_tasks"]) for row in selection["suites"].values()) == 24
    assert all(len(row["user_tasks"]) == 6 for row in selection["suites"].values())
    assert all(len(row["injection_tasks"]) == 2 for row in selection["suites"].values())
    assert len(selection["attacks"]) == 8

    manifest = json.loads((DATA / "injecagent_manifest.json").read_text(encoding="utf-8"))
    rows = manifest["records"]
    assert len(rows) == 48
    assert len({row["row_id"] for row in rows}) == 48
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], set()).add(row["variant"])
        assert row["real_external_endpoint_calls"] == 0
        assert row["tool_schemas"]
        assert all(tool["function"]["name"] for tool in row["tool_schemas"])
    assert len(pairs) == 24
    assert all(variants == {"clean", "poisoned"} for variants in pairs.values())


def test_no_generated_artifact_contains_outcome_labels_as_inputs():
    manifest = json.loads((DATA / "injecagent_manifest.json").read_text(encoding="utf-8"))
    forbidden = {"task_success", "attack_success", "utility", "security", "final_outcome"}
    for row in manifest["records"]:
        assert forbidden.isdisjoint(row)


def test_gate_accepts_a_complete_synthetic_pilot(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    run_root = tmp_path / "runs"
    attacks = protocol["agentdojo"]["published_attack_candidates"]
    seeds = protocol["victim_model"]["training_data_decoding"]["seeds"]
    messages = [
        {"role": "assistant", "tool_calls": [{"function": "a"}]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "tool_calls": [{"function": "b"}]},
    ]
    for seed in seeds:
        for suite, selection in protocol["agentdojo"]["pilot_suites"].items():
            for task in selection["user_tasks"]:
                clean_path = run_root / f"seed{seed}" / suite / task / "none" / "none.json"
                clean_path.parent.mkdir(parents=True, exist_ok=True)
                clean_path.write_text(
                    json.dumps(
                        {
                            "suite_name": suite,
                            "user_task_id": task,
                            "injection_task_id": None,
                            "attack_type": None,
                            "messages": messages,
                            "error": None,
                            "utility": True,
                        }
                    ),
                    encoding="utf-8",
                )
                for attack_index, attack in enumerate(attacks):
                    for injection in selection["injection_tasks"]:
                        path = (
                            run_root
                            / f"seed{seed}"
                            / suite
                            / task
                            / attack
                            / f"{injection}.json"
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        success = attack_index % 2 == 0
                        path.write_text(
                            json.dumps(
                                {
                                    "suite_name": suite,
                                    "user_task_id": task,
                                    "injection_task_id": injection,
                                    "attack_type": attack,
                                    "messages": messages,
                                    "error": None,
                                    "utility": success,
                                    "security": success,
                                }
                            ),
                            encoding="utf-8",
                        )

    manifest = json.loads((DATA / "injecagent_manifest.json").read_text(encoding="utf-8"))
    intervention_rows = []
    for row in manifest["records"]:
        intervention_rows.append(
            {
                **row,
                "completion": "done",
                "decision": {"kind": "text"},
                "runtime_error": None,
                "selected_attacker_tool": row["variant"] == "poisoned",
                "selected_user_tool": False,
            }
        )
    intervention = tmp_path / "injec.json"
    intervention.write_text(
        json.dumps({"complete": True, "records": intervention_rows}),
        encoding="utf-8",
    )
    output = tmp_path / "normalized"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "305_gate_llama31_8b_dualsource_pilot.py"),
            "--protocol",
            str(PROTOCOL),
            "--agentdojo-root",
            str(run_root),
            "--injecagent-output",
            str(intervention),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gate = json.loads((output / "pilot_gate.json").read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert gate["agentdojo"]["trajectories"] == 1224
    assert gate["injecagent"]["complete_pairs"] == 24
