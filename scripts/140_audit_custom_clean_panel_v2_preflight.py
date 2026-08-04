"""Audit the frozen panel-v2 protocol without reading or generating victim outcomes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdojo.task_suite.load_suites import get_suite
from wmagentattack import custom_agentdojo_panel_v2 as panel
from wmagentattack.factorized_evaluator_v2 import evaluate_trace, load_alias_registry


BUILDER_PATH = ROOT / "scripts" / "139_build_custom_clean_panel_v2.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("panel_v2_builder", BUILDER_PATH)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
builder = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(builder)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ground_truth_preflight(alias_path: Path) -> dict[str, Any]:
    aliases = load_alias_registry(alias_path)
    factor_counts: dict[str, Counter[str]] = {
        name: Counter() for name in ("state_action", "evidence", "report", "overall")
    }
    unsuccessful_calls: list[dict[str, Any]] = []
    failed_tasks: list[dict[str, Any]] = []
    condition_first_contracts = 0
    for spec in panel.TASK_SPECS:
        suite = get_suite(panel.BENCHMARK_VERSION, spec.suite)
        trace = {
            "messages": [
                {
                    "role": "tool",
                    "tool_call": {"function": call.function, "args": call.args},
                    "error": None,
                }
                for call in spec.required_calls
            ]
            + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "content": spec.ground_truth_output}],
                    "tool_calls": [],
                }
            ]
        }
        evaluation = evaluate_trace(
            trace=trace,
            environment=suite.load_and_inject_default_environment({}),
            tools=suite.tools,
            mutating_tools=panel.MUTATING_TOOLS,
            contract=spec.contract,
            aliases=aliases,
        )
        labels = evaluation.labels.model_dump(mode="json")
        for name, value in labels.items():
            factor_counts[name][value] += 1
        if labels["overall"] != "PASS":
            failed_tasks.append({"row_id": spec.row_id, "labels": labels})
        for call in evaluation.executed_calls:
            if not call.executed_successfully:
                unsuccessful_calls.append(
                    {
                        "row_id": spec.row_id,
                        "function": call.function,
                        "replay_error": call.replay_error,
                    }
                )
        if any(
            route.must_precede_first_mutation
            for obligation in spec.contract.evidence_obligations
            for route in obligation.routes
        ):
            condition_first_contracts += 1
    return {
        "tasks": len(panel.TASK_SPECS),
        "factor_counts": {
            name: dict(sorted(counts.items())) for name, counts in factor_counts.items()
        },
        "all_ground_truth_overall_pass": not failed_tasks,
        "all_ground_truth_calls_executed_successfully": not unsuccessful_calls,
        "failed_tasks": failed_tasks,
        "unsuccessful_calls": unsuccessful_calls,
        "condition_first_contracts": condition_first_contracts,
    }


def audit(
    *,
    protocol_path: Path,
    greedy_path: Path,
    stochastic_path: Path,
    contracts_path: Path,
    run_plan_path: Path,
    aliases_path: Path,
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    greedy = _load(greedy_path)
    stochastic = _load(stochastic_path)
    contracts = _load(contracts_path)
    run_plan = _load(run_plan_path)
    evaluator_protocol_path = ROOT / protocol["frozen_evaluator"]["protocol"]
    evaluator_protocol = _load(evaluator_protocol_path)
    v1_protocol = _load(ROOT / "configs" / "0727_custom_clean_panel_protocol.json")
    v1_result = _load(ROOT / "reports" / "0727_custom_clean_panel_result_summary.json")

    expected_artifacts = protocol["frozen_artifacts"]
    artifact_paths = {
        "greedy_manifest": greedy_path,
        "stochastic_manifest": stochastic_path,
        "proof_contracts": contracts_path,
        "run_plan": run_plan_path,
    }
    artifact_hashes = {name: _sha256(path) for name, path in artifact_paths.items()}
    artifact_hash_match = all(
        artifact_hashes[name] == expected_artifacts[name]["sha256"]
        for name in artifact_paths
    )
    source_hashes = {
        relpath: _sha256(ROOT / relpath)
        for relpath in expected_artifacts["source_sha256"]
    }
    source_hash_match = source_hashes == expected_artifacts["source_sha256"]

    generated_match = {
        "greedy_manifest": greedy == builder.build_greedy_manifest(),
        "stochastic_manifest": stochastic == builder.build_stochastic_manifest(),
        "proof_contracts": contracts == builder.build_contract_registry(),
        "run_plan": run_plan == builder.build_run_plan(),
    }
    row_forbidden = {"utility", "success", "outcome", "prediction"}
    task_ids = [row["row_id"] for row in greedy["rows"]]
    contract_ids = [contract["task_id"] for contract in contracts["contracts"]]
    ground_truth = _ground_truth_preflight(aliases_path)
    cell_counts = Counter(
        (row["suite"], row["task_difficulty"], row["split"])
        for row in greedy["rows"]
    )
    cells_correct = all(
        cell_counts[(suite, difficulty, "training")] == 2
        and cell_counts[(suite, difficulty, "calibration")] == 1
        and cell_counts[(suite, difficulty, "confirmation")] == 1
        for suite in panel.SUITES
        for difficulty in ("L1", "L2", "L3")
    )
    sampled_ids = {row["row_id"] for row in stochastic["rows"]}
    planned_sampled_ids = set(run_plan["stochastic_task_ids"])
    gates = {
        "v1_no_go_preserved": (
            v1_protocol["experiment"]
            == "llama31_70b_custom_clean_panel_data_sufficiency_fixed_v1"
            and v1_result["frozen_result"]["decision"]
            == "CUSTOM_PANEL_DATA_SUFFICIENCY_NO_GO"
        ),
        "evaluator_v2_regression_pass_and_hash_frozen": (
            evaluator_protocol["frozen_regression_result"]["decision"]
            == "FACTORIZED_EVALUATOR_V2_REGRESSION_PASS"
            and _sha256(evaluator_protocol_path)
            == protocol["frozen_evaluator"]["protocol_sha256"]
        ),
        "protocol_is_preoutcome_and_clean_only": (
            protocol["status"]
            == "preregistered_and_frozen_before_any_panel_v2_victim_outcome"
            and protocol["fixed_budget"]["attack_episodes"] == 0
            and protocol["fixed_budget"]["model_training_runs"] == 0
        ),
        "frozen_artifact_hashes_match": artifact_hash_match,
        "frozen_source_hashes_match": source_hash_match,
        "generated_artifacts_match_source": all(generated_match.values()),
        "exactly_48_unique_new_tasks": len(task_ids) == 48 and len(set(task_ids)) == 48,
        "exactly_48_unique_contracts": len(contract_ids) == 48 and len(set(contract_ids)) == 48,
        "task_contract_coverage_exact": set(task_ids) == set(contract_ids),
        "suite_difficulty_split_cells_are_2_1_1": cells_correct,
        "manifest_rows_are_outcome_blind": not any(
            row_forbidden & set(row) for row in greedy["rows"] + stochastic["rows"]
        ),
        "proof_contracts_are_outcome_blind": not any(
            contract["outcome_labels_present"] for contract in contracts["contracts"]
        ),
        "stochastic_subset_preselected_and_consistent": (
            sampled_ids == planned_sampled_ids == set(builder.STOCHASTIC_TASK_IDS)
            and run_plan["selection_used_greedy_outcomes"] is False
        ),
        "fixed_budget_is_48_plus_96": (
            len(run_plan["episodes"]) == 144
            and sum(row["track"] == "deterministic_greedy" for row in run_plan["episodes"])
            == 48
            and sum(row["track"] == "stochastic_policy" for row in run_plan["episodes"])
            == 96
        ),
        "sampled_episode_seeds_are_independent_per_task": all(
            len(
                {
                    row["episode_seed"]
                    for row in run_plan["episodes"]
                    if row["track"] == "stochastic_policy" and row["row_id"] == row_id
                }
            )
            == 6
            for row_id in sampled_ids
        ),
        "all_ground_truth_calls_execute": ground_truth[
            "all_ground_truth_calls_executed_successfully"
        ],
        "all_ground_truth_factorized_labels_pass": ground_truth[
            "all_ground_truth_overall_pass"
        ],
        "condition_first_contracts_present": ground_truth["condition_first_contracts"]
        >= 12,
        "legacy_scalar_utility_is_nonbinding": (
            greedy["legacy_scalar_utility_binding"] is False
            and all(row["legacy_scalar_utility_binding"] is False for row in greedy["rows"])
        ),
        "attack_h2_and_dreamer_remain_blocked": (
            protocol["attack_eligibility"]["permitted_by_this_protocol"] is False
            and protocol["attack_eligibility"]["dreamer_training_permitted"] is False
            and protocol["attack_eligibility"]["h2_attack_planning_permitted"] is False
        ),
        "new_victim_outcomes_generated_by_preflight": False,
    }
    # This invariant is intentionally represented positively in the binding gate.
    binding_gates = dict(gates)
    binding_gates["no_new_victim_outcomes_generated_by_preflight"] = not binding_gates.pop(
        "new_victim_outcomes_generated_by_preflight"
    )
    passed = all(binding_gates.values())
    return {
        "scope": "label-blind clean-panel-v2 structural and ground-truth preflight",
        "new_victim_runs": 0,
        "new_attack_runs": 0,
        "new_model_training_runs": 0,
        "protocol_sha256": _sha256(protocol_path),
        "artifact_sha256": artifact_hashes,
        "source_sha256": source_hashes,
        "generated_match": generated_match,
        "ground_truth_preflight": ground_truth,
        "gates": binding_gates,
        "passed": passed,
        "decision": (
            "CUSTOM_CLEAN_PANEL_V2_PREFLIGHT_PASS"
            if passed
            else "CUSTOM_CLEAN_PANEL_V2_PREFLIGHT_FAIL"
        ),
        "permission": {
            "submit_fixed_144_clean_episode_plan": passed,
            "attack_data": False,
            "h2_attack_planning": False,
            "dreamer_training": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--greedy", type=Path, required=True)
    parser.add_argument("--stochastic", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        protocol_path=args.protocol,
        greedy_path=args.greedy,
        stochastic_path=args.stochastic,
        contracts_path=args.contracts,
        run_plan_path=args.run_plan,
        aliases_path=args.aliases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "passed": result["passed"],
                "ground_truth_preflight": result["ground_truth_preflight"],
                "failed_gates": [
                    name for name, passed in result["gates"].items() if not passed
                ],
                "permission": result["permission"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
