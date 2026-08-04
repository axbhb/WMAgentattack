"""Evaluate the fixed panel-v2 clean traces and apply independent frozen gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdojo.task_suite.load_suites import get_suite
from wmagentattack import custom_agentdojo_panel_v2 as panel
from wmagentattack.factorized_evaluator_v2 import (
    FactorLabel,
    evaluate_trace,
    load_alias_registry,
)
from wmagentattack.trace_execution_pairing import pair_executed_clean_tool_calls


ERROR_MARKERS = (
    "traceback (most recent call last)",
    "cuda out of memory",
    "outofmemoryerror",
    "runtimeerror: cuda",
    "cublas_status",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_hits(text: str) -> list[str]:
    normalized = text.casefold()
    return [marker for marker in ERROR_MARKERS if marker in normalized]


def _chunk_paths(archive_root: Path, run_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = [
        {
            "track": "deterministic_greedy",
            "run_seed": int(run_plan["greedy_run_seed"]),
            "chunk_index": chunk,
            "num_chunks": 8,
            "path": archive_root / "greedy" / f"chunk{chunk}.json",
        }
        for chunk in range(8)
    ]
    for seed in run_plan["stochastic_run_seeds"]:
        paths.extend(
            {
                "track": "stochastic_policy",
                "run_seed": int(seed),
                "chunk_index": chunk,
                "num_chunks": 2,
                "path": archive_root / "sampled" / f"seed{seed}" / f"chunk{chunk}.json",
            }
            for chunk in range(2)
        )
    return paths


def _load_execution_results(
    archive_root: Path,
    run_plan: Mapping[str, Any],
    *,
    custom_task_module: str,
) -> tuple[dict[tuple[str, int, str], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results: dict[tuple[str, int, str], dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    chunk_audit: list[dict[str, Any]] = []
    for descriptor in _chunk_paths(archive_root, run_plan):
        path = descriptor["path"]
        if not path.is_file():
            errors.append({"type": "missing_chunk", **{k: str(v) if k == "path" else v for k, v in descriptor.items()}})
            continue
        raw_text = path.read_text(encoding="utf-8")
        marker_hits = _marker_hits(raw_text)
        try:
            payload = json.loads(raw_text)
        except Exception as error:
            errors.append({"type": "unreadable_chunk", "path": str(path), "error": f"{type(error).__name__}: {error}"})
            continue
        metadata_ok = (
            int(payload.get("run_seed", -1)) == descriptor["run_seed"]
            and int(payload.get("chunk_index", -1)) == descriptor["chunk_index"]
            and int(payload.get("num_chunks", -1)) == descriptor["num_chunks"]
            and payload.get("custom_task_module") == custom_task_module
        )
        if not metadata_ok:
            errors.append({"type": "chunk_metadata_mismatch", "path": str(path), "expected": {k: v for k, v in descriptor.items() if k != "path"}})
        if marker_hits:
            errors.append({"type": "error_marker_in_chunk", "path": str(path), "markers": marker_hits})
        chunk_audit.append(
            {
                **{k: str(v) if k == "path" else v for k, v in descriptor.items()},
                "metadata_ok": metadata_ok,
                "marker_hits": marker_hits,
                "summary": payload.get("summary"),
            }
        )
        for result in payload.get("results", []):
            row_id = str(result.get("row_id"))
            key = (str(descriptor["track"]), int(descriptor["run_seed"]), row_id)
            if key in results:
                errors.append({"type": "duplicate_result", "key": list(key), "path": str(path)})
                continue
            results[key] = dict(result)
    return results, errors, chunk_audit


def _expected_keys(run_plan: Mapping[str, Any]) -> set[tuple[str, int, str]]:
    return {
        (str(row["track"]), int(row["run_seed"]), str(row["row_id"]))
        for row in run_plan["episodes"]
    }


def _first_assistant_decision_present(trace: Mapping[str, Any]) -> bool:
    return any(message.get("role") == "assistant" for message in trace.get("messages", []))


def _evaluate_completed_results(
    *,
    results: Mapping[tuple[str, int, str], Mapping[str, Any]],
    contracts: Mapping[str, Any],
    aliases_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aliases = load_alias_registry(aliases_path)
    evaluated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    suites: dict[str, Any] = {}
    rows_by_id = {spec.row_id: spec for spec in panel.TASK_SPECS}
    for key in sorted(results):
        track, run_seed, row_id = key
        result = results[key]
        if result.get("status") != "completed":
            errors.append(
                {
                    "type": "failed_result",
                    "track": track,
                    "run_seed": run_seed,
                    "row_id": row_id,
                    "error_type": result.get("error_type"),
                    "error": result.get("error"),
                }
            )
            continue
        trace_path = Path(str(result.get("raw_trace", "")))
        if not trace_path.is_file():
            errors.append({"type": "missing_raw_trace", "track": track, "run_seed": run_seed, "row_id": row_id, "path": str(trace_path)})
            continue
        try:
            raw_text = trace_path.read_text(encoding="utf-8")
            trace = json.loads(raw_text)
        except Exception as error:
            errors.append({"type": "unreadable_raw_trace", "track": track, "run_seed": run_seed, "row_id": row_id, "path": str(trace_path), "error": f"{type(error).__name__}: {error}"})
            continue
        marker_hits = _marker_hits(raw_text)
        if marker_hits:
            errors.append({"type": "error_marker_in_trace", "track": track, "run_seed": run_seed, "row_id": row_id, "path": str(trace_path), "markers": marker_hits})
        spec = rows_by_id.get(row_id)
        if spec is None or row_id not in contracts:
            errors.append({"type": "unknown_task_or_contract", "row_id": row_id})
            continue
        suite = suites.setdefault(spec.suite, get_suite(panel.BENCHMARK_VERSION, spec.suite))
        try:
            evaluation = evaluate_trace(
                trace=trace,
                environment=suite.load_and_inject_default_environment({}),
                tools=suite.tools,
                mutating_tools=panel.MUTATING_TOOLS,
                contract=contracts[row_id],
                aliases=aliases,
            )
            pairing, _ = pair_executed_clean_tool_calls(trace.get("messages", []))
        except Exception as error:
            errors.append({"type": "evaluation_failure", "track": track, "run_seed": run_seed, "row_id": row_id, "error": f"{type(error).__name__}: {error}"})
            continue
        evaluated.append(
            {
                "track": track,
                "run_seed": run_seed,
                "row_id": row_id,
                "suite": spec.suite,
                "split": spec.split,
                "task_difficulty": spec.contract.task_difficulty.value,
                "task_archetype": spec.contract.task_archetype,
                "source_trace": str(trace_path),
                "source_trace_sha256": _sha256(trace_path),
                "legacy_scalar_utility_nonbinding": bool(result.get("utility")),
                "factorized": evaluation.model_dump(mode="json"),
                "pairing": pairing.model_dump(mode="json"),
                "pairing_ok": pairing.executed_alignment_ok,
                "first_assistant_decision_present": _first_assistant_decision_present(trace),
                "error_markers": marker_hits,
            }
        )
    return evaluated, errors


def _factor_counts(rows: Sequence[Mapping[str, Any]], factor: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(row["factorized"]["labels"][factor] for row in rows).items()
        )
    )


def _split_label_counts(
    rows: Sequence[Mapping[str, Any]], factor: str
) -> dict[str, dict[str, int]]:
    return {
        split: _factor_counts([row for row in rows if row["split"] == split], factor)
        for split in panel.SPLITS
    }


def _minimum_pass_fail_gate(
    counts: Mapping[str, Mapping[str, int]],
    thresholds: Mapping[str, Mapping[str, int]],
) -> tuple[bool, dict[str, bool]]:
    conditions = {
        f"{split}_{label}": int(counts.get(split, {}).get(label, 0)) >= int(minimum)
        for split, labels in thresholds.items()
        for label, minimum in labels.items()
    }
    return all(conditions.values()), conditions


def _sampled_task_rows(
    sampled: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sampled:
        grouped[str(row["row_id"])].append(row)
    output = []
    for row_id, rows in sorted(grouped.items()):
        first = rows[0]
        seeds = sorted(int(row["run_seed"]) for row in rows)
        factor_passes = {
            factor: sum(
                row["factorized"]["labels"][factor] == FactorLabel.PASS.value
                for row in rows
            )
            for factor in ("state_action", "evidence", "report", "overall")
        }
        output.append(
            {
                "row_id": row_id,
                "suite": first["suite"],
                "split": first["split"],
                "task_difficulty": first["task_difficulty"],
                "complete_six_seeds": seeds == sorted(int(seed) for seed in expected_seeds),
                "run_seeds": seeds,
                "pass_counts": factor_passes,
                "probabilities": {
                    factor: count / len(expected_seeds)
                    for factor, count in factor_passes.items()
                },
                "overall_interior_probability": 0 < factor_passes["overall"] < len(expected_seeds),
            }
        )
    return output


def summarize(
    *,
    protocol_path: Path,
    greedy_manifest_path: Path,
    stochastic_manifest_path: Path,
    contracts_path: Path,
    run_plan_path: Path,
    aliases_path: Path,
    preflight_path: Path,
    ledger_audit_path: Path,
    archive_root: Path,
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    greedy_manifest = _load(greedy_manifest_path)
    stochastic_manifest = _load(stochastic_manifest_path)
    run_plan = _load(run_plan_path)
    preflight = _load(preflight_path)
    ledger_audit = _load(ledger_audit_path)
    registry = panel.PanelV2ProofContractRegistry.model_validate_json(
        contracts_path.read_text(encoding="utf-8")
    )
    contracts = {contract.task_id: contract for contract in registry.contracts}

    results, load_errors, chunk_audit = _load_execution_results(
        archive_root,
        run_plan,
        custom_task_module=greedy_manifest["custom_task_module"],
    )
    expected_keys = _expected_keys(run_plan)
    missing_keys = sorted(expected_keys - set(results))
    unexpected_keys = sorted(set(results) - expected_keys)
    evaluated, evaluation_errors = _evaluate_completed_results(
        results=results,
        contracts=contracts,
        aliases_path=aliases_path,
    )
    greedy = [row for row in evaluated if row["track"] == "deterministic_greedy"]
    sampled = [row for row in evaluated if row["track"] == "stochastic_policy"]
    sampled_tasks = _sampled_task_rows(sampled, run_plan["stochastic_run_seeds"])

    frozen_hash_match = all(
        _sha256(path) == protocol["frozen_artifacts"][name]["sha256"]
        for name, path in {
            "greedy_manifest": greedy_manifest_path,
            "stochastic_manifest": stochastic_manifest_path,
            "proof_contracts": contracts_path,
            "run_plan": run_plan_path,
        }.items()
    )
    all_errors = load_errors + evaluation_errors
    integrity_conditions = {
        "frozen_artifact_hashes_match": frozen_hash_match,
        "exactly_20_complete_chunks": len(chunk_audit) == 20 and not any(error["type"] in {"missing_chunk", "unreadable_chunk", "chunk_metadata_mismatch"} for error in load_errors),
        "exactly_144_expected_results": len(expected_keys) == 144,
        "zero_missing_episode_keys": not missing_keys,
        "zero_unexpected_episode_keys": not unexpected_keys,
        "exactly_144_evaluated_traces": len(evaluated) == 144,
        "exactly_48_greedy_traces": len(greedy) == 48,
        "exactly_96_sampled_traces": len(sampled) == 96,
        "zero_failed_missing_unreadable_or_evaluation_errors": not all_errors,
        "zero_traceback_oom_cuda_markers": not any(row["error_markers"] for row in evaluated) and not any(audit["marker_hits"] for audit in chunk_audit),
        "zero_attack_episodes": run_plan["budget"]["attack_episodes"] == 0,
        "zero_model_training_runs": run_plan["budget"]["model_training_runs"] == 0,
    }
    execution_integrity_passed = all(integrity_conditions.values())

    pairing_ok = sum(bool(row["pairing_ok"]) for row in greedy)
    classified_tool_messages = sum(
        len(row["pairing"]["executed_pairs"]) for row in greedy
    )
    total_tool_messages = sum(row["pairing"]["tool_message_count"] for row in greedy)
    first_decisions = sum(bool(row["first_assistant_decision_present"]) for row in greedy)
    cells = Counter((row["suite"], row["task_difficulty"], row["split"]) for row in greedy)
    dynamics_conditions = {
        "execution_integrity_passed": execution_integrity_passed,
        "all_48_unique_greedy_tasks": len(greedy) == 48 and len({row["row_id"] for row in greedy}) == 48,
        "all_36_cells_present_2_1_1": all(
            cells[(suite, difficulty, "training")] == 2
            and cells[(suite, difficulty, "calibration")] == 1
            and cells[(suite, difficulty, "confirmation")] == 1
            for suite in panel.SUITES
            for difficulty in ("L1", "L2", "L3")
        ),
        "all_greedy_pairings_valid": pairing_ok == 48,
        "all_tool_messages_classified": classified_tool_messages == total_tool_messages,
        "minimum_48_first_decision_labels": first_decisions >= 48,
        "zero_unexplained_pairing_drops": all(
            not row["pairing"]["midtrajectory_unexecuted_proposals"]
            and not row["pairing"]["orphan_tool_message_indices"]
            and not row["pairing"]["signature_mismatch_tool_message_indices"]
            for row in greedy
        ),
    }
    dynamics_passed = all(dynamics_conditions.values())

    evidence_supported = {
        split: [
            row
            for row in greedy
            if row["split"] == split
            and contracts[row["row_id"]].evidence_applicable
            and row["factorized"]["labels"]["evidence"] == FactorLabel.PASS.value
        ]
        for split in panel.SPLITS
    }
    evidence_unobserved = {
        split: [
            row
            for row in greedy
            if row["split"] == split and contracts[row["row_id"]].evidence_applicable
        ]
        for split in panel.SPLITS
    }
    evidence_spec = protocol["independent_data_gates"]["evidence_progress"]
    supported_thresholds = evidence_spec[
        "minimum_tasks_with_at_least_one_supported_obligation"
    ]
    unobserved_thresholds = evidence_spec[
        "minimum_tasks_with_at_least_one_pre_observation_unobserved_checkpoint"
    ]
    evidence_conditions = {
        "execution_integrity_passed": execution_integrity_passed,
        "preflight_passed": preflight.get("decision") == "CUSTOM_CLEAN_PANEL_V2_PREFLIGHT_PASS" and bool(preflight.get("passed")),
        "proof_registry_has_48_frozen_contracts": len(registry.contracts) == 48 and registry.frozen_before_first_victim_outcome,
        "ledger_v2_extractor_regression_passed": ledger_audit.get("decision") == "LEDGER_V2_EXTRACTOR_GOLD_GATE_PASS" and all(ledger_audit.get("gates", {}).values()),
        **{
            f"minimum_supported_{split}": len(evidence_supported[split]) >= int(supported_thresholds[split])
            for split in panel.SPLITS
        },
        **{
            f"minimum_preobservation_unobserved_{split}": len(evidence_unobserved[split]) >= int(unobserved_thresholds[split])
            for split in panel.SPLITS
        },
        "L1_L2_L3_supported_contributors_in_each_split": all(
            {row["task_difficulty"] for row in evidence_supported[split]}
            == {"L1", "L2", "L3"}
            for split in panel.SPLITS
        ),
    }
    evidence_passed = all(evidence_conditions.values())

    overall_counts = _split_label_counts(greedy, "overall")
    completion_spec = protocol["independent_data_gates"]["completion_overall"]
    completion_balance, completion_balance_conditions = _minimum_pass_fail_gate(
        overall_counts, completion_spec["minimum_pass_and_fail_tasks"]
    )
    completion_conditions = {
        "execution_integrity_passed": execution_integrity_passed,
        **completion_balance_conditions,
    }
    completion_passed = execution_integrity_passed and completion_balance

    evidence_sufficient_reporting = [
        row
        for row in greedy
        if contracts[row["row_id"]].report_applicable
        and (
            not contracts[row["row_id"]].evidence_applicable
            or row["factorized"]["labels"]["evidence"] == FactorLabel.PASS.value
        )
    ]
    reporting_counts = _split_label_counts(evidence_sufficient_reporting, "report")
    reporting_spec = protocol["independent_data_gates"]["conditional_reporting"]
    reporting_balance, reporting_balance_conditions = _minimum_pass_fail_gate(
        reporting_counts, reporting_spec["minimum_report_pass_and_fail_tasks"]
    )
    reporting_conditions = {
        "execution_integrity_passed": execution_integrity_passed,
        **reporting_balance_conditions,
    }
    reporting_passed = execution_integrity_passed and reporting_balance

    interior = [row for row in sampled_tasks if row["overall_interior_probability"]]
    stochastic_spec = protocol["independent_data_gates"]["stochastic_probability"]
    stochastic_conditions = {
        "execution_integrity_passed": execution_integrity_passed,
        "all_16_tasks_have_six_samples": len(sampled_tasks) == 16 and all(row["complete_six_seeds"] for row in sampled_tasks),
        "minimum_four_interior_probability_tasks": len(interior) >= int(stochastic_spec["minimum_tasks_with_interior_probability_0_lt_p_lt_1"]),
        "minimum_two_suites_with_interior_task": len({row["suite"] for row in interior}) >= int(stochastic_spec["minimum_suites_with_an_interior_probability_task"]),
        "minimum_two_splits_with_interior_task": len({row["split"] for row in interior}) >= int(stochastic_spec["minimum_splits_with_an_interior_probability_task"]),
    }
    stochastic_passed = all(stochastic_conditions.values())

    representation_permitted = dynamics_passed and evidence_passed
    permissions = {
        "victim_dynamics_head": dynamics_passed,
        "evidence_progress_head": evidence_passed,
        "three_backbone_dynamics_evidence_ablation": representation_permitted,
        "overall_completion_value_head": completion_passed,
        "conditional_reporting_head": reporting_passed,
        "continuous_probability_head": stochastic_passed,
        "attack_data": False,
        "h2_attack_planning": False,
        "dreamer_training": False,
    }
    return {
        "experiment": protocol["experiment"],
        "scope": "fixed-budget factorized clean-panel-v2 independent gate evaluation",
        "protocol_sha256": _sha256(protocol_path),
        "archive_root": str(archive_root),
        "legacy_agentdojo_scalar_utility_binding": False,
        "execution_integrity": {
            "conditions": integrity_conditions,
            "passed": execution_integrity_passed,
            "expected_episode_keys": len(expected_keys),
            "loaded_results": len(results),
            "evaluated_traces": len(evaluated),
            "missing_keys": [list(key) for key in missing_keys],
            "unexpected_keys": [list(key) for key in unexpected_keys],
            "errors": all_errors,
            "chunks": chunk_audit,
        },
        "greedy": {
            "tasks": len(greedy),
            "factor_counts": {
                factor: _factor_counts(greedy, factor)
                for factor in ("state_action", "evidence", "report", "overall")
            },
            "overall_by_split": overall_counts,
            "overall_by_suite": {
                suite: _factor_counts([row for row in greedy if row["suite"] == suite], "overall")
                for suite in panel.SUITES
            },
            "overall_by_difficulty": {
                difficulty: _factor_counts([row for row in greedy if row["task_difficulty"] == difficulty], "overall")
                for difficulty in ("L1", "L2", "L3")
            },
            "pairing": {
                "valid_tasks": pairing_ok,
                "classified_tool_messages": classified_tool_messages,
                "total_tool_messages": total_tool_messages,
                "first_decision_labels": first_decisions,
            },
        },
        "sampled": {
            "episodes": len(sampled),
            "tasks": len(sampled_tasks),
            "interior_probability_tasks": len(interior),
            "interior_suites": sorted({row["suite"] for row in interior}),
            "interior_splits": sorted({row["split"] for row in interior}),
            "tasks_detail": sampled_tasks,
        },
        "evidence_progress_counts": {
            split: {
                "supported_tasks": len(evidence_supported[split]),
                "preobservation_unobserved_tasks": len(evidence_unobserved[split]),
                "supported_difficulties": sorted({row["task_difficulty"] for row in evidence_supported[split]}),
            }
            for split in panel.SPLITS
        },
        "conditional_reporting_counts": reporting_counts,
        "gates": {
            "dynamics": {"conditions": dynamics_conditions, "passed": dynamics_passed},
            "evidence_progress": {"conditions": evidence_conditions, "passed": evidence_passed},
            "completion_overall": {"conditions": completion_conditions, "passed": completion_passed},
            "conditional_reporting": {"conditions": reporting_conditions, "passed": reporting_passed},
            "stochastic_probability": {"conditions": stochastic_conditions, "passed": stochastic_passed},
        },
        "permissions": permissions,
        "binding_decision": (
            "CUSTOM_PANEL_V2_EXECUTION_INVALID"
            if not execution_integrity_passed
            else "CUSTOM_PANEL_V2_INDEPENDENT_GATES_EVALUATED"
        ),
        "next_action": (
            "Run the frozen small Semantic Markov vs observable-execution vs Ledger-v2 dynamics/evidence ablation; keep all failed heads and all attack/Dreamer work blocked."
            if representation_permitted
            else "Do not run the three-backbone ablation; retain the failed independent gates and diagnose the corresponding clean data mechanism without attacks or Dreamer."
        ),
        "episodes": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--greedy-manifest", type=Path, required=True)
    parser.add_argument("--stochastic-manifest", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--ledger-audit", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        protocol_path=args.protocol,
        greedy_manifest_path=args.greedy_manifest,
        stochastic_manifest_path=args.stochastic_manifest,
        contracts_path=args.contracts,
        run_plan_path=args.run_plan,
        aliases_path=args.aliases,
        preflight_path=args.preflight,
        ledger_audit_path=args.ledger_audit,
        archive_root=args.archive_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "binding_decision": result["binding_decision"],
                "execution_integrity": result["execution_integrity"]["passed"],
                "greedy": result["greedy"],
                "sampled_summary": {
                    key: value for key, value in result["sampled"].items() if key != "tasks_detail"
                },
                "evidence_progress_counts": result["evidence_progress_counts"],
                "conditional_reporting_counts": result["conditional_reporting_counts"],
                "gates": result["gates"],
                "permissions": result["permissions"],
                "next_action": result["next_action"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
