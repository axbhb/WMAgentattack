"""Execute the frozen, label-blind goal-progress v2 gold protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.goal_progress_v2 import (
    CompletionObservation,
    GoalAtomStatus,
    ProgressType,
    assess_goal_progress,
    build_environment_fact,
    compile_goal_plan,
)
from wmagentattack.state_storage_v2 import VisibilityScope
from wmagentattack.structured_ledger_v2 import (
    AttributeKind,
    ExecutionChannelStatus,
    ItemAttributeInput,
    StructuredEvidenceLedgerV2,
    build_item_linkage_record,
    load_adapter_registry,
    update_structured_ledger,
)


FORBIDDEN_KEYS = {
    "utility",
    "security",
    "task_success",
    "attack_success",
    "expert_trajectory",
    "ground_truth",
    "final_answer",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _junit_summary(path: Path) -> dict[str, int]:
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("JUnit report contains no test suites")
    return {
        field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }


def _forbidden_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = (*path, str(key))
            if str(key).lower() in FORBIDDEN_KEYS:
                found.append(".".join(child))
            found.extend(_forbidden_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_paths(item, (*path, str(index))))
    return found


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--test-junit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_execution":
        raise ValueError("goal-progress protocol was not preregistered")
    case_forbidden = _forbidden_paths(protocol["cases"])
    if case_forbidden:
        raise ValueError(f"goal fixtures contain forbidden fields: {case_forbidden}")
    registry = load_adapter_registry(args.registry)
    test_summary = _junit_summary(args.test_junit)

    status_checks = []
    coverage_checks = []
    completion_checks = []
    replay_checks = []
    probability_checks = []
    all_statuses = set()
    all_progress_types = set()
    environment_scopes = []
    ledger_scopes = []
    mismatch_rows = []
    outputs = []
    case_statuses = {}

    for case in protocol["cases"]:
        plan = compile_goal_plan(
            task_id=case["case_id"],
            trusted_goal=case["trusted_goal"],
            tool_schemas=case["tool_schemas"],
            atom_declarations=case["atoms"],
        )
        replayed_plan = compile_goal_plan(
            task_id=case["case_id"],
            trusted_goal=case["trusted_goal"],
            tool_schemas=case["tool_schemas"],
            atom_declarations=case["atoms"],
        )
        plan_replay_match = plan == replayed_plan

        ledger = StructuredEvidenceLedgerV2()
        for call_index, step in enumerate(case["ledger_steps"]):
            ledger = update_structured_ledger(
                ledger,
                registry,
                episode_id=f"goal-gold::{case['case_id']}",
                call_index=call_index,
                channel_status=ExecutionChannelStatus.EXECUTED_SUCCESS,
                tool_name=step["tool"],
                arguments=step["arguments"],
                runtime_output=step["runtime_output"],
            ).ledger
        for record_index, specification in enumerate(case["manual_records"]):
            attributes = tuple(
                ItemAttributeInput(
                    name=row["name"],
                    value=row["value"],
                    kind=AttributeKind(row["kind"]),
                )
                for row in specification["attributes"]
            )
            record = build_item_linkage_record(
                family="gold_fixture",
                entity_type=specification["entity_type"],
                episode_id=f"goal-gold::{case['case_id']}",
                call_index=len(case["ledger_steps"]),
                record_index=record_index,
                source_tool="gold_item_linker",
                source_arguments={},
                attributes=attributes,
                entity_key=specification.get("entity_key"),
                candidate_entity_keys=specification["candidate_entity_keys"],
            )
            ledger = StructuredEvidenceLedgerV2(
                records=(*ledger.records, record),
                conflicts=ledger.conflicts,
                execution_receipts=ledger.execution_receipts,
            )

        environment_facts = tuple(
            build_environment_fact(**row) for row in case["environment_facts"]
        )
        completion = (
            CompletionObservation.model_validate(case["completion_observation"])
            if case["completion_observation"] is not None
            else None
        )
        snapshot = assess_goal_progress(
            plan,
            ledger,
            environment_facts=environment_facts,
            completion_observation=completion,
        )
        replayed_snapshot = assess_goal_progress(
            plan,
            ledger,
            environment_facts=environment_facts,
            completion_observation=completion,
        )
        actual_statuses = {row.atom_id: row.status.value for row in snapshot.assessments}
        expected_statuses = case["expected_statuses"]
        statuses_match = actual_statuses == expected_statuses
        coverage_match = all(
            _close(snapshot.coverage_by_type[key], value)
            for key, value in case["expected_coverage"].items()
        ) and _close(
            snapshot.required_atom_coverage, case["expected_required_coverage"]
        )
        completion_match = (
            snapshot.completion_ready == case["expected_completion_ready"]
        )
        probability_match = all(
            abs(sum(row.status_probabilities.values()) - 1.0) <= 1e-12
            and row.status_probabilities[row.status.value] == 1.0
            for row in snapshot.assessments
        )
        status_checks.append(statuses_match)
        coverage_checks.append(coverage_match)
        completion_checks.append(completion_match)
        replay_checks.append(plan_replay_match and snapshot == replayed_snapshot)
        probability_checks.append(probability_match)
        all_statuses.update(actual_statuses.values())
        all_progress_types.update(atom.progress_type.value for atom in plan.atoms)
        environment_scopes.extend(fact.visibility_scope for fact in environment_facts)
        ledger_scopes.extend(record.observation_scope for record in ledger.records)
        case_statuses[case["case_id"]] = actual_statuses
        if not (statuses_match and coverage_match and completion_match):
            mismatch_rows.append(
                {
                    "case_id": case["case_id"],
                    "expected_statuses": expected_statuses,
                    "actual_statuses": actual_statuses,
                    "expected_coverage": case["expected_coverage"],
                    "actual_coverage": snapshot.coverage_by_type,
                    "expected_required_coverage": case[
                        "expected_required_coverage"
                    ],
                    "actual_required_coverage": snapshot.required_atom_coverage,
                    "expected_completion_ready": case[
                        "expected_completion_ready"
                    ],
                    "actual_completion_ready": snapshot.completion_ready,
                }
            )
        outputs.append(
            {
                "case_id": case["case_id"],
                "plan": plan.model_dump(mode="json"),
                "snapshot": snapshot.model_dump(mode="json"),
            }
        )

    output_forbidden = _forbidden_paths(outputs)
    same_entity_exact = (
        case_statuses["same_entity_supported"]["same"] == "SUPPORTED"
        and case_statuses["split_entity_contradicted"]["same"] == "CONTRADICTED"
    )
    ambiguous_unlinked_distinct = (
        case_statuses["ambiguous_item"]["address"] == "AMBIGUOUS"
        and case_statuses["unlinked_item"]["address"] == "PARTIALLY_SUPPORTED"
    )
    gates = {
        "all_five_atom_statuses_exercised": all_statuses
        == {status.value for status in GoalAtomStatus},
        "three_progress_types_exercised": all_progress_types
        == {progress_type.value for progress_type in ProgressType},
        "same_entity_relation_exact": same_entity_exact,
        "ambiguous_and_unlinked_distinct": ambiguous_unlinked_distinct,
        "gold_statuses_exact": all(status_checks),
        "gold_coverage_exact": all(coverage_checks),
        "completion_readiness_exact": all(completion_checks),
        "deterministic_plan_and_replay": all(replay_checks),
        "probabilities_normalized": all(probability_checks),
        "environment_facts_privileged_only": all(
            scope == VisibilityScope.PLANNER_PRIVILEGED for scope in environment_scopes
        ),
        "ledger_records_victim_observed_only": all(
            scope == VisibilityScope.VICTIM_OBSERVED for scope in ledger_scopes
        ),
        "future_or_outcome_fields_zero": not output_forbidden,
        "tests_pass": (
            test_summary["tests"] > 0
            and test_summary["failures"] == 0
            and test_summary["errors"] == 0
        ),
    }
    if protocol["frozen_gates"] != {name: True for name in gates}:
        raise ValueError("runtime gate names differ from frozen goal protocol")
    decision = protocol["pass_decision"] if all(gates.values()) else protocol["failure_decision"]
    audit = {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "gates": gates,
        "counts": {
            "gold_cases": len(protocol["cases"]),
            "goal_plans": len(outputs),
            "atoms": sum(len(row["atoms"]) for row in protocol["cases"]),
            "ledger_records": len(ledger_scopes),
            "environment_facts": len(environment_scopes),
            "mismatches": len(mismatch_rows),
        },
        "statuses_exercised": sorted(all_statuses),
        "progress_types_exercised": sorted(all_progress_types),
        "test_summary": test_summary,
        "mismatches": mismatch_rows,
        "safety": {
            "expert_trajectory_read": False,
            "future_calls_read": False,
            "outcome_labels_read": False,
            "victim_model_calls": 0,
            "attacks": 0,
            "dreamer": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "plans_and_snapshots.json", outputs)
    _write_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
