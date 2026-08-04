"""Run the preregistered, label-blind Travel ledger v2 gold gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.clean_state_instrumentation import canonical_call_signature
from wmagentattack.decision_state import canonical_json_value
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
    "expert_slot_coverage",
    "final_answer",
    "ground_truth_answer",
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


def _sorted_keys(values: Any) -> list[dict[str, Any]]:
    rows = [canonical_json_value(value) for value in values]
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False))


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _canonical_record_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize set-like fields while preserving record/item order."""

    row = canonical_json_value(dict(value))
    row["candidate_entity_keys"] = _sorted_keys(row["candidate_entity_keys"])
    row["attributes"] = sorted(
        row["attributes"],
        key=lambda attribute: json.dumps(
            attribute, sort_keys=True, ensure_ascii=False
        ),
    )
    return row


def _record_projection(record: Any) -> dict[str, Any]:
    return _canonical_record_projection({
        "entity_type": record.entity_type,
        "entity_key": canonical_json_value(record.entity_key),
        "link_status": record.link_status,
        "candidate_entity_keys": _sorted_keys(
            candidate.entity_key for candidate in record.entity_candidates
        ),
        "attributes": [
            {
                "name": attribute.name,
                "value": canonical_json_value(attribute.value),
                "kind": attribute.kind.value,
            }
            for attribute in record.attributes
        ],
        "context": canonical_json_value(record.context),
        "execution_status": record.execution_status,
        "observation_scope": _enum_value(record.observation_scope),
        "state_provenance": record.state_provenance,
    })


def _conflict_projection(conflict: Any) -> dict[str, Any]:
    return {
        "attribute_name": conflict.attribute_name,
        "reason": conflict.reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--test-junit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frozen = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if frozen.get("status") != "preregistered_before_execution":
        raise ValueError("gold fixtures were not preregistered before execution")
    if _forbidden_paths({"cases": frozen["cases"], "linkage_cases": frozen["linkage_cases"]}):
        raise ValueError("gold fixture inputs contain forbidden outcome fields")
    registry = load_adapter_registry(args.registry)
    if registry.benchmark_version != frozen["benchmark_version"]:
        raise ValueError("registry benchmark version differs from gold fixtures")
    if registry.suite != frozen["suite"]:
        raise ValueError("registry suite differs from gold fixtures")
    test_summary = _junit_summary(args.test_junit)

    record_checks = []
    conflict_checks = []
    replay_checks = []
    nonexecuted_update_counts = []
    all_records = []
    all_conflicts = []
    observed_tools = set()
    mismatches = []
    observed_cases = []

    for case in frozen["cases"]:
        ledger = StructuredEvidenceLedgerV2()
        episode_id = f"gold::{case['case_id']}"
        case_rows = []
        for call_index, step in enumerate(case["steps"]):
            status = ExecutionChannelStatus(step["status"])
            before_record_count = len(ledger.records)
            before_conflict_count = len(ledger.conflicts)
            result = update_structured_ledger(
                ledger,
                registry,
                episode_id=episode_id,
                call_index=call_index,
                channel_status=status,
                tool_name=step["tool"],
                arguments=step["arguments"],
                runtime_output=step.get("runtime_output"),
                error_type=step.get("error_type"),
                state_changed=bool(step.get("state_changed", False)),
                proposal_signature=canonical_call_signature(
                    step["tool"], step["arguments"]
                ),
            )
            additions = result.ledger.records[before_record_count:]
            conflicts = result.ledger.conflicts[before_conflict_count:]
            actual_records = [_record_projection(record) for record in additions]
            actual_conflicts = [_conflict_projection(row) for row in conflicts]
            expected_records = [
                _canonical_record_projection(row) for row in step["expected_records"]
            ]
            expected_conflicts = canonical_json_value(step.get("expected_conflicts", []))
            records_match = actual_records == expected_records
            conflicts_match = (
                len(conflicts) == int(step["expected_added_conflicts"])
                and actual_conflicts == expected_conflicts
            )
            record_checks.append(records_match)
            conflict_checks.append(conflicts_match)
            if not records_match or not conflicts_match:
                mismatches.append(
                    {
                        "case_id": case["case_id"],
                        "call_index": call_index,
                        "expected_records": expected_records,
                        "actual_records": actual_records,
                        "expected_conflicts": expected_conflicts,
                        "actual_conflicts": actual_conflicts,
                    }
                )
            replay = update_structured_ledger(
                result.ledger,
                registry,
                episode_id=episode_id,
                call_index=call_index,
                channel_status=status,
                tool_name=step["tool"],
                arguments=step["arguments"],
                runtime_output=step.get("runtime_output"),
                error_type=step.get("error_type"),
                state_changed=bool(step.get("state_changed", False)),
                proposal_signature=canonical_call_signature(
                    step["tool"], step["arguments"]
                ),
            )
            replay_checks.append(replay.ledger == result.ledger)
            if status in {
                ExecutionChannelStatus.PROPOSED,
                ExecutionChannelStatus.TERMINAL_UNEXECUTED,
                ExecutionChannelStatus.CENSORED,
            }:
                nonexecuted_update_counts.append(len(additions))
            all_records.extend(additions)
            all_conflicts.extend(conflicts)
            observed_tools.add(step["tool"])
            ledger = result.ledger
            case_rows.append(
                {
                    "call_index": call_index,
                    "tool": step["tool"],
                    "status": status.value,
                    "records": actual_records,
                    "conflicts": actual_conflicts,
                }
            )
        observed_cases.append({"case_id": case["case_id"], "steps": case_rows})

    linkage_checks = []
    linkage_records = []
    provisional_ids = []
    for record_index, case in enumerate(frozen["linkage_cases"]):
        record = build_item_linkage_record(
            family="hotel",
            entity_type="hotel",
            episode_id="gold::item_linkage",
            call_index=0,
            record_index=record_index,
            source_tool="gold_linker_fixture",
            source_arguments={},
            attributes=(
                ItemAttributeInput(
                    name="price",
                    value=180.0,
                    kind=AttributeKind.SINGLE_VALUED,
                ),
            ),
            entity_key=case.get("entity_key"),
            candidate_entity_keys=case["candidate_entity_keys"],
        )
        actual = {
            "link_status": record.link_status,
            "entity_key": canonical_json_value(record.entity_key),
            "candidate_entity_keys": _sorted_keys(
                candidate.entity_key for candidate in record.entity_candidates
            ),
            "has_provisional_id": record.provisional_entity_id is not None,
        }
        expected = {
            "link_status": case["expected_link_status"],
            "entity_key": canonical_json_value(case["expected_entity_key"]),
            "candidate_entity_keys": _sorted_keys(
                case["expected_candidate_entity_keys"]
            ),
            "has_provisional_id": bool(case["expected_has_provisional_id"]),
        }
        linkage_checks.append(actual == expected)
        if actual != expected:
            mismatches.append(
                {
                    "linkage_case_id": case["case_id"],
                    "expected": expected,
                    "actual": actual,
                }
            )
        if record.provisional_entity_id is not None:
            provisional_ids.append(record.provisional_entity_id)
        linkage_records.append(
            {"case_id": case["case_id"], "projection": actual}
        )
        all_records.append(record)

    registry_families = {
        registry.adapters[tool].family for tool in observed_tools
    }
    registry_modes = {
        registry.adapters[tool].mode.value for tool in observed_tools
    }
    unlinked_conflicts = sum(
        conflict.entity_id.startswith("PROVISIONAL::") for conflict in all_conflicts
    )
    forbidden_output_paths = _forbidden_paths(
        {"cases": observed_cases, "linkage_records": linkage_records}
    )
    gates = {
        "all_registry_families_covered": registry_families
        == set(frozen["expected_registry_families"]),
        "all_adapter_modes_covered": registry_modes
        == set(frozen["expected_adapter_modes"]),
        "gold_record_projection_exact": all(record_checks),
        "gold_conflict_projection_exact": all(conflict_checks),
        "item_linkage_projection_exact": (
            all(linkage_checks) and len(provisional_ids) == len(set(provisional_ids))
        ),
        "deterministic_replay": all(replay_checks),
        "nonexecuted_updates_zero": sum(nonexecuted_update_counts) == 0,
        "unlinked_derived_conflicts_zero": unlinked_conflicts == 0,
        "future_or_outcome_fields_zero": len(forbidden_output_paths) == 0,
        "all_records_victim_observed": all(
            _enum_value(record.observation_scope) == "VICTIM_OBSERVED"
            for record in all_records
        ),
        "tests_pass": (
            test_summary["tests"] > 0
            and test_summary["failures"] == 0
            and test_summary["errors"] == 0
        ),
    }
    if frozen["frozen_gates"] != {name: True for name in gates}:
        raise ValueError("runtime gate names differ from frozen gold protocol")
    decision = frozen["pass_decision"] if all(gates.values()) else frozen["failure_decision"]
    audit = {
        "fixture_id": frozen["fixture_id"],
        "decision": decision,
        "gates": gates,
        "counts": {
            "gold_cases": len(frozen["cases"]),
            "gold_steps": sum(len(case["steps"]) for case in frozen["cases"]),
            "linkage_cases": len(frozen["linkage_cases"]),
            "observed_records": len(all_records),
            "observed_conflicts": len(all_conflicts),
            "unlinked_derived_conflicts": unlinked_conflicts,
            "mismatches": len(mismatches),
        },
        "covered_families": sorted(registry_families),
        "covered_modes": sorted(registry_modes),
        "test_summary": test_summary,
        "mismatches": mismatches,
        "safety": {
            "outcome_labels_read": False,
            "expert_trajectory_read": False,
            "victim_model_calls": 0,
            "attacks": 0,
            "dreamer": False,
            "old_90_model_comparison": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "observed_projections.json", observed_cases)
    _write_json(args.output_dir / "observed_linkage.json", linkage_records)
    _write_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
