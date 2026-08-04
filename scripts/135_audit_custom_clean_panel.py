"""Audit custom clean tasks without reading any victim-model outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.task_suite.load_suites import get_suite
from wmagentattack import custom_agentdojo_panel_v1 as panel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {spec.spec_id: spec for spec in panel.TASK_SPECS}
    rows = manifest["rows"]
    row_ids = {f'{row["suite"]}::{row["user_task_id"]}' for row in rows}
    manifest_matches_registry = row_ids == set(expected)
    family_splits: dict[str, set[str]] = {}
    task_results = []
    for row in rows:
        family_splits.setdefault(row["template_family"], set()).add(row["split"])
        spec_id = f'{row["suite"]}::{row["user_task_id"]}'
        spec = expected[spec_id]
        suite = get_suite(panel.BENCHMARK_VERSION, row["suite"])
        task = suite.get_user_task_by_id(row["user_task_id"])
        tools = {function.name: function for function in suite.tools}
        schema_errors = []
        for call in task.ground_truth(suite.load_and_inject_default_environment({})):
            try:
                tools[call.function].parameters.model_validate(call.args)
            except Exception as error:
                schema_errors.append(f"{call.function}: {type(error).__name__}: {error}")
        environment = suite.load_and_inject_default_environment({})
        utility, security = suite.run_task_with_pipeline(
            GroundTruthPipeline(task),
            task,
            injection_task=None,
            injections={},
            environment=environment,
        )
        negative_environment = suite.load_and_inject_default_environment({})
        rejects_output_without_calls = task.utility_from_traces(
            task.GROUND_TRUTH_OUTPUT,
            negative_environment,
            negative_environment.model_copy(deep=True),
            [],
        ) is False
        task_results.append(
            {
                "spec_id": spec_id,
                "split": row["split"],
                "template_family": row["template_family"],
                "task_spec_sha256_matches": row["task_spec_sha256"] == spec.sha256,
                "registered_spec_matches": getattr(task, "PANEL_SPEC_ID", None) == spec_id,
                "tool_call_schema_errors": schema_errors,
                "ground_truth_utility": bool(utility),
                "ground_truth_security": bool(security),
                "rejects_correct_output_without_required_calls": rejects_output_without_calls,
            }
        )
    forbidden_manifest_keys = {
        "utility",
        "success",
        "outcome",
        "label",
        "prediction",
        "attack_success",
    }
    label_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.casefold() in forbidden_manifest_keys
        }
    )
    gates = {
        "manifest_has_24_unique_registry_tasks": manifest_matches_registry
        and len(rows) == 24,
        "all_template_families_belong_to_one_split": all(
            len(splits) == 1 for splits in family_splits.values()
        ),
        "no_outcome_or_attack_label_fields": not label_fields,
        "all_task_spec_hashes_match": all(
            row["task_spec_sha256_matches"] for row in task_results
        ),
        "all_registered_specs_match": all(
            row["registered_spec_matches"] for row in task_results
        ),
        "all_ground_truth_calls_validate": all(
            not row["tool_call_schema_errors"] for row in task_results
        ),
        "all_ground_truths_pass_utility": all(
            row["ground_truth_utility"] for row in task_results
        ),
        "all_clean_runs_have_security_true": all(
            row["ground_truth_security"] for row in task_results
        ),
        "all_tasks_reject_correct_text_without_required_calls": all(
            row["rejects_correct_output_without_required_calls"]
            for row in task_results
        ),
    }
    passed = all(gates.values())
    return {
        "scope": "label-blind custom AgentDojo panel implementation audit",
        "victim_model_outcomes_read": False,
        "attack_outcomes_read": False,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "counts": {
            "tasks": len(rows),
            "ground_truth_utility_passes": sum(
                row["ground_truth_utility"] for row in task_results
            ),
            "negative_checker_passes": sum(
                row["rejects_correct_output_without_required_calls"]
                for row in task_results
            ),
        },
        "template_family_splits": {
            family: sorted(splits) for family, splits in sorted(family_splits.items())
        },
        "forbidden_manifest_label_fields": label_fields,
        "gates": gates,
        "passed": passed,
        "decision": (
            "CUSTOM_CLEAN_PANEL_IMPLEMENTATION_GATE_PASS"
            if passed
            else "CUSTOM_CLEAN_PANEL_IMPLEMENTATION_GATE_FAIL"
        ),
        "tasks": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "gates": result["gates"], "decision": result["decision"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
