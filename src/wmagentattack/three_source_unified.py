"""Leakage-audited unification of AgentDojo, ToolSandbox, and InjecAgent.

The adapter intentionally keeps targets and outcome annotations outside the
causal model input.  AgentDojo skill actions and the two function-calling
sources are represented through the same candidate-schema interface.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import read_jsonl
from .multisource_suitability import (
    SUITABILITY_SCHEMA_VERSION,
    TEXT_ACTION,
    candidate_id,
    causal_model_input,
    normalized_goal,
    stable_hash,
)


UNIFIED_SCHEMA_VERSION = "wmagentattack.three_source_unified_action.v1"
COHORT_ORDER = ("original_test", "original_val", "train0", "train1", "train2")
FOLD_COHORTS = (
    ("original_test", "original_val"),
    ("original_val", "train0"),
    ("train0", "train1"),
    ("train1", "train2"),
    ("train2", "original_test"),
)
_FORBIDDEN_CAUSAL_KEYS = {
    "attack_action",
    "attack_family",
    "attack_success",
    "completion",
    "decision",
    "execution",
    "group_id",
    "policy_violation",
    "prompt_sha256",
    "reward",
    "run_seed",
    "runtime_error",
    "security",
    "simulator_audit_only",
    "task_success",
    "target",
    "utility",
    "variant",
}


def _task_name(row: Mapping[str, Any]) -> str:
    domain = row.get("suite", row.get("domain"))
    task = row.get("user_task_id", row.get("task_id"))
    if domain is None or task is None:
        raise ValueError("AgentDojo row lacks domain/task identity")
    return f"{domain}|{task}"


def _agentdojo_cohorts(metadata: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    by_domain_split: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    task_splits: dict[str, str] = {}
    for row in metadata:
        task = _task_name(row)
        split = str(row["task_split"])
        if task in task_splits and task_splits[task] != split:
            raise ValueError(f"AgentDojo task spans original splits: {task}")
        task_splits[task] = split
        domain, _ = task.split("|", 1)
        by_domain_split[domain][split].add(task)

    cohorts = {name: [] for name in COHORT_ORDER}
    for domain in sorted(by_domain_split):
        split_tasks = by_domain_split[domain]
        counts = {name: len(values) for name, values in split_tasks.items()}
        if counts != {"train": 3, "val": 1, "test": 1}:
            raise ValueError(f"unexpected AgentDojo task split for {domain}: {counts}")
        cohorts["original_test"].append(next(iter(split_tasks["test"])))
        cohorts["original_val"].append(next(iter(split_tasks["val"])))
        for index, task in enumerate(sorted(split_tasks["train"])):
            cohorts[f"train{index}"].append(task)
    for name in cohorts:
        cohorts[name] = sorted(cohorts[name])
        if len(cohorts[name]) != 4:
            raise ValueError(f"cohort {name} is not four tasks")
    if len(set().union(*(set(values) for values in cohorts.values()))) != 20:
        raise ValueError("AgentDojo cohorts do not partition 20 tasks")
    return cohorts


def _agentdojo_catalog_and_rows(
    steps: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[str]]]:
    metadata_by_trajectory = {str(row["trajectory_id"]): row for row in metadata}
    if len(metadata_by_trajectory) != len(metadata):
        raise ValueError("duplicate AgentDojo trajectory metadata")
    cohorts = _agentdojo_cohorts(metadata)
    cohort_by_task = {
        task: cohort for cohort, tasks in cohorts.items() for task in tasks
    }
    catalog: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for step in steps:
        trajectory_id = str(step["trajectory_id"])
        meta = metadata_by_trajectory.get(trajectory_id)
        if meta is None:
            raise ValueError(f"missing AgentDojo metadata: {trajectory_id}")
        task_name = _task_name(step)
        if task_name != _task_name(meta):
            raise ValueError(f"AgentDojo task identity mismatch: {trajectory_id}")
        descriptions = step["candidate_skill_descriptions"]
        legal: list[str] = []
        schemas: list[dict[str, Any]] = []
        for skill in step["candidate_skills"]:
            skill = str(skill)
            if skill == "finish":
                continue
            schema = {
                "type": "function",
                "function": {
                    "name": skill,
                    "description": str(descriptions.get(skill, "")),
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            key = candidate_id("agentdojo", skill, schema)
            descriptor = {
                "source": "agentdojo",
                "kind": "tool",
                "function": schema["function"],
            }
            previous = catalog.setdefault(key, descriptor)
            if stable_hash(previous) != stable_hash(descriptor):
                raise ValueError(f"AgentDojo candidate schema conflict: {key}")
            legal.append(key)
            schemas.append(schema)
        text_key = candidate_id("agentdojo", TEXT_ACTION)
        catalog.setdefault(
            text_key,
            {
                "source": "agentdojo",
                "kind": "text_or_stop",
                "function": {
                    "name": TEXT_ACTION,
                    "description": "Return a final textual response without another skill.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        )
        legal.append(text_key)
        selected = str(step["selected_skill"])
        if selected == "finish":
            target = text_key
        else:
            matches = [
                key
                for key, schema in zip(legal[:-1], schemas)
                if schema["function"]["name"] == selected
            ]
            if len(matches) != 1:
                raise ValueError(f"AgentDojo target is not uniquely legal: {selected}")
            target = matches[0]
        previous_skills = list(step.get("previous_skills", []))
        causal = {
            "source": "agentdojo",
            "trusted_goal": str(step["user_goal"]),
            "track": f"agentdojo:{step['domain']}",
            "tool_schemas": schemas,
            "legal_tool_names": [schema["function"]["name"] for schema in schemas],
            "visible_observation": str(step.get("current_observation", "")),
            "visible_prior_tool": (
                str(previous_skills[-1]) if previous_skills else "<START>"
            ),
        }
        rows.append(
            {
                "row_id": f"agentdojo::{trajectory_id}::step{int(step['step_id'])}",
                "source": "agentdojo",
                "task_key": stable_hash(
                    {"source": "agentdojo", "task": task_name}
                ),
                "task_name": task_name,
                "task_cohort": cohort_by_task[task_name],
                "group_id": (
                    f"{step['multiseed_group_id']}::step{int(step['step_id'])}"
                ),
                "repeat_id": trajectory_id,
                "variant": str(meta.get("attack_family", "clean")),
                "causal_model_input": causal,
                "causal_input_fingerprint": stable_hash(causal),
                "legal_candidate_ids": legal,
                "target_candidate_id": target,
                "target_is_tool": selected != "finish",
                "exact_outcome": {
                    "available": False,
                    "execution_error": False,
                    "state_changed": False,
                    "output_nonempty": False,
                },
            }
        )
    return rows, catalog, cohorts


def _aux_catalog_and_rows(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    catalog: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for record in records:
        source = str(record["source"])
        if source not in {"tool_sandbox", "injecagent"}:
            continue
        causal = causal_model_input(record)
        legal: list[str] = []
        for schema in causal["tool_schemas"]:
            name = str(schema["function"]["name"])
            key = candidate_id(source, name, schema)
            descriptor = {
                "source": source,
                "kind": "tool",
                "function": schema["function"],
            }
            previous = catalog.setdefault(key, descriptor)
            if stable_hash(previous) != stable_hash(descriptor):
                raise ValueError(f"auxiliary candidate schema conflict: {key}")
            legal.append(key)
        text_key = candidate_id(source, TEXT_ACTION)
        catalog.setdefault(
            text_key,
            {
                "source": source,
                "kind": "text_or_stop",
                "function": {
                    "name": TEXT_ACTION,
                    "description": "Return a textual response instead of calling a tool.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        )
        legal.append(text_key)
        target_name = (
            str(record["decision"]["name"])
            if record["decision"]["kind"] == "tool_call"
            else TEXT_ACTION
        )
        if target_name == TEXT_ACTION:
            target = text_key
        else:
            matches = [
                key
                for key, schema in zip(legal[:-1], causal["tool_schemas"])
                if schema["function"]["name"] == target_name
            ]
            if len(matches) != 1:
                raise ValueError(f"auxiliary target is not uniquely legal: {target_name}")
            target = matches[0]
        execution = record["execution"]
        exact = execution.get("tier") == "exact"
        replica = execution.get("replica_0", {}) if exact else {}
        rows.append(
            {
                "row_id": str(record["row_id"]),
                "source": source,
                "task_key": stable_hash(
                    {
                        "source": source,
                        "trusted_goal": normalized_goal(causal["trusted_goal"]),
                    }
                ),
                "task_name": str(record["group_id"]),
                "task_cohort": "auxiliary_training_only",
                "group_id": str(record["group_id"]),
                "repeat_id": str(record.get("run_seed", "unknown")),
                "variant": str(record["variant"]),
                "causal_model_input": causal,
                "causal_input_fingerprint": stable_hash(causal),
                "legal_candidate_ids": legal,
                "target_candidate_id": target,
                "target_is_tool": target_name != TEXT_ACTION,
                "exact_outcome": {
                    "available": exact,
                    "execution_error": bool(exact and replica.get("status") == "error"),
                    "state_changed": bool(exact and replica.get("state_changed") is True),
                    "output_nonempty": bool(exact and replica.get("output") is not None),
                },
            }
        )
    return rows, catalog


def _merge_catalogs(*catalogs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for catalog in catalogs:
        for key, descriptor in catalog.items():
            previous = output.setdefault(key, dict(descriptor))
            if stable_hash(previous) != stable_hash(descriptor):
                raise ValueError(f"candidate catalog conflict: {key}")
    return dict(sorted(output.items()))


def build_three_source_dataset(
    *,
    agentdojo_root: Path,
    base_records: Sequence[Mapping[str, Any]],
    replication_records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    steps = read_jsonl(agentdojo_root / "steps.jsonl")
    metadata = read_jsonl(agentdojo_root / "metadata.jsonl")
    ad_rows, ad_catalog, cohorts = _agentdojo_catalog_and_rows(steps, metadata)
    aux_rows, aux_catalog = _aux_catalog_and_rows(
        [*base_records, *replication_records]
    )
    rows = sorted([*ad_rows, *aux_rows], key=lambda row: row["row_id"])
    catalog = _merge_catalogs(ad_catalog, aux_catalog)

    source_counts = Counter(str(row["source"]) for row in rows)
    expected = protocol["sources"]["expected_action_rows"]
    row_ids = [str(row["row_id"]) for row in rows]
    ad_tasks = {row["task_name"] for row in ad_rows}
    fold_surfaces = []
    test_counts: Counter[str] = Counter()
    for fold, (test_cohort, validation_cohort) in enumerate(FOLD_COHORTS):
        test_tasks = set(cohorts[test_cohort])
        validation_tasks = set(cohorts[validation_cohort])
        train_tasks = ad_tasks - test_tasks - validation_tasks
        test_counts.update(test_tasks)
        fold_surfaces.append(
            {
                "fold": fold,
                "test_cohort": test_cohort,
                "validation_cohort": validation_cohort,
                "train_tasks": sorted(train_tasks),
                "validation_tasks": sorted(validation_tasks),
                "test_tasks": sorted(test_tasks),
            }
        )

    endpoint_calls = 0
    llm_contracts = set()
    for record in [*base_records, *replication_records]:
        if record.get("source") not in {"tool_sandbox", "injecagent"}:
            continue
        llm_contracts.add(str(record.get("llm_contract_sha256")))
        endpoint_calls += int(record.get("execution", {}).get("real_external_endpoint_calls", 0) or 0)

    ad_goal_norms = {normalized_goal(row["causal_model_input"]["trusted_goal"]) for row in ad_rows}
    aux_goal_norms = {normalized_goal(row["causal_model_input"]["trusted_goal"]) for row in aux_rows}
    forbidden = sorted(
        {
            key
            for row in rows
            for key in row["causal_model_input"]
            if key in _FORBIDDEN_CAUSAL_KEYS
        }
    )
    checks = {
        "expected_action_rows": dict(sorted(source_counts.items()))
        == dict(sorted((str(k), int(v)) for k, v in expected.items())),
        "unique_row_ids": len(row_ids) == len(set(row_ids)),
        "target_is_legal": all(
            row["target_candidate_id"] in row["legal_candidate_ids"] for row in rows
        ),
        "agentdojo_has_20_tasks": len(ad_tasks) == 20,
        "five_task_disjoint_folds": all(
            len(surface["train_tasks"]) == 12
            and len(surface["validation_tasks"]) == 4
            and len(surface["test_tasks"]) == 4
            and not (
                set(surface["train_tasks"])
                & set(surface["validation_tasks"])
                | set(surface["train_tasks"])
                & set(surface["test_tasks"])
                | set(surface["validation_tasks"])
                & set(surface["test_tasks"])
            )
            for surface in fold_surfaces
        ),
        "each_agentdojo_task_test_once": set(test_counts.values()) == {1}
        and set(test_counts) == ad_tasks,
        "zero_exact_goal_overlap_across_target_and_aux": not (ad_goal_norms & aux_goal_norms),
        "single_aux_llm_contract": len(llm_contracts) == 1
        and next(iter(llm_contracts), None)
        == protocol["sources"]["shared_aux_llm_contract_sha256"],
        "zero_real_external_endpoint_calls": endpoint_calls == 0,
        "forbidden_causal_keys_absent": not forbidden,
    }
    audit = {
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "source_rows": dict(sorted(source_counts.items())),
        "source_tasks": {
            source: len({row["task_key"] for row in rows if row["source"] == source})
            for source in sorted(source_counts)
        },
        "source_groups": {
            source: len({row["group_id"] for row in rows if row["source"] == source})
            for source in sorted(source_counts)
        },
        "source_repeat_ids": {
            source: len({row["repeat_id"] for row in rows if row["source"] == source})
            for source in sorted(source_counts)
        },
        "tool_rows": dict(
            sorted(
                Counter(row["source"] for row in rows if row["target_is_tool"]).items()
            )
        ),
        "text_rows": dict(
            sorted(
                Counter(row["source"] for row in rows if not row["target_is_tool"]).items()
            )
        ),
        "candidate_count": len(catalog),
        "agentdojo_cohorts": cohorts,
        "folds": fold_surfaces,
        "exact_goal_overlap_count": len(ad_goal_norms & aux_goal_norms),
        "forbidden_causal_keys": forbidden,
        "aux_llm_contracts": sorted(llm_contracts),
        "real_external_endpoint_calls": endpoint_calls,
        "counterevidence": {
            "agentdojo_action_abstraction": "semantic skills plus TEXT",
            "auxiliary_action_abstraction": "raw function schemas plus TEXT",
            "adjacent_transition_claim": "not asserted by this action-only experiment",
            "cross_source_candidate_alignment_is_learned_from_descriptions_not_ids": True,
        },
    }
    dataset = {
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "compatible_action_schema": SUITABILITY_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "rows": rows,
        "candidate_catalog": catalog,
        "agentdojo_cohorts": cohorts,
        "folds": fold_surfaces,
    }
    audit["dataset_content_sha256"] = stable_hash(dataset)
    return dataset, audit

