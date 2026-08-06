"""Label-blind panel selection and frozen gates for the tau3 horizon pilot."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def largest_remainder_quotas(
    counts: Mapping[str, int], total: int
) -> dict[str, int]:
    """Allocate an integer sample while preserving frozen strata proportionally."""

    available = sum(int(value) for value in counts.values())
    if total <= 0 or total > available:
        raise ValueError("invalid quota total")
    ideals = {
        name: total * int(count) / available for name, count in counts.items()
    }
    quotas = {name: math.floor(value) for name, value in ideals.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda name: (-(ideals[name] - quotas[name]), name),
    )
    for name in order[:remainder]:
        quotas[name] += 1
    if any(quotas[name] > int(counts[name]) for name in counts):
        raise ValueError("quota exceeds an available stratum")
    return quotas


def select_horizon_panel(
    parent_manifest: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the preregistered panel without reading any trajectory outcome."""

    panel = protocol["pilot_panel"]
    seeds = sorted(int(seed) for seed in panel["seeds"])
    rows = list(parent_manifest["rows"])
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_rows[str(row["task_key"])].append(dict(row))

    selected_tasks: list[str] = []
    quota_by_domain: dict[str, dict[str, int]] = {}
    available_by_domain: dict[str, dict[str, int]] = {}
    for domain in panel["domains"]:
        domain_tasks = {
            key: values
            for key, values in task_rows.items()
            if {str(item["domain"]) for item in values} == {str(domain)}
        }
        strata: dict[str, list[str]] = defaultdict(list)
        for task_key, values in domain_tasks.items():
            task_seeds = sorted(int(item["llm_seed"]) for item in values)
            if task_seeds != seeds:
                raise ValueError(f"task seed surface differs: {task_key}")
            names = {str(item["structural_stratum"]) for item in values}
            if len(names) != 1:
                raise ValueError(f"task stratum differs across seeds: {task_key}")
            strata[next(iter(names))].append(task_key)
        counts = {name: len(keys) for name, keys in sorted(strata.items())}
        quotas = largest_remainder_quotas(
            counts, int(panel["tasks_per_domain"])
        )
        available_by_domain[str(domain)] = counts
        quota_by_domain[str(domain)] = quotas
        for stratum, quota in sorted(quotas.items()):
            ranked = sorted(
                strata[stratum],
                key=lambda task_key: stable_hash(
                    {
                        "protocol_id": protocol["protocol_id"],
                        "domain": domain,
                        "stratum": stratum,
                        "task_key": task_key,
                    }
                ),
            )
            selected_tasks.extend(ranked[:quota])

    selected_set = set(selected_tasks)
    role_hash = stable_hash(
        {
            "parent_role_contract_sha256": parent_manifest[
                "role_contract_sha256"
            ],
            "single_mutable_mechanism": protocol["single_mutable_mechanism"],
        }
    )
    selected_rows = []
    for row in rows:
        if str(row["task_key"]) not in selected_set:
            continue
        candidate = copy.deepcopy(row)
        candidate["grandparent_episode_id"] = row.get("parent_episode_id")
        candidate["parent_episode_id"] = row["episode_id"]
        candidate["role_contract_sha256"] = role_hash
        selected_rows.append(candidate)
    selected_rows.sort(
        key=lambda row: (
            str(row["domain"]),
            str(row["task_key"]),
            int(row["llm_seed"]),
        )
    )

    split_tasks: dict[str, set[str]] = defaultdict(set)
    domains: Counter[str] = Counter()
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected_rows:
        split_tasks[str(row["experimental_split"])].add(str(row["task_key"]))
        domains[str(row["domain"])] += 1
        strata[str(row["domain"])][str(row["structural_stratum"])] += 1
    split_names = ("training", "calibration", "confirmation")
    overlaps = {
        f"{left}::{right}": sorted(split_tasks[left] & split_tasks[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    }
    manifest = {
        "schema_version": parent_manifest["schema_version"],
        "protocol_id": protocol["protocol_id"],
        "source_commit": parent_manifest["source_commit"],
        "parent_manifest_sha256": protocol["parent_result"][
            "manifest_sha256"
        ],
        "selection_contract_sha256": stable_hash(panel),
        "shared_model_identity_sha256": parent_manifest[
            "shared_model_identity_sha256"
        ],
        "role_contract_sha256": role_hash,
        "real_external_endpoint_calls": 0,
        "rows": selected_rows,
    }
    expected_episodes = int(panel["episodes"])
    expected_tasks = int(panel["tasks"])
    checks = {
        "expected_rows": len(selected_rows) == expected_episodes,
        "expected_tasks": len(selected_set) == expected_tasks,
        "expected_domain_rows": all(
            domains[str(domain)]
            == int(panel["tasks_per_domain"]) * len(seeds)
            for domain in panel["domains"]
        ),
        "both_frozen_seeds_per_task": all(
            sorted(
                int(row["llm_seed"])
                for row in selected_rows
                if str(row["task_key"]) == task_key
            )
            == seeds
            for task_key in selected_set
        ),
        "unique_episode_ids": len(selected_rows)
        == len({str(row["episode_id"]) for row in selected_rows}),
        "parent_episode_identity_preserved": all(
            row["episode_id"] == row["parent_episode_id"]
            for row in selected_rows
        ),
        "zero_task_overlap": not any(overlaps.values()),
        "single_shared_model_identity": len(
            {row["shared_model_identity_sha256"] for row in selected_rows}
        )
        == 1,
        "single_candidate_role_contract": len(
            {row["role_contract_sha256"] for row in selected_rows}
        )
        == 1,
        "zero_real_external_endpoints": manifest[
            "real_external_endpoint_calls"
        ]
        == 0,
    }
    audit = {
        "checks": checks,
        "passed": all(checks.values()),
        "rows": len(selected_rows),
        "tasks": len(selected_set),
        "selected_task_keys": sorted(selected_set),
        "domain_episode_counts": dict(sorted(domains.items())),
        "available_task_strata": available_by_domain,
        "selected_task_quotas": quota_by_domain,
        "selected_episode_strata": {
            domain: dict(sorted(values.items()))
            for domain, values in sorted(strata.items())
        },
        "split_task_counts": {
            split: len(split_tasks[split]) for split in split_names
        },
        "split_task_overlaps": overlaps,
        "selection_inputs_used": list(panel["selection_inputs_allowed"]),
        "forbidden_outcome_inputs_read": [],
        "manifest_content_sha256": stable_hash(manifest),
    }
    return manifest, audit


def compare_parent_prefixes(
    candidate: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify that extending the cap leaves every observed parent prefix fixed."""

    details: dict[str, Any] = {}
    all_equal = True
    for name in ("agent_decisions", "user_generations", "combined_tool_events"):
        parent_rows = list(parent[name])
        candidate_rows = list(candidate[name])
        equal = len(candidate_rows) >= len(parent_rows) and stable_hash(
            candidate_rows[: len(parent_rows)]
        ) == stable_hash(parent_rows)
        details[name] = {
            "parent_length": len(parent_rows),
            "candidate_length": len(candidate_rows),
            "prefix_equal": equal,
        }
        all_equal = all_equal and equal
    details["all_equal"] = all_equal
    return details


def evaluate_horizon_gate(
    metrics: Mapping[str, Any],
    integrity: Mapping[str, bool],
    gate: Mapping[str, Any],
) -> dict[str, bool]:
    """Evaluate every preregistered clause without changing a threshold."""

    checks = {
        "episodes_complete": int(metrics["episodes_complete"])
        == int(gate["expected_complete_episodes"]),
        "runtime_failures": int(metrics["runtime_failures"]) == 0,
        "agent_private_scenario_exposures": int(
            metrics["agent_private_scenario_exposures"]
        )
        == 0,
        "real_external_endpoint_calls": int(
            metrics["real_external_endpoint_calls"]
        )
        == 0,
        "forced_budget_stop_episodes": int(
            metrics["forced_budget_stop_episodes"]
        )
        <= int(gate["maximum_forced_budget_stop_episodes"]),
        "relative_forced_stop_reduction": float(
            metrics["relative_forced_stop_reduction_vs_parent"]
        )
        >= float(gate["minimum_relative_forced_stop_reduction_vs_paired_parent"]),
        "adjacent_assistant_tool_transitions": int(
            metrics["adjacent_assistant_tool_transitions"]
        )
        >= int(gate["minimum_adjacent_assistant_tool_transitions"]),
        "state_changed_assistant_transitions": int(
            metrics["state_changed_assistant_transitions"]
        )
        >= int(gate["minimum_state_changed_assistant_transitions"]),
        "state_unchanged_assistant_transitions": int(
            metrics["state_unchanged_assistant_transitions"]
        )
        >= int(gate["minimum_state_unchanged_assistant_transitions"]),
        "tasks_with_state_changed_assistant_transition": int(
            metrics["tasks_with_state_changed_assistant_transition"]
        )
        >= int(gate["minimum_tasks_with_state_changed_assistant_transition"]),
        "domains_with_state_changed_assistant_transition": int(
            metrics["domains_with_state_changed_assistant_transition"]
        )
        >= int(gate["minimum_domains_with_state_changed_assistant_transition"]),
        "paired_state_changed_transition_gain": int(
            metrics["paired_state_changed_transition_gain"]
        )
        >= int(gate["minimum_paired_state_changed_transition_gain"]),
        "supported_transition_targets": int(
            metrics["supported_transition_targets"]
        )
        >= int(gate["minimum_supported_transition_targets"]),
        "assistant_tool_error_rate_non_regression": float(
            metrics["assistant_tool_error_rate_increase_over_parent"]
        )
        <= float(gate["maximum_assistant_tool_error_rate_increase_over_paired_parent"]),
        "parent_prefix_equivalence": bool(
            integrity["all_parent_prefixes_equivalent"]
        ),
        "label_blind_panel_selection": bool(
            integrity["label_blind_panel_selection"]
        ),
    }
    checks.update(
        {f"integrity::{name}": bool(value) for name, value in integrity.items()}
    )
    return checks
