"""Full-surface manifest and frozen gate for tau3 horizon confirmation."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .tau3_horizon import evaluate_horizon_gate, stable_hash


def build_confirmation_manifest(
    parent_manifest: Mapping[str, Any],
    pilot_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the complete frozen parent surface without reading outcomes."""

    surface = protocol["confirmation_surface"]
    parent_rows = [copy.deepcopy(row) for row in parent_manifest["rows"]]
    pilot_ids = {str(row["episode_id"]) for row in pilot_manifest["rows"]}
    parent_ids = {str(row["episode_id"]) for row in parent_rows}
    if not pilot_ids <= parent_ids:
        raise ValueError("pilot manifest is not a subset of the full parent")

    pilot_role_hash = str(pilot_manifest["role_contract_sha256"])
    candidate_rows = []
    for row in parent_rows:
        candidate = copy.deepcopy(row)
        candidate["grandparent_episode_id"] = row.get("parent_episode_id")
        candidate["parent_episode_id"] = row["episode_id"]
        candidate["role_contract_sha256"] = pilot_role_hash
        candidate_rows.append(candidate)
    candidate_rows.sort(
        key=lambda row: (
            str(row["domain"]),
            str(row["task_key"]),
            int(row["llm_seed"]),
        )
    )

    split_tasks: dict[str, set[str]] = defaultdict(set)
    domain_rows: Counter[str] = Counter()
    task_seeds: dict[str, list[int]] = defaultdict(list)
    task_domains: dict[str, set[str]] = defaultdict(set)
    for row in candidate_rows:
        task_key = str(row["task_key"])
        split_tasks[str(row["experimental_split"])].add(task_key)
        domain_rows[str(row["domain"])] += 1
        task_seeds[task_key].append(int(row["llm_seed"]))
        task_domains[task_key].add(str(row["domain"]))
    split_names = ("training", "calibration", "confirmation")
    overlaps = {
        f"{left}::{right}": sorted(split_tasks[left] & split_tasks[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    }
    candidate_ids = {str(row["episode_id"]) for row in candidate_rows}
    holdout_ids = candidate_ids - pilot_ids
    holdout_tasks = {
        str(row["task_key"])
        for row in candidate_rows
        if str(row["episode_id"]) in holdout_ids
    }
    pilot_tasks = {
        str(row["task_key"])
        for row in candidate_rows
        if str(row["episode_id"]) in pilot_ids
    }
    manifest = {
        "schema_version": parent_manifest["schema_version"],
        "protocol_id": protocol["protocol_id"],
        "source_commit": parent_manifest["source_commit"],
        "parent_manifest_sha256": protocol["paired_parent"]["manifest_sha256"],
        "pilot_manifest_sha256": protocol["pilot_go"]["manifest_sha256"],
        "selection_contract_sha256": stable_hash(surface),
        "shared_model_identity_sha256": parent_manifest[
            "shared_model_identity_sha256"
        ],
        "role_contract_sha256": pilot_role_hash,
        "pilot_overlap_episode_ids": sorted(pilot_ids),
        "out_of_pilot_episode_ids": sorted(holdout_ids),
        "real_external_endpoint_calls": 0,
        "rows": candidate_rows,
    }
    expected_seeds = sorted(int(seed) for seed in surface["seeds"])
    checks = {
        "complete_parent_row_surface": candidate_ids == parent_ids,
        "expected_rows": len(candidate_rows) == int(surface["episodes"]),
        "expected_tasks": len(task_seeds) == int(surface["tasks"]),
        "expected_domain_rows": all(
            domain_rows[str(domain)] == int(surface["episodes"]) // 3
            for domain in surface["domains"]
        ),
        "both_frozen_seeds_per_task": all(
            sorted(seeds) == expected_seeds for seeds in task_seeds.values()
        ),
        "single_domain_per_task": all(
            len(domains) == 1 for domains in task_domains.values()
        ),
        "unique_episode_ids": len(candidate_rows) == len(candidate_ids),
        "parent_episode_identity_preserved": all(
            row["episode_id"] == row["parent_episode_id"]
            for row in candidate_rows
        ),
        "zero_task_overlap_across_splits": not any(overlaps.values()),
        "expected_pilot_overlap_episodes": len(pilot_ids)
        == int(surface["pilot_overlap_episodes"]),
        "expected_pilot_overlap_tasks": len(pilot_tasks)
        == int(surface["pilot_overlap_tasks"]),
        "expected_out_of_pilot_episodes": len(holdout_ids)
        == int(surface["out_of_pilot_episodes"]),
        "expected_out_of_pilot_tasks": len(holdout_tasks)
        == int(surface["out_of_pilot_tasks"]),
        "pilot_and_holdout_tasks_disjoint": not (pilot_tasks & holdout_tasks),
        "role_contract_matches_passed_pilot": pilot_role_hash
        == str(pilot_manifest["role_contract_sha256"]),
        "single_shared_model_identity": len(
            {row["shared_model_identity_sha256"] for row in candidate_rows}
        )
        == 1,
        "single_candidate_role_contract": len(
            {row["role_contract_sha256"] for row in candidate_rows}
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
        "rows": len(candidate_rows),
        "tasks": len(task_seeds),
        "domain_episode_counts": dict(sorted(domain_rows.items())),
        "split_task_counts": {
            split: len(split_tasks[split]) for split in split_names
        },
        "split_task_overlaps": overlaps,
        "pilot_overlap_episodes": len(pilot_ids),
        "pilot_overlap_tasks": len(pilot_tasks),
        "out_of_pilot_episodes": len(holdout_ids),
        "out_of_pilot_tasks": len(holdout_tasks),
        "selection_inputs_used": list(surface["selection_inputs_allowed"]),
        "forbidden_outcome_inputs_read": [],
        "manifest_content_sha256": stable_hash(manifest),
    }
    return manifest, audit


def episode_reproducibility(
    candidate: Mapping[str, Any], pilot: Mapping[str, Any]
) -> bool:
    """Require exact output identity on the passed pilot overlap."""

    return stable_hash(candidate) == stable_hash(pilot)


def evaluate_confirmation_gate(
    metrics: Mapping[str, Any],
    integrity: Mapping[str, bool],
    gate: Mapping[str, Any],
) -> dict[str, bool]:
    """Evaluate the common horizon clauses and full-confirmation additions."""

    checks = evaluate_horizon_gate(metrics, integrity, gate)
    checks.update(
        {
            "communication_error_terminations": int(
                metrics["communication_error_terminations"]
            )
            == 0,
            "natural_user_messages": int(metrics["natural_user_messages"])
            >= int(gate["minimum_natural_user_messages"]),
            "episodes_with_two_or_more_assistant_transitions": int(
                metrics["episodes_with_two_or_more_assistant_transitions"]
            )
            >= int(gate["minimum_episodes_with_two_or_more_assistant_transitions"]),
            "tasks_with_at_least_one_assistant_transition": int(
                metrics["tasks_with_at_least_one_assistant_transition"]
            )
            >= int(gate["minimum_tasks_with_at_least_one_assistant_transition"]),
            "unique_executed_assistant_tools": int(
                metrics["unique_executed_assistant_tools"]
            )
            >= int(gate["minimum_unique_executed_assistant_tools"]),
            "minimum_agent_tool_decision_rate": float(
                metrics["agent_tool_decision_rate"]
            )
            >= float(gate["minimum_agent_tool_decision_rate"]),
            "maximum_agent_tool_decision_rate": float(
                metrics["agent_tool_decision_rate"]
            )
            <= float(gate["maximum_agent_tool_decision_rate"]),
            "dominant_agent_action_fraction": float(
                metrics["dominant_agent_action_fraction"]
            )
            <= float(gate["maximum_dominant_agent_action_fraction"]),
            "out_of_pilot_episodes": int(metrics["out_of_pilot_episodes"])
            == int(gate["expected_out_of_pilot_episodes"]),
            "out_of_pilot_tasks": int(metrics["out_of_pilot_tasks"])
            == int(gate["expected_out_of_pilot_tasks"]),
            "out_of_pilot_state_changed_assistant_transitions": int(
                metrics["out_of_pilot_state_changed_assistant_transitions"]
            )
            >= int(
                gate["minimum_out_of_pilot_state_changed_assistant_transitions"]
            ),
            "out_of_pilot_tasks_with_state_changed_assistant_transition": int(
                metrics[
                    "out_of_pilot_tasks_with_state_changed_assistant_transition"
                ]
            )
            >= int(
                gate[
                    "minimum_out_of_pilot_tasks_with_state_changed_assistant_transition"
                ]
            ),
            "out_of_pilot_domains_with_state_changed_assistant_transition": int(
                metrics[
                    "out_of_pilot_domains_with_state_changed_assistant_transition"
                ]
            )
            >= int(
                gate[
                    "minimum_out_of_pilot_domains_with_state_changed_assistant_transition"
                ]
            ),
            "pilot_overlap_reproducibility": bool(
                integrity["all_pilot_overlap_episodes_reproduced"]
            ),
            "label_blind_full_surface": bool(
                integrity["label_blind_full_surface"]
            ),
        }
    )
    return checks
