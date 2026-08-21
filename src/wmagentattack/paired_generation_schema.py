"""Leakage-blind four-table contract for paired multi-source generation.

The contract deliberately separates per-step dynamics from episode-level
task/attack outcomes.  InjecAgent observation contrasts remain auxiliary and
cannot silently become exact environment transitions or joint-outcome gold.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


SOURCES = {"agentdojo", "tool_sandbox", "injecagent"}
TIERS = {"exact", "observed", "observation_contrast"}
JOINT_CLASSES = {
    "attack0_utility0",
    "attack0_utility1",
    "attack1_utility0",
    "attack1_utility1",
}
FORBIDDEN_TRANSITION_FIELDS = {
    "task_success",
    "attack_success",
    "joint_class",
    "utility",
    "security",
    "source",
    "source_id",
    "task_id",
    "pair_ref",
    "run_seed",
    "condition",
    "attack_payload",
    "final_report",
}


def _keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = set(map(str, value))
        for child in value.values():
            result.update(_keys(child))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: set[str] = set()
        for child in value:
            result.update(_keys(child))
        return result
    return set()


def _unique(rows: list[dict[str, Any]], key: str) -> bool:
    values = [str(row[key]) for row in rows]
    return len(values) == len(set(values))


def audit_paired_generation_tables(
    tables: Mapping[str, list[dict[str, Any]]],
    *,
    expected_episodes: int,
    expected_pairs: int,
    minimum_joint_per_cell: int,
    maximum_joint_class_fraction: float,
) -> dict[str, Any]:
    """Validate integrity without using outcomes to alter records or splits."""

    episodes = list(tables.get("episodes", []))
    transitions = list(tables.get("transitions", []))
    outcomes = list(tables.get("outcomes", []))
    pairs = list(tables.get("pairs", []))
    episode_by_ref = {str(row["episode_ref"]): row for row in episodes}
    outcome_by_ref = {str(row["outcome_ref"]): row for row in outcomes}
    pair_by_ref = {str(row["pair_ref"]): row for row in pairs}
    source_counts = Counter(str(row.get("source")) for row in episodes)
    joint_counts = Counter(
        str(row.get("joint_class")) for row in outcomes
        if row.get("joint_label_valid") is True
    )
    valid_joint = sum(joint_counts.values())
    split_components: dict[str, set[str]] = {}
    for row in episodes:
        split = str(row.get("split", ""))
        split_components.setdefault(split, set()).add(str(row.get("component_ref", "")))
    split_overlap = set()
    names = sorted(split_components)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            split_overlap.update(split_components[left] & split_components[right])
    transition_keys = set()
    for row in transitions:
        transition_keys.update(_keys(row.get("model_input", {})))
        transition_keys.update(_keys(row.get("transition_target", {})))
    checks = {
        "episode_count": len(episodes) == expected_episodes,
        "pair_count": len(pairs) == expected_pairs,
        "unique_episode_refs": _unique(episodes, "episode_ref"),
        "unique_transition_refs": _unique(transitions, "transition_ref"),
        "unique_outcome_refs": _unique(outcomes, "outcome_ref"),
        "unique_pair_refs": _unique(pairs, "pair_ref"),
        "known_sources": set(source_counts) <= SOURCES,
        "episode_outcome_links": all(
            str(row["outcome_ref"]) in outcome_by_ref for row in episodes
        ),
        "transition_episode_links": all(
            str(row["episode_ref"]) in episode_by_ref for row in transitions
        ),
        "pair_episode_links": all(
            str(row["control_episode_ref"]) in episode_by_ref
            and str(row["intervention_episode_ref"]) in episode_by_ref
            for row in pairs
        ),
        "episode_pair_links": all(
            str(row["pair_ref"]) in pair_by_ref for row in episodes
        ),
        "known_transition_tiers": all(
            str(row.get("transition_tier")) in TIERS for row in episodes
        ),
        "injecagent_is_observation_contrast": all(
            row.get("source") != "injecagent"
            or row.get("transition_tier") == "observation_contrast"
            for row in episodes
        ),
        "injecagent_not_joint_gold": all(
            episode_by_ref[str(row["episode_ref"])].get("source") != "injecagent"
            or row.get("joint_label_valid") is False
            for row in outcomes
        ),
        "transition_fields_are_outcome_blind": not (
            transition_keys & FORBIDDEN_TRANSITION_FIELDS
        ),
        "pair_invariants": all(
            row.get("same_initial_state") is True
            and row.get("same_task") is True
            and row.get("same_template") is True
            and row.get("same_seed") is True
            and int(row.get("changed_factor_count", -1)) == 1
            for row in pairs
        ),
        "split_component_overlap": not split_overlap,
        "all_joint_classes_present": set(joint_counts) == JOINT_CLASSES,
        "minimum_joint_cell_support": bool(joint_counts) and min(joint_counts.values()) >= minimum_joint_per_cell,
        "maximum_joint_class_fraction": bool(valid_joint) and max(joint_counts.values()) / valid_joint <= maximum_joint_class_fraction,
    }
    return {
        "schema_version": "wmagentattack.paired_generation_audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "episodes": len(episodes),
            "transitions": len(transitions),
            "outcomes": len(outcomes),
            "pairs": len(pairs),
            "source_counts": dict(sorted(source_counts.items())),
            "joint_counts": dict(sorted(joint_counts.items())),
            "forbidden_transition_fields": sorted(transition_keys & FORBIDDEN_TRANSITION_FIELDS),
            "split_component_overlap": sorted(split_overlap),
        },
    }

