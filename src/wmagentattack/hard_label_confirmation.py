"""Deterministic hard-label and held-out split view over the frozen v20 union."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .decision_state import canonical_json_value


HARD_LABEL_SCHEMA_VERSION = "wmagentattack.hard_label_confirmation.v21"
TOOL_FAMILIES = ("query_read", "create_send_reserve", "mutation")
SOURCE_FOLDS = ("v17_legal_fork", "v18_parameter_boundary", "v19_persistence_conflict")


def tool_family(tool_id: str) -> str:
    leaf = str(tool_id).rsplit("::", 1)[-1]
    if leaf.startswith(("get_", "search_", "read_")):
        return "query_read"
    if leaf.startswith(("create_", "send_", "reserve_")):
        return "create_send_reserve"
    if leaf.startswith(("update_", "share_", "add_")):
        return "mutation"
    raise ValueError(f"unregistered v21 tool family: {leaf}")


def hard_effect_tokens(tokens: list[str]) -> list[str]:
    """Remove only the mechanically action-implied source-tool target."""

    return sorted(token for token in tokens if not token.startswith("source="))


def _refs(rows: list[Mapping[str, Any]]) -> list[str]:
    return sorted(str(row["transition_ref"]) for row in rows)


def build_hard_label_confirmation(
    union: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if union.get("schema_version") != "wmagentattack.intervention_union.v20":
        raise ValueError("v21 requires the frozen v20 intervention union")
    rows = copy.deepcopy(list(union["transitions"]))
    removed = 0
    family_counts = Counter()
    for row in rows:
        original = list(row["model_target"]["effect_tokens"])
        filtered = hard_effect_tokens(original)
        removed += len(original) - len(filtered)
        if not filtered:
            raise ValueError("hard-label filtering removed every positive target")
        row["model_target"]["effect_tokens"] = filtered
        family_counts[tool_family(row["model_input"]["normalized_action"]["tool_id"])] += 1

    task_splits = {}
    for fold in range(3):
        test = [row for row in rows if int(row["confirmation_fold"]) == fold]
        train = [row for row in rows if int(row["confirmation_fold"]) != fold]
        task_splits[str(fold)] = {"train_refs": _refs(train), "test_refs": _refs(test)}
    family_splits = {}
    for family in TOOL_FAMILIES:
        test = [
            row for row in rows
            if tool_family(row["model_input"]["normalized_action"]["tool_id"]) == family
        ]
        train = [
            row for row in rows
            if tool_family(row["model_input"]["normalized_action"]["tool_id"]) != family
        ]
        family_splits[family] = {"train_refs": _refs(train), "test_refs": _refs(test)}
    source_splits = {}
    for source in SOURCE_FOLDS:
        test = [row for row in rows if source in row["source_versions"]]
        train = [row for row in rows if source not in row["source_versions"]]
        source_splits[source] = {"train_refs": _refs(train), "test_refs": _refs(test)}

    vocabulary = sorted({
        token for row in rows for token in row["model_target"]["effect_tokens"]
    })
    categories = Counter(
        token.split("=", 1)[0]
        for row in rows for token in row["model_target"]["effect_tokens"]
    )
    split_overlap = {}
    for split_type, splits in (
        ("task", task_splits), ("tool_family", family_splits), ("source", source_splits)
    ):
        for name, split in splits.items():
            overlap = sorted(set(split["train_refs"]) & set(split["test_refs"]))
            if overlap:
                split_overlap[f"{split_type}:{name}"] = overlap
    source_counts = {
        source: len(source_splits[source]["test_refs"]) for source in SOURCE_FOLDS
    }
    audit = {
        "rows": len(rows),
        "hard_vocabulary_size": len(vocabulary),
        "removed_source_token_occurrences": removed,
        "tool_family_counts": dict(sorted(family_counts.items())),
        "source_test_counts": source_counts,
        "task_test_counts": {
            name: len(split["test_refs"]) for name, split in task_splits.items()
        },
        "positive_category_occurrences": dict(sorted(categories.items())),
        "split_overlap": split_overlap,
        "all_rows_retain_positive_targets": all(
            bool(row["model_target"]["effect_tokens"]) for row in rows
        ),
        "no_source_tokens_remain": all(
            not token.startswith("source=")
            for row in rows for token in row["model_target"]["effect_tokens"]
        ),
        "model_inputs_unchanged": all(
            row["model_input"] == original["model_input"]
            for row, original in zip(rows, union["transitions"], strict=True)
        ),
    }
    dataset = {
        "schema_version": HARD_LABEL_SCHEMA_VERSION,
        "scope": "clean-only v20 hard-label confirmation and diagnostic held-out views",
        "loader_contract": {
            "model_input_is_byte_equivalent_to_v20": True,
            "source_tool_target_tokens_removed": True,
            "task_tool_family_and_source_splits_are_audit_only": True,
            "no_attack_utility_security_or_planning_labels": True,
        },
        "effect_token_vocabulary": vocabulary,
        "transitions": rows,
        "split_manifest": {
            "task_disjoint": task_splits,
            "tool_family_heldout_diagnostic": family_splits,
            "source_heldout_diagnostic": source_splits,
        },
    }
    return canonical_json_value(dataset), canonical_json_value(audit)
