"""Post-hoc mechanism audit for the frozen evidence-ledger NO-GO result.

This script is descriptive only.  It must not rewrite the preregistered gates,
select a new model, or trigger a rerun on the same 90 episodes.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.:/][A-Za-z0-9]+)*")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def _field(line: str, name: str) -> str:
    match = re.search(rf" {re.escape(name)} (\S+)", line)
    return match.group(1) if match else "<MISSING>"


def _span(line: str, left: str, right: str) -> str:
    start = line.find(left)
    end = line.find(right, start + len(left))
    if start < 0 or end < 0:
        return "<MISSING>"
    return line[start + len(left) : end]


def _donors(episodes: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        grouped[episode["task_id"]].append(episode["episode_id"])
    output = {}
    for episode_ids in grouped.values():
        ordered = sorted(episode_ids)
        for index, episode_id in enumerate(ordered):
            output[episode_id] = ordered[(index + 1) % len(ordered)]
    return output


def diagnose(episodes: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    evidence_lines = []
    for episode in episodes:
        for prefix in episode["prefixes"][1:]:
            evidence_lines.extend(
                line
                for line in prefix["features"]["new_evidence_text"].splitlines()
                if line
            )
    entities = Counter(_span(line, "entity ", " attribute ") for line in evidence_lines)
    attributes = Counter(_span(line, " attribute ", " value ") for line in evidence_lines)
    links = Counter(_field(line, "link") for line in evidence_lines)
    conflicts = Counter(_field(line, "conflict") for line in evidence_lines)
    novelty = Counter(_field(line, "novelty") for line in evidence_lines)
    provenance = Counter(_field(line, "provenance") for line in evidence_lines)
    overlaps = [float(_field(line, "goal_overlap")) for line in evidence_lines]
    unlinked_conflicts = sum(
        _span(line, "entity ", " attribute ") == "UNLINKED"
        and _field(line, "conflict") == "conflict"
        for line in evidence_lines
    )

    lookup = {episode["episode_id"]: episode for episode in episodes}
    donors = _donors(episodes)
    similarity_rows = []
    for episode in episodes:
        donor = lookup[donors[episode["episode_id"]]]
        for prefix in episode["prefixes"]:
            donor_index = min(prefix["prefix_index"], len(donor["prefixes"]) - 1)
            own_text = prefix["features"]["evidence_text"]
            donor_text = donor["prefixes"][donor_index]["features"]["evidence_text"]
            own_tokens = _tokens(own_text)
            donor_tokens = _tokens(donor_text)
            union = own_tokens | donor_tokens
            similarity_rows.append(
                {
                    "jaccard": len(own_tokens & donor_tokens) / len(union) if union else 1.0,
                    "exact": own_text == donor_text,
                    "both_nonempty": bool(own_tokens and donor_tokens),
                    "item_count_gap": abs(
                        float(prefix["features"]["evidence_length"]["item_count"])
                        - float(
                            donor["prefixes"][donor_index]["features"]["evidence_length"]["item_count"]
                        )
                    ),
                }
            )
    nonempty = [row for row in similarity_rows if row["both_nonempty"]]
    utility_successes = sum(episode["targets"]["final_utility"] for episode in episodes)
    metrics = summary["seed_averaged_metrics"]
    comparison = summary["comparisons"]["evidence_vs_event_state"]
    return {
        "status": "exploratory_posthoc_does_not_change_frozen_gate",
        "frozen_decision": summary["decision"],
        "counts": {
            "episodes": len(episodes),
            "tasks": len({episode["task_id"] for episode in episodes}),
            "utility_successes": utility_successes,
            "evidence_items": len(evidence_lines),
        },
        "ledger_v1_extraction_audit": {
            "entity_status_counts": dict(entities),
            "link_status_counts": dict(links),
            "conflict_status_counts": dict(conflicts),
            "novelty_counts": dict(novelty),
            "provenance_counts": dict(provenance),
            "unlinked_conflict_items": unlinked_conflicts,
            "conflict_fraction": conflicts["conflict"] / len(evidence_lines),
            "unlinked_conflict_fraction": unlinked_conflicts / len(evidence_lines),
            "mean_goal_overlap": statistics.mean(overlaps),
            "zero_goal_overlap_fraction": sum(value == 0 for value in overlaps) / len(overlaps),
            "top_attributes": attributes.most_common(25),
        },
        "within_task_shuffle_audit": {
            "prefix_pairs": len(similarity_rows),
            "nonempty_prefix_pairs": len(nonempty),
            "mean_token_jaccard_all": statistics.mean(row["jaccard"] for row in similarity_rows),
            "median_token_jaccard_all": statistics.median(row["jaccard"] for row in similarity_rows),
            "mean_token_jaccard_nonempty": statistics.mean(row["jaccard"] for row in nonempty),
            "median_token_jaccard_nonempty": statistics.median(row["jaccard"] for row in nonempty),
            "exact_text_fraction_all": sum(row["exact"] for row in similarity_rows) / len(similarity_rows),
            "exact_text_fraction_nonempty": sum(row["exact"] for row in nonempty) / len(nonempty),
            "mean_absolute_item_count_gap": statistics.mean(row["item_count_gap"] for row in similarity_rows),
        },
        "model_counterevidence": {
            "semantic_markov_progress_mae": metrics["semantic_markov"]["task_macro_progress_mae"],
            "event_state_progress_mae": metrics["semantic_markov_state"]["task_macro_progress_mae"],
            "evidence_progress_mae": metrics["semantic_markov_state_evidence"]["task_macro_progress_mae"],
            "static_utility_brier": metrics["static_length"]["task_macro_utility_brier"],
            "evidence_utility_brier": metrics["semantic_markov_state_evidence"]["task_macro_utility_brier"],
            "evidence_vs_event_state_progress_gain": comparison["task_macro_progress_mae_gain"],
            "evidence_vs_event_state_utility_gain": comparison["task_macro_utility_brier_gain"],
            "mixed_outcome_positive_direction": summary["mixed_outcome_direction"]["positive_direction_count"],
            "mixed_outcome_task_count": summary["mixed_outcome_direction"]["mixed_task_count"],
        },
        "mechanism_flags": {
            "record_boundary_or_entity_resolution_is_unreliable": (
                conflicts["conflict"] / len(evidence_lines) > 0.2
                and unlinked_conflicts > 0
            ),
            "within_task_shuffle_retains_substantial_content": (
                statistics.median(row["jaccard"] for row in nonempty) > 0.5
            ),
            "utility_supervision_is_sparse": utility_successes / len(episodes) < 0.2,
            "ledger_v1_has_preregistered_increment": False,
        },
        "next_protocol_constraints": [
            "Retain the frozen NO-INCREMENT decision; do not tune or rerun on these 90 episodes.",
            "Build ledger v2 from structured runtime outputs while preserving record/entity boundaries.",
            "Mark conflicts only for the same resolved entity and attribute; alternatives are not conflicts.",
            "Use item-level entity links and explicit ambiguous candidate sets rather than one call-level status.",
            "Develop extraction without utility labels, then evaluate once on a separately frozen clean panel.",
            "Do not authorize attack data or Dreamer until the independent durable clean gate passes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = [
        json.loads(line)
        for line in args.episodes.read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = diagnose(episodes, summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
