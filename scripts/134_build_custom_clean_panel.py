"""Build the frozen, label-blind manifest for custom AgentDojo clean tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdojo.task_suite.load_suites import get_suite
from wmagentattack import custom_agentdojo_panel_v1 as panel


FRESH_CLEAN_SCOPE = "AgentDojo sandbox only; clean-task solvability screen"
CUSTOM_TASK_MODULE = "wmagentattack.custom_agentdojo_panel_v1"


def build_manifest() -> dict[str, Any]:
    rows = panel.manifest_rows()
    if len(rows) != 24 or len({row["row_id"] for row in rows}) != 24:
        raise ValueError("The frozen custom panel must contain 24 unique tasks")
    split_order = {name: index for index, name in enumerate(panel.SPLITS)}
    suite_order = {name: index for index, name in enumerate(panel.SUITES)}
    rows.sort(
        key=lambda row: (
            split_order[row["split"]],
            suite_order[row["suite"]],
            row["user_task_id"],
        )
    )
    split_families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_families[row["split"]].add(row["template_family"])
        task = get_suite(panel.BENCHMARK_VERSION, row["suite"]).get_user_task_by_id(
            row["user_task_id"]
        )
        if getattr(task, "PANEL_SPEC_ID", None) != (
            f'{row["suite"]}::{row["user_task_id"]}'
        ):
            raise ValueError(f"Custom task registration mismatch: {row['row_id']}")
    for left_index, left in enumerate(panel.SPLITS):
        for right in panel.SPLITS[left_index + 1 :]:
            overlap = split_families[left] & split_families[right]
            if overlap:
                raise ValueError(f"Template leakage between {left} and {right}: {overlap}")
    prompt_hashes = {
        spec.spec_id: hashlib.sha256(spec.prompt.encode()).hexdigest()
        for spec in panel.TASK_SPECS
    }
    if len(set(prompt_hashes.values())) != len(prompt_hashes):
        raise ValueError("Every custom task prompt must be unique")
    counts_by_split = Counter(row["split"] for row in rows)
    counts_by_suite = Counter(row["suite"] for row in rows)
    counts_by_split_suite = Counter((row["split"], row["suite"]) for row in rows)
    if set(counts_by_split.values()) != {8} or set(counts_by_suite.values()) != {6}:
        raise ValueError("Custom panel is not split- and suite-balanced")
    if set(counts_by_split_suite.values()) != {2}:
        raise ValueError("Each split must have exactly two tasks from each suite")
    return {
        "scope": FRESH_CLEAN_SCOPE,
        "manifest_id": "0727_custom_clean_panel_v1_fixed",
        "custom_task_module": CUSTOM_TASK_MODULE,
        "benchmark_version": panel.BENCHMARK_VERSION,
        "safety_contract": {
            "synthetic_environment_only": True,
            "allow_real_network_endpoints": False,
            "external_side_effects": False,
            "clean_tasks_only": True,
            "attack_episodes": 0,
        },
        "independence_contract": {
            "prior_stock_task_development_exposure_disclosed": True,
            "custom_panel_victim_outcomes_read_during_task_authoring": False,
            "upstream_scaffold_screen_used_only_to_freeze_execution_scaffold": True,
            "outcome_labels_read_during_manifest_construction": False,
            "template_families_disjoint_across_splits": True,
            "task_ids_disjoint_across_splits": True,
            "confirmation_tasks_may_not_tune_scaffold_or_architecture": True,
            "all_tasks_are_frozen_before_first_custom_panel_victim_run": True,
        },
        "split_semantics": {
            "training": "direct lookup projections and explicit state mutations",
            "calibration": "pairwise comparisons and observed-condition actions",
            "confirmation": "cross-source joins and multi-constraint entity selection",
        },
        "summary": {
            "tasks": len(rows),
            "by_split": dict(sorted(counts_by_split.items())),
            "by_suite": dict(sorted(counts_by_suite.items())),
            "by_split_suite": {
                f"{split}::{suite}": counts_by_split_suite[(split, suite)]
                for split in panel.SPLITS
                for suite in panel.SUITES
            },
            "template_families_by_split": {
                split: sorted(split_families[split]) for split in panel.SPLITS
            },
        },
        "prompt_sha256": prompt_hashes,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
