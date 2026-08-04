"""Build the label-blind panel-v2 manifests, proof registry, and fixed run plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdojo.task_suite.load_suites import get_suite
from wmagentattack import custom_agentdojo_panel_v1 as panel_v1
from wmagentattack import custom_agentdojo_panel_v2 as panel
from wmagentattack.agentdojo_v2 import stable_episode_seed


FRESH_CLEAN_SCOPE = "AgentDojo sandbox only; clean-task solvability screen"
CUSTOM_TASK_MODULE = "wmagentattack.custom_agentdojo_panel_v2"
GREEDY_RUN_SEED = 293
STOCHASTIC_RUN_SEEDS = (307, 311, 313, 317, 331, 337)
STOCHASTIC_TASK_IDS = (
    "clean::banking::user_task_2000",
    "clean::banking::user_task_2001",
    "clean::banking::user_task_2006",
    "clean::banking::user_task_2011",
    "clean::slack::user_task_2100",
    "clean::slack::user_task_2101",
    "clean::slack::user_task_2106",
    "clean::slack::user_task_2111",
    "clean::travel::user_task_2200",
    "clean::travel::user_task_2204",
    "clean::travel::user_task_2206",
    "clean::travel::user_task_2211",
    "clean::workspace::user_task_2300",
    "clean::workspace::user_task_2306",
    "clean::workspace::user_task_2308",
    "clean::workspace::user_task_2311",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _base_manifest(rows: Iterable[dict[str, Any]], *, manifest_id: str) -> dict[str, Any]:
    rows = list(rows)
    split_order = {name: index for index, name in enumerate(panel.SPLITS)}
    suite_order = {name: index for index, name in enumerate(panel.SUITES)}
    difficulty_order = {"L1": 0, "L2": 1, "L3": 2}
    rows.sort(
        key=lambda row: (
            split_order[row["split"]],
            suite_order[row["suite"]],
            difficulty_order[row["task_difficulty"]],
            row["user_task_id"],
        )
    )
    prompt_hashes = {
        spec.row_id: _sha256_text(spec.prompt)
        for spec in panel.TASK_SPECS
        if spec.row_id in {row["row_id"] for row in rows}
    }
    return {
        "scope": FRESH_CLEAN_SCOPE,
        "manifest_id": manifest_id,
        "panel_id": panel.PANEL_ID,
        "custom_task_module": CUSTOM_TASK_MODULE,
        "benchmark_version": panel.BENCHMARK_VERSION,
        "factorized_evaluator_protocol": "configs/0728_factorized_evaluator_v2_protocol.json",
        "proof_contract_registry_id": "0728_custom_clean_panel_v2_contracts_frozen_v1",
        "legacy_scalar_utility_binding": False,
        "safety_contract": {
            "synthetic_environment_only": True,
            "allow_real_network_endpoints": False,
            "external_side_effects": False,
            "clean_tasks_only": True,
            "attack_episodes": 0,
            "model_training_runs": 0,
        },
        "independence_contract": {
            "old_24_tasks_are_evaluator_development_only": True,
            "old_24_tasks_are_barred_from_fresh_confirmation": True,
            "new_panel_victim_outcomes_read_during_task_authoring": False,
            "outcome_labels_read_during_manifest_construction": False,
            "task_ids_disjoint_from_v1": True,
            "prompts_disjoint_from_v1": True,
            "template_families_unique_and_split_disjoint": True,
            "split_assignment_frozen_before_first_victim_outcome": True,
            "stochastic_subset_frozen_before_any_greedy_outcome": True,
            "failed_tasks_may_not_be_removed": True,
        },
        "summary": {
            "tasks": len(rows),
            "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
            "by_suite": dict(sorted(Counter(row["suite"] for row in rows).items())),
            "by_difficulty": dict(
                sorted(Counter(row["task_difficulty"] for row in rows).items())
            ),
            "by_suite_difficulty_split": {
                f"{suite}::{difficulty}::{split}": sum(
                    row["suite"] == suite
                    and row["task_difficulty"] == difficulty
                    and row["split"] == split
                    for row in rows
                )
                for suite in panel.SUITES
                for difficulty in ("L1", "L2", "L3")
                for split in panel.SPLITS
            },
        },
        "prompt_sha256": dict(sorted(prompt_hashes.items())),
        "rows": rows,
    }


def build_greedy_manifest() -> dict[str, Any]:
    manifest = _base_manifest(
        panel.manifest_rows(), manifest_id="0728_custom_clean_panel_v2_greedy_frozen_v1"
    )
    if manifest["summary"]["tasks"] != 48:
        raise ValueError("greedy manifest must contain all 48 tasks")
    if manifest["summary"]["by_split"] != {
        "calibration": 12,
        "confirmation": 12,
        "training": 24,
    }:
        raise ValueError("greedy manifest split balance changed")
    if set(manifest["summary"]["by_suite_difficulty_split"].values()) != {1, 2}:
        raise ValueError("suite-difficulty-split cells are not 2/1/1")
    return manifest


def build_stochastic_manifest() -> dict[str, Any]:
    rows_by_id = {row["row_id"]: row for row in panel.manifest_rows()}
    if len(set(STOCHASTIC_TASK_IDS)) != 16 or not set(STOCHASTIC_TASK_IDS) <= set(rows_by_id):
        raise ValueError("invalid frozen stochastic subset")
    manifest = _base_manifest(
        (rows_by_id[row_id] for row_id in STOCHASTIC_TASK_IDS),
        manifest_id="0728_custom_clean_panel_v2_stochastic_subset_frozen_v1",
    )
    if manifest["summary"]["by_suite"] != {
        "banking": 4,
        "slack": 4,
        "travel": 4,
        "workspace": 4,
    }:
        raise ValueError("stochastic subset must contain four tasks per suite")
    if manifest["summary"]["by_split"] != {
        "calibration": 4,
        "confirmation": 4,
        "training": 8,
    }:
        raise ValueError("stochastic subset must preserve a 2/1/1 split ratio")
    if manifest["summary"]["by_difficulty"] != {"L1": 6, "L2": 5, "L3": 5}:
        raise ValueError("stochastic subset difficulty composition changed")
    return manifest


def build_run_plan() -> dict[str, Any]:
    greedy = build_greedy_manifest()
    sampled = build_stochastic_manifest()
    episodes: list[dict[str, Any]] = []
    for row in greedy["rows"]:
        episodes.append(
            {
                "episode_id": f'greedy::{row["row_id"]}',
                "track": "deterministic_greedy",
                "row_id": row["row_id"],
                "suite": row["suite"],
                "split": row["split"],
                "task_difficulty": row["task_difficulty"],
                "run_seed": GREEDY_RUN_SEED,
                "episode_seed": stable_episode_seed(GREEDY_RUN_SEED, row["row_id"]),
                "do_sample": False,
                "temperature": None,
                "top_p": None,
            }
        )
    for run_seed in STOCHASTIC_RUN_SEEDS:
        for row in sampled["rows"]:
            episodes.append(
                {
                    "episode_id": f'sampled::{row["row_id"]}::seed{run_seed}',
                    "track": "stochastic_policy",
                    "row_id": row["row_id"],
                    "suite": row["suite"],
                    "split": row["split"],
                    "task_difficulty": row["task_difficulty"],
                    "run_seed": run_seed,
                    "episode_seed": stable_episode_seed(run_seed, row["row_id"]),
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.95,
                }
            )
    if len(episodes) != 144 or len({row["episode_id"] for row in episodes}) != 144:
        raise ValueError("fixed run plan must contain 144 unique episodes")
    return {
        "run_plan_id": "0728_custom_clean_panel_v2_fixed_144_v1",
        "frozen_before_any_victim_outcome": True,
        "selection_used_greedy_outcomes": False,
        "budget": {
            "total_clean_episodes": 144,
            "greedy_independent_tasks": 48,
            "stochastic_tasks": 16,
            "stochastic_samples_per_task": 6,
            "stochastic_episodes": 96,
            "attack_episodes": 0,
            "model_training_runs": 0,
        },
        "greedy_run_seed": GREEDY_RUN_SEED,
        "stochastic_run_seeds": list(STOCHASTIC_RUN_SEEDS),
        "stochastic_task_ids": list(STOCHASTIC_TASK_IDS),
        "track_separation": {
            "greedy_label": "One deterministic binary factorized outcome per independent task",
            "stochastic_label": "Six sampled factorized outcomes per preselected task; never pooled with greedy as repeated deterministic seeds",
        },
        "episodes": episodes,
    }


def build_contract_registry() -> dict[str, Any]:
    return panel.build_contract_registry().model_dump(mode="json")


def audit_builds() -> None:
    greedy = build_greedy_manifest()
    sampled = build_stochastic_manifest()
    plan = build_run_plan()
    registry = build_contract_registry()
    old_ids = {spec.spec_id for spec in panel_v1.TASK_SPECS}
    old_prompts = {spec.prompt for spec in panel_v1.TASK_SPECS}
    if old_ids & {spec.spec_id for spec in panel.TASK_SPECS}:
        raise ValueError("panel-v2 task IDs overlap panel v1")
    if old_prompts & {spec.prompt for spec in panel.TASK_SPECS}:
        raise ValueError("panel-v2 prompts overlap panel v1")
    for row in greedy["rows"] + sampled["rows"]:
        task = get_suite(panel.BENCHMARK_VERSION, row["suite"]).get_user_task_by_id(
            row["user_task_id"]
        )
        if getattr(task, "PANEL_SPEC_ID", None) != f'{row["suite"]}::{row["user_task_id"]}':
            raise ValueError(f"custom registration mismatch: {row['row_id']}")
    forbidden = {"utility", "success", "outcome", "prediction"}
    if any(forbidden & set(row) for row in greedy["rows"] + sampled["rows"]):
        raise ValueError("an outcome-like field leaked into a manifest row")
    if any(contract["outcome_labels_present"] for contract in registry["contracts"]):
        raise ValueError("outcome labels leaked into the proof registry")
    if len(plan["episodes"]) != 144:
        raise ValueError("run-plan budget changed")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-output", type=Path, required=True)
    parser.add_argument("--stochastic-output", type=Path, required=True)
    parser.add_argument("--contracts-output", type=Path, required=True)
    parser.add_argument("--run-plan-output", type=Path, required=True)
    args = parser.parse_args()
    audit_builds()
    payloads = {
        args.greedy_output: build_greedy_manifest(),
        args.stochastic_output: build_stochastic_manifest(),
        args.contracts_output: build_contract_registry(),
        args.run_plan_output: build_run_plan(),
    }
    for path, payload in payloads.items():
        _write(path, payload)
    print(
        json.dumps(
            {
                "greedy_tasks": payloads[args.greedy_output]["summary"]["tasks"],
                "stochastic_tasks": payloads[args.stochastic_output]["summary"]["tasks"],
                "episodes": payloads[args.run_plan_output]["budget"]["total_clean_episodes"],
                "contracts": len(payloads[args.contracts_output]["contracts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
