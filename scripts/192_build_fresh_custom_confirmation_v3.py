"""Build the sealed twelve-task confirmation-v3 manifest and run plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdojo.task_suite.load_suites import get_suite

from wmagentattack import custom_agentdojo_confirmation_v3 as panel
from wmagentattack.agentdojo_v2 import stable_episode_seed


FRESH_CLEAN_SCOPE = "AgentDojo sandbox only; clean-task solvability screen"
CUSTOM_TASK_MODULE = "wmagentattack.custom_agentdojo_confirmation_v3"
RUN_SEEDS = (401, 409, 419)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def build_manifest() -> dict[str, Any]:
    rows = panel.manifest_rows()
    return {
        "scope": FRESH_CLEAN_SCOPE,
        "manifest_id": "0813_fresh_custom_confirmation_v3_frozen_v1",
        "panel_id": panel.PANEL_ID,
        "custom_task_module": CUSTOM_TASK_MODULE,
        "benchmark_version": panel.BENCHMARK_VERSION,
        "legacy_scalar_utility_binding": False,
        "safety_contract": {
            "synthetic_environment_only": True,
            "allow_real_network_endpoints": False,
            "external_side_effects": False,
            "clean_tasks_only": True,
            "attack_episodes": 0,
        },
        "independence_contract": {
            "all_stock_agentdojo_tasks_historically_screened": True,
            "panel_v1_and_v2_historically_used": True,
            "new_task_ids_prompts_and_template_families_disjoint": True,
            "victim_outcomes_read_during_authoring": False,
            "outcome_labels_read_during_manifest_construction": False,
            "all_twelve_tasks_retained_regardless_of_outcome": True,
            "run_seeds_frozen_before_first_outcome": True,
        },
        "summary": {
            "tasks": len(rows),
            "by_suite": dict(sorted(Counter(row["suite"] for row in rows).items())),
            "by_difficulty": dict(
                sorted(Counter(row["task_difficulty"] for row in rows).items())
            ),
        },
        "prompt_sha256": {
            spec.row_id: _sha256_text(spec.prompt) for spec in panel.TASK_SPECS
        },
        "rows": rows,
    }


def build_run_plan() -> dict[str, Any]:
    episodes = []
    for seed in RUN_SEEDS:
        for row in panel.manifest_rows():
            episodes.append(
                {
                    "episode_id": f'{row["row_id"]}::seed{seed}',
                    "row_id": row["row_id"],
                    "suite": row["suite"],
                    "task_difficulty": row["task_difficulty"],
                    "run_seed": seed,
                    "episode_seed": stable_episode_seed(seed, row["row_id"]),
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.95,
                }
            )
    return {
        "run_plan_id": "0813_fresh_custom_confirmation_v3_fixed_36_v1",
        "frozen_before_first_victim_outcome": True,
        "selection_used_historical_or_current_outcomes": False,
        "budget": {
            "tasks": 12,
            "run_seeds": 3,
            "total_clean_episodes": 36,
            "attack_episodes": 0,
            "real_external_endpoint_calls": 0,
        },
        "run_seeds": list(RUN_SEEDS),
        "episodes": episodes,
    }


def audit_builds() -> None:
    manifest = build_manifest()
    plan = build_run_plan()
    if manifest["summary"] != {
        "tasks": 12,
        "by_suite": {"banking": 3, "slack": 3, "travel": 3, "workspace": 3},
        "by_difficulty": {"L1": 4, "L2": 4, "L3": 4},
    }:
        raise ValueError("fresh confirmation balance drift")
    if len(plan["episodes"]) != 36 or len({row["episode_id"] for row in plan["episodes"]}) != 36:
        raise ValueError("fixed 36-episode run plan drift")
    forbidden = {"utility", "success", "outcome", "prediction"}
    if any(forbidden & set(row) for row in manifest["rows"]):
        raise ValueError("outcome-like field leaked into manifest")
    for row in manifest["rows"]:
        task = get_suite(panel.BENCHMARK_VERSION, row["suite"]).get_user_task_by_id(
            row["user_task_id"]
        )
        if getattr(task, "PANEL_SPEC_ID", None) != f'{row["suite"]}::{row["user_task_id"]}':
            raise ValueError(f"custom registration mismatch: {row['row_id']}")


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    args = parser.parse_args()
    audit_builds()
    _write(args.manifest, build_manifest())
    _write(args.contracts, panel.build_contract_registry())
    _write(args.run_plan, build_run_plan())
    print(json.dumps({"tasks": 12, "episodes": 36, "outcome_labels": 0}))


if __name__ == "__main__":
    main()
