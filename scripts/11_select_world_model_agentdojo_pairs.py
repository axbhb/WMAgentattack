"""Select real AgentDojo task/injection pairs for targeted replay.

The selector scores already-materialized AgentDojo traces with the learned
world model and writes three subsets:

- ``world_model_top``: high predicted risk while preserving utility
- ``low_score``: low predicted risk/utility control
- ``random``: deterministic random baseline

It does not create new attack text. It only selects existing
``user_task_*`` x ``injection_task_*`` combinations for replay.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.normalize_agentdojo import normalize_trace
from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import StepRecord
from wmagentattack.world_model import SklearnWorldModel


def _is_benchmark_attack_trace(raw: dict) -> bool:
    return (
        str(raw.get("user_task_id", "")).startswith("user_task_")
        and str(raw.get("injection_task_id", "")).startswith("injection_task_")
        and raw.get("attack_type") not in (None, "none")
    )


def _representative_step(steps: list[StepRecord]) -> StepRecord | None:
    with_untrusted = [step for step in steps if step.untrusted_content is not None]
    if with_untrusted:
        return with_untrusted[0]
    return steps[0] if steps else None


def _score_step(model: SklearnWorldModel, step: StepRecord | dict) -> dict:
    predictions = model.predict([step])
    classes = list(predictions["skill_classes"])
    target_skill = (
        step.target_skill if isinstance(step, StepRecord) else step.get("target_skill")
    )
    target_index = (
        classes.index(target_skill)
        if target_skill is not None and target_skill in classes
        else None
    )
    target_probability = (
        float(predictions["next_skill_proba"][0, target_index])
        if target_index is not None
        else 0.0
    )
    risk_score = float(predictions["risk_score"][0])
    utility_score = float(predictions["utility_score"][0])
    # Prioritize likely attack success, but avoid selecting only traces where
    # the task is predicted to collapse.
    selection_score = risk_score + 0.5 * utility_score + 0.3 * target_probability
    return {
        "risk_score": risk_score,
        "utility_score": utility_score,
        "target_skill_probability": target_probability,
        "selection_score": selection_score,
    }


def _dedupe_by_pair(rows: list[dict], max_per_user_task: int = 0) -> list[dict]:
    seen = set()
    per_user_task: dict[tuple[str, str], int] = {}
    output = []
    for row in rows:
        key = (row["suite"], row["user_task_id"], row["injection_task_id"])
        if key in seen:
            continue
        user_key = (row["suite"], row["user_task_id"])
        if max_per_user_task > 0 and per_user_task.get(user_key, 0) >= max_per_user_task:
            continue
        seen.add(key)
        per_user_task[user_key] = per_user_task.get(user_key, 0) + 1
        output.append(row)
    return output


def _pair_key(row: dict) -> tuple[str, str, str]:
    return (row["suite"], row["user_task_id"], row["injection_task_id"])


def _without_pairs(rows: list[dict], excluded: set[tuple[str, str, str]]) -> list[dict]:
    return [row for row in rows if _pair_key(row) not in excluded]


def _summarize_rows(rows: list[dict]) -> dict:
    observed_security = np.array([row["observed_security"] for row in rows])
    observed_utility = np.array([row["observed_utility"] for row in rows])
    return {
        "count": len(rows),
        "observed_asr": float(observed_security.mean()) if len(rows) else 0.0,
        "observed_bup": float(observed_utility.mean()) if len(rows) else 0.0,
        "mean_selection_score": float(
            np.mean([row["selection_score"] for row in rows])
        )
        if rows
        else 0.0,
        "mean_risk_score": float(np.mean([row["risk_score"] for row in rows]))
        if rows
        else 0.0,
        "mean_utility_score": float(np.mean([row["utility_score"] for row in rows]))
        if rows
        else 0.0,
        "mean_target_skill_probability": float(
            np.mean([row["target_skill_probability"] for row in rows])
        )
        if rows
        else 0.0,
    }


def _clean_prefix_states(run_root: Path) -> dict[tuple[str, str], dict]:
    states = {}
    for path in sorted(run_root.rglob("none/none.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not str(raw.get("user_task_id", "")).startswith("user_task_"):
            continue
        if raw.get("attack_type") not in (None, "none"):
            continue
        trajectory = normalize_trace(path)
        step = _representative_step(trajectory.steps)
        if step is None:
            continue
        states[(raw["suite_name"], raw["user_task_id"])] = step.model_dump(
            mode="json"
        )
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attack", default="important_instructions_no_model_name")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--scoring-mode",
        choices=["attack_trace_state", "clean_prefix"],
        default="attack_trace_state",
        help=(
            "attack_trace_state scores a state from the completed attacked trace; "
            "clean_prefix scores the matching clean user-task state with only "
            "hypothetical attack/target metadata."
        ),
    )
    parser.add_argument(
        "--allowed-trajectories",
        type=Path,
        help=(
            "Optional JSONL of TrajectoryRecord rows. When set, only raw traces "
            "whose normalized trajectory_id is present in this file are scored. "
            "Use the held-out split here to avoid train/test leakage."
        ),
    )
    parser.add_argument(
        "--max-per-user-task",
        type=int,
        default=2,
        help="Maximum selected pairs per (suite, user_task_id); <=0 disables.",
    )
    parser.add_argument(
        "--exclude-world-model-from-baselines",
        action="store_true",
        help=(
            "Build baseline/ablation selections after removing pairs selected "
            "by world_model_top. This avoids overlap-contaminated random controls."
        ),
    )
    args = parser.parse_args()

    model = SklearnWorldModel.load(args.model)
    clean_states = (
        _clean_prefix_states(args.run_root)
        if args.scoring_mode == "clean_prefix"
        else {}
    )
    allowed_ids = None
    if args.allowed_trajectories:
        allowed_ids = {
            row["trajectory_id"] for row in read_jsonl(args.allowed_trajectories)
        }

    candidates: list[dict] = []
    for path in sorted(args.run_root.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if raw.get("attack_type") != args.attack or not _is_benchmark_attack_trace(raw):
            continue
        trajectory = normalize_trace(path)
        if allowed_ids is not None and trajectory.trajectory_id not in allowed_ids:
            continue
        attack_step = _representative_step(trajectory.steps)
        if attack_step is None:
            continue
        if args.scoring_mode == "clean_prefix":
            step = clean_states.get((raw["suite_name"], raw["user_task_id"]))
            if step is None:
                continue
            step = {
                **step,
                "attack_action": f"AGENTDOJO_ATTACK_{raw['attack_type']}",
                "attack_location": None,
                "target_skill": attack_step.target_skill,
                "untrusted_content": None,
            }
        else:
            step = attack_step
        scores = _score_step(model, step)
        candidates.append(
            {
                "suite": raw["suite_name"],
                "user_task_id": raw["user_task_id"],
                "injection_task_id": raw["injection_task_id"],
                "attack": raw["attack_type"],
                "trajectory_id": trajectory.trajectory_id,
                "target_skill": attack_step.target_skill,
                "observed_utility": bool(raw.get("utility")),
                "observed_security": bool(raw.get("security")),
                "source_trace": str(path),
                **scores,
            }
        )

    if not candidates:
        raise RuntimeError(f"No attack candidates found under {args.run_root}")

    sorted_high = sorted(
        candidates, key=lambda row: row["selection_score"], reverse=True
    )
    sorted_risk = sorted(candidates, key=lambda row: row["risk_score"], reverse=True)
    sorted_utility = sorted(
        candidates, key=lambda row: row["utility_score"], reverse=True
    )
    sorted_target = sorted(
        candidates, key=lambda row: row["target_skill_probability"], reverse=True
    )
    sorted_low = sorted(candidates, key=lambda row: row["selection_score"])
    top_k = min(args.top_k, len(candidates))
    rng = random.Random(args.seed)
    random_rows = candidates[:]
    rng.shuffle(random_rows)

    world_model_top = _dedupe_by_pair(
        sorted_high, max_per_user_task=args.max_per_user_task
    )[:top_k]
    baseline_excluded = (
        {_pair_key(row) for row in world_model_top}
        if args.exclude_world_model_from_baselines
        else set()
    )

    selections = {
        "world_model_top": world_model_top,
        "risk_only_top": _dedupe_by_pair(
            _without_pairs(sorted_risk, baseline_excluded),
            max_per_user_task=args.max_per_user_task,
        )[:top_k],
        "utility_only_top": _dedupe_by_pair(
            _without_pairs(sorted_utility, baseline_excluded),
            max_per_user_task=args.max_per_user_task,
        )[:top_k],
        "target_skill_top": _dedupe_by_pair(
            _without_pairs(sorted_target, baseline_excluded),
            max_per_user_task=args.max_per_user_task,
        )[:top_k],
        "low_score": _dedupe_by_pair(
            _without_pairs(sorted_low, baseline_excluded),
            max_per_user_task=args.max_per_user_task,
        )[:top_k],
        "random": _dedupe_by_pair(
            _without_pairs(random_rows, baseline_excluded),
            max_per_user_task=args.max_per_user_task,
        )[:top_k],
    }
    summary = {name: _summarize_rows(rows) for name, rows in selections.items()}
    overlap = {}
    selection_sets = {
        name: {_pair_key(row) for row in rows} for name, rows in selections.items()
    }
    for left_name, left_pairs in selection_sets.items():
        for right_name, right_pairs in selection_sets.items():
            if left_name >= right_name:
                continue
            overlap[f"{left_name}__{right_name}"] = len(left_pairs & right_pairs)

    payload = {
        "scope": "world_model_selected_real_agentdojo_pairs",
        "run_root": str(args.run_root.resolve()),
        "model": str(args.model.resolve()),
        "attack": args.attack,
        "top_k": top_k,
        "seed": args.seed,
        "candidate_count": len(candidates),
        "scoring_mode": args.scoring_mode,
        "allowed_trajectories": (
            str(args.allowed_trajectories.resolve())
            if args.allowed_trajectories
            else None
        ),
        "allowed_trajectory_count": len(allowed_ids) if allowed_ids is not None else None,
        "max_per_user_task": args.max_per_user_task,
        "exclude_world_model_from_baselines": args.exclude_world_model_from_baselines,
        "summary": summary,
        "overlap": overlap,
        "selections": selections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
