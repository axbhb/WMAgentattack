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


def _score_step(model: SklearnWorldModel, step: StepRecord) -> dict:
    predictions = model.predict([step])
    classes = list(predictions["skill_classes"])
    target_index = (
        classes.index(step.target_skill)
        if step.target_skill is not None and step.target_skill in classes
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


def _dedupe_by_pair(rows: list[dict]) -> list[dict]:
    seen = set()
    output = []
    for row in rows:
        key = (row["suite"], row["user_task_id"], row["injection_task_id"])
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attack", default="important_instructions_no_model_name")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    model = SklearnWorldModel.load(args.model)
    candidates: list[dict] = []
    for path in sorted(args.run_root.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if raw.get("attack_type") != args.attack or not _is_benchmark_attack_trace(raw):
            continue
        trajectory = normalize_trace(path)
        step = _representative_step(trajectory.steps)
        if step is None:
            continue
        scores = _score_step(model, step)
        candidates.append(
            {
                "suite": raw["suite_name"],
                "user_task_id": raw["user_task_id"],
                "injection_task_id": raw["injection_task_id"],
                "attack": raw["attack_type"],
                "trajectory_id": trajectory.trajectory_id,
                "target_skill": step.target_skill,
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
    sorted_low = sorted(candidates, key=lambda row: row["selection_score"])
    top_k = min(args.top_k, len(candidates))
    rng = random.Random(args.seed)
    random_rows = candidates[:]
    rng.shuffle(random_rows)

    selections = {
        "world_model_top": _dedupe_by_pair(sorted_high)[:top_k],
        "low_score": _dedupe_by_pair(sorted_low)[:top_k],
        "random": _dedupe_by_pair(random_rows)[:top_k],
    }
    summary = {}
    for name, rows in selections.items():
        observed_security = np.array([row["observed_security"] for row in rows])
        observed_utility = np.array([row["observed_utility"] for row in rows])
        summary[name] = {
            "count": len(rows),
            "observed_asr": float(observed_security.mean()) if len(rows) else 0.0,
            "observed_bup": float(observed_utility.mean()) if len(rows) else 0.0,
            "mean_selection_score": float(
                np.mean([row["selection_score"] for row in rows])
            )
            if rows
            else 0.0,
        }

    payload = {
        "scope": "world_model_selected_real_agentdojo_pairs",
        "run_root": str(args.run_root.resolve()),
        "model": str(args.model.resolve()),
        "attack": args.attack,
        "top_k": top_k,
        "seed": args.seed,
        "candidate_count": len(candidates),
        "summary": summary,
        "selections": selections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
