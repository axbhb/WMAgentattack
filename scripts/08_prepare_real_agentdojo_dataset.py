"""Merge clean and attacked AgentDojo traces and produce a label audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.io_utils import write_jsonl
from wmagentattack.normalize_agentdojo import normalize_directory


def _infer_protocol(victim_model: str) -> str:
    return "native" if "-native-" in victim_model else "function_tags"


def _infer_prompt_profile(victim_model: str) -> str:
    parts = victim_model.split("-")
    if "robust" in parts:
        return "robust"
    if "base" in parts:
        return "base"
    return "unknown"


def _is_benchmark_user_task(task_id: str) -> bool:
    return task_id.startswith("user_task_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        help=(
            "AgentDojo run root containing both clean traces "
            "(`<suite>/<user_task>/none/none.json`) and attacked traces. "
            "When set, --clean-root and --attack-root are ignored."
        ),
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=ROOT
        / "runs"
        / "qwen2.5-7b-instruct-transformers-4bit-compact12000-robust",
    )
    parser.add_argument(
        "--attack-root",
        type=Path,
        default=ROOT / "runs" / "agentdojo_attacks",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data" / "real_agentdojo"
    )
    args = parser.parse_args()

    if args.run_root:
        all_trajectories = normalize_directory(args.run_root)
        standalone_injection_task_trajectories_excluded = sum(
            not _is_benchmark_user_task(trajectory.task_id)
            for trajectory in all_trajectories
        )
        clean = [
            trajectory
            for trajectory in all_trajectories
            if trajectory.steps
            and _is_benchmark_user_task(trajectory.task_id)
            and all(step.attack_action is None for step in trajectory.steps)
        ]
        attacked_all = [
            trajectory
            for trajectory in all_trajectories
            if trajectory.steps
            and _is_benchmark_user_task(trajectory.task_id)
            and any(step.attack_action is not None for step in trajectory.steps)
        ]
        raw_roots = [args.run_root]
    else:
        clean = normalize_directory(args.clean_root)
        attacked_all = normalize_directory(args.attack_root)
        raw_roots = [args.attack_root]
        standalone_injection_task_trajectories_excluded = 0

    attacked = [
        trajectory
        for trajectory in attacked_all
        if any(step.attack_action is not None for step in trajectory.steps)
        and trajectory.steps
    ]
    trajectories = clean + attacked
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    incomplete_raw_traces = 0
    for raw_root in raw_roots:
        for path in raw_root.rglob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if {"suite_name", "pipeline_name", "messages"}.issubset(raw) and (
                "utility" not in raw or "security" not in raw
            ):
                incomplete_raw_traces += 1

    write_jsonl(args.out_dir / "trajectories.jsonl", trajectories)
    write_jsonl(args.out_dir / "steps.jsonl", steps)

    by_attack = defaultdict(lambda: {"count": 0, "success": 0, "utility": 0})
    for trajectory in attacked:
        attack = trajectory.steps[0].attack_action or "NONE"
        by_attack[attack]["count"] += 1
        by_attack[attack]["success"] += int(
            trajectory.final_attack_success
        )
        by_attack[attack]["utility"] += int(trajectory.final_task_success)

    positive = sum(
        trajectory.final_attack_success for trajectory in attacked
    )
    audit = {
        "scope": "AgentDojo sandbox only",
        "clean_trajectories": len(clean),
        "attack_trajectories": len(attacked),
        "total_steps": len(steps),
        "standalone_injection_task_trajectories_excluded": (
            standalone_injection_task_trajectories_excluded
        ),
        "incomplete_raw_traces_excluded": incomplete_raw_traces,
        "steps_with_untrusted_content": sum(
            step.untrusted_content is not None for step in steps
        ),
        "finish_after_untrusted_content": sum(
            step.selected_skill == "finish"
            and step.untrusted_content is not None
            for step in steps
        ),
        "attack_positive_labels": positive,
        "attack_negative_labels": len(attacked) - positive,
        "real_risk_head_trainable": 0 < positive < len(attacked),
        "attack_success_rate": positive / len(attacked) if attacked else 0.0,
        "utility_preservation_rate_under_attack": (
            sum(t.final_task_success for t in attacked) / len(attacked)
            if attacked
            else 0.0
        ),
        "attack_breakdown": {
            attack: {
                **stats,
                "ASR": stats["success"] / stats["count"],
                "BUP": stats["utility"] / stats["count"],
            }
            for attack, stats in sorted(by_attack.items())
        },
        "selected_skill_counts": dict(
            Counter(step.selected_skill for step in steps)
        ),
        "domain_counts": dict(Counter(t.domain for t in trajectories)),
        "attack_domain_counts": dict(Counter(t.domain for t in attacked)),
        "victim_model_counts": dict(
            Counter(t.victim_model for t in trajectories)
        ),
        "protocol_counts": dict(
            Counter(_infer_protocol(t.victim_model) for t in trajectories)
        ),
        "prompt_profile_counts": dict(
            Counter(_infer_prompt_profile(t.victim_model) for t in trajectories)
        ),
        "notes": [
            "AgentDojo security=True on attacked traces means the attacker goal was achieved.",
            "Standalone injection-task utility checks are excluded from --run-root benchmark datasets.",
            "Do not train a real-data risk head unless both positive and negative attack labels exist.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
