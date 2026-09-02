"""Aggregate Qwen AgentDojo v1 results in the paper's metric vocabulary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITES = ("workspace", "travel", "banking", "slack")


def load_suite(root: Path, suite: str, model_label: str) -> dict:
    path = root / suite / f"summary_{model_label}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(success: int, count: int) -> float:
    return success / count if count else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--attack", default="important_instructions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clean_success = clean_count = 0
    attack_utility_success = attack_count = 0
    attack_success = joint_success = 0
    per_suite: dict[str, dict] = {}
    complete = True
    for suite in SUITES:
        summary = load_suite(args.run_root, suite, args.model_label)
        clean = summary["clean"][suite]
        attack = summary.get("attack", {}).get(args.attack, {}).get(suite)
        if attack is None:
            complete = False
            attack = {
                "utility_success": 0,
                "utility_count": 0,
                "security_success": 0,
                "security_count": 0,
                "joint_task_and_attack_success": 0,
            }
        clean_success += int(clean["utility_success"])
        clean_count += int(clean["utility_count"])
        attack_utility_success += int(attack["utility_success"])
        attack_count += int(attack["utility_count"])
        attack_success += int(attack["security_success"])
        joint_success += int(attack["joint_task_and_attack_success"])
        per_suite[suite] = {
            "benign_utility": ratio(int(clean["utility_success"]), int(clean["utility_count"])),
            "utility_under_attack": ratio(int(attack["utility_success"]), int(attack["utility_count"])),
            "targeted_asr": ratio(int(attack["security_success"]), int(attack["security_count"])),
            "joint_rate": ratio(int(attack["joint_task_and_attack_success"]), int(attack["security_count"])),
        }

    aggregate = {
        "schema_version": "wmagentattack.agentdojo_qwen_paper_parity.v1",
        "complete": complete and clean_count == 97 and attack_count == 629,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "benchmark_version": "v1",
        "attack": args.attack,
        "counts": {
            "clean_success": clean_success,
            "clean_total": clean_count,
            "attack_utility_success": attack_utility_success,
            "attack_total": attack_count,
            "targeted_attack_success": attack_success,
            "joint_success": joint_success,
        },
        "metrics": {
            "benign_utility": ratio(clean_success, clean_count),
            "utility_under_attack": ratio(attack_utility_success, attack_count),
            "targeted_asr": ratio(attack_success, attack_count),
            "joint_task_and_attack_rate": ratio(joint_success, attack_count),
            "conditional_targeted_asr_given_utility": ratio(joint_success, attack_utility_success),
        },
        "paper_reference_llama3_70b": {
            "benign_utility": 0.3450,
            "utility_under_attack": 0.1828,
            "targeted_asr": 0.2003,
        },
        "difference_qwen_minus_paper_llama3_70b": {},
        "per_suite": per_suite,
    }
    aggregate["difference_qwen_minus_paper_llama3_70b"] = {
        key: aggregate["metrics"][key] - value
        for key, value in aggregate["paper_reference_llama3_70b"].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
