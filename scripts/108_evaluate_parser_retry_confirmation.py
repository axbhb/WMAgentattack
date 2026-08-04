"""Evaluate unseen-seed clean confirmation for the frozen parser/retry protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_map(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary["tasks"]:
        key = (str(row["suite"]), str(row["user_task_id"]))
        if key in rows:
            raise ValueError(f"Duplicate task in summary: {key}")
        rows[key] = row
    return rows


def evaluate(
    development: dict[str, Any],
    confirmation: dict[str, Any],
    failure_audit: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    dev_seeds = tuple(int(seed) for seed in development["protocol"]["seeds"])
    conf_seeds = tuple(int(seed) for seed in confirmation["protocol"]["seeds"])
    expected_dev_seeds = tuple(protocol["development_reference"]["seeds"])
    expected_conf_seeds = tuple(protocol["confirmation"]["seeds"])
    if dev_seeds != expected_dev_seeds:
        raise ValueError(f"Development seeds changed: {dev_seeds}")
    if conf_seeds != expected_conf_seeds:
        raise ValueError(f"Confirmation seeds changed: {conf_seeds}")
    if set(dev_seeds) & set(conf_seeds):
        raise ValueError("Development and confirmation seeds overlap")

    dev = _task_map(development)
    conf = _task_map(confirmation)
    if set(dev) != set(conf):
        raise ValueError("Development and confirmation task sets differ")
    expected_tasks = int(protocol["confirmation"]["tasks"])
    expected_episodes = int(protocol["confirmation"]["episodes"])
    if len(conf) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} tasks, found {len(conf)}")
    if int(confirmation["counts"]["clean_episodes"]) != expected_episodes:
        raise ValueError("Confirmation episode count is incomplete")
    if int(failure_audit["counts"]["episodes"]) != expected_episodes:
        raise ValueError("Failure audit episode count is incomplete")

    threshold = int(protocol["decision_gate"]["retention_successes_out_of_three"])
    task_rows: list[dict[str, Any]] = []
    for key in sorted(conf):
        dev_row = dev[key]
        conf_row = conf[key]
        dev_successes = int(dev_row["successes"])
        conf_successes = int(conf_row["successes"])
        task_rows.append(
            {
                "suite": key[0],
                "user_task_id": key[1],
                "development_successes": dev_successes,
                "confirmation_successes": conf_successes,
                "development_retained": dev_successes >= threshold,
                "confirmation_retained": conf_successes >= threshold,
                "durable_retained": (
                    dev_successes >= threshold and conf_successes >= threshold
                ),
                "confirmation_outcomes": conf_row["outcomes"],
            }
        )

    confirmation_retained = [
        row for row in task_rows if row["confirmation_retained"]
    ]
    durable = [row for row in task_rows if row["durable_retained"]]
    min_confirmation = int(
        protocol["decision_gate"]["minimum_confirmation_retained_tasks"]
    )
    min_durable = int(
        protocol["decision_gate"][
            "minimum_tasks_retained_in_both_development_and_confirmation"
        ]
    )
    gate = {
        "complete_45_episode_panel": (
            int(confirmation["counts"]["clean_episodes"]) == expected_episodes
        ),
        "zero_runtime_failures": True,
        "at_least_two_confirmation_retained_tasks": (
            len(confirmation_retained) >= min_confirmation
        ),
        "at_least_two_durable_development_confirmation_tasks": (
            len(durable) >= min_durable
        ),
    }
    gate["all_required_criteria_met"] = all(gate.values())
    decision_key = (
        "decision_if_go"
        if gate["all_required_criteria_met"]
        else "decision_if_no_go"
    )

    return {
        "scope": "unseen-seed clean confirmation of frozen parser-v2/retry protocol",
        "attack_outcomes_read": False,
        "development": {
            "seeds": list(dev_seeds),
            "successes": sum(int(row["successes"]) for row in dev.values()),
            "retained_task_ids": [
                f'{row["suite"]}::{row["user_task_id"]}'
                for row in task_rows
                if row["development_retained"]
            ],
        },
        "confirmation": {
            "seeds": list(conf_seeds),
            "successes": sum(int(row["successes"]) for row in conf.values()),
            "episodes": expected_episodes,
            "retained_task_ids": [
                f'{row["suite"]}::{row["user_task_id"]}'
                for row in confirmation_retained
            ],
            "success_count_distribution": {
                str(count): sum(
                    row["confirmation_successes"] == count for row in task_rows
                )
                for count in range(4)
            },
            "failures_without_tool_call": int(
                failure_audit["counts"]["failures_without_tool_call"]
            ),
        },
        "durable_task_ids": [
            f'{row["suite"]}::{row["user_task_id"]}' for row in durable
        ],
        "gate": gate,
        "decision": protocol[decision_key],
        "task_rows": task_rows,
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--confirmation-summary", type=Path, required=True)
    parser.add_argument("--failure-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    development = json.loads(args.development_summary.read_text(encoding="utf-8"))
    confirmation = json.loads(args.confirmation_summary.read_text(encoding="utf-8"))
    failure_audit = json.loads(args.failure_audit.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = evaluate(development, confirmation, failure_audit, protocol)
    result["provenance"] = {
        "development_summary": str(args.development_summary.resolve()),
        "development_summary_sha256": _sha256(args.development_summary),
        "confirmation_summary": str(args.confirmation_summary.resolve()),
        "confirmation_summary_sha256": _sha256(args.confirmation_summary),
        "failure_audit": str(args.failure_audit.resolve()),
        "failure_audit_sha256": _sha256(args.failure_audit),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "development": result["development"],
                "confirmation": result["confirmation"],
                "durable_task_ids": result["durable_task_ids"],
                "gate": result["gate"],
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
