"""Audit whether critical first-step fields survive E5 truncation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from wmagentattack.dreamer_world_model import step_to_dreamer_text
from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import StepRecord


def critical_attack_text(step: StepRecord | dict[str, Any]) -> str:
    value = step.model_dump() if isinstance(step, StepRecord) else step
    target = value.get("target_skill") or "NONE"
    descriptions = value.get("candidate_skill_descriptions", {}) or {}
    return "\n".join(
        [
            f"target: {target}",
            f"target_description: {descriptions.get(target, '')}",
            f"attack: {value.get('attack_action') or 'NONE'}",
            f"domain: {value.get('domain', '')}",
            f"untrusted: {value.get('untrusted_content') or ''}",
            f"observation: {value.get('current_observation', '')}",
            f"goal: {value.get('user_goal', '')}",
            f"trusted: {value.get('trusted_instruction', '')}",
        ]
    )


def _first_attack_steps(data_root: Path) -> list[StepRecord]:
    selected: dict[str, StepRecord] = {}
    for split in ("train", "val", "test"):
        for payload in read_jsonl(data_root / f"{split}_steps.jsonl"):
            step = StepRecord.model_validate(payload)
            group = str(step.multiseed_group_id or "")
            if not group.startswith("attack::"):
                continue
            previous = selected.get(step.trajectory_id)
            if previous is None or step.step_id < previous.step_id:
                selected[step.trajectory_id] = step
    return list(selected.values())


def _marker_position(offsets: list[tuple[int, int]], text: str, marker: str) -> int:
    character = text.find(marker)
    if character < 0:
        raise ValueError(f"Missing marker: {marker}")
    return next(
        (index for index, (_, stop) in enumerate(offsets) if stop > character),
        len(offsets),
    )


def _view_rows(tokenizer, steps: list[StepRecord], view: str) -> list[dict[str, Any]]:
    builder = step_to_dreamer_text if view == "full" else critical_attack_text
    markers = ("untrusted:", "attack:", "target:")
    rows = []
    for step in steps:
        text = builder(step)
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            truncation=False,
        )
        row = {"domain": step.domain, "tokens": len(encoded["input_ids"])}
        for marker in markers:
            row[f"{marker[:-1]}_token"] = _marker_position(
                encoded["offset_mapping"], text, marker
            )
        rows.append(row)
    return rows


def _summarize(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    keys = ("tokens", "untrusted_token", "attack_token", "target_token")
    output: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([int(row[key]) for row in rows])
        if key == "tokens":
            truncated = values > limit
        else:
            truncated = values >= limit - 1
        output[key] = {
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": int(values.max()),
            "truncated_count": int(truncated.sum()),
            "truncated_fraction": float(truncated.mean()),
        }
    by_domain: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["domain"])].append(row)
    for domain, domain_rows in sorted(grouped.items()):
        by_domain[domain] = {}
        for key in keys:
            values = np.asarray([int(row[key]) for row in domain_rows])
            truncated = values > limit if key == "tokens" else values >= limit - 1
            by_domain[domain][key] = float(truncated.mean())
    output["by_domain_truncated_fraction"] = by_domain
    return output


def audit(data_root: Path, *, model_name: str, cache_dir: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    steps = _first_attack_steps(data_root)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, use_fast=True
    )
    limit = int(tokenizer.model_max_length)
    return {
        "scope": "label-blind E5 first-step truncation audit",
        "outcome_labels_read": False,
        "attack_first_step_trajectories": len(steps),
        "tokenizer": model_name,
        "token_limit": limit,
        "full_view": _summarize(_view_rows(tokenizer, steps, "full"), limit),
        "critical_view": _summarize(
            _view_rows(tokenizer, steps, "critical"), limit
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-name", default="intfloat/e5-base-v2")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.data_root, model_name=args.model_name, cache_dir=args.cache_dir
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
