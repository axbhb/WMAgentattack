"""Summarize multi-seed real AgentDojo selected-pair replays."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    security = np.asarray([bool(row["security"]) for row in rows], dtype=float)
    utility = np.asarray([bool(row["utility"]) for row in rows], dtype=float)
    asr = float(security.mean()) if count else 0.0
    bup = float(utility.mean()) if count else 0.0
    return {
        "attempt_count": count,
        "observed_asr": asr,
        "observed_bup": bup,
        "asr_plus_bup": asr + bup,
    }


def _cluster_bootstrap(
    pair_rows: dict[tuple[str, str, str], list[dict[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    keys = list(pair_rows)
    rng = np.random.default_rng(seed)
    values = np.empty((samples, 3), dtype=float)
    for index in range(samples):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        rows = [row for item in sampled for row in pair_rows[keys[item]]]
        rates = _rates(rows)
        values[index] = (
            rates["observed_asr"],
            rates["observed_bup"],
            rates["asr_plus_bup"],
        )
    return {
        "observed_asr_95ci": np.quantile(values[:, 0], [0.025, 0.975]).tolist(),
        "observed_bup_95ci": np.quantile(values[:, 1], [0.025, 0.975]).tolist(),
        "asr_plus_bup_95ci": np.quantile(values[:, 2], [0.025, 0.975]).tolist(),
    }


def _load_clean_rates(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(row["suite"]), str(row["user_task_id"])): float(
            row["base_success_rate"]
        )
        for row in payload.get("tasks", [])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="append", type=Path, required=True)
    parser.add_argument("--clean-solvability-json", type=Path)
    parser.add_argument("--min-base-success-rate", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clean_rates = _load_clean_rates(args.clean_solvability_json)
    by_selection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_seed: dict[str, dict[str, Any]] = defaultdict(dict)
    replay_metadata = []
    for path in args.replay:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = str(payload.get("seed"))
        replay_metadata.append(
            {
                "path": str(path.resolve()),
                "seed": payload.get("seed"),
                "do_sample": payload.get("do_sample"),
                "temperature": payload.get("temperature"),
                "top_p": payload.get("top_p"),
            }
        )
        for name, result in payload["results"].items():
            rows = result["rows"]
            by_selection[name].extend(rows)
            per_seed[name][seed] = _rates(rows)

    summary: dict[str, Any] = {
        "scope": "selected_real_agentdojo_multiseed_replay",
        "replays": replay_metadata,
        "bootstrap_unit": "task_injection_pair",
        "selections": {},
        "comparisons": {},
    }
    for name, rows in by_selection.items():
        pair_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            pair_rows[_pair_key(row)].append(row)
        pair_rates = []
        for key, attempts in sorted(pair_rows.items()):
            rates = _rates(attempts)
            pair_rates.append(
                {
                    "suite": key[0],
                    "user_task_id": key[1],
                    "injection_task_id": key[2],
                    **rates,
                }
            )
        eligible = [
            row
            for row in rows
            if clean_rates.get((str(row["suite"]), str(row["user_task_id"])), 0.0)
            >= args.min_base_success_rate
        ]
        summary["selections"][name] = {
            "pooled": _rates(rows),
            "per_seed": per_seed[name],
            "pair_count": len(pair_rows),
            "pair_rates": pair_rates,
            "variable_security_pair_count": sum(
                len({bool(row["security"]) for row in attempts}) > 1
                for attempts in pair_rows.values()
            ),
            "variable_utility_pair_count": sum(
                len({bool(row["utility"]) for row in attempts}) > 1
                for attempts in pair_rows.values()
            ),
            "conditional": {
                **_rates(eligible),
                "coverage": len(eligible) / len(rows) if rows else 0.0,
            },
            "cluster_bootstrap": _cluster_bootstrap(
                pair_rows,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
        }

    for left_name, right_name in itertools.combinations(sorted(by_selection), 2):
        left_rows = by_selection[left_name]
        right_rows = by_selection[right_name]
        left_pairs = {_pair_key(row) for row in left_rows}
        right_pairs = {_pair_key(row) for row in right_rows}
        shared = left_pairs & right_pairs
        left_only = left_pairs - shared
        right_only = right_pairs - shared
        left_rates = _rates(left_rows)
        right_rates = _rates(right_rows)
        common_seeds = sorted(set(per_seed[left_name]) & set(per_seed[right_name]))
        per_seed_difference = {}
        for seed in common_seeds:
            left_seed = per_seed[left_name][seed]
            right_seed = per_seed[right_name][seed]
            per_seed_difference[seed] = {
                "observed_asr": left_seed["observed_asr"]
                - right_seed["observed_asr"],
                "observed_bup": left_seed["observed_bup"]
                - right_seed["observed_bup"],
                "asr_plus_bup": left_seed["asr_plus_bup"]
                - right_seed["asr_plus_bup"],
            }
        summary["comparisons"][f"{left_name}__minus__{right_name}"] = {
            "shared_pair_count": len(shared),
            "left_only_pair_count": len(left_only),
            "right_only_pair_count": len(right_only),
            "pooled_difference": {
                "observed_asr": left_rates["observed_asr"]
                - right_rates["observed_asr"],
                "observed_bup": left_rates["observed_bup"]
                - right_rates["observed_bup"],
                "asr_plus_bup": left_rates["asr_plus_bup"]
                - right_rates["asr_plus_bup"],
            },
            "per_seed_difference": per_seed_difference,
            "left_only": _rates(
                [row for row in left_rows if _pair_key(row) in left_only]
            ),
            "right_only": _rates(
                [row for row in right_rows if _pair_key(row) in right_only]
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
