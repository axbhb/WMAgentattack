"""Audit the exact state-feature lineage used by frozen Slurm job 4722."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


STATE_VECTOR_VARIANTS = (
    "state_only",
    "semantic_markov_state",
    "semantic_markov_state_evidence",
    "semantic_markov_state_shuffled_evidence",
    "semantic_markov_state_output_length",
)
NON_STATE_VECTOR_VARIANTS = ("static_length", "semantic_markov")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _line_numbers(path: Path, patterns: list[str]) -> dict[str, list[int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        pattern: [index for index, line in enumerate(lines, start=1) if pattern in line]
        for pattern in patterns
    }


def _load_probe_module(source_root: Path):  # noqa: ANN202
    path = source_root / "src" / "wmagentattack" / "clean_evidence_probe.py"
    spec = importlib.util.spec_from_file_location("job4722_clean_evidence_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen probe module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feature_probe(module, prefix: dict[str, Any], hash_dimension: int = 64) -> dict[str, Any]:
    perturbed = copy.deepcopy(prefix)
    canonical = perturbed["features"]["canonical_state"]
    if not isinstance(canonical, dict):
        raise TypeError("audit prefix canonical state is not a mapping")
    canonical["__JOB4722_AUDIT_SENTINEL__"] = {
        "hidden_entity": "must-change-state-feature-only"
    }
    changed = {}
    for variant in (*STATE_VECTOR_VARIANTS, *NON_STATE_VECTOR_VARIANTS):
        before = module.vector_features(
            prefix, variant=variant, hash_dimension=hash_dimension
        )
        after = module.vector_features(
            perturbed, variant=variant, hash_dimension=hash_dimension
        )
        changed[variant] = bool(not np.array_equal(before, after))
    transformer_before = module.transformer_step_features(
        prefix, hash_dimension=hash_dimension
    )
    transformer_after = module.transformer_step_features(
        perturbed, hash_dimension=hash_dimension
    )
    changed["event_transformer_state_evidence"] = bool(
        not np.array_equal(transformer_before, transformer_after)
    )
    return changed


def _verify_snapshot(source_root: Path) -> dict[str, Any]:
    checksum_path = source_root / "code.prerun.sha256"
    rows = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = source_root / relative
        actual = _sha256(path)
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": expected == actual,
            }
        )
    return {"files": rows, "all_match": all(row["matches"] for row in rows)}


def audit(protocol: dict[str, Any], source_root: Path, dataset: Path) -> dict[str, Any]:
    snapshot = _verify_snapshot(source_root)
    if not snapshot["all_match"]:
        raise ValueError("frozen source snapshot checksum mismatch")
    if _sha256(dataset) != protocol["dataset_sha256"]:
        raise ValueError("frozen dataset checksum mismatch")
    episodes = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line
    ]
    prefixes = [prefix for episode in episodes for prefix in episode["prefixes"]]
    if len(episodes) != protocol["expected_episodes"]:
        raise ValueError("episode count mismatch")
    if len(prefixes) != protocol["expected_prefixes"]:
        raise ValueError("prefix count mismatch")
    if not all("canonical_state" in prefix["features"] for prefix in prefixes):
        raise ValueError("canonical state is missing from at least one prefix")

    builder = source_root / "scripts" / "121_build_clean_evidence_ledger_dataset.py"
    probe = source_root / "src" / "wmagentattack" / "clean_evidence_probe.py"
    training = source_root / "scripts" / "122_train_clean_evidence_ablation.py"
    source_evidence = {
        "builder": {
            "path": str(builder),
            "line_matches": _line_numbers(
                builder,
                [
                    '"canonical_state": canonical_json_value(pre_environment)',
                    '"canonical_state": transition.canonical_state_after',
                    '"last_state_changed": transition.state_changed',
                    'roots = _delta_roots(transition.canonical_state_delta)',
                ],
            ),
        },
        "feature_constructor": {
            "path": str(probe),
            "line_matches": _line_numbers(
                probe,
                [
                    'hashed_text(features["canonical_state"]',
                    'features["state_summary"], hash_dimension',
                    'elif variant == "state_only"',
                    'elif variant == "semantic_markov_state"',
                    'elif variant == "semantic_markov_state_output_length"',
                ],
            ),
        },
        "model_tensor": {
            "path": str(training),
            "line_matches": _line_numbers(
                training,
                [
                    "inputs, masks = _encode_variant(",
                    "inputs=inputs[train_global]",
                    "inputs=inputs[test_global]",
                ],
            ),
        },
    }
    if any(
        not rows
        for source in source_evidence.values()
        for rows in source["line_matches"].values()
    ):
        raise ValueError("expected frozen source lineage marker is missing")

    module = _load_probe_module(source_root)
    changed = _feature_probe(module, prefixes[0])
    state_variants_change = all(changed[variant] for variant in STATE_VECTOR_VARIANTS)
    non_state_ignore = all(not changed[variant] for variant in NON_STATE_VECTOR_VARIANTS)
    transformer_changes = changed["event_transformer_state_evidence"]
    root_keys = sorted(
        {
            str(key)
            for prefix in prefixes
            for key in prefix["features"]["canonical_state"].keys()
        }
    )
    state_hashes = {
        _canonical_hash(prefix["features"]["canonical_state"]) for prefix in prefixes
    }

    field_table = {
        "state_changed": {
            "used": True,
            "route": "state_summary.last_state_changed -> numeric state vector",
        },
        "state_delta_operation_count": {
            "used": False,
            "route": "canonical delta is reduced to roots; operation count is not stored",
        },
        "state_delta_roots": {
            "used": True,
            "route": "cumulative state_summary.delta_roots -> hashed state-summary vector",
        },
        "state_before_fingerprint": {"used": False, "route": "not in prefix features"},
        "state_after_fingerprint": {"used": False, "route": "not in prefix features"},
        "full_state_before_canonical_json": {
            "used": "initial_prefix_only",
            "route": "prefix 0 stores the pre-execution initial canonical environment; later prefixes do not separately store last-call state_before",
        },
        "full_state_after_canonical_json": {
            "used": True,
            "route": "each executed prefix stores transition.canonical_state_after as canonical_state",
        },
        "initial_current_structured_entity_content": {
            "used": True,
            "route": "full nested content is stored, then serialized and compressed by a 64-dimensional signed token hash; no entity-aware state encoder was used",
        },
    }
    gates = {
        "source_snapshot_checksums_valid": snapshot["all_match"],
        "dataset_hash_valid": True,
        "all_prefixes_have_canonical_state": True,
        "state_variants_respond_to_canonical_content_perturbation": (
            state_variants_change and transformer_changes
        ),
        "non_state_variants_ignore_canonical_content_perturbation": non_state_ignore,
        "no_job_rerun": True,
        "no_outcome_comparison": True,
    }
    decision = (
        protocol["interpretation"]["full_state_used"]
        if all(gates.values())
        else protocol["interpretation"]["summary_only"]
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "frozen_job_decision_unchanged": protocol["frozen_decision"],
        "none_accepted_unchanged": True,
        "job_rerun_required": False,
        "counts": {
            "episodes": len(episodes),
            "prefixes": len(prefixes),
            "distinct_canonical_states": len(state_hashes),
            "canonical_state_root_keys": root_keys,
        },
        "field_table": field_table,
        "functional_perturbation": changed,
        "source_lineage": source_evidence,
        "snapshot_integrity": snapshot,
        "gates": gates,
        "visibility_clarification": {
            "victim_proposal_model_present": False,
            "full_state_entered_progress_encoder_for_state_variants": True,
            "utility_head_used_frozen_progress_encoding": True,
            "full_state_was_causal_current_simulator_state": True,
            "full_state_was_not_proven_victim_observable": True,
            "privileged_victim_policy_leakage_claim_for_job4722": "not_applicable_because_no_victim_next-action_policy_was_trained",
        },
        "claim_boundary": "Job 4722 tested a full-current-state token-hash branch, not an entity-aware exact-state encoder. It must not be cited as evidence that exact structured canonical state has no value, and it must not be rerun on the old panel.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_execution":
        raise ValueError("audit protocol was not frozen before execution")
    result = audit(protocol, args.source_root, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
