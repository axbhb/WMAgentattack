"""Run the frozen multi-source current-method suitability comparison."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F

from wmagentattack.hybrid_semantic_world_model import (
    EVIDENCE_DELTA_TARGETS,
    HybridSemanticWorldModel,
    assert_no_planning_or_value_heads,
    tool_candidate_vector,
)
from wmagentattack.multisource_suitability import (
    candidate_text,
    file_sha256,
    representation_vector,
    tfidf_state_text,
)
from wmagentattack.multisource_suitability_experiment import (
    BASELINE_VARIANTS,
    NEURAL_VARIANTS,
    TRAINING_SCOPES,
    error_probe_supported,
    rows_for_scope,
    task_balanced_weights,
)


ERROR_TARGET_INDEX = EVIDENCE_DELTA_TARGETS.index("execution_error")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def _scope_arrays(
    rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    variant: str,
    hash_dimension: int,
) -> dict[str, Any]:
    candidates = sorted(
        {candidate for row in rows for candidate in row["legal_candidate_ids"]}
    )
    candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
    states = np.stack(
        [
            representation_vector(
                row, variant=variant, hash_dimension=hash_dimension
            )
            for row in rows
        ]
    )
    candidate_inputs = np.stack(
        [
            tool_candidate_vector(
                catalog[candidate], hash_dimension=hash_dimension
            )
            for candidate in candidates
        ]
    )
    legal = np.zeros((len(rows), len(candidates)), dtype=bool)
    targets = np.zeros(len(rows), dtype=np.int64)
    for row_index, row in enumerate(rows):
        for candidate in row["legal_candidate_ids"]:
            legal[row_index, candidate_index[candidate]] = True
        targets[row_index] = candidate_index[str(row["target_candidate_id"])]
        if not legal[row_index, targets[row_index]]:
            raise ValueError("target action is not legal")
    return {
        "candidates": candidates,
        "candidate_index": candidate_index,
        "states": states,
        "candidate_inputs": candidate_inputs,
        "legal": legal,
        "targets": targets,
    }


def _prediction_rows(
    *,
    scope: str,
    variant: str,
    seed: int,
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    error_probabilities: np.ndarray | None,
    error_authorized: bool,
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        target = int(targets[index])
        predicted = int(np.argmax(probabilities[index]))
        legal = predicted in {
            candidate_index
            for candidate_index, candidate in enumerate(candidates)
            if candidate in row["legal_candidate_ids"]
        }
        tool_probability = float(
            sum(
                probability
                for candidate, probability in zip(candidates, probabilities[index])
                if candidate in row["legal_candidate_ids"]
                and catalog_kind(candidate) == "tool"
            )
        )
        exact = bool(row["exact_outcome"]["available"])
        error_target = (
            float(bool(row["exact_outcome"]["execution_error"])) if exact else None
        )
        error_probability = (
            float(error_probabilities[index])
            if error_authorized and exact and error_probabilities is not None
            else None
        )
        error_bce = None
        error_brier = None
        if error_probability is not None and error_target is not None:
            clipped = min(max(error_probability, 1e-12), 1.0 - 1e-12)
            error_bce = float(
                -(error_target * math.log(clipped)
                  + (1.0 - error_target) * math.log(1.0 - clipped))
            )
            error_brier = float((error_probability - error_target) ** 2)
        output.append(
            {
                "prediction_type": "action_and_error",
                "scope": scope,
                "variant": variant,
                "training_seed": seed,
                "row_id": row["row_id"],
                "source": row["source"],
                "task_key": row["task_key"],
                "group_id": row["group_id"],
                "record_variant": row["variant"],
                "split": row["split"],
                "target_candidate_id": candidates[target],
                "predicted_candidate_id": candidates[predicted],
                "target_probability": float(probabilities[index, target]),
                "action_nll": float(
                    -math.log(max(float(probabilities[index, target]), 1e-12))
                ),
                "action_correct": float(predicted == target),
                "tool_brier": float(
                    (tool_probability - float(bool(row["target_is_tool"]))) ** 2
                ),
                "legal_prediction": float(legal),
                "exact_outcome_available": exact,
                "execution_error_target": error_target,
                "execution_error_probability": error_probability,
                "execution_error_bce": error_bce,
                "execution_error_brier": error_brier,
            }
        )
    return output


_CANDIDATE_KIND: dict[str, str] = {}


def catalog_kind(candidate: str) -> str:
    return _CANDIDATE_KIND[candidate]


def _frequency_probabilities(
    rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, Any]
) -> np.ndarray:
    candidates = arrays["candidates"]
    index = arrays["candidate_index"]
    counts = np.ones(len(candidates), dtype=np.float64)
    for row in rows:
        if row["split"] == "training":
            counts[index[str(row["target_candidate_id"])]] += 1.0
    probabilities = np.zeros_like(arrays["legal"], dtype=np.float64)
    for row_index, legal in enumerate(arrays["legal"]):
        values = counts * legal
        probabilities[row_index] = values / values.sum()
    return probabilities


def _frequency_error_probability(
    rows: Sequence[Mapping[str, Any]], *, authorized: bool
) -> np.ndarray | None:
    if not authorized:
        return None
    exact_training = [
        row
        for row in rows
        if row["split"] == "training" and row["exact_outcome"]["available"]
    ]
    errors = sum(bool(row["exact_outcome"]["execution_error"]) for row in exact_training)
    probability = (errors + 1.0) / (len(exact_training) + 2.0)
    return np.full(len(rows), probability, dtype=np.float64)


def _tfidf_probabilities(
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    training_pairs: list[str] = []
    labels: list[int] = []
    for row in rows:
        if row["split"] != "training":
            continue
        state = tfidf_state_text(row)
        for candidate in row["legal_candidate_ids"]:
            training_pairs.append(
                state + "\nCANDIDATE " + candidate_text(catalog[candidate])
            )
            labels.append(int(candidate == row["target_candidate_id"]))
    vectorizer = TfidfVectorizer(
        max_features=int(protocol["training"]["tfidf_max_features"]),
        ngram_range=tuple(protocol["training"]["tfidf_ngram_range"]),
        lowercase=True,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(training_pairs)
    estimator = LogisticRegression(
        C=float(protocol["training"]["tfidf_logistic_C"]),
        solver="liblinear",
        max_iter=1000,
        random_state=0,
    )
    estimator.fit(matrix, np.asarray(labels, dtype=np.int64))
    probabilities = np.zeros_like(arrays["legal"], dtype=np.float64)
    candidates = arrays["candidates"]
    for row_index, row in enumerate(rows):
        state = tfidf_state_text(row)
        legal_candidates = list(row["legal_candidate_ids"])
        pair_text = [
            state + "\nCANDIDATE " + candidate_text(catalog[candidate])
            for candidate in legal_candidates
        ]
        scores = estimator.decision_function(vectorizer.transform(pair_text))
        legal_probabilities = _softmax(np.asarray(scores, dtype=np.float64))
        for candidate, probability in zip(legal_candidates, legal_probabilities):
            probabilities[row_index, candidates.index(candidate)] = probability
    return probabilities, {
        "training_pairs": len(training_pairs),
        "positive_pairs": int(sum(labels)),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "iterations": int(np.max(estimator.n_iter_)),
    }


def _train_neural(
    *,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    protocol: Mapping[str, Any],
    seed: int,
    device: str,
    error_authorized: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    training_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == "training"],
        dtype=np.int64,
    )
    states = torch.as_tensor(
        arrays["states"][training_indices], dtype=torch.float32, device=device
    )
    candidates = torch.as_tensor(
        arrays["candidate_inputs"], dtype=torch.float32, device=device
    )
    legal = torch.as_tensor(
        arrays["legal"][training_indices], dtype=torch.bool, device=device
    )
    targets = torch.as_tensor(
        arrays["targets"][training_indices], dtype=torch.long, device=device
    )
    action_weights = torch.as_tensor(
        task_balanced_weights([rows[index]["task_key"] for index in training_indices]),
        dtype=torch.float32,
        device=device,
    )
    global_to_local = {
        int(global_index): local_index
        for local_index, global_index in enumerate(training_indices)
    }
    exact_global = [
        int(index)
        for index in training_indices
        if rows[int(index)]["exact_outcome"]["available"]
    ] if error_authorized else []
    error_local = torch.as_tensor(
        [global_to_local[index] for index in exact_global],
        dtype=torch.long,
        device=device,
    )
    error_candidates = torch.as_tensor(
        [
            arrays["candidate_index"][rows[index]["target_candidate_id"]]
            for index in exact_global
        ],
        dtype=torch.long,
        device=device,
    )
    error_targets = torch.as_tensor(
        [
            float(bool(rows[index]["exact_outcome"]["execution_error"]))
            for index in exact_global
        ],
        dtype=torch.float32,
        device=device,
    )
    error_weights = (
        torch.as_tensor(
            task_balanced_weights([rows[index]["task_key"] for index in exact_global]),
            dtype=torch.float32,
            device=device,
        )
        if exact_global
        else None
    )

    model = HybridSemanticWorldModel(
        state_size=int(arrays["states"].shape[1]),
        candidate_size=int(arrays["candidate_inputs"].shape[1]),
        argument_keys=1,
        hidden_size=int(protocol["training"]["hidden_size"]),
        dropout=float(protocol["training"]["dropout"]),
    ).to(device)
    assert_no_planning_or_value_heads(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    history = []
    for epoch in range(int(protocol["training"]["fixed_epochs"])):
        _set_seed(seed * 1009 + epoch)
        model.train()
        action_logits, _, evidence_logits = model(states, candidates)
        masked = action_logits.masked_fill(
            ~legal, torch.finfo(action_logits.dtype).min
        )
        per_action = F.cross_entropy(masked, targets, reduction="none")
        action_loss = (per_action * action_weights).sum() / action_weights.sum()
        error_loss = torch.zeros((), dtype=torch.float32, device=device)
        if exact_global and error_weights is not None:
            selected = evidence_logits[
                error_local, error_candidates, ERROR_TARGET_INDEX
            ]
            per_error = F.binary_cross_entropy_with_logits(
                selected, error_targets, reduction="none"
            )
            error_loss = (per_error * error_weights).sum() / error_weights.sum()
        loss = action_loss + float(
            protocol["training"]["evidence_error_loss_weight"]
        ) * error_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in (0, int(protocol["training"]["fixed_epochs"]) - 1):
            history.append(
                {
                    "epoch": epoch,
                    "total": float(loss.detach().cpu()),
                    "action": float(action_loss.detach().cpu()),
                    "execution_error": float(error_loss.detach().cpu()),
                }
            )

    model.eval()
    with torch.no_grad():
        all_states = torch.as_tensor(
            arrays["states"], dtype=torch.float32, device=device
        )
        all_legal = torch.as_tensor(
            arrays["legal"], dtype=torch.bool, device=device
        )
        action_logits, _, evidence_logits = model(all_states, candidates)
        probabilities = model.action_probabilities(
            all_states, candidates, all_legal
        ).cpu().numpy()
        error_probabilities = (
            torch.sigmoid(
                evidence_logits[
                    torch.arange(len(rows), device=device),
                    torch.as_tensor(arrays["targets"], device=device),
                    ERROR_TARGET_INDEX,
                ]
            )
            .cpu()
            .numpy()
            if error_authorized
            else None
        )
    return probabilities, error_probabilities, {
        "training_rows": len(training_indices),
        "training_tasks": len({rows[index]["task_key"] for index in training_indices}),
        "training_exact_rows": len(exact_global),
        "loss_history_endpoints": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preflight_passed_and_frozen_before_model_training":
        raise ValueError("protocol was not frozen before model training")
    frozen = protocol["frozen_adapter_dataset"]
    if file_sha256(args.dataset) != frozen["sha256"]:
        raise ValueError("frozen adapter dataset hash mismatch")
    if file_sha256(args.audit) != frozen["audit_sha256"]:
        raise ValueError("frozen adapter audit hash mismatch")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("frozen adapter audit did not pass")
    rows = dataset["rows"]
    catalog = dataset["candidate_catalog"]
    global _CANDIDATE_KIND
    _CANDIDATE_KIND = {
        candidate: str(descriptor["kind"])
        for candidate, descriptor in catalog.items()
    }
    if tuple(protocol["training"]["scopes"]) != TRAINING_SCOPES:
        raise ValueError("training scopes differ from frozen protocol")
    seeds = [int(seed) for seed in protocol["training"]["training_seeds"]]
    if seeds != [7, 17, 29]:
        raise ValueError("training seeds differ from frozen protocol")
    if int(protocol["fixed_budget"]["neural_training_runs"]) != (
        len(NEURAL_VARIANTS) * len(TRAINING_SCOPES) * len(seeds)
    ):
        raise ValueError("frozen neural run budget is inconsistent")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cpu":
        torch.set_num_threads(8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    runs: list[dict[str, Any]] = []
    support: dict[str, Any] = {}
    hash_dimension = int(protocol["training"]["hash_dimension"])

    for scope in TRAINING_SCOPES:
        selected = rows_for_scope(rows, scope)
        authorized, support_counts = error_probe_supported(
            selected, protocol["preflight_gate"]
        )
        support[scope] = {"authorized": authorized, **support_counts}
        baseline_arrays = _scope_arrays(
            selected,
            catalog,
            variant="structured_markov_v3",
            hash_dimension=hash_dimension,
        )

        frequency = _frequency_probabilities(selected, baseline_arrays)
        frequency_error = _frequency_error_probability(
            selected, authorized=authorized
        )
        _append_jsonl(
            predictions_path,
            _prediction_rows(
                scope=scope,
                variant="frequency_prior",
                seed=0,
                rows=selected,
                candidates=baseline_arrays["candidates"],
                targets=baseline_arrays["targets"],
                probabilities=frequency,
                error_probabilities=frequency_error,
                error_authorized=authorized,
            ),
        )
        runs.append(
            {
                "scope": scope,
                "variant": "frequency_prior",
                "training_seed": 0,
                "rows": len(selected),
                "candidates": len(baseline_arrays["candidates"]),
                "error_probe_authorized": authorized,
            }
        )

        tfidf, tfidf_diagnostics = _tfidf_probabilities(
            selected, baseline_arrays, catalog, protocol
        )
        _append_jsonl(
            predictions_path,
            _prediction_rows(
                scope=scope,
                variant="tfidf_candidate_logistic",
                seed=0,
                rows=selected,
                candidates=baseline_arrays["candidates"],
                targets=baseline_arrays["targets"],
                probabilities=tfidf,
                error_probabilities=None,
                error_authorized=False,
            ),
        )
        runs.append(
            {
                "scope": scope,
                "variant": "tfidf_candidate_logistic",
                "training_seed": 0,
                "rows": len(selected),
                "candidates": len(baseline_arrays["candidates"]),
                "error_probe_authorized": False,
                **tfidf_diagnostics,
            }
        )

        for variant in NEURAL_VARIANTS:
            arrays = _scope_arrays(
                selected,
                catalog,
                variant=variant,
                hash_dimension=hash_dimension,
            )
            if arrays["candidates"] != baseline_arrays["candidates"]:
                raise ValueError("candidate order changed across representations")
            for seed in seeds:
                _set_seed(seed)
                probabilities, error_probabilities, diagnostics = _train_neural(
                    rows=selected,
                    arrays=arrays,
                    protocol=protocol,
                    seed=seed,
                    device=device,
                    error_authorized=authorized,
                )
                _append_jsonl(
                    predictions_path,
                    _prediction_rows(
                        scope=scope,
                        variant=variant,
                        seed=seed,
                        rows=selected,
                        candidates=arrays["candidates"],
                        targets=arrays["targets"],
                        probabilities=probabilities,
                        error_probabilities=error_probabilities,
                        error_authorized=authorized,
                    ),
                )
                runs.append(
                    {
                        "scope": scope,
                        "variant": variant,
                        "training_seed": seed,
                        "rows": len(selected),
                        "candidates": len(arrays["candidates"]),
                        "error_probe_authorized": authorized,
                        **diagnostics,
                    }
                )

    neural_runs = [row for row in runs if row["variant"] in NEURAL_VARIANTS]
    frequency_runs = [row for row in runs if row["variant"] == BASELINE_VARIANTS[0]]
    tfidf_runs = [row for row in runs if row["variant"] == BASELINE_VARIANTS[1]]
    if len(neural_runs) != int(protocol["fixed_budget"]["neural_training_runs"]):
        raise ValueError("neural training run budget is incomplete")
    if len(frequency_runs) != int(protocol["fixed_budget"]["frequency_fits"]):
        raise ValueError("frequency fit budget is incomplete")
    if len(tfidf_runs) != int(protocol["fixed_budget"]["tfidf_fits"]):
        raise ValueError("TF-IDF fit budget is incomplete")
    run_metrics = {
        "protocol_sha256": file_sha256(args.protocol),
        "dataset_sha256": file_sha256(args.dataset),
        "audit_sha256": file_sha256(args.audit),
        "device": device,
        "scopes": list(TRAINING_SCOPES),
        "neural_variants": list(NEURAL_VARIANTS),
        "baseline_variants": list(BASELINE_VARIANTS),
        "training_seeds": seeds,
        "error_probe_support": support,
        "runs": runs,
        "neural_runs": len(neural_runs),
        "frequency_fits": len(frequency_runs),
        "tfidf_fits": len(tfidf_runs),
        "new_llm_calls": 0,
        "new_tool_executions": 0,
        "attack_examples": 0,
        "dreamer_runs": 0,
    }
    _write_json(args.output_dir / "run_metrics.json", run_metrics)
    print(json.dumps(run_metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
