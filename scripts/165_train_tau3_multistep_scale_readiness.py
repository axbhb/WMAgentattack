"""Train the frozen tau3 multi-step action and transition comparison."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.hybrid_semantic_world_model import (
    HybridSemanticWorldModel,
    assert_no_planning_or_value_heads,
)
from wmagentattack.tau3_multistep import file_sha256
from wmagentattack.tau3_multistep_experiment import (
    BASELINE_VARIANTS,
    NEURAL_VARIANTS,
    build_arrays,
    candidate_text,
    flatten_dataset,
    frequency_action_probabilities,
    frequency_transition_probabilities,
    set_seed,
    task_balanced_weights,
    tfidf_state_text,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
                + "\n"
            )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def _tfidf_probabilities(
    prefixes: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    definition: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    pairs: list[str] = []
    labels: list[int] = []
    pair_weights: list[float] = []
    training = [row for row in prefixes if row["split"] == "training"]
    row_weights = task_balanced_weights([row["task_id"] for row in training])
    for row, row_weight in zip(training, row_weights):
        state = tfidf_state_text(row)
        legal = list(row["source_prefix"]["features"]["legal_tools"])
        target = str(row["source_prefix"]["targets"]["next_action"])
        for candidate in legal:
            pairs.append(state + "\nCANDIDATE " + candidate_text(catalog[candidate]))
            labels.append(int(candidate == target))
            pair_weights.append(float(row_weight) / len(legal))
    vectorizer = TfidfVectorizer(
        max_features=int(definition["tfidf_max_features"]),
        ngram_range=tuple(definition["tfidf_ngram_range"]),
        lowercase=True,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(pairs)
    estimator = LogisticRegression(
        C=float(definition["tfidf_logistic_C"]),
        solver="liblinear",
        max_iter=1000,
        random_state=0,
    )
    estimator.fit(
        matrix,
        np.asarray(labels, dtype=np.int64),
        sample_weight=np.asarray(pair_weights, dtype=np.float64),
    )
    output = np.zeros_like(arrays["legal"], dtype=np.float64)
    for row_index, row in enumerate(prefixes):
        state = tfidf_state_text(row)
        legal = list(row["source_prefix"]["features"]["legal_tools"])
        pair_texts = [
            state + "\nCANDIDATE " + candidate_text(catalog[candidate])
            for candidate in legal
        ]
        scores = np.asarray(
            estimator.decision_function(vectorizer.transform(pair_texts)),
            dtype=np.float64,
        )
        probabilities = _softmax(scores)
        for candidate, probability in zip(legal, probabilities):
            output[row_index, arrays["candidate_index"][candidate]] = probability
    return output, {
        "training_pairs": len(pairs),
        "positive_pairs": int(sum(labels)),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "iterations": int(np.max(estimator.n_iter_)),
    }


def _train_neural(
    *,
    prefixes: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    supported_target_indices: Sequence[int],
    protocol: Mapping[str, Any],
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    training_prefix_indices = np.asarray(
        [index for index, row in enumerate(prefixes) if row["split"] == "training"],
        dtype=np.int64,
    )
    training_transition_indices = np.asarray(
        [
            index
            for index, row in enumerate(transitions)
            if row["split"] == "training"
        ],
        dtype=np.int64,
    )
    global_to_local = {
        int(global_index): local_index
        for local_index, global_index in enumerate(training_prefix_indices)
    }
    states = torch.as_tensor(
        arrays["states"][training_prefix_indices],
        dtype=torch.float32,
        device=device,
    )
    candidates = torch.as_tensor(
        arrays["candidate_inputs"], dtype=torch.float32, device=device
    )
    legal = torch.as_tensor(
        arrays["legal"][training_prefix_indices], dtype=torch.bool, device=device
    )
    action_targets = torch.as_tensor(
        arrays["targets"][training_prefix_indices], dtype=torch.long, device=device
    )
    action_weights = torch.as_tensor(
        task_balanced_weights(
            [prefixes[index]["task_id"] for index in training_prefix_indices]
        ),
        dtype=torch.float32,
        device=device,
    )
    transition_local_rows = torch.as_tensor(
        [
            global_to_local[int(transitions[index]["prefix_row_index"])]
            for index in training_transition_indices
        ],
        dtype=torch.long,
        device=device,
    )
    transition_candidates = torch.as_tensor(
        [
            arrays["candidate_index"][transitions[index]["action"]]
            for index in training_transition_indices
        ],
        dtype=torch.long,
        device=device,
    )
    transition_targets = torch.as_tensor(
        np.stack(
            [transitions[index]["target"] for index in training_transition_indices]
        )[:, supported_target_indices],
        dtype=torch.float32,
        device=device,
    )
    transition_weights = torch.as_tensor(
        task_balanced_weights(
            [transitions[index]["task_id"] for index in training_transition_indices]
        ),
        dtype=torch.float32,
        device=device,
    )
    supported = torch.as_tensor(
        list(supported_target_indices), dtype=torch.long, device=device
    )
    budget = protocol["method_training_budget_if_data_gate_passes"]
    definition = protocol["method_training_definition"]
    model = HybridSemanticWorldModel(
        state_size=int(arrays["states"].shape[1]),
        candidate_size=int(arrays["candidate_inputs"].shape[1]),
        argument_keys=1,
        hidden_size=int(budget["hidden_size"]),
        dropout=float(budget["dropout"]),
    ).to(device)
    assert_no_planning_or_value_heads(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(budget["learning_rate"]),
        weight_decay=float(budget["weight_decay"]),
    )
    history = []
    for epoch in range(int(budget["fixed_epochs"])):
        set_seed(seed * 1009 + epoch)
        model.train()
        action_logits, _, transition_logits = model(states, candidates)
        masked = action_logits.masked_fill(
            ~legal, torch.finfo(action_logits.dtype).min
        )
        per_action = F.cross_entropy(masked, action_targets, reduction="none")
        action_loss = (per_action * action_weights).sum() / action_weights.sum()
        selected_transition = transition_logits[
            transition_local_rows, transition_candidates
        ][:, supported]
        per_transition = F.binary_cross_entropy_with_logits(
            selected_transition, transition_targets, reduction="none"
        ).mean(dim=1)
        transition_loss = (
            per_transition * transition_weights
        ).sum() / transition_weights.sum()
        loss = (
            float(definition["action_loss_weight"]) * action_loss
            + float(definition["transition_loss_weight"]) * transition_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in (0, int(budget["fixed_epochs"]) - 1):
            history.append(
                {
                    "epoch": epoch,
                    "total": float(loss.detach().cpu()),
                    "action": float(action_loss.detach().cpu()),
                    "transition": float(transition_loss.detach().cpu()),
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
        action_probabilities = model.action_probabilities(
            all_states, candidates, all_legal
        ).cpu().numpy()
        _, _, transition_logits = model(all_states, candidates)
        transition_probabilities = torch.sigmoid(transition_logits).cpu().numpy()
    return action_probabilities, transition_probabilities, {
        "training_prefixes": len(training_prefix_indices),
        "training_transitions": len(training_transition_indices),
        "training_tasks": len(
            {prefixes[index]["task_id"] for index in training_prefix_indices}
        ),
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        "loss_history_endpoints": history,
    }


def _action_prediction_rows(
    *,
    variant: str,
    seed: int,
    prefixes: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(prefixes):
        target = int(arrays["targets"][index])
        predicted = int(np.argmax(probabilities[index]))
        legal = bool(arrays["legal"][index, predicted])
        output.append(
            {
                "variant": variant,
                "training_seed": seed,
                "row_id": row["row_id"],
                "episode_id": row["episode_id"],
                "task_id": row["task_id"],
                "suite": row["suite"],
                "domain": row["domain"],
                "split": row["split"],
                "prefix_index": row["prefix_index"],
                "target_action": arrays["candidates"][target],
                "predicted_action": arrays["candidates"][predicted],
                "target_probability": float(probabilities[index, target]),
                "action_nll": float(
                    -math.log(max(float(probabilities[index, target]), 1e-12))
                ),
                "action_correct": float(predicted == target),
                "legal_prediction": float(legal),
            }
        )
    return output


def _transition_prediction_rows(
    *,
    variant: str,
    seed: int,
    transitions: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    target_names: Sequence[str],
    supported_names: set[str],
    arrays: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(transitions):
        if arrays is None:
            predicted = probabilities[index]
        else:
            predicted = probabilities[
                int(row["prefix_row_index"]),
                arrays["candidate_index"][row["action"]],
            ]
        for target_index, name in enumerate(target_names):
            target = float(row["target"][target_index])
            probability = float(np.clip(predicted[target_index], 1e-12, 1.0 - 1e-12))
            output.append(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "row_id": row["row_id"],
                    "episode_id": row["episode_id"],
                    "task_id": row["task_id"],
                    "suite": row["suite"],
                    "domain": row["domain"],
                    "split": row["split"],
                    "prefix_index": row["prefix_index"],
                    "executed_action": row["action"],
                    "target_name": name,
                    "supported": name in supported_names,
                    "target": target,
                    "probability": probability,
                    "bce": float(
                        -(target * math.log(probability)
                          + (1.0 - target) * math.log(1.0 - probability))
                    ),
                    "brier": float((probability - target) ** 2),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "data_gate_passed_and_method_frozen_before_training":
        raise ValueError("method protocol is not frozen after the data gate")
    freeze = protocol["method_freeze"]
    for path, expected in (
        (args.dataset, freeze["dataset_sha256"]),
        (args.audit, freeze["dataset_audit_sha256"]),
        (args.pilot_gate, freeze["pilot_gate_sha256"]),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"frozen method input hash differs: {path}")
    for relative_path, expected in freeze["implementation_sha256"].items():
        if file_sha256(ROOT / relative_path) != expected:
            raise ValueError(f"frozen method implementation differs: {relative_path}")
    pilot_gate = json.loads(args.pilot_gate.read_text(encoding="utf-8"))
    if not pilot_gate["passed"] or pilot_gate["decision"] != (
        "DATA_FORM_GO__AUTHORIZE_FROZEN_METHOD_TEST"
    ):
        raise ValueError("data gate does not authorize method training")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    target_names = list(protocol["transition_targets"]["names"])
    supported_names = list(freeze["supported_transition_target_names"])
    if supported_names != list(pilot_gate["supported_transition_target_names"]):
        raise ValueError("supported transition target set differs from the data gate")
    supported_indices = [target_names.index(name) for name in supported_names]
    prefixes, transitions = flatten_dataset(dataset, target_names)
    budget = protocol["method_training_budget_if_data_gate_passes"]
    seeds = [int(seed) for seed in budget["training_seeds"]]
    if seeds != [7, 17, 29] or int(budget["neural_training_runs"]) != (
        len(NEURAL_VARIANTS) * len(seeds)
    ):
        raise ValueError("frozen method run budget is inconsistent")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cpu":
        torch.set_num_threads(8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_path = args.output_dir / "action_predictions.jsonl"
    transition_path = args.output_dir / "transition_predictions.jsonl"
    action_path.write_text("", encoding="utf-8")
    transition_path.write_text("", encoding="utf-8")
    hash_dimension = int(budget["hash_dimension"])
    base_arrays = build_arrays(
        prefixes,
        dataset["candidate_catalog"],
        variant="semantic_markov",
        hash_dimension=hash_dimension,
    )
    frequency_action = frequency_action_probabilities(prefixes, base_arrays)
    _append_jsonl(
        action_path,
        _action_prediction_rows(
            variant="frequency_prior",
            seed=0,
            prefixes=prefixes,
            arrays=base_arrays,
            probabilities=frequency_action,
        ),
    )
    frequency_transition = frequency_transition_probabilities(
        transitions, target_count=len(target_names)
    )
    _append_jsonl(
        transition_path,
        _transition_prediction_rows(
            variant="frequency_prior",
            seed=0,
            transitions=transitions,
            probabilities=frequency_transition,
            target_names=target_names,
            supported_names=set(supported_names),
        ),
    )
    tfidf_action, tfidf_diagnostics = _tfidf_probabilities(
        prefixes,
        base_arrays,
        dataset["candidate_catalog"],
        protocol["method_training_definition"],
    )
    _append_jsonl(
        action_path,
        _action_prediction_rows(
            variant="tfidf_candidate_logistic",
            seed=0,
            prefixes=prefixes,
            arrays=base_arrays,
            probabilities=tfidf_action,
        ),
    )
    runs = [
        {"variant": "frequency_prior", "training_seed": 0},
        {
            "variant": "tfidf_candidate_logistic",
            "training_seed": 0,
            **tfidf_diagnostics,
        },
    ]
    for variant in NEURAL_VARIANTS:
        arrays = build_arrays(
            prefixes,
            dataset["candidate_catalog"],
            variant=variant,
            hash_dimension=hash_dimension,
        )
        if arrays["candidates"] != base_arrays["candidates"]:
            raise ValueError("candidate order differs across representations")
        for seed in seeds:
            set_seed(seed)
            action_probabilities, transition_probabilities, diagnostics = _train_neural(
                prefixes=prefixes,
                transitions=transitions,
                arrays=arrays,
                supported_target_indices=supported_indices,
                protocol=protocol,
                seed=seed,
                device=device,
            )
            _append_jsonl(
                action_path,
                _action_prediction_rows(
                    variant=variant,
                    seed=seed,
                    prefixes=prefixes,
                    arrays=arrays,
                    probabilities=action_probabilities,
                ),
            )
            _append_jsonl(
                transition_path,
                _transition_prediction_rows(
                    variant=variant,
                    seed=seed,
                    transitions=transitions,
                    probabilities=transition_probabilities,
                    target_names=target_names,
                    supported_names=set(supported_names),
                    arrays=arrays,
                ),
            )
            runs.append(
                {
                    "variant": variant,
                    "training_seed": seed,
                    **diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "training_seed": seed,
                        "loss": diagnostics["loss_history_endpoints"][-1],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    neural_runs = [row for row in runs if row["variant"] in NEURAL_VARIANTS]
    if len(neural_runs) != int(budget["neural_training_runs"]):
        raise ValueError("neural method budget is incomplete")
    metrics = {
        "protocol_sha256": file_sha256(args.protocol),
        "dataset_sha256": file_sha256(args.dataset),
        "dataset_audit_sha256": file_sha256(args.audit),
        "pilot_gate_sha256": file_sha256(args.pilot_gate),
        "device": device,
        "prefixes": len(prefixes),
        "transitions": len(transitions),
        "candidate_count": len(base_arrays["candidates"]),
        "supported_transition_target_names": supported_names,
        "variants": list(NEURAL_VARIANTS),
        "training_seeds": seeds,
        "runs": runs,
        "neural_runs": len(neural_runs),
        "frequency_fits": 1,
        "tfidf_fits": 1,
        "new_llm_calls": 0,
        "new_tool_executions": 0,
        "attack_examples": 0,
        "dreamer_runs": 0,
        "action_predictions_sha256": file_sha256(action_path),
        "transition_predictions_sha256": file_sha256(transition_path),
    }
    _write_json(args.output_dir / "run_metrics.json", metrics)
    print("TAU3_MULTISTEP_METHOD_TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
