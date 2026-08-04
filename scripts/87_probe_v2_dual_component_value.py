"""Probe separate attack and utility value heads with a validation utility guard.

This is a one-factor follow-up to the frozen E5+structured joint-value probe.
The representation, estimator, and ridge alpha remain fixed.  Only the target
decomposition and deterministic selection recipe change.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts" / "85_probe_v2_semantic_configuration_value.py"
SPEC = importlib.util.spec_from_file_location("semantic_probe", PROBE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)

RIDGE_ALPHA = 10.0
ESTIMATOR = "pairwise_ridge"
UTILITY_ESTIMATORS = (*PROBE.ESTIMATORS, "hierarchical_domain_family")
RECIPES = (
    "joint_control",
    "dual_sum",
    "attack_plus_2utility",
    "attack_plus_4utility",
    "joint_plus_utility",
    "utility_only",
    "guarded_attack_delta_0.00",
    "guarded_attack_delta_0.05",
    "guarded_attack_delta_0.10",
    "guarded_joint_delta_0.00",
    "guarded_joint_delta_0.05",
    "guarded_joint_delta_0.10",
)


def _component_rows(
    rows: list[dict[str, Any]], target_field: str
) -> list[dict[str, Any]]:
    return [{**row, "target": float(row[target_field])} for row in rows]


def _domain_interaction_vocab(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    values = []
    for row in rows:
        domain = str(row["task_key"]).split("|", 1)[0]
        family, _, _ = PROBE._configuration_structure(str(row["group_id"]))
        values.append((domain, f"{domain}|{family}"))
    return {
        "domain": sorted({value[0] for value in values}),
        "domain_family": sorted({value[1] for value in values}),
    }


def _domain_interaction_matrix(
    rows: list[dict[str, Any]], vocab: dict[str, list[str]]
) -> np.ndarray:
    domain_lookup = {value: index for index, value in enumerate(vocab["domain"])}
    offset = len(domain_lookup)
    family_lookup = {
        value: offset + index for index, value in enumerate(vocab["domain_family"])
    }
    matrix = np.zeros((len(rows), offset + len(family_lookup)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        domain = str(row["task_key"]).split("|", 1)[0]
        family, _, _ = PROBE._configuration_structure(str(row["group_id"]))
        domain_column = domain_lookup.get(domain)
        family_column = family_lookup.get(f"{domain}|{family}")
        if domain_column is not None:
            matrix[row_index, domain_column] = 1.0
        if family_column is not None:
            matrix[row_index, family_column] = 1.0
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def _fit_hierarchical_utility(
    rows: list[dict[str, Any]], *, shrinkage: float
) -> dict[str, Any]:
    domain_values: dict[str, list[float]] = defaultdict(list)
    domain_family_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        domain = str(row["task_key"]).split("|", 1)[0]
        family, _, _ = PROBE._configuration_structure(str(row["group_id"]))
        target = float(row["target_bup"])
        domain_values[domain].append(target)
        domain_family_values[f"{domain}|{family}"].append(target)
    global_mean = float(np.mean([value for values in domain_values.values() for value in values]))
    domain_means = {
        domain: float(np.mean(values)) for domain, values in domain_values.items()
    }
    domain_family_means = {}
    for key, values in domain_family_values.items():
        domain = key.split("|", 1)[0]
        prior = domain_means[domain]
        domain_family_means[key] = float(
            (sum(values) + shrinkage * prior) / (len(values) + shrinkage)
        )
    return {
        "model_type": "hierarchical_domain_family",
        "shrinkage": shrinkage,
        "global_mean": global_mean,
        "domain_means": domain_means,
        "domain_family_means": domain_family_means,
    }


def _predict_hierarchical_utility(
    model: dict[str, Any], rows: list[dict[str, Any]]
) -> np.ndarray:
    predictions = []
    for row in rows:
        domain = str(row["task_key"]).split("|", 1)[0]
        family, _, _ = PROBE._configuration_structure(str(row["group_id"]))
        predictions.append(
            model["domain_family_means"].get(
                f"{domain}|{family}",
                model["domain_means"].get(domain, model["global_mean"]),
            )
        )
    return np.asarray(predictions, dtype=np.float64)


def _guarded_scores(
    rows: list[dict[str, Any]],
    *,
    base_scores: np.ndarray,
    utility_predictions: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Rank by base score only among candidates near predicted max utility."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["task_key"])].append(index)
    scores = np.zeros(len(rows), dtype=np.float64)
    for indices in grouped.values():
        index_array = np.asarray(indices, dtype=np.int64)
        task_base = base_scores[index_array]
        task_utility = utility_predictions[index_array]
        eligible = task_utility + 1e-12 >= float(task_utility.max()) - tolerance
        base_span = float(task_base.max() - task_base.min())
        normalized_base = (
            (task_base - task_base.min()) / base_span
            if base_span > 1e-12
            else np.zeros_like(task_base)
        )
        utility_span = float(task_utility.max() - task_utility.min())
        normalized_utility = (
            (task_utility - task_utility.min()) / utility_span
            if utility_span > 1e-12
            else np.zeros_like(task_utility)
        )
        task_scores = normalized_utility
        task_scores[eligible] = 2.0 + normalized_base[eligible]
        scores[index_array] = task_scores
    return scores


def _recipe_scores(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, np.ndarray]],
    recipe: str,
) -> np.ndarray:
    joint = predictions["joint"]["prediction"]
    attack = predictions["attack"]["prediction"]
    utility = predictions["utility"]["prediction"]
    if recipe == "joint_control":
        return predictions["joint"]["raw"]
    if recipe == "dual_sum":
        return attack + utility
    if recipe == "attack_plus_2utility":
        return attack + 2.0 * utility
    if recipe == "attack_plus_4utility":
        return attack + 4.0 * utility
    if recipe == "joint_plus_utility":
        return joint + utility
    if recipe == "utility_only":
        return predictions["utility"]["raw"]
    if recipe.startswith("guarded_attack_delta_"):
        tolerance = float(recipe.rsplit("_", 1)[1])
        return _guarded_scores(
            rows,
            base_scores=predictions["attack"]["raw"],
            utility_predictions=utility,
            tolerance=tolerance,
        )
    if recipe.startswith("guarded_joint_delta_"):
        tolerance = float(recipe.rsplit("_", 1)[1])
        return _guarded_scores(
            rows,
            base_scores=predictions["joint"]["raw"],
            utility_predictions=utility,
            tolerance=tolerance,
        )
    raise ValueError(recipe)


def _fit_models(
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    utility_estimator: str = ESTIMATOR,
    utility_alpha: float = RIDGE_ALPHA,
    utility_shrinkage: float = 4.0,
) -> dict[str, Any]:
    models = {
        "joint": PROBE._ridge_fit(
            matrix, rows, estimator=ESTIMATOR, alpha=RIDGE_ALPHA
        ),
        "attack": PROBE._ridge_fit(
            matrix,
            _component_rows(rows, "target_asr"),
            estimator=ESTIMATOR,
            alpha=RIDGE_ALPHA,
        ),
    }
    if utility_estimator == "hierarchical_domain_family":
        models["utility"] = _fit_hierarchical_utility(
            rows, shrinkage=utility_shrinkage
        )
    else:
        models["utility"] = PROBE._ridge_fit(
            matrix,
            _component_rows(rows, "target_bup"),
            estimator=utility_estimator,
            alpha=utility_alpha,
        )
    return models


def _predict_models(
    models: dict[str, Any], matrix: np.ndarray, rows: list[dict[str, Any]]
) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for name, model in models.items():
        if model.get("model_type") == "hierarchical_domain_family":
            prediction = _predict_hierarchical_utility(model, rows)
            raw = prediction.copy()
        else:
            raw, prediction = PROBE._ridge_predict(model, matrix)
        if name != "joint":
            prediction = np.clip(prediction, 0.0, 1.0)
        output[name] = {"raw": raw, "prediction": prediction}
    return output


def _evaluate_recipes(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, Any]]:
    return {
        recipe: PROBE._evaluate(
            rows,
            rank_scores=_recipe_scores(rows, predictions, recipe),
            predictions=predictions["joint"]["prediction"],
        )
        for recipe in RECIPES
    }


def _select_recipe(
    validation: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, bool]]:
    control_bup = float(validation["joint_control"]["top1_target_BUP"])
    eligible = {
        recipe: float(metrics["top1_target_BUP"]) + 1e-12 >= control_bup
        for recipe, metrics in validation.items()
    }
    order = {recipe: index for index, recipe in enumerate(RECIPES)}

    def correlation(recipe: str) -> float:
        value = validation[recipe]["mean_task_spearman"]
        return float(value) if value is not None else -np.inf

    selected = max(
        (recipe for recipe in RECIPES if eligible[recipe]),
        key=lambda recipe: (
            float(validation[recipe]["top1_target_ASR_plus_BUP"]),
            float(validation[recipe]["top1_target_BUP"]),
            correlation(recipe),
            -order[recipe],
        ),
    )
    return selected, eligible


def _build_matrices(
    rows: dict[str, list[dict[str, Any]]],
    *,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    domain_interactions: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    all_texts = sorted(
        {
            text
            for split_rows in rows.values()
            for row in split_rows
            for text in row["texts"]["full"]
        }
    )
    embeddings = PROBE._e5_embeddings(
        all_texts,
        model_name=model_name,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )
    semantic = {
        split: PROBE._mean_group_embeddings(
            rows[split], embeddings, view="full"
        )
        for split in ("train", "val", "test")
    }
    structured_vocab = PROBE._structured_vocab(rows["train"])
    structured = {
        split: PROBE._structured_matrix(rows[split], structured_vocab)
        for split in ("train", "val", "test")
    }
    metadata: dict[str, Any] = {"base": structured_vocab}
    if domain_interactions:
        interaction_vocab = _domain_interaction_vocab(rows["train"])
        interactions = {
            split: _domain_interaction_matrix(rows[split], interaction_vocab)
            for split in ("train", "val", "test")
        }
        structured = {
            split: PROBE._concatenate_normalized(
                structured[split], interactions[split]
            )
            for split in ("train", "val", "test")
        }
        metadata["domain_interactions"] = interaction_vocab
    matrices = {
        split: PROBE._concatenate_normalized(semantic[split], structured[split])
        for split in ("train", "val", "test")
    }
    return matrices, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-name", default="intfloat/e5-base-v2")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--utility-estimator", choices=UTILITY_ESTIMATORS, default=ESTIMATOR
    )
    parser.add_argument("--utility-alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--utility-shrinkage", type=float, default=4.0)
    parser.add_argument("--domain-interactions", action="store_true")
    parser.add_argument("--frozen-recipe", choices=RECIPES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {
        split: PROBE._configuration_rows(
            PROBE._steps(args.data_root / f"{split}_steps.jsonl")
        )
        for split in ("train", "val", "test")
    }
    matrices, structured_vocab = _build_matrices(
        rows,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.embedding_batch_size,
        domain_interactions=args.domain_interactions,
    )

    train_models = _fit_models(
        matrices["train"],
        rows["train"],
        utility_estimator=args.utility_estimator,
        utility_alpha=args.utility_alpha,
        utility_shrinkage=args.utility_shrinkage,
    )
    validation_predictions = _predict_models(
        train_models, matrices["val"], rows["val"]
    )
    validation = _evaluate_recipes(rows["val"], validation_predictions)
    validation_selected, eligible = _select_recipe(validation)
    selected_recipe = args.frozen_recipe or validation_selected

    combined_rows = rows["train"] + rows["val"]
    combined_matrix = np.concatenate((matrices["train"], matrices["val"]), axis=0)
    final_models = _fit_models(
        combined_matrix,
        combined_rows,
        utility_estimator=args.utility_estimator,
        utility_alpha=args.utility_alpha,
        utility_shrinkage=args.utility_shrinkage,
    )
    test_predictions = _predict_models(final_models, matrices["test"], rows["test"])
    test_scores = _recipe_scores(rows["test"], test_predictions, selected_recipe)
    test = PROBE._evaluate(
        rows["test"],
        rank_scores=test_scores,
        predictions=test_predictions["joint"]["prediction"],
    )

    result = {
        "scope": "dual-component E5+structured configuration-value probe",
        "protocol": {
            "representation": "e5_structured/full",
            "estimator": ESTIMATOR,
            "ridge_alpha": RIDGE_ALPHA,
            "utility_estimator": args.utility_estimator,
            "utility_alpha": args.utility_alpha,
            "utility_shrinkage": args.utility_shrinkage,
            "domain_interactions": args.domain_interactions,
            "changed_factor": "separate attack and utility targets plus utility guard",
            "validation_constraint": (
                "selected recipe target-BUP must be no lower than joint-control target-BUP"
            ),
            "validation_objective": (
                "maximize selected target ASR+BUP, then target BUP, then task Spearman"
            ),
            "frozen_recipe": args.frozen_recipe,
            "test_retuning": False,
        },
        "counts": {
            split: {
                "configurations": len(rows[split]),
                "tasks": len({row["task_key"] for row in rows[split]}),
            }
            for split in rows
        },
        "structured_vocabulary": structured_vocab,
        "validation": validation,
        "validation_utility_eligible": eligible,
        "validation_selected_recipe": validation_selected,
        "applied_test_recipe": selected_recipe,
        "test": test,
        "test_candidate_scores": [
            {
                "group_id": str(row["group_id"]),
                "task_key": str(row["task_key"]),
                "target_asr": float(row["target_asr"]),
                "target_bup": float(row["target_bup"]),
                "target": float(row["target"]),
                "observed_asr": float(row["observed_asr"]),
                "observed_bup": float(row["observed_bup"]),
                "observed": float(row["observed"]),
                "rank_score": float(test_scores[index]),
                "joint_prediction": float(test_predictions["joint"]["prediction"][index]),
                "attack_prediction": float(test_predictions["attack"]["prediction"][index]),
                "utility_prediction": float(test_predictions["utility"]["prediction"][index]),
            }
            for index, row in enumerate(rows["test"])
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
