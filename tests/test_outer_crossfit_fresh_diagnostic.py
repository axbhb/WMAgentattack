import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "58_diagnose_outer_crossfit_on_fresh_outcomes.py"
SPEC = importlib.util.spec_from_file_location("outer_crossfit_fresh", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _candidate(task, injection, clean, text, selected):
    def method(attack):
        return {
            "attack_rank": attack,
            "utility_rank": attack,
            "attack_probability": attack,
            "utility_probability": attack,
        }

    return {
        "suite": "banking",
        "user_task_id": task,
        "injection_task_id": injection,
        "all_methods": {
            "clean_raw": method(clean),
            "text_pointwise": method(text),
            "text_borda_alpha_0p25": method(selected),
        },
    }


def test_cohort_evaluation_uses_validation_frozen_method():
    candidates = [
        _candidate("task0", "inj0", 0.1, 0.2, 0.3),
        _candidate("task0", "inj1", 0.9, 0.8, 0.7),
        _candidate("task1", "inj0", 0.9, 0.8, 0.7),
        _candidate("task1", "inj1", 0.1, 0.2, 0.3),
    ]
    fresh = [
        {
            "suite": row["suite"],
            "user_task_id": row["user_task_id"],
            "injection_task_id": row["injection_task_id"],
            "observed_attack_probability": float(index % 2),
            "observed_utility_probability": float(index % 2),
        }
        for index, row in enumerate(candidates)
    ]
    result = MODULE._evaluate_cohort(
        fresh,
        {MODULE._key(row): row for row in candidates},
        selected_method="text_borda_alpha_0p25",
        text_method="text_pointwise",
        bootstrap_samples=20,
        bootstrap_seed=7,
    )
    assert result["pair_count"] == 4
    assert result["task_count"] == 2
    assert "text_borda_alpha_0p25__minus__text_pointwise" in result[
        "frozen_comparisons"
    ]
    assert len(result["posthoc_all_method_order"]) == 3
