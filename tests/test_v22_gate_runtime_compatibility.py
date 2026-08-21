from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = spec_from_file_location(name, ROOT / "scripts" / filename)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_task_surface_ignores_interleaved_outcome_rows_without_horizon():
    gate = load_script("gate_long_v22", "252_gate_long_horizon_controls_v22.py")
    rows = [
        {"arm": "candidate", "kind": "outcome", "training_seed": 7, "task_name": "suite|task"},
        {
            "arm": "candidate",
            "kind": "rollout",
            "horizon": 2,
            "training_seed": 7,
            "task_name": "suite|task",
            "action_nll": 1.25,
            "action_correct": 1.0,
            "legal_prediction": 1.0,
        },
    ]
    surface = gate.task_surface(rows, "candidate", {2})
    assert surface[(7, 2, "suite|task")]["nll"] == 1.25


def test_clause_counts_accepts_both_frozen_gate_schemas():
    finalizer = load_script("finalize_v22", "254_finalize_parallel_world_model_gates_v22.py")
    assert finalizer.clause_counts({"passed": 8, "total": 10}) == (8, 10)
    assert finalizer.clause_counts({"passed_clauses": 7, "total_clauses": 12}) == (7, 12)
