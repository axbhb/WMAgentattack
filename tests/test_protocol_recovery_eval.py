import json
from pathlib import Path
from wmagentattack.protocol_recovery_eval import make_plan, task_signflip_p, evaluate


def setup():
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / "configs/0831_protocol_recovery_v38_v40_protocol.json").read_text())
    tasks = json.loads((root / "configs/0831_clean_pairing_v37_protocol.json").read_text())["tasks"]
    stage = p["v39"]
    rows = make_plan(tasks, stage["seeds"], stage["arms"])
    for r in rows:
        r.update(status="completed", utility=(r["arm"] != "strict"), blocked_network_attempts=0, parsed_tool_calls=1)
    return tasks, stage, rows


def artifacts(row):
    raw = {"utility": row["utility"], "messages": [{"role": "assistant", "tool_calls": [{}] * row["parsed_tool_calls"]}]}
    return raw, {"first_prompt_ids": [1, 2], "events": [{"completion": "same", "output_tokens": 1}]}


def test_plan_is_balanced_and_seeds_are_shared_by_arm():
    tasks, stage, rows = setup()
    assert len(rows) == 180
    assert len({r["episode_seed"] for r in rows}) == 60
    assert rows[0]["arm"] != rows[3]["arm"]


def test_signflip_uses_tasks_not_duplicate_seed_records():
    assert task_signflip_p([1] * 6 + [0] * 14) == 1/64
    assert task_signflip_p([3] * 6 + [0] * 14) == 1/64
    assert task_signflip_p([0] * 20) == 1


def test_favorable_complete_results_pass_and_simpler_arm_selected():
    tasks, stage, rows = setup()
    gate = evaluate(rows, tasks, stage["seeds"], stage["arms"], stage["gate"], artifacts)
    assert gate["decision"] == "GO_PROTOCOL_RECOVERY"
    assert gate["selected_arm"] == "syntax"


def test_missing_or_wrong_seed_is_invalid():
    tasks, stage, rows = setup()
    assert evaluate(rows[:-1], tasks, stage["seeds"], stage["arms"], stage["gate"], artifacts)["decision"].startswith("INVALID")
    rows[0]["episode_seed"] += 1
    assert evaluate(rows, tasks, stage["seeds"], stage["arms"], stage["gate"], artifacts)["decision"].startswith("INVALID")


def test_first_generation_mismatch_invalidates_pairing():
    tasks, stage, rows = setup()
    def changed(row):
        raw, diag = artifacts(row)
        diag["events"][0]["completion"] = row["arm"]
        return raw, diag
    assert evaluate(rows, tasks, stage["seeds"], stage["arms"], stage["gate"], changed)["decision"].startswith("INVALID")


def test_toolfree_control_regression_is_not_hidden_by_large_total_gain():
    tasks, stage, rows = setup()
    for r in rows:
        if r["task"] == tasks[0]:
            r["utility"] = r["arm"] == "strict"
            r["parsed_tool_calls"] = 0
    gate = evaluate(rows, tasks, stage["seeds"], stage["arms"], stage["gate"], artifacts)
    assert gate["decision"] == "NO_GO_PROTOCOL_RECOVERY"
    assert not gate["comparisons"]["syntax"]["checks"]["preserve_successful_tool_free"]


def test_excess_correction_generations_are_invalid():
    tasks, stage, rows = setup()
    def changed(row):
        raw, diag = artifacts(row)
        if row["arm"] == "syntax_retry":
            diag["events"] *= 3
        return raw, diag
    assert not evaluate(rows, tasks, stage["seeds"], stage["arms"], stage["gate"], changed)["scientific_result"]


def test_independent_confirmation_requires_frozen_stable_task_overlap():
    tasks, stage, rows = setup()
    rows = [r for r in rows if r["arm"] != "syntax_retry"]
    threshold = dict(stage["gate"], minimum_stable_task_overlap_with_v39=10)
    prior = {"metrics": {"syntax": {"stable_tasks": tasks[:9]}}}
    gate = evaluate(rows, tasks, stage["seeds"], ["strict", "syntax"], threshold, artifacts, prior)
    assert gate["decision"] == "NO_GO_PROTOCOL_RECOVERY"
    assert not gate["comparisons"]["syntax"]["checks"]["stable_confirmation_overlap"]
