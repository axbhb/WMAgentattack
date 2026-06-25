from pathlib import Path

from wmagentattack.normalize_agentdojo import normalize_trace


def test_normalize_existing_agentdojo_trace():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "runs"
        / "qwen2.5-7b-instruct-transformers-4bit-compact12000-robust"
        / "workspace"
        / "user_task_1"
        / "none"
        / "none.json"
    )
    if not path.exists():
        return
    trajectory = normalize_trace(path)
    assert trajectory.dataset == "agentdojo-v1.2.2"
    assert trajectory.final_task_success is True
    assert len(trajectory.steps) >= 1
    assert trajectory.steps[0].selected_tool == "get_day_calendar_events"
    assert trajectory.steps[0].selected_skill == "calendar_read"
    assert trajectory.steps[0].selected_skill in trajectory.steps[0].candidate_skills
    assert trajectory.final_skill_path == [
        step.selected_skill for step in trajectory.steps
    ]
    assert trajectory.final_skill_path[-1] == "finish"


def test_tool_error_is_visible_in_next_observation():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "runs"
        / "qwen2.5-7b-instruct-transformers-4bit-compact12000-robust"
        / "workspace"
        / "user_task_0"
        / "none"
        / "none.json"
    )
    if not path.exists():
        return
    trajectory = normalize_trace(path)
    assert "ERROR:" in trajectory.steps[1].current_observation


def test_attack_observation_precedes_decision_and_finish_is_recorded():
    root = Path(__file__).resolve().parents[1]
    candidates = list(
        (
            root / "runs" / "agentdojo_attacks"
        ).rglob("user_task_1/direct/injection_task_0.json")
    )
    if not candidates:
        return
    trajectory = normalize_trace(candidates[0])
    assert trajectory.final_skill_path[-1] == "finish"
    assert trajectory.steps[0].untrusted_content is None
    assert trajectory.steps[-1].untrusted_content is not None
