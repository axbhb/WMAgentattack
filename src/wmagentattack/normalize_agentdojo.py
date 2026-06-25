"""Convert AgentDojo raw traces to standardized skill trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import get_text_content_as_str

from wmagentattack.schema import StepRecord, TrajectoryRecord
from wmagentattack.skill_registry import generalized_name, infer_risk_level


def _text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not content:
        return ""
    return get_text_content_as_str(content)


def _observation_text(message: dict[str, Any]) -> str:
    text = _text(message)
    error = message.get("error")
    if error:
        return f"{text}\nERROR: {error}".strip()
    return text


def _history(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{message.get('role', 'unknown').upper()}: {_observation_text(message)}"
        for message in messages
    )


def _tool_catalog(version: str, suite_name: str):
    suite = get_suite(version, suite_name)
    tool_names = [tool.name for tool in suite.tools]
    skill_descriptions: dict[str, list[str]] = {}
    for tool in suite.tools:
        skill = generalized_name(tool.name)
        skill_descriptions.setdefault(skill, []).append(
            f"{tool.name}: {tool.description}"
        )
    return (
        tool_names,
        list(skill_descriptions),
        {
            skill: " | ".join(descriptions)
            for skill, descriptions in skill_descriptions.items()
        },
    )


def normalize_trace(path: str | Path) -> TrajectoryRecord:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("benchmark_version") or "v1.2.2"
    suite_name = raw["suite_name"]
    candidate_tools, candidate_skills, candidate_descriptions = _tool_catalog(
        version, suite_name
    )
    candidate_skills = [*candidate_skills, "finish"]
    candidate_descriptions = {
        **candidate_descriptions,
        "finish": "Return a final response without calling another tool.",
    }
    messages = raw["messages"]
    system = next((m for m in messages if m["role"] == "system"), {})
    user = next((m for m in messages if m["role"] == "user"), {})
    attack_type = raw.get("attack_type")
    attack_action = (
        None
        if attack_type in (None, "none")
        else f"AGENTDOJO_ATTACK_{attack_type}"
    )
    # AgentDojo injection-task ``security`` functions return True when the
    # attacker goal is achieved. Clean runs also store True as a placeholder,
    # so the injection-task guard is essential.
    attack_success = bool(
        raw.get("injection_task_id") is not None and raw.get("security", False)
    )
    trajectory_id = hashlib.sha1(
        f"{raw['pipeline_name']}|{suite_name}|{raw['user_task_id']}|"
        f"{raw.get('injection_task_id')}|{attack_type}".encode()
    ).hexdigest()[:16]
    target_tools: list[str] = []
    injection_task_id = raw.get("injection_task_id")
    if injection_task_id is not None:
        suite = get_suite(version, suite_name)
        injection_task = suite.get_injection_task_by_id(injection_task_id)
        environment = suite.load_and_inject_default_environment({})
        try:
            target_tools = [
                call.function for call in injection_task.ground_truth(environment)
            ]
        except Exception:
            target_tools = []
    target_skill = (
        generalized_name(target_tools[-1]) if target_tools else None
    )

    steps: list[StepRecord] = []
    previous_skills: list[str] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        observation = (
            _observation_text(messages[index - 1])
            if index > 0
            else _text(user)
        )
        visible_untrusted = (
            observation
            if attack_action
            and index > 0
            and messages[index - 1].get("role") == "tool"
            else None
        )

        if not calls:
            finish_step = StepRecord(
                trajectory_id=trajectory_id,
                dataset=f"agentdojo-{version}",
                domain=suite_name,
                task_id=raw["user_task_id"],
                victim_model=raw["pipeline_name"],
                agent_architecture="tool_calling_llm",
                defense="none",
                step_id=len(steps),
                trusted_instruction=_text(system),
                user_goal=_text(user),
                agent_history=_history(messages[:index]),
                current_observation=observation,
                untrusted_content=visible_untrusted,
                candidate_tools=candidate_tools,
                candidate_skills=candidate_skills,
                candidate_skill_descriptions=candidate_descriptions,
                previous_skills=list(previous_skills),
                selected_tool=None,
                selected_skill="finish",
                skill_arguments={},
                skill_output=_text(message),
                tool_error=None,
                attack_action=attack_action,
                attack_location=(
                    ",".join(raw.get("injections", {}).keys())
                    if attack_action
                    else None
                ),
                target_skill=target_skill,
                task_success=bool(raw.get("utility")),
                attack_success=attack_success,
                target_skill_success=False,
                policy_violation=attack_success,
                risk_level=0,
            )
            steps.append(finish_step)
            previous_skills.append("finish")
            continue

        for call in calls:
            selected_tool = call["function"]
            is_valid_tool = selected_tool in candidate_tools
            selected_skill = (
                generalized_name(selected_tool)
                if is_valid_tool
                else "invalid_tool_call"
            )
            step_candidate_skills = list(candidate_skills)
            step_candidate_descriptions = dict(candidate_descriptions)
            if not is_valid_tool:
                step_candidate_skills.append("invalid_tool_call")
                step_candidate_descriptions["invalid_tool_call"] = (
                    "Malformed or unavailable tool requested by the agent."
                )
            tool_output = ""
            tool_error = None
            for later in messages[index + 1 :]:
                if later.get("role") == "tool" and later.get(
                    "tool_call", {}
                ).get("function") == selected_tool:
                    tool_output = _text(later)
                    tool_error = later.get("error")
                    break
                if later.get("role") == "assistant":
                    break
            step = StepRecord(
                trajectory_id=trajectory_id,
                dataset=f"agentdojo-{version}",
                domain=suite_name,
                task_id=raw["user_task_id"],
                victim_model=raw["pipeline_name"],
                agent_architecture="tool_calling_llm",
                defense="none",
                step_id=len(steps),
                trusted_instruction=_text(system),
                user_goal=_text(user),
                agent_history=_history(messages[:index]),
                current_observation=observation,
                untrusted_content=visible_untrusted,
                candidate_tools=candidate_tools,
                candidate_skills=step_candidate_skills,
                candidate_skill_descriptions=step_candidate_descriptions,
                previous_skills=list(previous_skills),
                selected_tool=selected_tool,
                selected_skill=selected_skill,
                skill_arguments=call.get("args") or {},
                skill_output=tool_output,
                tool_error=tool_error,
                attack_action=attack_action,
                attack_location=(
                    ",".join(raw.get("injections", {}).keys())
                    if attack_action
                    else None
                ),
                target_skill=target_skill,
                task_success=bool(raw.get("utility")),
                attack_success=attack_success,
                target_skill_success=(
                    attack_success
                    and target_skill is not None
                    and selected_skill == target_skill
                ),
                policy_violation=attack_success,
                risk_level=(
                    infer_risk_level(selected_tool) if is_valid_tool else 1
                ),
            )
            steps.append(step)
            previous_skills.append(selected_skill)

    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        dataset=f"agentdojo-{version}",
        domain=suite_name,
        task_id=raw["user_task_id"],
        victim_model=raw["pipeline_name"],
        steps=steps,
        final_task_success=bool(raw.get("utility")),
        final_attack_success=attack_success,
        final_policy_violation=attack_success,
        final_skill_path=[step.selected_skill for step in steps],
    )


def normalize_directory(root: str | Path) -> list[TrajectoryRecord]:
    trajectories = []
    for path in sorted(Path(root).rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not {
            "suite_name",
            "pipeline_name",
            "user_task_id",
            "messages",
            "utility",
            "security",
        }.issubset(raw):
            continue
        trajectories.append(normalize_trace(path))
    return trajectories
