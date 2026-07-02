"""Validated standardized records for skill-trajectory learning."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    dataset: str
    domain: str
    task_id: str
    victim_model: str
    agent_architecture: str
    defense: str
    step_id: int = Field(ge=0)
    trusted_instruction: str
    user_goal: str
    agent_history: str
    current_observation: str
    untrusted_content: str | None = None
    candidate_tools: list[str] = Field(default_factory=list)
    candidate_skills: list[str]
    candidate_skill_descriptions: dict[str, str]
    previous_skills: list[str]
    selected_tool: str | None = None
    selected_skill: str
    skill_arguments: dict[str, Any]
    skill_output: str
    tool_error: str | None = None
    attack_action: str | None = None
    attack_location: str | None = None
    target_skill: str | None = None
    task_success: bool
    attack_success: bool
    target_skill_success: bool
    policy_violation: bool
    risk_level: int = Field(ge=0, le=3)
    base_task_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    preservation_trainable: bool = True

    @model_validator(mode="after")
    def selected_skill_is_candidate(self):
        if self.selected_skill not in self.candidate_skills:
            raise ValueError("selected_skill must be present in candidate_skills")
        missing = set(self.candidate_skills) - set(
            self.candidate_skill_descriptions
        )
        if missing:
            raise ValueError(f"candidate descriptions missing for: {sorted(missing)}")
        return self


class TrajectoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    dataset: str
    domain: str
    task_id: str
    victim_model: str
    steps: list[StepRecord]
    final_task_success: bool
    final_attack_success: bool
    final_policy_violation: bool
    final_skill_path: list[str]

    @model_validator(mode="after")
    def path_matches_steps(self):
        expected = [step.selected_skill for step in self.steps]
        if self.final_skill_path != expected:
            raise ValueError("final_skill_path must match step selected skills")
        return self
