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
    preservation_weight: float = Field(default=1.0, ge=0.0)
    utility_probability_target: float | None = Field(default=None, ge=0.0, le=1.0)
    preservation_probability_target: float | None = Field(default=None, ge=0.0, le=1.0)
    attack_probability_target: float | None = Field(default=None, ge=0.0, le=1.0)
    joint_success_probability_target: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    probability_label_alpha: float | None = Field(default=None, gt=0.0)
    probability_label_beta: float | None = Field(default=None, gt=0.0)
    probability_label_variance: float | None = Field(default=None, ge=0.0)
    probability_label_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    attack_probability_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    joint_success_probability_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0
    )
    # Joint repeated-run evidence.  The key convention is deliberately
    # explicit: ``attack1_utility0`` means that the attacker goal succeeded
    # while the trusted user task failed.  These counts let newer models fit
    # a multinomial/Dirichlet-multinomial likelihood instead of treating a
    # posterior mean as a noise-free regression target.
    joint_outcome_counts: dict[str, int] | None = None
    joint_outcome_dirichlet_alpha: dict[str, float] | None = None
    joint_outcome_probability_target: dict[str, float] | None = None
    joint_outcome_trials: int | None = Field(default=None, ge=1)
    # Factorized utility target: a task-level clean prior plus the attacked
    # configuration's residual on the log-odds scale.  The legacy ratio
    # target above remains available for backward compatibility.
    clean_utility_logit_prior: float | None = None
    attack_utility_logit_residual_target: float | None = None
    probability_label_source: str | None = None
    multiseed_group_id: str | None = None
    multiseed_trials: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def selected_skill_is_candidate(self):
        if self.selected_skill not in self.candidate_skills:
            raise ValueError("selected_skill must be present in candidate_skills")
        missing = set(self.candidate_skills) - set(
            self.candidate_skill_descriptions
        )
        if missing:
            raise ValueError(f"candidate descriptions missing for: {sorted(missing)}")
        expected_joint_keys = {
            "attack0_utility0",
            "attack0_utility1",
            "attack1_utility0",
            "attack1_utility1",
        }
        if self.joint_outcome_counts is not None:
            if set(self.joint_outcome_counts) != expected_joint_keys:
                raise ValueError(
                    "joint_outcome_counts must use the four explicit "
                    "attack{0,1}_utility{0,1} keys"
                )
            if any(value < 0 for value in self.joint_outcome_counts.values()):
                raise ValueError("joint outcome counts must be non-negative")
            count_sum = sum(self.joint_outcome_counts.values())
            if self.joint_outcome_trials != count_sum:
                raise ValueError(
                    "joint_outcome_trials must equal the sum of joint counts"
                )
        elif self.joint_outcome_trials is not None:
            raise ValueError("joint_outcome_trials requires joint_outcome_counts")
        for name, values in (
            ("joint_outcome_dirichlet_alpha", self.joint_outcome_dirichlet_alpha),
            ("joint_outcome_probability_target", self.joint_outcome_probability_target),
        ):
            if values is None:
                continue
            if set(values) != expected_joint_keys:
                raise ValueError(f"{name} must use the four joint outcome keys")
            if name.endswith("alpha") and any(value <= 0.0 for value in values.values()):
                raise ValueError("Dirichlet alpha values must be positive")
            if name.endswith("target"):
                if any(not 0.0 <= value <= 1.0 for value in values.values()):
                    raise ValueError("joint probability targets must lie in [0, 1]")
                if abs(sum(values.values()) - 1.0) > 1e-6:
                    raise ValueError("joint probability targets must sum to one")
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
