"""Generalized skill definitions and risk mapping."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GeneralizedSkill(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    precondition: str = "Tool is available in the current sandbox runtime."
    effect: str
    risk_level: int = Field(ge=0, le=3)


READ_PREFIXES = (
    "get_",
    "search_",
    "list_",
    "find_",
    "read_",
)
GENERATE_PREFIXES = ("draft_", "summarize_", "compose_")
EXTERNAL_PREFIXES = ("send_", "book_", "transfer_", "pay_")
MODIFY_PREFIXES = (
    "create_",
    "update_",
    "delete_",
    "append_",
    "share_",
    "reschedule_",
    "cancel_",
)


def infer_risk_level(tool_name: str) -> int:
    name = tool_name.lower()
    if name.startswith(EXTERNAL_PREFIXES):
        return 3
    if name.startswith(MODIFY_PREFIXES):
        return 2
    if name.startswith(GENERATE_PREFIXES):
        return 1
    if name.startswith(READ_PREFIXES):
        return 0
    return 1


def generalized_name(tool_name: str) -> str:
    name = tool_name.lower()
    resource = next(
        (
            token
            for token in (
                "calendar",
                "email",
                "file",
                "drive",
                "slack",
                "message",
                "transaction",
                "hotel",
                "flight",
                "restaurant",
            )
            if token in name
        ),
        "record",
    )
    risk = infer_risk_level(name)
    operation = {0: "read", 1: "generate", 2: "modify", 3: "external"}[risk]
    return f"{resource}_{operation}"


def skill_from_tool(
    name: str, description: str, input_schema: dict[str, Any]
) -> GeneralizedSkill:
    risk = infer_risk_level(name)
    return GeneralizedSkill(
        name=generalized_name(name),
        description=description,
        input_schema=input_schema,
        effect=f"Executes sandbox tool `{name}` with risk level {risk}.",
        risk_level=risk,
    )

