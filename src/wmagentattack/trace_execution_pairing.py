"""Causal pairing of clean assistant proposals with executed tool messages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .clean_state_instrumentation import canonical_call_signature
from .decision_state import canonical_json_value


class ToolCallProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_message_index: int = Field(ge=0)
    proposal_index: int = Field(ge=0)
    function: str
    arguments: dict[str, Any]
    call_signature: str


class ExecutedToolCallPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal: ToolCallProposal
    tool_message_index: int = Field(ge=0)
    logged_error: bool


class CleanTraceExecutionPairing(BaseModel):
    """Pairing audit with no assistant text, tool output, or outcome labels."""

    model_config = ConfigDict(extra="forbid")

    proposal_count: int = Field(ge=0)
    tool_message_count: int = Field(ge=0)
    executed_pairs: tuple[ExecutedToolCallPair, ...]
    terminal_unexecuted_proposals: tuple[ToolCallProposal, ...]
    midtrajectory_unexecuted_proposals: tuple[ToolCallProposal, ...]
    orphan_tool_message_indices: tuple[int, ...]
    signature_mismatch_tool_message_indices: tuple[int, ...]
    assistant_multi_call_message_count: int = Field(ge=0)
    outcome_labels_present: bool = False

    @property
    def executed_alignment_ok(self) -> bool:
        """Terminal proposals are allowed; all other pairing defects fail."""

        return bool(
            not self.midtrajectory_unexecuted_proposals
            and not self.orphan_tool_message_indices
            and not self.signature_mismatch_tool_message_indices
            and len(self.executed_pairs) == self.tool_message_count
        )


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        value = block.get("content", block.get("text", ""))
        if value is not None:
            parts.append(str(value))
    return "\n".join(parts)


def _proposal(
    call: dict[str, Any], *, message_index: int, proposal_index: int
) -> ToolCallProposal:
    arguments = canonical_json_value(call.get("args") or {})
    if not isinstance(arguments, dict):
        raise TypeError("tool call arguments must be a mapping")
    function = str(call["function"])
    return ToolCallProposal(
        assistant_message_index=message_index,
        proposal_index=proposal_index,
        function=function,
        arguments=arguments,
        call_signature=canonical_call_signature(function, arguments),
    )


def pair_executed_clean_tool_calls(
    messages: Sequence[dict[str, Any]],
) -> tuple[CleanTraceExecutionPairing, str]:
    """Pair tool messages only with causally pending assistant proposals.

    A proposal left pending at end-of-trace is explicitly classified as a
    terminal unexecuted proposal and must not create a simulator transition.
    Pending proposals displaced by another assistant message are mid-trajectory
    defects. Tool messages without an exact pending signature are never guessed.
    """

    pending: list[ToolCallProposal] = []
    executed = []
    terminal = []
    midtrajectory = []
    orphan_tool_indices = []
    signature_mismatch_indices = []
    proposal_count = 0
    tool_message_count = 0
    multi_call_messages = 0
    final_output = ""

    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            if pending:
                midtrajectory.extend(pending)
                pending = []
            calls = list(message.get("tool_calls") or [])
            if len(calls) > 1:
                multi_call_messages += 1
            for proposal_index, call in enumerate(calls):
                pending.append(
                    _proposal(
                        call,
                        message_index=message_index,
                        proposal_index=proposal_index,
                    )
                )
                proposal_count += 1
            final_output = _message_text(message)
            continue

        if role != "tool":
            continue
        tool_message_count += 1
        raw_call = message.get("tool_call")
        if not isinstance(raw_call, dict) or not pending:
            orphan_tool_indices.append(message_index)
            continue
        tool_proposal = _proposal(
            raw_call, message_index=message_index, proposal_index=0
        )
        expected = pending[0]
        if tool_proposal.call_signature != expected.call_signature:
            signature_mismatch_indices.append(message_index)
            continue
        pending.pop(0)
        executed.append(
            ExecutedToolCallPair(
                proposal=expected,
                tool_message_index=message_index,
                logged_error=bool(message.get("error")),
            )
        )

    terminal.extend(pending)
    pairing = CleanTraceExecutionPairing(
        proposal_count=proposal_count,
        tool_message_count=tool_message_count,
        executed_pairs=tuple(executed),
        terminal_unexecuted_proposals=tuple(terminal),
        midtrajectory_unexecuted_proposals=tuple(midtrajectory),
        orphan_tool_message_indices=tuple(orphan_tool_indices),
        signature_mismatch_tool_message_indices=tuple(
            signature_mismatch_indices
        ),
        assistant_multi_call_message_count=multi_call_messages,
    )
    return pairing, final_output
