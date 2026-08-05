from __future__ import annotations

from pydantic import BaseModel

from wmagentattack.counterfactual_evidence import (
    ToolBindingSpec,
    bind_query,
    build_query_universe,
)
from wmagentattack.semantic_state_v3 import semantic_state_v3_payload


class _NoArgs(BaseModel):
    pass


class _OneArg(BaseModel):
    item: str


def _spec(candidate_id: str, model, *, mutating: bool = False):
    suite, name = candidate_id.split("::", 1)
    return ToolBindingSpec(
        candidate_id=candidate_id,
        tool_name=name,
        suite=suite,
        required_fields=tuple(model.model_json_schema().get("required", ())),
        mutating=mutating,
        validator=lambda args: model.model_validate(args).model_dump(mode="json"),
    )


def _features(prefix_index: int, *, action: str):
    return {
        "trusted_goal": "Find the blue record.",
        "track": "deterministic_greedy",
        "prefix_index": prefix_index,
        "legal_tools": ["demo::read", "demo::write", "STOP"],
        "last_action": {"function": action, "arguments": {}},
        "last_observation": "",
        "execution_receipt": {
            "status": "start" if prefix_index == 0 else "success",
            "error_type": None,
        },
        "causal_state_summary": {},
        "ledger_v2": {"records": [], "conflicts": [], "execution_receipts": []},
    }


def _datasets(*, terminal_only: bool):
    prefixes = [
        {
            "prefix_index": 0,
            "features": _features(0, action="<START>"),
            "targets": {"next_action": "STOP" if terminal_only else "demo::read"},
        }
    ]
    if not terminal_only:
        prefixes.append(
            {
                "prefix_index": 1,
                "features": _features(1, action="read"),
                "targets": {"next_action": "STOP"},
            }
        )
    episode = {
        "episode_id": "greedy::task-a",
        "task_id": "task-a",
        "suite": "demo",
        "split": "training",
        "track": "deterministic_greedy",
        "run_seed": 1,
        "task_difficulty": "L1",
        "task_archetype": "lookup",
        "prefixes": prefixes,
    }
    semantic = {
        **episode,
        "prefixes": [
            {
                "prefix_index": row["prefix_index"],
                "features": {
                    "semantic_state_v3": semantic_state_v3_payload(row["features"])
                },
                "targets": row["targets"],
            }
            for row in prefixes
        ],
    }
    return {"episodes": [episode]}, {"episodes": [semantic]}


def test_terminal_prefix_contributes_executable_queries_but_never_stop():
    raw, semantic = _datasets(terminal_only=True)
    specs = {
        "demo::read": _spec("demo::read", _NoArgs),
        "demo::write": _spec("demo::write", _OneArg, mutating=True),
    }
    universe = build_query_universe(
        raw, semantic, selected_task_ids=["task-a"], tool_specs=specs
    )

    assert universe["audit"]["terminal_decisions"] == 1
    assert universe["audit"]["states"] == 1
    assert {row["candidate_id"] for row in universe["queries"]} == {
        "demo::read",
        "demo::write",
    }
    assert all(row["candidate_id"] != "STOP" for row in universe["queries"])


def test_nonterminal_actual_action_is_observed_and_excluded_from_queries():
    raw, semantic = _datasets(terminal_only=False)
    specs = {
        "demo::read": _spec("demo::read", _NoArgs),
        "demo::write": _spec("demo::write", _OneArg, mutating=True),
    }
    universe = build_query_universe(
        raw, semantic, selected_task_ids=["task-a"], tool_specs=specs
    )

    first = [
        row for row in universe["queries"] if row["metadata"]["prefix_index"] == 0
    ]
    assert [row["candidate_id"] for row in first] == ["demo::write"]
    assert universe["audit"]["observed_transitions"] == 1
    assert universe["audit"]["terminal_decisions"] == 1


def test_binding_precedence_uses_empty_then_cross_task_then_same_task():
    empty = _spec("demo::read", _NoArgs)
    required = _spec("demo::write", _OneArg, mutating=True)
    base = {
        "query_ref": "q",
        "state_ref": "state",
        "candidate_id": "demo::write",
        "metadata": {"task_id": "target"},
    }
    donors = {
        "demo::write": [
            {
                "task_id": "target",
                "episode_id": "same",
                "track": "deterministic_greedy",
                "transition_index": 0,
                "arguments": {"item": "same"},
            },
            {
                "task_id": "other",
                "episode_id": "cross",
                "track": "stochastic_policy",
                "transition_index": 1,
                "arguments": {"item": "cross"},
            },
        ]
    }
    cross = bind_query(
        base,
        tool_specs={"demo::write": required},
        donors=donors,
        seed="fixed",
    )
    assert cross is not None
    assert cross.source == "CROSS_TASK_OBSERVED_DONOR"
    assert cross.arguments == {"item": "cross"}
    assert cross.donor_outcomes_read is False

    same = bind_query(
        base,
        tool_specs={"demo::write": required},
        donors={"demo::write": donors["demo::write"][:1]},
        seed="fixed",
    )
    assert same is not None and same.source == "SAME_TASK_CLEAN_DONOR"

    empty_query = {**base, "candidate_id": "demo::read"}
    empty_binding = bind_query(
        empty_query,
        tool_specs={"demo::read": empty},
        donors={},
        seed="fixed",
    )
    assert empty_binding is not None
    assert empty_binding.source == "SCHEMA_EMPTY"


def test_required_query_without_any_clean_donor_remains_unbound():
    required = _spec("demo::write", _OneArg, mutating=True)
    query = {
        "query_ref": "q",
        "candidate_id": "demo::write",
        "metadata": {"task_id": "target"},
    }
    assert (
        bind_query(
            query,
            tool_specs={"demo::write": required},
            donors={},
            seed="fixed",
        )
        is None
    )
