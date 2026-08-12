from __future__ import annotations

from collections import Counter
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite

from wmagentattack import custom_agentdojo_confirmation_v3 as panel
from wmagentattack import custom_agentdojo_panel_v1 as panel_v1
from wmagentattack import custom_agentdojo_panel_v2 as panel_v2
from wmagentattack.factorized_evaluator_v2 import (
    FactorLabel,
    evaluate_trace,
    load_alias_registry,
)


ROOT = Path(__file__).resolve().parents[1]
ALIASES = load_alias_registry(ROOT / "configs" / "0728_factorized_evaluator_v2_aliases.json")


def _ground_truth_trace(spec):
    return {
        "messages": [
            {
                "role": "tool",
                "tool_call": {"function": call.function, "args": call.args},
                "error": None,
            }
            for call in spec.required_calls
        ]
        + [
            {
                "role": "assistant",
                "content": [{"type": "text", "content": spec.ground_truth_output}],
                "tool_calls": [],
            }
        ]
    }


def test_v3_is_balanced_and_historically_disjoint() -> None:
    assert len(panel.TASK_SPECS) == 12
    assert Counter(spec.suite for spec in panel.TASK_SPECS) == {
        suite: 3 for suite in panel.SUITES
    }
    assert Counter(
        spec.contract.task_difficulty.value for spec in panel.TASK_SPECS
    ) == {"L1": 4, "L2": 4, "L3": 4}
    historical = (*panel_v1.TASK_SPECS, *panel_v2.TASK_SPECS)
    assert not (
        {spec.spec_id for spec in panel.TASK_SPECS}
        & {spec.spec_id for spec in historical}
    )
    assert not (
        {spec.prompt for spec in panel.TASK_SPECS}
        & {spec.prompt for spec in historical}
    )
    assert not (
        {spec.template_family for spec in panel.TASK_SPECS}
        & {spec.template_family for spec in historical}
    )


def test_v3_registry_and_manifest_are_outcome_blind() -> None:
    registry = panel.build_contract_registry()
    assert registry["frozen_before_first_victim_outcome"] is True
    assert len(registry["contracts"]) == 12
    assert all(contract["outcome_labels_present"] is False for contract in registry["contracts"])
    forbidden = {"utility", "success", "outcome", "prediction"}
    assert all(not (forbidden & set(row)) for row in panel.manifest_rows())


def test_v3_ground_truth_calls_and_factorized_contracts_pass() -> None:
    for spec in panel.TASK_SPECS:
        suite = get_suite(panel.BENCHMARK_VERSION, spec.suite)
        evaluation = evaluate_trace(
            trace=_ground_truth_trace(spec),
            environment=suite.load_and_inject_default_environment({}),
            tools=suite.tools,
            mutating_tools=panel.MUTATING_TOOLS,
            contract=spec.contract,
            aliases=ALIASES,
        )
        assert evaluation.labels.overall == FactorLabel.PASS, spec.spec_id
        assert all(call.executed_successfully for call in evaluation.executed_calls), spec.spec_id


def test_v3_conditioned_mutations_require_evidence_before_mutation() -> None:
    checked = 0
    for spec in panel.TASK_SPECS:
        if not any(
            route.must_precede_first_mutation
            for obligation in spec.contract.evidence_obligations
            for route in obligation.routes
        ):
            continue
        mutations = [call for call in spec.required_calls if call.function in panel.MUTATING_TOOLS]
        observations = [call for call in spec.required_calls if call.function not in panel.MUTATING_TOOLS]
        suite = get_suite(panel.BENCHMARK_VERSION, spec.suite)
        trace = {
            "messages": [
                {
                    "role": "tool",
                    "tool_call": {"function": call.function, "args": call.args},
                    "error": None,
                }
                for call in mutations + observations
            ]
            + _ground_truth_trace(spec)["messages"][-1:]
        }
        evaluation = evaluate_trace(
            trace=trace,
            environment=suite.load_and_inject_default_environment({}),
            tools=suite.tools,
            mutating_tools=panel.MUTATING_TOOLS,
            contract=spec.contract,
            aliases=ALIASES,
        )
        assert evaluation.labels.evidence == FactorLabel.FAIL, spec.spec_id
        checked += 1
    assert checked == 4
