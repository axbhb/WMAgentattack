from __future__ import annotations

from collections import Counter
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite

from wmagentattack import custom_agentdojo_panel_v1 as panel_v1
from wmagentattack import custom_agentdojo_panel_v2 as panel
from wmagentattack.factorized_evaluator_v2 import FactorLabel, evaluate_trace, load_alias_registry


ROOT = Path(__file__).resolve().parents[1]
ALIASES = load_alias_registry(ROOT / "configs" / "0728_factorized_evaluator_v2_aliases.json")


def _trace(spec, calls=None, output=None):
    selected = spec.required_calls if calls is None else tuple(calls)
    return {
        "messages": [
            {
                "role": "tool",
                "tool_call": {"function": call.function, "args": call.args},
                "error": None,
            }
            for call in selected
        ]
        + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "content": spec.ground_truth_output if output is None else output,
                    }
                ],
                "tool_calls": [],
            }
        ]
    }


def _evaluate(spec, *, calls=None, output=None):
    suite = get_suite(panel.BENCHMARK_VERSION, spec.suite)
    return evaluate_trace(
        trace=_trace(spec, calls=calls, output=output),
        environment=suite.load_and_inject_default_environment({}),
        tools=suite.tools,
        mutating_tools=panel.MUTATING_TOOLS,
        contract=spec.contract,
        aliases=ALIASES,
    )


def test_panel_v2_is_suite_difficulty_split_balanced():
    assert len(panel.TASK_SPECS) == 48
    assert len({spec.spec_id for spec in panel.TASK_SPECS}) == 48
    assert Counter(spec.suite for spec in panel.TASK_SPECS) == {
        "banking": 12,
        "slack": 12,
        "travel": 12,
        "workspace": 12,
    }
    assert Counter(spec.split for spec in panel.TASK_SPECS) == {
        "training": 24,
        "calibration": 12,
        "confirmation": 12,
    }
    cells = Counter(
        (spec.suite, spec.contract.task_difficulty.value, spec.split)
        for spec in panel.TASK_SPECS
    )
    for suite in panel.SUITES:
        for difficulty in ("L1", "L2", "L3"):
            assert cells[(suite, difficulty, "training")] == 2
            assert cells[(suite, difficulty, "calibration")] == 1
            assert cells[(suite, difficulty, "confirmation")] == 1


def test_panel_v2_is_new_and_template_disjoint():
    assert len({spec.template_family for spec in panel.TASK_SPECS}) == 48
    assert not ({spec.spec_id for spec in panel.TASK_SPECS} & {spec.spec_id for spec in panel_v1.TASK_SPECS})
    assert not ({spec.prompt for spec in panel.TASK_SPECS} & {spec.prompt for spec in panel_v1.TASK_SPECS})
    assert len({spec.prompt for spec in panel.TASK_SPECS}) == 48


def test_fresh_registry_contains_no_outcome_labels():
    registry = panel.build_contract_registry()
    assert registry.development_only is False
    assert registry.barred_from_fresh_confirmation is False
    assert registry.frozen_before_first_victim_outcome is True
    assert len(registry.contracts) == 48
    assert all(contract.outcome_labels_present is False for contract in registry.contracts)


def test_all_ground_truth_traces_pass_every_applicable_factor():
    for spec in panel.TASK_SPECS:
        evaluation = _evaluate(spec)
        assert evaluation.labels.overall == FactorLabel.PASS, spec.spec_id
        if spec.contract.state_action_applicable:
            assert evaluation.labels.state_action == FactorLabel.PASS, spec.spec_id
        if spec.contract.evidence_applicable:
            assert evaluation.labels.evidence == FactorLabel.PASS, spec.spec_id
        if spec.contract.report_applicable:
            assert evaluation.labels.report == FactorLabel.PASS, spec.spec_id
        assert all(call.executed_successfully for call in evaluation.executed_calls), spec.spec_id


def test_each_factor_has_a_negative_counterfactual():
    for spec in panel.TASK_SPECS:
        if spec.contract.state_action_applicable:
            assert _evaluate(spec, calls=()).labels.state_action == FactorLabel.FAIL, spec.spec_id
        if spec.contract.evidence_applicable:
            assert _evaluate(spec, calls=()).labels.evidence == FactorLabel.FAIL, spec.spec_id
        if spec.contract.report_applicable:
            assert _evaluate(spec, output="").labels.report == FactorLabel.FAIL, spec.spec_id


def test_condition_first_contracts_reject_evidence_after_mutation():
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
        evaluation = _evaluate(spec, calls=mutations + observations)
        assert evaluation.labels.evidence == FactorLabel.FAIL, spec.spec_id
        checked += 1
    assert checked >= 12


def test_read_only_contracts_reject_an_extra_successful_mutation():
    extras = {
        "banking": panel._call("update_password", password="unexpected"),
        "slack": panel._call("send_direct_message", recipient="Alice", body="unexpected"),
        "travel": panel._call(
            "create_calendar_event",
            title="Unexpected",
            start_time="2025-01-01 10:00",
            end_time="2025-01-01 11:00",
        ),
        "workspace": panel._call("create_file", filename="unexpected.txt", content="unexpected"),
    }
    for suite_name, extra in extras.items():
        spec = next(
            spec
            for spec in panel.TASK_SPECS
            if spec.suite == suite_name and not spec.contract.state_action_applicable
        )
        evaluation = _evaluate(spec, calls=spec.required_calls + (extra,))
        assert evaluation.safety_no_forbidden_side_effects is False
        assert evaluation.labels.overall == FactorLabel.FAIL


def test_manifest_rows_are_label_blind_and_structurally_complete():
    rows = panel.manifest_rows()
    assert len(rows) == 48
    forbidden = {"utility", "success", "outcome", "prediction"}
    for row in rows:
        assert not (forbidden & set(row))
        assert row["legacy_scalar_utility_binding"] is False
        assert len(row["task_spec_sha256"]) == 64
        assert len(row["proof_contract_sha256"]) == 64
        assert set(row["difficulty_features"]) == {
            "required_tool_count",
            "required_source_count",
            "goal_atom_count",
            "candidate_count",
            "has_condition",
            "has_mutation",
            "requires_cross_source_join",
            "requires_uniqueness_proof",
            "required_report_slot_count",
        }
