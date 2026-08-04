from collections import Counter, defaultdict

from agentdojo.task_suite.load_suites import get_suite

from wmagentattack import custom_agentdojo_panel_v1 as panel


def test_custom_panel_is_balanced_and_template_disjoint():
    assert len(panel.TASK_SPECS) == 24
    assert len({spec.spec_id for spec in panel.TASK_SPECS}) == 24
    assert Counter(spec.split for spec in panel.TASK_SPECS) == {
        "training": 8,
        "calibration": 8,
        "confirmation": 8,
    }
    assert Counter(spec.suite for spec in panel.TASK_SPECS) == {
        "banking": 6,
        "slack": 6,
        "travel": 6,
        "workspace": 6,
    }
    families = defaultdict(set)
    for spec in panel.TASK_SPECS:
        families[spec.template_family].add(spec.split)
    assert all(len(splits) == 1 for splits in families.values())


def test_custom_tasks_are_registered_without_stock_id_collision():
    for spec in panel.TASK_SPECS:
        assert spec.task_number >= 1000
        task = get_suite(panel.BENCHMARK_VERSION, spec.suite).get_user_task_by_id(
            spec.task_id
        )
        assert task.PANEL_SPEC_ID == spec.spec_id
        assert task.PROMPT == spec.prompt


def test_task_spec_hashes_are_unique_and_stable_length():
    hashes = [spec.sha256 for spec in panel.TASK_SPECS]
    assert len(set(hashes)) == 24
    assert all(len(value) == 64 for value in hashes)


def test_each_task_requires_at_least_one_observation_or_action():
    assert all(spec.required_calls for spec in panel.TASK_SPECS)
    for spec in panel.TASK_SPECS:
        suite = get_suite(panel.BENCHMARK_VERSION, spec.suite)
        task = suite.get_user_task_by_id(spec.task_id)
        environment = suite.load_and_inject_default_environment({})
        assert (
            task.utility_from_traces(
                task.GROUND_TRUTH_OUTPUT,
                environment,
                environment.model_copy(deep=True),
                [],
            )
            is False
        )
