import numpy as np

from wmagentattack.io_hmm_world_model import (
    IOHMMConfig,
    HierarchicalDiscreteIOHMM,
    SmoothedContextMarkovBaseline,
    evaluate_markov_baseline,
    evaluate_next_events,
)


def _synthetic_sequences():
    observations = []
    inputs = []
    for _ in range(20):
        observations.extend([["read", "finish"], ["write", "finish"]])
        inputs.extend([["clean", "clean"], ["attacked", "attacked"]])
    return observations, inputs


def test_io_hmm_learns_context_conditioned_victim_events(tmp_path):
    observations, inputs = _synthetic_sequences()
    model = HierarchicalDiscreteIOHMM(
        IOHMMConfig(
            num_states=2,
            max_iterations=40,
            restarts=2,
            random_seed=11,
            backoff_strength=0.5,
        )
    ).fit(observations, inputs)
    clean = model.predict_next_distribution([], [], "clean")
    attacked = model.predict_next_distribution([], [], "attacked")
    assert clean["read"] > attacked["read"]
    assert attacked["write"] > clean["write"]
    assert np.isfinite(model.training_log_likelihood[-1])
    metrics = evaluate_next_events(model, observations, inputs)
    assert metrics["next_event_accuracy"] >= 0.99

    path = tmp_path / "model.json"
    model.save(path)
    restored = HierarchicalDiscreteIOHMM.load(path)
    assert restored.predict_next_distribution([], [], "clean") == clean


def test_io_hmm_unseen_input_uses_pooled_backoff():
    observations, inputs = _synthetic_sequences()
    model = HierarchicalDiscreteIOHMM(
        IOHMMConfig(num_states=2, max_iterations=20, restarts=1)
    ).fit(observations, inputs)
    distribution = model.predict_next_distribution([], [], "unseen-family")
    assert abs(sum(distribution.values()) - 1.0) < 1e-9
    assert all(np.isfinite(value) for value in distribution.values())


def test_markov_counterbaseline_is_finite_on_unseen_context():
    observations, inputs = _synthetic_sequences()
    baseline = SmoothedContextMarkovBaseline().fit(observations, inputs)
    metrics = evaluate_markov_baseline(
        baseline,
        [["read", "finish"]],
        [["new-context", "new-context"]],
    )
    assert np.isfinite(metrics["mean_event_nll"])
