"""A small-data, hierarchical input-output HMM for victim skill events.

This is the deliberately conservative dynamic baseline for the factorized
world-model redesign.  Exogenous input tokens encode the suite/attacker
configuration, observations encode victim skill events, and latent states
capture compact execution phases.  Context-specific transition and emission
tables are shrunk toward a pooled table so unseen or rare inputs have a stable
backoff distribution.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


UNKNOWN_TOKEN = "<UNK>"
BEGIN_TOKEN = "<BOS>"


@dataclass(frozen=True)
class IOHMMConfig:
    num_states: int = 6
    max_iterations: int = 50
    tolerance: float = 1e-4
    smoothing: float = 1e-3
    backoff_strength: float = 2.0
    restarts: int = 3
    random_seed: int = 7

    def __post_init__(self) -> None:
        if self.num_states < 1:
            raise ValueError("num_states must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if self.smoothing <= 0.0:
            raise ValueError("smoothing must be positive")
        if self.backoff_strength < 0.0:
            raise ValueError("backoff_strength must be non-negative")
        if self.restarts < 1:
            raise ValueError("restarts must be positive")


def _normalize_last_axis(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=-1, keepdims=True)
    return values / np.maximum(totals, np.finfo(np.float64).tiny)


class HierarchicalDiscreteIOHMM:
    """Discrete IO-HMM trained by Baum-Welch with pooled context backoff."""

    def __init__(self, config: IOHMMConfig | None = None) -> None:
        self.config = config or IOHMMConfig()
        self.observation_vocab: dict[str, int] = {}
        self.input_vocab: dict[str, int] = {}
        self.initial_probabilities: np.ndarray | None = None
        self.transition_probabilities: np.ndarray | None = None
        self.emission_probabilities: np.ndarray | None = None
        self.training_log_likelihood: list[float] = []

    @property
    def fitted(self) -> bool:
        return all(
            value is not None
            for value in (
                self.initial_probabilities,
                self.transition_probabilities,
                self.emission_probabilities,
            )
        )

    @staticmethod
    def _build_vocab(sequences: Iterable[Sequence[str]]) -> dict[str, int]:
        tokens = sorted({str(token) for sequence in sequences for token in sequence})
        return {UNKNOWN_TOKEN: 0, **{token: index + 1 for index, token in enumerate(tokens)}}

    def _encode(
        self,
        observation_sequences: Sequence[Sequence[str]],
        input_sequences: Sequence[Sequence[str]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if len(observation_sequences) != len(input_sequences):
            raise ValueError("observation and input sequence counts differ")
        encoded: list[tuple[np.ndarray, np.ndarray]] = []
        for index, (observations, inputs) in enumerate(
            zip(observation_sequences, input_sequences)
        ):
            if not observations:
                raise ValueError(f"sequence {index} is empty")
            if len(observations) != len(inputs):
                raise ValueError(f"sequence {index} input/observation lengths differ")
            obs_ids = np.asarray(
                [self.observation_vocab.get(str(item), 0) for item in observations],
                dtype=np.int64,
            )
            input_ids = np.asarray(
                [self.input_vocab.get(str(item), 0) for item in inputs],
                dtype=np.int64,
            )
            encoded.append((obs_ids, input_ids))
        return encoded

    def _random_parameters(self, rng: np.random.Generator) -> None:
        states = self.config.num_states
        input_count = len(self.input_vocab)
        observation_count = len(self.observation_vocab)
        self.initial_probabilities = rng.dirichlet(np.ones(states))
        self.transition_probabilities = rng.dirichlet(
            np.ones(states), size=(input_count, states)
        )
        self.emission_probabilities = rng.dirichlet(
            np.ones(observation_count), size=(input_count, states)
        )

    def _parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("IO-HMM has not been fitted")
        assert self.initial_probabilities is not None
        assert self.transition_probabilities is not None
        assert self.emission_probabilities is not None
        return (
            self.initial_probabilities,
            self.transition_probabilities,
            self.emission_probabilities,
        )

    def _forward(
        self, observations: np.ndarray, inputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        initial, transitions, emissions = self._parameters()
        length = len(observations)
        alpha = np.zeros((length, self.config.num_states), dtype=np.float64)
        scales = np.zeros(length, dtype=np.float64)
        alpha[0] = initial * emissions[inputs[0], :, observations[0]]
        scales[0] = max(alpha[0].sum(), np.finfo(np.float64).tiny)
        alpha[0] /= scales[0]
        for time in range(1, length):
            alpha[time] = (
                alpha[time - 1] @ transitions[inputs[time]]
            ) * emissions[inputs[time], :, observations[time]]
            scales[time] = max(alpha[time].sum(), np.finfo(np.float64).tiny)
            alpha[time] /= scales[time]
        return alpha, scales, float(np.log(scales).sum())

    def _backward(
        self, observations: np.ndarray, inputs: np.ndarray, scales: np.ndarray
    ) -> np.ndarray:
        _, transitions, emissions = self._parameters()
        length = len(observations)
        beta = np.ones((length, self.config.num_states), dtype=np.float64)
        for time in range(length - 2, -1, -1):
            next_input = inputs[time + 1]
            beta[time] = transitions[next_input] @ (
                emissions[next_input, :, observations[time + 1]] * beta[time + 1]
            )
            beta[time] /= max(scales[time + 1], np.finfo(np.float64).tiny)
        return beta

    def _fit_once(
        self,
        encoded: Sequence[tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> list[float]:
        self._random_parameters(rng)
        history: list[float] = []
        states = self.config.num_states
        inputs_count = len(self.input_vocab)
        observations_count = len(self.observation_vocab)
        tiny = np.finfo(np.float64).tiny

        for _ in range(self.config.max_iterations):
            initial_counts = np.full(states, self.config.smoothing, dtype=np.float64)
            transition_counts = np.zeros(
                (inputs_count, states, states), dtype=np.float64
            )
            emission_counts = np.zeros(
                (inputs_count, states, observations_count), dtype=np.float64
            )
            total_log_likelihood = 0.0
            _, transitions, emissions = self._parameters()
            for observations, input_ids in encoded:
                alpha, scales, sequence_ll = self._forward(observations, input_ids)
                beta = self._backward(observations, input_ids, scales)
                gamma = alpha * beta
                gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), tiny)
                initial_counts += gamma[0]
                for time, (observation, input_id) in enumerate(
                    zip(observations, input_ids)
                ):
                    emission_counts[input_id, :, observation] += gamma[time]
                    if time == 0:
                        continue
                    numerator = (
                        alpha[time - 1, :, None]
                        * transitions[input_id]
                        * (
                            emissions[input_id, :, observation] * beta[time]
                        )[None, :]
                    )
                    transition_counts[input_id] += numerator / max(numerator.sum(), tiny)
                total_log_likelihood += sequence_ll

            self.initial_probabilities = initial_counts / initial_counts.sum()

            pooled_transition = _normalize_last_axis(
                transition_counts.sum(axis=0) + self.config.smoothing
            )
            pooled_emission = _normalize_last_axis(
                emission_counts.sum(axis=0) + self.config.smoothing
            )
            updated_transitions = np.empty_like(transition_counts)
            updated_emissions = np.empty_like(emission_counts)
            # Context id zero is the explicit unseen-context backoff table.
            updated_transitions[0] = pooled_transition
            updated_emissions[0] = pooled_emission
            for input_id in range(1, inputs_count):
                updated_transitions[input_id] = _normalize_last_axis(
                    transition_counts[input_id]
                    + self.config.backoff_strength * pooled_transition
                    + self.config.smoothing
                )
                updated_emissions[input_id] = _normalize_last_axis(
                    emission_counts[input_id]
                    + self.config.backoff_strength * pooled_emission
                    + self.config.smoothing
                )
            self.transition_probabilities = updated_transitions
            self.emission_probabilities = updated_emissions
            history.append(total_log_likelihood)
            if len(history) > 1:
                improvement = history[-1] - history[-2]
                if abs(improvement) <= self.config.tolerance * (1.0 + abs(history[-2])):
                    break
        return history

    def fit(
        self,
        observation_sequences: Sequence[Sequence[str]],
        input_sequences: Sequence[Sequence[str]],
    ) -> "HierarchicalDiscreteIOHMM":
        if not observation_sequences:
            raise ValueError("at least one sequence is required")
        self.observation_vocab = self._build_vocab(observation_sequences)
        self.input_vocab = self._build_vocab(input_sequences)
        encoded = self._encode(observation_sequences, input_sequences)

        best: dict[str, Any] | None = None
        for restart in range(self.config.restarts):
            rng = np.random.default_rng(self.config.random_seed + 104729 * restart)
            history = self._fit_once(encoded, rng)
            candidate = {
                "score": history[-1],
                "history": list(history),
                "initial": self.initial_probabilities.copy(),
                "transitions": self.transition_probabilities.copy(),
                "emissions": self.emission_probabilities.copy(),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        assert best is not None
        self.initial_probabilities = best["initial"]
        self.transition_probabilities = best["transitions"]
        self.emission_probabilities = best["emissions"]
        self.training_log_likelihood = best["history"]
        return self

    def sequence_log_likelihood(
        self, observations: Sequence[str], inputs: Sequence[str]
    ) -> float:
        encoded = self._encode([observations], [inputs])[0]
        return self._forward(*encoded)[2]

    def filtered_state(
        self, observations: Sequence[str], inputs: Sequence[str]
    ) -> np.ndarray:
        initial, _, _ = self._parameters()
        if not observations:
            if inputs:
                raise ValueError("inputs must be empty when the observation prefix is empty")
            return initial.copy()
        encoded = self._encode([observations], [inputs])[0]
        return self._forward(*encoded)[0][-1]

    def predict_next_distribution(
        self,
        observation_prefix: Sequence[str],
        input_prefix: Sequence[str],
        next_input: str,
    ) -> dict[str, float]:
        state = self.filtered_state(observation_prefix, input_prefix)
        _, transitions, emissions = self._parameters()
        input_id = self.input_vocab.get(str(next_input), 0)
        if observation_prefix:
            next_state = state @ transitions[input_id]
        else:
            next_state = state
        probabilities = next_state @ emissions[input_id]
        inverse_vocab = {index: token for token, index in self.observation_vocab.items()}
        return {
            inverse_vocab[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }

    def to_dict(self) -> dict[str, Any]:
        initial, transitions, emissions = self._parameters()
        return {
            "model_type": "hierarchical_discrete_io_hmm",
            "config": asdict(self.config),
            "observation_vocab": self.observation_vocab,
            "input_vocab": self.input_vocab,
            "initial_probabilities": initial.tolist(),
            "transition_probabilities": transitions.tolist(),
            "emission_probabilities": emissions.tolist(),
            "training_log_likelihood": self.training_log_likelihood,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HierarchicalDiscreteIOHMM":
        if payload.get("model_type") != "hierarchical_discrete_io_hmm":
            raise ValueError("unsupported IO-HMM serialization")
        model = cls(IOHMMConfig(**payload["config"]))
        model.observation_vocab = {
            str(key): int(value) for key, value in payload["observation_vocab"].items()
        }
        model.input_vocab = {
            str(key): int(value) for key, value in payload["input_vocab"].items()
        }
        model.initial_probabilities = np.asarray(
            payload["initial_probabilities"], dtype=np.float64
        )
        model.transition_probabilities = np.asarray(
            payload["transition_probabilities"], dtype=np.float64
        )
        model.emission_probabilities = np.asarray(
            payload["emission_probabilities"], dtype=np.float64
        )
        model.training_log_likelihood = [
            float(item) for item in payload.get("training_log_likelihood", [])
        ]
        return model

    def save(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> "HierarchicalDiscreteIOHMM":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def evaluate_next_events(
    model: HierarchicalDiscreteIOHMM,
    observation_sequences: Sequence[Sequence[str]],
    input_sequences: Sequence[Sequence[str]],
) -> dict[str, float | int]:
    if len(observation_sequences) != len(input_sequences):
        raise ValueError("observation and input sequence counts differ")
    total_nll = 0.0
    correct = 0
    event_count = 0
    epsilon = 1e-12
    for observations, inputs in zip(observation_sequences, input_sequences):
        if len(observations) != len(inputs):
            raise ValueError("input/observation sequence lengths differ")
        for time, target in enumerate(observations):
            distribution = model.predict_next_distribution(
                observations[:time], inputs[:time], inputs[time]
            )
            probability = distribution.get(str(target), distribution.get(UNKNOWN_TOKEN, 0.0))
            total_nll -= math.log(max(probability, epsilon))
            predicted = max(distribution, key=distribution.get)
            correct += predicted == str(target)
            event_count += 1
    if event_count == 0:
        raise ValueError("evaluation has no events")
    mean_nll = total_nll / event_count
    return {
        "sequence_count": len(observation_sequences),
        "event_count": event_count,
        "negative_log_likelihood": total_nll,
        "mean_event_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 700.0)),
        "next_event_accuracy": correct / event_count,
    }


class SmoothedContextMarkovBaseline:
    """Transparent first-order counterbaseline for the IO-HMM."""

    def __init__(self, alpha: float = 0.5) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.vocabulary: list[str] = []
        self.context_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.previous_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()

    def fit(
        self,
        observation_sequences: Sequence[Sequence[str]],
        input_sequences: Sequence[Sequence[str]],
    ) -> "SmoothedContextMarkovBaseline":
        if len(observation_sequences) != len(input_sequences):
            raise ValueError("observation and input sequence counts differ")
        vocabulary = {UNKNOWN_TOKEN}
        for observations, inputs in zip(observation_sequences, input_sequences):
            if len(observations) != len(inputs):
                raise ValueError("input/observation sequence lengths differ")
            previous = BEGIN_TOKEN
            for observation, input_token in zip(observations, inputs):
                target = str(observation)
                context = str(input_token)
                vocabulary.add(target)
                self.context_counts[(context, previous)][target] += 1
                self.previous_counts[previous][target] += 1
                self.global_counts[target] += 1
                previous = target
        self.vocabulary = sorted(vocabulary)
        return self

    def distribution(self, previous: str, context: str) -> dict[str, float]:
        if not self.vocabulary:
            raise RuntimeError("Markov baseline has not been fitted")
        counts = self.context_counts.get((str(context), str(previous)))
        if not counts:
            counts = self.previous_counts.get(str(previous))
        if not counts:
            counts = self.global_counts
        total = sum(counts.values()) + self.alpha * len(self.vocabulary)
        return {
            token: (counts.get(token, 0) + self.alpha) / total
            for token in self.vocabulary
        }


def evaluate_markov_baseline(
    model: SmoothedContextMarkovBaseline,
    observation_sequences: Sequence[Sequence[str]],
    input_sequences: Sequence[Sequence[str]],
) -> dict[str, float | int]:
    total_nll = 0.0
    correct = 0
    events = 0
    for observations, inputs in zip(observation_sequences, input_sequences):
        previous = BEGIN_TOKEN
        for observation, input_token in zip(observations, inputs):
            distribution = model.distribution(previous, str(input_token))
            target = str(observation)
            probability = distribution.get(target, distribution[UNKNOWN_TOKEN])
            total_nll -= math.log(max(probability, 1e-12))
            correct += max(distribution, key=distribution.get) == target
            previous = target
            events += 1
    if not events:
        raise ValueError("evaluation has no events")
    mean_nll = total_nll / events
    return {
        "sequence_count": len(observation_sequences),
        "event_count": events,
        "negative_log_likelihood": total_nll,
        "mean_event_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 700.0)),
        "next_event_accuracy": correct / events,
    }
