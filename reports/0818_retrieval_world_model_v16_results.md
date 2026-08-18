# Retrieval successor world model v16 results

## Conclusion

The frozen decision is `NO_GO_RETRIEVAL_SUPPORT_V16`. Semantic retrieval has broad coverage and its similarity score ranks easy versus hard rows, but retrieved successor actions are substantially worse than v6 even on supported held-out events. A v6 residual-fusion experiment is not authorized.

## Fixed experiment

- Slurm 7111 completed with exit code 0 in 39 seconds, CPU only.
- Five task-disjoint folds, horizons 1–5, paired with v6 seeds 7/17/29.
- 41,433 complete rows; all predictions legal; zero runtime failures; stderr empty; hashes verified.
- k=32, temperature 0.10, fixed block weights, and training-only leave-one-task-out support calibration.

## Exact results

- Supported fraction at horizons 2–5: **0.849693**.
- Uncovered minus supported retrieval NLL: **+0.187184**; support ranking is informative.
- Supported retrieval NLL gain over v6: **-0.820849**.
- Positive task fraction: **0.05** (1/20 tasks).
- All three paired seeds are negative.

## Interpretation

The failure is not lack of nearest neighbors. Many held-out states have close semantic prototypes, but those prototypes imply different next obligations and therefore different successor actions. This is evidence that the observational dataset does not identify the desired transition function from the current representation: semantic closeness is not causal equivalence.

Together with v13 dense reconstruction and v15 latent distillation failures, this closes the current architecture-search loop. The next stage must change the data. It should collect clean-only AgentDojo sandbox branches from matched prefixes: hold the canonical state fixed, intervene on legal actions or argument/evidence outcomes, and record multiple successor states. Such paired branches can separate action effects, task obligations, and stochastic evidence changes. No attack generation or large Dreamer training is authorized before a branching-coverage and identifiability gate passes.
