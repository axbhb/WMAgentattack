# Modular event world model v13 results

## Conclusion

Stage M1 passed: exact recombination of the v12 true-event-graph action branch and frozen v6 four-cell outcome branch preserves the outcome predictions exactly while retaining the v12 multi-step action gain. This establishes that the v12 failure was shared-representation negative transfer rather than an incompatibility between the two outputs.

Stage M2 is `NO_GO_PREDICTED_EVENT_GRAPH_WORLD_MODEL_V13`. The causal learned predictor does not make the full 263-dimensional event graph predictable under task-disjoint transfer and does not retain the oracle action gain. The deployed architecture remains v6; the modular separation is retained as a design constraint.

## Fixed experiment

- Slurm job: 7108, completed with exit code 0 in 8 minutes 31 seconds.
- Budget: 15 frozen Structured Markov teacher fits and 30 equal-capacity dynamics fits.
- Evaluation: five task-disjoint folds, seeds 7/17/29, horizons 1–5.
- Rows: 41,433 for v6, capacity control, candidate, and modular composite respectively.
- Integrity: all predictions legal; equal parameter counts; zero runtime failures; frozen data/code hashes verified; stderr empty.

## Exact task-macro effects

- Graph BCE gain over the training-fold prior: **-0.069719**. All three seeds and 19 of 20 held-out tasks are negative.
- Horizon-1 action NLL gain over v6: **+0.000947**.
- Horizon-1 accuracy gain over v6: **+0.002467**.
- Horizon-2–5 action NLL gain over v6: **-0.006191**.
- Horizon-2–5 gain over the equal-capacity unsupervised control: **+0.004009**, below the frozen +0.005 threshold.
- Positive held-out task fraction at horizons 2–5: **0.45**; positive seeds: **1/3**.
- Four-cell future joint CE difference from v6: **0.0**, exactly as designed.

Eight of thirteen gate clauses passed. Graph predictability, multi-step gains against v6 and capacity control, task breadth, and seed replication failed.

## Counterevidence and interpretation

The true event graph remains a strong oracle signal from v12, but reconstructing the entire graph is a poor causal learning target. It mixes deterministic action/tool protocol structure with stochastic receipt and evidence changes, and dense positive-weighted reconstruction is badly calibrated relative to a simple feature prior. This is not evidence that multi-step dynamics or modular outcomes are useless: one-step performance is preserved and the modular outcome path is exact.

The next direction therefore changes the target rather than enlarging the predictor. v14 will separate action/tool features that can be computed exactly from the chosen action and sandbox protocol from receipt/evidence residuals that must be learned. It will first test each partition as an oracle under the same task-disjoint gate. A learned residual predictor is authorized only if the stochastic partition retains the v12 action benefit.
