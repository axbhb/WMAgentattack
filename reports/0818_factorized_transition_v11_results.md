# Factorized semantic-transition v11: final result

## Conclusion

The frozen decision is `NO_GO_FACTORIZED_SEMANTIC_TRANSITION_V11`. The four deterministic labels pass their data-integrity gate, and the candidate preserves one-step action quality, but the factorized mechanism does not improve task-disjoint multi-step action prediction or the four-cell task/attack outcome. The failure persists when the true factors are supplied at confirmation time, so factor usefulness—not merely factor prediction—is the primary bottleneck.

The retained model remains **Structured Markov v3 + four-cell auxiliary + zero-initialized residual dynamics (v6)**. v11 is discarded as a replacement.

## Frozen experiment

- Dataset: 6,763 events, 2,060 trajectories, 4,703 adjacent transitions, 20 task-disjoint AgentDojo tasks.
- Label gate: two byte-identical builds; all 4,703 source events unique; 20/20 tasks; zero forbidden final-outcome or next-action factor leakage.
- Model budget: five task folds × three seeds; 15 v5 teacher fits, 15 capacity-control fits, and 15 factor-supervised fits. The two compared residual arms each contain exactly 94,584 trainable parameters.
- Confirmation outputs: 41,433 rows per arm for v6, capacity control, predicted-factor v11, and oracle-factor diagnostic.
- Runtime: Slurm 7106, zero failures, 8/8 tests, empty stderr, frozen source/data verification and archive checksum verification all pass.

## Exact paired effects

Positive values mean v11 is better.

| Metric | Effect | Frozen decision |
|---|---:|---|
| One-step action NLL vs v6 | +0.001480 | noninferiority pass |
| One-step action accuracy vs v6 | +0.000603 | noninferiority pass |
| Horizon 2–5 action NLL vs v6 | -0.008962 | fail |
| Horizon 2–5 action NLL vs equal-capacity control | -0.000285 | fail |
| Future four-cell cross-entropy vs v6 | -0.022366 | fail |
| Oracle factors vs predicted factors, horizon 2–5 | -0.000836 | diagnostic counterevidence |
| Oracle factors vs v6, horizon 2–5 | -0.009798 | diagnostic counterevidence |

Multi-step effects are positive on only 10/20 tasks and on only one of three seeds. The paired bootstrap interval for the v6 comparison is [-0.026284, 0.005318], so neither a reliable gain nor the preregistered +0.01 effect is present.

## Factor-prediction audit

| Factor | NLL gain over train-fold prior | Accuracy gain | Interpretation |
|---|---:|---:|---|
| Observation delta | -0.006455 | -0.000073 | not transferable; dominated by rewrite/loss |
| Evidence delta | +0.079254 | +0.109190 | transferable |
| Goal progress | +0.066447 | -0.011423 | probability improves but decisions do not |
| Execution status | +0.070442 | +0.037716 | transferable but highly imbalanced |

Only evidence and execution satisfy both frozen per-factor gain clauses. Observation delta fails both, while goal progress fails accuracy. The factor gate remains small (mean tanh gate -0.013 for the supervised model), and the factor-supervised model has slightly worse training horizon losses than its equal-capacity control.

## Counterevidence and bottleneck diagnosis

The oracle diagnostic is decisive for the direction change. If factor prediction alone were the bottleneck, replacing predicted factors with true factors should improve free rollout. Instead it worsens predicted v11 by 0.000836 NLL and remains 0.009798 worse than v6. Therefore more factor-head capacity, class reweighting, or a better semantic encoder cannot by itself solve this formulation.

The labels describe generic text-surface change rather than the action-relevant event that determines the next legal tool. In particular, `rewrite_or_loss` covers 87.05% of observation transitions; goal-token overlap confounds lexical disappearance with task regression; execution status is 90.58% productive continuation. These targets are valid observables but weak sufficient statistics for victim action dynamics.

## Authorized direction change

Stop generic state-delta factorization. The next candidate should model an **action-conditioned event graph**: predict entity-slot mutations, tool precondition satisfaction, newly enabled/disabled legal actions, and canonical receipt relations produced by the current tool call. These targets must be derived causally from the current and next sandbox-visible records, audited for support and leakage, and tested first with an oracle-sufficiency gate. Neural training is authorized only if true event-graph deltas materially improve held-out multi-step prediction over v6; this avoids spending another full budget on targets that are not useful even when known perfectly.

## Provenance

- Run commit: `b446c8b13aa3e44a2fb9a886a3921704b4002376`
- Archive: `/share/guozhix/wmagentattack/0818/factorized_semantic_transition_v11/stage_f2/formal_v1`
- Prediction SHA256: `9fec358472babcca7d76513dc144bda354ef371b55b207b82cdb80fc9a60948e`
- Gate SHA256: `bb6f577837a81615e088fe1942c93c3f0485608b13d44f36b80d8c5a57ebf68f`
- Archive checksum manifest SHA256: `cebe0e77debd01751638ecd46dc1f308c52df805d8d637240c915e12707e3830`
