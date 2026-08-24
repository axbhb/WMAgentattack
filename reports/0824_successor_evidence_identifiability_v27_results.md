# v27 typed successor-evidence identifiability results

## Conclusion

`GO_BOUND_SUCCESSOR_MODEL_V27` (23/23 frozen clauses passed).

The existing frozen counterfactual data can be represented as task-disjoint, leakage-free typed successor-evidence deltas without losing the exact v20/v21 effect labels. This closes the specific data-identifiability failure exposed by v26 and authorizes one small, separately preregistered bound-record successor model comparison. It does **not** establish a learned-model improvement and does not authorize data scale-up, attacks, a planner, Dreamer, or large world-model training.

## Frozen execution

- Remote Slurm job: `7312`; elapsed: 7 seconds; runtime failures: 0.
- Budget used: two deterministic CPU builds, 0 model fits, 0 GPU requests, 0 LLM calls, 0 sandbox calls, 0 post-result reruns.
- Frozen remote commit: `0df9270b994a6a36f9e2f27089f0dfe020b87a66`.
- Tests: 3 passed in 1.90 seconds.
- All implementation, protocol, and six source-data hashes verified before results.

## Exact results

- Dataset: 131 rows: 121 confirmation transitions from 12 tasks plus 10 support transitions from 4 disjoint sibling tasks.
- Exact recovery: 121/121 full v20 effects and 121/121 v21 hard effects.
- Relation coverage: entity 4/4 = 1.0; attribute 5/5 = 1.0; operation 9/9 = 1.0.
- Support cardinalities: 8 rows with zero matched terms, 1 with one, and 1 with three. Every task-disjoint training fold contains cardinalities 0, 1, and 3.
- Integrity: zero task overlap, missing next states, non-adjacent transitions, model-input leakage, semantic leakage, pointer errors, and record-binding errors.
- Independent replication: build A and B datasets and audits are byte-identical.

## Interpretation and counterevidence

v26's support gain was real, but independent atom heads discarded which entity, field, link status, and tool outcome belonged together. v27 repairs the *target contract*: a model predicts bound evidence records, conflicts, execution status, state-delta bits, and pointers to trusted goal terms. Source and matched count are then deterministically rendered rather than learned as independent labels.

This positive gate is intentionally narrow. Fold 1 contains no held-out entity/attribute relation occurrence, so its relation coverage is vacuous rather than additional evidence. The total support set is only 10 rows and the three-match case has one support example. Therefore the result demonstrates reconstructability and fold support, not statistical generalization or superiority over Structured Markov v3/v21.

## Next authorized experiment

Preregister exactly one small CPU comparison using the frozen 131-row dataset:

1. retained v21 closed-vocabulary model;
2. v26 independent-atom control;
3. a bound successor-record model that predicts execution/delta first, then typed records and goal-term pointers with permutation-invariant matching;
4. the same task folds and seeds, reporting unseen exact-record F1, rendered-effect recall/NLL, binding accuracy, matched-count-3 recall, one-step BCE, and v19 rollout non-inferiority.

The bound model must beat v26 and not underperform v21 on retained surfaces before any larger data generation is reconsidered.

## Archive and hashes

- Archive: `/share/guozhix/wmagentattack/0824/successor_evidence_identifiability_v27/formal_v1`
- Dataset SHA256: `88b9825b910f212c22dad91cab2274589e15302beaf660541f9c485567933070`
- Audit SHA256: `7930c6b35a513c45a81929dcfe2f6db2dc19e0133f5224a20b5ece296f338370`
- Gate SHA256: `f5c65119f1f669ac2d026a895d1b524c6ae50cc7b9614ea0efee632cf4e34bcd`
