# v26 support-conditioned compositional world model preregistration

## Frozen question

Can the clean, outcome-blind v25 atom-support rows repair task-disjoint unseen effect prediction without changing the retained v21 seen-label transition model? The exact intervention is a factorized category/entity/field/kind/value decoder plus a separate cumulative ordinal head for `matched_count=0..3`.

## Controls and single mechanism

All task splits, seeds, hashes and budgets are inherited unchanged from v21/v23. The fresh comparison is `factorized_ordinal_no_support_v26` versus `factorized_ordinal_support_v26`; the only difference is whether the ten frozen v25 rows contribute atom loss. Both use the same ordinal head. `fixed_v21` is reproduced in every split/seed unit and frozen `hybrid_e5_raw_v23` metrics are an external counterexample.

Seen canonical effect labels always use fixed-v21 probabilities. Only labels with zero positive examples in the training fold use the atom renderer. The support loader is forbidden from reading `audit_only`, task IDs, suites, utility, security, attacks or final outcomes.

## Budget and gate

The fixed budget is 45 CPU fits: 15 fixed-v21, 15 factorized without support, and 15 factorized with support. There are no LLM calls, environment executions, GPU requests, attacks, planners, Dreamer runs or result-driven reruns.

GO requires all 20 frozen clauses, including task-unseen recall at least 0.60; gains of at least 0.04 over both v23 raw and the no-support arm; positive gain in at least four of six affected fold/seed cells; unseen NLL at most 1.0; precision at least 0.10; FPR at most 0.05; count-3 recall at least 0.60; focused unseen recall at least 0.55; and non-inferiority for seen recall, one-step BCE, rollout BCE, query/read recall and pair assignment. Diagnostic held-out suites and parameter/integrity gates must also pass.

Thresholds will not be changed after any v26 prediction is produced.
