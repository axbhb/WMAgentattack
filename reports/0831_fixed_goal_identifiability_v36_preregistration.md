# v36 fixed-goal identifiability audit

Frozen after v35's support NO-GO, before calculating v36 outcomes. Existing
data only; one 15-minute CPU analysis; no model fits or new episodes.

Hypothesis: user-task-only ranking inflates support by comparing different
injection goals. Partition all 400 configurations into 80 fixed-goal blocks of
five variants. Keep all 20 tasks and all four suites. Evaluate p11 contrasts,
not utility-only preferences. Directly compare the allowed feature dictionaries
for outcome-contrasting pairs; identical dictionaries imply an input-resolution
limit for this representation, not necessarily absence of an attack effect.

The protocol freezes 12 informative tasks, 24 informative goals, 0.05 task-macro
oracle/random gap, at most 25% feature collisions, and three suites with at
least two informative tasks. All clauses must pass. These are coverage/design
criteria, not statistical significance claims. The oracle uses the same five
observations it maximizes and is optimistically biased. Independent posterior
draws do not recover missing episode-level seed pairing. Previously inspected
data are not a new independent confirmation set.

Literature mapping: [AutoInject](https://arxiv.org/html/2602.05746v2) learns
comparison feedback in a fixed user-task/injection-goal search. Here Bayesian
comparisons reuse old counts; they add no fresh feedback. The concrete repo
change is a fixed-goal grouping audit before any selector experiment.
[PIMiner](https://arxiv.org/html/2608.05108v1) motivates examining transferable
strategy diversity rather than merely increasing encoder capacity. Its target
environment differs; its external-agent execution is not imported and its
reported success rates are not directly comparable to our p11.

NO-GO leads to fresh clean-only eligibility and common-seed integrity work,
not a relaxed data gate or an enlarged neural model.
