# Parallel world-model gates v22: preregistration

Date: 2026-08-21

This fixed-budget cycle separates three questions that must not compensate for
one another.

1. **Open vocabulary.** Compare the retained fixed v21 effect head, a
   capacity control with independent label embeddings, and a candidate scorer
   that encodes category/entity/field/kind/value slots. Positive labels absent
   from a training fold are excluded from candidate training loss and become
   the primary confirmation target. No task, source, outcome, utility, or
   security identifier enters the model.
2. **Data generation.** Freeze four linked tables (episodes, transitions,
   outcomes, pairs), a common Llama-3.1-70B contract, same-seed intervention
   pairs, and connected-component leakage splits. Four-cell task/attack
   outcomes remain episode-level supervision and are not copied into each
   transition.
3. **Long horizon.** Audit real contiguous sequences before fitting a model.
   The corpus must contain enough task-disjoint 5- and 10-step sequences;
   concatenating unrelated transitions or repeating three-step sequences is
   forbidden.

The open-vocabulary budget is exactly 45 CPU fits. The data line is limited to
a 96-episode smoke protocol and cannot authorize large generation. The
long-horizon line performs zero model fits if its data-sufficiency gate fails.
No post-result threshold changes, task removal, attack generation, Dreamer,
planner, or real endpoint calls are allowed.

Primary evidence motivating this design is semantic-label transfer in
zero-shot classification and recent counterevidence that average prediction
quality can hide weak action controllability:

- https://proceedings.mlr.press/v38/li15d.html
- https://proceedings.mlr.press/v235/liu24bq.html
- https://arxiv.org/abs/2608.04653
- https://arxiv.org/abs/2607.22430
