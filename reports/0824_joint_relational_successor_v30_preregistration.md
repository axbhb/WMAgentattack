# v30 Joint Relational Successor — Preregistration

## Frozen mechanism

Replace only v28's independent record and global goal-pointer heads. The v30 model scores action-conditioned static record candidates and then scores a bipartite edge between each candidate and each current goal term. Global goal progress and matched-count probabilities are deterministic projections of those joint probabilities. Structured v3 state, zero-start recurrent residual dynamics, fixed-v21 control, tasks, split refs, seeds, support rows, renderer, and all non-inferiority surfaces remain fixed.

## Budget

Thirty CPU fits: 15 fixed-v21 reproductions and 15 v30 fits across the same task-disjoint, tool-family-heldout, and source-heldout units used by v28. No GPU, LLM, sandbox, attack, real endpoint, or post-result rerun is allowed.

## Gate rationale

The gate requires both local mechanism success (record, relation, and pointer metrics) and downstream canonical transfer (unseen recall/NLL, focused non-count recall, count-3, precision/FPR, per-cell stability, tool/source diagnostics) while preserving seen, one-step, rollout, query-read, and paired behavior. This prevents a low relation training loss from being accepted as world-model evidence.
