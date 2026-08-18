# Paired branch identifiability v17 preregistration

Date: 2026-08-18

Status: frozen scientific question and gate before manifest construction. No fresh branch outcome has been read.

Manifest freeze: two independent remote builds are byte-identical. Manifest SHA256 `fcf5f30d105c9b1eb56e4b888fcedf9bcb667fb195781168e151bd44dd499b13`; audit SHA256 `8114cf33b35d095fab642343c96fc939f3427af5f11bdc1328db5bc89060d575`. The manifest contains 47 exact query-plus-argument rows not present in the 0805 pilot and one structurally selected repeated control. Its fixed execution budget is 96 branch calls plus 152 observed-prefix replay calls, 248 synthetic sandbox tool executions total.

## Why this round exists

The v16 retrieval result rejected another observational-data architecture: semantic support covered 84.97% of multi-step rows, yet supported retrieval was 0.82085 NLL worse than Structured Markov v6 and only 1 of 20 tasks improved. The 0805 clean counterfactual collector proved exact prefix replay is feasible, but its 24 actions were spread over 22 states. Only two roots had more than one branch, with zero execution errors, zero conflicts, and one ambiguity event. It therefore did not identify within-state action effects.

The present question is narrower: can outcome-blind structural selection produce a small, reproducible dataset in which several legal actions are executed from exactly the same clean state and lead to non-trivially different causal effects?

## Frozen design

- Twelve clean training tasks: one task for each suite × difficulty cell.
- One exact reconstructed prefix per task.
- Four actions per prefix: two read-only and two mutating.
- Two independent fresh-state replicas per action.
- Exactly 48 bound queries and 96 branch executions; prefix replay calls are a deterministic consequence of the frozen manifest and must be frozen before execution.
- Zero victim-model calls, attacks, external endpoints, model training, Dreamer, utility heads, or planning.

Selection can use trusted goal structure, current Semantic State v3, legal tool schemas, and schema-valid arguments copied from already-observed clean calls. It cannot use donor outputs, counterfactual outcomes, expert future calls, task utility, security labels, or final reports. The 0805 exact query-plus-argument rows are deprioritized but not used as an outcome filter.

## Frozen gate

Collector integrity requires all 48 rows, all 96 replicas, byte-identical pairs, exact prefix replay, schema-valid arguments, zero infrastructure failures, and zero semantic-state leakage.

The scientific gate additionally requires all 12 complete four-action anchors, at least 10 anchors with two distinct action-effect projections, at least 6 with three, at least 50% pairwise effect differences, at least 6 boundary events, and at least two boundary-event types among execution error, conflict, and ambiguity. The effect projection excludes action identity, last-action text, raw output, and final labels, preventing a trivial pass from merely recording which action was executed.

Passing authorizes only a separate task-disjoint data expansion and a small intervention-masked entity-slot world-model probe. Failure stops learned action-effect latent modeling on the current AgentDojo collection; it does not authorize threshold changes or another encoder search.
