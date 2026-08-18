# Paired branch identifiability v17 results

Date: 2026-08-18

Final decision: `NO_GO_PAIRED_BRANCH_DATA_DIRECTION_V17`

## Outcome

Same-prefix action branching solved the ordinary transition-identifiability problem but failed the frozen boundary-event gate. The result is scientifically useful counterevidence: the current obstacle is no longer absence of action-conditioned signal; it is the absence of parameter and temporal interventions that expose failure, persistence, and conflict boundaries.

## Frozen execution

- Slurm job: `7113`
- Code commit: `e0353ca45515b4e9e0e1ebeade5e3f6715b32c1b`
- Archive: `/share/guozhix/wmagentattack/0818/paired_branch_identifiability_v17/formal_v1`
- Manifest SHA256: `fcf5f30d105c9b1eb56e4b888fcedf9bcb667fb195781168e151bd44dd499b13`
- Gate SHA256: `67e007bfc3b137a75484474060a06b8661308ff33ad9caf9266499cd8f24004d`
- Dataset SHA256: `c3c46f54bc17b53bbcd77a2ed21e64d0e8602fbe6bd5cd6a1758bd7ce55f0ce2`
- Tests: 18 passed
- Budget: 48 bound actions, two replicas, 96 branch calls, 152 prefix-replay calls, 248 sandbox calls total
- LLM, attack, model-training, Dreamer, and real-endpoint calls: all zero

All checksums and frozen input hashes pass. There were no runtime failures, replay mismatches, replica mismatches, schema failures, or semantic-state leakage findings.

## Exact metrics

| Metric | Frozen threshold | Result | Clause |
|---|---:|---:|---|
| Complete four-action anchors | 12 | 12 | pass |
| Anchors with at least two effects | 10 | 12 | pass |
| Anchors with at least three effects | 6 | 12 | pass |
| Pairwise effect difference | at least 0.50 | 0.95833 | pass |
| Boundary events | at least 6 | 3 | fail |
| Boundary-event types | at least 2 | 1 | fail |

The three boundary events were ambiguities. Execution errors and conflicts were both zero. Every anchor nevertheless showed execution-status or state-change diversity, and 69 of 72 same-root action pairs produced different action-independent effect projections.

## Interpretation and counterevidence

The 0805 pilot spread 24 actions over 22 states, whereas v17 places four actions at each of 12 exact roots. This change produces strong within-state causal contrast and confirms that ordinary read/write dynamics are identifiable when data are collected as forks.

However, argument values copied from successful clean calls remain valid across the highly permissive AgentDojo environments. Even cross-task values often return an empty result or create a valid mutation rather than an error. A representation learner cannot infer error or conflict boundaries that the dataset never traverses. The frozen v17 gate therefore remains NO-GO; its strong ordinary-effect diagnostics must not be relabeled as a pass.

## Literature mapping

- Bellot, Richens, and Everitt (ICML 2025) establish limits on predicting intentional agents from behavioural data alone. The repository implication is to add controlled interventions rather than another encoder.
- Causal-JEPA (2026) shows that structured masking can make interaction reasoning necessary, but explicitly cautions that masked completion induces a causal bias rather than recovering true causal interactions. The repository implication is to use entity-slot masking only after true action forks exist.
- TrajWorld (ICML 2025) gains from large heterogeneous action-labelled trajectory collections. The repository implication is that 6763 observational events are not a substitute for action-balanced, root-matched interventions.

## Authorized next mechanism

Run a new, separately frozen clean-only parameter-intervention pilot. For the same state and tool, pair one valid control with one schema-valid but simulator-precondition-invalid argument mutation, such as a missing transaction ID, nonexistent Slack entity, missing cloud-drive file ID, or invalid travel date. Require duplicate determinism and paired status flips. This tests whether hard failure supervision can be collected by design.

Conflict/persistence remains a separate data requirement. It should later use two-step `modify → read-back` forks; it is not authorized as satisfied by v17.
