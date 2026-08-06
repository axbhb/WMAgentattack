# tau3 bounded-horizon pilot results

## Frozen decision

`HORIZON_PILOT_GO__AUTHORIZE_FULL_96_CONFIRMATION`

The 24-episode same-task/same-seed pilot passed every preregistered clause.
Doubling only the two role call caps from 8 to 16 and the coherent orchestrator
horizon from 32 to 64 materially reduced truncation and recovered
assistant-side state-changing transitions. This result authorizes only a
separately frozen 96-episode confirmation under the identical contract.
Predictive-method training, large-scale collection, attacks, Dreamer, and
planner runs remain forbidden.

## Exact paired results

| Metric | Candidate | Paired parent | Frozen gate | Decision |
|---|---:|---:|---:|---|
| Complete episodes | 24 | 24 | 24 | PASS |
| Forced-budget-stop episodes | 5 | 18 | <=6 | PASS |
| Relative forced-stop reduction | 72.22% | - | >=50% | PASS |
| Adjacent assistant transitions | 103 | 74 | >=25 | PASS |
| State-changing assistant transitions | 10 | 1 | >=4 | PASS |
| State-unchanged assistant transitions | 93 | 73 | >=8 | PASS |
| Tasks with an assistant state change | 6 | 1 | >=2 | PASS |
| Domains with an assistant state change | 2 | 1 | >=2 | PASS |
| Paired changed-transition gain | +9 | - | >=3 | PASS |
| Supported transition targets | 4 | - | >=4 | PASS |
| Assistant tool-error rate | 22.33% | 25.68% | increase <=5 pp | PASS |
| Parent-prefix mismatches | 0 | - | 0 | PASS |

All 24 episodes completed with zero runtime failures, private-scenario
exposures, real endpoint calls, illegal tool names, split overlaps,
semantic-state leakage findings, or exact-replay disagreements. Every paired
parent prefix was reproduced exactly before the additional horizon suffix.
All logical, physical, retry, and exact-replay budgets were respected.

The four supported targets were `execution_error`, `goal_overlap_gained`,
`novel_observation`, and `state_changed`. `output_nonempty` remained
constant-positive, so it was correctly retained as unsupported rather than
fabricating a negative class.

## Counterevidence

- Airline recovered five state changes versus one in the paired parent; retail
  recovered five versus zero.
- Telecom remained at zero state-changing assistant transitions despite the
  longer horizon. The pilot therefore establishes a useful two-domain signal,
  not universal domain coverage.
- Five of 24 episodes still hit a role cap. The mechanism substantially reduced
  truncation but did not eliminate it.
- The assistant made more tool calls and more absolute errors (23 versus 19),
  while its error rate decreased from 25.68% to 22.33%. This passes the frozen
  rate-based non-regression clause but remains relevant for confirmation.

## Authorized next stage

Freeze the same 16/16 role caps and 64-step orchestrator contract over the full
original 96-episode panel before accessing confirmation outcomes. The full
data gate must be specified before submission and must retain task-disjoint
splits, exact parent-prefix checks, assistant-only transition labels, two fresh
replay replicas, and all safety boundaries. Only a full confirmation GO can
authorize the already-specified frequency/TF-IDF/Semantic Markov/Structured
v3/full-history/observed-v4 comparison.

## Reproducibility

- Generation array: `6534`
- Frozen summary: `6535`
- Execution commit: `69e23927f57e444befc77a92ebda367e1c3aac8d`
- Archive: `/share/guozhix/wmagentattack/0806/tau3_horizon_extension/pilot_v1`
- Frozen protocol SHA256: `537e5221236cc1f93e86ab6550f2d1a3cb01cfd30ac24589be3e948e82d32d69`
- Manifest SHA256: `5669e24c37ae2f5de5f9c1fb0eb1e07cafbd7f8587b4b4668ab67dba71d35ac0`
- Dataset SHA256: `23e368d8185b8ca8db151f3cf9b831cdca594d05c3c73b062fa2836353799327`
- Dataset audit SHA256: `568c1f615977c56e6eb6c3edd16b2faff1160d050633a8952da79307f53454ff`
- Gate SHA256: `4e870450406a1c4a32f6cdfc4ab5b0ed416cccb29f6528a918e333c4a74c4048`
- Archived report SHA256: `39b78e4b59ab3a8eff6ac29dc419dc8c34c9221a90c558b34a46a66927155db4`
- Archive checksum-file SHA256: `534e1a93facdbcbf061172d950cf733cab680eae93365ae9d1594558ba1fb750`
