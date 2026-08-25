# v33 attack-conditioned four-cell ranking results

## Decision

`NO_GO_ATTACK_CONDITIONED_RANKING_V33`

The fixed 45-fit CPU budget completed under Slurm 7331 with zero runtime
failures and no new attack, victim-model, sandbox-tool, or external-endpoint
execution. Eight of twelve frozen clauses passed.

## Exact task-level metrics

| Arm | Top-1 target p11 | Pairwise accuracy | p11 Brier | Four-cell CE |
|---|---:|---:|---:|---:|
| Frozen v5 | 0.228571 | 0.837163 | 0.023636 | 1.215592 |
| Structured attack residual | 0.257143 | 0.680320 | 0.041143 | 1.271805 |
| World + attack residual | 0.264286 | 0.755245 | 0.045020 | 1.288232 |
| Family-enabled diagnostic | 0.228571 | 0.726274 | 0.048747 | 1.302207 |
| Random expectation | 0.137143 | - | - | - |

The primary model improved top-1 p11 over frozen v5 by 0.035714 and replicated
that threshold in seeds 7 and 29. It nevertheless failed four binding clauses:

- pairwise accuracy was 0.081918 below v5 rather than at least 0.02 above it;
- p11 Brier worsened by 0.021384, exceeding the 0.005 non-inferiority margin;
- four-cell cross-entropy was 0.016427 worse than the structured attack-only arm;
- only 4/20 tasks improved, below the frozen 0.55 positive-task fraction.

All four positive task effects were in Banking. One Slack task fell by 0.285714;
the other fifteen tasks were unchanged. This is counterevidence against treating
the aggregate top-1 increase as a general attack-selection improvement.

## Retained conclusion

Retain Structured Markov v3/v5 and the structured attack-only model as
diagnostic baselines. Do not authorize a planner or large world-model run from
v33. The next experiment must use independently clean-solvable, task-disjoint,
same-seed paired attacks that alter one intervention factor at a time. The next
selector must use only information available before the candidate attack is
executed.
