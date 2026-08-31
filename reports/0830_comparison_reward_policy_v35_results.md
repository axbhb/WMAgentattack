# v35 comparison reward results

Decision: `NO_GO_COMPARISON_DATA_SUPPORT_V35`.

Slurm 7558 completed on 2026-08-31 in 21 seconds, exit 0:0, empty stderr.
The server tests passed 11/11. All 400 configurations, 20 tasks, five trials
per configuration, targets, and frozen fold assignments passed alignment.

| Frozen support clause | Observed | Required | Result |
|---|---:|---:|---|
| Confident comparison pairs | 1,267 | 400 | pass |
| Tasks with at least 20 pairs | 11/20 | 12/20 | fail |
| Tasks with multiple families in comparisons | 16/20 | 16/20 | pass |

Pair counts by suite: Banking 501, Slack 685, Travel 5, Workspace 76. Every
Banking and Slack task reached 20 pairs; only Workspace task 18 did outside
those suites. Four Travel tasks have no confident comparison at all.

The gate stopped before any fitting: **0 model fits**, no victim calls, no tool
calls, no attack generation. This neither proves nor disproves the comparison
architecture; it says the frozen data coverage is insufficient. We do not
lower 12 to 11. The 1,267 pairs are derived from the same 400 configurations,
not 1,267 new independent observations.

Next: a separately preregistered, read-only fixed-injection-goal contrast audit.
The current user-task grouping allows different injection goals to compete;
goal difficulty could explain apparent ranking signal. Historical v34 source
also derives the actual episode seed from variant-specific row IDs, so its
run-seed labels alone do not establish common-random-number pairing. Historical
metrics are retained unchanged with this methodological qualification.

Archive: `/share/guozhix/wmagentattack/0830/comparison_reward_policy_v35/formal_v1`.
