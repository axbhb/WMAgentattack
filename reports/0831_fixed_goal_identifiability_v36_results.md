# v36 result: fixed-goal coverage is insufficient

Decision: `NO_GO_FIXED_GOAL_SUPPORT_V36`. Slurm 7559, 11 seconds, exit 0:0,
empty stderr, 17 tests passed, zero model fits/new episodes.

All 400 configurations align to 80 fixed-goal blocks in 20 tasks, five victim
trials per candidate. v35's 1,267 reward comparisons decompose exactly into
**1,000 cross-goal (78.93%) and 267 within-goal (21.07%)** comparisons.
Thus the original pair count overstates evidence for selecting a better
strategy for one fixed goal. It does not by itself prove that every former
model gain was caused by this confound.

| Frozen criterion | Observation | Required | Gate |
|---|---:|---:|---|
| Tasks with confident within-goal p11 contrast | 9/20 | 12 | fail |
| Goals with confident p11 contrast | 29/80 | 24 | pass |
| Task-macro empirical oracle minus random p11 | 0.1405 | 0.05 | pass |
| Identical-feature fraction among contrasting pairs | 0/209 | <=25% | pass |
| Suites with >=2 informative tasks | 2 | 3 | fail |

Only Banking (5 tasks) and Slack (4) contain p11 contrast. Travel and Workspace
have zero empirical p11 throughout this archive; Slack task 10 also has zero.
There are 63 within-goal reward-confident pairs without confident p11 contrast:
reward support is not equivalent to joint-success support.

Counterevidence to an overly pessimistic conclusion: the current structured
features distinguish all 209 p11-contrasting pairs. Some selection signal may
be learnable in the two supported suites; the data do not show a need for a
larger encoder. Random p11 is 0.092; the **in-sample** oracle is 0.2325. The latter
uses outcomes to select candidates and is optimistically biased, not a method
score, independent confirmation, or an achievable held-out improvement.

Next: retain all 20 tasks and independently measure clean solvability under
fresh seeds, while implementing actual variant-invariant seed allocation.
These are familiar tasks with fresh random draws, **not unseen tasks**. No
attack generation before the new clean gate and a separate frozen pilot.

Archive: `/share/guozhix/wmagentattack/0831/fixed_goal_identifiability_v36/formal_v1`.
