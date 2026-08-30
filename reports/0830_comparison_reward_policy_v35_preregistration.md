# v35 uncertainty-aware comparison reward preregistration

## Frozen question

Can a pre-execution comparison-reward policy generalize across held-out AgentDojo
tasks better than an equal-capacity four-cell absolute predictor?

The experiment is motivated by the v34 counterevidence: only 10/96 attack trials
reached joint success, only two of eight tasks had more than one empirical p11
level, and both selectors underperformed random top-1 selection. This makes
absolute p11 regression a poor optimization signal.

## Mechanism under test

For every same-task candidate pair, v35 reconstructs the five-seed four-cell
counts and draws from independent Dirichlet posteriors. The soft comparison
label is the posterior probability that one candidate has larger constrained
reward. Joint task-and-attack success receives weight 1.0, attack success with
task failure receives -0.2, benign utility receives only 0.05, and complete
failure receives -0.05.

The candidate is a low-capacity Bradley--Terry reward head with a four-cell
outcome anchor. It sees only typed pre-execution attack and tool-affordance
features. Task identities, raw goals/payloads, victim trajectories, checker
fields, and final outcomes are excluded from inputs.

## Frozen gates and budget

The data-support gate runs before fitting. It requires at least 400 confident
pairs, at least 12/20 tasks with 20 confident pairs, and at least 16 tasks with
multiple attack families. Failure ends v35 as a data NO-GO.

If support passes, the fixed budget is five task-disjoint folds, three arms,
three seeds, and 45 CPU fits. The candidate must improve top-1 constrained
reward by 0.03, top-1 p11 by 0.02, and posterior pairwise accuracy by 0.03 over
the equal-capacity absolute baseline; improve at least 55% of tasks and two of
three seeds; preserve selected utility within 0.02; beat random p11 by 0.02;
and not require a family-name shortcut.

No threshold, task, feature, seed, or rerun may change after results are read.
This round generates no attacks, calls no victim model, and contacts no real
endpoint.
