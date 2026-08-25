# v33 attack-conditioned four-cell ranking preregistration

## Attack-focused question

This fixed-budget study asks whether the retained Structured Markov v3
four-cell signal becomes a better attack selector when it is conditioned on a
typed attacker action and optimized directly for
`P(task success AND attack success)`.

It reuses 400 existing AgentDojo attack configurations, each measured with
five Llama-3.1-70B seeds. No new payload, victim rollout, tool execution, real
endpoint, planner, Dreamer, or large model is run in this stage.

## Candidate

The candidate is a zero-initialized four-cell residual on top of the frozen v5
out-of-fold `p11` prediction. Its action representation contains attack
placement, timing, knowledge, endpoint policy, target tool sequence, typed
argument placeholders, injection-vector affordances, and independent clean
solvability. Raw task/injection IDs, raw payload/goal text, outcome labels, and
checker results are forbidden inputs.

The loss combines task-balanced four-cell soft cross-entropy with a within-task
pairwise ranking objective. Attack family/name/variant are excluded from the
primary model and enabled only in a diagnostic arm to search for template
memorization.

## Frozen controls and gate

The controls are frozen v5 p11, random within-task selection, a structured
attacker-only residual, and the family-enabled diagnostic. Five task-disjoint
folds, three model seeds, and three arms produce exactly 45 CPU fits.

The primary model advances only if it improves v5 top-1 target p11 by at least
0.02 and pairwise accuracy by at least 0.02, improves more than half the tasks,
replicates in at least two seeds, remains p11-Brier non-inferior, beats random,
and does not depend on the family-ID diagnostic. Thresholds are frozen before
reading v33 predictions.

A GO authorizes only a separately preregistered short-horizon sandbox pilot.
It does not authorize large world-model training or unrestricted attack
generation.
