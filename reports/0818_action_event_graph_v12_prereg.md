# Action-conditioned event graph v12 preregistration

v11 established that generic observation/evidence/progress/execution deltas are not action-dynamics sufficient: even true factors failed to improve v6. v12 therefore changes the research target rather than increasing encoder capacity.

The new representation is constructed from information visible after the current tool attempt: exact selected tool, typed argument schema, value-anonymized receipt structure, normalized error/format indicators, relation-to-goal bins, causal skill history, prior-to-current entity-schema changes, and legal-action changes. Raw values, raw text, task IDs, trajectory IDs, next actions, final task/attack labels, probability labels, and policy-violation labels are excluded.

Stage G1 requires two byte-identical 6,763-row builds, exact alignment to all canonical events and 20 tasks, complete tool identity for every non-finish event, noncollapsed signatures, at least five receipt formats, and zero forbidden features.

Only after G1 passes may Stage G2 train a parameter-matched zero-graph control and true-event-graph oracle adapter on the frozen five task folds and seeds 7/17/29. True future event graphs are diagnostic only. The direction advances to a learned graph predictor only if the oracle preserves h1, improves h2–h5 NLL by at least 0.02 over v6 and 0.01 over the equal-capacity control, covers at least 60% of tasks and two seeds, improves future four-cell CE by at least 0.005, and retains legal predictions.
