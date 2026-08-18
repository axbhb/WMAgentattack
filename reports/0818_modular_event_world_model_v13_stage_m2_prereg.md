# Modular event world model v13 — Stage M2 preregistration

Stage M1 established that v12's true event graph improves multi-step action dynamics while an independent frozen v6 branch preserves four-cell task/attack outcomes exactly. Stage M2 tests whether the event graph can be predicted causally rather than supplied by an oracle.

The candidate and capacity control have identical parameters. The candidate observes the current event graph and receives next-graph BCE supervision; the control receives a zero current graph and no graph labels. Both roll out only predicted future graphs. The next action is supplied only as teacher forcing during training; confirmation rollouts use the model's own probability-weighted action. The graph branch never receives four-cell outcome labels, and the final four-cell probabilities are copied exactly from frozen v6.

The fixed budget is 15 teacher fits plus 30 equal-capacity dynamics fits over five task-disjoint folds and seeds 7/17/29. No attacks, planners, LLM calls, task removal, threshold changes, or post-result reruns are authorized.

The candidate must beat a training-fold graph prior, preserve one-step performance, improve horizons 2–5 against both v6 and the equal-capacity control, show task and seed breadth, maintain legal predictions, and preserve the v6 four-cell output exactly. Failure of graph prediction and failure of downstream use are diagnosed separately.
