# Modular event world model v13 preregistration

v12 showed a replicated action-dynamics gain but failed because a shared graph-conditioned hidden state degraded the four-cell task/attack head. v13 is a sequentially exploratory architecture split; it does not revise the v12 gate.

Stage M1 performs no fitting. It pairs identical fold/seed/horizon/event rows, takes action NLL/accuracy/legality from the frozen v12 true-event-graph arm, and takes joint trainability/cross-entropy from the frozen v6 outcome arm. It requires exact preservation of both field groups, all 41,433 rows, the original v12 action thresholds and capacity-control gain, exact future-joint replication, and legal actions.

A complete M1 pass authorizes development of a learned future event-graph predictor only for the action branch. The v6 outcome branch must remain independent so graph-prediction errors cannot corrupt the four-cell result distribution.
