# Late-fusion action distillation v15 preregistration

v14 shows that exact protocol and evidence features are each useful but the full gain depends on their interaction. v15 retains separate current-state encoders and adds a zero-initialized interaction path. Future steps do not reconstruct graph features; instead an action-conditioned 32-dimensional residual is trained through task action losses and, only in the candidate arm, KL distillation from a frozen full-graph oracle.

The equal-capacity control has the identical architecture and data but no oracle KL. The full oracle is trained independently inside each task-disjoint fold and never supplies confirmation inputs. Four-cell outcomes remain an exact frozen v6 branch. The fixed budget is 15 structured teachers, 15 full-graph oracles, 15 controls, and 15 distilled students over seeds 7/17/29. No result-dependent reruns or threshold changes are allowed.
