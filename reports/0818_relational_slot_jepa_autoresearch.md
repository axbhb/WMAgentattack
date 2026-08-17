# Relational Slot-JEPA latent autoresearch

Status: `STAGE_A_PREREGISTERED_BEFORE_TRAINING`.

The fixed three-stage budget tests whether a canonical relational latent can replace the fixed Structured Markov v3 hash representation without reintroducing raw-text task memorization. Stage A changes only the state encoder by adding a zero-gated relational-slot residual to the retained v6 model. Stage B is independently authorized after an integrity-valid Stage A and adds action-conditioned JEPA plus semantic grounding. Stage C removes the old Structured Markov context only if Stage B passes its complete gate.

All stages retain task-disjoint folds, seeds 7/17/29, exact legal-action masking, the four-cell soft target, and the same AgentDojo sandbox dataset. Raw goal, observation, schema descriptions, task IDs, future fields, and outcome labels are excluded from slot inputs.
