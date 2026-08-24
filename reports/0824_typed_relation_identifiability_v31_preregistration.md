# v31 typed relation identifiability preregistration

## Question

Can a privacy-safe typed relation view distinguish the 74 frozen v29 record--goal edges from verified hard negatives, and is its held-out-fold support sufficient for another model fit?

## Single mechanism

Replace v30's raw hashed goal-term vector with typed entity, attribute, action-field, value-kind, and lexical-backoff units. Score each record--goal pair with a fixed 0.65 structural and 0.35 frozen E5 semantic mixture. E5 is revision pinned and receives only inference-visible goal/action text plus static record descriptions. Dataset outputs contain hashes, types, scores, and frozen relation labels, never raw terms or values.

Hard negatives share either the same record or the same goal unit with a positive pair and are required to be negative under the frozen v29 local-edge target. This avoids similarity-only false-negative mining.

## Budget and controls

- two byte-identical deterministic builds;
- two frozen E5 CPU passes;
- four hard negatives per positive edge;
- zero model fits, GPUs, victim-LLM calls, sandbox calls, attacks, or real endpoints;
- record-only goal-blind pair accuracy fixed at 0.5 as the representation control;
- unchanged task folds and 74 v29 positive edges.

## Frozen decision

Representation GO requires exact reconstruction, no leakage, at least 0.45 structural and typed positive coverage, at least 0.65 combined pair accuracy, at least +0.10 over the goal-blind control, and at least 0.55 in every fold. Model-fit data readiness additionally requires at least eight base positive edges in every fold.

If representation passes but fold support fails, only one 24-episode clean relation-support pilot is authorized. A small model fit is authorized only if both gates pass. No outcome can authorize attacks, large generation, Dreamer, a planner, or large training.
