# Hard-label confirmation v21 preregistration

Date: 2026-08-21

The v20 gate passed, but its 120-token objective included 121 occurrences of
`source=<tool>`, each mechanically implied by the action input. v21 removes
only that label family, leaving the model inputs and all non-source semantic
effects unchanged.

The primary split remains task-disjoint by frozen difficulty fold. Two
diagnostic split suites additionally hold out query/read,
create/send/reserve, or mutation tool families, and v17, v18, or v19 source
protocols. These diagnostics are explicitly not substitutes for the primary
task-disjoint inference.

The hard view must be built twice with byte-identical outputs before a model
comparison is authorized. No attacks, victim-model calls, real endpoints,
Dreamer, planner, utility head, or result-dependent rerun is allowed.

The view gate passed before model execution: both builds are byte-identical,
all 121 rows retain positive targets, the hard vocabulary contains 94 tokens,
and all 121 source-tool occurrences were removed. Dataset SHA256 is
`5ce08331f89f7e10da7512c98a26f44a7748bc8e3e755a38cf5c69edda6a4323`.

The frozen comparison has four arms: v6-style residual, full v20, v20 without
pair loss, and v20 without success/error effect experts. The task-disjoint
primary experiment uses three folds and seeds 7/17/29. Tool-family and source
held-out diagnostics use one seed and disable sequence supervision so no
partially held-out sequence can leak a target. Total budget is exactly 60
fits.

A modular candidate must improve hard-label task-macro BCE by at least 0.01,
positive-label NLL by 0.05, and rollout BCE by 0.01 over v6; preserve positive
recall and execution calibration; remain within 0.05 NLL/recall on both
held-out diagnostics; and repeat its positive-NLL and rollout advantage in at
least two folds and two seeds. If multiple variants pass, the frozen
simplicity order chooses no-experts, then no-pair, then full v20. Unseen-token
metrics remain mandatory counterevidence but are not an acceptance threshold,
because a fixed output head cannot learn a class absent from training.
