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
