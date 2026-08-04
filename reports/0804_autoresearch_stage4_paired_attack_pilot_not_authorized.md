# Autoresearch Stage 4: Paired attack pilot status

Date: 2026-08-04

Run tag: `aug4-semantic-wm-v3`

Decision: `NOT_AUTHORIZED__STAGE3_EVIDENCE_SUFFICIENCY_GATE_FAILED`

## Conditional design

Stage 4 was preregistered as conditional. It could freeze durable task IDs and construct a small same-task/same-seed AgentDojo sandbox pilot only if Stage 3 established both victim-action and evidence-dynamics sufficiency.

Stage 3 returned `NO_GO__STRUCTURED_MARKOV_V3_SUFFICIENCY_NOT_ESTABLISHED`. The victim-action clauses passed, but the evidence mean-gain and seed-replication clauses failed. The authorization condition is therefore false.

## Completed action

The conditional gate was evaluated and recorded without executing the pilot:

- durable task IDs frozen: 0
- attack examples generated: 0
- paired clean/attack executions: 0
- victim-model calls: 0
- external endpoint calls: 0
- Dreamer/value/planner runs: 0

This is a completed scientific outcome, not a runtime failure. Generating attacks anyway would invalidate the four-stage autoresearch contract and confound representation sufficiency with attack-data quality.

## Binding next boundary

No attack selector or Dreamer experiment is authorized by this loop. A future, separately preregistered loop should first redesign and expand clean evidence dynamics: more independent task identities; many more contradiction, error, and ambiguity transitions; non-degenerate labels; a relational candidate-by-constraint representation; head-specific encoders; and a genuinely fresh confirmation set. Only a successful clean gate may reopen the paired sandbox pilot.
