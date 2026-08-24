# Explicit atom-support data gate v25

## Frozen question

Can ten clean, outcome-blind branches on four sibling tasks expose the entity,
field, kind, and operation atoms missing from the v21 task-disjoint hard-label
folds, without making the exact composite confirmation labels available to the
effect head?

## Motivation and counterevidence

v24 failed despite a strong semantic encoder: unseen recall was 0.2619, false
positive rate 0.0762, precision 0.1349, and rollout BCE 0.0341. The pre-run
audit showed zero exact entity support and 18.75% exact field support. Earlier
v17--v19 experiments established that same-root action, parameter, and
persistence interventions are deterministic, but they used only the twelve
confirmation tasks. This round changes the data support, not the encoder.

The support panel targets `get_balance`, `send_direct_message`,
`get_users_in_channel`, `get_webpage`, and `create_file` on task-disjoint
sibling roots. Each tool is executed at two distinct canonical prefixes and
twice from fresh state. Full composite effects are retained only below an
`audit_only` boundary. A future loader may consume only factorized slot atoms.

## Fixed budget and gate

- 10 manifest rows, 20 fresh branch calls, no LLM/GPU/attack/model/Dreamer.
- Two independent manifest builds must be byte-identical.
- All rows must execute successfully with identical replicas and no leakage.
- Entity, field, kind, and operation coverage on the frozen unseen hard effects
  must each reach 1.0 while at least 16 unseen positive occurrences remain.
- Every clause is required. Failure is a completed data result and does not
  authorize model training.

Primary-method mapping: controlled-world-model identifiability requires
non-degenerate conditional action variation; CoCo warns that action features
without counterfactual consistency preserve shortcuts; executable synthetic
environments provide reliable transitions. Repository consequence: collect
matched executable support before changing the latent model again.
