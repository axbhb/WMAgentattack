# Autoresearch Stage 2: Hybrid Semantic World Model

Date: 2026-08-04

Run tag: `aug4-semantic-wm-v3`

Decision: `GO__FREEZE_HYBRID_ARCHITECTURE`

## Question and fixed budget

This stage asked whether an exact observed semantic-state transition can be composed with learned victim-action/argument and candidate-conditional evidence-delta heads, while leaving completion, reward, utility/value, actor/critic, planning, attack generation, and Dreamer disabled. The budget allowed one composition and at most three non-semantic learned-head implementation repairs. No optimizer update or victim-model call was permitted.

Protocol SHA256: `8f70a6853c0a96f2f71e8b71fcaa5d1884efcbc45a94405b0311de669e30a7da`

Input dataset SHA256: `099aecb7ab70b5d822f44e9007fb95c1d6ed78569505dbc32621efdd88f82a53`

## Frozen architecture

`src/wmagentattack/hybrid_semantic_world_model.py` contains three deliberately separated components:

1. `ExactObservedSemanticTransition` validates a one-step append to execution receipts, entity evidence, conflicts, and matched goal terms. It never learns or reconstructs state.
2. The victim head scores every legal tool candidate and predicts the argument-key set.
3. The candidate-conditional evidence head predicts five next-step binary deltas: record added, new goal term matched, conflict added, execution error, and ambiguous/unlinked record added.

The learned heads share a field-specific Structured Semantic State v3 encoder and a tool-candidate encoder. A legal-action mask is mandatory. There are no completion, reward, utility, value, actor, critic, or planner modules.

## Integrity results

- Episodes/states: 144 / 467
- Exact non-terminal observed transitions: 323 / 323
- Terminal rows: 144 / 144
- Candidate tools: 75
- Argument-key vocabulary: 51
- State/candidate feature sizes: 466 / 64
- Trainable parameters: 92,729
- Finite-gradient parameter tensors: 18 / 18
- Illegal-action probability mass: exactly 0
- Leakage failures: 0
- Focused remote tests: 35 passed
- Audit rebuilds: byte-identical
- Audit SHA256: `0931d2111e3b6984b2bac7c42224b2ad02771758179acf479abfba731b76f6e1`

Remote archive:

`/share/guozhix/wmagentattack/0804/autoresearch_semantic_wm_v3/stage2`

## Counterevidence and limits

- This exact component advances only after an AgentDojo sandbox observation exists. It is exact bookkeeping/replay, not a learned generator of unseen tool outputs.
- Evidence-delta labels are strongly imbalanced: record added 322/323, new goal term matched 218/323, ambiguous/unlinked added 15/323, execution error 8/323, and conflict added 2/323. Accuracy alone would therefore be misleading.
- The reported losses are deterministic forward/backward smoke values from an untrained initialization, not predictive results.
- Architecture validity does not establish Markov sufficiency or task-disjoint generalization.

## Gate decision and authorization

All preregistered architecture clauses passed. The composition is frozen. Stage 3 is authorized to compare exactly `semantic_markov`, `structured_markov_v3`, and `full_history_diagnostic` under seeds 7, 17, and 29. Stage 3 must report proper scores and rare-label-aware evidence metrics; it may not add representations after seeing confirmation results.
