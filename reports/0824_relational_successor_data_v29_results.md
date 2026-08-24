# v29 Relational Successor Data — Results

## Decision

`GO_RELATIONAL_SUCCESSOR_MODEL_V29` (19/19 frozen clauses).

This is a data-identifiability GO. It authorizes one small relational successor-model comparison; it does **not** establish predictive improvement and does not authorize large-scale generation, attacks, Dreamer, or a planner.

## Formal run

- Remote Slurm job: `7318`, `COMPLETED`, exit `0:0`.
- Resource contract: CPU only, two cores, 8 GB; zero GPUs, LLM calls, sandbox calls, attacks, or real endpoints.
- Slurm wall time: 25 s; two builds plus gate: 6 s.
- Tests: 2 passed; stdout and stderr were both empty.
- Two 2,113,326-byte datasets and their audits were byte-identical.

## Exact findings

- Preserved 121 confirmation transitions from 12 tasks and 10 support transitions from four disjoint tasks.
- Every global newly matched goal pointer equals the union of pointers bound inside its newly added evidence records: zero relation errors.
- 24 newly added records carry positive record-to-goal edges; one record binds as many as 16 current-goal terms.
- Static, outcome-blind contracts produced 133 typed record candidates for 55 tools, including `webpage`.
- Exact candidate coverage was 1.0 for every task fold, every tool-family split, and all three source-heldout splits. No signature was missing.
- A post-gate countercheck also found 196/196 record occurrences covered by the candidate set of their own action tool, rather than merely by the global inventory.
- No semantic-input leakage, raw-value/raw-term target leakage, outcome-label contamination, task overlap, runtime failure, or missing artifact was found.

## Comparison with v28

v28's learned whole-record mechanism achieved 0.8308 record F1 and 0.7460 task-disjoint unseen recall, but goal-pointer F1 was only 0.1717. Its exact candidate coverage was 0.4389 on tool-family-heldout data and 0.5690 on source-heldout data; webpage focused recall was zero.

v29 fixes the two corresponding prerequisites before another fit:

1. Goal progress is no longer a free global pointer set; pointers are attached to the evidence record that supplies them.
2. Candidate availability is no longer inferred from training outcomes; it is generated from frozen tool adapters and static AgentDojo return schemas.

## Counterevidence and limitation

- Most of the 196 added-record occurrences do not introduce a new goal term. They remain necessary negative relation examples, but severe relation sparsity may still hurt calibration.
- The static inventory deliberately includes all three link states for each successful schema. Coverage is complete, but this overcomplete set may increase false positives unless the next model conditions candidates on the selected tool and action arguments.
- No model was trained in v29, so none of v28's NLL, precision, webpage recall, or cross-source generalization failures has yet been resolved empirically.

## Retained method and next authorized experiment

Retain fixed v21 as the canonical effect control, v22 recurrent supervision as local multi-step evidence, and v28's whole-record scorer only as a mechanism component. The next experiment should replace v28's separate record and global-pointer heads with a joint relation scorer over `(action-conditioned record candidate, current goal term)` edges, while preserving the same task/tool/source splits, fixed v21 control, zero-start residual dynamics, deterministic matched-count rendering, and v28 non-inferiority gates.
