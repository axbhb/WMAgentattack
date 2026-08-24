# Explicit atom-support data gate v25: results

Date: 2026-08-24  
Decision: `GO_SUPPORT_CONDITIONED_MODEL_V25`

## Scientific conclusion

The direction change after v24 is validated at the data-gate level. Ten
clean-only sibling-task branches supplied all missing entity, field, kind, and
operation atoms for the frozen task-disjoint hard effects, while the exact
composite effect labels remained disabled in the model-facing support target.

This authorizes one small, separately frozen support-conditioned world-model
comparison. It does not show that the new model beats Structured Markov and
does not authorize attacks, a planner, Dreamer, LLM trajectory generation, or
dataset scale-up.

## Frozen execution and integrity

- Slurm job `7305`, completed in 9 seconds on CPU.
- 10 manifest rows, 20 fresh branch calls, 42 prefix replay calls, 62 total
  AgentDojo sandbox calls.
- 4 support tasks, disjoint from all 12 confirmation tasks.
- Five target tools, exactly two roots per tool.
- 10/10 branches succeeded; every replica pair was identical.
- 21 tests passed; all pre/post hashes passed; no runtime, replay, schema, or
  semantic-leakage failure occurred.
- Victim LLM, GPU, attack, model, Dreamer, and real-endpoint calls were zero.

The reused v17 collector reports its legacy `all_12_suite_difficulty_cells`
diagnostic as false because v25 intentionally targets only four rare-mechanism
cells. This clause was explicitly excluded before execution; every applicable
runtime-integrity check passed. It is not a scientific failure.

## Exact coverage gate

| Support dimension | Denominator | Coverage | Gate |
|---|---:|---:|---|
| entity | 9 unseen occurrences | 1.000 | pass |
| field | 5 unseen occurrences | 1.000 | pass |
| kind | 5 unseen occurrences | 1.000 | pass |
| operation/tool | 9 unseen occurrences | 1.000 | pass |

All 16 frozen unseen positive occurrences remain unseen to the ordinary
composite effect head. The support loader exposes only factorized atoms such as
`entity::webpage`, `field::content`, and `kind::SINGLE_VALUED`; full tokens such
as `attribute=webpage::content::SINGLE_VALUED` are retained below an
`audit_only` boundary.

## Counterevidence

Seven unseen occurrences are the scalar token `matched_count=3`. They do not
have entity or field slots and therefore are not evidence for or against the
atom-support gate. They remain a real prediction problem. The next model must
use a dedicated ordinal/exact matched-count head and report this metric
separately; it may not hide these seven cases inside a sparse composite BCE.

The support panel is small and deliberately targeted. Perfect coverage does
not prove predictive gain, calibration, or rollout fidelity. A model may still
overfit tool identity, so the next comparison must retain task-disjoint folds,
same seeds, v21/v23 controls, open-set precision/FPR, and the frozen v22 rollout
non-inferiority check.

## Authorized next architecture test

Freeze one comparison with:

1. the unchanged v21 Structured Markov probe;
2. the unchanged v23 E5 diagnostic;
3. a support-conditioned modular candidate that keeps Structured Semantic
   State v3 and v22 zero-start recurrent dynamics, adds entity/field/kind atom
   heads trained on v25 support rows, composes exact effects without reading
   support composite labels, and predicts `matched_count` with a separate
   ordinal head.

Only a task-disjoint gain in unseen recall and precision, with non-inferior
one-step and rollout BCE, can authorize a larger support dataset.

## Literature-to-repository mapping

- Controlled-world-model identifiability links counterfactual reliability to
  conditional action coverage; v25 supplies same-environment rare-action
  support rather than another encoder.
- CoCo shows that action injection alone can preserve statistical shortcuts;
  v25 keeps paired action roots and the next model must retain counterfactual
  consistency supervision.
- Agent World Model emphasizes executable, database-backed synthetic
  transitions; v25 uses only deterministic AgentDojo sandbox execution.

Primary sources:

- https://arxiv.org/abs/2607.22430
- https://arxiv.org/abs/2608.04653
- https://arxiv.org/abs/2602.10090

## Frozen artifacts

- Archive: `/share/guozhix/wmagentattack/0824/explicit_atom_support_v25/formal_v1`
- Manifest SHA256: `9bd4df644642e4b3e4886dc764294c1eb42f573039aa0679e8cb67a07d893678`
- Support dataset SHA256: `459d57cd3a75ad24709ee0a29569c1941bed4b1ac20f0cf6ccc3f8d83224bdfb`
- Gate SHA256: `ee80151926b1b9bb38574829c1cea090c606a46be84d7283072f106b7e3a4075`
- Archive checksum-list SHA256: `0975febe8f5e7ced58648a72e6326c45d7b5de1fdc505693c59e2f063bbf9266`
