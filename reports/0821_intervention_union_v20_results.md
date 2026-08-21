# Intervention-grounded modular world model v20: results

Date: 2026-08-21  
Formal job: Slurm 7196  
Archive: `/share/guozhix/wmagentattack/0821/intervention_union_v20/formal_v1`

## Scientific conclusion

The frozen v20 gate passed: `GO_INTERVENTION_MODULAR_V20`. This is evidence
that the intervention-grounded bundle is a better *small transition probe* on
the current task-disjoint v17/v18/v19 panel. It is not yet evidence that it is
the final WMagentattack world model, nor does it authorize large-scale attack
generation or planning.

The deterministic union contained 144 raw transition occurrences and 121
canonical transitions from 12 tasks. All three independent sources were
retained, two builds were byte-identical, and all task/root/pair/sequence
cross-fold and semantic-leakage findings were empty.

## Frozen comparison

| Arm | Parameters | One-step task-macro BCE ↓ | Micro F1 ↑ | Execution Brier ↓ | Pair assignment ↑ | v19 rollout BCE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Structured Markov v3 probe | 50,713 | 0.12875 | 0.87529 | 0.06294 | 0.98990 | 0.13328 |
| Structured residual v6-style probe | 52,794 | 0.07194 | 0.90971 | 0.000704 | 0.98653 | 0.06005 |
| Intervention modular v20 | 47,737 | **0.04744** | **0.95126** | **0.00000296** | **0.99327** | **0.03519** |

Relative to the v6-style recurrent baseline, v20 reduced one-step BCE by
34.1% and three-step rollout BCE by 41.4%, while using 9.6% fewer trainable
parameters. It won the rollout comparison in all three folds and all three
seeds. All 27 frozen fits completed, 15 tests passed, and no runtime failure
occurred.

## What changed in the candidate

The candidate keeps Structured Semantic State v3 as a causal input contract,
but replaces the single shared prediction path with:

1. a normalized action encoder;
2. a zero-initialized action-conditioned latent residual;
3. a separate execution-error gate;
4. success- and error-conditional semantic-effect experts;
5. same-root pair-response supervision; and
6. recurrent supervision on the frozen three-step v19 sequences.

This design is consistent with recent evidence that latent-state residuals can
adapt dynamics efficiently (ReDRAW), and that merely injecting actions is not
enough without counterfactual action-consistency constraints (CoCo). The
current dataset construction also follows the controlled-world-model
identifiability requirement that the same state must be observed under varied
actions, rather than relying only on on-policy trajectories.

Primary references:

- ReDRAW: https://proceedings.mlr.press/v331/lanier26a.html
- CoCo: https://arxiv.org/abs/2608.04653
- Controlled-world-model identifiability: https://arxiv.org/abs/2607.22430
- Executable synthetic agent environments: https://arxiv.org/abs/2602.10090

## Counterevidence and limitations

The positive result must be bounded by four observations.

First, pair assignment is almost saturated for every arm (0.987--0.993), so
the small pair gain does not establish that pair supervision is the source of
the improvement.

Second, all 121 rows contain a `source=<tool>` target token that is directly
implied by the input tool identifier. These 121 occurrences are 7.3% of the
1,654 positive target-token occurrences. Entity and attribute effects can also
be strongly correlated with tool identity. The full-vocabulary BCE therefore
contains an easy shortcut component.

Third, folds 0 and 2 contain respectively 10 and 11 positive token
occurrences that never appear in their training partitions. Because BCE is
averaged over a 120-token sparse vocabulary, those hard unseen positives can
be underweighted by the many easy negatives. A harder metric is required
before scale-up.

Fourth, v20 changes execution factorization, latent residual dynamics, pair
loss, and sequence loss together. This round validates the bundle, not the
individual causal contribution of each component. The v3 and v6 names denote
transition probes retrained on the v20 target, not reuse of old action-policy
checkpoints.

## Next authorized research direction

Do not expand the dataset or start attack/planning experiments yet. Freeze one
v21 confirmation that reuses the same 27-fit budget or less and changes only
evaluation/ablation:

1. remove mechanically implied `source=<tool>` labels from the primary metric;
2. report positive-only recall and macro metrics for unseen entity/attribute/
   conflict tokens, not only sparse full-vocabulary BCE;
3. use source-held-out or tool-family-held-out confirmation in addition to the
   existing task-disjoint folds; and
4. run a minimal component ablation: v20 full versus no pair loss versus no
   execution experts, keeping parameters and seeds fixed.

Only if the rollout advantage survives the hard-label and source-held-out
checks should v20 become the replacement for Structured Markov v3 in the
larger WMagentattack pipeline.

## Integrity

- Union SHA256: `cf5eb6fb2e92f8175e399ca1230d40d31b3d954a99b0f32258ba9789190b50ca`
- Metrics SHA256: `6c7fae00f644323cec328176eb2505e5ef7600264d393f90f0a2631ee9ba7c09`
- Gate SHA256: `cc4ccaa06cefef9f4f8a2bcbe8939997aa09479fcb3b7fa780239f61687d22b6`
- Frozen code commit: `db4435164cfbfba77af85aa2fa5bd129d5452a34`
