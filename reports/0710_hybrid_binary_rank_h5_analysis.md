# 0710 Hybrid binary utility, continuous preservation, and ranking analysis

## Question

Does the evidence-driven hybrid objective improve the formal Full DreamerV3
model on the same Llama-3.1-70B trajectory split?

The tested hybrid keeps five-step imagination and changes the utility stack to:

```text
utility head       <- observed binary task success
critic reward      <- observed binary task success
preservation head  <- continuous preservation posterior
ordering           <- same-task margin ranking, margin 0.2, scale 1.0
batching            <- task-grouped batches with positive/negative interleaving
```

The selector also adds:

```text
expected_attacked_utility_score
    = clean_success_rate * preservation_score
```

and rejects validation configurations with conditional coverage below 0.5.

## Verification

- Local and server test suite: 51 passed.
- Existing continuous H5 checkpoint loads with backward-compatible defaults.
- One-epoch GPU smoke job 4231 passed end to end.
- Smoke ranking pairs: 3.28 valid pairs per optimizer update on average.
- Formal three-seed Slurm array 4232 completed for seeds 7, 13, and 21.
- No traceback, OOM, CUDA error, killed process, or runtime exception.
- Parameter count remains 5,952,588; the experiment changes losses and sampling,
  not model capacity.

Archive:

```text
/share/guozhix/wmagentattack/0710/full_dreamer_v3_hybrid_binary_rank_h5
```

## Strict frozen-threshold result

One configuration is selected by mean validation ASR+BUP across model seeds,
subject to conditional coverage >= 0.5. Each model seed's numeric validation
threshold is copied to test.

| Variant | Val-selected configuration | Test ASR | Test BUP | ASR+BUP |
|---|---|---:|---:|---:|
| sklearn | K16, utility q0.80 | 0.6250 | **0.6875** | **1.3125** |
| Full binary H5 | K16, value q0.80 | 0.6875 | 0.5833 | 1.2708 |
| Full continuous H5 | K16, utility q0.90 | **0.7083** | 0.5417 | 1.2500 |
| Full hybrid H5 | K16, utility q0.60 | 0.5625 | 0.5000 | 1.0625 |
| Full continuous H1 | K16, final utility q0.80 | 0.5833 | 0.4583 | 1.0417 |

Hybrid seed variation is high:

```text
ASR       0.5625 +/- 0.1654
BUP       0.5000 +/- 0.0625
ASR+BUP   1.0625 +/- 0.2253
```

The validation objective falls from 1.3958 on validation to 1.0625 on test.
Removing the minimum-coverage rule selects the same configuration and produces
the same test result, so the regression is not caused by coverage filtering.

## Prediction and ordering diagnostics

| Variant | Next-skill top-3 | Step utility AUC | Candidate risk AUC | Candidate utility AUC | Candidate value AUC |
|---|---:|---:|---:|---:|---:|
| Full binary H5 | 0.8552 | **0.9014** | 0.9153 | **0.8714** | **0.8850** |
| Full continuous H5 | **0.8750** | 0.7849 | **0.9206** | 0.7858 | 0.7983 |
| Full hybrid H5 | 0.7753 | 0.8438 | 0.8735 | 0.7619 | 0.8736 |

The ranking objective is active and converges: final training ranking losses are
0.0244, 0.0494, and 0.0138 for the three seeds. Nevertheless, held-out utility,
risk, and skill ordering degrade. This is training-objective improvement without
selection generalization.

The most likely mechanism is that scale-1 ranking plus task-grouped batches
changes the shared RSSM/encoder training distribution too aggressively. The
current experiment does not fully separate that effect from the preservation
auxiliary loss, so this attribution remains a hypothesis rather than a proven
ablation result.

## Weighted selector counterevidence

The hybrid is not uniformly bad. Its validation-selected weighted score gives:

| Variant | Weighted ASR | Weighted BUP | ASR+BUP |
|---|---:|---:|---:|
| Full binary H5 | 0.6458 | 0.4792 | 1.1250 |
| Full continuous H5 | 0.5833 | 0.3750 | 0.9583 |
| Full hybrid H5 | **0.7083** | **0.5208** | **1.2292** |

The binary critic reward therefore appears useful: hybrid candidate value AUC
is 0.8736, close to binary H5 at 0.8850 and much higher than continuous H5 at
0.7983. The failure is concentrated in utility-head ranking and broad Pareto
configuration transfer, not in every part of the hybrid design.

## Expected attacked utility and coverage constraint

The expected-utility transformation improves candidate BUP AUC:

| Model | Raw preservation AUC | Expected attacked utility AUC |
|---|---:|---:|
| Full continuous H5 | 0.6611 | 0.7901 |
| Full hybrid H5 | 0.5683 | 0.7764 |

For hybrid H5, pre-registering the expected-utility key and selecting only its
K/quantile on validation transfers to:

```text
ASR 0.6875 / BUP 0.5208 / ASR+BUP 1.2083
```

This is better than the globally val-selected hybrid Pareto result but remains
below Full binary H5 and sklearn. For continuous H5, the expected-utility key
transfers to only 1.0417, so higher global AUC does not automatically improve
the attack/utility top-K tradeoff.

The coverage constraint behaves as intended. Raw hybrid `preservation_score`
has no configuration reaching 0.5 conditional coverage; its maximum is 0.4583,
so it is excluded rather than producing a small-denominator headline result.

## Decision

The combined hybrid is **not useful as a replacement for the formal continuous
H5 or binary H5 model**. Its strict headline metric is lower and substantially
less stable.

Keep:

1. five-step imagination;
2. binary-dominant critic reward;
3. expected attacked utility as a diagnostic/constraint feature;
4. minimum conditional-coverage filtering;
5. the frozen val-to-test protocol.

Redesign:

1. do not group the entire world-model epoch by task;
2. do not backpropagate scale-1 ranking through the shared RSSM representation;
3. use a small separate ranking replay batch or detach latent features for the
   ranking branch;
4. reduce ranking scale to 0.1-0.25 and preservation auxiliary scale to 0.25;
5. compare `binary + preservation, no ranking` against `head-only ranking 0.1`
   before another broad hyperparameter grid.

The current formal checkpoint should remain Full continuous H5 for highest ASR,
while Full binary H5 remains the strongest Dreamer-family ASR+BUP model.

## Repo mapping

| Change | File |
|---|---|
| Hybrid utility/reward/ranking controls | `src/wmagentattack/full_dreamer_v3.py` |
| Hybrid CLI flags | `scripts/23_train_full_dreamer_v3.py` |
| Config-aware evaluation objective | `scripts/24_eval_full_dreamer_v3.py` |
| Expected utility and coverage filter | `scripts/18_pareto_utility_selection.py` |
| Strict transfer and per-key counterevidence | `scripts/25_compare_val_selected_transfer.py` |
| Formal three-seed job | `scripts/server/run_full_dreamer_hybrid_70b.sbatch` |

Machine-readable result:

```text
/share/guozhix/wmagentattack/0710/full_dreamer_v3_hybrid_binary_rank_h5/strict_val_selected_transfer.json
```
