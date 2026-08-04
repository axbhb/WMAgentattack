# 0710 Hybrid-lite utility-preservation round

## Objective

The previous scale-1 hybrid used task-grouped world-model batches and allowed
ranking gradients to update the shared RSSM representation. It reduced strict
ASR+BUP to 1.0625. This round tests whether a lighter intervention can preserve
the useful binary critic reward without damaging dynamics learning.

Two three-seed variants use the same Llama-3.1-70B trajectories, Full
DreamerV3 architecture, five-step imagination, validation split, and frozen
val-to-test protocol:

```text
hybrid_lite_norank:
  binary utility head and critic reward
  continuous preservation auxiliary scale 0.25
  random world-model batches
  no ranking

hybrid_lite_headrank:
  same base model
  head-only ranking scale 0.1
  ranking latent is detached
  one explicit positive/negative pair injected per random batch
```

## Verification

- Local and server tests: 52 passed.
- Head-ranking smoke job 4236 completed end to end.
- Smoke ranking pairs: 1.13 pairs per optimizer update.
- Formal six-task Slurm array 4237 completed.
- No traceback, OOM, CUDA error, killed process, or runtime exception.
- Formal head-ranking pair counts: 1.18-1.20 per update.

Archive:

```text
/share/guozhix/wmagentattack/0710/full_dreamer_v3_hybrid_lite
```

## Strict frozen-threshold comparison

| Variant | Val-selected configuration | ASR | BUP | ASR+BUP |
|---|---|---:|---:|---:|
| sklearn | K16, utility q0.80 | 0.6250 | **0.6875** | **1.3125** |
| Full binary H5 | K16, value q0.80 | 0.6875 | 0.5833 | 1.2708 |
| Full continuous H5 | K16, utility q0.90 | **0.7083** | 0.5417 | 1.2500 |
| Hybrid-lite no ranking | K16, final utility q0.80 | 0.5833 | 0.6667 | 1.2500 |
| Hybrid-lite head ranking | K16, final utility q0.90 | 0.5625 | 0.5000 | 1.0625 |
| Previous scale-1 hybrid | K16, utility q0.60 | 0.5625 | 0.5000 | 1.0625 |

Hybrid-lite no-ranking uncertainty:

```text
ASR       0.5833 +/- 0.0955
BUP       0.6667 +/- 0.0361
ASR+BUP   1.2500 +/- 0.1083
```

It provides a new utility-preserving point on the Dreamer Pareto frontier. It
does not improve total ASR+BUP over continuous H5, but increases BUP by 0.1250
while reducing ASR by 0.1250. Relative to binary H5, it increases BUP by 0.0834
and decreases ASR by 0.1042.

## Conditional preservation

Hybrid-lite no-ranking reaches:

```text
conditional coverage   0.6667
conditional ASR        0.5333
conditional BUP        0.8121
```

This is much stronger utility preservation than continuous H5 conditional BUP
0.5861, but sklearn remains stronger at conditional ASR 0.5833, BUP 0.8333,
and coverage 0.7500.

## Prediction and candidate ordering

| Variant | Step top-3 | Step utility AUC | Candidate risk AUC | Candidate utility AUC | Candidate value AUC |
|---|---:|---:|---:|---:|---:|
| Full binary H5 | 0.8552 | 0.9014 | 0.9153 | 0.8714 | **0.8850** |
| Full continuous H5 | **0.8750** | 0.7849 | **0.9206** | 0.7858 | 0.7983 |
| Hybrid-lite no ranking | 0.8572 | 0.8964 | 0.9167 | **0.8141** | 0.8735 |
| Hybrid-lite head ranking | 0.8709 | **0.9030** | 0.8484 | 0.7621 | 0.8307 |

The head-ranking variant improves held-out step metrics, including utility AUC
and Brier, without collapsing next-skill accuracy. However, imagined candidate
risk, utility, and value ordering all degrade. Its q0.90 final-utility
thresholds saturate near 0.97-1.00, and strict selection is unstable.

The ranking loss is active and converges from 0.34-0.50 to 0.02-0.04. The
failure is therefore a mismatch between step-level ranking and imagined
trajectory selection, not an inactive loss or shared-latent damage.

## Cross-model ensemble countercheck

An inference-free ensemble combines continuous H5 risk/target predictions with
hybrid-lite no-ranking utility/value predictions. Validation still selects K16
final utility q0.80. Frozen test performance is:

```text
ASR 0.5833 / BUP 0.6458 / ASR+BUP 1.2292
```

This is dominated by hybrid-lite no-ranking itself. Its weighted result is
0.6667 / 0.5417 / 1.2083, also below the best strict baselines. The missing gain
cannot be recovered by simply splicing the strongest existing risk and utility
heads.

## Decision

Useful:

1. binary utility/critic reward;
2. five-step imagination;
3. preservation auxiliary at 0.25;
4. random world-model batches;
5. hybrid-lite no-ranking as the BUP-oriented Dreamer checkpoint;
6. coverage-aware reporting and frozen thresholds.

Not useful:

1. scale-1 ranking through shared latent features;
2. detached step-head ranking at scale 0.1;
3. explicit pair injection for a step-level utility head;
4. simple continuous-risk/no-ranking-utility score splicing.

The method now has three non-dominated Dreamer operating points:

| Operating point | Recommended checkpoint | ASR | BUP |
|---|---|---:|---:|
| attack-oriented | Full continuous H5 | 0.7083 | 0.5417 |
| balanced | Full binary H5 | 0.6875 | 0.5833 |
| utility-oriented | Hybrid-lite no ranking | 0.5833 | 0.6667 |

The next ranking method should be a separate candidate/trajectory-level head
trained on imagined rollout summaries. It should not backpropagate into the
world model. Training/selection must use grouped cross-validation on validation
candidates, followed by one frozen test report. Relevant features are predicted
risk, utility, final utility, value, target probability, clean solvability, and
rollout skill statistics.

## Reproduction

- `src/wmagentattack/full_dreamer_v3.py`
- `scripts/23_train_full_dreamer_v3.py`
- `scripts/18_pareto_utility_selection.py`
- `scripts/25_compare_val_selected_transfer.py`
- `scripts/server/run_full_dreamer_hybrid_lite_70b.sbatch`
- `/share/guozhix/wmagentattack/0710/full_dreamer_v3_hybrid_lite/hybrid_lite_norank/strict_val_selected_transfer.json`
- `/share/guozhix/wmagentattack/0710/full_dreamer_v3_hybrid_lite/hybrid_lite_headrank/strict_val_selected_transfer.json`
- `/share/guozhix/wmagentattack/0710/full_dreamer_v3_hybrid_lite/ensemble_continuous_risk_norank_utility/strict_val_selected_transfer.json`
