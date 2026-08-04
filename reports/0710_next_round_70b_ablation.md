# 0710 70B Full DreamerV3 label and imagination ablation

## Scope and protocol

This round tests the two main explanations left open by the formal continuous-
probability run:

1. whether Full DreamerV3 beats the TF-IDF + logistic-regression baseline on
   the same Llama-3.1-70B trajectories;
2. whether the continuous utility target and five-step imagination improve
   strict validation-to-test selection.

All learned selectors use the same candidate pool, task split, `K` grid, utility
quantiles, clean-solvability annotations, and Pareto selector. A single shared
configuration is selected by mean validation ASR+BUP. Each model seed's numeric
validation threshold is then frozen and applied to test. No test-selected
configuration is used in the headline results.

Slurm array: `4223`

Archive:

```text
/share/guozhix/wmagentattack/0710/next_round_70b_ablation
```

All seven array tasks completed. The logs contain no traceback, OOM, CUDA
error, killed process, or runtime exception.

## Variants

| Variant | Model seeds | Utility target | Actor/critic imagination |
|---|---:|---|---:|
| sklearn | deterministic | observed binary task success | heuristic text rollout |
| Full binary H5 | 7, 13, 21 | observed binary task success | 5 steps |
| Full continuous H5 | 7, 13, 21 | posterior probability target | 5 steps |
| Full continuous H1 | 7, 13, 21 | posterior probability target | 1 step |

The Full binary model uses the same underlying 70B trajectories and the same
Full DreamerV3 architecture as the formal model. The preservation head has no
continuous target in this ablation and is therefore excluded from its Pareto
utility-key grid.

## Strict frozen-threshold selection

| Variant | Val-selected configuration | Test ASR | Test BUP | ASR+BUP | Conditional coverage |
|---|---|---:|---:|---:|---:|
| sklearn | K16, utility q0.80 | 0.6250 | **0.6875** | **1.3125** | 0.7500 |
| Full binary H5 | K16, value q0.80 | 0.6875 +/- 0.0000 | 0.5833 +/- 0.0361 | 1.2708 +/- 0.0361 | 0.6458 |
| Full continuous H5 | K16, utility q0.90 | **0.7083 +/- 0.0361** | 0.5417 +/- 0.1301 | 1.2500 +/- 0.1250 | 0.7292 |
| Full continuous H1 | K16, final utility q0.80 | 0.5833 +/- 0.1443 | 0.4583 +/- 0.1573 | 1.0417 +/- 0.2954 | 0.7083 |

ASR+BUP is a sum of two rates and can exceed one.

The current Full continuous H5 model remains the strongest attack selector,
but it does not beat sklearn on BUP or total ASR+BUP. Full binary H5 closes most
of that gap and has much lower seed variance than either continuous model.

## Candidate-level ordering

| Variant | Attack-risk AUC | Utility AUC | Value AUC |
|---|---:|---:|---:|
| sklearn | 0.8994 | **0.8960** | n/a |
| Full binary H5 | 0.9153 | 0.8714 | **0.8850** |
| Full continuous H5 | 0.9206 | 0.7858 | 0.7983 |
| Full continuous H1 | **0.9312** | 0.7175 | 0.6405 |

Continuous labels preserve strong attack-risk ordering but substantially weaken
utility and value ordering. The binary Full model selects `value_score`, while
the formal continuous model selects `utility_score`; this is evidence that the
binary utility reward trains a more useful critic for constrained selection.

## Step-level metrics

| Variant | Next-skill top-3 | Risk AUC | Binary utility AUC |
|---|---:|---:|---:|
| sklearn | **0.9078** | 0.9073 | **0.9131** |
| Full binary H5 | 0.8552 | 0.9041 | 0.9014 |
| Full continuous H5 | 0.8750 | **0.9139** | 0.7849 |
| Full continuous H1 | 0.8709 | 0.9110 | 0.8056 |

The one-step model has slightly better step-level binary utility AUC than the
five-step continuous model, yet much worse candidate value AUC and strict
selection. Step prediction alone therefore does not explain selection quality;
multi-step behavior/value learning contributes useful information.

## Counterevidence and limitations

1. sklearn is still the strongest total-utility baseline on this exact 70B
   split. The Full DreamerV3 result cannot yet be presented as an across-the-
   board baseline win.
2. sklearn is deterministic and has only one model fit; Full models report three
   initialization seeds. Final uncertainty should use paired bootstrap or
   repeated task splits rather than treating the sklearn point estimate as
   variance-free.
3. Horizon one is a weakened-imagination ablation, not a true no-actor/critic
   ablation. It supports keeping multi-step imagination but does not isolate
   every actor/critic component.
4. Full continuous H1 selected epochs 29/30, 30/30, and 30/30. Extra epochs may
   improve its validation objective, but its candidate utility/value ordering
   is already far below H5 and is not the highest-value next experiment.
5. Continuous-probability Brier scores and binary Brier/AUC values target
   different labels and must not be compared as if they measured the same
   quantity.

## Decision

Keep five-step imagination. Do not continue with a pure continuous utility
target as the sole utility/value signal.

The next model should use a hybrid objective:

```text
utility/value target: observed binary task success
preservation target: continuous posterior probability (auxiliary)
ordering target: same-task pairwise utility/preservation ranking
critic reward: binary utility or a binary-dominant binary/soft mixture
```

This preserves the binary model's candidate utility/value ordering, retains the
continuous head's uncertainty representation, and keeps the five-step model's
stronger and more stable attack selection.

## Repo mapping for the next implementation

| Finding | Repo change |
|---|---|
| Continuous target hurts ordering | Add binary + soft + pairwise hybrid losses in `src/wmagentattack/full_dreamer_v3.py`. |
| Critic is better with binary utility | Add separate world-head and reward-target controls to `scripts/23_train_full_dreamer_v3.py`. |
| Raw preservation ratio mismatches BUP | Add `clean_rate * preservation_score` and minimum-coverage constraints to `scripts/18_pareto_utility_selection.py`. |
| Strict transfer must remain the headline | Reuse `scripts/25_compare_val_selected_transfer.py` for every future variant. |
| Simple baseline remains strong | Keep `src/wmagentattack/world_model.py` in every formal comparison. |

## Reproduction files

- `scripts/server/run_next_round_70b_ablation.sbatch`
- `scripts/25_compare_val_selected_transfer.py`
- `/share/guozhix/wmagentattack/0710/next_round_70b_ablation/sklearn/strict_transfer.json`
- `/share/guozhix/wmagentattack/0710/next_round_70b_ablation/full_binary/strict_transfer.json`
- `/share/guozhix/wmagentattack/0710/next_round_70b_ablation/full_horizon1/strict_transfer.json`
- `/share/guozhix/wmagentattack/0710/full_dreamer_v3_llama31_70b/strict_val_selected_transfer.json`
