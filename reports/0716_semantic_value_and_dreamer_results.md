# AgentDojo-v2 semantic value and Dreamer integration results

Date: 2026-07-16

Decision: **NO-GO for formal multi-fold scaling of the current semantic Dreamer.** The experiments establish that semantic text features contain useful attack-success signal, but they do not yet improve preservation reliably enough to justify a larger training run.

## Fixed research budget completed

The round was stopped after the following precommitted sequence rather than after the first favorable result:

1. replicate the grouped utility-head-only result across checkpoint seeds;
2. test a direct configuration-value head inside Dreamer;
3. replace hashed text with frozen E5 plus structured features in a strict five-fold probe;
4. separate attack and preservation/value components, including pointwise, pairwise, domain-family, and hierarchical variants;
5. integrate precomputed E5 observations into the full DreamerV3 training path;
6. test train-fit, validation-selected affine-logit utility calibration;
7. test early fusion of hashed, E5, and structured observations.

No test labels were used to select a formal method. Diagnostic test-informed hybrids were kept out of the claim.

## Reference points

Two different baselines are reported because they answer different questions.

| Baseline | Scope | Top-1 ASR | Top-1 BUP | Top-1 joint |
|---|---|---:|---:|---:|
| Frozen Dreamer OOF | 20 held-out tasks, five folds | 0.23 | 0.49 | 0.72 |
| Hash Dreamer head-only | fold 1, seed 7, matched 30 epochs | 0.30 | 0.30 | 0.60 |

The first baseline is used for cross-task statistical comparisons. The second is used only for matched architecture/representation diagnostics on fold 1.

## Sequential findings

### 1. Three-seed replication removed the apparent single-seed gain

The utility-head-only model improved test utility Brier in two of three seeds and preserved mean risk AUC, but the frozen three-checkpoint ensemble had exactly the same Top-1 result as the baseline:

| Three-seed ensemble | ASR | BUP | Joint |
|---|---:|---:|---:|
| Hash baseline | 0.15 | 0.30 | 0.45 |
| Utility-head-only | 0.15 | 0.30 | 0.45 |
| Delta | 0.00 | 0.00 | 0.00 |

The per-seed joint deltas were `+0.15`, `-0.05`, and `+0.15`. This is direct counterevidence to treating seed 7 as a stable method improvement. The broader-fold gate failed.

Archive: `/share/guozhix/wmagentattack/0715/grouped_continuous_utility/seedrep_20260715_group_utility_head_only_v1`.

### 2. Direct configuration-value supervision was not sufficient

Adding a configuration-value head directly to the hashed Dreamer reduced fold-1 Top-1 joint performance from `0.45` to `0.20` and did not produce useful held-out ranking. This rejected adding another scalar head without first improving representation and target factorization.

Archive: `/share/guozhix/wmagentattack/0715/configuration_value/pilot30_20260715_configuration_value_v1`.

### 3. Frozen E5 plus structured fields found attack signal, but lost preservation

The frozen method was `E5-base-v2 query embedding (768) + structured hash features (32) -> pairwise ridge`. E5 recommends a `query:` prefix for feature/linear-probe use; the model is English-only and truncates at 512 tokens ([paper](https://arxiv.org/abs/2212.03533), [model card](https://huggingface.co/intfloat/e5-base-v2)).

| Five-fold OOF | ASR | BUP | Joint |
|---|---:|---:|---:|
| Frozen Dreamer baseline | 0.23 | 0.49 | 0.72 |
| E5 + structured ridge | 0.32 | 0.46 | 0.78 |
| Delta | +0.09 | -0.03 | +0.06 |

The joint bootstrap 95% interval was `[-0.07, 0.21]`, and the exact one-sided sign-flip p-value was `0.2646`. Domain joint deltas were banking `+0.16`, slack `+0.20`, travel `0.00`, and workspace `-0.12`. The method therefore failed both the nonnegative-BUP and worst-domain gates despite the best aggregate point estimate.

Archive: `/share/guozhix/wmagentattack/0715/semantic_value_probe/frozen_e5_structured_oof_components_20260716_v2`.

### 4. Explicit dual-component value models did not repair BUP transfer

The frozen pointwise dual model used a pairwise attack head and a pointwise utility head with the validation-selected recipe `attack + 2 * utility`.

| Five-fold OOF | ASR | BUP | Joint | Joint delta |
|---|---:|---:|---:|---:|
| Dual-component E5 model | 0.27 | 0.48 | 0.75 | +0.03 |

Its joint interval still crossed zero, the sign-flip p-value was `0.3594`, and workspace remained negative (`-0.08`). Pairwise utility, domain-family interaction, and hierarchical shrinkage discovery variants produced fold-1 joint scores of `0.60`, `0.50`, and `0.60`; none beat the frozen gate.

Archive: `/share/guozhix/wmagentattack/0715/dual_component_value/dual_component_pointwise_frozen_oof_20260716_v1`.

### 5. E5 observations are now supported by the full Dreamer path

The implementation now supports a strict precomputed-observation mode. The E5 cache contains 4,749 unique observations over 4,564 unique texts and occupies about 13.6 MB. Each observation is 800-dimensional (`768 E5 + 32 structured`); missing or mismatched cache entries fail closed. The resulting Dreamer has about 5.97 million trainable parameters.

The matched 30-epoch run selected epoch 5, so additional epochs did not solve transfer:

| Fold-1 metric | Hash head-only | E5 Dreamer |
|---|---:|---:|
| Test grouped utility Brier | 0.0890 | 0.1647 |
| Test utility AUC | 0.7680 | 0.7946 |
| Test risk AUC | 0.8991 | 0.9329 |
| Top-1 joint | 0.60 | 0.60 |
| Top-2 joint | 0.40 | 0.575 |
| Top-4 joint | 0.4625 | 0.525 |

E5 improves discrimination and broader-budget selection, but its utility probabilities transfer much worse. It ties rather than beats Top-1, which is the primary endpoint.

Archive: `/share/guozhix/wmagentattack/0715/semantic_dreamer/semantic_dreamer_pilot30_20260716_v2`.

### 6. Scalar utility calibration was not the missing component

Affine-logit calibrators with regularization `0`, `0.001`, `0.01`, `0.1`, and `1` were fitted on train and selected only by validation. Validation selected identity (`0.0891` Brier), so test Brier remained `0.1647` and downstream selections were unchanged. The failure is task/domain-conditional rather than a single global scale or bias error.

### 7. Hash + E5 early fusion did not pass the smoke gate

The early-fusion observation was 1,568-dimensional (`768 E5 + 768 hash + 32 structured`) and produced a 6.36-million-parameter Dreamer. At its best smoke checkpoint:

| Fold-1 metric | Hash | E5 only | Hash + E5 |
|---|---:|---:|---:|
| Test grouped utility Brier | 0.0890 | 0.1647 | 0.1426 |
| Test utility AUC | 0.7680 | 0.7946 | 0.7098 |
| Test risk AUC | 0.8991 | 0.9329 | 0.9456 |
| Top-1 joint | 0.60 | 0.60 | 0.60 |
| Top-2 joint | 0.40 | 0.575 | 0.55 |
| Top-4 joint | 0.4625 | 0.525 | 0.50 |

Fusion partially repairs calibration and improves risk AUC, but destroys the E5 utility-ranking advantage and does not improve Top-1. A 30-epoch repeat was therefore not justified.

Archive: `/share/guozhix/wmagentattack/0715/semantic_dreamer/hash_e5_structured_dreamer_smoke5_20260716_v3`.

## What the evidence says

The main bottleneck is no longer whether the model can recognize attack text. E5 improves ASR and risk discrimination. The unstable quantity is preservation: observed BUP mixes the intrinsic solvability of a user task with the extra damage caused by a particular attack. A single utility target asks one head to learn both effects from only 20 held-out tasks, and workspace shows that this does not transfer.

The next method should therefore factor preservation rather than add another generic head:

1. estimate a cross-fitted clean-task solvability prior from clean multi-seed runs;
2. train a residual head for attack-induced preservation change conditional on task representation, attack family, and trajectory state;
3. use a beta-binomial posterior (or an equivalent repeated-trial likelihood) so five-seed uncertainty remains explicit;
4. rank with predicted ASR plus a conservative posterior estimate of BUP;
5. freeze the recipe on validation and evaluate over all five task-held-out folds.

Formal scaling should resume only if this residual model gives at least `+0.05` Top-1 joint, nonnegative BUP, a joint confidence interval excluding zero, and no domain below `-0.10` relative to the frozen Dreamer baseline.

## Reproducibility status

- Remote Slurm queue was empty after the final smoke run.
- All listed archives exist and contain model/evaluation artifacts.
- The semantic observation loader, cache builder, value probes, fold summaries, and calibration path have focused unit tests.
- No large formal run is currently queued because the predeclared gate was not met.
