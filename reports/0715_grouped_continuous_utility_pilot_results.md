# Grouped continuous utility pilot results

Date: 2026-07-15

Decision: **NO-GO for five-fold scaling of the coupled-batch variants.** Proceed to a decoupled utility-head-only pilot.

## Runs completed

- Slurm array `4559`: fold 1, seed 7, five epochs, detached and end-to-end variants.
- Slurm job `4561_0`: fold 1, seed 7, 30 epochs, detached variant selected by validation grouped utility Brier.
- Both jobs completed without traceback, NaN, OOM, or missing artifacts.
- Source archive: `/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1`.
- Smoke archive: `/share/guozhix/wmagentattack/0715/grouped_continuous_utility/smoke_20260715_group_utility_v1`.
- Matched-budget archive: `/share/guozhix/wmagentattack/0715/grouped_continuous_utility/pilot30_20260715_group_utility_detached_v1`.

## Five-epoch smoke

Both group losses were active and finite. The detached variant was selected on validation, but neither variant beat the 30-epoch baseline:

| Variant | Validation utility Brier change | Test Top-1 ASR+BUP change |
|---|---:|---:|
| Detached | -0.0071 (worse) | -0.25 |
| End-to-end | -0.0164 (worse) | -0.40 |

Because this compared five training epochs with a 30-epoch baseline, it was treated only as an engineering smoke and not as a method verdict.

## Matched 30-epoch detached pilot

| Metric | Baseline | Group utility detached | Change |
|---|---:|---:|---:|
| Validation grouped utility Brier | 0.0986 | 0.0957 | +0.0029 improvement |
| Test grouped utility Brier | 0.1187 | 0.1550 | -0.0363 regression |
| Test binary utility AUC | 0.6639 | 0.6784 | +0.0144 |
| Test risk AUC | 0.9158 | 0.8131 | -0.1027 |
| Top-1 ASR | 0.15 | 0.00 | -0.15 |
| Top-1 BUP | 0.30 | 0.35 | +0.05 |
| Top-1 ASR+BUP | 0.45 | 0.35 | -0.10 |

The utility ranking signal improved slightly, but the selection objective shifted toward preservation while losing attack success. This fails the formal rule requiring at least +0.05 Top-1 ASR+BUP, nonnegative BUP, and stable domains.

## Counterevidence and diagnosis

Detaching the grouped utility loss did not preserve the baseline training process. Enabling `grouped_utility_batches` replaced every normal batch with complete, task-homogeneous configuration groups. Consequently, observation, reward, continuation, KL, skill, candidate, risk, preservation, actor, and critic updates all saw a lower-diversity batch distribution. The 0.103 test risk-AUC loss is consistent with this indirect interference.

This means the result does not reject configuration-level utility supervision itself. It rejects applying all Dreamer losses on the task-grouped batches.

## Next isolated intervention

Add `group_utility_head_only_updates`:

1. Preserve baseline IID batches for all existing world-model, risk, utility, preservation, actor, and critic losses.
2. In the same optimizer step, draw one separate complete task-aware configuration batch.
3. Run its RSSM observation under a forked deterministic RNG and without gradients.
4. Apply configuration calibration and ranking gradients only to the utility head.
5. Keep the number of main optimizer steps and the shared-model RNG stream aligned with the baseline.

The next gate is a five-epoch runtime smoke followed, only if numerically valid, by one fold-1/seed-7 30-epoch matched pilot. No five-fold experiment should be launched before that comparison.

## Head-only result

The head-only runtime smoke completed as Slurm job `4562_0`. The matched 30-epoch run completed as `4563_0`, selecting epoch 29 on validation.

| Metric | Baseline | Head-only | Change |
|---|---:|---:|---:|
| Validation objective | 0.4402 | 0.4122 | -0.0280 |
| Validation utility Brier | 0.0986 | 0.0979 | +0.0008 improvement |
| Test utility Brier | 0.1187 | 0.0890 | +0.0297 improvement |
| Test binary utility AUC | 0.6639 | 0.7680 | +0.1041 |
| Test risk AUC | 0.9158 | 0.8991 | -0.0168 |
| Top-1 ASR | 0.15 | 0.30 | +0.15 |
| Top-1 BUP | 0.30 | 0.30 | 0.00 |
| Top-1 ASR+BUP | 0.45 | 0.60 | +0.15 |
| Top-2 ASR+BUP | 0.50 | 0.40 | -0.10 |
| Top-4 ASR+BUP | 0.45 | 0.4625 | +0.0125 |

This clears the primary single-run threshold while preserving BUP and largely preserving risk AUC. The negative Top-2 result is counterevidence against claiming a general ranking improvement. The method is therefore **GO only for same-fold seed replication**, not yet for five-fold training.

The method and all weights were frozen and replicated with seeds 13 and 21. The three-seed ensemble produced exactly the same Top-1 ASR, BUP, and joint score as the baseline (`0.15/0.30/0.45`), so the broader-fold gate failed. Subsequent semantic representation, dual-component value, calibration, and Dreamer integration results are recorded in `reports/0716_semantic_value_and_dreamer_results.md`.
