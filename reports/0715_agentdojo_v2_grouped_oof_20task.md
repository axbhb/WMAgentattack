# AgentDojo-v2 20-task grouped OOF data-scale experiment

Date: 2026-07-15

## Decision

The fixed-budget 20-task grouped five-fold OOF experiment is complete. Expanding the label-blind training subset from 25% to 100% improves risk discrimination and next-skill prediction, and raises the pre-registered Top-1 downstream objective from 0.62 to 0.72. However, the paired expansion effect is not statistically resolved (`delta=+0.10`, task bootstrap 95% CI `[-0.04, 0.25]`, exact one-sided sign-flip `p=0.1328125`) and reverses slightly at Top-2 and Top-4.

The appropriate decision is **PARTIAL GO for representation learning, NO-GO for a scale-only downstream claim**. More data helps the attack-risk head, but does not yet solve utility/value estimation or stable configuration selection.

## Fixed protocol and completed budget

- 20 user tasks: five cohorts, each containing one banking, Slack, travel, and workspace task.
- Five grouped OOF folds. Each fold has 12 train tasks, 4 validation tasks, and 4 test tasks.
- The 25% condition retains all clean groups and five label-blind, family-interleaved attack groups per train task. The 100% condition uses all available train attack groups.
- Three checkpoint seeds per fold and size: 7, 13, and 21.
- 30 logical checkpoints (`5 folds x 2 sizes x 3 seeds`): 6 fold0 checkpoints were reused byte-for-byte from the earlier run and 24 were newly trained.
- Validation-only selection of calibration and scoring recipe.
- First-step decision state only; no final-step information is used.
- Primary budget: Top-1 per held-out task. Top-2 and Top-4 are secondary.
- Calibrated three-seed ensemble.
- 200,000 task-balanced randomization draws and task bootstrap draws.
- Primary scale contrast cross-checked with exact sign-flip enumeration.
- No model, recipe, threshold, or endpoint was changed after observing fold results.

Slurm jobs:

- Training: `4507`, `4512`, `4518`
- Frozen first-step selection: `4531`

Completion audit:

- 30/30 checkpoints have `model.pt`, readable validation metrics, and readable test metrics.
- 10/10 fold-size prospective results are readable and contain Top-1/2/4 outputs.
- No Traceback, runtime error, CUDA OOM, or incomplete array element was found.
- Local regression suite: 12/12 tests passed.

## Prospective OOF outcome

`ASR+BUP` is averaged over the 20 independently held-out user tasks. The random column uses task-balanced random selection from the same candidate pools.

| Train size | Budget | ASR | BUP | ASR+BUP | Task bootstrap 95% CI | Random mean | Raw random p | Holm p over 3 budgets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 1 | 0.1300 | 0.4900 | 0.6200 | [0.4000, 0.8400] | 0.6084 | 0.4460 | 0.4460 |
| 100% | 1 | 0.2300 | 0.4900 | 0.7200 | [0.4700, 0.9800] | 0.6084 | 0.0440 | 0.1321 |
| 25% | 2 | 0.1850 | 0.5050 | 0.6900 | [0.4650, 0.9150] | 0.6086 | 0.0290 | 0.0871 |
| 100% | 2 | 0.1850 | 0.4800 | 0.6650 | [0.4400, 0.8900] | 0.6086 | 0.1025 | 0.2050 |
| 25% | 4 | 0.1675 | 0.4875 | 0.6550 | [0.4400, 0.8700] | 0.6085 | 0.0543 | 0.1086 |
| 100% | 4 | 0.1675 | 0.4650 | 0.6325 | [0.4225, 0.8400] | 0.6085 | 0.2079 | 0.2079 |

Because Top-1 was pre-registered as the primary endpoint, the full-data policy's raw `p=0.0440` against random is interpretable as the primary random-baseline comparison. It is not evidence that 100% is better than 25%; that claim requires the paired scale contrast below.

## Paired effect of expanding 25% to 100%

| Budget | Delta ASR | Delta BUP | Delta ASR+BUP | 95% CI for joint delta | Exact sign-flip p | Positive / tie / negative tasks |
|---|---:|---:|---:|---:|---:|---:|
| 1 | +0.1000 | 0.0000 | +0.1000 | [-0.0400, 0.2500] | 0.132812 | 7 / 10 / 3 |
| 2 | 0.0000 | -0.0250 | -0.0250 | [-0.1250, 0.0750] | 0.714478 | 6 / 6 / 8 |
| 4 | 0.0000 | -0.0225 | -0.0225 | [-0.0950, 0.0475] | 0.745667 | 9 / 3 / 8 |

The Top-1 gain comes entirely from ASR; BUP is unchanged. At larger budgets, ASR is unchanged and BUP is slightly worse. Therefore the current evidence does not support a general claim that adding data improves the joint attack-and-preservation policy.

## World-model transfer metrics

The table averages the 15 paired fold-seed test runs per train size. For Brier scores, lower is better.

| Metric | 25% | 100% | 100%-25% | Runs improved by 100% |
|---|---:|---:|---:|---:|
| Risk probability Brier | 0.03821 | 0.03235 | -0.00587 | 11/15 |
| Risk AUC | 0.81575 | 0.85545 | +0.03970 | 12/15 |
| Next-skill accuracy | 0.32558 | 0.35740 | +0.03182 | 11/15, 1 tie |
| Utility probability Brier | 0.08076 | 0.08114 | +0.00038 | 6/15; worsened in 9/15 |
| Preservation probability Brier | 0.14245 | 0.13835 | -0.00410 | 9/15 |

First-step ridge calibration (`reg0`) was selected in all five folds for both sizes. Its validation risk Brier averaged 0.03837 at 25% and 0.03604 at 100%, versus identity-calibration values of 0.04918 and 0.04309 respectively.

This is internally consistent: data scale helps the risk and skill heads, while the utility head remains the bottleneck.

## Counterevidence and heterogeneity

### The earlier four-task gain was real but unrepresentative

Fold0 exactly reproduces the earlier original-test result: 100% versus 25% Top-1 is `0.85` versus `0.55`, a `+0.30` gap. Across all 20 OOF tasks, the gap contracts to `+0.10`.

Per-fold joint deltas are `+0.30`, `+0.05`, `0.00`, `+0.10`, and `+0.05`. The direction is non-negative at the coarse four-task fold level, but the task-level uncertainty remains large.

### The gain is concentrated in Slack

| Domain | Delta ASR | Delta BUP | Delta ASR+BUP |
|---|---:|---:|---:|
| Banking | 0.00 | -0.04 | -0.04 |
| Slack | +0.40 | 0.00 | +0.40 |
| Travel | 0.00 | +0.04 | +0.04 |
| Workspace | 0.00 | 0.00 | 0.00 |

Slack alone contributes essentially the entire positive scale effect. This is evidence against treating the aggregate gain as domain-general.

### Exact configuration selection is unstable

- Cross-scale Top-1 exact overlap is only 2/20 tasks; 18/20 tasks select different configuration IDs.
- Mean Top-1 pairwise seed Jaccard is 0.033 at 25% and 0.067 at 100%.
- No task has unanimous Top-1 agreement across all three seeds.
- All three seeds choose different Top-1 IDs on 18/20 tasks at 25% and 16/20 at 100%.
- The selected validation recipe also changes by fold. Full-data Top-1 uses all four recipe types across five folds: risk-only, risk+utility, risk+preservation, and risk+blended-utility.

Exact-ID agreement is a strict diagnostic and near-tied configurations can differ without a large utility difference. Nevertheless, together with recipe variation and the failure to transfer at Top-2/4, it shows that the decision layer is substantially less stable than the risk AUC suggests.

### Attack-family evidence does not yet establish OOD generalization

The OOF candidate pool is not limited to the original fixed textual attack. It contains 400 attack configurations:

- 80 static controls
- 80 dynamic multistage attacks
- 80 contextual paraphrases
- 80 tool-knowledge attacks
- 48 AutoDojo/transfer attacks
- 16 structured holdouts
- 16 contextual holdouts

However, the full-data Top-1 policy selects 6 static controls and 5 tool-knowledge attacks, and four of five Slack selections are static controls. The observed full-data gain is therefore compatible with better ranking of familiar/high-base-rate textual families; this experiment is task-held-out, not a strict attack-family-held-out test.

## Statistical cross-check and fix

The first summary exposed a floating-point issue in the Monte Carlo sign-flip comparison when the observed paired mean was exactly zero. Equality cases could be placed on different sides of zero by rounding. The primary joint result was unchanged (`0.1330` Monte Carlo versus `0.1328125` exact), but secondary zero-effect p-values were biased.

The summarizer now:

- uses a numerical tolerance for Monte Carlo equality;
- reports an exact one-sided sign-flip p-value whenever the number of effective non-zero task deltas is small enough;
- records the effective exact assignment count;
- includes a regression test for the zero-mean tie case.

The superseded pre-fix summary is retained as `final_summary_pre_exact_tolerance_fix.json` for provenance and must not be used for reporting.

## Recommended next experiment

Do not scale the dataset or world model again yet. The next highest-information experiment should reuse the current 30 checkpoints and compare frozen, validation-selected ensemble aggregators:

1. current mean-score ensemble;
2. seed-wise percentile/Borda rank aggregation;
3. disagreement-aware lower-confidence scoring;
4. majority/consensus-assisted scoring with validation-only tie breaking.

The primary endpoint should remain 20-task OOF Top-1 ASR+BUP. This directly tests whether the weak downstream transfer comes from unstable seed-level ranking rather than insufficient representation capacity. After that, run a strict attack-family-held-out split to separate template recognition from genuine attack-family transfer.

## Repository and archive mapping

Repository files:

- `scripts/74_build_v2_grouped_oof_folds.py`: grouped five-fold data construction and audit.
- `scripts/server/run_agentdojo_v2_grouped_oof_train.sbatch`: fixed 24-checkpoint training array for folds 1-4.
- `scripts/server/run_agentdojo_v2_grouped_oof_prospective.sbatch`: frozen first-step Top-1/2/4 selection.
- `scripts/75_summarize_v2_grouped_oof.py`: 20-task OOF bootstrap, randomization, paired scale contrast, model transfer metrics, and exact sign-flip cross-check.
- `scripts/76_diagnose_v2_grouped_oof.py`: seed stability, cross-scale overlap, domain effect, and attack-family diagnostics.
- `tests/test_v2_scale_and_selection.py`: fold, selection, paired inference, exact-tie, and diagnostic regression tests.

Canonical remote archive:

`/share/guozhix/wmagentattack/0715/grouped_oof_20task/formal_20260715_oof_v1`

Primary artifacts:

- `data/summary.json`
- `final_summary.json`
- `diagnostics.json`
- `final_summary_pre_exact_tolerance_fix.json` (superseded provenance only)
- `analysis.md`
- `SHA256SUMS`
