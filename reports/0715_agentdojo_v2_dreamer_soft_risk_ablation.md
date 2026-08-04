# AgentDojo v2-final DreamerV3 risk-label ablation

Date: 2026-07-15

## Question

Does replacing per-trajectory binary attack outcomes with five-seed posterior
attack probabilities improve the SheepRL DreamerV3 world model on the final
AgentDojo v2 dataset?

## Data and protocol

- Dataset: `/share/guozhix/wmagentattack/0714/agentdojo_v2/final_dataset`
- 2,060 trajectories and 6,763 steps: 2,000 attacked trajectories from 400
  configurations repeated over seeds 7/17/29/43/61, plus 60 clean
  trajectories.
- Split sizes: 1,236/412/412 trajectories and 3,776/1,426/1,561 steps for
  train/validation/test.
- Model: SheepRL DreamerV3 offline skeleton with 5,952,588 parameters
  (5,492,813 trainable), auxiliary risk, utility, preservation, skill, and
  candidate-skill heads.
- Formal Slurm job: `4449`; 30 epochs; seeds 7, 13, and 21; one validation-best
  checkpoint per run.
- Variant selection: lowest mean validation objective across the three fixed
  seeds. Test metrics were not used for variant selection.
- Evaluation unit: one final step per trajectory followed by averaging the five
  repeated trajectories in each attack configuration. The group-aware
  objective is risk Brier + utility Brier + preservation Brier +
  0.25 * (1 - next-skill accuracy).
- The two variants differed only in `binary_risk_loss_scale`,
  `soft_risk_loss_scale`, and `risk_reward_binary_mix`. Utility and
  preservation supervision, architecture, optimization, and splits were held
  fixed.

Raw output:
`/share/guozhix/wmagentattack/0715/dreamer_v2_final/formal_20260715_softprob_v1`

## Frozen results

All values are mean +/- population standard deviation over three model seeds.
For objectives and Brier scores, lower is better; for AUC and accuracy, higher
is better. Delta is soft-risk minus binary-risk.

| Metric | Split | Binary risk | Soft risk | Delta |
|---|---:|---:|---:|---:|
| Group-aware objective | val | 0.431830 +/- 0.020186 | 0.453205 +/- 0.023912 | +0.021375 |
| Group-aware objective | test | 0.324058 +/- 0.029032 | 0.333956 +/- 0.002682 | +0.009898 |
| Group risk-probability Brier | val | 0.016313 +/- 0.006293 | 0.016584 +/- 0.000585 | +0.000272 |
| Group risk-probability Brier | test | 0.020194 +/- 0.003962 | 0.030522 +/- 0.006322 | +0.010328 |
| Step risk-probability Brier | val | 0.071875 +/- 0.012497 | 0.048463 +/- 0.002288 | -0.023412 |
| Step risk-probability Brier | test | 0.068367 +/- 0.013015 | 0.055344 +/- 0.007643 | -0.013023 |
| Binary risk AUC | val | 0.912810 +/- 0.010680 | 0.902244 +/- 0.009505 | -0.010567 |
| Binary risk AUC | test | 0.898044 +/- 0.001471 | 0.861364 +/- 0.002540 | -0.036681 |
| Group utility-probability Brier | val | 0.080247 +/- 0.005220 | 0.086691 +/- 0.006328 | +0.006444 |
| Group utility-probability Brier | test | 0.071187 +/- 0.015337 | 0.069233 +/- 0.002180 | -0.001954 |
| Group preservation-probability Brier | val | 0.177778 +/- 0.011803 | 0.195769 +/- 0.012488 | +0.017990 |
| Group preservation-probability Brier | test | 0.064355 +/- 0.015164 | 0.063424 +/- 0.001133 | -0.000931 |
| Next-skill accuracy | val | 0.370033 +/- 0.007385 | 0.383357 +/- 0.029001 | +0.013324 |
| Next-skill accuracy | test | 0.326714 +/- 0.013407 | 0.316891 +/- 0.012091 | -0.009823 |

Per-seed group-aware objective:

| Seed | Binary val | Soft val | Soft - binary | Binary test | Soft test | Soft - binary |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.459813 | 0.467601 | +0.007788 | 0.303872 | 0.337221 | +0.033349 |
| 13 | 0.422733 | 0.472507 | +0.049773 | 0.303188 | 0.333997 | +0.030809 |
| 21 | 0.412945 | 0.419508 | +0.006563 | 0.365114 | 0.330651 | -0.034463 |

Validation selected **binary-risk**. Its mean test objective is also lower, but
that test result is confirmatory and was not part of selection.

## What improved and what did not

Pure soft-risk supervision consistently improved the all-step risk-probability
Brier score. This is evidence that posterior targets provide a useful
calibration signal. It did not improve the actual selection target:

- Validation group objective became worse in all three paired seeds.
- Test group risk Brier became worse in all three paired seeds.
- Test binary risk AUC fell in all three paired seeds, by 0.036681 on average.
- On validation, most of the total regression came from utility and
  preservation heads, indicating negative transfer through the shared latent
  state rather than only a weak risk head.

The one-epoch smoke test was counterevidence: it showed a 0.019919 validation
objective improvement for soft-risk. The 30-epoch, three-seed result reversed
that direction, so the smoke result must not be reported as an effectiveness
result.

## Interpretation

The five posterior labels in a configuration are repeated versions of the same
five Bernoulli observations; they are not five new independent probability
measurements. Replacing every outcome with the shared posterior mean removes
trajectory-level outcome information. Binary likelihood training keeps that
information, and averaging predictions across the five seeds already produces
a configuration-level probability estimator.

There is also an objective mismatch: the current risk loss supervises every
step independently, whereas formal evaluation uses only the final step and then
averages by multi-seed configuration. That explains how all-step calibration
can improve while the group-level metric degrades.

## Decision and next experiment

Keep binary-risk as the current v2-final baseline. Do not use pure soft-risk as
the formal model.

The next experiment should retain binary Bernoulli likelihood and add the
posterior probability only as a group-final calibration regularizer:

1. Construct batches containing all five trajectories from each attack
   configuration.
2. Apply binary risk BCE to each trajectory's final latent state.
3. Average the five final risk predictions and match that average to the
   Jeffreys posterior target with a separate calibration loss.
4. Keep imagined risk reward binary for the first ablation so only the new
   calibration loss changes.
5. Sweep calibration weights 0.1, 0.25, and 0.5 on validation; freeze the best
   setting before test evaluation.

This directly aligns training with the group-aware evaluation unit while
preserving the outcome evidence that pure soft-risk discarded.
