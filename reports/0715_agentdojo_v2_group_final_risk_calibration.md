# AgentDojo v2 group-final risk calibration ablation

Date: 2026-07-15

## Research question

Can the five-seed posterior attack probability improve the v2-final DreamerV3
model when it is used as a configuration-level calibration regularizer rather
than replacing binary trajectory outcomes?

## Implementation and protocol

The ablation added four controls to the SheepRL DreamerV3 learner:

1. Every attack configuration is packed with all five victim-model seed
   trajectories in the same batch. The 240 train attack groups were verified
   to be rectangular, with no split or missing group.
2. Binary attack BCE is applied at the final trajectory state instead of being
   repeated over every step.
3. The mean of five final risk probabilities is matched to the Jeffreys
   posterior target with a confidence-weighted Brier loss.
4. Calibration latent features are detached so the calibration loss directly
   updates the risk head rather than the shared RSSM representation.

Because final-step supervision and grouped batching differ from the previous
binary baseline, a new `lambda=0` control was trained under exactly the same
protocol. The fixed validation budget was:

- Weights: 0, 0.1, 0.25, and 0.5.
- Model seeds: 7, 13, and 21.
- 30 epochs with validation-best checkpoint selection.
- Slurm validation job: `4457` (12 runs).
- Selection used only mean group-aware validation objective.
- After selection was frozen, test was evaluated only for the control and the
  selected weight with Slurm job `4469` (6 evaluations).

Archive:
`/share/guozhix/wmagentattack/0715/dreamer_v2_group_calibration/formal_20260715_groupcal_v1`

Frozen selection SHA-256:
`c95b505df1f0dcfecf668487b4448573446f815b802acd82b5c39d45807c0b0c`

Final summary SHA-256:
`60c57294bd9314732bb6eed161905737d9d339ff12b6197812b16dacf44d551b`

## Validation selection

Values are mean +/- population standard deviation over three model seeds.
Lower objective/Brier is better; higher AUC/accuracy is better.

| Weight | Group objective | Group risk Brier | Risk AUC | Skill accuracy |
|---:|---:|---:|---:|---:|
| 0 | 0.435793 +/- 0.031399 | 0.018243 +/- 0.005366 | 0.829217 +/- 0.003168 | 0.352735 +/- 0.027810 |
| 0.1 | **0.430454 +/- 0.021428** | 0.014537 +/- 0.003159 | 0.820643 +/- 0.005367 | 0.350397 +/- 0.023499 |
| 0.25 | 0.434615 +/- 0.035546 | 0.015345 +/- 0.003755 | **0.833818 +/- 0.005319** | **0.361851 +/- 0.027268** |
| 0.5 | 0.455560 +/- 0.038198 | **0.012583 +/- 0.002167** | 0.828969 +/- 0.005905 | 0.338476 +/- 0.019788 |

Validation selected `lambda=0.1`. Its paired objective deltas relative to the
new control were -0.016597, -0.008228, and +0.008809 for seeds 7, 13, and 21.
The mean improvement was -0.005339, but it was present in only two of three
seeds and its paired standard deviation was 0.010572.

The weight-0.5 result is important counterevidence: it produced the best risk
calibration but the worst total objective. Stronger risk calibration therefore
interfered with utility, preservation, or skill quality.

## Frozen test result

| Metric | New control | Selected weight 0.1 | Selected - control |
|---|---:|---:|---:|
| Group objective | 0.427425 +/- 0.020759 | **0.422755 +/- 0.025641** | -0.004670 |
| Group risk Brier | 0.038013 +/- 0.020030 | **0.036408 +/- 0.016075** | -0.001605 |
| Group utility Brier | 0.099737 +/- 0.004593 | **0.097559 +/- 0.006626** | -0.002178 |
| Group preservation Brier | 0.120446 +/- 0.009222 | **0.116462 +/- 0.016591** | -0.003984 |
| Risk AUC | **0.772731 +/- 0.004641** | 0.770254 +/- 0.004983 | -0.002477 |
| Skill accuracy | **0.323083 +/- 0.009303** | 0.310698 +/- 0.007544 | -0.012385 |

The per-seed test objective deltas were +0.040297, -0.024335, and -0.029974.
The mean improvement was -0.004670 with a paired standard deviation of
0.031880. Thus the small mean gain is not seed-stable. Risk AUC and skill
accuracy became worse in all three paired seeds.

## Cross-check against the previous binary baseline

The scientifically relevant comparison is not only against the new control.
The protocol change itself must also beat the previously frozen binary model.

| Metric | Split | Previous binary | Selected group calibration | Delta |
|---|---:|---:|---:|---:|
| Group objective | val | 0.431830 | 0.430454 | -0.001376 |
| Group objective | test | **0.324058** | 0.422755 | +0.098697 |
| Group risk Brier | val | 0.016313 | **0.014537** | -0.001776 |
| Group risk Brier | test | **0.020194** | 0.036408 | +0.016214 |
| Risk AUC | val | **0.912810** | 0.820643 | -0.092168 |
| Risk AUC | test | **0.898044** | 0.770254 | -0.127790 |
| Group utility Brier | test | **0.071187** | 0.097559 | +0.026372 |
| Group preservation Brier | test | **0.064355** | 0.116462 | +0.052107 |

The selected method nearly ties the old validation objective but fails to
transfer. Its test objective is worse by 0.098697, and every major test
component regresses. It must not replace the previous binary baseline.

## Diagnosis

- Final-only binary supervision reduces train-split risk observations from
  3,776 labeled steps to 1,236 trajectory endpoints. The large AUC drop indicates
  that the repeated step supervision was useful for trajectory-level risk
  discrimination even though formal probability evaluation is group-level.
- Grouped batches changed the optimization distribution for every world-model
  head, not only risk. This is consistent with the large utility and
  preservation regressions on test.
- Detaching the calibration latent does not fully isolate the shared model.
  Calibration changes risk-head weights; subsequent binary-risk gradients pass
  through those changed weights into the latent state. This is an inference
  from the architecture and the observed cross-head changes.
- The calibration effect is smaller than its across-seed variability. Mean-only
  selection therefore overstates its reliability.

## Decision

**NO-GO as a replacement training protocol.** Retain the previous all-step,
IID-batched binary-risk DreamerV3 model as the current v2-final baseline.

The posterior probabilities remain useful, but only as a calibration target
that must not perturb the backbone or other heads.

## Next controlled experiment

Reuse the three frozen previous-binary checkpoints and keep the complete
DreamerV3 model fixed. Fit only a monotonic affine risk calibrator
`sigmoid(softplus(a) * logit + b)` from train multi-seed groups, with identity
regularization selected on validation. This design has three useful controls:

1. Utility, preservation, skill policy, RSSM, and actor/critic outputs remain
   bit-for-bit unchanged.
2. Positive slope preserves per-step binary risk ordering and therefore should
   preserve AUC, apart from numerical ties.
3. Test evaluation can compare calibrated and unmodified probabilities from
   the exact same frozen checkpoint, isolating calibration from retraining.

This is the next informative test; another full Dreamer retraining sweep would
confound calibration with backbone optimization again.
