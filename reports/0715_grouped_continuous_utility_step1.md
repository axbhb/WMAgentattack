# AgentDojo-v2 grouped continuous utility: step-1 implementation

Date: 2026-07-15

Status: implementation and local regression complete; remote smoke not submitted because the NTU VPN route is currently absent.

## Why this is the first intervention

The two frozen ensemble rounds did not improve the full-data Top-1 result. The existing full-data model improved binary utility discrimination, but its grouped utility Brier did not improve over the smaller-data model. This points to utility supervision and decision-time alignment, rather than checkpoint aggregation, as the next bottleneck.

The old utility path applies continuous BCE at every trajectory step and binary ranking at the final step. Prospective selection, however, uses the first trajectory step and averages predictions across the five victim-model seeds of one attack configuration. The training unit, target, and decision unit were therefore misaligned.

## Implemented method

The new path is opt-in and leaves old checkpoint loading and default training unchanged.

1. Pack every attack `multiseed_group_id` as a complete five-trajectory unit. In ranking mode, batches preferentially contain multiple configurations from the same user task.
2. At trajectory step 0, average the five utility probabilities for a configuration.
3. Add a confidence-weighted configuration-level Brier loss against `utility_probability_target`.
4. Within each user task, form configuration pairs whose posterior targets differ by at least 0.1. Apply pairwise logistic loss to the higher- versus lower-target configuration, weighted by target gap and geometric-mean posterior confidence.
5. Cap ranking pairs at eight per user task per batch.
6. Support two ablations: utility-head-only gradients through detached latents and full end-to-end gradients.
7. Select the best epoch with first-step grouped validation metrics when the new mode is enabled.

Implementation map:

- `src/wmagentattack/full_dreamer_v3.py`: configuration, complete-group batching, continuous pair construction, grouped utility losses, and first/final grouped validation.
- `scripts/23_train_full_dreamer_v3.py`: command-line controls for the new losses and first-step validation.
- `scripts/24_eval_full_dreamer_v3.py`: explicit first-step reevaluation override for a fair baseline comparison.
- `scripts/server/run_agentdojo_v2_group_utility_smoke.sbatch`: fold-0/seed-7 detached versus end-to-end smoke, including baseline reevaluation and downstream selection.
- `scripts/81_compare_v2_group_utility_smoke.py`: validation-only variant selection, loss-activation checks, and diagnostic test comparison.
- `configs/0715_grouped_continuous_utility_protocol.json`: frozen protocol and formal success rule.

## Verification completed

- Python compilation: passed.
- Patch whitespace validation: passed.
- Slurm Bash syntax: passed.
- Relevant local regression: 23 passed.
- Tested countercases include incomplete-group rejection, task separation, target-gap filtering, pair caps, first-step versus final-step grouped metrics, and smoke-gate selection using validation only.

The full repository test collection was not a code failure: the lightweight local Python used for tests does not have AgentDojo and several optional provider packages installed. The directly affected WM-AgentAttack tests all pass.

## Smoke gate

The smoke is fixed to fold 1, checkpoint seed 7, five epochs, and both gradient variants. Fold 1 is used because the source archive has fold-0 data but no fold-0 baseline checkpoints; folds 1-4 each contain the expected six checkpoints. It proceeds to formal five-fold training only if:

- both new losses are finite;
- complete configuration groups and ranking pairs are actually nonzero during training; and
- the validation-selected variant's grouped utility Brier is no more than 0.01 worse than the first-step-recomputed baseline.

The smoke test split is diagnostic only and cannot select the variant. The formal success rule remains Top-1 ASR+BUP improvement of at least 0.05, nonnegative BUP change, utility-Brier improvement in at least 10 of 15 paired runs, and no domain losing more than 0.10.

## Current blocker and exact next action

The configured hostname currently returns DNS `NXDOMAIN`, all four previously observed `10.96.x.x` addresses time out, and Windows shows no NTU VPN interface/route. No Slurm job has been claimed or submitted.

When the VPN route is restored, sync the six implementation/experiment files to `/share/guozhix/WMagentattack`, submit the two-task smoke array, monitor both tasks, run the comparison script, and use its validation-only gate to decide whether to launch the fixed formal experiment.
