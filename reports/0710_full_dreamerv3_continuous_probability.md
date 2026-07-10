# Full SheepRL DreamerV3 with continuous preservation probabilities

## Scope

This round replaces the previous confidence-weighted binary utility target with
an explicit continuous probability target and adds a complete offline
DreamerV3 behavior-learning stack. The previous RSSM-only adapter remains intact
as a baseline.

## Continuous probability target

For each attacked trajectory, the posterior combines four sources:

1. training-split global attacked-utility rate, strength 1;
2. smoothed clean task solvability, strength 2;
3. training-split `(domain, attack_location)` evidence, capped at strength 4;
4. the trajectory's observed Bernoulli utility, strength 1.

Training rows use leave-one-out global and attack-location evidence. Validation
and test rows use training evidence only. The hyperparameters were selected from
a fixed grid by validation posterior-predictive Brier score; test statistics
were inspected only after selection.

The generated 70B dataset is:

`/share/guozhix/WMagentattack/data/agentdojo_full_llama31_70b/splits_continuous_probability`

All attacked steps have interior probabilities. At trajectory level:

| Split | Attacked trajectories | Observed utility | Mean target | Unique targets |
|---|---:|---:|---:|---:|
| Train | 669 | 0.1973 | 0.2305 | 56 |
| Val | 138 | 0.2029 | 0.2252 | 37 |
| Test | 142 | 0.1690 | 0.2228 | 38 |

## Full DreamerV3 architecture

- Stable 768-dimensional text hashing observation encoder.
- SheepRL MLP encoder and decoder.
- SheepRL RSSM with a 256-dimensional deterministic state and `32 x 32`
  discrete stochastic state; concatenated latent size 1280.
- Action-conditioned two-hot reward model and Bernoulli continue model.
- Auxiliary skill, candidate-skill, attack-risk, continuous-utility, and
  continuous-preservation heads.
- SheepRL discrete Actor over 25 skills.
- Two-hot Critic and EMA target Critic.
- Five-step latent imagination, lambda returns, percentile return scaling,
  actor entropy, and offline behavior cloning with candidate-skill masks.
- Validation-based best-epoch restoration and deterministic inference RNG.

The formal-size smoke model contains 5,952,588 parameters, of which 5,492,813
are trainable. Llama-3.1-70B is not loaded during this stage.

## Verification

- Local test suite: 47 passed.
- CPU reduced-size world/actor/critic backward pass: passed.
- Deterministic repeat inference: maximum difference 0.0.
- Save/load inference: maximum difference 0.0.
- Formal-size GPU smoke job 4218: passed without OOM or CUDA errors.
- Full replay-selection backend smoke job 4217: passed and emitted learned
  risk, utility, preservation, value, reward, and imagined-skill fields.

## Formal run entrypoint

`scripts/server/run_sheeprl_full_dreamer_v3_70b.sbatch`

This is a three-seed Slurm array (`7, 13, 21`). Each task trains with validation
checkpoint selection, evaluates val/test, builds latent-rollout selection
caches, and produces Pareto utility grids. It has intentionally not been
submitted as part of the smoke-validation round.
