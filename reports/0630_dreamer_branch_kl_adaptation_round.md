# 0630 Dreamer Branch/KL Adaptation Round

## Research budget executed

This round used a fixed budget before summarizing:

1. Implement a task-adapted RSSM imagination change: candidate-action
   branching instead of greedy self-rollout.
2. Implement a closer DreamerV3-style offline KL objective: dynamic KL,
   representation KL, and free-nats.
3. Run tuned-branch validation/test selection grids.
4. Train a new KL-balanced model and evaluate its step metrics.
5. Run KL-branch validation/test selection through candidate-cache sweeps after
   the grid runner exposed repeated-process SIGKILL behavior.
6. Cross-check against initial Dreamer, tuned greedy Dreamer, and sklearn
   rollout.
7. Search for counterevidence and map every finding to repo locations.

## Implemented repo changes

| Change | Repo location |
|---|---|
| Candidate-action RSSM branching | `src/wmagentattack/dreamer_world_model.py::rollout_score_step` |
| Compact branch diagnostics | `src/wmagentattack/dreamer_world_model.py::rollout_score_step` |
| DreamerV3-style KL dynamic/repr/free-nats | `src/wmagentattack/dreamer_world_model.py::_OfflineDreamerModule.kl_loss` |
| KL CLI controls | `scripts/15_train_dreamer_world_model.py` |
| KL-branch training job | `scripts/server/run_sheeprl_dreamer_kl_branch_e5.sbatch` |

## Step-level model metrics

| Model | Risk AUC | Utility AUC | Brier | Next-skill top-3 |
|---|---:|---:|---:|---:|
| Initial Dreamer | 0.8173 | 0.8093 | 0.1760 | 0.6754 |
| Tuned Dreamer | 0.8519 | 0.8352 | 0.1532 | 0.6716 |
| KL-branch Dreamer | 0.8336 | 0.8570 | 0.1533 | 0.6549 |
| sklearn | 0.8722 | 0.9795 | 0.1116 | 0.8694 |

Interpretation: KL/free-nats improved utility AUC over tuned Dreamer but hurt
risk AUC and next-skill top-3. The sklearn baseline remains the strongest
supervised predictor.

## Selection results

World-model-top, averaged over seeds 7/13/21:

| Split | Backend | K | ASR | BUP | ASR+BUP |
|---|---|---:|---:|---:|---:|
| val | tuned greedy | 16 | 0.1667 | 0.2292 | 0.3958 |
| val | tuned branch | 16 | 0.2500 | 0.1667 | 0.4167 |
| val | KL branch | 16 | 0.1250 | 0.1875 | 0.3125 |
| val | tuned greedy | 24 | 0.2083 | 0.1667 | 0.3750 |
| val | tuned branch | 24 | 0.2361 | 0.1667 | 0.4028 |
| val | KL branch | 24 | 0.1250 | 0.2500 | 0.3750 |
| val | tuned greedy | 32 | 0.1875 | 0.1771 | 0.3646 |
| val | tuned branch | 32 | 0.1875 | 0.1979 | 0.3854 |
| val | KL branch | 32 | 0.1875 | 0.2188 | 0.4063 |
| test | initial greedy | 16 | 0.3333 | 0.1875 | 0.5208 |
| test | tuned greedy | 16 | 0.3750 | 0.2500 | 0.6250 |
| test | tuned branch | 16 | 0.4167 | 0.2083 | 0.6250 |
| test | KL branch | 16 | 0.5000 | 0.2500 | 0.7500 |
| test | sklearn rollout | 16 | 0.4375 | 0.4375 | 0.8750 |
| test | initial greedy | 24 | 0.3333 | 0.1944 | 0.5278 |
| test | tuned greedy | 24 | 0.3611 | 0.2361 | 0.5972 |
| test | tuned branch | 24 | 0.3333 | 0.2361 | 0.5694 |
| test | KL branch | 24 | 0.3750 | 0.2917 | 0.6667 |
| test | sklearn rollout | 24 | 0.3333 | 0.4583 | 0.7917 |
| test | initial greedy | 32 | 0.2917 | 0.2396 | 0.5313 |
| test | tuned greedy | 32 | 0.3021 | 0.2604 | 0.5625 |
| test | tuned branch | 32 | 0.3229 | 0.2292 | 0.5521 |
| test | KL branch | 32 | 0.3125 | 0.2188 | 0.5313 |
| test | sklearn rollout | 32 | 0.3125 | 0.3750 | 0.6875 |

## Findings

1. Candidate-action branching is useful for attack discovery, but it trades away
   utility. On test K=16 it increases ASR from tuned greedy 0.3750 to 0.4167,
   while BUP drops from 0.2500 to 0.2083.

2. KL/free-nats shifts the model toward utility representation but does not
   solve BUP. KL branch reaches the strongest Dreamer ASR at test K=16
   (0.5000), but BUP remains 0.2500 and still trails sklearn.

3. The best Dreamer setting in this round is KL branch at test K=16/K=24, but
   it is not uniformly better. For test K=32 it is worse than tuned greedy on
   ASR+BUP.

4. The sklearn rollout remains a strong counterexample. It has lower or similar
   ASR at K=24/K=32 but much higher BUP, so the Dreamer adapter still does not
   preserve benign task utility well enough.

5. Grid runner instability exposed an engineering issue: branch diagnostics can
   become too large when repeated across many subprocesses. The fix was to
   store only top-3 branch summaries and use candidate-cache sweeps for
   repeated scoring.

## Counterevidence

- KL branch is worse than tuned greedy on validation K=16 by ASR+BUP
  (-0.0833).
- Tuned branch is worse than tuned greedy on test K=24 and K=32 by ASR+BUP.
- KL branch is worse than tuned greedy on test K=32 by ASR+BUP (-0.0313).
- KL/free-nats increases utility AUC but decreases risk AUC relative to tuned
  Dreamer.
- All Dreamer variants still trail sklearn rollout on test BUP.

## Why the current Dreamer adapter is still not enough

The current adapter is now more than a shallow RSSM wrapper, but it is still
not a full DreamerV3-style agent model:

- It has no actor/critic objective over imagined returns.
- It does not train a reward/continue model in the same way as DreamerV3.
- Its observation is hashed text rather than a learned language-state encoder.
- Its branch selector optimizes existing pair ranking, not generation of a
  utility-preserving attack policy.
- It does not explicitly model the causal relation between untrusted content,
  tool outputs, and downstream task utility.

## Next repo changes recommended

| Goal | Concrete repo change |
|---|---|
| Improve BUP | Add an explicit final-utility value head trained on trajectory outcomes and use it in branch scoring. |
| Reduce branch utility loss | Add a constrained scorer: rank by risk only among candidates above a predicted utility threshold. |
| Improve language state | Replace/augment hash features with frozen embedding features cached per step. |
| Make Dreamer more faithful | Add reward/continue heads and train them with symlog/two-hot style losses where appropriate. |
| Avoid selection overfit | Use validation to choose scorer constraints, then report test only once. |

## External cross-check

DreamerV3 and SheepRL both emphasize that the world model objective includes
more than latent dynamics: KL balancing/free-nats, reward prediction, continue
prediction, and imagined value learning are part of the full method. This
supports the conclusion that the current offline adapter needs task-specific
world-model and utility/value objectives rather than simply plugging AgentDojo
features into RSSM.

