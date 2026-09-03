# Llama-3.1-8B dual-source pilot result

## Scientific decision

`NO_SCALE_RETAIN_COUNTEREVIDENCE_AND_REDESIGN_ONE_CAUSE`

The frozen pilot completed on the friend V100 server without runtime failures, OOM, CUDA errors, missing selected outputs, or real external endpoint calls. The full-scale AgentDojo plus InjecAgent collection is **not authorized** because the preregistered scientific support gates failed.

## Frozen budget and artifacts

- Victim: `meta-llama/Meta-Llama-3.1-8B-Instruct`, 4-bit NF4, FP16 compute.
- AgentDojo: 24 user tasks, four suites, three seeds, eight published attack candidates, 72 clean plus 1,152 attacked trajectories (1,224 selected trajectories total).
- Twenty-four extra injection-task utility diagnostics were produced by AgentDojo and excluded by a label-blind task-identity rule; all expected selected trajectory keys were present and unique.
- InjecAgent: eight published cases, three seeds, clean/poisoned pairs, 48 decisions and 24 complete pairs.
- Remote archive: `/home/pth/outputs/wmagentattack/0903/llama31_8b_dualsource_pilot/formal_v1`.
- Runtime: 2026-09-03 16:15:16 to 21:13:17 (Asia/Shanghai), approximately 4 h 58 min.

## Exact gate results

| Gate | Observed | Required | Result |
|---|---:|---:|---|
| AgentDojo selected trajectories | 1,224 | 1,224 | pass |
| InjecAgent decisions | 48 | 48 | pass |
| Runtime failures | 0 | 0 | pass |
| Stable clean tasks (success in at least 2/3 seeds) | 3 | at least 12 | fail |
| Stable clean tasks by suite | banking 0; slack 2; travel 0; workspace 1 | at least 2 per suite | fail |
| Tasks with at least two distinct joint-success rates | 2 | at least 12 | fail |
| Task-success plus attack-success trajectories | 22 | at least 30 | fail |
| Tasks with a multistep trajectory | 23 | at least 12 | pass |
| Multistep suite coverage | all four suites | all four suites | pass |

The InjecAgent intervention diagnostic also supplied counterevidence: both clean and poisoned attacker-tool-use rates were 0.0, with zero discordant pairs in either direction. This signal was descriptive rather than a release threshold, but it shows that the current adapter/model combination did not express the intended clean-versus-poisoned intervention effect.

## Interpretation

The failure is not an infrastructure or dataset-integrity failure. It is a scientific support failure. Llama-3.1-8B produced many multistep trajectories, but it reliably solved too few clean tasks: only three of 24 were stable, with no eligible Banking or Travel task. Consequently, most attack outcomes cannot distinguish attack quality from victim task-execution failure. The data also contain too little task-level variation across attack candidates and too few joint task-and-attack successes to train or validate the intended selector without severe shortcut risk.

Therefore, scaling the same recipe to the frozen 21,711-interaction plan would mostly replicate low-clean-utility and low-contrast outcomes. The correct retained result is the complete negative pilot and its normalized artifacts, not a relaxed gate.

## Authorized next step

Do not start full-scale dual-source collection. Under a new preregistration, change exactly one cause and run a small clean-only diagnostic of Llama-3.1-8B native tool-use reliability. The diagnostic should preserve task-disjoint evaluation and compare a minimal protocol/prompt repair on the same tasks and seeds before any further attack data are generated. InjecAgent alignment should be reconsidered only after the clean execution gate is restored.
