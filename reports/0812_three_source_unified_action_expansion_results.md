# AgentDojo + ToolSandbox + InjecAgent unified expansion results

## Conclusion

The frozen experiment concludes:

`NO_GO_NO_REPLICATED_AGENTDOJO_OOF_BENEFIT_FROM_THREE_SOURCE_EXPANSION`

The three sources can be represented in one leakage-audited action dataset,
but simply adding ToolSandbox and InjecAgent to the current Semantic Markov
training objective does **not** reliably improve task-disjoint AgentDojo action
prediction. The unified dataset should be retained as an audited resource, not
claimed as an effective training expansion for the current architecture.

## Unified dataset

The dataset contains 13,372 action-decision records and 138 candidate action
schemas:

| source | action rows | task units | groups |
|---|---:|---:|---:|
| AgentDojo | 6,763 | 20 | 2,265 step-level multiseed groups |
| ToolSandbox | 285 | 70 | 95 |
| InjecAgent | 6,324 | 17 | 1,054 |

AgentDojo semantic skills and `finish` were adapted to the same candidate-schema
interface as ToolSandbox/InjecAgent functions and textual responses. Source
identity was retained. Targets, outcomes, security labels, attack labels, and
identifiers were excluded from the causal model input.

Two independent builds were byte-identical. The dataset SHA256 is
`1ab07868cf0ee9a6ec021c98489444c0c0255dd4622a3c0883b6b3ed88d5c3c3`.
All preflight checks passed, including the five task-disjoint AgentDojo folds,
one OOF confirmation appearance per AgentDojo task, zero exact normalized-goal
overlap between AgentDojo and the auxiliary sources, one frozen auxiliary LLM
contract, and zero real external endpoint calls.

## Frozen comparison

The comparison used the same 20 AgentDojo OOF tasks, folds, rows, model capacity,
and seeds in both conditions:

1. AgentDojo-only training.
2. AgentDojo + ToolSandbox + InjecAgent training, with source mass fixed at
   0.50/0.25/0.25 and equal task mass within each source.

There were 60 neural runs: five folds, two data conditions, two representations,
and seeds 7/17/29. No early stopping, validation selection, hyperparameter grid,
new LLM calls, new tool calls, attack generation, or Dreamer training was used.

| representation | AD-only NLL | expanded NLL | NLL gain | AD-only accuracy | expanded accuracy | accuracy gain | positive tasks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Semantic Markov | 1.616682 | 1.599539 | +0.017143 | 0.485753 | 0.468413 | -0.017339 | 50% |
| Structured Markov v3 | 1.578435 | 1.572566 | +0.005868 | 0.470143 | 0.440362 | -0.029781 | 55% |

Semantic Markov NLL gains by seed were `+0.021846`, `+0.022097`, and
`+0.007487`. Accuracy gains were `-0.056770`, `+0.000490`, and `+0.004261`.
Thus two seeds crossed the NLL threshold but the mean did not, and the accuracy
effect was not replicated.

The Semantic Markov paired task bootstrap interval was
`[-0.018587, +0.054355]`; its exact sign test was 10 wins and 10 losses
(`p=1.0`). Structured v3 had interval `[-0.023541, +0.035178]` and 11 wins
versus 9 losses (`p=0.8238`). These were preregistered counterevidence, not
additional post-hoc gates.

## Counterevidence and diagnosis

- Semantic Markov improved mean NLL slightly, so the auxiliary data is not
  completely unrelated. The effect is nevertheless below the frozen `+0.02`
  threshold and is not stable across tasks or seeds.
- Accuracy decreased for both representations. The expanded model appears to
  spread probability more smoothly without improving the top-ranked action.
- Domain effects disagree: Semantic Markov NLL improved for banking
  (`+0.0424`) and workspace (`+0.0549`), but worsened for Slack (`-0.0138`) and
  Travel (`-0.0150`). Structured v3 was especially negative on Travel
  (`-0.0469`).
- AgentDojo targets semantic skill classes, whereas the auxiliary sources target
  raw function schemas. Candidate descriptions provide a transfer channel, but
  the current shared encoder has no explicit cross-source action ontology.
- The sources also expose different horizon structures: AgentDojo contributes
  trajectory steps, while the retained auxiliary records are mainly isolated
  decisions. More rows therefore do not automatically provide compatible
  transition supervision.

## Method implication

The result rejects **simple pooled expansion**, not all multi-source learning.
The next justified modification is to isolate alignment before another scale
claim: define a label-blind shared action ontology (read/search/write/send/stop
plus domain object), preserve source-specific candidate heads or adapters, and
test it first on the same frozen AgentDojo OOF surface. A future experiment
should distinguish ontology alignment from additional data volume; changing
source weights or tuning the current pooled model after seeing this result would
not constitute confirmation.

## Execution integrity

- Slurm job: `6715`
- Runtime: 2026-08-12 05:08:39--05:11:24 UTC
- Device: NVIDIA RTX 6000 Ada Generation, 49,140 MiB
- Runtime failures: 0
- Tests: 19 passed
- Prediction rows: 81,156
- Frozen code and input post-run verification: PASS
- Full archive checksum verification: PASS
- Archive: `/share/guozhix/wmagentattack/0812/three_source_unified_action/formal_v1`
- Summary SHA256: `7f3166b9dd6f52432ed1e1d8a3118e63f54ec6d9ca357aaf10eed2691c52f327`
- Predictions SHA256: `11147f9fa64e264239ad5591f297c9a6d2ecadbe39e0089bbf0a1005dff3821c`
