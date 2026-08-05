# Multi-source Semantic v3 large-build results

## Decision

`LARGE_MULTISOURCE_V1_COMPLETE`

The frozen large-build gate passed. The released archive contains 2,367 rows from ToolSandbox, InjecAgent, and tau3 under one Meta-Llama-3.1-70B-Instruct 4-bit contract. All rows are complete, all merged-source checks pass, and no real external endpoint was called.

This conclusion is a data-construction result. It does not claim attack effectiveness and does not by itself authorize a large Dreamer training run.

## Frozen budget and execution

- Generation array: Slurm `6354`, 21 shards, at most four concurrent one-GPU workers.
- Array wall interval: 2026-08-05 09:09:58--10:43:11 UTC (`01:33:13`).
- Original after-ok summary: Slurm `6355`, cancelled after diagnosing a shard-local pair-scope defect.
- Label-blind recovery and final summary: Slurm `6374`, `COMPLETED`, exit `0:0`, elapsed `00:01:06`.
- Seed: `307` for every large-build source.
- Hyperparameter searches: `0`.
- Shared victim model: `meta-llama/Meta-Llama-3.1-70B-Instruct`, frozen snapshot, 4-bit quantization.
- LLM contract SHA256: `d8a459b64106de53d58915cd5da4ff9f1c667c6077f16b8d9c73364048af9bc2`.

## Release-gate metrics

| Source | Rows | Text responses | Tool calls | Tool-call rate | Exact executions | Non-deterministic exact executions | Runtime failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| ToolSandbox | 95 | 72 | 23 | 0.242105 | 23 | 0 | 0 |
| InjecAgent | 2,108 | 1,464 | 644 | 0.305503 | 0 (observation-only) | 0 | 0 |
| tau3 | 164 | 35 | 129 | 0.786585 | 129 | 0 | 0 |
| Total | 2,367 | 1,571 | 796 | 0.336291 | 152 | 0 | 0 |

All 152 executable model actions were run on two fresh replicas and produced identical results. The merged dataset has unique row IDs, non-empty completions, valid parsed tool names, one frozen LLM contract, and zero real external endpoint calls.

## InjecAgent paired counterevidence

The merged data contains all 1,054 clean/poison pairs.

- Clean rows selecting an attacker tool: `4/1054 = 0.003795`.
- Poisoned rows selecting an attacker tool: `136/1054 = 0.129032`.
- Discordant clean-only pairs: `3`.
- Discordant poisoned-only pairs: `135`.

This is descriptive paired counterevidence. It shows that the published poisoned observations materially alter attacker-tool selection under this victim configuration, but it is not an attack-success-rate measurement and was not used as a release gate.

## Label-blind orchestration repair

The original InjecAgent workers used modulo sharding. Clean and poisoned members of a pair therefore resided in different shards, making pair completeness unidentifiable within any single shard. Every complete shard had zero record-level failures but failed only the local `injecagent_pair_completeness` check. The frozen global merge gate was always the scientifically valid scope for this condition.

The repair did not read outcomes, change tasks or seeds, regenerate completions, call the LLM, or overwrite an original output. Sixteen immutable-shard recovery audits were produced, and the final merged gate retained the preregistered clean/poison completeness condition.

One shard (`injecagent` chunk 5) completed all 132 frozen completions but hit an audit-stage `TypeError` during a transient code handoff before emitting its original audit. It was reparsed from the immutable completion file into a separate recovery path with zero LLM calls. The incident and traceback remain archived as counterevidence; the recovered shard passed all label-blind integrity checks.

## Integrity and archive

- Archive: `/share/guozhix/wmagentattack/0806/multisource_semantic_v3/large_v1`
- Merged dataset SHA256: `9a930150e70c718c9ceb72d119212f485358897487e3a683c123e8725707ba1d`
- `large_gate.json` SHA256: `85cd5701a0e8017ce63c660b352a3506b5a969daf9df591e611435997491ce1f`
- Frozen protocol SHA256: `78473c00ac014d9ec904dc18654f52c28acd42313002a8701f1bb5144364cb70`
- Archive `SHA256SUMS` SHA256: `6b702832640673fea342dfb15bd275ca89dfbbdbcbef1bf898b3dbebdfc0f368`
- Frozen runner SHA256: `712a6489742a8d1247570a728960dc3174fb6b08220f1a10509d0aa742727ca9`
- Frozen semantic-data implementation SHA256: `e5cabd1e912d33de95dff81f1f624e5eff04ff65d3296783e650c4dd93aa8d8c`
- Real external endpoint calls: `0`.

## Gate interpretation and next authorization

The three-source construction pipeline is now released for downstream semantic-state adaptation and task-disjoint data-quality/sufficiency evaluation. The next research cycle should freeze those splits and measure whether adding the three sources improves Semantic Markov sufficiency over the AgentDojo-only dataset. No attack generation or large Dreamer training was started in this cycle.

