# Parameter intervention v18 results

Date: 2026-08-18

Decision: `GO_PARAMETER_INTERVENTION_DATA_DIRECTION_V18`

## Question

After v17 showed that ordinary clean action forks were reproducible but rarely crossed execution boundaries, v18 held the exact reconstructed state and tool fixed and changed exactly one schema-valid argument. The pair selection, sentinels, tasks, seeds, budgets, and gates were frozen before any new branch result was executed.

## Fixed execution

- Slurm job: 7114 (CPU-only AgentDojo synthetic sandbox)
- Coverage: 12 task pairs, one for every suite × difficulty cell
- Rows: 24 canonical branches, each executed from fresh state twice
- Budget: 48 branch calls + 76 prefix replay calls = 124 total sandbox calls
- External endpoint, victim LLM, attack, model training, and Dreamer calls: 0
- Independent manifest builds: byte-identical
- Tests: 18 passed
- Runtime failures, replay mismatches, replica mismatches, and semantic leakage: 0

## Exact results

| Frozen metric | Required | Observed |
|---|---:|---:|
| Complete pairs | 12 | 12 |
| Valid-control successes | 12 | 12 |
| Corrupted execution errors | >=10 | 12 |
| Paired success→error flips | >=10 | 12 |
| Pairs with changed effect | >=10 | 12 |
| Suites with a status flip | 4 | 4 |

All corrupted branches produced a deterministic `ValueError`. The four suite-specific interventions were transaction ID, Slack user, hotel start date, and workspace file ID. Every valid control succeeded and every one-field corruption changed the model-visible successor effect.

## What this establishes

The observational dataset's central weakness was not merely encoder choice: it lacked controlled same-root action and argument interventions. Combining v17 and v18 gives two validated supervision layers:

1. v17 legal action forks provide broad ordinary action-effect diversity: all 12 roots had at least three distinct effects and 95.83% of same-root action pairs differed.
2. v18 minimal parameter forks provide sharp executability boundaries: 12/12 pairs flipped from success to error across all four suites.

The suitable research direction is therefore an intervention-grounded modular world model, not another latent encoder trained only on observational trajectories. Its data should contain a shared canonical root, exact action/tool identity, normalized argument slots, intervention identity, execution status, semantic successor delta, and uncertainty/coverage metadata. Deterministic simulator-known effects remain exact; learned residual heads are restricted to stochastic receipt/evidence and downstream outcome probabilities. The existing Structured Markov v3 + four-cell outcome branch remains the conservative reference until this new paired dataset clears task-disjoint model gates.

## Counterevidence and limits

This is not yet proof that a learned model beats Structured Markov v3. The v18 corruptions are deliberately strong precondition violations and all produce the same error class, so a model could learn a shallow validity detector. v18 also does not test whether a successful mutation persists into a later readback, whether two legal actions conflict, whether evidence updates correctly, or whether any branch improves task-success-and-attack-success selection. These claims remain unauthorized.

Before a neural comparison, the next data gate should therefore use two-step `modify → readback` forks and legal competing updates, with the second observation testing state persistence and conflict resolution. Only after that gate should v17 + v18 + persistence pairs be scaled and used in a task-disjoint comparison against Structured Markov v3 and v6.

## Frozen artifacts

- Archive: `/share/guozhix/wmagentattack/0818/parameter_intervention_v18/formal_v1`
- Manifest SHA256: `306d80996fd40f099cc4214751adff07c245ac5fdb16d5ad198f5365eefa1499`
- Dataset SHA256: `36f20f1fb2c5850b40fabb50bb4d687c3b4643b7da80fa7dee40f42501c84002`
- Gate SHA256: `2c03f730452f29ee4cb5bf259de4a07373667bee275cdb95a3c63e875930c0f9`
- Archive checksum-list SHA256: `c7297a0b0c1223fcb95bd687153e000b5ec4654586d3abc0c2e6bc51fdf1631b`
