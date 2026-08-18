# Persistence and legal-conflict v19 results

Date: 2026-08-18

Decision: `GO_PERSISTENCE_CONFLICT_DATA_DIRECTION_V19`

## Scientific question

v17 showed that legal actions from one reconstructed root usually produce distinguishable effects. v18 showed that one-field parameter interventions expose execution boundaries. v19 tested the remaining requirement for a multi-step world model: whether a successful write is observable after a later read and whether a second legal write changes that readback in a stable, model-visible way.

## Runtime-integrity history

Slurm 7115/formal_v1 is preserved as `INVALID_RUNTIME_INTEGRITY`, not scientific NO-GO. Six Travel sequences failed because a frozen `datetime` subclass could not be represented by the YAML formatter. Three Slack removal steps partially mutated state and then raised `KeyError` because the default Slack record lacked the expected inbox key. Only the failed mechanisms were repaired before a complete result: JSON-native output formatting and a documented channel-message append/readback sequence. Tasks, roots, seeds, budget, metrics, and thresholds were unchanged.

The valid formal result is Slurm 7116/formal_v2.

## Fixed budget and integrity

- 12 paired roots, covering all four suites and all three difficulty levels
- 24 canonical three-step sequences, each reconstructed and executed twice
- 144 sequence-step executions
- 76 observed-prefix replay calls
- 220 total synthetic sandbox calls
- 19 tests passed
- Zero runtime failures, replica mismatches, prefix mismatches, leakage findings, LLM calls, attack examples, GPU jobs, model runs, Dreamer runs, and external endpoint calls

## Exact gate metrics

| Frozen metric | Required | Observed |
|---|---:|---:|
| Complete sequences | 24 | 24 |
| Complete task pairs | 12 | 12 |
| Persistence-control readbacks matched | 12 | 12 |
| Competing-update readbacks matched | 12 | 12 |
| Shared first-write states identical | 12 | 12 |
| Final semantic states differed | 12 | 12 |
| Final observations differed | 12 | 12 |
| Suites with both effects | 4 | 4 |
| Successful sequence steps | all | all |

## Retained data and model direction

The autoresearch direction switch is now supported by three complementary intervention layers:

1. v17 same-root legal-action forks provide broad action-effect diversity: 95.83% of action pairs differed, although the broad boundary-event gate failed.
2. v18 same-state/same-tool one-parameter pairs provide sharp success/error boundaries: 12/12 paired flips.
3. v19 three-step write/read and competing-write/read sequences provide persistent multi-step evidence: all 12 task pairs and all four suites passed.

The authorized next data artifact is a deterministic union with explicit group keys, not a flat concatenation. Every row must retain root/pair/sequence identity, action and normalized argument slots, execution status, semantic successor state, intervention type, horizon, and source version. Splits remain task-disjoint, so paired branches from one root can never cross train/validation/test boundaries.

The retained model direction is an intervention-grounded modular world model:

- exact action/tool/schema and deterministic transition layer;
- Structured Markov v3/v6 state and four-cell outcome branch as the frozen conservative reference;
- a small action-conditioned residual trained on v17 ordinary forks;
- an executability head trained on v18 paired parameter boundaries;
- a persistence/conflict head trained on v19 sequences;
- calibrated coverage uncertainty and abstention for unsupported roots.

This maps the repository to the same-state counterfactual-control principle emphasized by recent action-controllable world-model work, while respecting the need for diverse action-labelled trajectories in heterogeneous world models. It also follows the evidence from code-backed synthetic agent environments that reliable executable transitions are preferable to unconstrained LLM-simulated state changes.

## Counterevidence and limits

v19 is a small, deliberately constructed identifiability pilot. Its writes use fixed synthetic values, only three horizons are exercised, and each suite uses one transition family. A model could memorize these values if the union builder fails to normalize entities and intervention roles. The Slack v1 partial-mutation bug also demonstrates that a tool may return an error after changing state; execution status alone cannot be treated as an atomic transition label.

Therefore this GO does not show that a learned model beats Structured Markov, does not validate attack selection, and does not authorize large-scale generation. The next experiment must first build the union twice, pass duplicate/leakage/group audits, and compare a tiny modular model against unchanged Structured Markov v3/v6 on task-disjoint folds. It must include source-wise and suite-wise ablations so gains cannot come only from the fixed v19 sentinels.

## Frozen artifacts

- Valid archive: `/share/guozhix/wmagentattack/0818/persistence_conflict_v19/formal_v2`
- Invalid runtime archive: `/share/guozhix/wmagentattack/0818/persistence_conflict_v19/formal_v1`
- Manifest SHA256: `b81c88bde8b20caed0a3dfb64389cc6610c714a7346034dd9b0aa48d65ecbcc4`
- Dataset SHA256: `895294f17261d553e578317220a1fd0be1543ae85e9205f8a9e1fd467aa42ca0`
- Gate SHA256: `56c71e3b17a551faeb61cd1e41d164087643660d9fa3f0159bce436de3460913`
- Archive checksum-list SHA256: `737ebc860760a2ab14b067ca92b798d62a64382e3df0f43b622260299686e633`

## Literature mapping

- Trajectory World Models for Heterogeneous Environments: https://proceedings.mlr.press/v267/yin25f.html
- Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning: https://arxiv.org/abs/2602.10090
- Overcoming Statistical Bias in Action-Controllable World Models: https://arxiv.org/abs/2608.04653
