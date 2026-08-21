# Parallel world-model gates v22: final results

Date: 2026-08-21

## Overall conclusion

The frozen cycle concludes
`NO_GO_FORMAL_SCALE_V22__RETAIN_INDEPENDENT_COUNTEREVIDENCE`.
The three lines were evaluated independently and were not allowed to compensate
for one another.

| Research line | Frozen result | Clauses | Scientific interpretation |
|---|---:|---:|---|
| Open-vocabulary effect prediction | `NO_GO_OPEN_VOCABULARY_V22` | 7/12 | Slot hashing recovers some unseen-label recall, but calibration and seen/rollout fidelity regress badly. |
| Paired multi-source data design | `GO_DATA_GENERATION_PROTOCOL_READY_V22__96_EPISODE_SMOKE_NOT_RUN` | 15/15 | The linked-table schema and 96-episode budget are coherent; no episode was generated in this cycle. |
| Real long-horizon dynamics | `NO_GO_LONG_HORIZON_H1_H5_V22` | 9/10 | The new recurrent residual passes every action-dynamics clause but inherits the failed v19 effect-rollout fidelity clause. |

No 96-episode generation smoke, medium/large data generation, large world-model
training, or attack-selector training is authorized by the combined gate.

## 1. Open-vocabulary gate

The compositional candidate scorer used category/entity/field/kind/value slots
and 41,490 parameters. It improved task-disjoint unseen positive recall from
0.0000 for fixed v21 to 0.2646. This recall is not sufficient evidence of
transfer:

- task-disjoint unseen NLL worsened from 7.2074 to 8.4660;
- task-disjoint one-step BCE worsened from 0.04426 to 0.14441;
- v19 rollout BCE worsened from 0.02269 to 0.16701;
- source-held-out unseen recall was only 0.05166;
- an independent candidate-ID control reached 0.50794 unseen recall and 2.39379
  unseen NLL, which is direct counterevidence against interpreting recall alone
  as compositional generalization.

The appropriate next direction is a pretrained semantic label encoder with
candidate ranking, matched hard negatives, and explicit seen/unseen
calibration—not further tuning of handcrafted hash slots.

## 2. Data-generation protocol gate

All 15 design clauses passed. The frozen 96-episode smoke protocol contains:

- AgentDojo: 48 episodes;
- ToolSandbox: 24 episodes;
- InjecAgent: 24 observation-only episodes;
- two common seeds, 431 and 433;
- 48 same-seed intervention pairs;
- one shared Meta-Llama-3.1-70B-Instruct 4-bit generation contract;
- four linked tables: `episodes`, `transitions`, `outcomes`, and `pairs`;
- episode-level four-cell task/attack outcomes, without copying terminal labels
  into every transition;
- connected-component train/calibration/confirmation splitting and zero real
  external endpoints.

This result validates a protocol, not generated data. It does not establish
four-cell support, paired semantic validity, tool-use quality, or trajectory
length coverage for the proposed 96 episodes.

## 3. Long-horizon gate

The real AgentDojo corpus passed all 10 data-sufficiency checks: 2,060
trajectories, 6,763 event rows, and 4,703 adjacent transitions across 20 tasks
and four suites. Contiguous window support is H1=4,703, H2=3,450, H3=2,499,
H5=1,320, and H10=206. H10 remains diagnostic because it spans only 9/20
tasks and one frozen fold has zero H10 windows.

All 15 teacher fits and 15 zero-initialized residual fits completed with zero
runtime failure. The model passed nine of ten frozen clauses:

- H1 NLL gain versus the repeated one-step teacher: -0.00288, within the
  non-inferiority margin;
- H1 accuracy gain: -0.000073, also non-inferior;
- H2-H5 NLL gain over the frozen typed-v4 control: +1.46719;
- positive-task fraction: 1.0;
- positive seeds: 3/3;
- all predicted actions were legal.

Free-rollout accumulation versus a repeated one-step baseline was +0.0680 at
H2, +0.1444 at H3, +0.0572 at H5, and +0.0527 at diagnostic H10. The only
failed clause was `v19_effect_rollout_noninferiority`, inherited from the open
vocabulary/effect branch. Therefore the scientific conclusion is nuanced:
long-horizon action dynamics are promising and should be retained as a local
mechanism, but they cannot yet authorize the combined world model because its
effect-state representation is not sufficiently faithful.

## Runtime integrity and counterevidence

Jobs 7220 and 7223 produced the scientific model results. Job 7221 was
cancelled before start only to replace an unnecessarily long Slurm wall-time
request with job 7223; the frozen code, data, seeds, and thresholds were
unchanged. Job 7224 exposed a frozen-v4 row-schema reader bug, and job 7225
exposed NumPy JSON scalar serialization. Both failures were archived. Job 7226
performed label-blind gate-only repairs and did not refit a model or regenerate
predictions. Three compatibility tests passed, the final archive is complete,
and recorded checksums verify successfully.

## Recommended next fixed-budget cycle

Keep the v22 recurrent residual action-dynamics branch frozen. Replace only the
effect-label representation with a pretrained semantic candidate encoder and
calibrated ranking objective. The next gate should require both open-label NLL
and recall improvements over fixed v21 and the independent-ID control, while
retaining v21 one-step and v19 rollout non-inferiority. Only after that gate
passes should the already frozen 96-episode multi-source smoke be executed.

Formal archive:
`/share/guozhix/wmagentattack/0821/parallel_world_model_gates_v22/formal_v2`

Key immutable hashes:

- protocol: `7c362fcb8fc3b6a53390e22b805ac3d4a89dcb83bd7029ff851721c4da57a8d2`;
- open gate: `6629b6d6f0d335e87b91f9183634bc6146aad8e8ff4767895686d22652f80ff0`;
- long predictions: `d4b71fb0cbe3a541a9ed2c0ab8bcacb7628fad2719f235d5ec61d02a7e5d43f0`;
- long gate: `78e5c3e5c4603e2c33fcadb2317068aae66d31c993e50e0f6d2042852ee57238`.
