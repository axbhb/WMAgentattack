# Shared action ontology repair results

Date: 2026-08-12

Decision: `NO_GO_SHARED_ACTION_ONTOLOGY_DOES_NOT_REPAIR_MULTI_SOURCE_TRANSFER`

## What was tested

This fixed-budget experiment tested whether a label-blind action ontology could
repair the negative transfer observed when AgentDojo, ToolSandbox, and
InjecAgent were simply pooled. Candidate schemas were mapped from public names,
descriptions, and parameter schemas into operation, object, effect,
communication-scope, and terminal components. No selected action, outcome,
security label, task utility, or confirmation metric was used to build the
ontology.

Two failed preflight versions are retained. The final hierarchical v1.2 mapping
passed all preflight gates with 138/138 candidates annotated, seven AgentDojo
bridge keys shared with at least one auxiliary source, six bridge keys shared by
all three sources, and no within-interface collision for the local-residual
representation. The ontology-only representation collapsed distinct legal
actions in all 6,763 AgentDojo rows, all 285 ToolSandbox rows, and 2,874
InjecAgent rows; this was preregistered counterevidence rather than repaired
after training.

## Frozen comparison

The formal run used the same five task-disjoint AgentDojo folds and training
seeds 7/17/29 as the parent experiment. It ran 60 neural fits: two candidate
representations, two state variants, five folds, and three seeds. The primary
candidate was the shared ontology plus a 20% source-local schema residual.

| primary comparison | NLL gain | accuracy gain | positive tasks |
|---|---:|---:|---:|
| ontology residual vs raw pooled | -0.074762 | -0.059529 | 35% |
| ontology residual vs AgentDojo-only | -0.057619 | -0.076868 | 45% |

All twelve preregistered acceptance clauses failed. Against raw pooling, the
paired task NLL bootstrap interval was `[-0.145305, -0.010592]`, providing
counterevidence that the degradation was not only a seed-average artifact.
Against AgentDojo-only, 9/20 tasks improved and 11/20 worsened.

## Scientific conclusion

Action-name mismatch is not the dominant explanation for the failed
three-source expansion. A coarse shared ontology discards source-local
distinctions, while adding a local residual does not create compatible
trajectory dynamics. The ontology audit remains useful metadata, but neither
ontology candidate representation is retained as the main predictive model.

The next authorized independent repair is to restore real adjacent-step
supervision: learn action-conditioned observable outcomes and the next victim
action from actual AgentDojo trajectory transitions before testing
source-specific adapters.

## Integrity

- Slurm: `6733` on CPU; pending GPU job `6732` was cancelled before producing an
  archive or training result.
- Runtime: 2026-08-12 11:49:15--12:31:02 UTC; zero runtime failures.
- Tests: 15 passed.
- Predictions: 81,156 rows from exactly 60 frozen runs.
- New LLM calls, tool executions, real endpoint calls, attack generation, and
  Dreamer runs: all zero.
- Archive: `/share/guozhix/wmagentattack/0813/action_ontology/formal_v3`
- Dataset SHA256: `5260df08d4dfe7c9bc03b4dfe7078a3946179091566950e286e6b89c1d421a4d`
- Summary SHA256: `8d5bc497b454a6b0785044c0c8a6f29d283efcdc9dfac9cc5a2b64cf3cedbcc2`
- Predictions SHA256: `80e404f6a42627c5ae08866f5d645c842a5fd9d20dea1ec518bf4ddbbd607708`
- Full archive checksum verification: PASS.
