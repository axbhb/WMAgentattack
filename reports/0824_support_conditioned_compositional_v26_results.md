# v26 support-conditioned compositional world model results

## Decision

`NO_GO_SUPPORT_CONDITIONED_COMPOSITIONAL_V26` (12/20 frozen clauses passed).

Slurm 7306 completed all 45/45 CPU fits and 45/45 metric rows in 265 seconds with zero runtime failures. Five focused tests passed before training, fixed-v21 reproduction error was exactly zero, stderr was empty, and the complete archive checksum verifies.

## Main evidence

The v25 support mechanism is genuinely active, but the v26 decoder is not sufficiently general or stable to replace the retained open-vocabulary diagnostic.

| Task-disjoint metric | no support | v25 atom support | frozen v23 raw |
|---|---:|---:|---:|
| unseen positive recall | 0.1667 | **0.5053** | 0.5291 |
| unseen positive NLL | 3.2020 | **1.5831** | 0.9965 |
| focused entity/attribute recall | 0.0000 | **0.3571** | not recorded |
| matched-count=3 recall | 0.4286 | 0.3810 | not recorded |
| unseen FPR | 0.0000 | 0.0260 | not recorded |
| unseen precision | 0.3333 | 0.3574 | not recorded |
| one-step BCE | 0.0404 | 0.0428 | 0.0423 |
| rollout BCE | 0.0225 | 0.0278 | 0.0293 |

Adding support improved unseen recall by 0.3386 and reduced unseen NLL by 1.6190 relative to the identical no-support architecture. This is strong positive evidence that explicit sibling-task branches are useful. It nevertheless missed the frozen 0.60 recall floor, trailed v23 raw recall by 0.0238, and had substantially worse NLL than v23 raw.

## Stability and counterevidence

The gain was not task-fold stable. Fold 0 contained seven banking/Slack entity and attribute effects. The support arm reached 5/7 recall (0.7143) for every seed, versus zero without support. Fold 2 contained two webpage entity/attribute effects plus seven `matched_count=3` effects. Webpage recall remained zero; count-3 recall was only 2/7 or 3/7 depending on seed. Support improved only three of the six affected fold/seed cells, below the frozen four-cell requirement.

This isolates two architecture failures:

1. Independent atom BCE discards bindings. Knowing `entity::webpage`, `field::content`, and `tool::get_webpage` separately does not guarantee that the latent binds them into one successor evidence record.
2. `matched_count` is a derived successor-state quantity. A cumulative ordinal head trained mostly on counts 0 and 1 does not reliably extrapolate count 3 in a task-disjoint fold. It should be computed from predicted evidence-state change, not treated as an independent semantic label.

Diagnostic transfer confirms the problem: tool-family unseen recall was 0.0220 and source-held-out unseen recall was 0.0742, both far below their frozen floors. At the same time, seen recall (0.9749), pair assignment (0.9931), one-step BCE and rollout BCE remained non-inferior, validating the decision to preserve fixed-v21 outputs for seen labels.

## Retained method and next direction

Retain fixed v21 for seen effects, v22 recurrent supervision, and the v25 support dataset as positive mechanism evidence. Reject the v26 independent-atom renderer as the primary open-vocabulary effect head. No data smoke, large generation, attack study, Dreamer or planner is authorized.

The next research direction is a typed successor-evidence world model:

1. reconstruct current/next Semantic State v3 pairs from the frozen v17-v19 source datasets and v25 support branches;
2. predict bound evidence-record slots such as `(source_tool, entity_type, attribute_name, kind, link_status)` rather than independent atoms;
3. apply the exact deterministic state-delta renderer to produce canonical entity, attribute, source and matched-count effects;
4. run a data-identifiability and leakage gate before any new model comparison.

This changes the failed mechanism instead of relaxing the v26 gate or fitting another label scorer to the same 121 rows.

## Archive

- Archive: `/share/guozhix/wmagentattack/0824/support_conditioned_compositional_v26/formal_v1`
- Slurm: `7306`
- Metrics SHA256: `4ec9cee78d210668c0af8925012fc9ff5152f3493c26786f9f342def034c6771`
- Gate SHA256: `ab99f4299d3f098c271d6bcc495785461b400de69dd49d98f25f7948a863b3e7`
- Archive checksum SHA256: `087501fe249bb05ae541221a58d6a60808a8bc56c06a5a39a65ec985db36f088`
