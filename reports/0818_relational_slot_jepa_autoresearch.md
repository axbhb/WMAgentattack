# Relational Slot-JEPA latent autoresearch

Status: `STAGE_A_PREREGISTERED_BEFORE_TRAINING`.

The fixed three-stage budget tests whether a canonical relational latent can replace the fixed Structured Markov v3 hash representation without reintroducing raw-text task memorization. Stage A changes only the state encoder by adding a zero-gated relational-slot residual to the retained v6 model. Stage B is independently authorized after an integrity-valid Stage A and adds action-conditioned JEPA plus semantic grounding. Stage C removes the old Structured Markov context only if Stage B passes its complete gate.

All stages retain task-disjoint folds, seeds 7/17/29, exact legal-action masking, the four-cell soft target, and the same AgentDojo sandbox dataset. Raw goal, observation, schema descriptions, task IDs, future fields, and outcome labels are excluded from slot inputs.

Pre-formal, label-blind capacity audit: representing every local entity mention as a separate padded node caused avoidable truncation. Before any formal result, entity mentions were therefore deterministically aggregated by `(semantic type, goal/observation/shared role)` with a log-count feature. This retains equality and cardinality signals, removes raw values, bounds the graph, and avoids result-conditioned architecture changes.

Runtime integrity record: Slurm 7066 exited before any model fit or prediction because the teacher configuration used the key `epochs` while the frozen v5 trainer expects `fixed_epochs`. The sole repair renamed that key while preserving the value 120. No data, architecture, seed, loss, or gate changed; exactly one retry is authorized.

## Stage A result

Slurm 7067 completed all 45 fixed fits with zero runtime failures and exact v6 metric replication. Decision: `NO_GO_RELATIONAL_SLOT_STAGE_A` (7/10 clauses passed). The slot arm preserved h1 (NLL 1.771204 versus 1.771762; accuracy 0.447158 versus 0.446900), but h2--h5 NLL worsened by 0.004760 on average, all three seed effects were negative, and future joint CE worsened by 0.010795, just beyond the 0.01 noninferiority margin. Eleven of 20 tasks improved h2--h5, so the representation has local signal but no reliable transferable gain. This rules out relational slots trained only through the existing downstream losses.

All integrity conditions passed: 27,052 fold-specific slot rows, maximum 25 nodes, zero truncation, zero raw values, exact entity-renaming invariance tests, and legal predictions only. The pre-registered Stage B is therefore authorized to test whether predictive latent geometry and semantic grounding can turn the structurally valid slots into a useful state representation.

Stage A archive: `/share/guozhix/wmagentattack/0818/relational_slot_jepa_v7/stage_a/formal_v1`.

## Stage B result

Slurm 7068 completed all 45 fixed fits with zero failures. Decision: `NO_GO_GROUNDED_JEPA_STAGE_B` (6/10 clauses passed). Anti-collapse succeeded (mean latent-dimension standard deviation 0.468933; only 3.125% below 0.05), h1 remained within noninferiority, and future four-cell CE improved by 0.012066. Nevertheless h2--h5 NLL worsened by 0.011363 versus v6 and 0.006603 versus Stage A; only 9/20 tasks improved and all three seed effects were negative.

The counterevidence is domain-structured: several Banking and Slack tasks improve, while Travel tasks degrade strongly (for example `travel|user_task_2`: -0.096824 NLL gain). The representation is diverse and predictable, so collapse is not the explanation. The anonymized slot state omitted action-relevant domain concepts such as hotel, restaurant, flight, rating, and price. JEPA faithfully learned this incomplete state geometry and therefore could not recover the missing next-tool information. Stage C pure replacement is `NOT_AUTHORIZED` under the frozen protocol.

Stage B archive: `/share/guozhix/wmagentattack/0818/relational_slot_jepa_v7/stage_b/formal_v1`.

Binding retained model after this loop: `Structured Markov v3 + four-cell head + zero-initialized v6 residual`. The next independent hypothesis must preserve normalized interface-aligned intent content rather than adding more latent regularization to the same lossy slots.
