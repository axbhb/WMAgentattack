# Action-conditioned event graph v12 results

## Frozen conclusion

The complete decision is `NO_GO_EVENT_GRAPH_ORACLE_SUFFICIENCY_V12`. Eleven of twelve preregistered clauses pass; the future four-cell outcome clause fails. Therefore v12 does **not** authorize training a learned event-graph predictor under its original shared-hidden architecture.

This is nevertheless the first post-v6 representation with strong, replicated task-disjoint multi-step action evidence. The appropriate direction change is modular head isolation, not another generic latent encoder.

## Data gate

Two independent deterministic builds produce byte-identical 6,763-row datasets and audits, aligned to all 2,060 trajectories and 20 tasks. The explicit catalog contains 263 value-anonymized features. Every non-finish action has exact tool identity; seven receipt formats are represented; the largest repeated graph signature is only 2.17%. All task/attack outcomes, target skills, probability labels, policy-violation labels, task IDs in features, raw values, and runtime IDs are excluded.

## Oracle-sufficiency experiment

- Five task-disjoint folds × seeds 7/17/29.
- Fifteen v5 teacher fits, fifteen equal-capacity zero-graph fits, fifteen true-event-graph fits.
- Both trained residual arms have exactly 126,151 parameters.
- Each of v6, zero-graph, and true-graph arms contains 41,433 paired confirmation rows.
- Slurm 7107, zero runtime failures, 8/8 tests, empty stderr, all frozen and archive checksums verified.

Positive effects favor the true-event-graph arm.

| Metric | Effect | Evidence |
|---|---:|---|
| h1 NLL vs v6 | +0.000960 | noninferiority pass |
| h1 accuracy vs v6 | +0.005818 | CI [+0.000898,+0.013088] |
| h2–h5 NLL vs v6 | +0.036183 | CI [+0.015641,+0.054255], 18/20 tasks, 3/3 seeds |
| h2–h5 NLL vs zero-graph capacity | +0.038463 | CI [+0.025313,+0.050221], 18/20 tasks |
| future four-cell CE vs v6 | -0.008871 | fail; 7/20 tasks improve |

The action effect is not a capacity artifact: the parameter-matched zero-graph arm is materially worse. The remaining failure is localized to the shared future-joint head. The graph-conditioned hidden state is useful for next-action dynamics but shifts the representation away from the trajectory-level task/attack outcome statistic.

## Direction change

Retain v6 as the deployed baseline and reject v12's shared-hidden replacement. The next sequentially exploratory candidate will use two paths:

1. an action-event-graph branch for next-action and free-rollout dynamics;
2. an independent retained v6 branch for four-cell task/attack outcomes.

The new branch must first demonstrate that modular recombination preserves the strong v12 action effect while eliminating joint-outcome degradation. Only then may an event-graph predictor replace true future graphs. This is a new protocol and does not relax or reinterpret the v12 result.

## Provenance

- Build commit: `68a8bae4ff8af724e7046b94294813f34295c0f4`
- Run commit: `a7c4d6c0fb31ac40b9d5b5d501dbfe191007d437`
- Data archive: `/share/guozhix/wmagentattack/0818/action_event_graph_v12/data_gate/formal_v1`
- Oracle archive: `/share/guozhix/wmagentattack/0818/action_event_graph_v12/oracle_sufficiency/formal_v1`
- Prediction SHA256: `50547423cd0ee6392fdfd9aa26bbf5cc2e974e55e49db49b4857ddb82f4760b5`
- Oracle gate SHA256: `c0a3b155e4bff9e9b21e1300f950907da6f7e372c80588c506d217e68f4a0625`
- Archive manifest SHA256: `8ac6c2c039e42d8a13a300771a1e3a68c6fb06b9db6fe85ccd6b8362bacbc328`
