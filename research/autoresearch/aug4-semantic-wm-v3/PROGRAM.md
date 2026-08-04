# WMagentattack autoresearch: semantic hybrid world model v3

Run tag: `aug4-semantic-wm-v3`

Branch: `codex/autoresearch-aug4-semantic-wm-v3`

## Objective

Determine whether a causal structured semantic state is sufficient for task-disjoint victim and evidence dynamics, then use it in a hybrid AgentDojo world model and, only if the clean gate passes, test a small paired attack-selection pilot.

## Fixed operating rules

- Preserve all prior reports and confirmation decisions.
- Develop only on training/calibration tasks; use a frozen independent confirmation surface once.
- Use task-level paired inference, identical seeds, legal-action masks, and label-blind causal features.
- Reject future/expert/final-output/utility/security leakage and hidden simulator state unavailable at decision time.
- Change one mechanism per candidate and retain it only when the complete stage gate passes.
- Record negative results, crashes, OOM, hashes, exact commands, and Slurm state.
- Use only AgentDojo's synthetic sandbox; never call real external endpoints.
- Do not start Dreamer training in this program.

## Fixed stage budgets

1. Structured state: one frozen schema plus at most three implementation repairs that do not alter semantics; unit, replay, causality, entity, and leakage audits required.
2. Hybrid world model: one exact-transition composition and at most three learned-head repairs; completion/value/planning heads remain disabled.
3. Sufficiency experiment: exactly three representations (`semantic_markov`, `structured_markov_v3`, `full_history_diagnostic`) and seeds `7/17/29`; no post-result threshold changes.
4. Attack pilot: conditional on Stage 3 GO; one frozen durable-task panel, same-task/same-seed clean/attack pairs, and four selector controls. If Stage 3 is NO-GO, record `NOT_AUTHORIZED` without generating attacks.

## Stage reports

Write one compact stage report containing integrity, data, metrics, gate clauses, counterevidence, decision, retained architecture, and next authorization. Append every candidate to `results.tsv`.
