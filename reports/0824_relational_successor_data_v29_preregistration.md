# v29 Relational Successor Data — Preregistration

## Frozen question

v28 improved whole-record recovery but failed goal-pointer and open diagnostic gates. This round tests two data hypotheses without fitting a model: (1) bind each newly matched goal-term index to the exact newly added evidence record that supplied it; (2) construct record candidates from pre-existing, outcome-blind tool adapters and AgentDojo return schemas rather than training/test outcomes.

## Fixed budget and controls

- Two deterministic builds of 121 confirmation transitions and 10 previously frozen support transitions.
- Zero model fits, GPU requests, victim-model calls, sandbox tool calls, attacks, or real endpoints.
- Existing v21 task/tool/source split refs remain unchanged.
- Raw values and raw goal terms are audit-only. The model target contains typed fields and goal-term indices only.

## Gate

Both builds must be byte-identical; all 121+10 rows and task disjointness must be preserved; every global new-goal pointer must equal the union of its record-local pointers; at least one record must bind three goal terms; static candidates must include webpage and cover every exact record signature in every task/tool/source test split with coverage 1.0; leakage and outcome-label audits must be clean.

GO authorizes only one small, separately frozen relational successor-model comparison. It does not authorize large-scale generation, attack data, Dreamer, or a planner.
