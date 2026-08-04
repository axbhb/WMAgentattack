# 0727 custom clean panel preregistration

## Objective

Replace the exhausted stock-task development pool with a task- and template-disjoint clean panel. This panel is the data-sufficiency gate that precedes any Ledger-v2 model comparison.

## Panel construction

The panel contains 24 synthetic AgentDojo tasks, balanced across Banking, Slack, Travel, and Workspace. Each split contains two tasks per suite:

| Split | Frozen semantic families |
|---|---|
| Training | direct lookup projection; explicit state mutation |
| Calibration | pairwise entity comparison; observed condition then action |
| Confirmation | cross-source join action; multi-constraint entity selection |

No semantic family, task ID, or prompt crosses a split. Tasks and checkers are frozen before any victim outcome. The task source uses only AgentDojo's in-memory environments and tools.

The authors had access to prior stock-task development results and to the scaffold-screen decision. This is disclosed development context. No outcome from any of the 24 custom tasks was available while their prompts, split assignments, ground-truth calls, or checkers were authored.

## Victim scaffold

The scaffold is fixed from the completed screening result:

- Meta-Llama-3.1-70B-Instruct, 4-bit;
- `constraint_checklist` prompt profile;
- `function_tags_repair_retry` parser;
- greedy decoding;
- 8,192 input tokens, 256 output tokens, and 12,000 tool-output characters.

Greedy decoding means different seeds may not induce different actions. The six-seed design is retained to detect runtime nondeterminism and enforce separate development/confirmation execution records; task/template separation is the primary generalization barrier.

## Fixed clean budget

- Development seeds: 233, 239, 241.
- Confirmation seeds: 251, 257, 263.
- 24 tasks per seed.
- 144 clean episodes in total.
- 0 attack episodes.
- 0 model-training runs.

## Data-sufficiency gate

Dynamics/progress research requires all of the following:

- 144 complete episodes with zero runtime, trace, or pairing failures;
- at least 8 tasks retained under both seed panels;
- at least 3 durable tasks in each of training, calibration, and confirmation;
- at least 1 durable task in Banking, Slack, and Workspace;
- at least two suites with two or more durable tasks.

Travel remains a reported stress/OOD suite and is not a blocking core-suite condition.

The completion/value head has an additional balance gate: training and confirmation must each contain at least three durable-success tasks and at least two tasks that fail all six runs. If this balance gate fails, only dynamics/progress modeling may proceed.

## Claim and action boundary

Confirmation-task outcomes cannot tune the scaffold, tasks, checkers, representation, or architecture. Failed tasks cannot be removed. Any revision creates a new panel version. Until later clean forecasting gates pass, attack data generation and Dreamer training remain prohibited.
