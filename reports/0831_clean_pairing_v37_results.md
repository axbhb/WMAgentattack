# v37 final result: NO_GO_CLEAN_PAIRING_V37

结论：本轮60条clean轨迹完整，成功30条，稳定任务10/20，未达到12个稳定任务及3个领域的门槛。没有启动攻击pilot或world-model训练。

## Execution and integrity

- Slurm 7562, one physical GPU 0; Llama-3.1-70B-Instruct 4bit.
- Observed start 2026-08-31 08:49:22 UTC; terminal COMPLETE marker 09:18:17
  UTC (28m55s start-to-marker). The scheduler's earlier overnight start estimate
  was superseded. No rerun or second submission occurred.
- 60/60 completed records, 60 readable raw traces, exact task/seed keys, actual
  seed alignment, all raw utility labels agreeing with results.json.
- CUDA cuInit=0, one visible device aligned to physical GPU0. No runner failures,
  raw error fields, Traceback/OOM/CUDA-error log matches, or blocked Python
  network attempts. This is an application-level offline network guard, not
  proof of OS-level egress isolation.
- Independently recomputing the **unchanged** archived gate matches every
  recorded field. The original protocol, gate, raw traces and labels are intact.
- 23 tests passed at submission/run start; 28 passed at closeout after adding
  five descriptive-audit fixture tests. The new audit neither executes episodes
  nor fits models and is explicitly post-hoc.
- Slurm purged the job from scontrol before final inspection; sacct is disabled.
  Therefore the exact scheduler exit code is unavailable. Completion follows
  from the terminal script sentinel and complete, validated artifacts; we do
  not invent an exit-code record.

## Frozen gate

| Criterion | Observed | Required | Result |
|---|---:|---:|---|
| Complete clean episodes | 60 | 60 | pass |
| Runtime failures | 0 | 0 | pass |
| Stable tasks (>=2 successes / 3 seeds) | 10/20 | >=12 | fail |
| Suites with >=2 stable tasks | 2/4 | >=3 | fail |

Clean success by seed: 601 = 9/20, 607 = 11/20, 613 = 10/20.

| Suite | Clean successes | Stable tasks | Failed, zero parsed calls | Failed despite parsed calls |
|---|---:|---:|---:|---:|
| Banking | 13/15 | 5/5 | 2 | 0 |
| Slack | 10/15 | 3/5 | 3 | 2 |
| Travel | 3/15 | 1/5 | 10 | 2 |
| Workspace | 4/15 | 1/5 | 10 | 1 |
| Total | 30/60 | 10/20 | 25 | 5 |

All twenty tasks remain in the result. Familiar tasks with new seeds are not
unseen-task confirmation. No attack-success or p11 improvement was measured
by this clean-only study.

## Descriptive failure diagnosis and counterevidence

Of 30 failures, **25 (83.33%) have zero parsed tool calls**. Thirteen of these
contain function-tag text without a parsed call; twelve contain no function
tag. For example, a Travel response emits a noncanonical function-tag/parameter
format rather than a callable message. Other failures announce an intention
to look up information and then terminate without executing that lookup.

This localizes a major observed failure mode to the victim's tool-use/protocol
boundary, upstream of world-model fitting. It does **not** establish that a
parser change would fix all these tasks. Five failures have parsed calls and
still miss the task checker; argument, planning or task-constraint mistakes
remain possible and are not inferred from a simple keyword rule.

Important counterevidence: **five successful episodes also have zero parsed
tool calls**. Tool-free completion can be valid. Never force every answer to
call a tool or relabel zero-call traces as failure. Also, two suites have good
clean performance, so this is not a universal inability of the 70B model.

## Three-round conclusion and retained method

1. v35 stopped before 45 planned fits: 11/20 tasks met pair-count coverage,
   short of 12. Reweighting old binary outcomes did not add new information.
2. v36 found 1,000/1,267 reward comparisons crossed injection goals. Only
   9/20 tasks in two suites had same-goal p11 contrast. No identical-feature
   collisions were seen among 209 contrasting pairs, which argues against
   assuming an input-collision bottleneck but does not prove encoder adequacy.
3. v37 independently confirms weak clean eligibility in the same 20 tasks:
   only 10 stable tasks, with most failures making no parsed tool call.

**No new model has been validated as better than Structured Markov.** Retain
the existing baseline and four-cell outcome formulation. The next potentially
useful research question is a separately preregistered **clean-only** comparison
of the fixed base protocol against bounded syntax repair/tool-intent handling,
holding prompts, tasks and actual episode seeds constant. Existing repair
paths in qwen_agentdojo.py are candidates, not validated improvements here.
Use new evaluation seeds, preserve legitimate tool-free answers, and separate
syntax recovery from prompt or policy changes. Do not retune against these
60 outcomes and retroactively call it confirmation.

This cycle stops with the clean NO-GO. New attacks, selector fitting, dataset
expansion and large world-model training are **NOT_AUTHORIZED by this gate**.

## Artifacts

Remote archive: `/share/guozhix/wmagentattack/0831/clean_pairing_v37/formal_v1`.
Core files: `preregistered_protocol.json`, `submission.json`, `run_plan.json`,
`cuda_preflight.json`, `results.json`, `gate.json`, `raw/`, `COMPLETE`.
Added after completion: `posthoc_failure_audit.json` and `final_report/`.
Run code version: `ea4b6e4`; descriptive audit version: `77947d4`.
Content checksums were not generated. Original archived inputs/results were
not overwritten; result metadata and reports are stored separately.
