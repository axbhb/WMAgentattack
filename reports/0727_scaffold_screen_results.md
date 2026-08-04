# 0727 clean-only scaffold screen

## Decision

The frozen screen selected `constraint_checklist_greedy` for construction of the new custom clean panel:

`SCAFFOLD_SCREEN_SELECT_CONSTRAINT_CHECKLIST_GREEDY`

This is a scaffold-selection result only. It is not an independent task-confirmation result, a world-model result, or permission to construct attack data.

## Frozen design

- Model: Meta-Llama-3.1-70B-Instruct, 4-bit.
- Tool protocol: `function_tags_repair_retry`.
- Screening tasks: 16 stock AgentDojo tasks, four per suite.
- Seeds: 151, 157, and 163.
- Candidates: base/checklist crossed with sampled/greedy decoding.
- Budget: 192 clean episodes; zero attack episodes and zero model-training runs.
- Retention: at least two successes out of three runs per task.
- Replacement gate: at least two additional retained tasks, at least three additional successes, and no suite-level retained-task regression relative to `base_sampled`.

The first 24-element Slurm layout was not submitted because the controller's global `MaxJobCount=24` had one occupied record. A scheduler-only v2 packed two frozen candidates into each of 12 array elements. The manifest, model, prompts, seeds, candidate settings, episode budget, and selection gate were unchanged.

## Results

| Candidate | Successes / 48 | Retained / 16 | Zero-tool failures |
|---|---:|---:|---:|
| `base_sampled` | 21 | 6 | 5 |
| `base_greedy` | 18 | 6 | 3 |
| `constraint_checklist_sampled` | 21 | 7 | 3 |
| `constraint_checklist_greedy` | **27** | **9** | **0** |

Against `base_sampled`, `constraint_checklist_greedy` produced 7 paired wins, 1 loss, and 40 ties. It gained 6 successful episodes and 3 retained tasks and did not reduce the retained-task count in any suite, so it passed every preregistered material-gain condition. The exact two-sided sign-test value was 0.0703125; this was a diagnostic, not a post-hoc additional gate.

Retained tasks for the selected scaffold were distributed as follows:

- Banking: 1 of 4.
- Slack: 4 of 4.
- Travel: 1 of 4.
- Workspace: 3 of 4.

All 192 episodes completed with raw traces. There were no missing pairs, runtime failures, trace failures, OOMs, CUDA errors, or detected Tracebacks.

## Counterevidence and limitations

1. The paired sign test does not cross 0.05, so this screen is not strong statistical confirmation by itself.
2. The improvement is concentrated in Slack and Workspace. Banking and Travel each retain only one task.
3. The sampled checklist did not improve total success over the sampled baseline, showing that the prompt improvement depends on the decoding regime.
4. Under greedy decoding, every task was either 0/3 or 3/3. The seed split is therefore effectively a replay-consistency check, not evidence of stochastic robustness.
5. All 16 stock tasks have prior benchmark-development exposure and are barred from final model confirmation.

## Next frozen step

Use the selected scaffold only to collect clean trajectories on `wmagentattack_custom_clean_panel_v1`:

- 24 new custom tasks;
- 8 training, 8 calibration, and 8 confirmation tasks;
- two tasks per suite in each split;
- semantic template families disjoint across splits;
- separate development and confirmation replay seeds;
- no post-hoc task deletion or checker edits.

Before any 70B run, every task must pass tool-schema validation, executable ground-truth utility, a negative checker test, split-disjointness checks, and label-blind manifest checks. Attack data and Dreamer remain blocked.
