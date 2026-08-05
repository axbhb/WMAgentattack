# Multi-source current-method suitability: frozen results

## Scientific conclusion

The frozen decision is `NO_SOURCE_PASSES_CURRENT_METHOD_ACTION_GATE__DO_NOT_SCALE`.
None of ToolSandbox, InjecAgent, or tau3 should be expanded immediately into a
large core world-model dataset using the current one-step adapter and
Structured Semantic State v3 model.  This is a method/data-form decision, not a
claim that the original benchmarks are intrinsically unusable.

The run used the preregistered task-disjoint splits, three fixed seeds, 36
neural fits, four TF-IDF fits, and four frequency fits.  It produced 52,074
predictions with zero runtime failures, zero new LLM calls, zero new tool
executions, zero attack generation, and zero Dreamer runs.  Slurm job `6399`
completed at `2026-08-05T11:38:04+00:00`; every archived checksum verifies.

## Confirmation metrics

| Scope | Frequency NLL / accuracy | TF-IDF NLL / accuracy | Semantic Markov NLL / accuracy | Structured v3 NLL / accuracy | Full history NLL / accuracy | Frozen action gate |
|---|---:|---:|---:|---:|---:|---|
| ToolSandbox | 0.5178 / 0.8929 | 1.3687 / 0.8929 | 0.8053 / 0.8095 | 0.8638 / 0.7857 | 0.7935 / 0.7857 | FAIL |
| InjecAgent | 1.0222 / 0.8347 | 0.6464 / 0.8347 | 0.6586 / 0.8172 | 0.9871 / 0.6297 | 0.8510 / 0.6929 | FAIL |
| tau3 | 1.2359 / 0.6471 | 2.7773 / 0.2059 | 0.9864 / 0.7353 | 1.0739 / 0.6471 | 1.0057 / 0.7353 | FAIL |
| Combined | 0.9684 / 0.7277 | 2.0792 / 0.4392 | 0.8556 / 0.7564 | 0.9307 / 0.7212 | 0.8530 / 0.7639 | FAIL |

The acceptance arm is Structured v3.  Semantic Markov and full history are
frozen counterevidence, not alternatives selected after seeing the result.

## Source-level interpretation

### ToolSandbox: no-go for scaling

The confirmation surface contains 16 rows from 14 tasks; 14/16 targets are the
text action.  The frequency baseline therefore already reaches 0.8929
accuracy.  Structured v3 is worse by 0.3460 NLL and 0.1071 accuracy.  Although
10/14 tasks have a positive NLL gain, three tasks lose more than 0.5 NLL and
the paired bootstrap interval is `[-0.9893, 0.1897]`.  A few large failures
dominate the mean, so majority-task wins do not justify scaling.  Exact outcome
support is also too small to authorize the execution-error probe.

Required change: collect a less text-dominated, action-balanced multi-step
surface with exact adjacent execution states before retesting.  More replicas
of the current one-step distribution would mostly strengthen the prior.

### InjecAgent: auxiliary robustness data only

The confirmation set has 496 rows but only four held-out trusted-goal task
units; 83.5% of targets are text.  Structured v3 improves NLL over frequency by
only 0.0351 (below 0.05), loses 0.2050 absolute accuracy, and trails the TF-IDF
baseline by 0.3407 NLL.  Structured v3 is worse than Semantic Markov on all four
confirmation tasks.

The 248 clean/poison confirmation pairs provide a useful intervention surface:
the target action changes in 27.0% of pairs.  However, Structured v3 changes its
prediction in only 8.1%-12.9% across seeds, showing that it under-reacts to the
clean/poison observation difference.  InjecAgent therefore remains valuable
for robustness and intervention evaluation, but not as the core transition
distribution for the current world model.  Observation-only records also
cannot supervise exact execution dynamics.

### tau3: strongest candidate, but still no-go under the current method

tau3 has the clearest nontrivial action signal.  Structured v3 improves NLL
over frequency by 0.1619, does so in all three seeds, and wins 24/34 paired
tasks (exact sign-test `p=0.0243`).  Nevertheless, action accuracy does not
improve at all on average, violating the frozen accuracy gate.  The paired NLL
bootstrap interval `[-0.2120, 0.5078]` includes zero and seven tasks lose more
than 0.5 NLL.

The simpler Semantic Markov arm is materially better than Structured v3
(0.9864 vs 1.0739 NLL; 0.7353 vs 0.6471 accuracy), and Structured v3 is worse
than Semantic Markov on 73.5% of tau3 tasks.  This is direct evidence that the
current structured adapter adds harmful abstraction or sparsity on this
one-step surface rather than merely needing more rows.

tau3 alone authorizes the exact execution-error probe, but that probe also
fails: frequency BCE is 0.4866 versus Structured-v3 BCE 0.5088.  All three seed
gains are negative (`-0.0043`, `-0.0324`, `-0.0300`).  The model separates
errors from successes weakly, but assigns only 0.18-0.27 mean error probability
to actual errors.  Its Brier score improves while BCE worsens, so a few
high-confidence or poorly calibrated cases remain consequential.

tau3 is the best source for a redesigned, small, action-balanced, adjacent-step
pilot.  It is not authorized for immediate large-scale collection with the
current Structured-v3 endpoint.

## Combined training and counterevidence

The combined Structured-v3 model does not materially degrade any source under
the preregistered 0.05-NLL threshold: the combined-minus-source-specific gaps
are -0.3515 for ToolSandbox, -0.0639 for InjecAgent, and +0.0298 for tau3.
Thus source mixing is not the main bottleneck.  Even so, the combined action
gate fails: its NLL gain over frequency is only 0.0377, its accuracy is lower,
the paired bootstrap interval is `[-0.2102, 0.2732]`, and the sign-test is
`p=0.3317`.  The combined execution-error probe passes, but an error head cannot
rescue a failed action model.

Counterevidence is mixed rather than uniformly negative.  TF-IDF is strong on
InjecAgent but weak on ToolSandbox and tau3, so lexical memorization is not a
complete explanation.  Semantic Markov and full history consistently beat the
accepted Structured-v3 arm on the combined and tau3 surfaces, which points to
representation mismatch as the more actionable issue.

## Why full world-model scaling is structurally blocked

All 2,367 records are one-step decisions.  The three sources contain zero
adjacent semantic transitions.  Clean/poison pairs are interventions, not
successive states.  Consequently, the five-label evidence-transition head and
free-running rollouts cannot be trained or validated from this dataset, and
every source fails the full-world-model gate independently of action metrics.

The next authorized research unit is therefore a small data-form repair, not a
large training run:

1. Use tau3 first and retain InjecAgent only as a paired robustness evaluation.
2. Record adjacent pre-action and post-action semantic states, exact receipts,
   legal action sets, and execution outcomes for each step.
3. Balance text/tool targets and recurrent action classes at the task level.
4. Compare the simpler Semantic Markov representation against a revised
   structured adapter on the same tasks and seeds before any scale-up.
5. Require action accuracy, NLL, execution-error calibration, and multi-step
   rollout consistency to pass jointly; only then authorize a larger dataset.

## Reproducibility

- Remote archive: `/share/guozhix/wmagentattack/0806/multisource_method_suitability/fixed_v1`
- Git commit used by Slurm: `bbef0e5250cb328cab420d26c2c6af03b137fc95`
- Preregistered protocol SHA256: `6dc32945e66bde87d90b1e49fb3ee6ef67a56b3ff8f8506130e6c81c9977e840`
- Summary SHA256: `20606cfdcf983af85c23b4edb7d7b9e3e02d357d14ae6d9a329224a5ee72e22d`
- Predictions SHA256: `86090d4781eaeea8ae4ad18caf4abd882caf7d16f658a841984a6824c8d2c531`
- Archive checksum verification: PASS
