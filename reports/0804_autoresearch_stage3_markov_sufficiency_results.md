# Autoresearch Stage 3: Markov-sufficiency results

Date: 2026-08-04

Run tag: `aug4-semantic-wm-v3`

Slurm job: `6288`

Decision: `NO_GO__STRUCTURED_MARKOV_V3_SUFFICIENCY_NOT_ESTABLISHED`

## Execution integrity

The frozen protocol ran exactly the preregistered three representations and seeds 7/17/29, for nine training runs. It used only the 24-task training split for gradients and evaluated the fixed 12-task calibration and 12-task confirmation splits without selection or reruns.

- Protocol SHA256: `aa1dd2e5abc674f39ff0e20d9e27fc32e8174eda25adc3a4018822b32c231778`
- Raw dataset SHA256: `c01a0c29a8e2ce99e6f3dd81c82f1711131868ebe81da62a0298fe7e42745746`
- Semantic dataset SHA256: `099aecb7ab70b5d822f44e9007fb95c1d6ed78569505dbc32621efdd88f82a53`
- Final summary SHA256: `db37c1418afdd54d627b9f96d978e673f4813ea0122a92f5686c1ab8dfcbd3c5`
- Run metrics SHA256: `b9b1760cc72a10556ce404fc6f5318ca2f67900eddd9523101a56113a2add32e`
- Predictions SHA256: `261dedd649dfa5979f180dbcac85ecc13d35b5f87599971e7fa7f55e3ff5048c`
- Prediction rows: 7,110 / 7,110
- Runtime: 28 seconds, zero failures, empty stderr
- Victim-model calls / attacks / Dreamer runs: 0 / 0 / 0
- Every archived checksum and post-run code hash verifies.

Remote archive:

`/share/guozhix/wmagentattack/0804/autoresearch_semantic_wm_v3/stage3/fixed_v1`

## Primary results

Task-macro means over the three seeds:

| Split | Representation | Action NLL ↓ | Action accuracy ↑ | Evidence BCE ↓ | Evidence Brier ↓ |
|---|---|---:|---:|---:|---:|
| Training | semantic_markov | 0.389156 | 0.964098 | 0.072930 | 0.011951 |
| Training | structured_markov_v3 | 0.400511 | 0.961057 | 0.071805 | 0.011454 |
| Training | full_history_diagnostic | 0.386338 | 0.963008 | 0.070534 | 0.011235 |
| Calibration | semantic_markov | 3.455404 | 0.259722 | 0.288776 | 0.079236 |
| Calibration | structured_markov_v3 | 3.351649 | 0.295255 | 0.380448 | 0.115093 |
| Calibration | full_history_diagnostic | 3.374515 | 0.272685 | 0.309746 | 0.088265 |
| Confirmation | semantic_markov | 3.883626 | 0.264614 | 0.274206 | 0.076320 |
| Confirmation | structured_markov_v3 | 3.726833 | 0.311625 | 0.286989 | 0.083802 |
| Confirmation | full_history_diagnostic | 3.981574 | 0.254985 | 0.279245 | 0.079952 |

## Frozen gate evaluation

### Victim-action dynamics

Structured v3 improves confirmation action NLL by `0.156794` over Semantic Markov. All three seed gains exceed the frozen 0.02 threshold (`0.055459`, `0.252377`, `0.162546`), and 7/12 paired tasks improve. It also beats full visible history by `0.254741` NLL. All four action clauses pass.

The required counterevidence weakens the breadth of this result: the paired-task bootstrap interval is `[-0.055570, 0.485609]`, the exact sign test is 7 wins versus 5 losses (`p=0.7744`), and one Slack task contributes a gain of `1.7522`. The action signal is real enough to retain as a hypothesis, but it is not uniformly distributed across tasks.

### Evidence dynamics

Structured v3 worsens confirmation evidence BCE by `0.012784` relative to Semantic Markov. Every seed is negative (`-0.015835`, `-0.011130`, `-0.011386`), so the mean-gain and seed-replication clauses fail. Exactly 6/12 tasks improve, which passes only the paired-task count clause. The bootstrap interval is `[-0.044858, 0.018414]` and the sign test is 6 wins versus 6 losses (`p=1.0`).

Structured v3 is within the frozen non-inferiority margin of full history (`+0.007744` BCE), but full history is itself worse than the simpler Semantic Markov baseline. Thus similarity to full history does not rescue the incremental evidence gate.

The component labels expose severe support problems on the 112 confirmation transitions:

| Evidence delta | Positive rows | Structured-minus-Semantic paired task BCE gain |
|---|---:|---:|
| record_added | 112 | -0.019808 |
| goal_term_newly_matched | 72 | +0.034848 |
| conflict_added | 1 | -0.028328 |
| execution_error | 4 | -0.016931 |
| ambiguous_or_unlinked_added | 4 | -0.033698 |

`record_added` is constant on the complete confirmation surface, while three failure/ambiguity labels have only one to four positives. The only component with a positive mean gain is `goal_term_newly_matched`.

## Cross-check against the July architecture ablation

The independent target redesign reproduces the earlier qualitative pattern in `reports/0729_custom_panel_v2_architecture_ablation_results_v2.md`: richer observable state improved confirmation action NLL there by `0.2013`, but worsened evidence NLL by `0.1205`. In this Stage 3 experiment, structured v3 improves action NLL by `0.1568` and again worsens evidence proper loss, now by `0.0128` BCE. Because the evidence targets and representation differ, the magnitudes are not directly comparable; the repeated cross-head reversal is the relevant counterevidence.

## Interpretation

1. Structured semantic state is useful for predicting the victim's next tool choice, and its advantage over full raw history suggests the semantic compression removes some irrelevant textual noise.
2. The same state/head does not establish task-disjoint evidence dynamics. Calibration evidence BCE deteriorates even more strongly (`0.380448` versus `0.288776`), so the confirmation failure is not an isolated split fluctuation.
3. All arms reach about 0.96 training action accuracy but only 0.25–0.31 confirmation accuracy. The dominant bottleneck remains task generalization from 24 training identities, not optimization.
4. The current evidence-delta formulation is partly degenerate and extremely sparse. A generic MLP can fit the training rows but cannot learn robust failure, contradiction, or ambiguity transitions from this panel.
5. This result supports retaining v3 as an action-dynamics research component, not accepting it as a complete world-model state and not using Dreamer as the main framework yet.

## Binding decision

The preregistered gate required both action and evidence increments. Two evidence clauses fail, so Stage 3 is a binding NO-GO. No threshold is changed and no result-conditioned rerun is submitted.

Stage 4 attack construction is therefore `NOT_AUTHORIZED`. The next scientifically defensible research loop, if separately preregistered later, is to collect substantially more independent tasks and informative contradiction/error/ambiguity transitions, replace degenerate evidence labels with a relational candidate-by-constraint state, and evaluate a head-specific evidence encoder on a genuinely fresh confirmation surface.
