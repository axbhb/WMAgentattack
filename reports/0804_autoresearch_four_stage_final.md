# WMagentattack four-stage autoresearch final report

Date: 2026-08-04

Branch: `codex/autoresearch-aug4-semantic-wm-v3`

Overall decision: `ACTION_SIGNAL_RETAINED__EVIDENCE_GATE_NO_GO__ATTACK_PILOT_NOT_AUTHORIZED`

## Research contract

The loop adapted the fixed-budget, fixed-evaluation, keep/discard ledger pattern from `karpathy/autoresearch` to the existing WMagentattack research constraints. It used an isolated local branch and remote worktree, preserved all prior dirty worktrees and July archives, recorded negative evidence, and permitted no attack generation or Dreamer training before a clean representation gate.

## Four-stage outcome

| Stage | Question | Decision | Main evidence |
|---|---|---|---|
| 1 | Can a causal, label-blind, entity-preserving semantic state be built reproducibly? | GO | 144 episodes/467 states; zero leakage or hidden-oracle dependence; two byte-identical builds |
| 2 | Can exact observed updates be composed with learned victim/evidence heads without a value/planner? | GO | 323/323 exact transitions; finite gradients; zero illegal action mass; no value/planning heads |
| 3 | Is v3 task-disjoint sufficient relative to Semantic Markov and full visible history? | NO-GO | Action NLL improves 0.1568, but evidence BCE worsens 0.0128 and all three evidence seeds regress |
| 4 | May a paired sandbox attack-selection pilot run? | NOT_AUTHORIZED | The conditional Stage 3 evidence gate failed; zero attack examples and zero victim calls |

## Retained method

The retained research architecture is not DreamerV3. It is a hybrid semantic model:

1. deterministic Structured Semantic State v3 built from the trusted goal, legal tools, executed actions, observed receipts/output summaries, and entity-preserving evidence records;
2. exact state bookkeeping after an actual AgentDojo sandbox observation;
3. a learned legal-candidate victim-action/argument head;
4. an experimental candidate-conditional evidence-delta head.

The first three components are technically valid. The action head has a repeatable directional signal. The evidence head is not accepted as task-general, so the assembly is not yet a sufficient world model for attack selection.

## Binding empirical findings

- Structured v3 confirmation action NLL is `3.726833`, versus `3.883626` for Semantic Markov and `3.981574` for full history. All three seeds exceed the action gain threshold, but the paired-task interval crosses zero and one Slack task drives much of the mean.
- Structured v3 confirmation evidence BCE is `0.286989`, versus `0.274206` for Semantic Markov and `0.279245` for full history. All three structured-versus-semantic seed gains are negative.
- Calibration independently shows the same evidence problem: `0.380448` for structured v3 versus `0.288776` for Semantic Markov.
- Training action accuracy is about 0.96 for every arm, while confirmation accuracy is only 0.25–0.31. The dominant limitation is task-identity generalization, not optimizer convergence.
- Confirmation evidence support is inadequate: `record_added` is 112/112 positive, while conflict/error/ambiguity labels have only 1/4/4 positives. Only `goal_term_newly_matched` shows a positive v3 gain.
- The action-improves/evidence-regresses pattern independently repeats the July architecture ablation despite a different state and evidence target.

## Current problems

1. **Evidence target degeneracy.** Several labels are constant or nearly absent, so the model cannot learn calibrated transition structure.
2. **Relational structure is still weak.** Hashed entity records do not explicitly represent candidate × constraint × source × coverage/uniqueness relations.
3. **Too few independent training identities.** There are 24 training tasks; larger networks fit them but generalize poorly.
4. **Confirmation is not pristine.** It is task-disjoint but was examined in earlier July studies, so it cannot serve as a final never-used test.
5. **Exact transition is observation-conditioned.** It guarantees bookkeeping after sandbox execution; it does not imagine unseen tool outputs.
6. **One shared representation/head objective is not enough.** Action benefits from semantic structure while evidence consistently prefers the simpler baseline.
7. **Dreamer remains mismatched.** A latent rollout/value framework cannot repair missing, sparse, or poorly defined semantic evidence transitions.

## Next research loop, not executed here

The next loop should be separately preregistered and should remain clean-only:

1. define a relational evidence transition target over candidate entities, goal constraints, source links, coverage, uniqueness, contradiction, and error recovery;
2. construct a larger independent clean panel with explicit minimum positive support for every evidence event and a new untouched confirmation set;
3. compare a head-specific relational evidence encoder against Semantic Markov and simple calibrated frequency/logistic baselines;
4. require both proper-score improvement and per-task replication before reopening the same-seed paired attack pilot;
5. consider utility/value learning only after that gate; consider Dreamer only if learned multi-step stochastic rollouts add value beyond exact sandbox replay and simpler sequence models.

## Artifacts

- Stage 1 report: `reports/0804_autoresearch_stage1_structured_semantic_state_v3.md`
- Stage 2 report: `reports/0804_autoresearch_stage2_hybrid_semantic_world_model.md`
- Stage 3 report: `reports/0804_autoresearch_stage3_markov_sufficiency_results.md`
- Stage 4 report: `reports/0804_autoresearch_stage4_paired_attack_pilot_not_authorized.md`
- Experiment ledger: `research/autoresearch/aug4-semantic-wm-v3/results.tsv`
- Remote root: `/share/guozhix/wmagentattack/0804/autoresearch_semantic_wm_v3`
