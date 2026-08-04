# Autoresearch Stage 1: Structured Semantic State v3

Date: 2026-08-04

Run tag: `aug4-semantic-wm-v3`

Decision: `GO__FREEZE_STRUCTURED_SEMANTIC_STATE_V3`

## Question and fixed budget

This stage asked whether the frozen clean AgentDojo panel can be represented by a deterministic, entity-preserving, causal semantic state without hidden simulator state, expert/future trajectories, proof contracts, or outcome labels. The budget allowed one schema and at most three non-semantic implementation repairs; it allowed no model training, victim-model calls, or attack generation.

The frozen protocol is `configs/0804_structured_semantic_state_v3_protocol.json` with SHA256 `913768b033fe1dfdeaf2aac44462b4b9270060590b66efe769484eee2b2b80a4`.

## Retained representation

`src/wmagentattack/semantic_state_v3.py` retains only information available at the decision prefix:

- trusted user goal parsed into generic lexical, logical, comparison, coverage, uniqueness, and typed-mention fields;
- policy track, prefix index, legal action identifiers, and the previous canonical action;
- observed execution status, error/retry summary, and output shape statistics;
- entity-separated evidence records, attributes, conflicts, source tools, and source arguments;
- lexical links between observed evidence and fact terms in the trusted goal.

The builder accepts old hidden fields only for input compatibility and proves that changing them does not change the emitted state. It rejects unknown top-level features and future, expert, checker, target, reward, utility, security, proof-contract, and final-output fields.

## Frozen data and integrity

- Source: `/share/guozhix/wmagentattack/0729/custom_panel_v2_architecture_ablation/preflight_v2/panel_v2_architecture_dataset.json`
- Source SHA256: `c01a0c29a8e2ce99e6f3dd81c82f1711131868ebe81da62a0298fe7e42745746`
- Episodes: 144, balanced as 36 each for banking, slack, travel, and workspace
- Prefix states: 467
- Unique semantic-state fingerprints: 265
- Mean serialized state size: 4305.0107 bytes
- Task counts: training 24, calibration 12, confirmation 12
- Cross-split task overlap: zero for every split pair
- Emitted-state leakage failures: 0
- Hidden-oracle invariance failures: 0
- Runtime-identifier invariance failures: 0
- Remote focused tests: 34 passed

Two independent builds were byte-identical. Both emitted dataset SHA256 `099aecb7ab70b5d822f44e9007fb95c1d6ed78569505dbc32621efdd88f82a53`.

Remote archive:

`/share/guozhix/wmagentattack/0804/autoresearch_semantic_wm_v3/stage1`

## Gate decision

Every preregistered Stage 1 clause passed: 144 episodes, 467 states, task-disjoint splits, zero leakage, zero hidden-state dependence, zero runtime-ID dependence, byte-identical rebuild, and all focused tests passing. The schema is therefore frozen for Stage 2.

## Counterevidence and limits

- The reduction from 467 prefixes to 265 unique fingerprints is evidence of abstraction, not evidence of Markov sufficiency. Predictive sufficiency remains an open Stage 3 question.
- The deterministic goal parser is primarily English and lexical. It does not claim deep compositional language understanding.
- No predictive model was trained in this stage, so this result cannot support a performance-improvement claim.
- The state still averages about 4.3 KB; future compression is possible, but changing it before the frozen sufficiency comparison would invalidate the planned control.

## Authorization

Stage 2 is authorized: compose exact deterministic semantic-state updates with learned victim-action and evidence-dynamics heads. Completion, utility/value, planning, attack generation, and Dreamer training remain disabled.
