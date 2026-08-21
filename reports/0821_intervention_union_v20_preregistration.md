# Intervention union v20 and model preregistration

Date: 2026-08-21

Status: union gate passed; model protocol frozen before any v20 model result.

The union contains the already frozen v17 legal-action forks, v18 parameter-boundary pairs, and v19 persistence/competing-update sequences. It is not a flat concatenation. Exact duplicate transitions are merged while all source, root, pair, and sequence memberships remain available only for grouping and audits. Identical model inputs with different targets are an immediate NO-GO.

Model inputs contain only Structured Semantic State v3 and a value-normalized action descriptor. Task IDs, suite/difficulty labels, source IDs, group IDs, raw action values, simulator audit state, utility, security, attack, and final outcomes are excluded. Parameter values are represented by type/category and whether the exact value is observable in the current semantic state. Targets are execution error, five evidence-delta bits, and value-blind semantic effect tokens.

The three confirmation folds are fixed by difficulty: L1, L2, and L3. Tasks, roots, pairs, and sequences inherit one fold and cannot cross partitions. Two independent builds must be byte-identical and pass all source-count, duplicate, conflicting-label, leakage, adjacency, and grouping checks before any model protocol is frozen.

The double build passed: 144 raw transitions became 121 canonical
transitions, 23 shared occurrences were deduplicated without target conflicts,
and the fold sizes are 40/40/41. All cross-fold and leakage findings are empty.
The dataset SHA256 is
`cf5eb6fb2e92f8175e399ca1230d40d31b3d954a99b0f32258ba9789190b50ca`;
the audit SHA256 is
`ee208b2c89b007856a1097458742c5e60f88c56efb3873c75d54342dad958cd4`.

Exactly 27 CPU fits are authorized: three arms, three frozen folds, and seeds
7/17/29. The candidate uses a zero-initialized latent residual, an
executability gate, success/error-conditional effect experts, pair-response
supervision, and closed-loop v19 sequence supervision. Group memberships are
used only to form training losses and audits; they are never model inputs.

Acceptance requires at least 0.01 lower v19 closed-loop effect BCE than the
v6-style baseline, one-step BCE and execution Brier within 0.01 of the best
baseline, pair assignment accuracy at least 0.75 and within 0.02 of the best
baseline, and rollout gains in at least two folds and two seeds. No threshold
change or post-result rerun is authorized.
