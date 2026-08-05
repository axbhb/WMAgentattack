# Candidate × Constraint Evidence Dataset pilot preregistration

Date: 2026-08-05

Status: frozen before the pilot build

## Purpose

This development-only pilot tests the normalized dataset schema proposed after the Stage 3 evidence NO-GO. It does not test a model and cannot reopen attack or Dreamer experiments.

## Fixed sample

Use only deterministic-greedy episodes from the existing training split. Select the lexicographically first task ID in each of the 12 suite × difficulty cells. Selection uses metadata only. Calibration and confirmation rows are forbidden.

The frozen source preview predicts 12 tasks, 19 observed non-terminal transitions, 19 states, 98 unique trusted-goal fact-term constraints, 203 observed candidate × constraint labels, and 3,834 unlabeled counterfactual candidate queries.

## Label contract

For the actually executed action, each trusted-goal fact term receives exactly one adjacent-state label: `ALREADY_SUPPORTED`, `NEWLY_SUPPORTED`, or `UNCHANGED_UNSUPPORTED`. `ALREADY_SUPPORTED` is fully known from the input state, so those rows are marked `STATE_CONSISTENCY_ONLY`; only initially unsupported constraints are eligible for future transition-gain training. Non-executed legal candidates are retained as `UNLABELED_COUNTERFACTUAL`, with no fabricated negative label and no invented arguments.

Proof contracts, required calls, hidden state deltas, future observations, final outcomes, utility/security labels, and attacks are excluded. Reference IDs and task metadata are grouping fields and may never be encoded as model features.

## Two gates

The schema gate requires exact counts, all three progress classes with at least 20 rows and six task identities each, zero state leakage, strict observed/unlabeled separation, deterministic rebuilding, and passing tests.

The separate training-readiness gate requires at least 25% of the legal candidate query space to have observed labels and at least five error, conflict, and ambiguity transitions. If the schema passes but readiness fails, the correct result is to freeze the schema and collect clean counterfactual outcomes; no model training is authorized.
