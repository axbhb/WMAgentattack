# Autoresearch Stage 3 preregistration: Markov sufficiency

Date: 2026-08-04

Run tag: `aug4-semantic-wm-v3`

Status: frozen before model training

## Fixed question

The experiment tests whether Structured Semantic State v3 adds task-disjoint predictive signal over the old Semantic Markov representation and preserves essentially the same predictive information as a causal full visible-history diagnostic. Both victim next-action dynamics and candidate-conditional next-evidence deltas must pass; a single-head improvement is insufficient to authorize attack construction.

## Fixed controls

Exactly three representations are allowed: `semantic_markov`, `structured_markov_v3`, and `full_history_diagnostic`. Their vectors have the same dimension. They use the same hybrid learned heads, tool descriptors, legal masks, train-only task-balanced loss, optimizer, 120 epochs, and seeds 7/17/29. No hyperparameter grid, early stopping, calibration selection, or post-result rerun is allowed.

The full-history arm contains only actions, tool observations, and execution receipts visible at or before the current prefix. Hidden simulator deltas, future rows, expert/proof contracts, final outputs, utility/security labels, and attack labels are forbidden.

## Fixed gate

On the 12 task-disjoint confirmation tasks, v3 must satisfy all of the following:

- mean action-NLL gain over Semantic Markov at least 0.02;
- that action threshold reached by at least two of three seeds and positive paired gain on at least 6/12 tasks;
- mean evidence-BCE gain over Semantic Markov at least 0.01;
- that evidence threshold reached by at least two of three seeds and positive paired gain on at least 6/12 tasks;
- action NLL no more than 0.05 worse than full visible history;
- evidence BCE no more than 0.02 worse than full visible history.

Paired task bootstrap intervals and exact sign tests are mandatory counterevidence but are not additional hard gates because only 12 independent confirmation tasks are available.

## Known limitations before results

- The confirmation tasks are identity-disjoint from training, but their identities/results were examined in earlier July representation experiments. They are not a pristine never-used final test set.
- Evidence labels are strongly imbalanced, so unweighted proper BCE/Brier and every component label will be reported even though training uses capped train-only positive weights.
- This is a 48-task clean-only synthetic-panel comparison. Passing would authorize only a small paired sandbox pilot, not large attack data, value learning, Dreamer, or a general security claim.
