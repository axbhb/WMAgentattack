# Retrieval successor world model v16 preregistration

The neural future-state path is stopped after v13 and v15. v16 indexes only training-task transition anchors and retrieves observed successor-action targets for held-out tasks. Queries combine normalized Structured Markov v3 state, current value-anonymized event graph, and current action embedding. No raw text, outcome label, confirmation event, or future query field enters the index.

Support is calibrated without confirmation labels: for each fold and horizon, the threshold is the 25th percentile of training leave-one-trajectory-out top-1 similarity. Only supported confirmation rows can authorize a later conservative fusion with v6. The retrieval model uses fixed k=32, temperature 0.10, and legal-action masking. No hyperparameter sweep or post-result rerun is allowed.
