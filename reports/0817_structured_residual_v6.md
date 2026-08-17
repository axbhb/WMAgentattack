# Structured joint zero-residual v6

Status: `PREREGISTERED_AND_FROZEN_BEFORE_TRAINING`.

This candidate starts from the retained Structured Markov v3 plus four-cell auxiliary teacher. The teacher is frozen before residual training. A zero-initialized recurrent residual receives only the structured context and action embeddings; raw goal, observation, and schema text nodes are absent. One-step predictions are anchored to the teacher with KL and a noninferiority gate. Horizons 2--5 use teacher-forced training and free-running expected-action feedback at confirmation. A latent consistency term aligns imagined hidden states with future frozen structured contexts, while a small future four-cell loss preserves the updated outcome objective.
