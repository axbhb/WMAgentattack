# v34 paired single-factor attack intervention preregistration

## Question

v33 improved aggregate top-1 selection but failed calibration, pairwise ranking,
four-cell likelihood, and cross-task consistency. v34 tests whether the missing
signal is causal pairing rather than additional world-model capacity.

## Frozen design

Eight independently 3/3 clean-solvable AgentDojo tasks are held out from all
historical training rows. Each task receives one fresh clean control and four
attacks under the same injection task, injection goal, endpoint set,
environment, victim model, and seed. `fixed_plain`, `contextual_prefix`,
`contextual_suffix`, and `structured_record` are four levels of one declared
text-carrier factor. These levels use normalized position/stage categories
already present in the historical training data. Endpoint selection is held
fixed because most stable confirmation tasks expose only one injection vector.

Three Llama-3.1-70B seeds produce 24 clean and 96 attack episodes. At most one
GPU is used concurrently. No external endpoint, payload optimizer, planner,
Dreamer, or large world-model training is permitted.

## Selector

The primary selector separately encodes pre-execution task state and attacker
action, combines them with a multiplicative interaction, and predicts all four
task/attack outcome cells through a zero-start residual. Raw goals, payloads,
task IDs, trajectories, and outcome fields are excluded. The baseline is the
v33 structured pre-execution concatenation model; neither arm can use a v5
score that requires the candidate trajectory to exist.

The model trains on 240 historical configurations from the other twelve tasks
and is evaluated only on the 32 fresh paired attack configurations. The frozen
gate requires complete clean/attack execution, identifiable intervention
effects in at least five tasks, task-level top-1 and pairwise gains, positive
effects in at least five of eight tasks, two-seed replication, calibration
non-inferiority, improved four-cell likelihood, and a gain over random.
