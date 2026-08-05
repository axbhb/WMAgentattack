# tau3 multi-step scale-readiness pilot results

## Decision

`DATA_FORM_NO_GO__DO_NOT_SCALE_OR_RUN_METHOD_TEST`

The frozen pilot completed normally, but only 5 of 163 adjacent exact tool
transitions changed simulator state, below the preregistered minimum of 15.
The observation-aware Semantic Markov v4 comparison is therefore not
authorized, and this dataset must not be expanded as the core world-model
training source.

## Frozen results

| Metric | Result | Gate | Result |
|---|---:|---:|---|
| Complete episodes | 96 | 96 | PASS |
| Prefixes | 259 | diagnostic | — |
| Adjacent transitions | 163 | >=100 | PASS |
| Episodes with >=2 transitions | 46 | >=36 | PASS |
| Tasks with >=1 transition | 42 | >=30 | PASS |
| Unique executed tools | 19 | >=10 | PASS |
| Tool-decision rate | 0.6525 | 0.35--0.90 | PASS |
| Dominant-action fraction | 0.1236 | <=0.65 | PASS |
| State-changing transitions | 5 | >=15 | **FAIL** |
| State-unchanged transitions | 158 | >=30 | PASS |
| Supported transition targets | 3 | >=3 | PASS |

All 96 episodes completed with zero runtime failures, zero OOM/CUDA errors,
zero nondeterministic exact replicas, zero illegal tools, zero task overlap,
no detected semantic-state label leakage, and zero real endpoint calls.

The supported transition labels were `execution_error`,
`goal_overlap_gained`, and `output_nonempty`.  `state_changed` was not usable:
training had 1 positive, calibration had 4, and confirmation had 0.

## Counterevidence and diagnosis

The failure is concentrated rather than a generic lack of tool use:

- airline: 45 transitions, 5 state changes;
- retail: 64 transitions, 0 state changes;
- telecom: 54 transitions, 0 state changes.

Every observed state change came from `cancel_reservation`.  The victim made
many valid reads, but almost never reached a successful write.  This rejects
the hypothesis that simply collecting adjacent calls from the one-agent
adapter is sufficient.

A source audit also exposed an interaction-boundary mismatch.  The official
tau3 task schema states that `UserScenario` is information sent to the user
simulator, while the official runner alternates a user simulator, agent, and
environment.  The v1 adapter instead flattened this private scenario into the
agent's initial goal and omitted the multi-turn user role.  Its label-blindness
audit passed, but it was not a faithful tau3 conversational interface.  This
is a stronger reason not to scale v1 than the mutation count alone.

The official repository describes text tau3 as half-duplex, turn-based
tool-agent-user interaction and exposes independent agent/user LLM settings;
the tau2 paper further models telecom as a dual-control Dec-POMDP with a
tool-coupled user simulator.  See the
[official tau3 repository](https://github.com/sierra-research/tau2-bench) and
[tau2 paper](https://arxiv.org/abs/2506.07982).

## Retained evidence and next mechanism

The 163 exact transitions remain useful as negative evidence and read/error
dynamics, but not as a balanced core transition source.  The next fixed
candidate will:

1. use the same frozen Llama-3.1-70B 4-bit snapshot for both roles;
2. run official half-duplex user--agent turn-taking in the in-memory sandbox;
3. keep `UserScenario` private to the user role, while the agent sees only
   policy, legal tools, user messages, and its own tool receipts;
4. support user-side tools where the task requires them;
5. replay the complete user/agent tool-call sequence twice from a fresh state;
6. compare a deterministic task/seed subset against the archived v1 results;
7. require restored state-changing support before any predictive-method test.

No attack data, Dreamer training, planner, real endpoint, or large data build
was run.

## Reproducibility

- Jobs: generation array `6404`, dependent summary `6405`
- Archive:
  `/share/guozhix/wmagentattack/0806/tau3_multistep_scale_readiness/pilot_v1`
- Frozen manifest SHA256:
  `ae31d21babab8e88b5e308fb0aec9b59a90a15ad8fc9233663168fea8c18a01e`
- Dataset SHA256:
  `9973ec2d1756e441b30f1ad03b2452a524f6e8e3d5678386f4ddfef4bb865acc`
- Gate SHA256:
  `ef5273491193b841a0a82cc6ffa91538066e5ed11d1d0fbb6b1339e8ea4a48cd`
- Result report SHA256:
  `1cce0edbfd7e6b3f1eabc2022323585f010df4fb1db3706a345c442e57754418`
