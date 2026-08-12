# AgentDojo observed adjacent-transition repair results

Decision: `GO_RETAIN_OBSERVED_ADJACENT_TRANSITION_OBJECTIVE`

The frozen AgentDojo-v2 traces yielded 6,763 observed action events from 2,060
trajectories, including 4,703 real adjacent transitions and 1,253 multistep
trajectories. All 20 tasks contain adjacent transitions; 19 tasks contain an
execution error, for 374 error events. Two independent builds were byte
identical and all causal-boundary checks passed.

The formal five-fold OOF experiment compared the same transition model trained
with next-action loss alone against next-action plus observable-outcome loss.
It used variants Semantic Markov and Structured Markov v3, seeds 7/17/29, and
60 total runs.

| representation | tail NLL gain | tail accuracy gain | outcome BCE gain vs prior | error BCE gain | positive tasks |
|---|---:|---:|---:|---:|---:|
| Semantic Markov | +0.035664 | +0.011359 | +0.177766 | +0.009756 | 65% |
| Structured Markov v3 | +0.028608 | +0.002233 | +0.213377 | +0.060251 | 75% |

All nine frozen clauses passed. Semantic Markov tail-action NLL improved in all
three seeds; accuracy improved in two. Its paired task bootstrap interval
`[-0.006547, 0.093207]` crosses zero and the exact sign test is 13 wins versus
7 losses (`p=0.2632`), so heterogeneous task effects remain counterevidence.

The retained method now has a genuine action-conditioned dynamics component:
given causal state and the selected action, it predicts the next victim action
and the observable probabilities of execution error, nonempty output, and
trajectory continuation. It does not predict task utility, security, hidden
simulator state, or counterfactual actions.

- Slurm: `6734`; zero runtime failures and empty stderr.
- Tests: 15 passed; 60 runs and 81,156 prediction rows.
- Archive: `/share/guozhix/wmagentattack/0813/adjacent_transition/formal_v1`
- Dataset SHA256: `066b69fb580634ac5354c4aa59463b86ff26fadfc3f7436b6e0646f3c0cafa5f`
- Summary SHA256: `1b1152e0af5403d67fb3a331c8d361a6155a55fbc6c3d6666b938c0adc06f5ae`
- Predictions SHA256: `5821ea63d36f4b64f400ec7ca3e0d6f49b40a7dee38cf02cdaf5f7e063bc31e9`
- Full archive checksum verification: PASS.
