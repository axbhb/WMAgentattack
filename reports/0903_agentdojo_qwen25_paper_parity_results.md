# AgentDojo Qwen2.5-7B paper-parity result

## Conclusion

The frozen AgentDojo v1 evaluation completed on the friend's two V100-32GB GPUs. All 97 clean tasks and all 629 `important_instructions` security cases were evaluated with Qwen2.5-7B-Instruct, native tool calling, greedy decoding, and 4-bit NF4 quantization. There were zero episode-level runtime failures.

Qwen2.5-7B reached a targeted ASR of **19.08%**, only **0.95 percentage points below** the paper's reported Llama-3-70B value of 20.03%. Its benign utility was **29.90%**, 4.60 points below the paper reference, while utility under attack was **25.91%**, 7.63 points above the paper reference.

This establishes a complete Qwen-aligned AgentDojo attack baseline. It does not establish attack improvement by WMagentattack: that requires a paired comparison against this fixed victim and baseline attack under identical tasks and seeds.

## Aggregate results

| Metric | Qwen2.5-7B | Paper Llama-3-70B | Difference |
|---|---:|---:|---:|
| Benign utility | 0.29897 (29/97) | 0.3450 | -0.04603 |
| Utility under attack | 0.25914 (163/629) | 0.1828 | +0.07634 |
| Targeted ASR | 0.19078 (120/629) | 0.2003 | -0.00952 |
| Joint task and attack success | 0.03339 (21/629) | not reported | n/a |
| Conditional ASR given task utility | 0.12883 (21/163) | not reported | n/a |

The last two metrics are stricter diagnostics introduced in this project. They should not be compared directly with the paper's unconditional targeted ASR.

## Suite breakdown

| Suite | Benign utility | Utility under attack | Targeted ASR | Joint rate |
|---|---:|---:|---:|---:|
| Workspace | 0.3000 | 0.2708 | 0.0458 | 0.0000 |
| Travel | 0.1000 | 0.0643 | 0.2571 | 0.0000 |
| Banking | 0.3750 | 0.3958 | 0.1597 | 0.0347 |
| Slack | 0.4286 | 0.3048 | 0.4762 | 0.1524 |

The aggregate ASR hides large domain variation. Slack is highly attackable, whereas Workspace rarely satisfies the targeted attack objective. Travel has a moderate ASR but very low victim utility, so it contributes no joint successes. This supports reporting both unconditional ASR and joint/conditional metrics in later WMagentattack comparisons.

## Integrity and counterevidence

- The clean gate passed: 97/97 cases completed, zero runtime failures, and every suite had at least one clean success.
- Attack evaluation completed exactly 629/629 cases: Workspace 240, Travel 140, Banking 144, and Slack 105.
- The initial smoke attempt exposed an AgentDojo model-name compatibility restriction. The repair only registered a local alias and did not alter prompts, tasks, decoding, or outputs.
- The initial attack-worker attempt exposed a Pydantic forward-reference loading incompatibility. The repair rebuilt the stored result type namespace; the failed logs were preserved.
- No OOM, CUDA failure, or traceback occurred in the successful clean or attack runs.
- The generation library warned that sampling flags were ignored. This is consistent with the frozen greedy-decoding contract and is not a runtime failure.
- No content checksums were generated, following the repository's explicit no-checksum policy.

## Scientific interpretation

The near-paper targeted ASR shows that Qwen2.5-7B is a viable aligned victim for the next paired attack study despite its lower benign utility. However, only 21 of 629 cases simultaneously achieved user-task utility and the targeted attack objective. Optimizing unconditional ASR alone would therefore risk selecting attacks that exploit already-failing trajectories. The next attack-method experiment should use this exact evaluation as the fixed baseline and optimize/report joint success, conditional ASR, and benign-utility preservation alongside paper ASR.

## Archived evidence

- Remote archive: `/home/pth/outputs/wmagentattack/0902/agentdojo_qwen25_paper_parity/formal_v1`
- Machine-readable summary: `paper_parity_summary.json`
- Clean gate: `clean_gate.json`
- Successful worker logs and original compatibility-failure logs are retained under `logs/` and `status/`.

