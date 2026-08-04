# 0725 clean evidence-ledger 消融结果与下一步约束

## 结论

本轮冻结实验的正式结论为：

`EVIDENCE_LEDGER_NO_INCREMENT_CLEAN_GATE_BLOCKED`

这不是运行失败，而是一个完整、可复核的 NO-GO 结果。当前 evidence-ledger v1 没有在预注册的 task-macro、分任务、三训练种子和反事实控制条件下证明增量价值。攻击数据构建与 Dreamer 大训练仍不允许启动。

## 执行完整性

- Slurm 作业：`4722`，运行时间约 7 分 55 秒。
- 数据：90 个 clean Travel episode、15 个任务、545 个前缀。
- 因果配对：456 个 proposal = 455 个 executed call + 1 个 terminal-unexecuted proposal。
- clean utility：14/90。
- 网格：8 个冻结 variant × 5 个任务分组 × 3 个训练种子 = 120/120 个运行。
- OOF 预测：13,080 条。
- 训练时没有 OOM、Traceback、运行时失败或外部端点调用。
- 作业封存时全仓测试：251 passed；最终勘误与新增诊断测试后再次验证为 254 passed。两次均只有 5 个已知 Transformer 性能提示。
- 归档：16 个文件的 SHA256 独立复核为 0 个失败；执行前后冻结源码哈希一致。

固定结果归档：

`/share/guozhix/wmagentattack/0725/clean_evidence_ledger_ablation/fixed_v1`

作业实际源码快照：

`/share/guozhix/wmagentattack/0725/clean_evidence_ledger_ablation/source_snapshot_job4722`

## 冻结网格结果

以下指标均为三训练种子平均预测后的 task-macro 结果，越低越好。

| Variant | Progress MAE | Utility Brier | Utility log loss |
|---|---:|---:|---:|
| `static_length` | 0.148961 | **0.131627** | **0.432730** |
| `state_only` | 0.140651 | 0.135470 | 0.446180 |
| `semantic_markov` | **0.133791** | 0.132244 | 0.434929 |
| `semantic_markov_state` | 0.142200 | 0.133874 | 0.440083 |
| `semantic_markov_state_evidence` | 0.153147 | 0.133823 | 0.440482 |
| `semantic_markov_state_shuffled_evidence` | 0.136688 | 0.133310 | 0.438345 |
| `semantic_markov_state_output_length` | 0.136004 | 0.134483 | 0.444333 |
| `event_transformer_state_evidence` | 0.166392 | 0.138123 | 0.457070 |

最清楚的三个现象是：

1. `semantic_markov` 的 progress 最好，加入 canonical state 后反而从 0.133791 变差到 0.142200，符合 Travel 绝大多数调用只读、state 分支稀疏的反证。
2. 加入 evidence v1 后 progress 进一步变差到 0.153147；相对 Event+State 的增益为 -0.010947，仅 4/15 个任务方向为正。
3. utility 最好的不是复杂模型，而是 `static_length`。Evidence 相对 Event+State 的 Brier 只改善 0.000051，精确配对符号翻转值为 0.986145，仅 5/15 个任务方向为正，不能解释为有效增益。

## 预注册门槛

只有完整性门槛和“utility 数值上略低于 Event+State”这一条通过；后者的变化小到没有研究意义。其余关键门槛均失败：

- Evidence 没有改善 progress。
- Evidence 没有同时击败 within-task shuffle。
- Evidence 没有同时击败 output-length control。
- 三个训练种子没有保持 progress 与 utility 同时正增益。
- mixed-outcome utility 方向仅为 5/8，低于 6/8 门槛。
- leave-one-task-out 最差 progress 增益为 -0.013975，最差 utility 增益为 -0.001745。

因此不能使用“某一个平均指标略好”覆盖预注册的多重反证。

## Semantic Markov 与 Event Transformer

在使用相同 evidence 输入时，Semantic Markov 比小型 Event Transformer 更好：progress MAE 低 0.013245，utility Brier 低 0.004299。这个结果不支持继续堆叠 Transformer 容量。

但两者都没有通过 evidence 增量门槛，且两者的 utility 均不如 static baseline。因此正确结论不是“接受 Semantic Markov+Evidence”，而是当前没有被接受的 clean architecture。冻结 `summary.json` 中原先的 `preferred_clean_architecture=semantic_markov_state_evidence` 是报告逻辑错误，不影响门槛或正式 decision；后续代码已修正为 NO-GO 时输出 `none_accepted`，原归档保持不改写。

## 事后机制诊断（不改变冻结结论）

独立诊断归档：

`/share/guozhix/wmagentattack/0725/clean_evidence_ledger_counterevidence/posthoc_v1`

诊断发现 evidence-ledger v1 本身存在两个实质问题：

### 1. 实体边界与 conflict 语义不可靠

- 共生成 1,568 个 evidence item。
- 504 个 item 被标为 conflict，占 32.14%。
- 这 504 个 conflict 全部属于 `UNLINKED` entity。
- 例如多个备选航班共享 `UNLINKED + price`，后续不同价格会被误判为同一实体价格冲突；它们实际是不同候选，而不是冲突事实。
- 32.33% item 的朴素 goal-overlap 为 0。

这说明 v1 把文本字段扁平化后丢失了航班、酒店、餐厅等记录边界，随后又把 call-level argument link 状态复制到每个 evidence item。该表示会向模型注入大量错误的冲突和实体关系。

### 2. shuffled control 保留了大量语义

冻结规则采用同任务 cyclic donor。对 545 个对应前缀：

- 非空 evidence 的 token Jaccard 中位数为 0.7111，均值为 0.6998。
- 27.17% 的非空 evidence 文本完全相同。
- 若包含空前缀，完全相同占 37.80%。

这解释了为什么 shuffled evidence 没有明显变差：相同任务的确定性工具常返回相同候选与属性，shuffle 破坏了 episode 对齐，却保留了大部分 task-specific 内容。它仍然是有效的预注册反证，所以本轮必须 NO-GO；但它也提示下一版控制需要在独立协议中更精确地区分“任务事实”与“轨迹实际获得的事实”。

### 3. 监督目标与样本量仍有限

- Expert-slot coverage 本质上由“是否执行了与 expert 完全相同的 call”构成，因而 last event 可直接解释很大一部分 target；它并不等价于语义 evidence completeness。
- Utility 只有 14 个正例，且只有 15 个独立任务，静态/长度先验已经取得最佳 Brier。当前数据不足以训练可靠的 completion/value head。
- 以上问题解释“为什么 v1 可能失败”，但不能把失败改写成 evidence 思路有效；后者必须由新的独立数据证明。

## 已完成的代码修改

- 新增确定性 evidence ledger 与泄漏检查：`src/wmagentattack/evidence_ledger.py`。
- 新增冻结 clean probe 模型、任务平衡训练和 task-macro 指标：`src/wmagentattack/clean_evidence_probe.py`。
- 新增数据重放、120-run 训练网格和冻结汇总：`scripts/121_*`、`scripts/122_*`、`scripts/123_*`。
- 新增 post-hoc 机制诊断：`scripts/124_diagnose_clean_evidence_counterevidence.py`。
- 新增相应测试；远端当前代码最终全仓验证为 254 passed。
- 修正未来汇总中 NO-GO 仍推荐未通过模型的问题；没有重写 job 4722 的冻结结果。

## 下一步研究约束

当前不应在这 90 条轨迹上调 hash dimension、hidden size、训练轮数或追加新 control；这会把同一面板从验证集变成调参集。

下一版应作为新的、独立冻结研究：

1. 从 exact simulator 的结构化 runtime output 构建 ledger v2，而不是先转成扁平文本再解析。
2. 保留 record/entity 边界；每个航班、酒店、餐厅作为独立实体。
3. 只有“同一已解析实体 + 同一属性 + 不同值”才标 conflict；不同候选不是 conflict。
4. 使用 item-level entity candidate set，并保留 `UNIQUE/AMBIGUOUS/UNLINKED`，不能把 call-level 状态复制给所有 item。
5. Goal linking 只能读取 trusted user goal 与 tool schema，不得读取 expert path 或 utility。
6. 在不看 utility label 的条件下完成抽取器单元测试，然后一次性评估新的独立 stronger clean panel。
7. 新面板必须分离 development/confirmation seeds，并恢复至少两个预注册 durable task 交集后，才允许小规模 paired attack pilot。
8. 在此之前继续禁止攻击数据生成、真实外部端点和 Dreamer 大训练。

因此，下一阶段的合理目标不是“让 v1 在旧面板上变好”，而是先构建不丢失实体结构的 ledger v2，并把它留给独立 clean panel 做一次真正的确认。
