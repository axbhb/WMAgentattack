# Candidate × Constraint Evidence Dataset 小规模试验结果

日期：2026-08-05

正式结论：`GO_SCHEMA__NO_GO_TRAINING__COUNTERFACTUAL_COLLECTION_REQUIRED`

## 1. 完成了什么

本轮只使用远端服务器上的冻结 clean-only 数据，构建了一个 12 个训练任务的小规模 Candidate × Constraint Evidence Dataset v1。正式构建在 `/share/guozhix/WMagentattack-evidence-pilot-aug5` 执行，结果归档在：

`/share/guozhix/wmagentattack/0805/candidate_constraint_evidence_pilot/fixed_v1`

正式作业为 Slurm `6342`，状态 `COMPLETED`、退出码 `0:0`、运行 26 秒。运行期间没有 victim LLM 调用、反事实工具执行、模型训练、攻击样本或 Dreamer 运行。

数据采用四层分离结构：

1. `state_catalog`：冻结的 Structured Semantic State v3；
2. `constraint_catalog`：从 trusted goal 提取的原子约束；
3. `observed_rows`：实际执行动作与每个约束之间的相邻状态监督；
4. `unlabeled_counterfactual_queries`：合法但未执行的动作查询，明确标为未标注，不伪造为负样本。

已满足因果约束：不使用 proof contract、未来动作、最终任务结果、utility/security/attack 标签或隐藏模拟器状态。任务、轨迹、状态和行 ID 仅作引用与分组，禁止作为模型特征。

## 2. 数据规模与标签分布

| 指标 | 结果 |
|---|---:|
| suite × difficulty 单元 | 12/12 |
| 任务 / episode | 12 / 12 |
| semantic state | 19 |
| 真实非终止转移 | 19 |
| 唯一 goal constraint | 98 |
| 有监督 candidate × constraint 行 | 203 |
| 未标注反事实查询 | 3,834 |
| 合法候选查询空间 | 4,037 |
| 已观测比例 | 5.0285% |

有监督标签分布：

- `ALREADY_SUPPORTED`：42 行，覆盖 6 个任务；这些行只用于状态一致性检查，不用于学习转移增益；
- `NEWLY_SUPPORTED`：60 行，覆盖 10 个任务；
- `UNCHANGED_UNSUPPORTED`：101 行，覆盖 10 个任务；
- 可用于预测性训练的行共 161 行，状态一致性行共 42 行。

## 3. 验证结果

- 本机测试：23/23 通过；
- 远端测试：23/23 通过；
- 两次远端独立重建逐字节一致；
- dataset SHA256：`e1851c0e80a11421cbb4e2c0fac60ce6e127da48713d287e28fb5f491667fc3c`；
- audit SHA256：`6209bcba50239e8feb56ecc26e7b9f448063a59d870c6d434f2777348a16ebcc`；
- 原始轨迹与 Semantic v3 输入哈希均与预注册值一致；
- 所有引用有效、观测行与未标注行无交集、未标注行没有 target、状态泄漏为零；
- 使用独立审计程序直接从原始轨迹和相邻 Semantic v3 状态重新推导后，19 条转移、203 个观测标签、3,834 个未标注查询和 42/60/101 标签分布完全一致；
- Slurm 标准错误为空，精确扫描未发现 Traceback、OOM、CUDA、RuntimeError 或 Exception。

## 4. 门槛判定与反证

Schema gate 全部通过，因此当前数据表示方式可以保留。但 training-readiness gate 四项全部失败：

| 训练准备条件 | 实际值 | 预注册门槛 | 判定 |
|---|---:|---:|---|
| 已观测候选比例 | 5.0285% | ≥25% | FAIL |
| execution error 转移 | 0 | ≥5 | FAIL |
| conflict 新增转移 | 0 | ≥5 | FAIL |
| ambiguity/unlinked 新增转移 | 0 | ≥5 | FAIL |

主要反证是：当前 203 个标签只描述 victim 实际选择的动作；其余 3,834 个合法候选没有执行结果。若把它们当作“不支持约束”的负例，模型会学习到人为标签而不是环境动力学。此外，19 条转移只覆盖 13 种实际动作，部分动作只有一次观测；L1 中也没有 `ALREADY_SUPPORTED` 行。这个样本适合验证数据接口，不足以比较模型，更不能证明 evidence world model 有效。

## 5. 调度异常记录

第一次作业 `6341` 在 0 秒处以 `NonZeroExitCode` 退出。原因是新 worktree 中没有 Slurm 输出所需的 `logs/` 目录；脚本主体没有启动，归档目录也未创建，因此没有数据污染。创建空日志目录后，以完全相同的提交、协议和输入运行 `6342` 成功。仓库现已保留 `logs/.gitkeep`，避免新 worktree 再次出现同一非语义故障。

## 6. 当前保留架构与下一步

保留 `Candidate × Constraint Evidence Dataset v1` 的规范化 schema，但维持训练 NO-GO。下一步应单独预注册一个 clean-only counterfactual collection pilot：从这 19 个状态中按 suite、difficulty、动作类型和当前约束状态分层抽取少量合法替代动作，在 AgentDojo 合成沙箱中从同一 canonical state 执行，并记录真实的参数绑定、状态增量、错误、冲突与歧义结果。达到标签覆盖和稀有事件门槛后，才能冻结 task-disjoint 划分并启动小模型 probe；当前不得启动 Dreamer、攻击数据生成或大规模训练。
