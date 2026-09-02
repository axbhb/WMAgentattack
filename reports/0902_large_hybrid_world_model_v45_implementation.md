# Large Hybrid World Model v45：代码实现说明

## 当前结论

本轮只完成代码和冻结实验设计，不启动训练，也不占用服务器 GPU。v45 将当前系统中三个主要可学习部分同时升级为大模型，但仍保留 AgentDojo 沙箱的精确工具约束与任务互斥评估。

## 固定架构

### 1. Large Structured State Encoder

- 使用本地 `intfloat/e5-base-v2` 将五类推理时可见字段编码为 768 维语义节点；
- 五类字段为 trusted goal、visible observation、legal tools、visible prior tool 和 source/track；
- 再使用 8 层、12 头、隐藏维度 768 的 Structured Transformer 融合字段；
- 禁止输入任务 ID、轨迹 ID、真实结果标签和未来信息。

### 2. Large Victim Action Dynamics

- 31 个候选动作分别作为 query；
- 使用 6 层 candidate-to-state cross-attention Transformer 读取结构化状态节点；
- 同时预测合法动作分布、五个结果变量和任务成功/攻击成功四组合；
- 合法动作 mask 和沙箱执行仍由确定性环境处理，不交给神经网络猜测。

### 3. Large Multi-step Residual Dynamics

- 使用 8 个 latent memory token 和 6 层 Transformer 进行 H1–H5 rollout；
- residual 输出严格零初始化，训练开始时等价于一步 teacher；
- 一步使用 KL/non-inferiority 约束，防止多步训练破坏 Structured State 的一步能力；
- teacher 阶段完成后冻结 state encoder 与 victim dynamics，只训练 residual dynamics。

## 完整数据实验

正式实验固定使用已有完整 AgentDojo 数据：2,060 条轨迹、6,763 个事件、4,703 个相邻转移、20 个任务、4 个 suite、31 个候选动作。采用五折任务互斥划分和三个种子，共 15 个模型拟合，不允许将同一任务分到训练与测试两侧。

结果必须与两个保留基线比较：Structured Markov v3 + joint outcome head，以及其零初始化 residual 版本。此前 E5 开放词表实验的召回失败和中等容量模型的任务词汇过拟合均作为反证保留，不能只报告 v45 的有利指标。

## 风险与判定

扩大三块模型并不保证提升。最主要风险仍是 6,763 个事件不足以支撑约亿级参数端到端更新，造成 task-disjoint 过拟合。因此正式门槛同时要求：

- 一步 action NLL 比小模型改善；
- 一步准确率基本不下降；
- 四组合校准不恶化；
- residual 的 H1 不劣于 teacher；
- H2–H5 的加权 NLL 相对旧 residual 有稳定改善；
- 改善覆盖多数任务、至少两个种子，并且不存在 suite 级大幅退化。

任何一项关键门槛失败都记为 NO-GO，而不是在看过测试集后调整阈值。

## 当前运行状态

- 代码：已实现，正在完成本地静态检查与测试；
- 训练：未提交；
- Slurm 作业：无；
- 外部端点：不使用；
- 新攻击数据：不生成。
