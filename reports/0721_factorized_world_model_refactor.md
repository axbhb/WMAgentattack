# WMagentattack 因子化世界模型重构（2026-07-21）

## 结论先行

本轮不再把 DreamerV3 视为默认主模型，而是将它冻结为历史对照 D0。共享对话指出的核心问题成立：旧实现把攻击者配置、受害者的工具选择、AgentDojo 的确定性工具状态转移和最终 checker 标签混在同一个单智能体 MDP 中；在当前数据规模下，这会让模型更容易记忆文本或配置模式，而不是学习可迁移的受害者响应动力学。

新实现把系统拆成三层：攻击者在 rollout 前选择静态干预；受害者模型只预测 tool/skill、参数签名和 stop；AgentDojo 已知的工具执行、状态变化及 checker 继续由精确模拟器完成。效用建模改成 task clean prior 加 attack residual，重复采样标签改成四格联合计数似然。

远端冻结 clean confirmation 仍是硬性 NO-GO：45 条确认轨迹完整、零运行失败，但只有一个确认集稳定任务，开发/确认稳定任务交集为零。因此本轮允许代码重构和现有数据上的诊断实验，但不允许生成新攻击数据、启动 Dreamer 大训练或形成论文级效果结论。

## 共享对话中的发现如何落到仓库

| 发现 | 仓库改进 | 文件 |
|---|---|---|
| attacker action 与 victim tool action 被混为一个 action | 定义静态 `AttackerIntervention` 与动态 `VictimActionEvent` | `src/wmagentattack/decision_state.py` |
| 训练输入容易混入完整轨迹、工具输出或 checker 标签 | manifest 采用正向字段白名单；被忽略的 outcome 路径进入审计；语义指纹对 outcome 改变保持不变 | `src/wmagentattack/decision_state.py`, `scripts/109_build_decision_states.py` |
| AgentDojo 的工具状态转移本来就是已知的，不应由神经网络重建 | 新增确定性 executor/checker 适配器和 canonical state delta | `src/wmagentattack/exact_simulator.py` |
| 数据量较小时，Dreamer latent dynamics 容易过拟合 | 新增层级离散 IO-HMM，按攻击上下文收缩到 pooled backoff，并加入一阶 Markov 反证基线 | `src/wmagentattack/io_hmm_world_model.py`, `scripts/110_train_io_hmm.py` |
| 主模型应学习 victim response，而不是原始文本重建 | 新增因果 Event Transformer；无 actor/critic、无 observation reconstruction | `src/wmagentattack/event_world_model.py`, `scripts/111_train_event_world_model.py` |
| posterior mean 被当作无噪声回归标签 | 保留旧字段，同时加入 `attack0/1_utility0/1` 四格计数、Jeffreys Dirichlet 后验和 count likelihood | `src/wmagentattack/multiseed_labels.py`, `src/wmagentattack/schema.py`, `scripts/64_build_agentdojo_v2_final.py` |
| utility 受任务本身可解性强烈影响 | 新增 clean utility logit prior 与 attacked utility logit residual | 同上 |
| 旧随机 trajectory split 存在 task 泄漏 | 新训练入口默认遇到 task-group overlap 直接失败；只可用显式 smoke 标志绕过，并标记非确认性 | `scripts/110_train_io_hmm.py`, `scripts/111_train_event_world_model.py` |
| 第一条看似正向的结果可能是偶然 | 固定三档 IO-HMM、三个 Event seed、明确反基线与事前阈值 | `configs/0721_factorized_world_model_protocol.json`, `scripts/112_summarize_factorized_smoke.py` |

## 新骨架

```text
manifest / trusted task context
          │
          ▼
CanonicalDecisionState ── AttackerIntervention (rollout 前固定)
          │
          ▼
Victim dynamics
  W0: hierarchical IO-HMM
  W1: causal Event Transformer
          │ predicts
          ▼
VictimActionEvent(tool, argument signature, stop)
          │
          ▼
ExactSandboxSimulator / AgentDojo in-memory tools
          │
          ├── canonical state delta
          ├── trusted utility checker
          └── attack/security checker
```

Event Transformer 的序列头预测下一 skill、参数签名和 stop；终局头输出四格 Dirichlet concentration。由四格分布可得到 utility、attack success 与 joint success 三个边际概率。模型还单独输出 utility logit residual，并把 clean solvability prior 作为输入。

## 已完成验证

1. 本地 schema smoke：400 个 v2 manifest 行生成 400 个唯一决策状态指纹。由于尚未提供 trusted goal、完整工具 schema 与初始环境映射，审计正确标记 `training_ready=false`；这些状态只能用于 schema smoke，不能直接作为正式训练输入。
2. outcome 不变性：改变同一 manifest 行的 utility、security、selected tool 和 skill output，不会改变 canonical decision-state fingerprint。
3. 联合标签：测试样例正确得到四格计数 `00=2, 01=1, 10=0, 11=2`，后验概率和为 1。
4. 本地旧数据诊断：6-state IO-HMM 的 validation mean event NLL 为 1.2945，一阶 Markov 为 1.5506，差值 −0.2560；test NLL 为 1.3306，对照为 1.5141。该旧划分存在大量 task overlap，结果仅说明代码可学习，不能作为方法有效性的证据。
5. 远端回归：新增测试与标签测试 14/14 通过；修复一个此前 provenance helper 对测试替身字段的兼容问题后，全仓库 206/206 通过。唯一剩余输出是 PyTorch Transformer 的 nested-tensor 性能提示，不影响正确性。

## 当前所有关键问题

### 1. clean eligibility 是最高优先级阻塞

冻结确认中，开发集稳定任务为 Travel 18/7，确认集稳定任务仅 Travel 1，两者无交集。当前 victim/scaffold 对任务的完成存在强随机性，攻击效果与基础任务失败无法可靠区分。任何新 world model 的正向离线指标都不能解除这一问题。

### 2. canonical task context 尚未补齐

现有 v2 manifest 有攻击目标、目标工具序列和 payload，但没有一套独立、结构化且在 rollout 前可用的 trusted user goal、完整 candidate tool schema、初始环境 canonical state 与 goal-slot 分解。`scripts/109_build_decision_states.py` 因此默认拒绝正式构建；必须显式提供 task-context 文件。

### 3. “是否真的需要多步 MBRL”仍未证明

当前多数 attacker 配置在 episode 开始前一次性确定。若攻击者不能根据中间 observation 再选择新干预，问题更接近 contextual bandit/configuration ranking，而不是标准多步控制。H>1 imagined planning 只有在定义并采集真实的中间 attacker interventions 后才有方法学意义。

### 4. v2 数据虽为 task-group split，但 clean gate 已否定其正式资格

v2 final 数据的 train/val/test task overlap 为零，且有五 seed 重复采样；这是比旧随机划分更好的诊断数据。然而其任务来自先前协议，不能绕开新 unseen-seed clean confirmation 的 durable-task 要求。

### 5. 参数建模仍是第一版

Event Transformer 目前把工具参数压缩成 key-set signature，尚未实现实体指针、schema-constrained value decoder 或参数级 exact-match/NLL。若 skill 预测有效而 exact simulator rollout 仍失败，参数头将是首要改进点。

### 6. exact AgentDojo adapter 目前是通用接口

`ExactSandboxSimulator` 已固定函数式 executor/checker 契约，但尚未为四个 AgentDojo suite 写具体 state clone、tool registry 和 checker adapter。当前测试使用纯内存状态；正式 H=2 验证前必须补齐 suite adapter，并验证 replay 与 AgentDojo 原生结果逐步一致。

### 7. Event value 头需要配置级反事实评估

即使 joint count NLL 改善，也不能自动说明候选攻击排序改善。下一轮需要同任务、同 seed 的 matched candidate set，按 task-cluster bootstrap 评估 top-1 utility/ASR、pairwise ordering、Brier/NLL 与 regret，并与现有静态 ranker、IO-HMM 和 D0 Dreamer 同时比较。

## 本轮固定预算与判定

远端诊断预算已经预注册在 `configs/0721_factorized_world_model_protocol.json`：

- W0：4/6/8 latent states，各 50 EM iterations、3 restarts；与 smoothed context Markov 比较。
- W1：Event Transformer 64 hidden、2 layers、8 epochs；seeds 7/17/29。
- D0：历史 Dreamer 只作为冻结对照，不重跑。
- 数据：仅使用现有 AgentDojo-v2 sandbox 数据，先离线重建联合 count label；不执行 Llama victim，不生成 payload，不调用真实端点。

无论 W0/W1 是否达到架构阈值，clean gate 都保持 false。因此本轮最高只能得到 `ARCHITECTURE_SIGNAL_ONLY_CLEAN_GATE_BLOCKED`，不能进入攻击数据扩充或大训练。

### 固定预算实际结果

Slurm 4696 已完成，归档有 `COMPLETE` 标志，未出现 Traceback、OOM、CUDA/runtime error。40 个归档 checksum、17 个数据 checksum 和 9 个代码 checksum 均通过。

| 模型 | Validation | Test | 结论 |
|---|---:|---:|---|
| Markov | NLL 1.8036；accuracy 46.00% | NLL 2.2692；accuracy 30.37% | 强低容量反基线 |
| IO-HMM 4-state | NLL 1.8650；accuracy 38.43% | NLL 2.4321；accuracy 29.34% | validation/test 均劣于 Markov |
| IO-HMM 6-state | NLL 1.8669；accuracy 37.17% | NLL 2.7072；accuracy 31.33% | 增大状态数后 test NLL 更差 |
| IO-HMM 8-state | NLL 1.8673；accuracy 40.67% | NLL 2.7344；accuracy 31.20% | 同样不通过门槛 |
| Event Transformer（3 seed 均值） | tool accuracy 43.67%；joint NLL 相对 constant −1.6107 | tool accuracy 37.35%；joint NLL 相对 constant −1.9140 | value signal 稳定，但 dynamics 结果混合 |

Event seed 的 validation tool accuracy 标准差为 2.73 个百分点，通过事前 5 点稳定性阈值；三个 seed 的 joint count NLL 都优于训练集 constant baseline。可是 W0 的两个门槛均失败，所以按照未修改的事前规则，最终决定为 `NO_GO_REVISE_FACTORIZED_MODEL`，而不是架构 GO。

### 必须保留的反证与解释边界

1. Event validation accuracy 43.67%，低于 Markov 的 46.00%（−2.34 点）；test 高于 Markov 6.98 点。它不是跨 split 一致胜出。
2. validation/test 各有 19.42% 轨迹使用训练未见的 attack context；对应 event 占比分别为 20.20% 和 15.76%。当前 categorical context 只能退化到 `<UNK>`，不具备攻击语义迁移能力。
3. test 中有 17 个训练未见的 `restaurant_generate` skill，被当前 loader 折叠成 `<UNK>`。因此 test tool accuracy 并非完整的真实 skill identity accuracy。
4. train 四格计数为 `00=502, 01=533, 10=24, 11=141`；`attack1_utility0` 仅占 2%。联合头面对明显不平衡，必须报告逐格校准而不能只给总 NLL。
5. joint NLL 使用 teacher-forced victim event prefix。模型尚未证明在自由生成的 event rollout 上仍能保持 value 改善；观察到的优势可能部分来自真实轨迹提供的中介信息。
6. IO-HMM 的 4/6/8 state 全部输给一阶 Markov，说明当前离散 context/state factorization 不成立；不能因为旧泄漏划分上的正结果而保留该主张。

完整结果位于 `/share/guozhix/wmagentattack/0721/factorized_refactor/fixed_budget_v1`，其中 `summary.json` SHA256 为 `ff7591583afaa1147fe077f38ae3af55b1f95289be1a353685f93101028fa9e6`。

## 后续顺序

1. 本固定预算循环在 `NO_GO_REVISE_FACTORIZED_MODEL` 处停止，不继续提交实验。
2. 下一协议首先把 attack name categorical ID 改为可迁移语义：family/role/trigger/payload-position/target-tool-depth 分解，并给未见 family 设置显式层级 backoff。
3. 将 skill 表示改成 tool schema 与参数槽组合，测试时保留 `restaurant_generate` 身份；禁止把未见真实 skill 当作普通 `<UNK>` 命中。
4. 新增三种严格分开的 value 评估：H=0/BOS-only、teacher-forced prefix、free-running victim rollout；报告 teacher-forcing gap，并用 exact simulator 检查 H=1/2 replay consistency。
5. 事前加入 Event-vs-Markov 的 next-event NLL/accuracy 门槛、四格逐类 NLL/Brier/ECE，以及按 `(suite, user_task_id)` cluster bootstrap 的候选排序 regret。
6. IO-HMM 降为反基线；只有加入语义 context 或显式 duration 后能稳定击败 Markov，才重新考虑 HSMM 扩展。
7. 并行解决真正阻塞：扩大 victim/scaffold 或 clean task pool，重新预注册 unseen-seed durable-task gate。只有 clean gate GO 且至少两个 durable tasks 后，才构建 matched attack pilot或讨论正式大训练。
