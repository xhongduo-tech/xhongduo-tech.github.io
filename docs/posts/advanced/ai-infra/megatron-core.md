---
title: Megatron-Core 与 Megatron-LM 的关系及新特性
date: 2026-08-07
---

# Megatron-Core 与 Megatron-LM 的关系及新特性

<div class="epigraph">
<p>重构不是推倒重来，而是把经验沉淀成可复用的核心。</p>
<footer>—— 利伯尔 · 维克勒克（Libor Vykoukal，NVIDIA 系统软件工程师）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ Megatron-Core 官方文档与 NVIDIA 技术博客 · 训练框架篇 ｜ 2026-08-07</p>
</div>

## 为什么从 Megatron-Core 开始

上一节我们解剖了 Megatron-LM。但如果你现在打开 NVIDIA 的仓库，会发现主线已演化为 **Megatron-Core（NVIDIA/Megatron-LM 仓库中的 `megatron/core` 目录）**。Megatron-LM 像一台为特定 Transformer 量身定做的机器，而 Megatron-Core 是把它拆成「引擎」与「车身」——**引擎（core 库）与模型无关，车身（应用层）才是具体模型**。

为什么需要这次重构？因为训练框架的演进太快：FP8、Transformer Engine（TE）、chunked 注意力、CPU offload……老代码每加一个新特性就要动一遍模型定义。Megatron-Core 把「分布式切分、通信组、数据加载、checkpoint」这些**横切能力**抽成独立库，让新模型、新特性都能直接复用。读懂它，你就读懂了 2024 年之后大模型训练框架的默认形态。

## 1 Megatron-LM 的痛点：一切和 Transformer 耦合

Megatron-LM 的问题不是功能不够，而是**耦合太深**。在它的代码里，张量并行的实现方式（Column/Row 切分）与 Transformer 的具体结构（attention、MLP 的前后顺序）绑在一起：

- 想支持一个新的模型结构（比如把 FFN 换成 MoE、把 attention 换成线性注意力），你得**理解整条并行切分链路**，再逐算子改写。
- 想接入新硬件特性（FP8、TE 的 fused kernel），同样要动模型层。
- 想换个 optimizer、换个调度、换个 dataset，全在一个大包里互相牵扯。<span class="marginnote">这正是「通用性 vs 效率」的经典冲突：Megatron-LM 追求极致效率，代价是难扩展。而 AI 训练框架的用户从「跑一个模型」变成了「跑无数新模型」，通用性成了硬需求。</span>

Megatron-Core 的回应是**分层架构**：把与模型无关的并行基础设施下沉为核心库，把模型定义留在应用层，中间用清晰的抽象接缝隔开。

## 2 Megatron-Core 的架构：core 与应用分离

Megatron-Core 的仓库结构清晰地分为两层：

**核心库（`megatron/core`）**，与具体模型无关：

`megatron/core/transformer`：**Transformer 模块的并行实现**（这仍是核心中的核心，但用「可配置 spec」而非写死结构）。
`megatron/core/parallel_state.py`：并行状态、通信组、TP/PP/DP/CP 组变体。
`megatron/core/optimizer`：混合精度、分布式优化器。
`megatron/core/tensor_parallel`、`megatron/core/pipeline_parallel`、`megatron/core/sequence_parallel`：各并行维度的底层算子。
`megatron/core/datasets`、`megatron/core/checkpointing`、`megatron/core/pipeline_parallel/schedules`：数据、检查点、调度。

**应用层**：具体的模型与训练脚本，如 Llama、Mistral、以及各类 pretrain 入口。

**核心思想**：模型定义通过「层规格（spec）」（描述每一层用哪种注意力、哪种 MLP、用不用 TE）与核心库解耦。换模型＝换 spec，而不是改并行代码。<span class="marginnote">这个「spec 驱动」设计是 Megatron-Core 的灵魂：一个 `ModelConfig` 对象描述隐藏维、层数、头数、用不用 flash-attention、用不用 TE，核心库据此装配出「已经切好并行」的模型。要支持新模型，主要是写新 spec，而非重写分布式逻辑。</span>

## 3 新特性一：与 Transformer Engine（TE）深度集成

Megatron-Core 最重要的工程合作是 **Transformer Engine（TE）**。TE 是 NVIDIA 提供的「fused + 自动精度选择」的 Transformer 算子库：它内部把 LayerNorm、矩阵乘、激活融合成少数 kernel，并根据输入动态选择 FP8/BF16/FP16。

在 Megatron-Core 里，spec（如 `ModuleSpec`）可以指定注意力与 MLP 用 TE 实现。好处：

**自动 FP8**：TE 在 forward 里检测激活范围，自动切换到 FP8 权重与激活，训练提速且省显存。
**kernel fusion**：LN + QKV 投影、attention + 输出投影等融合，减少 kernel 启动与显存读写。
**精度安全**：TE 内置了「延迟缩放（delayed scaling）」等机制保证 FP8 不爆精度。<span class="marginnote">FP8 的细节在第四篇第八节《FP8 训练》详述。这里只需记住：TE 是 FP8 训练的「搬运工」，Megatron-Core 通过 TE 把 FP8 能力无缝接进分布式框架——这是「框架」与「算子库」分工协作的范例。</span>

## 4 新特性二：更灵活的调度与上下文并行

Megatron-Core 把调度从「内置」变成「可配置」：

**调度模块（`megatron/core/pipeline_parallel/schedules.py`）支持 1F1B、Interleaved、以及 Chunked 调度**（把 attention 按 chunk 切，减少中间激活）。
**Context Parallel（CP）作为一等公民**：`context_parallel_size` 配置直接可用，与 TP、PP、DP 组合成「4D 并行」。
**MoE 支持**：专家并行、负载均衡 loss 都进入核心库，`num_experts`、`moe_router_load_balancing_type` 等参数开箱即用。<span class="marginnote">对比 Megatron-LM：它的 MoE、CP 需要用户自己拼装；Megatron-Core 把这些「新维度」沉淀成核心参数。这正是「重构把经验沉淀成核心」的具象化。</span>

## 5 公式解析：从「写死的层」到「spec 驱动的层」

在 Megatron-LM 中，一个 Transformer 层的构造大致是写死的：QKV 用 Column 并行、输出用 Row 并行、MLP 两段都切。而 Megatron-Core 的构造变成了「按 spec 装配」：

$$\text{Layer} = \text{Assemble}(\text{AttentionSpec}, \text{MLPSpec}, \text{Config})$$

其中：

- **AttentionSpec**：描述注意力用 standard / flash / TE / ring 中的哪一种，以及是否 chunked、是否 sequence parallel。
- **MLPSpec**：描述 MLP 是 dense 还是 MoE，专家数多少，是否用 TE。
- **Config**：隐藏维、头数、层数、并行大小、重计算开关等全局配置。

拆解这个装配过程：

- **并行切分被隐藏**：spec 只描述「层长什么样」，切分逻辑（Column/Row、stage 划分、通信组）由核心库根据 Config 里的并行参数统一施加。
- **新特性即新 spec**：要加一个「chunked attention 的线性注意力变体」，就是写一个新 spec 并注册进核心库，不用改并行引擎。
- **可组合**：不同 spec 可以组合出不同模型，同一个核心库同时支撑 Llama、Mistral、MoE 等。

**一句话概括**：Megatron-LM 是「代码写死结构」，Megatron-Core 是「配置驱动结构」。数据与模型都是 spec 的输入，并行能力是核心库的输出。<span class="marginnote">工程上这个抽象的成本是「学习曲线变陡」：你不再直接看到「ColumnParallelLinear」散布在模型代码里，而是藏在核心库。调试时反而要习惯「查 spec 是否配置正确」这种新思维。</span>

## 6 辨析｜易错点：Megatron-Core 的常见误区

**辨析｜易错点：**
- **Megatron-Core 不是 Megatron-LM 的替代品那么简单**：旧仓库 `megatron/model` 目录仍保留供兼容，但新开发都落在 `megatron/core`。读文档先分清楚自己在看哪代。
- **TE 不是必需的**：Megatron-Core 可不用 TE（fallback 到原生 PyTorch 实现），用 TE 主要是为了 FP8 与融合 kernel。
- **spec 不是超参**：它描述「结构」，Config 里的并行参数描述「怎么切」，两者职责不同，别混。
- **CP 不是自动开的**：CP（`context_parallel_size`）需要网络拓扑支持（跨节点高带宽），且要配合 flash-attn 的 CP 实现才有效。

## 7 小结

- **重构动机**：Megatron-LM 与 Transformer 耦合太深，新特性难加。
- **分层架构**：核心库（并行/优化器/调度/checkpoint）与应用层（模型 spec）解耦。
- **spec 驱动**：模型结构由 spec 描述，并行能力由核心库统一施加，换模型不改并行代码。
- **两大新特性**：Transformer Engine 集成（FP8 + 融合 kernel）与一等公民的 Context Parallel / MoE。
- **演化关系**：Megatron-Core 是 Megatron-LM 的「引擎重构版」，当前 NVIDIA 大模型训练的默认底座。

## 8 进阶与延伸

**动手比较两个版本**：分别读 Megatron-LM 旧版与 Megatron-Core 新版里「Transformer 层」的构造代码——旧版是「结构写死 + 并行写进层」，新版是「spec 描述结构 + core 施加并行」。对照完，你就能给同事讲清楚「重构」带来的东西。

**几个值得进一步挖的方向**：

- **`ModelConfig` 字段地图**：`tensor_model_parallel_size`、`pipeline_model_parallel_size`、`data_parallel_size`、`context_parallel_size`、`recompute_method`——每个字段对应并行或显存决策的哪个旋钮？填一份「字段 → 决策」对照表。
- **spec 与并行解耦的边界**：哪些「模型差异」能被 spec 表达（attention 类型、MLP 结构），哪些不能（需要改 core）？划清这条线，你就理解了框架抽象的能力边界。
- **TE 的 FP8 开关**：`ModelConfig` 的 `fp8` 与 `transformer_engine` 选项如何影响前向——配合《FP8 训练》篇，看 TE 的「延迟缩放」在框架层怎么配。

**自测题**：为什么说「换模型＝换 spec，而不是改并行代码」？如果模型里有一个「全新算子」core 库不支持，你会怎么做——这暴露了 spec 抽象的边界。

## 9 动手实践清单

- 读 Megatron-Core 的 `megatron/core/transformer/config.py` 源码，填一张「字段 → 并行决策」对照表。
- 对比旧版 `megatron/model/transformer.py` 与新版 `megatron/core/transformer/transformer_layer.py` 的 Transformer 层构造。
- 开 FP8（`--fp8 e4m3`）跑一个小模型，观察 FP8 的算力与显存收益。
- 试 Context Parallel（`--context-parallel-size 2`），验证长序列训练能否跑通。
- 用一个自定义 spec 注册新模型，体会「换模型不改并行代码」。
- 用 profiler 对比「用 TE vs 不用 TE」的 kernel 效率。
- 验证「spec 描述结构、Config 描述切分」的分工。

在下一节，我们从 NVIDIA 生态转向微软的答案——**DeepSpeed**，看它的 ZeRO 如何用配置文件与训练流程组织起另一个庞大的框架。
