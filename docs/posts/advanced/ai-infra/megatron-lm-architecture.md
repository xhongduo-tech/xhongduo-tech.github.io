---
title: Megatron-LM 整体架构与并行切分的代码结构
date: 2026-08-07
---

# Megatron-LM 整体架构与并行切分的代码结构

<div class="epigraph">
<p>要让千亿参数模型跑起来，框架本身就是一台精密的并行机器。</p>
<footer>—— 肖比 · 默罕默德（Mohammad Shoeybi，Megatron-LM 作者）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ Megatron-LM 论文与开源代码 · 训练框架篇 ｜ 2026-08-07</p>
</div>

## 为什么从 Megatron-LM 开始

前一篇我们用五步决策法配好了 TP/PP/DP，但「配好」不等于「跑起来」。要落地，需要一套**工业级训练框架**来承载这些并行策略——而 Megatron-LM 正是这个领域的标杆。它是 NVIDIA 开源的、支撑 GPT-3 级模型训练的事实标准，也是理解 Megatron-Core、DeepSpeed、甚至所有后续框架的地基。

本篇不追求逐行读代码，而是画出 Megatron-LM 的**架构地图**：它把模型定义成什么、并行切分在代码里发生在哪、训练循环长什么样。读完你对「一个 3D 并行训练框架由哪几块拼成」会有清晰的全局观。

## 1 Megatron-LM 的定位

Megatron-LM（2019–2020）由 NVIDIA 发布，目标是让 Transformer 模型在 **TP × PP × DP** 三维并行的语义下训练。它的核心贡献有三：

- **张量并行层**：把 Transformer 的每个矩阵乘改写为「列并行 + 行并行」的切分形式（`ColumnParallelLinear` / `RowParallelLinear`）。
- **流水线并行**：把层列表切成若干段，每段放到一个 PP stage，micro-batch 按 1F1B（或 Interleaved）调度流过。
- **一体化训练循环**：内置数据并行、学习率调度、混合精度、checkpoint 等，开箱即用。

它最有价值的遗产，是把「并行切分」从「论文里的图」变成了「代码里可读的 API」。后续 Megatron-Core 是对它的重构（下一篇专门讲）。<span class="marginnote">Megatron-LM 与 HuggingFace 的 transformers 定位不同：后者追求「模型种类丰富」，前者追求「把一个大模型高效切到几千张卡」。你用 Megatron 不是为了换模型结构，而是为了换「分布式能力」。</span>

## 2 核心抽象：模型并行区与序列并行区

Megatron-LM 把所有算子分成两类，这是它切分模型的总纲：

- **模型并行区（model parallel region）**：需要跨 TP 卡联合计算的算子，主要是**矩阵乘与注意力投影**。这里的权重被按行/按列切开，算子内部有 AllReduce 通信。
- **序列并行区（sequence parallel region）**：LayerNorm、Dropout 这类按「序列 × 隐藏」作用的算子。在开启序列并行后，它们只处理本卡那 $1/t$ 段序列，无需通信。<span class="marginnote">这个「分区」思想直接对应上一篇《序列并行》里的 Megatron-SP：矩阵乘走 TP（通信），LN/Dropout 走 SP（无通信）。Megatron 把这条边界固化进了代码结构，任何算子都要先归位到这两区之一，才能正确并行。</span>

**这两区的分界是 Megatron 切分模型的第一性原理**：能省通信的算子坚决不通信，必须联合的算子才通信。

## 3 张量并行的代码结构：Column 与 Row

Megatron 把权重矩阵的切分封装成两个模块：

- **`ColumnParallelLinear`**：把权重 $W$（形状 $h_{in} \times h_{out}$）沿**输出维**切成 $t$ 块，每卡持有一块。输入 $X$ 完整，各卡算 $X \cdot W_i$，输出是**各卡的局部片段**——下游算子需要把它拼起来或继续切分。无通信（前向），后向时 AllReduce 梯度。
- **`RowParallelLinear`**：把权重沿**输入维**切成 $t$ 块。每卡用 $X_i \cdot W_i$ 算自己的部分和，最后 **AllReduce 求和**得到完整输出。

Attention 的 QKV 投影用 Column（输出维 = 3×hidden 被切），输出投影用 Row（输入维被切）。**一次「Column → Row」配对，就完成了一个切分的线性层，且通信只发生在 Row 之后的一次 AllReduce**。<span class="marginnote">这正是《张量并行》那篇的公式落地：$Y = X \cdot W = \sum_i X_i W_i$。代码里把「切」写进层定义，而不是在用户脚本里手动拼 AllReduce——这是框架存在的意义。</span>

## 4 流水线并行的代码结构：p2p 通信

流水线并行在 Megatron 里的体现更直接：

- 模型的 `layers` 列表按 `world_size / tp_size` 个 PP stage 切分，每张卡只实例化属于自己 stage 的那几层。
- 相邻 stage 之间用**点对点通信**（`torch.distributed.send/recv` 封装）传激活与梯度。
- 调度逻辑（1F1B / Interleaved）由「每个 micro-batch 的前向/后向何时发何时收」决定，Megatron 在 `forward_backward_pipelining_with_interleaving` 等函数里实现。

**流水线并行在代码层面的本质：把「整模型」变成「每卡只拥有模型的一段」，并用收发消息串联起 micro-batch 的流经**。<span class="marginnote">工程上 PP 与 TP 的交互最易出错：TP 组的边界必须恰好卡在 PP stage 边界内，否则通信组会串。Megatron 用「rank 编号 → 先按 TP 分组、再按 PP 分段」的固定映射规避了这个问题。</span>

## 5 训练循环与代码地图

Megatron-LM 的训练入口是 `pretrain.py`，核心循环非常朴素：

```python
for step in range(total_steps):
    optimizer.zero_grad()
    losses = forward_backward_no_pipelining(...)   # 或 pipelining 版本
    optimizer.step()
    if step % eval_interval == 0: evaluate(...)
```

分布式的一切都被藏在框架内部。项目的关键目录：

| 目录 / 文件 | 职责 |
| --- | --- |
| `megatron/model/transformer.py` | Transformer 层定义，含 Column/Row 并行层装配 |
| `megatron/model/parallel_state.py` | 通信组初始化：TP/PP/DP 组映射 |
| `megatron/core/parallel_state.py` | Megatron-Core 中的并行状态（新版本） |
| `megatron/arguments.py` | 命令行参数：`--tensor-model-parallel-size` 等 |
| `megatron/optimizer/` | 混合精度优化器、主权重管理 |

**读代码的推荐路径**：先读 `parallel_state.py`（理解通信组怎么建），再读 `transformer.py`（理解层怎么切），最后读训练循环（理解调度怎么串）。<span class="marginnote">`arguments.py` 里那一排 `--xxx-parallel-size` 参数，就是上一篇《并行策略选型》的五个决策输入在代码里的落点：`--tensor-model-parallel-size=8 --pipeline-model-parallel-size=4 --data-parallel-size=...`。配比最终就是填这张参数表。</span>

## 6 公式解析：Column→Row 的通信为什么只有一次 AllReduce

设输入 $X$（形状 $b \times s \times h_{in}$），权重 $W$（$h_{in} \times h_{out}$），TP 规模 $t$。Row 并行的计算与通信：

$$Y = X W = X\,[W_0, W_1, \ldots, W_{t-1}] = \sum_{i=0}^{t-1} X_i W_i \tag{1}$$

其中 $X_i$ 是 $X$ 沿隐藏维的第 $i$ 块，$W_i$ 是对应块。拆解：

- **$X_i W_i$（每卡局部乘积）**：每卡只算与自己权重块对应的部分，无通信、纯本地计算。
- **$\sum_{i=0}^{t-1}$（跨卡求和）**：这是**唯一一次通信**，用一次 AllReduce 完成（不是 $t$ 次）。

所以一次 Row 并行的线性层，通信成本恒定为 **1 次 AllReduce**，与 $t$ 无关。而 Column 并行在数学上「不需要」通信（输出本就是片段，谁用谁拼），真正通信发生在它后向算梯度时。**一列一行配对，整个 Transformer 层的通信次数被压到每算子常数次**——这就是 Megatron 能高效扩展 TP 的数学基础。<span class="marginnote">对照之前 3D 混合并行篇说的「TP 每个算子都要通信」，这里的精确版本是：Column 前向不通信、Row 前向通信 1 次、每个切分层后向各通信 1 次。总量仍是「每算子常数次」，只是常数有差别。</span>

## 7 辨析｜易错点：Megatron-LM 的常见误解

**辨析｜易错点：**
- **Megatron-LM ≠ 只用 TP**：它默认 TP×PP×DP 三维全开，TP 只是它的金字招牌。
- **`ColumnParallelLinear` 前向真的不通信吗**：前向不通信，但输出是「碎片」，下游必须有配套切分；后向梯度要 AllReduce。别把「前向不通信」误读成「整个层不通信」。
- **`--tensor-model-parallel-size` 不是随便填的**：它必须能整除隐藏维与注意力头数，且最好等于节点内 GPU 数。
- **旧的 `megatron` 包与新的 `megatron.core` 是两代**：代码结构差异不小，查资料时先分清版本。

## 8 小结

- **Megatron-LM**：NVIDIA 开源的工业级 3D 并行训练框架，TP 切分是它的核心贡献。
- **两区划分**：模型并行区（矩阵乘，通信）与序列并行区（LN/Dropout，无通信）。
- **TP 代码结构**：`ColumnParallelLinear`（输出维切、前向无通信）与 `RowParallelLinear`（输入维切、前向 1 次 AllReduce）。
- **PP 代码结构**：层列表按 stage 切分，相邻 stage 用 send/recv 串起 micro-batch。
- **训练循环朴素**：分布式全部封装在框架内，用户只填并行参数表。

## 9 进阶与延伸

**动手读一段代码**：打开 Megatron-LM 的 `megatron/model/transformer.py`，找到 `ColumnParallelLinear` 与 `RowParallelLinear` 的 `forward`——对照本篇公式（Column 前向无通信、Row 前向一次 AllReduce），你会在代码里找到对应的 `all_reduce` 调用。这是「论文公式 → 代码实现」的最好对照练习。

**几个值得进一步挖的方向**：

- **并行状态初始化**：`parallel_state.py` 里 `initialize_model_parallel` 用 rank 编号推导 TP/PP/DP 组——理解它，你就理解了 3D 通信组在代码里怎么落地。
- **`forward_backward_pipelining`**：Megatron 的流水线调度函数如何实现 1F1B？对比 `no_pipelining` 版本，你会看到「发前收后」的调度骨架。
- **从 Megatron 到 Megatron-Core 的代码演进**：同一个 Transformer 层，旧版写死 Column/Row、新版用 spec 装配——对照两个版本，体会「重构」的意图。

**自测题**：`ColumnParallelLinear` 前向不通信，那它的「输出片段」怎么被下游正确使用？如果你能说清「谁消费片段、谁触发通信」，就真的读懂了 Megatron 的切分设计。

在下一节，我们看看 Megatron 的**新一代形态**：Megatron-Core 如何重构并统一 Megatron-LM 与 Transformer Engine。
