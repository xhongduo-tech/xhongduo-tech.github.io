---
title: 序列并行（SP）与 Context Parallel：长序列训练的切分
date: 2026-08-07
---

# 序列并行（SP）与 Context Parallel：长序列训练的切分

<div class="epigraph">
<p>记忆的长度，决定了模型能处理多复杂的依赖关系。</p>
<footer>—— 沃特 · 康特尼（Wouter Koolen，机器学习研究者）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ Megatron-SP 论文（Korthikanti et al., 2022）与长序列文献 · 并行策略篇 ｜ 2026-08-07</p>
</div>

## 为什么从序列并行开始

前两篇切分模型的方式，要么沿层切（PP）、要么沿隐藏维切（TP），都对**序列长度**不敏感。但大模型的上下文正在越拉越长：8K、32K、128K 甚至 1M token。序列一长，**激活显存随序列长度平方增长**，光靠 TP/PP 救不回来——这正是**序列并行（Sequence Parallelism, SP）**与 **Context Parallel（CP）**登场的理由。

SP 不是一个统一算法，而是一族「把序列维度切开、分摊到多个设备」的技术。它回答同一个问题：当一条序列太长、一张 GPU 放不下时，怎么让多张卡合起来处理它？本篇把这一族技术的两条主流路线讲清：Megatron-SP（与 TP 缝合，省 LayerNorm/Dropout 的激活）与 Context Parallel（把注意力也沿序列切，含 Ring Attention 与 DeepSpeed Ulysses 两种实现）。

## 1 序列长度如何引爆激活显存

先看清敌人。一个 Transformer 层的**激活**（前向为了后向而暂存的中间张量），主要两块：

- **注意力部分**：Q、K、V 以及注意力分数矩阵。注意力分数矩阵的形状是 $\text{batch} \times \text{heads} \times s \times s$，其中 $s$ 是序列长度——它随 $s$ **平方**增长。
- **MLP / LayerNorm 部分**：形状约 $\text{batch} \times s \times \text{hidden}$，随 $s$ **线性**增长。

<p>粗略估算单层激活量：</p>

$$\text{Act}_{\text{layer}} \approx 34 b s h + 5 b s^2 a$$

其中 $b$ 是 batch size，$s$ 是序列长度，$h$ 是隐藏维，$a$ 是注意力头数。当 $s$ 很大时，$s^2$ 那一项（来自注意力分数矩阵）会压过 $s$ 那一项，成为显存的主导者。<span class="marginnote">这个「34bsh + 5bs²a」是 Megatron 论文（Reducing Activation Recomputation in Large Transformer Models）给出的经验公式，常用于估算是否需要重计算与序列并行。数值不必精确记忆，抓住「注意力随 s²、MLP 随 s」的结构即可。</span>

## 2 第一路线：Megatron-SP，把 LayerNorm/Dropout 沿序列切开

在纯张量并行（TP）里，每张卡持有模型的**一部分隐藏维**。但 LayerNorm 和 Dropout 这两个算子在计算时天然需要**整条序列**的统计量，没法按隐藏维切开算——所以 TP 下它们每张卡都要复制一份完整的序列激活，造成冗余。

Megatron-SP 的思路很朴素：**既然 LayerNorm/Dropout 是按「序列 × 隐藏」作用的，那就把序列维切成 TP 份，让每张卡只算自己那份**。于是每个 TP rank 上，LayerNorm/Dropout 只持有 $\text{seq}/p$ 长的序列激活，省下的激活量与 $p$ 成正比。<span class="marginnote">注意这里的 SP 是 TP 的「补充」而不是「替代」：矩阵乘部分仍按隐藏维切（TP），只有 LayerNorm/Dropout 按序列维切（SP）。两者结合后，几乎每一份激活都被切分过，冗余消失。今天 Megatron 默认就是 TP+SP 一起开。</span>

**Megatron-SP 省的是「TP 下 LayerNorm/Dropout 那部分冗余激活」，而注意力部分仍由 TP 按注意力头切分**——它不是为长序列而生，而是为「让 TP 更彻底」而生。

## 3 第二路线：Context Parallel，把注意力也沿序列切

当 $s$ 大到 TP 也救不了时，就要让**注意力本身**沿序列维度分布式地算。这一类方法统称 **Context Parallel（CP）**，核心问题是：Q 在第 0 号设备，K、V 却可能在第 1 号设备，怎么算注意力？

两个代表性方案给出不同答案：

**Ring Attention（2023）**：每台设备持有整条序列的**一个块**的完整 Q、K、V。通过环形通信，K、V 块像传花鼓一样在设备间轮转；每转一圈，当前设备就把自己这份 Q 与传来的 K、V 块做一次局部注意力，用 **online softmax** 把分块的 softmax 正确合并。通信量 $\propto \text{序列长度}$，且通信与计算重叠，理论上可扩展到百万级 token。
**DeepSpeed Ulysses（2023）**：先用 **All-to-All** 把 Q/K/V 从「序列分块」重排成「按设备分块」（每个设备拿到完整的一段序列、但只有一部分注意力头），各设备算局部注意力，再用一次 All-to-All 把结果拼回。通信量 $\propto s h$，对**全注意力**的扩展性很好。<span class="marginnote">Ring Attention 通信与计算重叠、适合超长序列；Ulysses 每次 All-to-All 一步到位、实现简单、对较短的序列更划算。业界常把两者甚至 flash-attention 的序列并行版一起做「组合拳」，按序列长度动态选择。</span>

## 4 公式解析：切分后激活如何摊薄

以注意力分数矩阵为例。原始激活量：

$$\text{AttnAct} = 2 b s^2 a \quad (\text{attn scores 的前向与后向各存一份})\tag{1}$$

Context Parallel 把 $s$ 切成 $p$ 份，每个设备只管 $\frac{s}{p}$ 长度的序列。对 Ring Attention，每个设备持有的注意力分数矩阵变为：

$$\text{AttnAct}_{\text{per device}} = 2 b \cdot \frac{s}{p} \cdot s \cdot a \approx \frac{2 b s^2 a}{p} \tag{2}$$

逐项拆解公式 (2)：

- **$b$（batch size）**：不变，batch 不参与切分。
- **$\frac{s}{p}$（本设备的序列块长）**：每个设备只处理序列的 $1/p$ 段，这是切分的核心。
- **$s$（完整序列长）**：$Q$ 是本设备的块，但 $K$、$V$ 是完整序列（Ring 轮转时要看完全部 K/V），所以分数矩阵仍是「本地块 × 全序列」。
- **$a$（注意力头数）**：每个头独立算，不变。

关键洞察：公式 (2) 比公式 (1) 小了约 $p$ 倍——**Context Parallel 把「随 $s^2$ 增长」的注意力激活也按 $p$ 摊薄了**，这正是长序列训练能跑起来的数学原因。若再叠加 TP（头维切分）与 SP，摊薄倍数还会继续乘上去。<span class="marginnote">一条经验法则：当单卡显存装不下「$s$ 到 $s+1$ 的激活增量」时，就该考虑 CP 了。CP 的 p 通常等于节点内或跨节点的网络最优点——它吃带宽，选择 p 时要结合后文第七篇的网络拓扑。</span>

## 5 辨析｜易错点：SP、CP、TP、PP 别混

| 切分维度 | 方法 | 切什么 | 典型通信 | 何时用 |
| --- | --- | --- | --- | --- |
| 隐藏维 $h$ | 张量并行（TP） | 权重、激活的列/行 | AllReduce（每次算子） | 单节点内、带宽充裕 |
| 层维 | 流水线并行（PP） | 模型层段 | 点对点（每 micro-batch） | 跨节点、$p \le 16$ |
| 序列维（LayerNorm 等） | 序列并行（SP / Megatron-SP） | LN/Dropout 的序列份 | 少量 AllReduce | 配合 TP 省冗余激活 |
| 序列维（注意力） | Context Parallel（CP） | 注意力整条序列 | Ring 点对点 或 All-to-All | 超长上下文（$s \ge 32K$） |

**辨析｜易错点：**
- **SP 不切注意力**：Megatron-SP 只处理 LayerNorm/Dropout；真正切注意力的是 CP。不要把两者混为一谈。
- **CP 不等于 TP 的序列版**：TP 按隐藏维/头切，通信密集（每个算子一次集合通信）；CP 按序列切，通信次数少但单次数据量大（尤其 All-to-All）。
- **SP/CP 不是数据并行**：数据并行复制整个模型、切 batch；SP/CP 复制 batch、切序列——一个 batch 要由多卡共同完成。
- **CP 与 FlashAttention 不冲突**：CP 的「分块注意力」内部仍可用 FlashAttention 加速，二者是不同层次的技术。

## 6 小结

- **长序列的敌人是 $s^2$ 激活**：注意力分数矩阵随序列长度平方增长，TP/PP 都救不了。
- **Megatron-SP**：把 LayerNorm/Dropout 沿序列维切开，消除 TP 下的冗余激活，是 TP 的增强。
- **Context Parallel**：把注意力本身沿序列切开，代表方案为 Ring Attention（环形轮转 K/V + online softmax）与 DeepSpeed Ulysses（All-to-All 重排）。
- **核心收益**：激活显存按切分数 $p$ 摊薄，长序列（数十万 token）训练成为可能。
- **选型**：短序列用 TP+SP 足够；长序列叠加 CP，具体用 Ring 还是 Ulysses 视序列长度与网络拓扑。

## 7 进阶与延伸

**动手体验长序列的显存压力**：用 PyTorch 把序列长度从 4096 逐步加到 128K，同时观察 attention 分数矩阵的显存（$s^2$）。你会看到它按 $s^2$ 暴涨——这正是 CP 存在的全部理由。

**几个值得进一步挖的方向**：

- **Ring Attention 的 online softmax 精度**：分块 softmax 用「校正因子」合并，浮点误差与分块粒度有关。长到 1M token 时，误差会不会累积到影响训练？这是序列并行与数值稳定性（第六篇）的交汇点。
- **CP 与 FlashAttention 的组合**：CP 是「序列切分」、FlashAttention 是「IO 优化」——两者嵌套时，通信块与计算块怎么对齐才能既省显存又省带宽？这是当前长序列训练的前沿工程。
- **Ulysses 的 All-to-All 代价**：Ulysses 每次注意力要两次 All-to-All，通信量 $O(sh)$。当 $s$ 极大时，Ring 的 $O(s)$ 通信更优——两种方案的切换点在哪？可以用「通信量公式」画一条交叉曲线。

**自测题**：为什么说「CP 吃网络带宽，而 TP 吃节点内 NVLink」？如果 CP 的通信组跨了节点，你会怎么设计网络拓扑来配合它？

## 8 动手实践清单

- 用一个小 Transformer 把序列长度从 4K 加到 64K，观察 attention 激活按 $s^2$ 增长的曲线。
- 开 Megatron-SP，对比「开/关」时 LayerNorm/Dropout 的激活显存。
- 用 Ring Attention 的实现跑一次长序列训练，观察通信与计算的重叠。
- 对比 DeepSpeed Ulysses 与 Ring Attention 在相同序列长度下的吞吐。
- 把「序列并行」与「张量并行」同时开，验证激活被双重摊薄。
- 用 profiler 检查「CP 通信是否被计算盖住」，算出重叠率。
- 尝试把 CP 的通信组跨节点配置，观察网络带宽对吞吐的影响。
- 对比「CP 切分 vs TP 切分」对激活摊薄的差异。
- 用 online softmax 的校正因子，验证分块合并的数值正确性。
- 画出「序列长度 vs 最优切分方式」的选择曲线。

在下一节，我们转向 MoE 模型的特殊需求——当不同 token 被路由到不同专家，**专家并行（EP）**如何承担起通信重任。
