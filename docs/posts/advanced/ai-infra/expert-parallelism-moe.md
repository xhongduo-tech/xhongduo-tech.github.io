---
title: 专家并行（EP）：MoE 路由与负载均衡问题
date: 2026-08-07
---

# 专家并行（EP）：MoE 路由与负载均衡问题

<div class="epigraph">
<p>让每个 token 只调用它需要的那部分智能。</p>
<footer>—— 雅各布 · 德弗林（Jacob Devlin，语言模型研究者）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ GShard / Switch Transformer 论文（Shazeer et al., 2017; Fedus et al., 2021）· 并行策略篇 ｜ 2026-08-07</p>
</div>

## 为什么从专家并行开始

密集 Transformer 里，每一层 FFN 都作用于所有 token，算力被平均摊薄。**Mixture-of-Experts（MoE）** 反其道而行：把 FFN 换成一组「专家」，每个 token 只激活其中少数几个。于是**同样算力可以撑起大得多的参数量**——GPT-4、Mixtral、DeepSeek-V3 无不如此。<span class="marginnote">MoE 的参数量（总参数）远大于「激活参数」（每个 token 实际用到的参数）。比如 Mixtral 8x7B 有约 47B 总参数，但每个 token 只激活 2 个专家，激活参数约 13B。推理成本主要由激活参数决定，这就是 MoE 划算的根本原因。</span>

但要落地 MoE，训练系统必须先回答：**成百上千个专家放在哪、token 怎么找专家、专家之间怎么通信**。这就是**专家并行（Expert Parallelism, EP）**。它不像 TP 那样切矩阵，而是**把「专家」这个单位当成切分对象**，并由此引入 MoE 独有的两大工程问题：All-to-All 通信，与路由的负载均衡。

## 1 从密集 FFN 到稀疏 MoE

在标准 Transformer 中，每个 token 依次经过注意力与一个 FFN。MoE 把 FFN 替换为 $E$ 个专家（每个专家就是一个小的 FFN），再配一个**路由器（router/gating）**。对每个 token，路由器输出在 $E$ 个专家上的概率分布，只挑概率最高的 $k$ 个专家（通常 $k=1$ 或 $2$）来算：

$$y = \sum_{i \in \text{top-}k} g_i \cdot \text{FFN}_i(x)$$

其中 $g_i$ 是路由器给专家 $i$ 的权重。**参数量随 $E$ 增长，但每个 token 的算力只随 $k$ 增长**——这就是「稀疏激活」的含义。<span class="marginnote">这里的 sparse 指「对每个 token 来说，大部分专家是关闭的」。注意它和「稀疏矩阵」不是一回事：MoE 的稀疏是结构性的、显式的路由选择。</span>

## 2 EP 的切分思想：专家是切分单位

假设模型有 64 个专家、64 张 GPU。EP 的最朴素形式：**每张卡放 1 个专家，token 被路由器送到它该去的卡**。相比 TP 每层都切、每层都通信，EP 的结构更自然——专家本来就是「各管一摊」的独立计算单元，切分的边界清晰。

EP 的关键特性：

**专家参数不复制**：每个专家只存在于一张卡上，总参数量大也不怕，因为卡之间不冗余。
**token 是流动的**：数据并行里数据在本地、模型复制；EP 里**模型分片、token 跨卡流动**。
**与数据并行天然兼容**：EP 可以视为数据并行在 MoE 层的一种变形——非专家层照常数据并行，专家层按专家切分。

**专家并行把「搬运数据」与「搬运模型」的角色对调**：DP 搬梯度、TP 搬激活、EP 搬 token。这一句话是理解 MoE 训练通信的核心。

## 3 通信模式：token dispatch 与 combine

EP 的通信是 **All-to-All**：每一张源卡上都有若干 token 被路由到各个目标卡上的专家，所有卡需要同时互相交换数据。一次完整的专家调用分两半：

**Dispatch（分发）**：源卡把本地 token 按目标专家分组，发给对应目标卡。目标卡收齐后，把「本地专家要处理的所有 token」排成 batch 计算。
**Combine（回收）**：专家算完，把结果 token 按来源原路送回，各源卡再把「自己 token 在不同专家上的结果」加权求和（乘回路由权重 $g_i$）。

通信量估算：设每张卡有 $b$ 个 token，每个 token 的隐藏向量 $h$ 维，路由到专家时每个 token 的中间数据约 $O(h)$。dispatch 与 combine 各一次，**每张卡每次专家调用的通信量约 $O(b h)$**，与专家数 $E$ 无关——这很反直觉，因为通信只发生在「本卡 token 与它们的目标专家」之间。<span class="marginnote">All-to-All 的通信量与「每卡本地 token 数」成正比，而不是与总 token 数成正比。所以当 batch 很大、每卡 token 很多时，EP 通信会显著变大——这就是为什么 MoE 训练中 batch size 不能无限增大。</span>

## 4 负载均衡：EP 的头号敌人

EP 最大的工程难题不是通信，而是**不均衡**。如果路由器的决策高度偏斜——比如某几个专家被几乎所有 token 选中，其余专家空转——那么：

- **计算热点**：热门专家所在的卡算力打满，其他卡闲着，整体利用率暴跌。
- **通信热点**：所有卡都往同一张卡发数据，网络拥塞，All-to-All 性能崩坏。
- **显存失衡**：热门专家的输入 batch 超过容量，被迫丢弃 token，损失模型质量。

负载均衡问题有专门的定义：设 $f_i$ 是「实际被路由到专家 $i$ 的 token 比例」，$P_i$ 是「路由器对专家 $i$ 的平均路由概率」。**理想情况是 $f_i = P_i = 1/E$**，即 token 均匀铺开且路由概率与频率一致。任何偏离都会在公式层面被量化出来（见下节公式解析）。

## 5 公式解析：负载均衡 loss 与专家容量

**负载均衡损失（auxiliary load balancing loss）** 是 Switch Transformer 的经典设计：

$$\mathcal{L}_{aux} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

- **$E$（专家数）**：归一化系数，让 loss 尺度与专家数脱钩。
- **$f_i$（分配比例）**：这批 token 中，真正落到专家 $i$ 的比例。由数据驱动，训练时统计得到。
- **$P_i$（平均路由概率）**：路由器给专家 $i$ 的概率，所有 token 上的平均。
- **$\alpha$（系数）**：通常很小（如 0.01），保证它作为辅助 loss 不喧宾夺主。

直觉：若均衡，$f_i = P_i = 1/E$，则每一项 $f_i P_i = 1/E^2$，求和 $\sum = 1/E$，乘 $E$ 得 1——loss 为常数 $\alpha$。若某专家过热，$f_i$ 与 $P_i$ 同时升高，乘积变大，loss 上升，梯度就会压低路由器的偏斜。<span class="marginnote">为什么用 $f_i \cdot P_i$ 的乘积而不是只用 $f_i$？因为 $f_i$ 来自采样（不可导），$P_i$ 来自路由器（可导）。乘积让 loss 对路由参数可微，同时惩罚「实际分配」与「路由概率」双重偏斜。这是「把不可导的统计量裹进可导的代理」的经典手法。</span>

**专家容量（expert capacity）** 是防止单专家过载的硬约束：

$$\text{capacity} = \text{ceil}\left(\frac{T}{E} \cdot \text{capacity factor}\right)$$

其中 $T$ 是每批 token 总数，capacity factor 通常取 $1.0$–$1.25$。每个专家最多收 capacity 个 token，超出部分被丢弃（或走「drop token」旁路）。capacity factor 越大，丢得越少但浪费越多；$1.0$ 时若分布不均必丢 token，$1.25$ 是常用的折中。<span class="marginnote">「drop token」不是免费午餐：被丢弃的 token 在这层等于没被模型看到，梯度传不回来。Switch 论文发现 top-1 路由下丢少量 token 对质量影响可接受，但对训练稳定性敏感的任务，倾向用「no-drop」或高 capacity。</span>

## 6 辨析｜易错点：EP 与 DP/TP 的边界

| 方法 | 切什么 | token 在哪 | 通信 | 适合 |
| --- | --- | --- | --- | --- |
| 数据并行（DP） | 复制模型、切 batch | 本地 | 梯度 AllReduce | 小模型、大 batch |
| 张量并行（TP） | 权重矩阵 | 本地（分块） | 每算子 AllReduce | 单节点、隐藏维大 |
| 专家并行（EP） | 专家 | **跨卡流动** | All-to-All | MoE 模型、专家多 |

**辨析｜易错点：**
- **EP ≠ 数据并行**：DP 里每个数据副本走同一个模型；EP 里不同 token 走不同专家，模型是被切开的。
- **EP 不替代 TP**：EP 管专家层的切分；非专家层（注意力、embedding）仍需 DP/TP。MoE 模型通常是「非专家层用 DP/TP，专家层用 EP」的混合体。
- **All-to-All 不是 AllReduce**：AllReduce 是「大家各持一份、汇总成一份发给所有人」；All-to-All 是「每人都发不同的数据给每个人」，形状完全不一样（第二篇已详述）。
- **路由不等同于注意力**：路由器是一个小线性层，输出在专家上的 softmax，它没有跨 token 的信息交互，这是与注意力最本质的差别。

## 7 小结

- **MoE 的收益**：参数量随专家数 $E$ 增长，而每个 token 算力只随 top-$k$ 增长，稀疏激活划算。
- **EP 的本质**：把专家当切分单位，token 跨卡流动，通信靠 dispatch/combine 两次 All-to-All，量级约 $O(bh)$。
- **两大工程问题**：All-to-All 通信（吃带宽）与路由负载均衡（吃利用率）。
- **负载均衡双保险**：辅助 loss（软约束，可导地惩罚偏斜）与专家容量（硬约束，超容丢 token）。
- **与其它并行关系**：EP 管专家层，非专家层仍用 DP/TP；这是 MoE 训练的标准组合。

## 8 进阶与延伸

**动手观察 MoE 的负载分布**：跑一个 MoE 模型（如 Mixtral），打印每个专家实际收到的 token 数——你会看到即使有负载均衡 loss，分布也远非均匀。这正是 EP 里「热点专家」的来源。

**几个值得进一步挖的方向**：

- **capacity factor 的调参**：1.0 时几乎必丢 token，1.25 是常用值——但不同路由分布下最优值不同。怎么根据「路由熵」动态调 capacity？这是 MoE 推理与训练共同的调参课题。
- **EP 与 TP 在 MoE 层如何叠**：专家层用 EP、非专家层用 TP/DP——切分边界在哪、通信怎么衔接？DeepSeek-V3 的「细粒度专家 + 共享专家」路由设计是很好的案例。
- **All-to-All 的通信压缩**：dispatch 前能否量化/压缩 token 的隐藏向量？通信量 $O(bh)$ 里，$h$ 维能不能压——这是 MoE 通信优化的前沿。

**自测题**：为什么 EP 的通信量与专家数 $E$ 无关，只与每卡本地 token 数有关？想清楚这一点，你就理解了「batch 越大、EP 通信越贵」的直觉。

## 9 动手实践清单

- 跑一个 MoE 模型，打印每个专家的 token 分布，观察负载不均的程度。
- 调整 capacity factor 从 1.0 到 1.5，观察「丢 token 数」与「训练质量」的权衡。
- 用 profiler 拆出 dispatch 与 combine 两次 All-to-All 的时间占比。
- 计算不同 batch 下 EP 通信量（$O(bh)$），画出「batch vs 通信」曲线。
- 开启负载均衡 loss，观察路由分布的熵是否上升。
- 对照 Switch Transformer 的 loss 公式，手算一次「均衡 vs 偏斜」的 loss 值。
- 验证「EP 管专家层、DP/TP 管非专家层」的混合配置能否跑通。

在下一节，我们从「切分模型」转向「切分冗余」：**ZeRO-1/2/3** 如何通过砍掉 DP 里被复制的参数、梯度与优化器状态，让数据并行撑起更大的模型。
