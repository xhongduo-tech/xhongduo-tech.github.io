---
title: 分块注意力 Blockwise
date: 2026-08-11
---

# 分块注意力 Blockwise

<div class="epigraph">
<p>简单性是可靠性的先决条件。</p>
<footer>—— 艾兹格 · 迪杰斯特拉（Edsger W. Dijkstra）</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · 长上下文与注意力优化 ｜ 对标教材：Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention (NeurIPS 2022) ｜ 2026-08-11</p>
</div>

## 为什么从分块注意力开始

上一节我们说 FlashAttention 靠「三板斧」省掉了 $n \times n$ 矩阵的 HBM 往返：分块、在线 softmax、重算。这一节把第一板斧和第二板斧的**逐块细节**摊开——这正是「Blockwise」的全部含义。为什么分块之后 softmax 还是精确的？重缩放的公式长什么样？因果掩码怎么处理？这三个问题，是理解 FlashAttention 实现、乃至后来 FlashAttention-2 与各种衍生版本（flash-decoding、PagedAttention）的钥匙。

先说结论：**分块注意力是「把整行 softmax 换成多次局部 softmax + 按比例重缩放」**。数学上一个字都不改，只是把计算顺序从「先整行归一化、再加权」改成了「先局部加权、再统一缩放」。

## 1 分块主循环：一次内层迭代在算什么

FlashAttention 的前向是一个**双重循环**：外层遍历查询块 $Q_i$（大小 $B_r \times d$），内层遍历键值块 $K_j, V_j$（大小 $B_c \times d$）。对每一对块：

1. 从 HBM 载入 $Q_i, K_j, V_j$ 到 SRAM；
2. 在片上算分数块 $S_{ij} = Q_i K_j^{\top} / \sqrt{d_k}$（大小 $B_r \times B_c$，正好装进 SRAM）；
3. 对 $S_{ij}$ 做**局部 softmax** 并重缩放，累加进输出块 $O_i$（$B_r \times d$）；
4. 内层循环结束时，把 $O_i$ 与运行统计量写回 HBM。

关键约束是块大小的选择：SRAM 里要同时放下 $Q_i, K_j, V_j, S_{ij}, O_i$ 及两个统计向量，于是

$$
B_r = \left\lfloor \frac{M}{4d} \right\rfloor, \qquad B_c = \left\lfloor \frac{M}{4d} \right\rfloor
$$

$M$ 是 SRAM 容量。A100 上 $M \approx 192$ KB、$d = 64$ 时，块大约是 $64 \times 64$ 的量级。块大小的取法本身是一场三角权衡：$B$ 越大，SRAM 利用率越高、块间循环次数越少；但 $B$ 超过 SRAM 容量就装不下 $S_{ij}$，$B$ 太小则循环开销与带宽利用率双输。实际实现会在目标 GPU 上**自动搜索**最优块尺寸——这是「算力—带宽—容量」三角在工程侧的经典博弈。块取方阵（$B_r = B_c$）是为了让 $Q$ 与 $K$ 的搬运量对称；因果场景对角块可以放宽形状——块形状本身也是可调超参。把完整算法写成伪码（论文 Algorithm 1 的骨架），结构一目了然：

```text
设 Q, K, V 已按块划分；初始化 O = 0, l = 0, m = -inf
for 每个查询块 Q_i:
    for 每个键值块 K_j, V_j:
        S_ij = Q_i K_j^T / sqrt(d_k)          # 片上计算
        m_new = max(m_i, rowmax(S_ij))
        P_ij = exp(S_ij - m_new)
        l_i = l_i * exp(m_i - m_new) + rowsum(P_ij)
        O_i = O_i * exp(m_i - m_new) + P_ij V_j
        m_i = m_new
    O_i = O_i / l_i                             # 块结束，归一化
输出 O
```

双重循环的每一轮只接触 SRAM 内的三个块，HBM 只进出一次 $O_i$——「全程片上」不是修辞，是代码结构。这个伪码里没有任何一处出现「整张 $n \times n$ 矩阵」——HBM 全程只见 $O$ 与统计量，这就是 IO 优化的全部形态。一句话概括分块与朴素的差别：**朴素实现每层把 $n \times n$ 矩阵写进 HBM 一次，分块实现每层只把 $O(n)$ 的输出写进 HBM——显存墙塌了，因为「中间产物」不再有平方级的落地。**<span class="marginnote">这正是一节 CUDA 矩阵乘法课的分块做法：用共享内存手动切块、避免反复访问全局内存。FlashAttention 的贡献在于把这个老手艺扩展到「注意力这个端到端算子」的粒度，并顺带解决了 softmax 的跨块归一化问题。</span>

## 2 在线 softmax 的重缩放：核心公式

每一块 $S_{ij}$ 只能看到「整行」的一部分，它的局部最大值与局部总和并不是整行的。**在线 softmax（online softmax）**维护两个运行量：运行最大值 $m_i$ 与运行指数和 $l_i$。遇到新块时：

$$
\begin{aligned}
m_{\text{new}} &= \max(m_i, \ \mathrm{rowmax}(S_{ij})) \\
P_{ij} &= \exp(S_{ij} - m_{\text{new}}) \\
l_{\text{new}} &= l_i \cdot e^{m_i - m_{\text{new}}} + \mathrm{rowsum}(P_{ij}) \\
O_i &\leftarrow O_i \cdot e^{m_i - m_{\text{new}}} + P_{ij} V_j
\end{aligned}
$$

新块带来一个**更大的最大值** $m_{\text{new}}$（若它确实更大），那么之前所有已累加的输出 $O_i$ 与统计量 $l_i$，在旧的 $m_i$ 下算的指数都要打一个折扣 $e^{m_i - m_{\text{new}}}$——把「旧账」统一换算到新尺度上，再叠加新块的贡献。循环结束后，输出为 $O_i = O_i / l_i$（逐行除以总和）。

## 3 公式解析：重缩放为什么精确等价

拿一个两块的例子验证。设一行只有两个块，第一块的分数是 $s_1, s_2$（最大 $m_1$），第二块是 $s_3, s_4$（最大 $m_2 \ge m_1$）。整行 softmax 的权重应为

$$
w_j = \frac{e^{s_j - m_2}}{\sum_{j=1}^{4} e^{s_j - m_2}}
$$

分块实现走的路是：

- **第一块后**：$l^{(1)} = e^{s_1 - m_1} + e^{s_2 - m_1}$，$O^{(1)} = e^{s_1 - m_1} v_1 + e^{s_2 - m_1} v_2$。
- **第二块后**：$l^{(2)} = l^{(1)} \cdot e^{m_1 - m_2} + e^{s_3 - m_2} + e^{s_4 - m_2} = \sum_{j=1}^{4} e^{s_j - m_2}$。<span class="marginnote">关键的一步：$l^{(1)} \cdot e^{m_1 - m_2} = (e^{s_1 - m_1} + e^{s_2 - m_1}) e^{m_1 - m_2} = e^{s_1 - m_2} + e^{s_2 - m_2}$——旧的指数项被精确地重标定到新的最大值 $m_2$ 下。</span>

同理 $O^{(2)} = O^{(1)} \cdot e^{m_1 - m_2} + e^{s_3 - m_2} v_3 + e^{s_4 - m_2} v_4 = \sum_{j=1}^{4} e^{s_j - m_2} v_j$。

最终 $O / l = \frac{\sum_j e^{s_j - m_2} v_j}{\sum_j e^{s_j - m_2}}$，与整行一次 softmax 的凸组合**逐位相同**——唯一的差别是浮点舍入的次序。

**辨析｜易错点：** 重缩放系数用的是 $e^{m_i - m_{\text{new}}}$（旧值减新值），不是反过来。想想：$m_{\text{new}}$ 更大，旧的指数项在更大尺度下**变轻**，所以要乘以一个小于 1 的因子 $e^{m_i - m_{\text{new}}} \le 1$——方向反了，指数和与输出就全乱了。这是手写 FlashAttention 最常见的 bug。重缩放还带来一个数值上的隐藏收益：所有指数项都保持在「相对当前最大值」的尺度上，$e^{s_j - m}$ 恒有上界 1——无论序列多长，在线 softmax 的中间值永远不会真的大到溢出。

## 4 数值稳定性的意外收获

标准 softmax 要防溢出：若分数达到上千，直接算 $e^{s}$ 会爆。标准实现因此要先减整行最大值。FlashAttention 的在线版本把**减最大值内嵌进了循环**：每个局部块都减自己的 rowmax，而旧块的 $m_i$ 一直被保存——所以**无论序列多长、分数多大，指数始终不会溢出**。<span class="marginnote">这是在线 softmax 的经典论文 Milakov & Gimelshein 2018 的核心思想：把「减最大值」从一次性操作改成增量维护，既保证稳定性，又允许流式处理。</span>

于是分块注意力得到两项免费的保证：**数值稳定**（全程指数有界）与**精确等价**（重缩放是恒等变换），外加 **$O(n)$ 显存**（只存 $O_i$ 与统计量，不存 $n \times n$ 矩阵）。

## 5 工程细节：因果掩码与 FlashAttention-2

实际训练都是**因果注意力**（causal）：$i$ 位置的 token 不能看到 $j > i$。分块后掩码怎么处理？

**核心要点：** 把 $S_{ij}$ 中 $j \cdot B_c > (i+1) \cdot B_r$ 的整个块跳过（块在矩阵对角线上方），对角线上**部分可见**的块则把越界元素设为 $-\infty$。这个处理让 FlashAttention 对因果模型同样成立，且省掉近一半的块计算。<span class="marginnote">严格说，论文中因果版把对角块再细分：只取下半三角的微块，最大化跳过率。这是实现层面的细节优化，但直接影响训练吞吐。</span>

**反向传播同样分块重算**：不存 $n \times n$ 的 $P$，而是把 $O_i$、$l_i$ 与 $m_i$ 这些小块统计量存下，反向时从 SRAM 里的 $Q_i, K_j$ 重新生成 $P_{ij}$，再按注意力的四个梯度公式（对 $Q, K, V$ 与 $O$ 的梯度）就地更新。这是「重算」板斧在反向上的镜像：**用两倍前向的 FLOPs 换回 $O(n^2)$ 的显存**，对训练而言是决定性的交易。

**代价被写进了代码里**：分块注意力的所有好处，都建立在「手动把整段注意力融进一个 CUDA kernel」之上——`flash-attn` 的核心是手写 CUDA/CUTLASS，而非 PyTorch 组合算子。这让实现难度陡增：块大小调度、寄存器分配、线程布局、在线 softmax 的精确分支，处处是坑。**「分块」既带来了 IO 收益，也把注意力从「可组合的算子」变成了「手工调优的黑盒」**——这是长上下文优化里反复出现的隐性成本。<span class="marginnote">反向的块循环与正向完全对称，唯一新增的是对 $P_{ij}$ 的梯度累加。这也解释了为何 FlashAttention 训练能一口气跑到 64k+ 长度——显存与序列长度完全解耦。</span>

FlashAttention-2 沿用了分块与在线 softmax 的骨架，改动在于**并行策略**：让外层循环遍历 $K, V$ 块（而非 $Q$ 块），把块间循环交给更多线程并行，并在对角块用更小的 $B_c$ 避免浪费。它在 A100 上把速度再提约 2 倍。这一改动也说明：**分块骨架与线程级调度在每一层都互相咬合**——算法的块结构，直接决定了硬件怎么把活儿分给成千上万个核。事实上，flash-attn 在推理侧也几乎全程使用分块（flash-decoding 亦如此）——**分块已经不只是「FlashAttention」，而是注意力实现的默认形态**。正确性如何验收：社区常用「与朴素实现在随机输入上的逐位误差 < $10^{-5}$」作为门槛——数值等价不是口头承诺，而是可复现的测试。**窗口与分块的天然亲缘**：窗口掩码恰好是「按块跳过」的特殊情形，所以 flash-attention 用一个 `window_size` 参数就同时支持了稠密、窗口与全局——这是分块框架的设计红利。<span class="marginnote">若你读过本专题下一节，会发现滑动窗口注意力恰好也能写成「分块 + 跳过块外」的形式——分块骨架把稀疏与稠密两种注意力统一在了同一套实现里。</span>

## 6 小结

- 分块注意力 = 双重循环遍历 $(Q_i, K_j)$ 块，块内全程片上、不写中间矩阵。
- 块大小由 SRAM 容量决定：$B_r = B_c = \lfloor M / 4d \rfloor$ 的量级。
- 在线 softmax 用 running max + running sum 做重缩放，数学上与整行 softmax **精确等价**。
- 重缩放系数 $e^{m_i - m_{\text{new}}}$ 的方向是**旧减新**，方向反了是常见 bug。
- 减最大值内嵌进循环，数值稳定性是免费附赠的。
- 因果掩码 = 整块跳过 + 对角块裁半三角；FlashAttention-2 在并行策略上改进。
- 反向重算同样分块；分块实现是手写 CUDA kernel，工程成本是被低估的隐性代价。
- 块稀疏版在块粒度跳过窗口外块——稀疏与 IO 优化在块层握手，是两大流派的汇合点。
- 在线 softmax 是 FlashAttention 的「算法心脏」：它让分块不必向归一化妥协。

在下一节，我们将把稀疏与分块两套思路拼起来：**滑动窗口注意力**——既像 Longformer 一样只关注局部窗口，又能像 FlashAttention 一样按块高效实现，它是现代长上下文模型（Mistral、Gemma 等）默认架构的基石。
