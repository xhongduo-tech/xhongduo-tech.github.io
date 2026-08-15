---
title: FlashAttention-1：IO 感知的精确注意力
date: 2026-08-07
---

# FlashAttention-1：IO 感知的精确注意力

<div class="epigraph">
<p>计算不是瓶颈，搬运才是。</p>
<footer>—— 李兆隆（Tri Dao），FlashAttention 作者</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ FlashAttention 论文（Dao et al., 2022, NeurIPS） ｜ 2026-08-07</p>
</div>

## 为什么从 FlashAttention 开始

前面讲了一路的访存瓶颈，现在终于到了直接对注意力动手的优化。标准注意力实现有一个几乎荒谬的浪费：**计算 $QK^T$、softmax、$PV$ 的每一步，都把中间结果（巨大的 $S = QK^T$ 矩阵）写回显存，下一次运算再读出来**。注意力算起来快，但这个「写-读-写-读」的乒乓把时间全耗在显存 IO 上。FlashAttention 的核心理念只有一句话：**让计算在 SRAM（片上高速内存）里完成，中间矩阵根本不出芯片**。<span class="marginnote">FlashAttention-1（2022）是后来 FA2、FA3、FlashDecoding 的起点。理解它的关键是分层存储：<strong>SRAM（~20MB、快）vs HBM（大、慢）</strong>。算力与带宽的鸿沟在注意力上尤其致命——注意力是典型的「算得少、搬得多」。</span>

本篇讲 FlashAttention 的动机、分块（tiling）与 online softmax 两项核心技术，以及为什么它是「精确」（无损）算法。

## 1 标准注意力为什么慢：IO 复杂度

标准的缩放点积注意力（以单头为例）：

$$S = QK^T, \quad P = \text{softmax}(S), \quad O = PV$$

三个矩阵运算，每个都要把中间结果经 HBM 搬一遍。设序列长度 $N$、头维度 $d$，$Q, K, V$ 各是 $N \times d$。标准实现的总 IO（以读写 HBM 的字节计）约为：

$$\text{IO} = O(N^2 + N d)$$

其中 $N^2$ 是 $S$、$P$ 矩阵的体积。序列越长，$N^2$ 项越大——**注意力的 IO 随序列长度平方增长**，这正是长序列推理慢的根本原因之一。<span class="marginnote">对比计算量也是 $O(N^2 d)$，但 GPU 的算力远大于带宽，所以<strong>IO 是注意力的实际瓶颈</strong>。Roofline 模型下，注意力位于「带宽受限区」。</span>

更糟的是，标准实现需要**显存足够装下 $N \times N$ 的 $S$ 矩阵**——序列 128k 时，$S$ 有 128k² 个元素，128k×128k×2 字节 ≈ 32 GB，直接爆显存。**注意力是「算得起、存不起」的经典案例**。

这里要厘清一个关键区分：**注意力的计算量（FLOPs）与访存量（IO）是两回事**。$QK^\top$ 的矩阵乘法本身有 $O(N^2 d)$ 次运算，A100 的 Tensor Core 一眨眼就能算完；真正拖住时间的，是把 $N \times N$ 的中间矩阵写进、读出 HBM 的搬运。Roofline 模型下注意力落在「带宽受限区」——**优化注意力就是优化 IO，不是优化 FLOPs**，这正是 FlashAttention 的全部出发点，也是它与本专题《访存瓶颈》那一篇共用同一套语言的原因。

## 2 分块：把注意力切成 SRAM 装得下的块

FlashAttention 的核心是**分块（tiling）**：不把整个 $QK^T$ 算出来，而是把 $K$、$V$ 沿序列维切成块，逐个读入 SRAM，在片上完成「部分注意力」，再增量累积输出。

算法骨架（每块）：

1. 加载 $Q$ 的一个块 $Q_i$ 与 $K$ 的一个块 $K_j$ 进 SRAM；
2. 算部分分数 $S_{ij} = Q_i K_j^T$（在 SRAM 内，不落 HBM）；
3. 对 $S_{ij}$ 做**在线 softmax**（见下节），得到部分 $P_{ij}$；
4. 算部分输出 $O_i \mathrel{+}= P_{ij} V_j$，并同步更新「迄今的 softmax 统计量」。

遍历所有 $K_j$、$V_j$ 块后，$O_i$ 就是正确的输出——**全程 $S$、$P$ 从未整体写入 HBM**。

IO 复杂度因此降为：

$$\text{IO} = O(N^2 \cdot d^2 / M)$$

其中 $M$ 是 SRAM 容量。当 $M$ 足够大时，IO 从 $O(N^2)$ 降到接近 $O(N)$——**长序列的 IO 瓶颈被打开**。

分块为什么能把 IO 拉回线性，直觉上想一遍：块与块之间只有「把 $Q_i$、$K_j$ 读进 SRAM」和「把 $O_i$ 写回」这两类搬运，前者总量是 $(N/d_m)^2$ 块 × 每块 $O(d_m \cdot d)$ 字节 ≈ $N^2 d^2 / M$，后者是 $O(Nd)$。**真正被消灭的是那块 $N \times N$ 的 $S$ 矩阵——它从头到尾没有完整出现过**。序列越长，这块被消灭的矩阵越大，收益越悬殊。

## 3 在线 softmax：不用等全局 max 的归一化

标准 softmax 需要先扫一遍算全局最大值（数值稳定），再算 exp 和。分块后**不能先看完整列再归一化**，于是 FlashAttention 用「在线（online）softmax」增量维护统计量：维护当前块序列中的「滚动最大值」$m$ 与「滚动指数和」$l$，每处理一个新块就**修正**之前块的归一化系数。

具体地，处理到块 $j$ 时：

$$m_{\text{new}} = \max(m, \max(S_{ij})), \quad l_{\text{new}} = l \cdot e^{m - m_{\text{new}}} + \sum e^{S_{ij} - m_{\text{new}}}$$

同时把已累积的输出按 $e^{m - m_{\text{new}}}$ 缩放修正。**每块都带着「还没归一化」的部分输出前进，最后统一除以 $l$**——数学结果与一次性 softmax 逐位一致。

**FlashAttention 是精确算法**：online softmax 的中间修正保证了最终输出与标准实现**数值相同**（除浮点舍入外），不引入任何近似。这与投机解码、量化形成鲜明对比——它是「无损」的加速。<span class="marginnote">这是 FlashAttention 系列最容易被低估的性质：<strong>它优化的是 IO 路径，不是数学</strong>。模型精度零损失，任何模型都可以直接换用。</span>

在线 softmax 的修正为什么成立，值得停下来想一遍：$l \cdot e^{m - m_{\text{new}}}$ 这一项，是把「之前所有块的指数和」从旧的 max 基准 $m$ 平移到新的 max 基准 $m_{\text{new}}$——因为每个 $e^{s_{ij} - m}$ 都要乘 $e^{m - m_{\text{new}}}$ 才能变成 $e^{s_{ij} - m_{\text{new}}}$，输出侧的累积量也按同一因子缩放。这套「滚动基准」的做法与数值分析里的 log-sum-exp trick 一脉相承，只是把「先扫一遍找全局 max」改成了「边扫边维护滚动 max」，让归一化也能分块进行。

## 4 数值算例：序列长度与 IO 收益

把「IO 随序列平方增长」落成具体数字。设 $d = 128$、FP16（$b = 2$ 字节）、SRAM 容量 $M \approx 20$ MB，对比标准实现与 FlashAttention 在不同序列长度 $N$ 下每步写 HBM 的字节量级：

| 序列长度 $N$ | 标准实现 IO（$N^2$ 项主导） | FlashAttention IO（接近线性） | 相对收益 |
| --- | --- | --- | --- |
| 1k | $\approx N^2$（约 $10^6$ 元素级） | $O(Nd)$，线性 | 约 2 倍 |
| 8k | $64 \times 10^6$ 元素级 | 线性 | 约 4–8 倍 |
| 32k | $10^9$ 量级 | 线性 | 约一个数量级 |
| 128k | $1.6 \times 10^{10}$，**先爆显存** | 线性，可跑 | 一个数量级以上 |

读这张表抓住两点：

- **标准实现被 $N^2$ 项拖死**：序列从 1k 涨到 128k，中间矩阵从几 MB 涨到 32 GB，先爆显存、再谈速度——长上下文推理对标准实现是「内存墙」，不只是「带宽墙」。
- **FlashAttention 把平方项打开**：分块让 IO 降到 $O(N)$ 量级，序列越长收益越大，这是长上下文（128k、1M）推理几乎必须用 FlashAttention 系 kernel 的原因。

一个反直觉的结论：**FlashAttention 对短序列（几十 token）的收益微乎其微**。短序列时 $N^2$ 项本来就小，IO 与算力都吃不饱，分块反而引入额外调度开销。它真正的主场是长 prefill 与长上下文——这与本专题《Prefill 与 Decode》里「Prefill 决定 TTFT」的结论正好呼应：**长 prefill 正是靠 FlashAttention 这类把 IO 压到极限的 kernel 才撑得住**。

## 5 公式解析：IO 复杂度与 Roofline

把标准与分块实现的 IO 复杂度对比放在一起：

$$T_{\text{standard}} \approx \frac{O(N^2 d + N^2)}{B}, \qquad T_{\text{flash}} \approx \frac{O(N^2 d^2 / M + N d)}{B}$$

- **第一步，读标准项**：$N^2 d$ 是 $S$ 与 $P$ 的读写（各 $N^2$ 元素），$N d$ 是 $Q,K,V,O$ 的读写。**序列平方项主导**。
- **第二步，读分块项**：$N^2 d^2 / M$ 来自「每对块读入 $Q_i K_j$ 的量」：块数 $(N/d_m)^2$ 乘以每次读入的字节 $O(d_m \cdot d)$，其中 $d_m$ 是块大小，受 $M$ 约束。当 $d_m \approx \sqrt{M}$ 时，该项降为 $N^2 d^2 / M$。
- **第三步，看阈值**：当 $N \gg d^2/M$（长序列）时，分块把平方项压到可忽略，IO 逼近 $O(Nd)$（线性）——**序列越长，FlashAttention 相对标准的收益越大**。典型收益：序列 1k 时约 2 倍，序列 64k 时可达一个数量级。

## 6 辨析｜易错点

- **误区一：以为 FlashAttention 改变了注意力公式。** 没有。它输出的数学结果与标准 softmax 注意力逐位一致，只是 IO 路径不同。任何把 FA 说成「近似注意力」的介绍都是错的——它的卖点恰恰是「精确」。
- **误区二：以为分块会让长序列变慢。** 分块的调度开销是常数级的，且随序列增长被 $N^2$ 项的节省彻底淹没。短序列收益小、长序列收益大，但不会「变慢」。
- **误区三：以为在线 softmax 是近似。** 它只是改变归一化的计算顺序——最终除以的 $l$ 与标准实现的指数和逐位一致，是精确的。
- **误区四：以为 FA 只对训练有用。** 训练与推理都受益：推理侧的长 prefill 是 FA 的主场；而 Decode 阶段每步只有一个新 token，FA 的收益有限，要靠 FlashDecoding 接力（见本专题《FlashDecoding 与长序列 Decode 加速》）。

**FlashAttention 收尾一句**：它是「让注意力不出芯片」的 IO 优化——分块进 SRAM、在线 softmax 精确归一化，把注意力的 IO 从 $O(N^2)$ 拉到 $O(N)$ 量级。理解它，就理解了长上下文时代几乎所有注意力 kernel 的底层逻辑。

## 7 小结

- **标准注意力的瓶颈是 IO**：$S$、$P$ 矩阵反复经 HBM 读写，IO 随 $N^2$ 增长，且 $N \times N$ 矩阵本身存不下。
- **分块（tiling）**：$K$、$V$ 切成块进 SRAM，部分注意力在片上算完，$S$、$P$ 不落 HBM。
- **在线 softmax**：滚动维护 max 与指数和，每块修正归一化，最终结果与标准实现逐位一致。
- **精确算法**：优化 IO 路径而非数学，模型精度零损失，任何模型可直接换用。
- **IO 复杂度**：从 $O(N^2)$ 降到 $O(N)$ 量级，序列越长收益越大。
- **适用场景**：长 prefill 与长上下文是 FA 的主场，短序列收益有限；Decode 阶段由 FlashDecoding 接力。

在下一节，我们看 FlashAttention 的第一次迭代——**FlashAttention-2：更好的并行与工作分配**，把 GPU 的并行度与线程利用率做到极限。
