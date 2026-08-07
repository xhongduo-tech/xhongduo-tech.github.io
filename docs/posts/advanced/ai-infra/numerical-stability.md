---
title: 数值稳定性问题：溢出、下溢与 attention softmax 的精度
date: 2026-08-07
---

# 数值稳定性问题：溢出、下溢与 attention softmax 的精度

<div class="epigraph">
<p>数值误差是训练里最安静、也最致命的 bug。</p>
<footer>—— 约翰 · 威尔金森（James H. Wilkinson，数值分析先驱）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ 数值线性代数与 FlashAttention 论文 · 训练稳定性篇 ｜ 2026-08-07</p>
</div>

## 为什么从数值稳定性开始

前面讲尖刺时提到「数值层」是成因之一。但数值问题远不止尖刺——它是训练里**最隐蔽的慢性病**：不报错、不崩溃，只是悄悄让模型收敛变差、训练更脆。溢出（overflow）、下溢（underflow）、softmax 的精度陷阱，每一个都值得单独开课。

为什么大模型训练格外容易踩数值坑？因为混合精度（FP16/BF16）把数值的动态范围压窄，而 attention 里的 softmax 又天然包含「大数相减」的高危操作。本篇从三个层面讲透数值稳定性：数怎么溢出、softmax 怎么爆、以及工程上怎么防。

## 1 溢出与下溢：浮点的边界

回忆浮点表示：一个数 $x = (-1)^s \times (1+m) \times 2^e$，由符号、指数、尾数决定。

- **溢出（overflow）**：$x$ 超过格式上限 → 变成 `inf`。FP16 上限 65504，BF16 上限与 FP32 相同（$3.4\times10^{38}$）。
- **下溢（underflow）**：$x$ 小于最小正规数 → 舍入成 0。FP16 下约 $6\times10^{-5}$ 以下直接变 0。

在训练里，溢出和下溢的后果不同：**溢出会传播**（inf 参与后续运算产生 NaN），**下溢会悄悄丢信息**（小梯度变成 0，模型学不到）。<span class="marginnote">下溢比溢出更危险，因为溢出至少会炸出 NaN 引起注意，下溢则是「无声的精度损失」——你甚至不知道哪些小梯度被舍入成了 0。这就是混合精度里 loss scaling 存在的深层原因：把梯度整体抬离下溢区。</span>

## 2 softmax 的数值陷阱：大数相减

attention 的核心是 softmax：$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$。它的数值风险在 `e^{z_i}` 上：

- 若 $z_i$ 较大（如 50），$e^{50} \approx 5\times10^{21}$，FP16 直接溢出成 inf。
- 溢出后 $\frac{inf}{inf} = NaN$，整个 attention 矩阵报废。

标准对策是**最大值平移（max subtraction）**：

$$\text{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \qquad m = \max_j z_j$$

**因为 softmax 对平移不变**：分子分母同乘 $e^{-m}$，结果不变。减去最大值后，$z_i - m \le 0$，$e^{z_i-m} \le 1$，永不溢出。<span class="marginnote">最大值平移是「用数学恒等式消除数值风险」的教科书案例：softmax 的平移不变性让「减去最大值」在数学上是自由操作，却把指数从「可能爆」变成「必然安全」。任何手写 attention 都必须做这一步，否则训练必炸。</span>

## 3 FlashAttention 的在线 softmax：把精度问题升级成 IO 问题

FlashAttention 引入的 **online softmax** 更精巧：它不预先知道整行的最大值，而是**在线统计**——边遍历分块边更新最大值与归一化常数，最后用校正因子把分块结果合并。

核心公式（简化）：

$$\text{softmax}(z) \approx \frac{\sum_{\text{block}} e^{z - m_{\text{global}}} \cdot V_{\text{block}}}{\sum_{\text{block}} e^{z - m_{\text{global}}}}$$

- 每块算局部最大值 $m_{\text{block}}$，全局最大值 $m_{\text{global}} = \max(m_{\text{global}}, m_{\text{block}})$。
- 若块最大值更新，旧块的结果要乘校正因子 $e^{m_{\text{old}} - m_{\text{new}}}$。
- 全程只保留 $O(1)$ 的统计量，数值上仍等价于「减全局最大值」的经典 softmax。<span class="marginnote">online softmax 的妙处在于「先分块计算、再精确校正」：它不需要看完整行就能保证数值安全，这让 attention 可以分块流式处理而不牺牲正确性。它把「数值稳定性」从「必须看全局」解耦成「可以在线」，是 FlashAttention 能省 IO 的前提。</span>

## 4 数值稳定性的工程防线

工程上防数值问题有几道防线，按性价比排序：

1. **最大值平移**：所有手写 softmax/exp 都做（无脑、免费）。
2. **梯度裁剪**：把梯度范数钳住，防一步爆炸。
3. **loss scaling（FP16）**：把梯度抬离下溢区。
4. **BF16 替代 FP16**：范围同 FP32，彻底免去溢出担忧。
5. **LayerNorm 的 epsilon**：分母加一个小数 $\epsilon$，防除零（$\epsilon$ 太小在低精度下会下溢）。
6. **累积用 FP32**：attention 分数、softmax 分母、以及任何跨张量累加，在 FP32 里做再转回。<span class="marginnote">最后一条「FP32 累加」被反复强调：FP16/BF16 的矩阵乘累加器在 Tensor Core 上本就是 FP32 的（硬件保证），但用户手写的 reduce、mean、sum 如果不显式升到 FP32，就会在下溢区悄悄损失。检查代码里每个 `torch.sum`/`.mean()` 的精度，是数值调试的日常。</span>

## 5 公式解析：最大值平移为什么数值安全

对比两种情况。设 logits 向量 $z = [z_1, \ldots, z_n]$，其中 $z_{\max} = 100$。

**不做平移**（危险）：

$$e^{z_{\max}} = e^{100} \approx 2.7 \times 10^{43} \longrightarrow \text{FP16/BF16 溢出为 } \infty$$

**做平移**（安全）：

$$e^{z_i - z_{\max}} \le e^{0} = 1, \quad \forall i$$

- **$z_i - z_{\max}$（平移后的 logit）**：最大值为 0，其余为负。所有指数项落在 $[0, 1]$，无溢出风险。
- **分母 $\sum_j e^{z_j - z_{\max}}$**：在 $[1, n]$ 之间，安全。
- **结果不变**：分子分母同乘 $e^{-z_{\max}}$，$\frac{e^{z_i - z_{\max}}}{\sum e^{z_j - z_{\max}}} = \frac{e^{z_i}}{\sum e^{z_j}}$。

**一句话**：最大值平移把「指数可能爆成 inf」的输入，硬生生搬进 $[0,1]$ 的安全区，而数学结果一字不差。它是数值稳定性里「零成本、必然收益」的典范。<span class="marginnote">顺带一提，这也能解释为什么 attention 的 logits 常被缩放到「除以 $\sqrt{d}$」：缩放本身把 logits 的典型量级压到 $O(1)$，配合最大值平移，double 保险。缩放还有个正交的目的——保持 softmax 的梯度合理（《大模型原理》篇的 attention 缩放讨论）。</span>

## 6 辨析｜易错点：数值稳定性的常见误区

**辨析｜易错点：**
- **「BF16 不会溢出所以不用管精度」是错觉**：BF16 范围大但尾数只有 7 位，累加精度仍要 FP32 兜底。
- **「softmax 平移只影响精度不影响结果」不准确**：平移在数学上严格等价，但低精度下「减最大值」仍会损失小 logit 的相对精度——极端情况要配合更高精度的累加。
- **「梯度裁剪会破坏收敛」过度担心**：裁剪引入偏差，但大模型训练里经验上利大于弊；真在意可以用「按层裁剪」等精细变体。
- **`epsilon` 不是随便填的**：LayerNorm 的 $\epsilon$ 若低于当前精度能分辨的最小量（如 FP16 下 $10^{-5}$），就形同虚设。
- **别忽视日志里的 NaN/Inf 告警**：一次 NaN 往往预示「后续所有步全废」，发现即止损（回滚/跳过）。

## 7 小结

- **两类错误**：溢出（inf/NaN，传播快）与下溢（舍入成 0，无声损失）。
- **softmax 的陷阱**：`e^{z}` 可能溢出，标准对策是最大值平移（softmax 平移不变性）。
- **FlashAttention 的 online softmax**：分块在线统计 + 校正因子，把数值安全与 IO 优化统一。
- **六道防线**：最大值平移、梯度裁剪、loss scaling、BF16、epsilon、FP32 累加。
- **核心心法**：把「高危运算」搬进安全区（平移、缩放、FP32 累加），数学结果不变，数值风险消失。

## 8 进阶与延伸

**动手复现一次 softmax 溢出**：写一个不含「最大值平移」的朴素 softmax，输入一个含 100 的 logits 向量——BF16 下你会得到 inf/NaN。加上 `m = max(z)` 平移后重跑，一切正常。这就是「一次复现胜过十次讲解」。

**几个值得进一步挖的方向**：

- **`torch.nn.functional.softmax` 的隐式平移**：PyTorch 的 softmax 内部已经做了最大值平移——但手写 kernel、手写 attention 时没人替你兜底。这是「框架安全 ≠ 手写安全」的经典案例。
- **FP16 下的 LayerNorm epsilon**：`eps=1e-5` 在 FP16 下「形同虚设」——因为 FP16 的最小可分辨量约 $6\times10^{-5}$。换 BF16 或调大 eps，你会看到数值行为的变化。
- **FlashAttention 的 online softmax 与「全局 vs 分块」**：分块统计的校正因子 $e^{m_{\text{old}} - m_{\text{new}}}$ 在极端数值下会怎样？这联系到 FP8 篇的「延迟缩放」——都是「用历史信息做此刻决策」。

**自测题**：为什么「最大值平移」在数学上严格等价、在数值上却救命？如果你能说清「平移不改变结果、只改变计算中间值的范围」，就抓住了数值稳定的精髓。

## 9 动手实践清单

- 写一个不含最大值平移的朴素 softmax，用含 100 的输入复现溢出。
- 加上 `m = max(z)` 平移后重跑，确认结果不变且不再溢出。
- 检查你代码里每个 `torch.sum`/`.mean()` 的累加精度。
- 验证 LayerNorm 的 epsilon 在 FP16 下是否「形同虚设」。
- 用 BF16 替代 FP16，观察溢出是否消失。
- 在 attention 里检查「除以 √d」的缩放是否到位。
- 用 profiler 找出「FP32 累加」在哪些算子被自动使用。

在下一节，我们转向训练稳定性的「保险丝」工程——**断点续训（checkpointing）**：保存什么、保存频率与一致性。
