---
title: 鞅差序列与 Azuma 不等式
date: 2026-08-07
---

# 鞅差序列与 Azuma 不等式

<div class="epigraph">
<p>即使每一步都依赖过去，只要期望不偏不倚，总和也会像独立变量一样紧紧聚拢。</p>
<footer>—— 平野 · 阿祖马（Kazuoki Azuma）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§6.5 ｜ 2026-08-07</p>
</div>

## 相依变量也能「集中」

独立同分布的 Chebyshev / Hoeffding 不等式说「独立变量之和紧密聚集在均值附近」。但现实里的变量几乎总是**相依的**：随机梯度下降每步的噪声依赖当前参数、在线算法每步的损失依赖已学的模型。独立假设不成立时，还能不能得到集中不等式？

答案是**鞅差序列（martingale difference sequence）**——它只要求「每步的条件期望为零」，不要求独立。而 **Azuma 不等式**在「鞅差有界」的条件下，给出与 Hoeffding 几乎一样漂亮的指数界。**它是「相依情形下的 Hoeffding」，是现代机器学习理论（随机算法、在线学习、MCMC 收敛）最常用的概率工具之一。**<span class="marginnote">鞅差与独立差的区别：<strong>独立差要求 $D_n$ 与全部历史独立；鞅差只要求 $E[D_n \mid \text{过去}] = 0$</strong>。后者允许「大小依赖过去」，只要「期望方向不偏」。这个放宽让 Azuma 不等式适用于几乎所有「迭代更新」算法。</span>

本节目标：定义鞅差序列、陈述并证明 Azuma 不等式、掌握它在算法分析中的用法。

## 1 鞅差序列

**鞅差序列（martingale difference sequence, MDS）**：随机序列 $D_1, D_2, \dots$ 称为鞅差，若对每个 $n$，
$$
E\big[ D_n \mid D_1, \dots, D_{n-1} \big] = 0.
$$
**每个 $D_n$ 在给定过去的条件下均值为 0——公平的增量。** 部分和 $X_n = \sum_{i=1}^n D_i$ 是鞅；反之任何鞅的增量都是鞅差。

**例（随机梯度下降的噪声）**：$D_n = $ 第 $n$ 步随机梯度的偏差（$g_n - E[g_n \mid \text{过去}]$）——条件期望为零，是鞅差，即使 $g_n$ 之间强相关。<span class="marginnote">SGD 收敛性分析的经典入口：<strong>把随机梯度写成「真梯度 + 鞅差噪声」，鞅差部分用 Azuma 控制，真梯度部分用凸性收缩</strong>。鞅差是「去均值后的噪声」的通用框架。</span>

## 2 Azuma 不等式

**Azuma-Hoeffding 不等式**：设 $D_1, D_2, \dots$ 是鞅差序列，且 $|D_n| \le c_n$（几乎必然有界）。则对任意 $a > 0$，
$$
P\Big( \Big| \sum_{i=1}^n D_i \Big| \ge a \Big) \le 2 \exp\!\left( -\frac{a^2}{2 \sum_{i=1}^n c_i^2} \right).
$$

**读法**：鞅差之和偏离 0 超过 $a$ 的概率，以指数衰减，衰减率由「步长界平方和」$\sum c_i^2$ 决定。**与独立 Hoeffding 的差别只有「$c_i$ 可以每步不同」**——$c_i$ 都等于 $c$ 时，正是 Hoeffding。<span class="marginnote">Azuma 不等式是「相依 + 有界」下最优的集中界：<strong>常数 2 与指数核 $e^{-a^2/(2\sum c_i^2)}$ 不能改进（在无更多假设时）</strong>。它把 Hoeffding 从「独立」推广到「鞅差」，而代价只是把方差和换成步长界平方和——在很多应用里够用了。</span>

## 3 公式解析：Azuma 不等式的证明

**目标：用「条件 Hoeffding」逐步证明 Azuma 不等式，理解指数核从哪来。**

第一步，用 Markov 不等式的指数化版本（Chernoff 技巧）。对 $\theta > 0$：
$$
P\Big(\sum D_i \ge a\Big) = P\Big(e^{\theta \sum D_i} \ge e^{\theta a}\Big) \le e^{-\theta a}\, E\big[ e^{\theta \sum D_i} \big].
$$
第二步，逐项剥条件期望。因为 $D_n$ 是鞅差且 $|D_n| \le c_n$，Hoeffding 引理（条件版本）给出
$$
E\big[ e^{\theta D_n} \mid \text{过去} \big] \le \exp\!\Big( \frac{\theta^2 c_n^2}{2} \Big).
$$
第三步，链条式相乘。把 $E[e^{\theta\sum D_i}]$ 从后往前逐项取条件期望：
$$
E\big[ e^{\theta \sum D_i} \big] \le \prod_{i=1}^n \exp\!\Big( \frac{\theta^2 c_i^2}{2} \Big) = \exp\!\Big( \frac{\theta^2}{2} \sum_{i=1}^n c_i^2 \Big).
$$
第四步，优化 $\theta$。对 $\theta$ 最小化 $e^{-\theta a + \theta^2 \sum c_i^2/2}$，取 $\theta = a/\sum c_i^2$，得单边界 $e^{-a^2/(2\sum c_i^2)}$；两边合起来乘 2，即 Azuma 不等式。

**这个推导为什么重要**：它展示了「Chernoff 技巧 + 条件 Hoeffding」的完整流程——**只要增量是鞅差且有界，指数矩就能被步长界控制，从而得到指数集中**。这套「逐项条件期望」的手法，是几乎所有鞅集中不等式的共同骨架。

## 4 应用：随机算法与在线学习的集中性

**例（在线学习 / 专家建议）**：在线梯度下降第 $n$ 轮的遗憾 $R_n$，其增量（一步的即时遗憾减去条件期望）是鞅差且通常有界。Azuma 不等式给出
$$
P\big( |R_n - E[R_n]| \ge a \big) \le 2 e^{-a^2/(2nc^2)}.
$$
**遗憾围绕期望的波动是指数集中的**——配合期望上界，直接得到「以高概率遗憾 $\le O(\sqrt n)$」的在线学习经典结论。<span class="marginnote">这是「遗憾界」的黄金套路：<strong>期望遗憾用在线凸优化分析，波动用 Azuma 控制，合起来就是高概率界</strong>。几乎每一篇在线学习论文都有「By Azuma's inequality…」这一步。</span>

**例（随机梯度下降）**：SGD 的参数在 $T$ 步内的累计噪声 $\sum_{t=1}^T D_t$（鞅差）有界，Azuma 给出「参数以高概率不跑偏 $O(\sqrt T)$」——这是「SGD 以 $1/\sqrt T$ 速率收敛」的概率版本。

**例（MCMC）**：马尔可夫链蒙特卡洛估计的偏差可用鞅差表示，Azuma 型界给出「估计以高概率接近真值」——这是 MCMC 诊断（第十篇）的误差分析基础。<span class="marginnote">Azuma 的普适性来源：<strong>它只需要「迭代 + 有界增量」两个结构，不要求独立、不要求分布</strong>。凡是「一步步更新、每步不偏不倚、增量有界」的算法，都能套 Azuma——这正是它在现代统计与机器学习里无处不在的原因。</span>

## 5 与 Chernoff/Hoeffding 的对照

| 不等式 | 要求 | 界的形式 |
| --- | --- | --- |
| Hoeffding | 独立有界 | $2e^{-a^2/(2n c^2)}$ |
| Azuma | 鞅差有界 | $2e^{-a^2/(2\sum c_i^2)}$ |
| Chernoff | 独立 + 矩生成 | $e^{-a^2/(2\sigma^2)}$ 型 |

**Azuma 是 Hoeffding 的「相依推广」**：把「独立」换成「鞅差」，把 $nc^2$ 换成 $\sum c_i^2$，其余不变。**记忆 Azuma = 记住「相依情形下把 Hoeffding 的方差换成步长界平方和」。**<span class="marginnote">还有一个延伸：<strong>McDiarmid 不等式</strong>（有界差不等式）可以看成 Azuma 的一个推论——它把「函数对输入的敏感性」翻译成鞅差的界。Azuma 家族是现代「集中不等式」的大本营，从随机算法到差分隐私都用它。</span>

## 6 小结

- **鞅差序列**：$E[D_n \mid \text{过去}] = 0$——每步条件期望为零，允许相依。
- **Azuma 不等式**：$P(|\sum D_i| \ge a) \le 2e^{-a^2/(2\sum c_i^2)}$，$|D_n| \le c_n$。
- **证明**：Chernoff 技巧 + 条件 Hoeffding 引理 + 逐项剥条件期望 + 优化 $\theta$。
- **应用**：SGD 噪声、在线学习遗憾、MCMC 误差——凡「迭代 + 有界增量」皆可套。
- **关系**：Azuma = Hoeffding 的鞅差推广；McDiarmid 是其推论。

在下一节，我们看看鞅差与 Azuma 在算法分析里的具体打法：**鞅在算法分析中的应用**——随机算法的期望与高概率复杂度分析。
