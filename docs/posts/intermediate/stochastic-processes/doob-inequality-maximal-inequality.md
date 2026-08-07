---
title: Doob 不等式与极大不等式
date: 2026-08-07
---

# Doob 不等式与极大不等式

<div class="epigraph">
<p>过程中的每一个高点都被终值「管着」——极大值不会比终值过分嚣张。</p>
<footer>—— 约瑟夫 · 杜布（Joseph Doob）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§6.5 ｜ 2026-08-07</p>
</div>

## 管住「过程的最大值」

鞅的期望守恒只约束「每时刻的值」。但很多问题关心的是**过程中的最大值**：随机算法的最大误差、赌徒在过程中曾达到的最高赌本、估计算法运行中的最坏偏离。最大值 $\max_{k\le n} X_k$ 是「整条路径」的极端，比单个 $X_n$ 难控得多。

**Doob 不等式（Doob's maximal inequality）**给出了一个惊人的结论：**下鞅的最大值，可以被终值「管住」**——虽然 $\max_{k\le n} X_k \ge X_n$（显然更大），但它在概率意义下不「过分大」：
$$
P\Big( \max_{0 \le k \le n} X_k \ge \lambda \Big) \le \frac{E[X_n^+]}{\lambda}, \qquad X_n \text{ 下鞅}.
$$
<span class="marginnote">这像「概率版的极大值原理」：<strong>终值的期望定了，过程中的高点就有明确的概率上界</strong>。直觉：下鞅平均向上，若过程中某点摸到 $\lambda$，则从「此刻已很高」的起点出发，终值「大概率不低」——于是「摸高概率 × 终值期望」被联系起来。</span>

本节目标：陈述 Doob 极大不等式与其 $L^2$ 版本、给出证明直觉、并看到它在算法分析里的用途。

## 1 极大不等式

**Doob 极大不等式**：设 $X_n$ 是**非负**下鞅（或鞅的绝对值），则对 $\lambda > 0$，
$$
P\Big( \max_{0 \le k \le n} X_k \ge \lambda \Big) \le \frac{E[X_n]}{\lambda}.
$$

**这是 Markov 不等式的「过程版」**：Markov 说 $P(X_n \ge \lambda) \le E[X_n]/\lambda$，Doob 把 $X_n$ 换成过程中的最大值 $\max_{k\le n} X_k$，结论几乎一样——**最大值也被 $E[X_n]/\lambda$ 控制**。

**证明直觉（首达分解）**：设 $\tau = \min\{k : X_k \ge \lambda\}$（停时）。$\{\max \ge \lambda\} = \{\tau \le n\}$。在 $\tau$ 处停止后，$X_\tau \ge \lambda$。由下鞅的停止过程仍下鞅：
$$
E[X_n] \ge E[X_{\min(\tau, n)}] \ge \lambda\, P(\tau \le n),
$$
因为 $X_{\min(\tau,n)} \ge \lambda$ 当 $\tau \le n$，且非负。移项即得不等式。<span class="marginnote">证明的美妙：<strong>用「首次摸高时刻」这个停时把「最大值事件」翻译成「停时前终值的下界」</strong>。停时概念再次立功——把「整条路径的极端」化成了「一个停时 + 终值期望」。</span>

## 2 L² 版本：Doob 不等式

**Doob 不等式（$L^2$ 版本）**：设 $X_n$ 是鞅（或下鞅），则
$$
E\Big[ \max_{0 \le k \le n} X_k^2 \Big] \le 4\, E\big[ X_n^2 \big].
$$

**最大值平方的期望，被终值平方期望的 4 倍控制。** 常数 4 是通用的（对任意 $n$ 与鞅），不依赖过程结构。这个不等式把「路径极值」与「终值」在 $L^2$ 范数下等价起来——**$L^2$ 鞅的空间中，极大算子有界**。<span class="marginnote">$L^2$ 版本是「鞅的平方可积理论」的支柱：<strong>它让「整条路径」的分析降维成「终点」的分析</strong>。在随机积分（第八篇）里，它正是证明 Itô 积分等距性、鞅表示定理的关键工具。</span>

**证明骨架**：从极大不等式的「穷举版本」出发，用积分表示
$$
E\big[ \max X_k^2 \big] = \int_0^\infty 2\lambda\, P\big( \max X_k \ge \lambda \big)\, d\lambda \le \int_0^\infty 2\lambda \cdot \frac{E[X_n^2 \mathbb{1}[\max X_k \ge \lambda]]}{\lambda^2}\, d\lambda,
$$
再交换积分次序，用 Cauchy-Schwarz 收拾常数，得到 $\le 4E[X_n^2]$。

## 3 公式解析：证明极大不等式

**目标：把 $P(\max_{k\le n} X_k \ge \lambda) \le E[X_n]/\lambda$ 完整走一遍。**

第一步，设停时。$\tau = \min\{k : X_k \ge \lambda\}$（首次摸高），则 $\{\max_{k\le n} X_k \ge \lambda\} = \{\tau \le n\}$。

第二步，用停止过程。$X_{\min(\tau,n)}$ 是下鞅（截断停时保持下鞅性），故
$$
E\big[ X_{\min(\tau,n)} \big] \le E[X_n].
$$
第三步，拆期望。在 $\{\tau \le n\}$ 上 $X_{\min(\tau,n)} = X_\tau \ge \lambda$；在 $\{\tau > n\}$ 上 $X_{\min(\tau,n)} = X_n \ge 0$（非负）。于是
$$
E[X_{\min(\tau,n)}] \ge \lambda\, P(\tau \le n) + 0.
$$
第四步，联立。$\lambda P(\tau \le n) \le E[X_n]$，即 $P(\max X_k \ge \lambda) \le E[X_n]/\lambda$。

**这个推导为什么重要**：它示范了「停时化」的标准动作——**把「最大值事件」化为「停时事件」，再用「停止过程仍下鞅」的期望单调性收口**。Doob 不等式的全部版本都是这个剧本的变体。

## 4 应用：随机算法的最坏偏离

设某随机算法第 $k$ 步的误差 $X_k$ 是鞅（公平更新），我们关心「$n$ 步内误差曾超过 $a$」的概率。由 Doob 极大不等式（$X_n^2$ 是下鞅）：
$$
P\Big( \max_{k \le n} |X_k| \ge a \Big) = P\Big( \max_{k \le n} X_k^2 \ge a^2 \Big) \le \frac{E[X_n^2]}{a^2}.
$$
**只需知道终值方差 $E[X_n^2]$，就能给「过程中最大误差」一个上界。** 若 $E[X_n^2] \le C$（均匀有界），则
$$
P\big( \text{过程中最大误差} \ge a \big) \le \frac{C}{a^2}.
$$
这是「路径极值」的 Chebyshev 型控制——**不比单点误差困难多少**。<span class="marginnote">这条不等式在随机梯度下降、随机网络的误差分析里是标配：<strong>「过程某时刻误差大」的概率，用「终值方差」就能控制</strong>。它把「最坏情况」的分析变成「终值二阶矩」的计算——后者通常容易得多。</span>

## 5 Doob 不等式 vs 收敛定理

| 工具 | 控制的对象 | 条件 |
| --- | --- | --- |
| Doob 极大不等式 | 过程中的最大值 | 下鞅 + 非负 |
| Doob $L^2$ 不等式 | 最大值平方的期望 | 鞅 + 平方可积 |
| 鞅收敛定理 | 过程的极限 | $L^1$ 有界 |

**三件套的配合**：收敛定理说「极限存在」，Doob 不等式说「途中不会太野」——**先管极值、再谈极限**，是鞅理论的标准进攻路线。<span class="marginnote">三件套在随机积分里缺一不可：Itô 积分构造用 $L^2$ Doob 不等式保证「部分和的最大值受控」，再用鞅收敛定理取极限。<strong>「有界性 → 收敛性」是概率论反复使用的论证链</strong>。</span>

## 6 小结

- **Doob 极大不等式**：$P(\max_{k\le n} X_k \ge \lambda) \le E[X_n]/\lambda$（非负下鞅）——最大值的概率被终值管住。
- **Doob $L^2$ 不等式**：$E[\max_{k\le n} X_k^2] \le 4E[X_n^2]$——极大算子 $L^2$ 有界。
- **证明套路**：首次摸高停时 + 停止过程仍下鞅 + 期望单调。
- **应用**：随机算法「过程最大误差」的上界，只需终值方差。
- 与收敛定理配合：先管极值、再谈极限。

在下一节，我们把「鞅差」作为独立的工具引入：**鞅差序列与 Azuma 不等式**——相依随机变量也能有 Hoeffding 型的集中不等式。
