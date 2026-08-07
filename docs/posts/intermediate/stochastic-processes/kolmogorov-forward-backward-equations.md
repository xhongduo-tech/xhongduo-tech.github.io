---
title: Kolmogorov 向后方程与向前方程
date: 2026-08-07
---

# Kolmogorov 向后方程与向前方程

<div class="epigraph">
<p>微分方程把「遥远的未来」切成「眼前的一步」——向前与向后，只是切的方向不同。</p>
<footer>—— 安德雷 · 柯尔莫哥洛夫（Andrey Kolmogorov）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§5.2 ｜ 2026-08-07</p>
</div>

## 给 P(t) 写微分方程

CTMC 的转移概率 $P(t) = e^{Qt}$ 是「闭式解」，但很多问题不需要闭式，只需要「$P(t)$ 如何随时间演化」。而演化由微分方程描述——**Kolmogorov 方程**。它把 $Q$ 与 $P(t)$ 连接起来，让「算转移概率」变成「解微分方程组」——这是工程上处理 CTMC 最常用的途径。

Kolmogorov 方程有两个版本，名字对应「条件在哪一端」：

- **向后方程（backward equation）**：对**起点** $i$ 取无穷小——「从 $i$ 先走一小步，再走剩下的」；
- **向前方程（forward equation）**：对**终点** $j$ 取无穷小——「先走大部分，最后一步落入 $j$」。

两者在**有限状态**时等价（都给出 $P' = QP$ 与 $P' = PQ$），但推导视角不同，适用场景也不同。<span class="marginnote">向后/向前的直觉：<strong>向后方程把『条件于起点』往前推（从起点先迈一步），向前方程把『条件于终点』往后推（终点前最后一步）</strong>。排队论里向后方程常用于分析「吸收时间」，向前方程常用于算平稳分布。</span>

本节目标：推导并理解两个方程、弄清它们的适用条件、并用泊松过程与两态例演示。

## 1 向后方程

**向后方程（Kolmogorov backward equation）**：对每个 $i, j$ 与 $t \ge 0$，
$$
p'_{ij}(t) = \sum_{k \in S} q_{ik}\, p_{kj}(t), \qquad p_{ij}(0) = \delta_{ij}.
$$
**矩阵形式**：$P'(t) = Q\, P(t)$。

**推导（直觉）**：把「从 $i$ 走 $t$ 到 $j$」拆成「先在极短 $h$ 内从 $i$ 跳到某处 $k$，再走 $t$ 从 $k$ 到 $j$」：
$$
p_{ij}(t + h) = \sum_k p_{ik}(h)\, p_{kj}(t) \approx (1 - q_i h)\, p_{ij}(t) + \sum_{k \ne i} q_{ik} h\, p_{kj}(t).
$$
移项、除以 $h$、令 $h \to 0$：
$$
p'_{ij}(t) = -q_i\, p_{ij}(t) + \sum_{k \ne i} q_{ik} p_{kj}(t) = \sum_k q_{ik} p_{kj}(t).
$$
**名字由来**：方程对「起点 $i$ 的那一步」求导——条件固定在起点，向未来推进。

## 2 向前方程

**向前方程（Kolmogorov forward equation）**：对每个 $i, j$ 与 $t \ge 0$，
$$
p'_{ij}(t) = \sum_{k \in S} p_{ik}(t)\, q_{kj}, \qquad p_{ij}(0) = \delta_{ij}.
$$
**矩阵形式**：$P'(t) = P(t)\, Q$。

**推导（直觉）**：把「从 $i$ 走 $t$ 到 $j$」拆成「从 $i$ 先走 $t$ 到某处 $k$，最后极短 $h$ 内 $k \to j$」：
$$
p_{ij}(t + h) = \sum_k p_{ik}(t)\, p_{kj}(h) \approx p_{ij}(t)(1 - q_j h) + \sum_{k \ne j} p_{ik}(t)\, q_{kj} h.
$$
同样取极限：
$$
p'_{ij}(t) = -q_j\, p_{ij}(t) + \sum_{k \ne j} p_{ik}(t) q_{kj} = \sum_k p_{ik}(t) q_{kj}.
$$
**名字由来**：方程对「终点 $j$ 的最后一步」求导——条件固定在终点，向过去回推。<span class="marginnote">向前方程的分量式还可以写成「$\pi$ 的演化」：设 $\pi_j(t) = P(X(t) = j)$，则 $\pi'(t) = \pi(t) Q$——<strong>分布随时间演化的方程</strong>。平稳分布 $\pi Q = 0$ 正是它的不动点，这也是为什么向前方程与平稳性天然亲近。</span>

## 3 公式解析：从无穷小展开到向前方程

**目标：把向前方程 $p'_{ij}(t) = \sum_k p_{ik}(t) q_{kj}$ 用 C-K 方程 + 无穷小展开完整推一遍。**

第一步，C-K 方程把「$t + h$」拆开。对终点「追加 $h$」：
$$
p_{ij}(t + h) = \sum_{k} p_{ik}(t)\, p_{kj}(h).
$$
第二步，展开 $p_{kj}(h)$。$k = j$ 时 $p_{jj}(h) = 1 - q_j h + o(h)$；$k \ne j$ 时 $p_{kj}(h) = q_{kj} h + o(h)$：
$$
p_{ij}(t+h) = p_{ij}(t)\big(1 - q_j h\big) + \sum_{k \ne j} p_{ik}(t)\, q_{kj} h + o(h).
$$
第三步，移项取极限。
$$
\frac{p_{ij}(t+h) - p_{ij}(t)}{h} = -q_j p_{ij}(t) + \sum_{k\ne j} p_{ik}(t) q_{kj} + o(1) \;\to\; \sum_k p_{ik}(t) q_{kj}.
$$
第四步，写成 $P'(t) = P(t)Q$。把终点指标展开：$q_{jj} = -q_j$，故 $\sum_k p_{ik} q_{kj}$ 正好是「$P$ 右乘 $Q$」的第 $(i,j)$ 元。

**这个推导为什么重要**：它演示了「无穷小展开 + C-K 方程」这套标准动作——**任何 CTMC 的微分方程都从这一步起步**。向后方程只需把「拆 $h$」换到起点端，完全相同的手法。

## 4 例子：泊松过程满足向前方程

泊松过程 $N(t) \sim \mathrm{Poisson}(\lambda t)$ 是 CTMC 的特例：状态 $S = \mathbb{Z}_{\ge 0}$，速率 $q_{i, i+1} = \lambda$（其余为 0），即 $Q$ 是「上双对角」：
$$
q_{i,i+1} = \lambda, \qquad q_{ii} = -\lambda.
$$
向前方程给 $p_{ij}(t) = P(N(t) = j \mid N(0) = i) = \frac{(\lambda t)^{j-i}}{(j-i)!} e^{-\lambda t}$（$j \ge i$）。验证：对 $t$ 求导满足 $p'_{ij} = \lambda p_{i,j-1} - \lambda p_{ij}$。**泊松过程的向前方程就是泊松分布递推 $p'_j = \lambda p_{j-1} - \lambda p_j$ 的矩阵化。**<span class="marginnote">这个例子把第二篇与第五篇打通：<strong>泊松过程 = 只有「+1 跳跃」的 CTMC</strong>。反过来看，CTMC 是「多方向泊松流」的合成——每个 $q_{ij}$ 都是一条独立泊松流。</span>

## 5 两个方程的关系与适用

**有限状态时**：向后与向前都成立且解相同（$e^{Qt}$），可任选。
**无穷状态时**：需要条件——向后方程在「离开速率有界」时成立；向前方程在「无爆炸」等条件下成立。直觉上，向后方程对边界条件更宽容，向前方程更容易出现「质量泄漏」到无穷的问题。<span class="marginnote">工程建议：<strong>算「从某状态出发的首次事件」（吸收、破产）用向后方程；算「某时刻的状态分布、平稳分布」用向前方程</strong>。这是排队论、可靠性、生物模型里的通行分工。</span>

## 6 小结

- **向后方程** $P'(t) = QP(t)$：条件于起点，$p'_{ij} = \sum_k q_{ik} p_{kj}$。
- **向前方程** $P'(t) = P(t)Q$：条件于终点，$p'_{ij} = \sum_k p_{ik} q_{kj}$。
- 推导套路：C-K 方程 + 无穷小展开 + 除以 $h$ 取极限——两方程只有「拆哪端」的区别。
- 泊松过程是「只有 +1 跳跃」的 CTMC，其向前方程退化为泊松分布递推。
- 有限状态两方程等价；无穷状态向后更稳、向前需防爆炸。

**例（两态 CTMC 的向前方程）**：$Q = \begin{pmatrix} -\lambda & \lambda \\ \mu & -\mu \end{pmatrix}$。向前方程 $\pi'(t) = \pi(t)Q$ 写出 $\pi_0' = -\lambda\pi_0 + \mu\pi_1$、$\pi_1' = \lambda\pi_0 - \mu\pi_1$。解之得 $\pi_0(t) = \frac{\mu}{\lambda+\mu} + \big(\pi_0(0) - \frac{\mu}{\lambda+\mu}\big) e^{-(\lambda+\mu)t}$——**指数逼近平稳分布，收敛速率 $\lambda + \mu$**。这个显式解演示了向前方程的完整玩法：写方程、解指数型、看稳态。

这个结论也说明两态链的混合速度由 $\lambda + \mu$ 决定——速率之和越大，平稳越快。把「收敛速率 = 速率之和」这条直觉记下来，后面连续时间链的收敛分析都以此为准绳。

在下一节，我们问 CTMC 的长期问题：它有没有平稳分布？怎么求？这就是**平稳分布与长期行为：连续时间情形**。
