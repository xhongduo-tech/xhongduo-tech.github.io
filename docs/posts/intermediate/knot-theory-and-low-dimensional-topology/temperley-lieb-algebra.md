---
title: Temperley-Lieb 代数
date: 2026-08-07
---

# Temperley-Lieb 代数

<div class="epigraph">
<p>图是代数的心脏——Temperley-Lieb 代数把乘法画成了不相交的配对。</p>
<footer>—— 本文作者按</footer>
</div>

<div class="article-byline">
<p>第二级 · 纽结理论与低维拓扑 ｜ Lickorish《An Introduction to Knot Theory》第13章 ｜ 2026-08-07</p>
</div>

## 为什么从「Temperley-Lieb」开始

上一节我们说「$(n, n)$-tangle 的 skein 模往往有限维」。这个有限维代数有一个名字：**Temperley-Lieb 代数** $TL_n$。它最初由统计物理学家 Temperley 与 Lieb（1971）在研究**六顶点模型**的转移矩阵时发现，后来被琼斯认出正是 Jones 多项式的代数根源。

Temperley-Lieb 代数是「图 + 乘法」的最简代数：基是**不相交配对**（non-crossing pairings）的图，乘法是「堆叠 + 消圈」。它同时是拓扑的（编码缠绕）、代数的（有生成元与关系）、组合的（Catalan 数维数）、物理的（统计力学转移矩阵）——一个四合一的对象。<span class="marginnote">Temperley-Lieb 代数出现在远超结理论的地方：量子 Heisenberg 模型、自旋链、张量网络（TNR、MPS）、量子计算（匹配门电路）。它是「低维拓扑与凝聚态物理的公共语言」——第3篇之四的量子不变量正是它的表示论化身。</span>

## 1 图基：不相交配对

**Temperley-Lieb 代数** $TL_n(\delta)$ 的基由「$2n$ 个点的**不相交配对**」组成：把 $n$ 个点排成一行（下方），$n$ 个点排成另一行（上方），用不相交的弧把上下两排的点两两配对，弧不出格、互不相交。每个这样的配对图是一个基元素；$\delta$ 是「圈」的标量因子。

**乘法**：把两个图**上下堆叠**——$\alpha$ 的底排与 $\beta$ 的顶排对齐连接，中间的弧合并，出现闭圈就替换为因子 $\delta$。得到的图仍是不相交配对。

**例子**（$TL_3$）：基有 $5$ 个元素（Catalan 数 $C_3 = 5$）：恒等图、三个「帽子/杯」图、一个「连通」图。

维数公式：$\dim TL_n = C_n$（第 $n$ 个 **Catalan 数** $C_n = \frac{1}{n+1}\binom{2n}{n}$）。
$TL_1$ 维数 1，$TL_2$ 维数 2，$TL_3$ 维数 5，$TL_4$ 维数 14，……<span class="marginnote">Catalan 数的出现绝非巧合：不相交配对正是 Catalan 组合的经典计数对象（括号配对、凸多边形三角剖分、二叉树的兄弟）。Temperley-Lieb 代数把 Catalan 数嵌入成一个代数的维数，让「组合计数」获得了「代数结构」——这是组合数学与代数互惠的典范。</span>

## 2 生成元与关系

Temperley-Lieb 代数也可用代数生成元与关系定义。设 $e_i$（$i = 1, \ldots, n-1$）为「把第 $i$ 与第 $i+1$ 根弦连成帽」的图，则 $TL_n(\delta)$ 由 $e_1, \ldots, e_{n-1}$ 生成，满足：

$$
e_i^2 = \delta\, e_i,
$$

$$
e_i e_{i \pm 1} e_i = e_i, \qquad
e_i e_j = e_j e_i \quad (|i - j| \ge 2).
$$

- **第一条**：$e_i^2 = \delta e_i$——叠两次出现一个圈，圈 = $\delta$。
- **第二条**：相邻帽子互相「吸收」——叠三层的图可化简为单层。
- **第三条**：不相邻的帽子互不干扰、可交换。

**与辫群的关系**：Temperley-Lieb 与辫群 $B_n$ 是「表亲」——$B_n$ 的关系是 $\sigma_i\sigma_{i+1}\sigma_i = \sigma_{i+1}\sigma_i\sigma_{i+1}$，$TL_n$ 的关系是 $e_i e_{i\pm 1} e_i = e_i$。后者是前者的「幂等版本」（把 $\sigma_i + \sigma_i^{-1}$ 归约后剩下的对象）。

## 3 从 Temperley-Lieb 到 Jones 多项式

Temperley-Lieb 代数最辉煌的应用是给出 Jones 多项式的**代数定义**。关键步骤：

1. 取 $\delta = -A^2 - A^{-2}$（Kauffman 括号的圈因子）。
2. 在 $TL_n(\delta)$ 上定义 **Markov 迹（Markov trace）** $\operatorname{tr}_n$：一个线性泛函，把图映到「闭包图的括号值」的合适归一化（第3篇之三详述）。
3. 对辫子 $\beta \in B_n$，通过辫群到 $TL_n$ 的**表示**（把 $\sigma_i$ 映到 $A e_i - A^{-1}$ 之类）把 $\beta$ 送进 $TL_n$，再取迹。

于是 Jones 多项式 = 辫子在 Temperley-Lieb 代数中的**迹**：

$$
V_{\widehat{\beta}}(t) = \text{适当的迹求值}(\beta \in B_n \to TL_n \to \mathbb{Z}[A^{\pm 1}]).
$$

这正是琼斯 1984 年的原始路线——他在研究算子代数的有限维子因子时发现 Temperley-Lieb 代数的迹能产生结不变量。<span class="marginnote">琼斯的原始证明不画任何图：他算的是「$TL_n$ 上满足 Markov 性质的迹」的唯一性。图只是后来的「可视化」。这个「代数 → 迹 → 不变量」的模板后来被量子群推广：用李代数的量子化表示替代 $TL_n$，就得到一整族量子不变量（第3篇之四）。</span>

## 4 公式解析：$TL_n$ 上的迹怎么取

Markov 迹 $\operatorname{tr}_n : TL_n \to \mathbb{Z}[A^{\pm 1}]$ 是满足下列性质的唯一线性泛函（对合适的标度）：

$$
\operatorname{tr}_n(1) = 1, \qquad
\operatorname{tr}_n(x e_n) = \frac{\operatorname{tr}_n(x)}{\delta}, \quad
\operatorname{tr}_n(e_n x) = \frac{\operatorname{tr}_n(x)}{\delta},
$$

其中 $x \in TL_n$，$e_n$ 是「最右帽」。

- **第一步，归一化**：恒等图的迹为 1——「没有缠绕」的闭包是平凡结，括号值归一化为 1。
- **第二步，Markov 性质**：$x e_n$ 的迹与 $x$ 的迹只差因子 $1/\delta$——在 $x$ 右边加一个帽，闭包后多一个圈，圈因子 $\delta$ 给出 $1/\delta$。
- **第三步，唯一性**：这两条性质（加上线性与乘积相容）足以唯一确定 $\operatorname{tr}_n$。这个「存在唯一」是 Jones 多项式良定义性的代数核心——比用 Reidemeister 移动验证不变性更本质。

**易错点｜$\delta$ 何时为 0**：$TL_n$ 的表示结构随 $\delta$ 取值剧烈变化。当 $\delta = 0$（$A^4 = -1$，即 $t = -1$ 之类），代数退化（非半单）；当 $\delta$ 为一般参数，$TL_n$ 半单。这个「临界值」现象对应量子群表示论里的「$q$ 为单位根」——它决定了不变量何时「奇异」。理解 $\delta$ 何时退化，是读量子不变量文献的第一道坎。

## 5 Temperley-Lieb 代数的结构

- **维数**：$\dim TL_n = C_n$（Catalan 数），图基给出组合实现。
- **半单性与分解**：对一般 $\delta$，$TL_n$ 分解为矩阵代数的直和；不可约表示的标号对应「$n$ 的整数划分」。
- **本原幂等**：$TL_n$ 含本原幂等元（Jones-Wenzl 幂等），它们是「不可约投影」的图实现——量子不变量构造中「沿表示投影」的必需品。
- **与 Kauffman 括号 skein 模**：$TL_n \cong$「$(n, n)$-tangle 的 KBSM」——第3篇之一的商空间就是这个代数。<span class="marginnote">Jones-Wenzl 幂等是 Temperley-Lieb 代数最深的结构之一：用图语言写的「全对称投影」。它在张量网络、量子计算的「可计算性刻画」中举足轻重——最近的研究（Leverrier 等人的「匹配门」工作）用它统一了匹配门电路，是 Temperley-Lieb 代数在新物理中的又一次出场。</span>

### Jones-Wenzl 幂等：不可约投影的图实现

$TL_n$ 里最深刻的结构是 **Jones-Wenzl 幂等（Jones-Wenzl idempotent）**——它是「全对称投影」的图实现。记作 $f_n$，它满足：

$$
f_n^2 = f_n, \qquad f_n e_i = e_i f_n = 0 \quad (\text{对 } i \le n-1).
$$

- **第一条**：$f_n$ 是幂等（投影）——施加两次等于施加一次。
- **第二条**：$f_n$ 被所有生成元「杀死」——它与任何帽正交，是「不可约」的纯投影。

**递归构造**：$f_n$ 可由 $f_{n-1}$ 递归造出（Wenzl 公式），每个 $f_n$ 在 $TL_n$ 的不可约分解里对应「最高权表示」。$f_n$ 在量子不变量里的作用：给 tangle 的「颜色」做投影——沿表示投影后再取迹，正是着色 Jones 多项式的构造。

**为什么重要**：$f_n$ 是 $TL_n$ 不可约理论的「原子」——把 $TL_n$ 分解为矩阵代数的直和时，每个块由一个 $f_k$ 的「主幂等」支撑。理解 $f_n$ = 理解 $TL_n$ 的表示论。

### TL 代数在物理中的第二次出场

Temperley-Lieb 代数从统计力学出发，却在现代物理里不断重现：

- **自旋链**：一维量子自旋链的转移矩阵用 $TL_n$ 的生成元写出——「六顶点模型 → TL → 可积自旋链」是同一条线。
- **张量网络**：MPS（矩阵乘积态）、TNR 里出现的「腿的收缩」正是 $TL_n$ 的图乘法——图基的「配对」就是张量缩并。
- **拓扑量子计算**：$TL_n$ 在 $q$ 为单位根处的表示给出 anyon 的编织——「TL 表示 = 拓扑量子比特」的代数基础（第9级《拓扑量子计算》）。

**一个代数，三重物理**：从统计力学（起源）到凝聚态（自旋链、张量网络）到量子计算（anyon）——$TL_n$ 是「低维物理的公共代数语言」这句话的最好注脚。

## 6 小结

- **Temperley-Lieb 代数** $TL_n(\delta)$：基 = $2n$ 点的不相交配对，乘法 = 堆叠 + 消圈。
- 维数 = **Catalan 数** $C_n$；生成元 $e_i$ 满足 $e_i^2 = \delta e_i$、$e_i e_{i\pm1} e_i = e_i$。
- **Jones 多项式 = $TL_n$ 上 Markov 迹对辫子的求值**——代数定义优于图定义。
- 取 $\delta = -A^2 - A^{-2}$ 时，$TL_n$