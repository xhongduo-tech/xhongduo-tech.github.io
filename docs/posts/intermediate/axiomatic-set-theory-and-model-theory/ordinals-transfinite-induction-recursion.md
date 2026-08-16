---
title: 序数（超限归纳与递归）
date: 2026-08-07
---

# 序数（超限归纳与递归）

<div class="epigraph">
<p>在数学中，你不理解事物，你只是习惯它们。</p>
<footer>—— 约翰 · 冯 · 诺伊曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 公理集合论与模型论 ｜ Jech, <em>Set Theory</em> 第2章；Kunen 第2章 ｜ 2026-08-07</p>
</div>

## 为什么从「无限步之后」开始

上一篇我们把 ZFC 的十条公理逐条解剖，其中无穷公理给了自然数集 $\omega$。自然数的次序有一个特点：**每一步都只有「上一步」**——要走到 $n$，必须先经过 $n-1$。可数学里处处有「无限步之后还要继续」的需求：第2篇（index 第1篇）里，$V_\omega$ 之后还有 $V_{\omega+1}$；第2篇基数篇里，$\aleph_0, \aleph_1, \aleph_2, \dots$ 走完一整个 $\omega$ 后还有 $\aleph_\omega$。<span class="marginnote">「无限步之后的那一步」在日常语言里没有名字，在集合论里有——它叫 $\omega$。von Neumann 的天才在于：序数不是别的东西，就是「所有比它小的序数的集合」。这句话读起来像废话，却是整座序数大厦的地基。</span>

把「序」从有限推向超限，是康托尔 1883 年完成的工作。他在研究三角级数时被迫追问：「收敛点集经过 $\omega$ 次、$\omega+1$ 次……取导集之后还剩什么？」答案就是超穷序数。**序数 = 良序的类型**：任意两个序数可比，每个序数恰好是「小于它的序数全体」。这一篇我们把这三件事讲透：von Neumann 序数、超限归纳、超限递归——它们是后续一切（基数、可构造宇宙、力迫）的公共脚手架。

## 1 传递集与 von Neumann 序数

**传递集（transitive set）**：集合 $T$ 的元素仍是 $T$ 的元素，即 $x \in y \in T \Rightarrow x \in T$。

**von Neumann 序数**：一个传递集 $\alpha$，且其元素被 $\in$ 关系良序。当且仅当它是「全体小于它的序数之集」。前几个序数是

$$0 = \emptyset, \quad 1 = \{0\}, \quad 2 = \{0,1\}, \quad \dots, \quad \omega = \{0,1,2,\dots\}$$

- **后继序数**：$\alpha+1 = \alpha \cup \{\alpha\}$，如 $1, 2, \omega+1$。
- **极限序数**：既非 0 又非后继，如 $\omega, \omega\cdot2, \omega^2$。

**为什么序数由「所有比它小的序数」构成**：这保证了「$\beta < \alpha$」与「$\beta \in \alpha$」完全同义——次序关系就是属于关系。<span class="marginnote">这一招把「次序」翻译成「元素关系」，于是普通集合论的公理自动适用于序数：传递性给出行文顺序，良基性（正则公理）保证没有无穷下降链，全序性保证任意两个序数可比。次序成了集合的「内建属性」。</span>

**核心定理（序数的基本性质）**：

- 序数的元素仍是序数；
- 任意两个序数 $\alpha, \beta$，恰有 $\alpha < \beta$、$\alpha = \beta$、$\beta < \alpha$ 之一成立（三歧性）；
- 每个非空序数类有 $\in$-极小元——全体序数构成真类 $\mathrm{On}$，它不是集合（布拉里-福蒂悖论）。

## 2 超限归纳原理

**超限归纳原理（transfinite induction）**：设 $P$ 是序数的一个性质。若

$$\forall \alpha \in \mathrm{On} \bigl( \forall \beta < \alpha \, P(\beta) \rightarrow P(\alpha) \bigr)$$

则 $\forall \alpha \in \mathrm{On} \, P(\alpha)$。<span class="marginnote">与自然数归纳相比，超限归纳多了一个「极限步」：不仅要证明 $P(0)$ 与「$P(\alpha) \Rightarrow P(\alpha+1)$」，还要证明「若所有 $\beta<\lambda$ 都满足 $P$，则 $P(\lambda)$ 也满足」——其中 $\lambda$ 跑遍极限序数。处理极限步，是超限理论的日常。</span>

**公式解析：为什么「对一切 $\beta < \alpha$」就够。** 归纳的通常形式是「$P(0)$ 且 $P(\alpha) \Rightarrow P(\alpha+1)$」。但在超限世界里：

- **第零步**：$\alpha = 0$ 时「$\forall \beta < 0$」是空的，前提恒真，故必须单独验证 $P(0)$；
- **后继步**：$\alpha = \beta+1$ 时「$\forall \gamma < \beta+1$」包含 $P(\beta)$，由归纳前提推出 $P(\beta+1)$；
- **极限步**：$\alpha = \lambda$ 时没有「前一步」，必须用「所有更小序数都成立」这个强前提直接证明 $P(\lambda)$。

三步合起来，就是「从 0 出发，沿后继步爬，在极限处跃迁」——这套节奏会重复出现在秩函数、基数幂、Gödel 的 $L_\alpha$ 层级、力迫的 $\alpha$-阶段构造里。

## 3 超限递归原理与序数算术

**超限递归原理（transfinite recursion）**：设 $G$ 是定义在「一切序数序列」上的函数，则存在唯一的函数 $F$ 满足

$$F(\alpha) = G\bigl( F \upharpoonright \alpha \bigr), \quad \alpha \in \mathrm{On}$$

即：**每个新值由「此前所有值」唯一决定**。这是「用已经算好的部分定义下一步」的严格化——编程里的动态规划、递归下降，数学里的阶乘、Fibonacci，都是它的特例。<span class="marginnote">为什么「此前所有值」要写成 $F \upharpoonright \alpha$（把 $F$ 限制在前 $\alpha$ 个序数上）而不是「$F(\alpha-1)$」？因为极限序数没有前一项。把整个历史打包递出去，正是为了照顾极限步。</span>

序数算术由超限递归给出。加法：

$$\alpha + 0 = \alpha, \quad \alpha + (\beta+1) = (\alpha+\beta)+1, \quad \alpha + \lambda = \sup_{\beta < \lambda} (\alpha + \beta) \;(\lambda \text{ 极限})$$

乘法、指数同理。三个反直觉的实例值得记住：

- **$1 + \omega = \omega \ne \omega + 1$**：加法不交换；
- **$2 \cdot \omega = \omega \ne \omega \cdot 2$**：乘法不交换；
- **$2^{\aleph_0} = \mathfrak{c}$**：指数一路爆炸，直接通向连续统假设。

**为什么次序不能颠倒**：$1+\omega$ 是把 $1$ 排在 $\omega$ 个元素之前，仍是 $\omega$ 个元素；$\omega+1$ 是在 $\omega$ 之后再添一个，比 $\omega$ 多。**「有限个元素在无限之前」不改变大小，「在无限之后」就多出一个**——序数的次序不是交换的算术，而是「排队」的算术。

## 4 核心对比表：自然数归纳与超限归纳

| 对比项 | 自然数归纳 | 超限归纳 |
| --- | --- | --- |
| 定义域 | $\omega$ | 全体序数 $\mathrm{On}$ |
| 归纳前提 | $P(0)$ 且 $P(n)\Rightarrow P(n+1)$ | $\forall \beta<\alpha\, P(\beta) \Rightarrow P(\alpha)$ |
| 特殊步骤 | 无 | **极限步**（$\lambda$ 处直接验证） |
| 递归依据 | 前一项 $n-1$ | **整个历史** $F\upharpoonright\alpha$ |
| 实例 | 阶乘、求和公式 | 秩函数、$\aleph_\alpha$、$L_\alpha$、力迫层级 |

**辨析｜易错点：** 初学者常以为「超限递归 = 普通递归 + 对极限序数取并」。取并只是常见做法之一，原理上递归定理允许**任何**从历史到新值的定义——极限步要验证的是「定义在极限处也唯一」，而取并往往是验证起来最省事的写法。

## 5 动手推导：从递归定义算出 ω·2 与 ω²

用超限递归的实际定义算一遍，把「极限步取 sup」落到实处。

**$\omega + \omega$（即 $\omega \cdot 2$）**：加法定义给 $\omega + \lambda = \sup_{\beta < \lambda} (\omega + \beta)$。于是

$$\omega + \omega = \sup_{n < \omega} (\omega + n) = \sup\{ \omega, \omega+1, \omega+2, \dots \}$$

结果比 $\omega$ 大（多出一整列 $\omega$），但仍是可数的：$|\omega + \omega| = \aleph_0$。

**$2 \cdot \omega$ 与 $\omega \cdot 2$ 的差别**：乘法定义 $\alpha \cdot (\beta+1) = \alpha \cdot \beta + \alpha$，$\alpha \cdot \lambda = \sup_{\beta<\lambda} \alpha \cdot \beta$。

- $2 \cdot \omega = \sup_n (2 \cdot n) = \sup\{0, 2, 4, \dots\} = \omega$：两个一组排无限轮，还是 $\omega$ 个位置。
- $\omega \cdot 2 = \omega \cdot 1 + \omega = \omega + \omega$：先排完整一列 $\omega$，再排第二列——多一格。

**为什么「无穷之前」与「无穷之后」判若云泥**：$1 + \omega = \omega$（把 1 塞进队首，总量不变），$\omega + 1 > \omega$（队尾再添一个，序型变大）。序数的次序不是交换律的家，而是「队列」的家——**有限段在前不影响大小，在后就多一格**。

**一个记号陷阱**：序数指数 $2^\omega = \sup_n 2^n = \omega$，而基数幂 $2^{\aleph_0} = \mathfrak{c}$。同一套记号、两种语义——看到指数运算务必先问「这是序数还是基数」。

## 6 术语速查表：序数语言

| 术语 | 含义 |
| --- | --- |
| 传递集 | $x \in y \in T \Rightarrow x \in T$ |
| 序数 | 传递且被 $\in$ 良序的集合 |
| 后继序数 | $\alpha+1 = \alpha \cup \{\alpha\}$ |
| 极限序数 | 非 0 非后继，如 $\omega$ |
| $\mathrm{On}$ | 全体序数的真类 |
| 超限归纳 | 第 0 步 + 后继步 + 极限步 |
| 超限递归 | $F(\alpha) = G(F\upharpoonright\alpha)$ |
| $\omega$ | 最小极限序数 = 自然数集 |
| 序数加法 | $\alpha+\lambda = \sup_{\beta<\lambda}(\alpha+\beta)$ |
| 序型 | 良序集的同构类代表序数 |

## 7 小结

- **von Neumann 序数** = 传递且被 $\in$ 良序的集合，$\beta < \alpha$ 即 $\beta \in \alpha$。
- 序数分**后继序数**（$\alpha+1 = \alpha \cup \{\alpha\}$）与**极限序数**（$\omega$ 为首例）。
- **超限归纳**比自然数归纳多一个**极限步**；**超限递归**用「整个历史」定义新值。
- 序数算术**不交换**：$1+\omega=\omega$，$\omega+1>\omega$；「无限之后」才真正增大。
- 全体序数 $\mathrm{On}$ 是真类而非集合。

在下一节，我们将回答「集合一样大吗」：**基数与基数算术**——怎么给无穷集合配数，$\aleph_0, \aleph_1, \aleph_\omega$ 如何像梯子一样无限延伸，以及连续统假设 $2^{\aleph_0}=\aleph_1$ 到底难在哪里。
