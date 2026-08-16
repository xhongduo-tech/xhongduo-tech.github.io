---
title: 基数（基数运算、连续统假设）
date: 2026-08-07
---

# 基数（基数运算、连续统假设）

<div class="epigraph">
<p>我看见了，但我不相信。</p>
<footer>—— 格奥尔格 · 康托尔（Georg Cantor），1877 年致戴德金的信</footer>
</div>

<div class="article-byline">
<p>第二级 · 公理集合论与模型论 ｜ Jech, <em>Set Theory</em> 第3章；Kunen 第3章 ｜ 2026-08-07</p>
</div>

## 为什么从「哪个无穷更大」开始

序数篇给出了「排队的无穷」：$\omega, \omega+1, \omega\cdot2, \dots$。但「数数」有两种：排队（顺序）与配数（个数）。两个无穷集合一样大吗？答案由**双射**决定：存在一一对应则一样大。1877 年，康托尔证明了线段 $[0,1]$ 与平面 $\mathbb{R}^2$ **等势**（一样大），他在给戴德金的信里写下了这句「我看见了，但我不相信」——直观上平面比线段「多得多」，双射却说它们一样大。<span class="marginnote">这句话点破了基数的本质：<strong>数量只看双射，不看几何维度</strong>。$[0,1]$ 与 $[0,1]\times[0,1]$ 等势，这与「维度」毫无关系——直到拓扑学（本课程后续层级）才重新抓住「维度」这个概念。</span>

这一篇要回答三个问题：怎么给无穷集合配数（基数是什么）；这些数怎么算（基数算术）；最大的那个未定问题——连续统假设 $2^{\aleph_0}=\aleph_1$ 到底成不成立。最后一问将把我们一路引向可构造宇宙与力迫法（第3篇）。

## 1 基数的定义：用秩给等价类挑代表

**等势**：$|X| = |Y|$ 当且仅当存在双射 $f: X \to Y$。等势是一个等价关系，**基数**就是「等势类」的大小指标。麻烦在于：一个等价类不是一个集合（它是真类），不能直接拿来做对象。解决办法是用序数当「标尺」：

**基数定义**：$\mathrm{card}(X) = $ 与 $X$ 等势的**最小序数**。<span class="marginnote">这个最小序数存在，靠的是良序定理（选择公理）：任意集合可被良序，于是必然落进某个序数。没有选择公理时这套定义要打补丁（见第2篇选择公理专文），所以「基数 = 最小等势序数」在 ZFC 中才干净。</span>

**有限基数**就是自然数；**无穷基数**是那些不落在 $\omega$ 之内的序数，记作 $\aleph_\alpha$（阿列夫）：

$$\aleph_0 = \omega, \quad \aleph_1 = \omega_1, \quad \aleph_2 = \omega_2, \quad \dots, \quad \aleph_{\alpha+1}, \quad \aleph_\lambda = \sup_{\alpha<\lambda} \aleph_\alpha$$

$\aleph_0$ 是可数基数，$\aleph_1$ 是**最小的不可数基数**。<span class="marginnote">「最小的不可数基数」这个名字里有深意：它说的是「比 $\mathbb{N}$ 大的第一个阿列夫」，而<strong>未必</strong>是「实数集的大小」。实数集 $\mathbb{R}$ 到底排在 $\aleph_1$ 还是更高，正是连续统假设所问——注意这跟「$\aleph_1$ 存在」是两回事。</span>

**核心定理（康托尔）**：$|X| < |\mathcal{P}(X)|$，对一切集合 $X$。证明用对角线法：任给 $f: X \to \mathcal{P}(X)$，构造 $D = \{x : x \notin f(x)\}$，$D$ 不在 $f$ 的像里。由幂集公理，$2^{\aleph_0} > \aleph_0$，于是实数集 $|\mathbb{R}| = 2^{\aleph_0}$ 不可数——对角线法还是哥德尔不完全性定理（本专题第5篇）的骨架。

## 2 公式解析：无穷基数的加法与乘法

有限数的算术，在无穷基数那里几乎全部「坍缩」：

$$\aleph_0 + \aleph_0 = \aleph_0, \qquad \aleph_0 \cdot \aleph_0 = \aleph_0$$

更一般地，对无穷基数 $\kappa, \lambda$，

$$\kappa + \lambda = \kappa \cdot \lambda = \max\{\kappa, \lambda\}$$

三步拆解这条「吸星大法」：

- **第一步**：$\aleph_0 \cdot \aleph_0 = \aleph_0$。把 $\omega \times \omega$ 沿「对角线之字形」编号：$(0,0), (0,1),(1,0), (0,2),(1,1),(2,0),\dots$。每条对角线有限，全部对角线可数——这给出双射，是康托尔 1874 年的第一个结果。
- **第二步**：任取 $\kappa, \lambda$ 中较大者，比如 $\kappa \ge \lambda$，则 $\kappa \le \kappa+\lambda \le \kappa \cdot 2$（直和拆两半）。而 $\kappa \cdot 2 \le \kappa \cdot \kappa = \kappa$（第一步推广到无穷序数），由 Cantor–Bernstein 定理夹逼得 $\kappa+\lambda = \kappa$。
- **第三步**：乘法同理由 $\kappa \cdot \lambda \le \max\{\kappa,\lambda\}^2$ 与「$\kappa^2 = \kappa$」（对所有无穷基数成立，用超限递归构造双射）给出。

结论：**无穷基数对加法乘法是「惰性」的**——只有取幂才可能真正变大，而取幂正是全部难点所在。

## 3 指数爆炸与连续统假设

基数幂 $\kappa^\lambda$ = 从 $\lambda$ 到 $\kappa$ 的全体函数之基数。当指数为 $\aleph_0$ 时：

$$2^{\aleph_0} = \bigl|\mathcal{P}(\mathbb{N})\bigr| = |\mathbb{R}| = \mathfrak{c}$$

**连续统假设（Continuum Hypothesis, CH）**：$2^{\aleph_0} = \aleph_1$——实数集恰好是「最小的不可数基数」。广义连续统假设（GCH）：$\kappa < 2^\kappa = \aleph_{\alpha+1}$ 当 $\kappa = \aleph_\alpha$，对一切 $\kappa$。<span class="marginnote">1900 年希尔伯特把 CH 列为其二十三个问题的<strong>第一问</strong>；1938 年哥德尔证明 CH 与 GCH 在 ZFC 中<strong>不可否证</strong>（可构造宇宙 $L$ 满足它们）；1963 年科恩用力迫证明它们<strong>不可证明</strong>。一前一后夹出「独立」，成了本专题第3篇的全部动机。</span>

**公式解析：为什么 $2^{\aleph_0} \ge \aleph_1$ 是平凡真、$=$ 却难上天。** 康托尔定理只给出 $2^{\aleph_0} > \aleph_0$。把实数集良序化（选择公理），它的序型是一个不可数序数，于是 $|\mathbb{R}| \ge \aleph_1$——这是硬不等式，不依赖 CH。CH 断言的是：这个不等式取到**最小**可能。反过来，$\aleph_1 = \omega_1$ 是「全体可数序数之集」，要证 $|\mathbb{R}| \le \aleph_1$ 需要把实数逐个「翻译」成可数序数——翻译不存在，因为 $L$ 与力迫扩张里 $2^{\aleph_0}$ 可以是任意高的 $\aleph_\alpha$。**CH 的难，难在它同时要「序数侧」和「实数侧」的两套计数吻合**。

**几个里程碑（事实，记住结论即可）**：

- **Easton 定理（1970）**：在 ZFC 中，正则基数上的指数函数几乎可以任意地大——$2^{\aleph_0}$ 可以等于 $\aleph_1, \aleph_{17}, \aleph_{\omega+1}$ 等等，只要满足 König 不等式 $\mathrm{cf}(2^{\aleph_0}) > \aleph_0$。
- **Silver 定理（1974）**：对奇异基数，指数行为受「低层」控制——$\mathrm{cf}(\aleph_\delta)=\aleph_0$ 时，若所有 $\aleph_\alpha (\alpha<\delta)$ 的 $2^{\aleph_\alpha} = \aleph_{\alpha+1}$，则 $2^{\aleph_{\aleph_\delta}} = \aleph_{\aleph_\delta+1}$。GCH 不能「在最底层就崩」。

## 4 核心对比表：序数算术 vs 基数算术

| 对比项 | 序数算术 | 基数算术 |
| --- | --- | --- |
| 度量什么 | 顺序（排队第几位） | 大小（有几个） |
| 交换律 | 不交换 | 加乘都交换 |
| $\omega$ 的加/乘 | $1+\omega=\omega$，$\omega+1>\omega$ | $\aleph_0+1=\aleph_0$ |
| 决定大小的运算 | 后继与极限 | **取幂** |
| 例 | $\omega^\omega$ 还是可数个 | $2^{\aleph_0}$ 不可数，大小未定 |

**辨析｜易错点：** 「$\aleph_0 < \aleph_1 \le \mathfrak{c}$」与「$\mathfrak{c} = \aleph_1$」是两句话。前者是定理（由 $\mathfrak{c} > \aleph_0$ 与最小性得 $\aleph_1 \le \mathfrak{c}$），后者是 CH——独立于 ZFC。把「不可数」当成「就是 $\aleph_1$」是把 CH 偷渡进了定义。

## 5 动手推导：可数个可数集的并为什么仍可数

用「之字形编号」证明 $\aleph_0 \cdot \aleph_0 = \aleph_0$ 之后，立刻推出一个影响深远的结论。

**命题**：可数个可数集的并是可数集。设每个 $A_n$（$n<\omega$）可数，$X = \bigcup_n A_n$。把每个 $A_n$ 的成员横排成行，$X$ 就是 $\omega \times \omega$ 的子集。对 $\omega \times \omega$ 沿对角线编号（康托尔 1874）：

- 第 0 号：$(0,0)$；
- 第 1、2 号：$(0,1),(1,0)$；
- 第 3、4、5 号：$(0,2),(1,1),(2,0)$；……

第 $k$ 条对角线上「坐标之和」恒为 $k$，共 $k+1$ 个元素，全部对角线穷尽 $\omega \times \omega$。于是存在双射 $F: \omega \times \omega \to \omega$，$|X| \le \aleph_0$；又 $X$ 含某个可数集，$|X| \ge \aleph_0$。夹逼得 $|X| = \aleph_0$。

**两个直接推论**：

- **$\mathbb{Q}$ 可数**：把每个有理数写成既约分数 $\frac{p}{q}$，映射到 $(p,q)$ 的之字形编号——「处处稠密」的有理数居然是「可数个」。
- **代数数可数**：每个代数数是某整系数多项式的根，而多项式可数——于是「绝大多数实数都是超越数」（由 $|\mathbb{R}|$ 不可数）。

**辨析｜易错点：** 「可数个可数集的并可数」是定理，但依赖**可数选择公理**（每个 $A_n$ 的枚举要一起选）。另一个坑：它说的是「并」而不是「积」——可数个可数集的**笛卡尔积** $\omega^\omega$ 不可数，正是 $2^{\aleph_0}$ 的另一个面孔。

## 6 术语速查表：基数论核心词

| 术语 | 含义 |
| --- | --- |
| 等势 | 存在双射的两集合同大小 |
| 基数 | 等势类的最小序数代表 |
| $\aleph_0$ | 可数无穷基数 $=|\mathbb{N}|$ |
| $\aleph_\alpha$ | 第 $\alpha$ 个无穷基数 |
| $\mathfrak{c}$ | 连续统 $= 2^{\aleph_0} = |\mathbb{R}|$ |
| 正则基数 | $\mathrm{cf}(\kappa) = \kappa$ |
| 奇异基数 | $\mathrm{cf}(\kappa) < \kappa$（如 $\aleph_\omega$） |
| CH / GCH | 连续统假设 / 广义连续统假设 |
| Easton 定理 | 正则基数上指数的自由度 |
| Silver 定理 | 奇异基数上指数的刚性 |

## 7 小结

- **基数** = 与给定集合等势的最小序数；$\aleph_0 < \aleph_1 < \aleph_2 < \cdots$ 无限向上。
- 无穷基数的**加法乘法都坍缩**：$\kappa+\lambda = \kappa\cdot\lambda = \max\{\kappa,\lambda\}$。
- **康托尔定理** $|X| < |\mathcal{P}(X)|$ 给出 $2^{\aleph_0} > \aleph_0$，对角线法是它的引擎。
- **CH**：$2^{\aleph_0}=\aleph_1$；**独立于 ZFC**（Gödel 1938、Cohen 1963）。
- Easton 与 Silver 定理刻画了指数函数的「自由度」与「刚性」。

在下一节，我们将回到「每个非空集族都能选元素吗」：**良序定理与选择公理**——Zorn 引理、良序定理与选择公理如何被证明等价，为什么最「显然」的选择公理偏偏是 ZFC 里最不显然的一条。
