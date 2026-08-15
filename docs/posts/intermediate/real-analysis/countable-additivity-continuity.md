---
title: 测度的可数可加性与连续性
date: 2026-08-07
---

# 测度的可数可加性与连续性

<div class="epigraph">
<p>测度对集合列的极限是连续的：就像质量守恒——逐块累加与整体称量，永远给出同一个数。</p>
<footer>—— 恩里科 · 波莱尔（Émile Borel）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》§3.6 ｜ 2026-08-07</p>
</div>

## 为什么从测度的可数可加性与连续性开始

前几节我们有了可测集 σ-代数 $\mathcal{M}$ 与可数可加的测度 $m$。本节把这些「静态」性质升级为「动态」性质：**当集合列递增地并拢或递减地收缩时，测度如何随极限变化？** 测度的连续性回答：测度与集合列的极限「交换顺序」——$\lim_k m(E_k)=m(\lim E_k)$。

连续性是一切极限定理（Levi 单调收敛、Fatou 引理、控制收敛）的**测度论引擎**。没有它，积分号下的极限交换就无从谈起。本节是「静态测度」迈向「动态分析」的转折点。<span class="marginnote">测度的连续性本质上是<strong>可数可加性的「积分形式」</strong>：把递增并拆成不相交差，可数可加性就在每一项上生效。反过来，从「单调连续性」也能推出可数可加性（取 $E_k$ 的递增部分和）——<strong>可数可加性 ⇔ 有限可加性 + 下连续</strong>，这是测度论里一对漂亮的等价关系。</span>

## 1 集合列的上极限与下极限回顾

先回顾第二篇的记号（见《集合列的极限》一篇）。设 $\{E_k\}$ 是一列集合：

$$\liminf_{k\to\infty}E_k=\bigcup_{n=1}^{\infty}\bigcap_{k=n}^{\infty}E_k,\qquad \limsup_{k\to\infty}E_k=\bigcap_{n=1}^{\infty}\bigcup_{k=n}^{\infty}E_k$$

直观：下极限是「最终都在」的点集（除有限个外全在）；上极限是「无限次在」的点集。若两者相等，称 $\{E_k\}$ 收敛，极限记 $\lim E_k$。**递增并是特殊情况**：$E_1\subset E_2\subset\cdots$ 时 $\lim E_k=\bigcup_kE_k$；递减时 $\lim E_k=\bigcap_kE_k$。

**重点：可测集列的上下极限仍是可测集。** 因为可数并、可数交在 σ-代数内封闭——这是上节 σ-代数「运算底盘」的直接回报。

## 2 测度的连续性：递增与递减

**定理（测度的下连续）**：若 $E_1\subset E_2\subset\cdots$ 可测，则

$$m\left(\bigcup_{k=1}^{\infty}E_k\right)=\lim_{k\to\infty}m(E_k)$$

证明：令 $F_1=E_1$，$F_k=E_k\setminus E_{k-1}$（$k\ge2$）。$\{F_k\}$ 两两不相交，$\bigcup_kE_k=\bigcup_kF_k$（不相交并），且 $m(E_k)=\sum_{i=1}^{k}m(F_i)$（有限可加）。由可数可加性：

$$m\left(\bigcup_{k=1}^{\infty}E_k\right)=\sum_{i=1}^{\infty}m(F_i)=\lim_{k\to\infty}\sum_{i=1}^{k}m(F_i)=\lim_{k\to\infty}m(E_k)$$

**定理（测度的上连续）**：若 $E_1\supset E_2\supset\cdots$ 可测，且 $m(E_1)<+\infty$，则

$$m\left(\bigcap_{k=1}^{\infty}E_k\right)=\lim_{k\to\infty}m(E_k)$$

证明：由德摩根律 $\bigcap_kE_k=E_1\setminus\bigcup_k(E_1\setminus E_k)$，而 $\{E_1\setminus E_k\}$ 递增，用下连续：

$$m\left(\bigcap_kE_k\right)=m(E_1)-m\left(\bigcup_k(E_1\setminus E_k)\right)=m(E_1)-\lim_km(E_1\setminus E_k)=\lim_km(E_k)$$

**辨析｜易错点：上连续需要「$m(E_1)<+\infty$」。** 若 $E_k=[k,+\infty)$，递减且 $m(E_k)=+\infty$ 恒成立，但 $\bigcap_kE_k=\varnothing$，$m=0\neq\lim\infty$。**无穷测度会「漏走」**——这是「无穷」在测度论里的经典陷阱，也是后续控制收敛定理要求「被控制函数可积」的原因。<span class="marginnote">上连续的条件「$m(E_1)<\infty$」等价于概率论里的<strong>概率测度连续</strong>：概率测度自动满足 $m(X)=1<\infty$，故上下连续无条件成立。这是概率测度特别「好」的原因，也是 Fatou 引理在概率里总成立的原因。</span>

## 3 上下极限的测度不等式

对一般的集合列（不一定单调），测度与上下极限之间有一对著名的联系：

**定理（上极限的次可加估计）**：若 $\sum_{k=1}^{\infty}m(E_k)<+\infty$，则

$$m\left(\limsup_{k\to\infty}E_k\right)=0$$

证明：$\limsup E_k=\bigcap_n\bigcup_{k\ge n}E_k\subset\bigcup_{k\ge n}E_k$ 对每个 $n$，故 $m(\limsup E_k)\le m(\bigcup_{k\ge n}E_k)\le\sum_{k\ge n}m(E_k)\to0$（尾部趋于零，因级数收敛）。**「总测度有限」⇒「无限次出现的点集零测」**——这是 Borel–Cantelli 引理的测度论版本，概率论里它是「无穷多次事件几乎不发生」的判据。

**定理（Fatou 型不等式）**：对任意可测列 $\{E_k\}$，

$$m\left(\liminf_{k\to\infty}E_k\right)\le\liminf_{k\to\infty}m(E_k)$$

证明：$\liminf E_k=\bigcup_n\bigcap_{k\ge n}E_k$，而 $\bigcap_{k\ge n}E_k$ 关于 $n$ 递增，由下连续：

$$m(\liminf E_k)=\lim_n m\left(\bigcap_{k\ge n}E_k\right)\le\liminf_n\inf_{k\ge n}m(E_k)=\liminf_k m(E_k)$$

**重点：这两条不等式是「积分号下取极限」三大定理的集合版预言。** Borel–Cantelli 对应「收敛级数的尾效应」，Fatou 型对应「下极限单调」，它们将在第六篇以积分形式（Levi、Fatou、控制收敛）重新登场。<span class="marginnote">Fatou 型不等式的集合版与<strong>Fatou 引理</strong>（第六篇）同名同源：都是「下极限的测度 ≤ 测度的下极限」。在概率论中它正是「事件概率的下极限」控制，用于证明大数定律的初等形式。</span>

## 4 公式解析：下连续证明里的「环形分解」

下连续的全部秘密在「把递增并改写成不相交并」：

$$m\left(\bigcup_{k=1}^{\infty}E_k\right)=\sum_{k=1}^{\infty}m(F_k),\qquad F_k=E_k\setminus E_{k-1}$$

- **第一步，读「$F_k=E_k\setminus E_{k-1}$」**：$F_k$ 是「第 $k$ 圈新增的环形地带」——递增并的增量。这些环形地带两两不相交（$E_k\supset E_{k-1}$ 保证），且拼起来恰是整个并集。**「整体 = 环形带之和」**，这正是积分里「把总量切成增量」的集合版。
- **第二步，读「$m(E_k)=\sum_{i=1}^{k}m(F_i)$」**：前 $k$ 圈环形带之和 = 前 $k$ 个集合的并的测度 = $m(E_k)$（递增保证并即最大者）。**有限可加性在这里做功**——环形带不相交，求和无重叠。
- **第三步，读「$\lim_k\sum_{i=1}^k=\sum_{i=1}^\infty$」**：把「部分和」推向「无穷和」。级数收敛（每一项非负）保证极限存在（允许 $+\infty$）。**可数可加性在此刻完成「有限 → 无穷」的跳跃**——这正是它比有限可加更强大的全部意义。

**「环形分解 + 有限可加 + 级数极限」三步**，是测度论所有连续性证明的标准模板。记住 $F_k=E_k\setminus E_{k-1}$ 这个「增量」操作，就掌握了整套连续性论证。

## 6 数值演练与连续性速查

**算例一（下连续的数值验证）**：$E_k=[0,1-\tfrac1k]$。递增并 $\to[0,1)$，$m(E_k)=1-\tfrac1k\to1=m([0,1))$——下连续成立。环形分解：$F_k=[1-\tfrac1{k-1},1-\tfrac1k]$，$m(F_k)=\tfrac1{k-1}-\tfrac1k$，$\sum_k m(F_k)=1$（望远镜求和）。

**算例二（上连续的无穷陷阱）**：$E_k=[k,\infty)$。递减且 $m(E_k)=+\infty$ 恒成立，但 $\bigcap_kE_k=\varnothing$，$m=0\neq\lim\infty$——**无穷测度「漏走」**，上连续需要 $m(E_1)<\infty$。

**对照表：连续性与可加性**

| 性质 | 陈述 | 条件 |
| --- | --- | --- |
| 下连续 | $m(\bigcup E_k)=\lim m(E_k)$ | 递增 |
| 上连续 | $m(\bigcap E_k)=\lim m(E_k)$ | 递减 + $m(E_1)<\infty$ |
| 可数可加 | $m(\bigcup E_k)=\sum m(E_k)$ | 不相交 |
| Borel–Cantelli | $\sum m(E_k)<\infty\Rightarrow m(\limsup E_k)=0$ | 级数收敛 |

**术语速查**

| 记号 | 含义 |
| --- | --- |
| $\limsup E_k$ | 无限次出现的点集 |
| $\liminf E_k$ | 最终都在的点集 |
| 环形分解 | $F_k=E_k\setminus E_{k-1}$ |
| Borel–Cantelli | 总测度有限则无限出现零测 |

**辨析｜易错点：上连续与下连续不对称——上连续多一个「$m(E_1)<\infty$」。** 概率测度自动满足（$P(X)=1$），故概率里上下连续无条件成立。**「无穷测度」是上连续失效的唯一原因**，也是控制收敛要求「可积控制」的测度论根源。

### 三步记住「环形分解」

- **切环**：$F_k=E_k\setminus E_{k-1}$（递增并的增量）。
- **求和**：$m(E_k)=\sum_{i=1}^k m(F_i)$（不相交 + 有限可加）。
- **取极限**：$\lim_k\sum_{i=1}^k=\sum_{i=1}^\infty$（级数极限）。

**延伸（与概率论连接）**：Borel–Cantelli 引理——「$\sum P(E_k)<\infty$ ⇒ $P(\text{无限多次出现})=0$」——是大数定律、强大数定律、随机游走理论的基石。**「几乎必然」在概率里正是「零测」的另一副面孔**，而本节的测度不等式全部逐字搬入概率。

**一道收束练习**：用环形分解证明「可数可加 ⇔ 有限可加 + 下连续」——它是测度论「可加性与连续性互为表里」的核心等价。

## 7 小结

- **下连续**：递增可测并的测度 = 测度列的极限；$m(\bigcup E_k)=\lim m(E_k)$。
- **上连续**：递减可测交的测度 = 测度列的极限；需 $m(E_1)<+\infty$，否则无穷会漏走。
- **Borel–Cantelli 型**：$\sum m(E_k)<+\infty\Rightarrow m(\limsup E_k)=0$——总测度有限则无限出现点零测。
- **Fatou 型**：$m(\liminf E_k)\le\liminf m(E_k)$——下极限与测度可交换但只给不等号。
- **可数可加 ⇔ 有限可加 + 下连续**：连续性与可加性互为表里。
- **数值**：$E_k=[0,1-\tfrac1k]$ 递增并测度 $\to1$；$E_k=[k,\infty)$ 递减但无穷测度漏走。

在下一节，我们把连续性与可测性结合，证明**可测集的逼近定理**：用开集从外逼近、闭集从内逼近、$G_\delta$ 与 $F_\sigma$ 精确夹逼，误差任意小。
