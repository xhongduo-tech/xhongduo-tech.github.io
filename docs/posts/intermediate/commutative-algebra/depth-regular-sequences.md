---
title: 深度与正则序列
date: 2026-08-07
---

# 深度与正则序列

<div class="epigraph">
<p>维数告诉你模有多「宽」，深度告诉你模有多「深」。</p>
<footer>—— 同调代数进入交换代数后的常识（Auslander 语意转述）</footer>
</div>

<div class="article-byline">
<p>第二级 · 交换代数 ｜ Matsumura Ch. 6 / Eisenbud Ch. 17-18 ｜ 2026-08-07</p>
</div>

## 为什么从深度开始

上一节的 Koszul 复形告诉我们：正则序列「层层压低维数」。但一个环/模里，正则序列最长能到多长？这个数就是**深度（depth）**。它像维数一样衡量「大小」，但衡量的方向不同：维数看「能塞进多大的坐标系统」，深度看「最多能取几个横截的方程」。两者之间永远有不平等 $\operatorname{depth} \leq \dim$——等号成立时，就是下一篇的主角 **Cohen–Macaulay 环**。<span class="marginnote">深度的另一个名字是「同调维数」（homological dimension）——因为深度可以被 $\operatorname{Ext}^i(k, M)$ 精确测定，这是交换代数与同调代数最早的汇合点之一。Auslander 与 Buchsbaum 在 1950 年代把它做成一把通用尺。</span>

这一篇建立深度的两条道路：组合的（正则序列的最大长度）与同调的（$\operatorname{Ext}$ 消失的起点），再用 **Auslander–Buchsbaum 公式**把深度与自由分解的射影维数配平。

## 1 深度的定义与基本性质

设 $(R, \mathfrak{m}, k)$ Noether 局部环，$M$ 有限生成 $R$-模。

**深度（depth）**：$\mathfrak{m}$ 中 $M$ 上的正则序列的最大长度，记为 $\operatorname{depth} M$（或 $\operatorname{depth}_R M$）。<span class="marginnote">几何直觉：深度是「在 $\mathfrak{m}$ 里能横截地切 $M$ 的次数」。$M = R$ 时，深度 = 能取多少个「处处都非零因子」的元素——像「能切几刀」。</span>

基本例子：
$R = k[x,y]_{(x,y)}$：$\operatorname{depth} R = 2$（$x, y$ 是正则序列，且不可能更长）。
- $R = k[x,y]/(xy)$：$\operatorname{depth} R = 1$（$x + y$ 非零因子，但 $x$ 是零因子——任何长 2 的序列里总会被 $(x,y)$ 处某个元素杀死）。
- $R = k[x,y]/(x^2, xy)$：$\operatorname{depth} R = 0$（$x$ 本身就是零因子，$xy = 0$）。

**核心对照表：深度的计算**

| 环 $R$（局部化处） | $\dim R$ | $\operatorname{depth} R$ | 说明 |
| --- | --- | --- | --- |
| $k[x,y]_{(x,y)}$ | 2 | 2 | 正则序列 $x, y$ |
| $k[x,y]/(xy)$ | 1 | 1 | 节点曲线，CM |
| $k[x,y]/(x^2, xy)$ | 1 | 0 | 嵌入分支，深度 0 |
| $k[x]/(x^n)$ | 0 | 0 | 零维，深度 0 |
| $\mathbb{Z}_{(p)}$ | 1 | 1 | DVR，深度 1 |
| $k$（域） | 0 | 0 | 平凡情形 |

竖着读：**深度看「能在 $\mathfrak{m}$ 里取几个非零因子」，维数看「能砌多长的素理想链」**。$k[x,y]/(x^2,xy)$ 维数仍 1 但深度已 0——两者在这张表里第一次分开。

几何上，深度回答「在 $\mathfrak{m}$ 处能横截地切 $M$ 几次」：每切一刀（除以正则元素），维数降 1 而商模仍干净；切到不能切为止的刀数就是深度。$k[x,y]/(x^2,xy)$ 的第一刀就切不动——$x$ 是零因子，正是第1篇《准素分解》的嵌入素理想在作祟。

**重点：$\operatorname{depth} M \leq \dim M$，且对短正合列行为良好。** 核心性质：
1. $\operatorname{depth} M \leq \dim M$；
2. 短正合列 $0 \to M' \to M \to M'' \to 0$ 给出 $\operatorname{depth} M \geq \min\{\operatorname{depth} M', \operatorname{depth} M''\}$，以及两端夹逼的不等式组。<span class="marginnote">不等式 $\operatorname{depth} \leq \dim$ 的证明常用正则序列 + 高度：$d$ 个元素的生成理想高度 ≤ $d$，而长度 $d$ 的正则序列对应的商维数至少掉 $d$——两条合流即得。它是下一篇「CM = 等号」的前提。</span>

**辨析｜易错点：** 深度是**局部**概念（依赖 $\mathfrak{m}$），但 $\operatorname{depth} M = \inf\{\operatorname{depth} R_{\mathfrak{p}} M_{\mathfrak{p}} : \mathfrak{p} \in \operatorname{Supp} M\}$ 给出整体还原——**深层信息在极小相伴素处取到**（见《相伴素与支集》）。不要一看到深度就把环当全局对象。

**辨析｜深度的两个端点：** $\operatorname{depth} M = 0$ ⇔ 存在非零 $m$ 被 $\mathfrak{m}$ 杀死 ⇔ $\mathfrak{m} \in \operatorname{Ass} M$ ⇔ $H^0_{\mathfrak{m}}(M) \neq 0$（见最后一篇局部上同调）；$\operatorname{depth} M = \dim M$ ⇔ $M$ 是 CM 模（下一篇）。深度的两个端点，恰好是两篇论文的起点——本篇与《Cohen–Macaulay》。这张「深度看两点」的地图，从第一篇《相伴素与支集》的「$\operatorname{Ass}$ 是病历卡」一路铺到末篇局部上同调。

## 2 深度的同调刻画：$\operatorname{Ext}$ 语言

**重点：$\operatorname{depth} M = \inf\{\, i \mid \operatorname{Ext}^i_R(k, M) \neq 0\,\}$，且当 $M \neq 0$ 时该下确界 ≤ $\dim M$。** 换句话说：

$$\operatorname{Ext}^i_R(k, M) = 0 \ (\forall\, i < d) \qquad\text{当且仅当}\qquad \operatorname{depth} M \geq d.$$

这条公式把「存在正则序列」翻译成「前 $d$ 个 $\operatorname{Ext}$ 消失」，其证明套路固定：用 $\operatorname{Ext}$ 对短正合列的长正合列归纳，正则序列每个元素对应一次「$\operatorname{Ext}^i(k, \cdot) = 0$ 的平移」。<span class="marginnote">为什么是 $k$？因为 $k = R/\mathfrak{m}$ 是「唯一的点」，而 $\operatorname{Ext}^i(k, M)$ 度量「$M$ 在 $\mathfrak{m}$ 上的第 $i$ 阶扩张」——深度就是「$M$ 不被 $\mathfrak{m}$ 过早杀掉」的阶数。这是局部上同调（本专题最后一篇）的同调前奏。</span>

**为什么要用 $k$ 测深度：** 因为 $k = R/\mathfrak{m}$ 是「唯一的点」，$\operatorname{Ext}^i(k, M)$ 度量「$M$ 在 $\mathfrak{m}$ 上的第 $i$ 阶扩张」。另一个视角：$\operatorname{depth} M = \inf_{\mathfrak{p}} \operatorname{depth}_{R_{\mathfrak{p}}} M_{\mathfrak{p}}$，最小值在极小相伴素处取到（见《相伴素与支集》）。**深度是「处处取最小」的量，维数是「处处取最大」的量**——二者在 CM 时重合，正是下一篇的等号。

**辨析｜易错点：** $\operatorname{Ext}^i(k, M) = 0$（$i < d$）与「存在长度 $d$ 的正则序列」是同一个事实的两张脸。常见误区是把「$\operatorname{Ext}^d(k, M) \neq 0$」当成「深度 $= d$」的判据——实际深度是「**首个**非零」而非「最后一个非零」的下标。

## 3 Auslander–Buchsbaum 公式

对**正则局部环**上的有限生成模，深度与投射维数被一条公式锁定：

**Auslander–Buchsbaum 公式**：$(R, \mathfrak{m})$ 正则局部，$M$ 有限生成 $R$-模，$\operatorname{pd} M < \infty$，则

$$\operatorname{pd} M + \operatorname{depth} M = \dim R.$$

**「射影维数 + 深度 = 环的维数」**——模越是「不深」，自由分解越要拉长补足。$M = k$ 时：$\operatorname{depth} k = 0$，故 $\operatorname{pd} k = \dim R$；正则局部环的自由分解长度恰为维数，$k[x_1,\dots,x_n]$ 的 Koszul 复形正是这条长度的实现。<span class="marginnote">这是本专题的「守恒律」：信息总量固定，深度高则分解短，深度低则分解长。Koszul 复形（《Koszul 复形》）在正则环上给出长度 = 维数的自由分解，两者在此合流。</span>

证明的核心是「自由模上 $\operatorname{Ext}^i(k, \cdot)$ 的消失 + 归纳剥自由模」，最终归结为对 $\operatorname{depth}$ 的递推——与深度定义的递归性（逐层取非零因子）严丝合缝。

**辨析｜易错点：** Auslander–Buchsbaum 公式要求 $\operatorname{pd} M < \infty$（正则环上自动成立，Serre 定理）。非正则环上该条件不自动满足——$R = k[x,y]/(x^2, xy)$ 上 $k$ 的投射维数无穷，公式失效。**看到公式先确认「正则局部环」前提。**

用公式算一笔：$R = k[x,y]_{(x,y)}$，$M = k$。$\dim R = 2$，$\operatorname{depth} k = 0$，故 $\operatorname{pd} k = 2$。$k$ 的自由分解是 $0 \to R \xrightarrow{\binom{-y}{x}} R^2 \xrightarrow{(x\ \ y)} R \to k \to 0$，长度恰为 2，正是《Koszul 复形》里那条复形——**Auslander–Buchsbaum 保证这条分解不能再短**。

## 4 公式解析：正则序列与 $\operatorname{Ext}$ 的握手

把两条道路写进一条公式。设 $(R, \mathfrak{m})$ Noether 局部，$M \neq 0$ 有限生成，则

$$\operatorname{depth} M = \min\{ i \geq 0 \mid \operatorname{Ext}^i_R(R/\mathfrak{m}, M) \neq 0 \} = \max\{\, r \mid \exists\, M\text{-正则序列 } x_1,\dots,x_r \subseteq \mathfrak{m}\,\}.$$

- **第一步，左侧**：$\operatorname{Ext}^0(k, M) = \operatorname{Hom}(k, M) = \{m \in M \mid \mathfrak{m}m = 0\}$——「被 $\mathfrak{m}$ 整体杀死的元素」（socle）。它非零当且仅当深度 0。这就是「$\operatorname{Ext}$ 消失的起点 = 深度」在 $i = 0$ 处的对照。
- **第二步，中间**：对短正合列 $0 \to M' \to M \to M'' \to 0$，$\operatorname{Ext}^\bullet(k, \cdot)$ 的长正合列把「杀掉 $\mathfrak{m}$」逐阶传递；正则序列每延长一个元素，就迫使 $\operatorname{Ext}^i(k, \cdot)$ 对下一个 $i$ 消失——**组合定义与同调定义逐阶对齐**。
- **第三步，收束**：两条定义的等价证明靠归纳：$x_1$ 正则 ⇒ $0 \to M \xrightarrow{x_1} M \to M/x_1 M \to 0$ 正合 ⇒ $\operatorname{Ext}$ 长正合列 ⇒ $\operatorname{Ext}^i(k, M) = 0$（$i \leq d$）随 $M/x_1 M$ 的深度 $d - 1$ 一并得到。

**辨析｜易错点：** $\operatorname{Ext}^0(k, M) = \operatorname{Hom}_R(k, M)$ 不是「$\mathfrak{m}$ 幂归零」的全体（那是 $\Gamma_{\mathfrak{m}}(M)$，见最后一篇局部上同调），而是「**一步**被 $\mathfrak{m}$ 杀死」的子模（socle）。深度 0 的正确判别是「存在非零元素被 $\mathfrak{m}$ 杀死」，别和「被某幂杀死」混为一谈。

做一道完整算例。$R = k[x]_{(x)}$，$M = R$：$\mathfrak{m} = (x)$，正则序列取 $\{x\}$（$x$ 在 $R$ 上是非零因子，$R/(x) = k \neq 0$），长度 1，无法再长——$\operatorname{depth} R = 1 = \dim R$，$R$ 是 CM（事实上是 DVR，见第1篇《离散赋值环》）。$\operatorname{Ext}$ 侧验证：$\operatorname{Ext}^0(k, R) = \operatorname{socle} R = 0$（整环里没有非零元被 $x$ 杀死），$\operatorname{Ext}^1(k, R) \cong k \neq 0$——首个非零下标恰是 1，两条道路一致。

再验 $R = k[x,y]/(x^2, xy)$ 的深度 0：$\bar{x} \in R$ 满足 $\bar{x} \cdot \mathfrak{m} = (\bar{x}^2, \bar{x}\bar{y}) = 0$，故 $\bar{x} \in \operatorname{Hom}(k, R) \neq 0$——**「被 $\mathfrak{m}$ 一步杀死」正是深度 0 的同调面貌**。

至此三条算例在深度 0、1、2 三个档位上，用「数正则序列」与「数 $\operatorname{Ext}$ 首非零下标」两条路都给出同一答案——本篇的「两条道路」就此闭环。

**术语速查表**

| 术语 | 一句话含义 |
| --- | --- |
| 深度 $\operatorname{depth} M$ | $\mathfrak{m}$ 中正则序列最大长度 |
| 正则序列 | 每个元素在逐层商上非零因子 |
| socle | 被 $\mathfrak{m}$ 一步杀死的元素 |
| $\operatorname{Ext}^i(k, M)$ | $M$ 在 $\mathfrak{m}$ 上的第 $i$ 阶扩张 |
| 投射维数 $\operatorname{pd}$ | 自由分解的最小长度 |
| Auslander–Buchsbaum | 正则局部环上 $\operatorname{pd} M + \operatorname{depth} M = \dim R$ |

## 5 小结

- **深度** = $\mathfrak{m}$ 中正则序列最大长度；$\operatorname{depth} M \leq \dim M$。
- **同调刻画**：$\operatorname{depth} M = \min\{i : \operatorname{Ext}^i(k, M) \neq 0\}$。
- **Auslander–Buchsbaum 公式**：正则局部环上 $\operatorname{pd} M + \operatorname{depth} M = \dim R$——深度高则分解短。
- 例子：$k[x,y]$ 深度 2，$k[x,y]/(xy)$ 深度 1，$k[x,y]/(x^2,xy)$ 深度 0。

在下一节，深度取到上限——**Cohen–Macaulay 模与 Gorenstein 环**：当 $\operatorname{depth} = \dim$，模有了「处处均匀」的好脾气，而 Gorenstein 环更把自对偶性也收入囊中。
