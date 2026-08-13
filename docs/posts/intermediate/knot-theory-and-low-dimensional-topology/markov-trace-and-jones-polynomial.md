---
title: Markov 迹与 Jones 多项式
date: 2026-08-07
---

# Markov 迹与 Jones 多项式

<div class="epigraph">
<p>一条迹若能通过 Markov 移动的考验，它就能识别闭辫——进而识别一切结。</p>
<footer>—— 本文作者按</footer>
</div>

<div class="article-byline">
<p>第二级 · 纽结理论与低维拓扑 ｜ Lickorish《An Introduction to Knot Theory》第14章 ｜ 2026-08-07</p>
</div>

## 为什么从「迹」开始

Alexander 定理说「任何结都是闭辫」，但反过来**同一个结对应很多条辫子**。要定义「不依赖辫子选择的结不变量」，必须弄清：两条辫子何时给出同一个结？答案是 **Markov 定理**：当且仅当它们可以通过两类**Markov 移动**互相转化。于是，「在 Markov 移动下不变的量」自动成为结不变量——这类量就是 **Markov 迹（Markov trace）**。

Jones 多项式的代数定义（上一节）依赖 Temperley-Lieb 代数上的迹；而「迹」与「Markov 移动」的结合，正是 Jones 多项式良定义性的根源。这一节把「迹」单独推上舞台：它为什么唯一、怎么构造、如何给出 Jones。<span class="marginnote">Markov 移动由 Markov 在 1935 年（给维诺格拉多夫的信）提出，是「辫子等价」的完整刻画。Alexander 定理给出「到辫子的满射」，Markov 定理给出「辫子上的等价关系」——两者合起来，把「结的分类」翻译成「辫群上的迹分类」，这是纽结理论最成功的代数归约之一。</span>

## 1 闭辫与 Markov 移动

设 $B_n$ 是 $n$ 股辫群，$\beta \in B_n$，$\widehat{\beta}$ 为闭辫。两条辫子 $\beta, \beta'$ 的闭辫同痕，当且仅当 $\beta'$ 可由 $\beta$ 经有限次两类操作得到：

**共轭（conjugation）**：$\beta \sim \gamma \beta \gamma^{-1}$（$\gamma \in B_n$）。几何上：在闭辫里「把一个圈移到另一个位置」。
**稳定化（stabilization）**：$\beta \sim \beta \sigma_n^{\pm 1}$（把 $B_n$ 视为 $B_{n+1}$ 的子群，$\sigma_n$ 是新加的最右生成元）。几何上：给闭辫「加或去一股」，同时加或去一个交叉。

**定理（Markov，1935）**：两个闭辫表示同一个链环，当且仅当两条辫子可以用有限次共轭与稳定化（及其逆）相互转化。

**易错点｜稳定化改变股数**：共轭保持 $n$ 不变，稳定化把 $B_n$ 升级到 $B_{n+1}$（加一股）或降级（去一股）。所以「闭辫等价」不是固定股数的辫群内部的等价，而是一个跨股数的等价——这正是 Markov 定理比「辫群内共轭」更广的原因。

## 2 Markov 迹的定义

**Markov 迹（Markov trace）**：一族线性泛函 $\operatorname{tr}_n : B_n \to R$（$R$ 为某环，如 $\mathbb{Z}[A^{\pm 1}]$），满足：

1. **迹性（trace property）**：$\operatorname{tr}_n(\alpha\beta) = \operatorname{tr}_n(\beta\alpha)$。
2. **Markov 性质**：对 $\beta \in B_n$，$\operatorname{tr}_{n+1}(\beta \sigma_n^{\pm 1}) = z^{\pm 1} \operatorname{tr}_n(\beta)$，其中 $z$ 是固定标量。

迹性保证「共轭不变」；Markov 性质保证「稳定化只乘标量」。于是**闭辫的不变量**为：

$$
f(\widehat{\beta}) = \alpha^{\,w(\beta)}\, \operatorname{tr}_n(\beta),
$$

其中 $\alpha$ 是修正因子、$w(\beta)$ 是辫子的「指数和」（把 $\beta$ 写成 $\sigma_i^{\pm 1}$ 的字，指数之和）。适当的 $\alpha, z$ 选择使 $f$ 在 Markov 移动下完全不变——这就是结不变量。

**辨析｜迹性 vs 交换性**：迹性 $\operatorname{tr}(\alpha\beta) = \operatorname{tr}(\beta\alpha)$ 比交换性弱——它不要求 $\alpha\beta = \beta\alpha$，只要求两者的**迹**相等。辫群非交换，但非交换元素仍可有交换的迹。正是这个「弱交换性」让迹能穿过共轭而不要求辫群可交换。

## 3 公式解析：为什么 Markov 迹定义出结不变量

要证明 $f(\widehat{\beta}) = \alpha^{w(\beta)}\operatorname{tr}_n(\beta)$ 是良定义的结不变量，逐一检查两类 Markov 移动：

$$
f(\widehat{\gamma\beta\gamma^{-1}}) \overset{?}{=} f(\widehat{\beta}), \qquad
f(\widehat{\beta\sigma_n}) \overset{?}{=} f(\widehat{\beta}).
$$

- **第一步，共轭不变**：$\operatorname{tr}_n(\gamma\beta\gamma^{-1}) = \operatorname{tr}_n(\beta)$（迹性：把 $\gamma$ 挪到最右）。指数和 $w$ 在共轭下不变，所以 $f$ 不变。
- **第二步，稳定化不变**：$w(\beta\sigma_n) = w(\beta) + 1$，而 $\operatorname{tr}_{n+1}(\beta\sigma_n) = z\, \operatorname{tr}_n(\beta)$。修正因子 $\alpha$ 若满足 $\alpha \cdot z = 1$（即 $\alpha = z^{-1}$），则 $\alpha^{w+1} \cdot z \operatorname{tr}_n = \alpha^{w} \operatorname{tr}_n$——不变！
- **第三步，唯一性**：两类移动生成所有等价，所以 $f$ 对所有闭辫良定义；Alexander 定理保证覆盖所有结；于是 $f$ 是结不变量。

**关键教训**：Markov 迹是「把辫群上的迹变成结不变量」的唯一障碍——满足迹性与 Markov 性质的迹，自动给出良定义的结不变量。<span class="marginnote">这解释了 Jones 多项式「为何存在」：Temperley-Lieb 代数上恰好存在唯一满足 Markov 性质的迹（归一化后）。「存在唯一」是构造的基石——没有唯一性，同一结可能算出不同值；唯一性保证「怎么构造都一个样」。</span>

## 4 Jones 多项式：Markov 迹的特例

把上一节与本节拼起来，Jones 多项式获得完整代数定义：

**定理**：存在唯一一族满足「迹性 + Markov 性质」的 Markov 迹 $\operatorname{tr}_n$ 于 Temperley-Lieb 代数 $TL_n$（$\delta = -A^2 - A^{-2}$），且对辫子 $\beta \in B_n$（经表示 $\sigma_i \mapsto A e_i - A^{-1}$ 进入 $TL_n$），

$$
V_{\widehat{\beta}}(t) = \left( \frac{1 - A^{-4}}{A} \right)^{\,w(\beta) - 1} \operatorname{tr}_n(\beta), \qquad t = A^{-4}.
$$

- 指数 $w(\beta) - 1$ 与因子 $\frac{1-A^{-4}}{A}$ 一起处理稳定化修正与归一化。
- 对平凡结（空辫子），$V = 1$。
- 验证三叶结：取 $n = 2$、$\beta = \sigma_1^3$，代入公式得 $V_{3_1}(t) = -t^{-4} + t^{-3} + t^{-1}$，与括号定义一致。

**易错点｜代数定义 ≠ 组合定义**：代数定义（Markov 迹）与图定义（Kauffman 括号）殊途同归，但证明策略不同：括号用 Reidemeister 移动验证不变性；迹用 Markov 移动验证。前者是「局部检查」，后者是「全局移动」——两条路都成立，说明 Jones 多项式的性质「被两种结构同时保证」。

## 5 辫子里的手性：为什么 $\sigma_1^3$ 与 $\sigma_1^{-3}$ 不可互化

把三叶结写进辫群 $B_2$：$B_2 = \{\sigma_1^k : k \in \mathbb{Z}\}$ 是无穷循环群。正三叶结对应 $\sigma_1^3$，镜像对应 $\sigma_1^{-3}$。**Markov 定理告诉我们它们不等价**，并给出「辫子层面」的验证路径：

$$
\sigma_1^3 \not\sim_{\text{Markov}} \sigma_1^{-3}.
$$

- **第一步，先看共轭**：$B_2$ 是交换群（只有 $\sigma_1$ 一个生成元，且无关系），所以共轭在 $B_2$ 里是恒等操作——$\gamma \sigma_1^3 \gamma^{-1} = \sigma_1^3$。共轭完全不能改变 $\sigma_1$ 的指数。
- **第二步，看稳定化**：稳定化把 $B_2$ 升级到 $B_3$。$\sigma_1^3$ 与 $\sigma_1^{-3}$ 的差别是「指数符号」，而稳定化只加/去「最右生成元 $\sigma_2$」——它不能把 $\sigma_1$ 的指数从正变负。
- **第三步，结论**：两类 Markov 移动都无法翻转 $\sigma_1$ 的指数符号，所以 $\sigma_1^3$ 与 $\sigma_1^{-3}$ 不等价——三叶结 ≠ 镜像。**Jones 多项式正是把这条「辫子层面不可互化」翻译成了「$t$ 多项式不对称」。**

**辨析｜辫子指数与卷绕数**：$\sigma_1^3$ 的指数 $w = 3$ 正是三叶结的卷绕数。对 $B_2$ 里的结，辫子指数 = 卷绕数——「绕了几圈」在辫子语言里就是「$\sigma_1$ 的指数」。

**延伸｜一般股数的复杂性**：对 $n \ge 3$，辫群非交换（$\sigma_1\sigma_2\sigma_1 \neq \sigma_2\sigma_1\sigma_2$），共轭不再是恒等操作——「把圈移来移去」真的能改变辫子的写法。此时判断两条辫子是否 Markov 等价需要真正处理共轭与稳定化的组合，这正是 Markov 定理的深度所在：它不是「看一眼就能判断」的平凡标准，而是一条需要计算的判定准则。

## 6 Markov 迹的结构与推广

Markov 迹不是 Jones 的专利——它是「从辫群提取不变量」的通用机制：

- **起源**：Jones 研究**冯 · 诺依曼代数的子因子（subfactor）**时，用 Markov 迹构造「Jones 指标」；当指标落在「离散禁区」$[4, \infty)$ 的特殊值时，子因子理论自动吐出 Markov 迹，进而吐出结不变量——这是「算子代数意外发明结理论」的传奇。
- **唯一性**：对 $TL_n$（与 Jones 相关的代数），满足归一化与 Markov 性质的迹**存在且唯一**——这是 Jones 多项式的「代数唯一性」证明。
- **一般化**：把 $TL_n$ 换成任意带 $R$-矩阵的辫群表示（量子群表示），同样构造 Markov 迹，得到 **Reshetikhin-Turaev 不变量**（第3篇之四）。迹的「标记」从单参数推广到表示论的标号。
- **多变量**：HOMFLY、Kauffman 多项式也有 Markov 迹形式——只是底层代数从 $TL_n$ 换成更一般的 **Hecke 代数**（$q$-对称群）或 **Birman-Wenzl 代数**（正交/辛型）。「迹 → 不变量」的配方不变，变的只是代数。
- **归一化的选择**：Markov 迹的定义含标量 $z$ 与修正因子 $\alpha$ 的自由度；不同选择给出相差归一化的不变量。约定「$\operatorname{tr}(1) = 1$」并取 $z$ 使平凡结映到 1，则唯一固定——这就是各教材 Jones 多项式一致的原因。

**定理（Jones）**：$TL_n$ 上的 Markov 迹是唯一的——若 $\operatorname{tr}_n$ 与 $\operatorname{tr}'_n$ 都满足迹性与 Markov 性质且 $\operatorname{tr}_1(1) = \operatorname{tr}'_1(1)$，则二者相等。<span class="marginnote">唯一性定理是 Jones 1984 论文的「脚手架」：它保证了「用 Temperley-Lieb 代数造出的不变量与用括号造出的不变量是同一个」。数学中「同一对象被多条独立路径唯一确定」往往是深刻信号——Jones 多项式后来在 Khovanov 同调、Chern-Simons 理论、量子群中反复出现，正是这种「结构性必然」的体现。</span>

## 7 小结

- **Markov 定理**：两个闭辫同结 ⟺ 可由**共轭**与**稳定化**（加/去一股）相互转化。
- **Markov 迹**：满足迹性与 Markov 性质的线性泛函族；共轭不变 × 稳定化只乘标量。
- 修正因子 $\alpha = z^{-1}$ 使 $f(\widehat{\beta}) = \alpha^{w(\beta)}\operatorname{tr}_n(\beta)$ 成为良定义的结不变量。
- **Jones 多项式** = Temperley-Lieb 代数上唯一 Markov 迹对辫子的求值；三叶结 $V = -t^{-4}+t^{-3}+t^{-1}$。
- 迹机制是通用的：换底层代数即得 HOMFLY、Kauffman 与量子不变量——「迹 → 不变量」是量子拓扑的标准引擎。
- 对 $B_2$ 中的三叶结，辫子指数 $w = 3$ 即卷绕数；$\sigma_1^3$ 与 $\sigma_1^{-3}$