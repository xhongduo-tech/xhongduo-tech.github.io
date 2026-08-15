---
title: 量词与否定
date: 2026-08-07
---

# 量词与否定

<div class="epigraph">
<p>我们必须知道，我们必将知道。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第一级 · 数学证明导论（证明方法） ｜ Velleman, How to Prove It §2.1–2.2 ｜ 2026-08-07</p>
</div>

## 为什么从量词开始

上一课《命题与逻辑联结词》处理的是「没有变元」的句子。可数学里几乎每句话都带着「所有」或「存在」：**所有**偶数都能被 $2$ 整除，**存在**一个质数大于 $100$，**对任意** $\varepsilon>0$ **都存在** $\delta>0$……这些「所有」「存在」就是**量词（quantifier）**。量词让数学摆脱了「逐条列举」的无力感，一句话就能管住无穷多个对象；而**正确地否定带量词的命题**，则是反证法、构造反例的共同基本功。<span class="marginnote">为什么把量词和否定放在一课？因为否定会「穿过」量词并翻转它：否定「所有」得「存在……不」，否定「存在」得「所有……不」。这条法则加上 De Morgan 律，是逻辑变形里最常出错、也最值得练熟的一组。</span>

## 1 开句：带有变元的句子

**开句（open sentence）**：含变元的陈述句，本身无真假，把变元代以具体对象后才成为命题。<span class="marginnote">开句又叫<strong>命题函数（propositional function）</strong>、谓词（predicate）。「$x > 3$」「$n$ 是偶数」都是开句。高中数学里「方程 $x^2=1$ 的解集」就是「找使开句为真的 $x$」，与集合论的描述法 $\{x \mid P(x)\}$ 直接相通，见《集合的概念》。</span>例如：

$$P(x): \quad x > 3$$

当 $x = 5$ 时 $P(5)$ 为真；当 $x = 1$ 时 $P(1)$ 为假。开句本身不断真假，量词的作用，就是**把开句「关起来」变成命题**。

$x$ 可以取值的范围叫**论域（domain of discourse）**，也叫个体域。论域必须事先声明：说「所有 $x$」，总要交代 $x$ 是自然数、整数还是实数，否则「所有」无从谈起。

## 2 全称量词与存在量词

**全称量词（universal quantifier）**记作 $\forall$（取自英文 All 倒写），读作「对所有」「对任意」。命题

$$
\forall x \in \mathbb{N},\; P(x)
$$

意思是「论域 $\mathbb{N}$ 里每一个 $x$ 都使 $P(x)$ 为真」——例如「所有自然数都是非负的」。**证明全称命题只有一条路：取一个任意的 $x$，仅凭它是论域的一员、不附加任何额外假设，推出 $P(x)$。** 这就是《直接证明》那篇要练的核心动作。

**存在量词（existential quantifier）**记作 $\exists$（取自英文 Existence），读作「存在」。命题

$$
\exists x \in \mathbb{Z},\; P(x)
$$

意思是「论域 $\mathbb{Z}$ 里**至少有一个** $x$ 使 $P(x)$ 为真」——例如「存在一个整数 $x$ 满足 $x^2 = 4$」。**证明存在命题通常是给出一个具体的例子**，比如指出 $x = 2$ 即可，这类证法叫**构造性证明**，见《存在性与唯一性证明》。

量词管辖的那部分式子叫它的**辖域（scope）**。被量词约束的变元是**约束变元（bound variable）**，没被任何量词约束的是**自由变元（free variable）**。例如 $\forall x\,(x + y = 0)$ 中 $x$ 是约束的、$y$ 是自由的——这个式子仍是开句，只有再对 $y$ 量化或赋值才成为命题。

## 3 公式解析：否定如何穿过量词

现在来到本课的主角——**量词否定法则**：

$$
\neg\big(\forall x\,P(x)\big) \Longleftrightarrow \exists x\,\neg P(x)
$$

$$
\neg\big(\exists x\,P(x)\big) \Longleftrightarrow \forall x\,\neg P(x)
$$

- **第一步，读懂第一条**：「并非（所有 $x$ 都满足 $P$）」等价于「存在某个 $x$ 不满足 $P$」。想推翻「全班都及格」，不必证明每个人都不及格，只要**找出一个**不及格的人就够了。<span class="marginnote">「并非所有」不等于「所有都不」！「并非所有人都喜欢你」只是「至少有一个不喜欢你」，完全允许其余人都喜欢。用记号：$\neg\forall x\,P(x)$ 推出的是 $\exists x\,\neg P(x)$，而非 $\forall x\,\neg P(x)$。这是初学者最常翻车的地方。</span>
**第二步，读懂第二条**：「并非（存在 $x$ 满足 $P$）」等价于「所有 $x$ 都不满足 $P$」。想否认「有人迟到」，必须证明**每个人都**没迟到——只指出「某个人没迟到」远远不够，因为那不能排除其他人迟到。
**第三步，合并记忆**：否定把 $\forall$ 与 $\exists$ **互换**，并让否定号 $\neg$ **落到谓词身上**。类比 De Morgan 律：$\neg(P \land Q) \Leftrightarrow \neg P \lor Q$ 中「且」变「或」，量词的 $\forall \leftrightarrow \exists$ 互换与之同构——全称是「无穷合取」，存在是「无穷析取」。

举一组对比例子，论域为全体动物：

- 「所有狗都有尾巴」$\forall x\,(D(x) \to T(x))$。其否定是「存在一条狗没有尾巴」$\exists x\,(D(x) \land \neg T(x))$。
- 「存在一只会飞的猫」$\exists x\,(C(x) \land F(x))$。其否定是「所有猫都不会飞」$\forall x\,(C(x) \to \neg F(x))$。

注意全称命题的否定中，蕴含变成了合取：$\neg\forall x\,(D(x)\to T(x)) \Leftrightarrow \exists x\,(D(x)\land\neg T(x))$，这正是蕴含 $P\to Q$ 的否定 $\neg(P\to Q) \Leftrightarrow P\land\neg Q$ 在量词下的推广。

## 4 嵌套量词：顺序即意义

量词可以叠着用，**顺序不同，意思可能完全不同**。这是量词世界最微妙的地方。<span class="marginnote">两个量词的基本读法：$\forall x\,\exists y$ 读作「对每个 $x$，都能（各自地）找到一个 $y$」；$\exists y\,\forall x$ 读作「存在一个 $y$，对所有的 $x$ 都成立」。差别在 $y$ 能否依赖 $x$。</span>看论域为实数的两个命题：

$$
\forall x \in \mathbb{R},\; \exists y \in \mathbb{R},\; x + y = 0 \qquad (\text{真})
$$

它说「每个 $x$ 都有个相反数」——对每个 $x$ 取 $y = -x$ 即可，$y$ 可以随 $x$ 变。这是真的。

$$
\exists y \in \mathbb{R},\; \forall x \in \mathbb{R},\; x + y = 0 \qquad (\text{假})
$$

它说「存在一个 $y$，加谁都得 $0$」——任何固定的 $y$ 都无法对每个 $x$ 成立。顺序一换，真值翻转。

这种「$\forall$ 在前的 $y$ 可以依赖 $x$，$\exists$ 在前的 $y$ 必须固定」的差别，是微积分里 $\varepsilon$–$\delta$ 定义的骨架。函数 $f$ 在点 $a$ 连续，说的是

$$
\forall \varepsilon > 0,\; \exists \delta > 0,\; \forall x \in \mathbb{R},\; \big(|x - a| \lt  \delta \to |f(x) - f(a)| \lt  \varepsilon\big)
$$

$\delta$ 可以依赖 $\varepsilon$（和 $a$），这是「对每个容忍度都能找到应对」；如果顺序写成 $\exists \delta > 0,\forall \varepsilon > 0$，那就变成「一个 $\delta$ 通吃所有 $\varepsilon$」，是强得多的（也常是错的）要求。学到这里，你已经提前预习了数学分析最精密的读题法。

## 5 否定嵌套量词：一串翻转

否定多层量词时，把上节的法则**逐层套用**：每个量词翻一次，谓词最后落一个 $\neg$。例如：

$$
\neg\Big(\forall x\,\exists y\,(x \lt  y)\Big) \Longleftrightarrow \exists x\,\forall y\,\neg(x \lt  y) \Longleftrightarrow \exists x\,\forall y\,(x \ge y)
$$

「并非（每个 $x$ 都小于某个 $y$）」等于「存在一个 $x$，比所有 $y$ 都大」。**先不管中间步骤，记住终点形态：否定号一路扫过去，$\forall$ 与 $\exists$ 逐个互换。**<span class="marginnote">这也是解证明题时的反向工程法：要证 $\neg(\forall x\,\exists y\,\cdots)$，先把它化简成 $\exists x\,\forall y\,\cdots$，于是你心里清楚<strong>要构造一个怎样的 $x$</strong>。量词否定是把「要证明什么」翻译成「要找什么」的翻译机。</span>

## 6 辨析｜易错点

**易错点一：「并非所有」与「全都不」混为一谈。** 这是全量词否定最经典的坑。用表格钉死：

| 中文表述 | 逻辑形式 | 否定形式 |
| --- | --- | --- |
| 所有学生都到齐了 | $\forall x\,A(x)$ | $\exists x\,\neg A(x)$：至少一个没到 |
| 没有学生缺席 | $\forall x\,\neg A(x)$ | $\exists x\,A(x)$：至少一个缺席 |

「所有都」的否定是「有一个不」，「全都不」的否定才是「有一个」。否定到底落在哪个量词后面，差之毫厘谬以千里。

**易错点二：论域没声明。** 「存在 $x$ 使 $x^2 = -1$」在实数域为假，在复数域为真。同一条开句，论域不同真值不同——写证明时永远先问：$x$ 的论域是什么？<span class="marginnote">正是这个「不够用就扩论域」的冲动，把数系从 $\mathbb{N}$ 一路扩到 $\mathbb{C}$，见《集合的概念》里数系扩充的叙事。量词视角下，这等于在更换量化对象的论域。</span>

**易错点三：以为 $\forall$ 与 $\exists$ 可以随便换序。** 上节已见：$y$ 依赖 $x$ 时 $\forall x\exists y$ 真、$\exists y\forall x$ 可能假。只有连续同种量词才可换序：$\forall x\forall y$ 与 $\forall y\forall x$ 等价，$\exists x\exists y$ 与 $\exists y\exists x$ 等价。

## 7 小结

- **量词**把开句变成命题：$\forall$ 管「所有」，$\exists$ 管「至少一个」；变元有**约束**与**自由**之分。
- **否定法则**：$\neg\forall x\,P(x) \Leftrightarrow \exists x\,\neg P(x)$，$\neg\exists x\,P(x) \Leftrightarrow \forall x\,\neg P(x)$；否定扫过量词时 $\forall \leftrightarrow \exists$ 互换。
- **嵌套量词的顺序有意义**：$\forall x\exists y$ 允许 $y$ 依赖 $x$，$\exists y\forall x$ 要求 $y$ 固定；连续同种量词才可交换。
- 量词否定是**反证法、构造反例、$\varepsilon$–$\delta$