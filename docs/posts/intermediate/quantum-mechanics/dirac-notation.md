---
title: 狄拉克符号
date: 2026-08-07
---

# 狄拉克符号

<div class="epigraph">
<p>bra 和 ket 这两个词，来自 bracket（括号）的拆解——我把「括号」从中间劈开，一半记内积，一半记态。</p>
<footer>—— 保罗 · 狄拉克（Paul Dirac），《量子力学原理》</footer>
</div>

<div class="article-byline">
<p>第二级 · 量子力学 ｜ 曾谨言《量子力学》卷一 第4章 / Griffiths《量子力学概论》§3.3 ｜ 2026-08-07</p>
</div>

## 为什么从狄拉克符号开始

我们已经累积了不少记号：$\Psi(x)$、$\psi_n$、$c_n$、$\langle\psi|\hat{A}|\psi\rangle$……它们能工作，但略显笨重，且总让人忘记「态是矢量、表象是坐标系」。1939 年，狄拉克发明了一套记号，把量子力学的书写彻底简化：用 `|ket⟩` 表示态矢量，用 `⟨bra|` 表示对偶矢量，内积就是 `⟨bra|ket⟩` 的「括号」拼合。**这套记号把「态是抽象的、表象只是坐标」这一思想焊进了符号本身**，从此成为量子力学的通用语言。<span class="marginnote">狄拉克符号的深层价值不是「省几个字」，而是「强制正确」：它把内积、对偶、投影、算符作用全部变成不可误读的拼图。量子计算里的量子比特、量子门（第四级）、量子场论的创生湮灭算符，全都用狄拉克符号书写。学透它，等于学会量子物理的「母语」。</span>

## 1 右矢与左矢

**右矢（ket）**：量子态矢量，记作 $|\psi\rangle$。它就是希尔伯特空间里的矢量本身，与表象无关。

**左矢（bra）**：右矢的对偶矢量，记作 $\langle\psi|$。左矢与右矢的关系是「取复共轭 + 转置」：若 $|\psi\rangle = \sum_n c_n|e_n\rangle$，则

$$
\langle\psi| = \sum_n c_n^*\langle e_n|
$$

<span class="marginnote">左矢与右矢构成「对偶空间」：左矢是「把右矢变成复数的线性函数」。物理上，$\langle\psi|$ 是「测量态 $|\psi\rangle$ 的重叠」这个动作的化身。矩阵语言里，右矢是列向量，左矢是行向量（取复共轭），两者拼成内积恰好是「行乘列」——狄拉克记号是矩阵记号的升华版。</span>左右矢之间的**内积**：

$$
\langle\phi|\psi\rangle = \langle\psi|\phi\rangle^*
$$

它给出两个态的重叠振幅，是复共轭对称的。

## 2 算符与期望值的狄拉克记法

算符 $\hat{A}$ 作用在右矢上：$|\hat{A}\psi\rangle$，简写为 $\hat{A}|\psi\rangle$。左矢的对应写法是 $\langle\hat{A}\psi| = \langle\psi|\hat{A}^\dagger$（取厄米共轭）。**矩阵元**：

$$
\langle\phi|\hat{A}|\psi\rangle = \text{态} \ |\psi\rangle \xrightarrow{\hat{A}} \text{态} \ |\phi\rangle \ \text{的跃迁振幅}
$$

**期望值**是它取 $\phi = \psi$ 的特例：

$$
\langle \hat{A} \rangle = \langle\psi|\hat{A}|\psi\rangle
$$

**厄米性**在狄拉克记号下最简洁：

$$
\langle f|\hat{A}g\rangle = \langle\hat{A}f|g\rangle \quad \Longleftrightarrow \quad \hat{A} = \hat{A}^\dagger
$$

一个常用的技巧是**插入完备性关系**：

$$
\mathbb{I} = \sum_n |e_n\rangle\langle e_n|
$$

把它插进任何表达式的任何位置，都不改变物理内容，却能把态展开成需要的基。这是狄拉克记号最强大的演算工具。<span class="marginnote">「插入一个单位算符」是狄拉克记号里最常用的戏法：比如 $\langle x|\hat{A}|x'\rangle$ 想换成能量表象，就在中间插 $\sum_n|E_n\rangle\langle E_n|$，立刻得到 $\sum_n\langle x|E_n\rangle\langle E_n|\hat{A}|E_n\rangle\langle E_n|x'\rangle$——把「不知道的积分」换成一串「知道的分量乘积」。这种「插入恒等分解」的技巧在微扰论、跃迁计算里反复使用。</span>

## 3 投影算符与测量

狄拉克记号让投影算符变得无比自然：

$$
\hat{P}_n = |e_n\rangle\langle e_n|
$$

它作用于任意态：

$$
\hat{P}_n|\psi\rangle = |e_n\rangle\langle e_n|\psi\rangle
$$

读出两个关键部件：$\langle e_n|\psi\rangle$ 是标量（展开系数），$|e_n\rangle$ 是留下的态——**投影把态「切」到 $|e_n\rangle$ 方向**。投影算符的性质：

$$
\hat{P}_n^2 = \hat{P}_n, \qquad \hat{P}_n^\dagger = \hat{P}_n, \qquad \sum_n \hat{P}_n = \mathbb{I}
$$

<span class="marginnote">这套记号把测量公设压缩成了四行：展开 $|\psi\rangle = \sum_n |e_n\rangle\langle e_n|\psi\rangle$，概率 $P(a_n) = |\langle e_n|\psi\rangle|^2$，坍缩后 $|\psi'\rangle = |e_n\rangle$，重复测量 $\hat{P}_n|\psi'\rangle = |\psi'\rangle$（幂等性）。整个测量理论在狄拉克记号下几乎「自动书写」。</span>完备性 $\sum_n\hat{P}_n = \mathbb{I}$ 也可以写成连续谱版本：$\int|x\rangle\langle x|\,dx = \mathbb{I}$，坐标与动量表象之间的变换正是通过它来建立。

## 4 公式解析：从狄拉克记号回到波函数

把最核心的「抽象 ↔ 具体」换算写清楚：

$$
\langle x | \psi \rangle = \Psi(x), \qquad \langle x | \hat{p} | \psi \rangle = -i\hbar\frac{d\Psi}{dx}
$$

- **第一步，$\Psi(x) = \langle x|\psi\rangle$**：态矢量 $|\psi\rangle$ 在位置本征态上的投影，就是坐标表象的波函数——抽象的右矢通过「插入 $\int|x\rangle\langle x|dx$」回到具体的函数。
- **第二步，$\hat{p}$ 的作用**：动量算符在坐标表象下化为微分算符 $-i\hbar\frac{d}{dx}$。这是狄拉克记号的「坐标化」环节——把抽象算符翻译成熟悉的微分操作。
- **第三步，换表象**：想换到动量表象，插入动量完备性：$\langle x|\hat{p}|\psi\rangle = \int\langle x|p\rangle\langle p|\hat{p}|\psi\rangle dp$，其中 $\langle x|p\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$ 是坐标—动量重叠（平面波）。
- **第四步，几何意义**：全套记号保证了「同一个态 $|\psi\rangle$，坐标下是 $\Psi(x)$、动量下是 $\phi(p)$、能量下是 $c_n$」——狄拉克记号让「换坐标」变成纯粹的代换，物理内容一丝不动。

## 5 狄拉克符号的使用约定

几条使用约定，避免常见的坑：

- **内积方向**：$\langle\phi|\psi\rangle$ 是「左矢配右矢」，不能写 $\langle\psi\rangle|\phi$；$|\psi\rangle\langle\phi|$ 是算符（外积），$\langle\phi|\psi\rangle$ 是标量（内积）——**内外积的位置决定它的身份**。
- **厄米共轭翻转顺序**：$(|\psi\rangle\langle\phi|)^\dagger = |\phi\rangle\langle\psi|$，$(\hat{A}|\psi\rangle)^\dagger = \langle\psi|\hat{A}^\dagger$——共轭时翻转所有顺序。
- **算符在左矢上的作用**：$\langle\psi|\hat{A}$ 表示「先 $\hat{A}$ 作用在右矢上再取共轭」的等价左矢，等价于 $\hat{A}^\dagger|\psi\rangle$ 的共轭，书写时避免歧义。<span class="marginnote">初学最容易混的一处：$\langle\psi|\hat{A}|\psi\rangle$ 里的 $\hat{A}$ 作用在右矢上；而 $\langle\psi\hat{A}|$ 是「$\hat{A}^\dagger$ 作用后的态对应的左矢」。狄拉克记号强调「算符总是作用在右边的 ket 上」，读式子时保持这个习惯即可。</span>

### 易错辨析

**辨析｜易错点：$|\psi\rangle\langle\phi|$（外积）是算符，$\langle\phi|\psi\rangle$（内积）是标量。** 两者记号只差一个「位置」，身份完全不同。外积是「把态射到另一个态的投影」，内积是「重叠振幅」。把外积当标量、内积当算符，是狄拉克记号里最根本的混淆。

**辨析｜易错点：左矢不是右矢的「取共轭」，是「复共轭 + 转置（对偶）」。** 若 $|\psi\rangle = \sum_n c_n|e_n\rangle$，则 $\langle\psi| = \sum_n c_n^*\langle e_n|$——系数取复共轭，基矢变左矢。对列向量而言这正是「取厄米共轭」$|\psi\rangle^\dagger = \langle\psi|$。

**辨析｜易错点：完备性 $\sum_n|e_n\rangle\langle e_n| = \mathbb{I}$ 只能插在「两个右矢/左矢之间」的合适位置。** 常见错误是在 $|\psi\rangle$ 和 $\langle\psi|$ 之间乱插导致重复计数或维度错误。检验：插入恒等分解后，左边缩并掉一个 bra 和一个 ket，总的对象类别不改变。

**辨析｜易错点：算符的厄米共轭翻转「所有因子的顺序」。** $(\hat{A}\hat{B})^\dagger = \hat{B}^\dagger\hat{A}^\dagger$，$(|a\rangle\langle b|)^\dagger = |b\rangle\langle a|$，$(\hat{A}|\psi\rangle)^\dagger = \langle\psi|\hat{A}^\dagger$。顺序翻转是共轭操作的铁律——写错顺序就会得到完全不同的算符。

## 6 小结

- **右矢 $|\psi\rangle$** 表示态矢量，**左矢 $\langle\psi|$** 是它的对偶（复共轭转置），内积 $\langle\phi|\psi\rangle$ 是重叠振幅。
- 期望值 $\langle\hat{A}\rangle = \langle\psi|\hat{A}|\psi\rangle$；厄米性即 $\hat{A} = \hat{A}^\dagger$。
- **完备性关系** $\sum_n|e_n\rangle\langle e_n| = \mathbb{I}$ 是「插入恒等分解」的技巧核心，用于换基与展开。
- **投影算符** $\hat{P}_n = |e_n\rangle\langle e_n|$：幂等、厄米，测量公设四行写完。
- 抽象态与具体波函数的换算：$\Psi(x) = \langle x|\psi\rangle$，$\hat{p}$ 在坐标表象化为 $-i\hbar\frac{d}{dx}$。

在下一节，我们用这套符号回答一个贯穿始终的问题：两个不对易的可观测量，能同时被精确知道吗？答案——**不确定度关系**——即将给出量子力学最著名的定量约束。
