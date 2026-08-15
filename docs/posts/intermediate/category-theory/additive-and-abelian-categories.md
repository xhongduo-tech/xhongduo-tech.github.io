---
title: 加性范畴与 Abel 范畴
date: 2026-08-07
---

# 加性范畴与 Abel 范畴

<div class="epigraph">
<p>Abel 范畴是「可以做线性代数的范畴」。</p>
<footer>—— 依据亚历山德鲁 · 格罗滕迪克（Alexandre Grothendieck）</footer>
</div>

<div class="article-byline">
<p>第二级 · 范畴论 ｜ Mac Lane Ch. VIII ｜ 2026-08-07</p>
</div>

## 为什么从加性范畴开始

线性代数的核心本领——求和、数乘、矩阵、核与像——依赖的不是向量空间本身，而是「所有向量空间的态射之间都能相加」这个事实。范畴论把这个事实抽象成**加性范畴（additive category）**，并进一步要求「每个态射都有核与余核」，得到**Abel 范畴（abelian category）**。今天代数几何、代数数论与同调代数讨论的对象——交换群的层、模的复形、凝聚层——全都生活在 Abel 范畴里。<span class="marginnote">机器学习里，向量空间范式的「线性、可加、可核化」正是 Abel 结构直觉；而当你要处理「不是向量空间的对象」时，Abel 范畴给了你一套超越具体载体、只靠抽象性质做线性代数的语言。</span>

## 1 预加性与加性范畴

**预加性范畴（preadditive category）**：每个 $\mathrm{Hom}(a, b)$ 都是交换群（对象是 $0$ 态射），且复合是双线性的：

$$h \circ (f + g) = h \circ f + h \circ g, \qquad (f + g) \circ k = f \circ k + g \circ k$$

**零对象（zero object）**：既是始对象又是终对象的对象，记作 $\mathbf{0}$。在 $\mathbf{Vect}$ 中是零空间。**加性范畴（additive category）**：预加性 + 有零对象 + 每对对象有**双积（biproduct）**。

在预加性范畴里，$\mathrm{Hom}$ 集的群结构一律写成**加法记号**：$0$ 表示零态射，$f + g$ 表示逐点相加。这让「矩阵」成为可能——态射之间的加、减、零在逻辑上先行于一切具体对象，这正是后面同调代数里复形条件 $d \circ d = 0$ 能成立的前提。<span class="marginnote">对比 $\mathbf{Set}$：那里两个函数之间没有自然的「和」，所以「$f + g$」无从谈起；预加性把「可以相加」变成范畴的一种结构性质，而不是某个对象的偶然能力。</span>

一旦有双积，态射就可以「排成矩阵」：$\mathrm{Hom}(a_1 \oplus a_2,\ b_1 \oplus b_2)$ 的元素一一对应到 $2 \times 2$ 分块 $\begin{pmatrix} f_{11} & f_{12} \\ f_{21} & f_{22} \end{pmatrix}$，其中 $f_{ij}: a_j \to b_i$，而复合就是矩阵乘法。这是「线性代数语言在任意加性范畴里都能用」的最直观证据——也是后面谱序列、导出函子的记号基础。

**双积（biproduct）** $a \oplus b$ 同时是乘积与余积，由四态射 $i_1, i_2, p_1, p_2$ 及其关系唯一刻画（见公式解析）。<span class="marginnote">在 $\mathbf{Ab}$、$\mathbf{Vect}$、$\mathbf{Mod}_R$ 里直和 = 直积，双积的存在说明「乘积与余积在这个范畴里长一个样」——这是线性代数「直和/直积不区分」的范畴论根源。</span>

## 2 核、余核与正合列

在加性范畴上引入分析工具：

**核（kernel）**$\ker f$：$f: a \to b$ 的核是「被 $f$ 杀死」的最大的对象，配态射 $k: \ker f \to a$ 满足 $f k = 0$ 且对一切满足 $f k' = 0$ 的 $k'$ 有唯一分解。
- **余核（cokernel）**$\mathrm{coker}\, f$：对偶概念，是「把 $f$ 的像压掉」得到的商。
- 于是可以定义**像（image）**$\mathrm{im}\, f = \ker(\mathrm{coker}\, f)$。

**正合列（exact sequence）**：$A \xrightarrow{f} B \xrightarrow{g} C$ 满足 $\ker g = \mathrm{im}\, f$。短正合列 $0 \to A \to B \to C \to 0$ 把「$B$ 由 $A$ 与 $C$ 拼成」这件事精确化。<span class="marginnote">同调代数的全部动力来自正合列：长正合列、蛇形引理、五引理——都是「在 Abel 范畴里做线性代数」的推论。层上同调、导出函子、谱序列都建立其上。</span>

**辨析｜易错点：** 在一般范畴里「单态射」不等于「内射」，「满态射」不等于「满射」。Abel 范畴之所以重要，正是因为它提供**核、余核、像**这套线性结构，让「像 = 核」这类等式有意义；而在普通范畴里这些概念根本无从谈起。

**数值算例：在 $\mathbf{Vect}_{\mathbb{R}}$ 里算核与余核。** 取 $f: \mathbb{R}^2 \to \mathbb{R}^2$，$f(x, y) = (x, 0)$（向 $x$ 轴的投影）。$\ker f = \{(0, y)\}$ 是 $y$ 轴这条一维子空间，$\mathrm{im}\, f = \{(x, 0)\}$ 是 $x$ 轴；余核 $\mathrm{coker}\, f = \mathbb{R}^2 / \mathrm{im}\, f$ 商掉 $x$ 轴后同构于 $\mathbb{R}$。

$$0 \to \ker f \to \mathbb{R}^2 \xrightarrow{q} \mathrm{coker}\, f \to 0$$

短正合列「$\mathbb{R}^2$ 由两条一维线拼成」由此被精确写下来：每个向量 $(x, y)$ 唯一分解为 $(x, 0) + (0, y)$，且这条正合列是**分裂的**——存在回缩 $s: \mathbb{R} \to \mathbb{R}^2$，$s(t) = (0, t)$ 使 $q \circ s = 1$。<span class="marginnote">一般结论：$0 \to A \to B \to C \to 0$ 分裂当且仅当 $B \cong A \oplus C$。机器学习里残差块 $y = x + f(x)$ 正是这种分裂结构——恒等回线保证信息不丢失。</span>

**实例对照：哪些范畴是 Abel 的？** 「能做线性代数」不是一句空话，下表把它落实成「预加性 / 加性 / Abel」三张检查表：

| 范畴 | 预加性 | 加性 | Abel |
| --- | --- | --- | --- |
| $\mathbf{Set}$ | ✗ | ✗ | ✗ |
| $\mathbf{Grp}$（非交换群） | ✗ | ✗ | ✗ |
| $\mathbf{Ab}$ | ✓ | ✓ | ✓ |
| $\mathbf{Vect}_k$ | ✓ | ✓ | ✓ |
| $\mathbf{Mod}_R$ | ✓ | ✓ | ✓ |
| $\mathbf{Sh}(\mathcal{T})$ 层范畴 | ✓ | ✓ | ✓ |

关键的否定是前两行：$\mathbf{Set}$ 里根本没有「态射相加」，$\mathbf{Grp}$ 里两个群同态的「逐点积」不再是同态（除非目标群交换）。这两条失败恰好说明**加法与 Abel 结构是一份稀缺财产**，而不是所有范畴的标配。

## 3 Abel 范畴

**Abel 范畴（abelian category）**：加性范畴 + 每个态射都有核与余核 + **正则性**：每个单态射都是其余核的核，每个满态射都是其核的余核。

这三条看似技术，实则保证：单态射 = 内射（在核的语言下）、满态射 = 满射（在余核的语言下）、且 $\mathrm{im}$ 与 $\mathrm{coim}$ 一致。<span class="marginnote">Grothendieck 在 1957 年《论某些代数几何范畴》里系统研究了加性范畴的若干公理（AB1–AB5），把「足够好」的 Abel 范畴推上代数几何与同调代数的舞台。</span>典型实例：**$\mathbf{Ab}$、$\mathbf{Vect}_k$、$\mathbf{Mod}_R$、域上范畴 $\mathbf{Sh}(\mathcal{T})$ 上的层范畴**都是 Abel 范畴，而 $\mathbf{Set}$、$\mathbf{Grp}$（非交换）**不是**。

## 4 公式解析：双积的特征

双积存在与否，是加性范畴与普通范畴的分水岭。$X$ 是 $a$ 与 $b$ 的双积，当且仅当存在态射 $i_1: a \to X$、$i_2: b \to X$、$p_1: X \to a$、$p_2: X \to b$ 满足五条方程：

$$
p_1 i_1 = 1_a,\ \ p_2 i_2 = 1_b,\ \ i_1 p_1 + i_2 p_2 = 1_X,\ \ p_1 i_2 = 0,\ \ p_2 i_1 = 0
$$

- **第一步，两条「还原律」**：$p_j i_j = 1$ 保证「塞进去再取出来」是恒等——$i_j$ 是分裂单态射、$p_j$ 是分裂满态射。
- **第二步，交叉项为零**：$p_1 i_2 = 0$、$p_2 i_1 = 0$——一个因子的内容不会泄露进另一个。
- **第三步，分解恒等式**：$i_1 p_1 + i_2 p_2 = 1_X$——$X$ 中每个元素可拆成「第一分量 + 第二分量」，且这个拆分一致。
- **第四步，线性代数的回响**：这五条方程正是「$a \oplus b$ 里的元素写成 $(x, y)$、态射写成 $2 \times 2$ 分块矩阵」的范畴论翻译。一旦满足，$X$ 自动既是乘积又是余积。

## 5 为什么 Abel 范畴是必要的地基

有了 Abel 范畴，才能谈论以下一切：

- **正合性与长正合列**：把短正合列拉长成链，是导出函子（$\mathrm{Ext}$、$\mathrm{Tor}$、上同调）的起点；
- **复形与同调**：$\ker d / \mathrm{im}\, d$ 在 Abel 范畴里有意义，层上同调由此展开；
- **函子的正合性**：右伴随保极限、左伴随保余极限在 Abel 范畴里化为「$\mathrm{Hom}$ 的 $\lim^1$」等精细工具。<span class="marginnote">机器学习里的「分解—重建」「编码—解码」若在 Abel 范畴上表述，正合列给出「信息是否无损」的精确判据——重构误差为零等价于某个正合列分裂。</span>

**辨析｜易错点：** 不要误以为「Abel 范畴 = 交换群范畴的推广所以必然关于函子良态」。函子把正合列映成正合列（左/右/正合）是**额外性质**，不是免费午餐；这是同调代数花费一整章讨论的问题。

**一个小实验：在 $\mathbf{Vect}$ 里算 $\mathrm{Ext}^1$。** 短正合列 $0 \to A \to B \to C \to 0$ 的「非分裂程度」由 $\mathrm{Ext}^1(C, A)$ 度量：当 $C = A = \mathbb{R}$ 时 $\mathrm{Ext}^1(\mathbb{R}, \mathbb{R}) = 0$，说明所有以 $\mathbb{R}$ 为核、以 $\mathbb{R}$ 为余核的扩张都分裂——维数可加性保证中间层必同构于 $\mathbb{R}^2$。而在 $\mathbf{Ab}$ 中 $\mathrm{Ext}^1_{\mathbb{Z}}(\mathbb{Z}/2, \mathbb{Z}) \cong \mathbb{Z}/2$ 非零：$\mathbb{Z}/2$ 的扩张确有真分歧。这个对比正是 Abel 范畴让「同调代数可计算」的原因。

**正合函子的数值检验。** 右正合与左正合是两种不同性质：$-\otimes_{\mathbb{Z}} \mathbb{Z}/2$ 是右正合而非左正合——对短正合列 $0 \to \mathbb{Z} \xrightarrow{2} \mathbb{Z} \to \mathbb{Z}/2 \to 0$ 张量 $\mathbb{Z}/2$，中间的「乘以 2」变成零映射，正合性在左端断掉，而 $\mathrm{Tor}^{\mathbb{Z}}_1(\mathbb{Z}/2, \mathbb{Z}/2) \cong \mathbb{Z}/2$ 恰好度量了这个「断」。

## 6 术语速查表

| 术语 | 英文 | 一句解释 |
| --- | --- | --- |
| 预加性范畴 | preadditive category | $\mathrm{Hom}$ 集是交换群且复合双线性 |
| 零对象 | zero object | 既是始对象又是终对象；$\mathbf{Vect}$ 中是零空间 |
| 双积 | biproduct | 同时是乘积与余积，由五条方程刻画 |
| 核 | kernel | 被 $f$ 杀死的最大的对象 |
| 余核 | cokernel | 把像压掉得到的商对象 |
| 像 | image | $\mathrm{im}\, f = \ker(\mathrm{coker}\, f)$ |
| 正合列 | exact sequence | $\ker g = \mathrm{im}\, f$ 的态射链 |
| 分裂正合列 | split exact | $B \cong A \oplus C$ 的特殊情形 |
| Abel 范畴 | abelian category | 加性 + 核/余核存在 + 正则性 |

## 7 小结

- **预加性范畴**：$\mathrm{Hom}$ 集是交换群，复合双线性；**加性范畴** = 预加性 + 零对象 + 双积。
- **双积**由五条方程刻画，同时是乘积与余积——直和 = 直积。
- **Abel 范畴** = 加性 + 核/余核存在 + 正则性（单态射 = 核、满态射 = 余核）。
- **正合列**、复形、同调都在 Abel 范畴里有意义；$\mathbf{Ab}$、$\mathbf{Vect}$、$\mathbf{Mod}_R$、层范畴都是，$\mathbf{Set}$、$\mathbf{Grp}$ 不是。
- 导出函子与上同调的地基，正建立在 Abel 范畴之上。

在下一节，我们走出「具体对象」回到「表示」本身：**Yoneda 引理**将告诉你，任何对象都可以完全由它到所有其他对象的态射来认识——这是整个范畴论的支点。
