---
title: 群同态基本定理
date: 2026-08-07
---

# 群同态基本定理

<div class="epigraph">
<p>同态像的一切，都藏在原群「压掉核」之后的那一小片里。</p>
<footer>—— 自 题（近世代数课堂笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§4.5 ｜ 2026-08-07</p>
</div>

## 为什么从群同态基本定理开始

前两节我们有了两件武器：同态（把群映射到另一群）与商群（把群按正规子群压缩）。**群同态基本定理（First Isomorphism Theorem）**把它们焊成一条最深刻的等式：

$$
G / \ker f \ \cong \ \operatorname{Im} f
$$

翻译过来：**同态的像，就是原群「压掉核」得到的商群。** 同态「压缩掉核之后剩下的那部分」与「它真正照到的地方」是同一个群。这一定理是群论里最漂亮的统一——它把核、像、商、同构四个核心概念放进一条公式，是后续三个同构定理、环论同态定理、域论 Galois 对应的共同母版。

本节先把定理的陈述与证明讲透，再用三个不同层次的例子展示它的「通用性」：从最简单的 $\mathbb{Z} \to \mathbb{Z}_n$，到矩阵群的行列式同态，再到一个不平凡的构造。理解了同态基本定理，你就抓住了抽象代数「由商而生新结构」这条主线的核心。

## 1 定理的陈述

**定理（群同态基本定理）：** 设 $f : G \to H$ 是群同态，则

1. $\ker f \trianglelefteq G$（核是正规子群——上一节等价条件 4 已证）；
2. $\operatorname{Im} f \le H$（像是子群——第二篇已证）；
3. **$G / \ker f \cong \operatorname{Im} f$**（商群与像同构）。

把第三条展开：存在一个同构

$$
\bar{f} : G / \ker f \longrightarrow \operatorname{Im} f, \qquad \bar{f}(a \cdot \ker f) = f(a)
$$

$\bar{f}$ 由 $f$ 诱导，称为 $f$ 的**诱导同构**。<span class="marginnote">诱导同构的记号 $\bar{f}$ 提醒我们：它把每个陪集 $a \ker f$ 映到「$a$ 的像」$f(a)$。良定义性正是关键验证：若 $a\ker f = b\ker f$，则 $a^{-1}b \in \ker f$，于是 $f(b) = f(aa^{-1}b) = f(a)f(a^{-1}b) = f(a)e_H = f(a)$——同一陪集里的元素共享同一个像，「按陪集分组」不冲突。</span>

## 2 证明：三步走

证明分三段：良定义、同态、双射。

**第一步（良定义）。** 设 $a \ker f = b \ker f$，则 $a^{-1}b \in \ker f$，$f(a^{-1}b) = e_H$，于是 $f(b) = f(a) f(a^{-1}b) = f(a)$。故 $\bar{f}(a\ker f)$ 不依赖代表元 $a$ 的选择。$\checkmark$

**第二步（同态）。**

$$
\bar{f}\big( (a \ker f)(b \ker f) \big) = \bar{f}(ab \ker f) = f(ab) = f(a)f(b) = \bar{f}(a\ker f)\,\bar{f}(b\ker f)
$$

中间每一步都是代入定义：陪集乘法、诱导映射、$f$ 的同态性、再拆回诱导映射。$\checkmark$<span class="marginnote">第二步为什么能「一路畅通」？因为它只是把 $f$ 的同态性质「翻译」到陪集语言上，每换一步都用定义衔接。当你知道要证什么、定义是什么时，这类证明往往是机械的——关键是第一步的良定义，它保证「翻译」不会因为代表元不同而翻车。</span>

**第三步（双射）。** 满射：$\operatorname{Im} f$ 中每个 $f(a)$ 都是 $\bar{f}(a\ker f)$ 的像。单射：若 $\bar{f}(a\ker f) = \bar{f}(b\ker f)$，则 $f(a) = f(b)$，$f(a^{-1}b) = e_H$，$a^{-1}b \in \ker f$，故 $a\ker f = b\ker f$。$\blacksquare$

注意第三步的单射性论证与第一步的良定义性论证**互为镜像**——一个保证「不同代表元不造成混乱」，一个保证「不同陪集不被压在一起」。这正呼应第二篇「$f$ 单射 $\iff \ker f = \{e\}$」：现在 $f$ 不单射时，恰恰是「按核折叠之后」变成单射。

## 3 三个例子：定理的三种用法

**例 1（最平凡也最基础）：模同态。** $f : \mathbb{Z} \to \mathbb{Z}_n$，$f(k) = k \bmod n$。$\ker f = n\mathbb{Z}$，$\operatorname{Im} f = \mathbb{Z}_n$。基本定理给出 $\mathbb{Z} / n\mathbb{Z} \cong \mathbb{Z}_n$——这正是商群的「出身证明」，我们在商群一节已经见过，现在它有名字了。

**例 2（矩阵群）：行列式同态。** $\det : GL_n(\mathbb{R}) \to \mathbb{R}^\ast$，核为 $SL_n(\mathbb{R})$（行列式为 1 的矩阵），像为全体非零实数 $\mathbb{R}^\ast$。基本定理给出

$$
GL_n(\mathbb{R}) / SL_n(\mathbb{R}) \cong \mathbb{R}^\ast
$$

「一般线性群压掉特殊线性群，剩下的正是非零实数」——$SL_n$ 把「行列式信息」压平，商群忠实保留了这个信息。<span class="marginnote">这个例子说明：商群不是抽象的玄学，而是「把不想要的信息压掉、保留想要的信息」的精确工具。$\det$ 把 $GL_n$ 里的矩阵压成单个实数，商群 $GL_n/SL_n$ 就是这单个实数的全体。线性代数里「行列式 ≠ 0 可逆」的全部信息，浓缩在 $\mathbb{R}^\ast$ 里。</span>

**例 3（交换化）：** 换位子群 $[G, G] = \langle aba^{-1}b^{-1} \mid a, b \in G \rangle$ 是 $G$ 的特征子群（从而是正规子群）。自然同态 $G \to G/[G,G]$ 的像自然是 $G/[G,G]$。更深刻的是**泛性质**：任何同态 $f : G \to A$ 到交换群 $A$，都「穿过」交换化——存在唯一同态 $\bar{f} : G/[G,G] \to A$ 使 $f = \bar{f} \circ \pi$。交换化是「$G$ 到交换群的最近投影」，这是同态基本定理思想在泛性质语言中的延伸。<span class="marginnote">「穿过（factor through）」是抽象代数的高频动词：$f$ 穿过 $G/[G,G]$ 意味着 $f$ 的信息完全由 $G/[G,G]$ 承载。这类「万有/泛性质」描述在环论、域论（第十篇）里会反复出现——基本定理是理解泛性质的第一个据点。</span>

## 4 公式解析：G/ker f ≅ Im f 的意义分层

把这条等式从四个层面读透。

- **第一层：作为「大小」的等式。** 有限群时，$|G/\ker f| = |G|/|\ker f|$，而 $|G/\ker f| = |\operatorname{Im} f|$，故 $|G| = |\ker f| \cdot |\operatorname{Im} f|$——**「原群的大小 = 核 × 像」**。这是一个「保底」的计数关系：同态越「胖」（核越大），像就越小；反之亦然。

- **第二层：作为「结构」的等式。** 不只是大小相等，而是**同构**：$G/\ker f$ 与 $\operatorname{Im} f$ 具有完全相同的群结构。它说：同态 $f$ 造成的「信息损失」恰好等于「压掉核」造成的信息损失——不多不少。$f$ 不单射丢掉的，正是核里那部分；$f$ 不满射够不到的，正是像外那部分。

- **第三层：作为「分解」的等式。** 任何同态 $f$ 都能写成「满射 × 同构 × 单射」的三段式：

$$
G \ \xrightarrow{\pi}\ G/\ker f \ \xrightarrow{\ \bar{f}\ }\ \operatorname{Im} f \ \xrightarrow{\ i\ }\ H
$$

其中 $\pi$ 是自然同态（满射）、$\bar{f}$ 是同构、$i$ 是包含映射（单射）。**同态 = 先压、再同构、再嵌入。** 这与线性代数里「线性映射 = 满射 × 同构 × 单射」的分解（秩-零化度定理的几何形态）完全平行——事实上，秩-零化度定理 $\dim V = \dim \ker T + \dim \operatorname{Im} T$ 就是同态基本定理在向量空间里的投影。<span class="marginnote">线性代数的秩-零化度定理是同态基本定理在「加法群 + 线性结构」下的化身：向量空间的商空间 $V/\ker T \cong \operatorname{Im} T$ 与群论版本 $G/\ker f \cong \operatorname{Im} f$ 结构同源。抽象代数与线性代数在这里共用同一张图纸。</span>

**第四层：作为「替换」的等式。** 要研究 $\operatorname{Im} f$ 却嫌它藏在 $H$ 里看不清？换成同构的 $G/\ker f$，它的一切陪集结构都是明摆着的。**同态基本定理允许我们在「像」与「商群」之间自由切换**——哪边好算用哪边。

## 5 应用：判定同构的标准流程

同态基本定理最常见的应用是「证明 $X \cong Y$」：找一个同态 $f$，让 $\ker f$ 与 $\operatorname{Im} f$ 恰好落在 $X$、$Y$ 上。标准流程四步：

1. **构造同态** $f : G \to H$，目标是让 $\operatorname{Im} f = Y$、$G / \ker f = X$；
2. **算核** $\ker f$（或证明它平凡）；
3. **算像** $\operatorname{Im} f$；
4. **套定理**：$G/\ker f \cong \operatorname{Im} f$，即 $X \cong Y$。

**例：** 证明 $S_n / A_n \cong \mathbb{Z}_2$。取 $f = \mathrm{sgn} : S_n \to \{\pm 1\}$。$\ker f = A_n$（偶置换），$\operatorname{Im} f = \{\pm 1\} \cong \mathbb{Z}_2$。由基本定理 $S_n / A_n \cong \{\pm 1\} \cong \mathbb{Z}_2$。$\blacksquare$——这是「交错群指标 2」的同态语言证明，比数元素快得多。

**例：** 证明 $(\mathbb{R}, +) / \mathbb{Z} \cong S^1$（圆周群）。取 $f : \mathbb{R} \to S^1$，$f(x) = e^{2\pi i x}$。$\ker f = \mathbb{Z}$，$\operatorname{Im} f = S^1$。由基本定理 $\mathbb{R}/\mathbb{Z} \cong S^1$——「实数压掉整数，得到圆周」。这条同构在傅里叶分析与拓扑群理论中地位崇高，也是《复变函数与积分变换》里周期函数的代数解释。<span class="marginnote">$\mathbb{R}/\mathbb{Z} \cong S^1$ 是「周期性的代数编码」：实数轴上差一个整数就视为相同，剩下的正是 $[0,1)$ 模 1 的圆周。傅里叶级数里的 $e^{2\pi i n x}$ 一族，本质上是 $S^1$ 上的「对偶群」。这条桥把抽象代数与《数学分析》《复变函数》连了起来。</span>

## 6 小结

- **群同态基本定理**：$G / \ker f \cong \operatorname{Im} f$，诱导同构 $\bar{f}(a\ker f) = f(a)$。
- **证明三步**：良定义（同陪集共享同像）→ 同态（翻译 $f$ 的性质）→ 双射（满射显然、单射与良定义互为镜像）。
- **大小层面**：$|G| = |\ker f| \cdot |\operatorname{Im} f|$；**分解层面**：$f = i \circ \bar{f} \circ \pi$。
- **应用流程**：构造同态 → 算核 → 算像 → 套定理得同构。
- 经典例：$\mathbb{Z}/n\mathbb{Z} \cong \mathbb{Z}_n$、$GL_n/SL_n \cong \mathbb{R}^\ast$、$S_n/A_n \cong \mathbb{Z}_2$、$\mathbb{R}/\mathbb{Z} \cong S^1$。

在下一节，我们给同态基本定理配齐另外两个兄弟：**第一、第二、第三同构定理**。它们分别处理「商群的再商」、「子群与正规子群的对偶」与「对应定理」，把同构思想的工具箱补全。
