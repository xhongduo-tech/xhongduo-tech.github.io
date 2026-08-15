---
title: 辛几何与几何表示论的桥梁
date: 2026-08-07
---

# 辛几何与几何表示论的桥梁

<div class="epigraph">
<p>表示论研究对称，辛几何研究守恒——moment map 把前者交给后者。</p>
<footer>—— 纪尧姆 · 洛姆（Guillaume Laumon）教学传统</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ Cannas 第11章；McDuff & Salamon 第1章 ｜ 2026-08-07</p>
</div>

## 为什么从几何表示论开始

这一篇是辛几何课程的收官：把前面全部装备——moment map、辛约化、Hamiltonian 群作用、Lagrangian 子流形——投向一个更古老的数学王国：**表示论**。表示论研究「群如何作用在向量空间上」，而辛几何研究「群如何守恒地作用在流形上」；**几何表示论**（geometric representation theory）的核心想法正是用「哈密顿群作用 + 辛约化」来构造和研究表示。三条大动脉：**轨道方法**（Kirillov：共伴随轨道是辛流形，表示 = 几何量子化）、**Borel-Weil 定理**（旗簇上的线束截面给出不可约表示）、以及 **Nakajima 箭图簇**（用辛约化构造 Kac-Moody 表示）。这一篇让读者看见：辛几何不是孤岛，而是连接代数几何、表示论与数学物理的立交桥。<span class="marginnote">在课程地图上：这是全专题的终篇，把可积系统篇（环面纤维化）、moment map 篇（约化）、几何量子化篇（线束截面）全部调用起来。它也是「从极限到大模型」里「对称 → 结构」这一主线的数学版本。</span>

## 1 轨道方法：共伴随轨道与辛结构

**轨道方法（orbit method）**（Kirillov）是「用辛几何构造酉表示」的纲领：对李群 $G$，考虑其在对偶李代数 $\mathfrak{g}^*$ 上的**共伴随作用**，轨道

$$
\mathcal{O}_\lambda = \{ \mathrm{Ad}^*_g \lambda : g \in G \} \subset \mathfrak{g}^*
$$

**基本定理（Kirillov-Kostant-Souriau）**：每个共伴随轨道 $\mathcal{O}_\lambda$ 都是**辛流形**，其辛形式由 Kirillov 形式

$$
\omega_\mathcal{O}(\xi_M, \eta_M) = \langle \lambda, [\xi, \eta] \rangle
$$

给出（$\xi_M, \eta_M$ 是无穷小生成元）。<span class="marginnote">验证辛性的关键：$\omega_\mathcal{O}$ 的闭性来自 <strong>Jacobi 恒等式</strong>（李代数的），非退化性来自「轨道 = 迷向子群的齐次空间」。所以「群表示的空间」自动带辛结构——这是轨道方法的地基。</span>

**轨道方法的信条**：$G$ 的（酉）表示对应「整」共伴随轨道 + 其上的几何量子化。$\mathcal{O}_\lambda$ 上的几何量子化（前篇）给出表示空间。**表示论问题 ⟶ 辛几何问题**。

**例（$SU(2)$）**：$\mathfrak{su}(2)^* \cong \mathbb{R}^3$，共伴随轨道是球面 $S^2_{\|\lambda\|}$（半径由 $\lambda$ 决定）。球面是辛流形（面积形式），几何量子化给出自旋 $j = \|\lambda\|/2$ 的表示。**自旋表示 = 球面上的几何量子化**——这是轨道方法最干净的例。

## 2 Borel-Weil 定理：旗簇上的线束

**旗簇（flag variety）**：$G$ 的旗簇 $G/B$（$B$ 是 Borel 子群）是复流形，如 $\mathbb{CP}^n = \mathrm{GL}_{n+1}/B$。它上面有 $G$ 作用，且是**辛流形**（配 Kähler 形式，即 Fubini-Study 型）。

**Borel-Weil 定理**：$G$ 的（正则）不可约表示 $V_\lambda$ 可实现为旗簇上某个全纯线束 $L_\lambda$ 的**全纯截面空间**：

$$
V_\lambda \cong H^0(G/B, L_\lambda)
$$

**几何量子化的语言**：$V_\lambda$ = Kähler 量子化（$G/B$, 线束 $L_\lambda$）的态空间。<span class="marginnote">Borel-Weil 把「表示论」变成「线束截面的几何」——这是几何量子化（前篇）在齐次空间上的完美实现：辛流形 $G/B$ + 线束 = 表示。Bott 的推广（Borel-Weil-Bott）用更高的 Dolbeault 上同调覆盖所有表示。</span>

**为什么旗簇是辛的**：$G/B$ 是共伴随轨道（$G/B \cong \mathcal{O}$，通过「稳定子」对应）。所以 Borel-Weil 其实是「轨道方法 + 几何量子化」在旗簇上的具体化——**三条线在这里会师**。

## 3 辛约化与表示论：Nakajima 箭图簇

**Nakajima 箭图簇（quiver variety）**：给箭图（quiver：顶点 + 箭头）配「表示维数」，构造

$$
\mathcal{M}(v, w) = \mu^{-1}(0)/G
$$

其中 $\mu$ 是哈密顿约化的 moment map，$G$ 是「基底变换群」（各顶点上的 $\mathrm{GL}$ 乘积）。**箭图簇 = 辛约化的实例**，它把箭图表示论几何化。<span class="marginnote">箭图簇是「用辛约化造表示」的巅峰：$v$、$w$ 是维数向量，$\mathcal{M}(v,w)$ 是带辛结构的代数簇。Nakajima 证明其同调上 $K$-理论给出 Kac-Moody 李代数的表示——<strong>表示空间从「辛约化的几何」长出来</strong>。</span>

**Nakajima 定理**：Kac-Moody 代数 $\mathfrak{g}$ 的不可约最高权表示由箭图簇 $\mathcal{M}(v,w)$ 的（等变）同调实现：

$$
V(\Lambda) \cong \bigoplus_v H_*(\mathcal{M}(v,w))
$$

**生成元与关系**：升降算子由「拉格朗日子流形/对应（correspondence）的积分」给出——**Lagrangian 子流形（第1篇）在这里成为表示论算子的几何载体**。<span class="marginnote">这就是「辛几何与几何表示论的桥梁」最具体的表现：表示空间的基由「箭图簇的中维（Lagrangian）同调」给出，算子由「Lagrangian 对应」作用——第1篇的 Lagrangian 子流形概念直接进入表示论核心。</span>

**Coulomb 分支**：箭图簇的「对偶」构造（Coulomb 分支，Braverman-Finkelberg-Nakajima）同样由辛约化出发，给出 3d 镜像对称下的另一组表示——是当代最活跃的方向。

## 4 公式解析：Kirillov 形式

**核心公式：**

$$
\omega_\mathcal{O}(\xi_M, \eta_M) = \langle \lambda, [\xi, \eta] \rangle
$$

拆解：

- **第一步，读两边**：左边是共伴随轨道 $\mathcal{O}_\lambda$ 上的 2-形式作用在两个无穷小生成元上；右边是李代数括号 $[\xi,\eta]$ 与 $\lambda$ 的配对。**左边「几何量」被右边「代数量」定义**——辛结构来自李代数。
- **第二步，验证反对称**：$\langle\lambda, [\xi,\eta]\rangle = -\langle\lambda, [\eta,\xi]\rangle$（括号反对称）⇒ $\omega$ 反对称。✓
- **第三步，验证非退化**：$\omega_\mathcal{O} = 0$（对所有 $\eta$）⟺ $\lambda$ 在 $\xi$ 方向不变 ⟺ $\xi$ 属于 $\lambda$ 的迷向子群。轨道上「迷向 = 零」，所以非退化。**关键：轨道切空间与迷向子群商**。
- **第四步，验证闭性**：$d\omega_\mathcal{O} = 0$ 等价于 **Jacobi 恒等式** $\sum_{\mathrm{cyc}} [\xi, [\eta, \zeta]] = 0$——李代数的核心恒等式在这里变成「辛形式闭」的几何条件。

**直觉总结：** Kirillov 形式是「用李括号造辛形式」的模板。**表示论里的代数（李括号）直接决定几何（辛结构）**——这就是「几何表示论」名字的含义：把代数的结构读成几何的对象。

## 5 统一图景：moment map 的三重身份

收官之际，值得盘点 **moment map**（第2篇）在表示论中的三重身份——它是贯穿全课程的「万能接口」：

1. **力学**：moment map = 守恒量（角动量、动量）——Noether 定理（哈密顿向量场篇）；
2. **几何**：moment map 的纤维化 → 辛约化（moment map 篇）→ Delzant 多胞形（环面作用篇）；
3. **表示论**：moment map 的轨道 = 共伴随轨道（轨道方法）；约化 = 箭图簇（Nakajima）；旗簇 = 约化的特例。

**一条主线**：**哈密顿群作用 + moment map = 表示论的几何工厂**。可积系统、量子化、镜面对称、表示论——全部从这里「长」出来。

**辨析｜易错点：** 几何表示论的「几何」有两层：一是「用几何对象构造表示」（轨道、线束、箭图簇），二是「用同调/拓扑工具研究表示」（等变上同调、K 理论）。初学者易混淆「表示本身」（向量空间）与「表示的实现」（几何构造）——**Borel-Weil 给出实现，Nakajima 给出实现**，而表示论问题（分解、张量、分支）在这些实现里获得几何证明。

**Springer 理论与几何 Satake**：这条桥梁还有两支重要延伸，值得记住名字。**Springer 理论**把 Weyl 群表示实现为「旗簇上等变相交上同调」的分解——用代数几何的层论工具研究表示，其核心对象（Springer 纤维）正是辛约化的特例。**几何 Satake 等价**（Lusztig、Ginzburg、Mirković-Vilonen）把「群的表示范畴」实现为「仿射格拉斯曼流形上等变层的范畴」，是朗兰兹纲领的几何基石——**表示论、辛几何与代数几何在朗兰兹纲领处完全合一**。<span class="marginnote">这些主题各有一整片天地，这里只需记住：<strong>辛约化与 moment map 是它们共同的技术起点</strong>。对想深入几何表示论的读者，Nakajima 的《Lectures on Hilbert schemes of points on surfaces》与 Chriss-Ginzburg《Representation Theory and Complex Geometry》是经典入口。</span>

**终篇回望**：从第1篇的「反对称双线性形式」到这里的「几何 Satake」，辛几何走完了一条从「线性代数」到「朗兰兹纲领」的弧线。贯穿始终的只有三样东西：**辛形式（面积）、moment map（守恒）、Lagrangian（自正交）**——它们在不同语境里反复变形，却始终是同一个几何。

**给读者的下一步**：若这一系列点燃了你对辛几何的兴趣，标准进阶路线是 McDuff-Salamon《Introduction to Symplectic Topology》第三版（本专题的母本）配合 Cannas da Silva《Lectures on Symplectic Geometry》。再往前，Fukaya 的 $A_\infty$ 范畴、导出范畴与 Kähler 几何，以及 Audin《Torus Actions on Symplectic Manifolds》都是成熟的进阶路径——**把今天的「桥梁」走成你自己的「高速公路」**。

## 6 小结

- **轨道方法（Kirillov）**：共伴随轨道 $\mathcal{O}_\lambda$ 是辛流形（Kirillov 形式）；表示 = 轨道上的几何量子化。$SU(2)$ 的自旋表示 = 球面 $S^2_{\|\lambda\|}$ 上的量子化。
- **Borel-Weil 定理**：不可约表示 $V_\lambda \cong H^0(G/B, L_\lambda)$——表示 = 旗簇上全纯线束的截面空间。
- **Nakajima 箭图簇**：$\mathcal{M}(v,w) = \mu^{-1}(0)/G$ 是辛约化的实例；其同调给出 Kac-Moody 表示，Lagrangian 对应充当算子。
- **moment map 三重身份**：力学（守恒量）、几何（约化 → Delzant 多胞形）、表示论（共伴随轨道 / 箭图簇）——贯穿全课程。
- **延伸**：Springer 理论与几何 Satake 把表示论、辛几何、代数几何在朗兰兹纲领处合一。
- **方法论**：哈密顿群作用 + moment map = 表示的几何工厂；要区分「表示本身」与「表示的实现」（Borel-Weil、Nakajima 都给出实现）。

我们走到「辛几何」专题的最后一篇。回望这条弧线：第1篇的辛线性代数与 Darboux 定理立住「局部平凡、整体刚性」；第2篇的哈密顿流、可积系统与 moment map 把力学与对称性变成几何；第3篇的 Gromov 曲线把刚性变成计数；第4篇的 Floer 同调、量子上同调、镜面对称与几何表示论，把这一切织成当代数学的立交桥。

如果你跟着走到了这里，接下来最自然的动作是翻开 McDuff–Salamon《Introduction to Symplectic Topology》第三版，把这一系列文章里的每个定理亲自证明一遍——**今天读懂的所有桥梁，都会成为你日后自己的高速公路**。