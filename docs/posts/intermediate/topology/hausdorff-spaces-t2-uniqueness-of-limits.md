---
title: Hausdorff 空间（T2）：极限唯一性
date: 2026-08-07
---

# Hausdorff 空间（T2）：极限唯一性

<div class="epigraph">
<p>在 Hausdorff 空间里，极限只有一个家。</p>
<footer>—— 对 Hausdorff 性与极限唯一性的概括</footer>
</div>

<div class="article-byline">
<p>第二级 · 拓扑学 ｜ 尤承业《基础拓扑学讲义》第五章 ｜ Munkres《Topology》§17、§31 ｜ 2026-08-07</p>
</div>

## 为什么 Hausdorff 性是「默认配置」

分析课程里「极限唯一」被视为理所当然，但它其实依赖一个拓扑假设：空间是 **Hausdorff（T2）** 的。Hausdorff 空间要求「任意两点有互不相交的开邻域」——比 T1 的「互避」强在「不交」。这条看似温和的公理带来一系列深远的后果：极限唯一、紧集自动闭、连续双射自动同胚（在紧定义域下）、单点紧化保持 Hausdorff……几乎一切「好空间」的默认配置都是 Hausdorff。理解 Hausdorff 性，就是理解「为什么我们能放心地谈极限」的拓扑根源。<span class="marginnote">「两点有不交开邻域」——T2 是分离公理阶梯上最常用的一级。几乎所有数学中的「标准空间」（度量空间、流形、解析簇的通常拓扑）都是 Hausdorff；而 Zariski 拓扑、Sierpiński 空间等「病态」空间恰恰不是。</span>

## 1 Hausdorff 空间的定义

**Hausdorff 空间（T2）**：拓扑空间 $X$，若任意两个不同点 $x \neq y$，都存在互不相交的开邻域 $U \ni x$、$V \ni y$（$U \cap V = \emptyset$），则称 $X$ 是 **Hausdorff** 空间。

与 T1 对比：

- **T1**：$x$ 有避 $y$ 的邻域、$y$ 有避 $x$ 的邻域——两个邻域不必不交。
- **T2**：$x$ 与 $y$ 有**互不相交**的开邻域——两个邻域彻底隔开。

T2 比 T1 强：$U \cap V = \emptyset$ 蕴含「$y \notin U$ 且 $x \notin V$」。于是蕴含链

$$T2 \Longrightarrow T1 \Longrightarrow T0$$

看例子：

- **度量空间都是 Hausdorff**：$d(x,y) = \delta > 0$，取 $U = B_{\delta/2}(x)$、$V = B_{\delta/2}(y)$，两球不交（三角不等式）。
- **$\mathbb{R}^n$、$S^n$、流形**：Hausdorff（作为度量空间的子空间）。
- **余有限拓扑（无限集）**：**不是 Hausdorff**——两个开集都是「去掉有限个点」，必然相交。它是 T1 但不是 T2。
- **Zariski 拓扑**（代数几何）：不是 Hausdorff——开集是「多项式零点集的补」，两个非空开集总相交。这解释了代数几何里「点不能分离」的特殊现象。<span class="marginnote">Zariski 拓扑是「T1 不 T2」的深刻例子：代数簇上的开集太少（余有限多项式零点集），任意两个开集都相交。代数几何被迫发展出「泛点」「层」等工具来弥补「不能分离」的缺陷——这是 Hausdorff 性缺席的代价。</span>

## 2 极限唯一性定理

Hausdorff 性最重要的推论是**极限唯一性**。先给网（net）的版本，它适用于任意拓扑空间：

**定理（极限唯一）**：$X$ Hausdorff，网 $(x_\lambda)$ 收敛，则极限**唯一**。

证明：设 $(x_\lambda) \to x$ 且 $(x_\lambda) \to y$，$x \neq y$。Hausdorff 性给不交开邻域 $U \ni x$、$V \ni y$。由收敛定义，最终 $x_\lambda \in U$ 且最终 $x_\lambda \in V$——但 $U \cap V = \emptyset$，不可能同时。矛盾，故 $x = y$。∎<span class="marginnote">证明只有四行：两个极限给两套「最终进入」，Hausdorff 给两个不交邻域，「最终进入」叠加后自相矛盾。关键是「最终」两条同时成立——收敛定义保证这一点。</span>

序列版本（度量/第一可数空间）：$x_n \to x$ 且 $x_n \to y$ ⟹ $x = y$。这正是分析里「极限唯一」的严格表述。

**反例对照**：非 Hausdorff 空间里极限可以不唯一。取平凡拓扑的多点空间，任何序列收敛到任何点。取 Sierpiński 空间，序列可以「双收敛」。所以「极限唯一」不是拓扑的默认事实，是 Hausdorff 性的礼物。<span class="marginnote">「非 Hausdorff ⟹ 极限可不唯一」用平凡拓扑就够：所有序列收敛到所有点。「极限唯一」的默认成立，掩盖了 Hausdorff 性的存在——本课把它揭开。</span>

## 3 Hausdorff 性带来的「好结论」清单

回顾前面学过的、依赖 Hausdorff 性的结论，把它们集中起来：

- **紧集闭**：Hausdorff 空间中紧子集是闭的（第四篇）——「紧 ⟺ 闭」在紧 Hausdorff 空间的基石。
- **连续双射同胚**：$X$ 紧、$Y$ Hausdorff、$f$ 连续双射 ⟹ $f$ 同胚（第四篇）——同胚判据。
- **极限唯一**：本课——收敛的唯一性。
- **单点紧化保持 Hausdorff**：$X$ 局部紧 Hausdorff ⟹ $X^*$ 紧 Hausdorff（第四篇）——紧化的「体面」。
- **收敛的子列唯一**（度量）：列紧收敛的唯一极限。

这张清单说明：**Hausdorff 性是「极限/紧集/同胚」这些核心概念的幕后保证**。没有它，前面很多定理都要失效。

## 4 公式解析：T2 的定义式

$$X \text{ Hausdorff} \iff \forall x \neq y,\ \exists U, V \text{ 开},\quad x \in U,\ y \in V,\ U \cap V = \emptyset$$

- **第一步，读量词**：对每对不同的点（$\forall x \neq y$），要能找到（$\exists U, V$）两个开集。顺序是「每对点 → 存在邻域」，不能换。
- **第二步，读条件**：$U$ 包 $x$、$V$ 包 $y$、$U \cap V = \emptyset$——三点合起来是「彻底隔开」。互不相交是关键，比 T1 的「互避」强。
- **第三步，对比 T1**：T1 只要求「$y\notin U$ 且 $x \notin V$」（互避）；T2 要求「$U \cap V = \emptyset$」（不交）。「互避」允许邻域在别处重叠，「不交」彻底杜绝。

## 5 辨析｜易错点（续）

**辨析｜易错点：** Hausdorff 性有四个高频误区：

- **T1 ≠ T2**：余有限拓扑（无限集）是 T1 非 T2。判定 T2 必须找「互不相交」的开邻域。
- **「极限唯一」只在 Hausdorff 成立**：非 Hausdorff（平凡拓扑、Sierpiński）里极限不唯一。分析里「极限唯一」的直觉不能移植到一般拓扑。
- **Hausdorff 不保证「正则/正规」**：T2 只分离「点与点」，不分离「点与闭集」「闭集与闭集」——那是 T3、T4 的职责。别把 T2 当分离能力的终点。
- **Hausdorff 是拓扑性质**：同胚保持 Hausdorff。判别不同胚时，$X$ Hausdorff 而 $Y$ 不 ⟹ 不同胚（如 $\mathbb{R}$ 与 Zariski 拓扑下的直线）。

## 6 小结

- **Hausdorff（T2）**：任意两点有互不相交的开邻域。
- **蕴含链**：T2 ⟹ T1 ⟹ T0；余有限拓扑 T1 非 T2。
- **极限唯一**：Hausdorff 空间中收敛网/序列极限唯一——分析「极限唯一」的拓扑根源。
- **好结论清单**：紧集闭、连续双射同胚、单点紧化保 Hausdorff。
- **反例**：度量空间 Hausdorff；Zariski 拓扑、平凡拓扑非 Hausdorff。
- **易错**：T1 ≠ T2；「极限唯一」依赖 Hausdorff。
- **记忆锚点**：度量空间 Hausdorff（三角不等式）；余有限拓扑 T1 非 T2；Zariski 拓扑非 Hausdorff。
- **连续函数图闭**：$Y$ Hausdorff ⟹ 连续函数 $f:X\to Y$ 的图是 $X\times Y$ 的闭集——「图像闭」的拓扑根据。
- **判定练习**：$X$ 是二点空间（平凡拓扑），$X$ 是否 Hausdorff？否——两点无不交开邻域，极限不唯一。
- **与紧致性的联动**：Hausdorff + 紧 ⟹「紧 ⟺ 闭」——本课与第四篇的接口。
- **本课一句话**：Hausdorff 性是「极限唯一」的拓扑保证，也是「紧集闭」「图闭」的幕后功臣。
- **自测**：度量空间 Hausdorff 的证明靠三角不等式；余有限拓扑为何非 T2？——任意两个开集都相交。
- **Hausdorff 在积空间**：$X, Y$ Hausdorff ⟹ $X \times Y$ Hausdorff（逐坐标分离，矩形基元）；逆也真（投影）。
- **Hausdorff 在子空间**：Hausdorff ⟹ 子空间 Hausdorff（开集限制）——与正规（T4 不继承）形成对照。
- **本课小结**：T2 是「极限唯一 + 紧集闭 + 图闭」的幕后保证，也是分离公理最常用的一级。

在下一节，我们将攀登下一级：**正则空间（Regular, T3）**，它要求点与闭集能分离。

### 极限唯一性的反例推演

把「非 Hausdorff ⟹ 极限不唯一」推演到具体例子，能彻底钉死这个概念：

- **平凡拓扑**（$X$ 至少两点）：任何序列都收敛到任何点。$x_n \to x$ 且 $x_n \to y$（$x \neq y$）——极限完全不唯一。
- **Sierpiński 空间**（$X = \{a,b\}$，开集 $\emptyset, \{a\}, \{a,b\}$）：常值序列 $x_n = a$ 收敛到 $a$（邻域 $\{a\}, \{a,b\}$ 都含 $a$）；它也收敛到 $b$ 吗？$b$ 的邻域只有 $\{a,b\}$，含 $a$，故 $x_n \to b$ 也成立。于是 $x_n \to a$ 且 $x_n \to b$。
- **余有限拓扑**：序列 $x_n \to x$ 当且仅当「几乎处处等于 $x$」或「几乎处处互异且逃逸」——极限同样可以不唯一。

三个例子说明：**没有 Hausdorff，极限就「不老实」**。分析里「极限唯一」的安心感，全部来自 Hausdorff 性的暗中守护。

### Hausdorff 与「连续映射的图」

Hausdorff 性还有一个漂亮的几何推论：**连续映射的图是闭集**。

- 设 $f : X \to Y$ 连续，$Y$ Hausdorff，则 $f$ 的图 $\Gamma_f = \{(x, f(x)) \mid x \in X\}$ 是 $X \times Y$ 的闭子集。
- 证明：$(x,y) \notin \Gamma_f$ ⟹ $f(x) \neq y$，Hausdorff 给不交邻域 $U \ni f(x)$、$V \ni y$；$f$ 连续 ⟹ $f^{-1}(U)$ 是 $x$ 的邻域，$(f^{-1}(U)) \times V$ 是 $(x,y)$ 的邻域且与 $\Gamma_f$ 不交。
- 反过来，若 $Y$ 非 Hausdorff，「连续函数图闭」可能失败。

「图闭」在分析里被默认（连续函数的图像是闭集），后台正是 Hausdorff 性。这条推论在微分几何（图作为子流形）、控制理论里是常用工具。
