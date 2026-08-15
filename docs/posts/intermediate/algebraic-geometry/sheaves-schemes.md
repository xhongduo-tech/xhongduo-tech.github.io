---
title: 层与概形
date: 2026-08-07
---

# 层与概形

<div class="epigraph">
<p>概形论使代数几何第一次真正成为统一的理论。</p>
<footer>—— 亚历山大 · 格罗滕迪克（Alexander Grothendieck）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数几何 ｜ Hartshorne, Algebraic Geometry (GTM 52) Ch. II §1-2 ｜ 2026-08-07</p>
</div>

## 为什么从层与概形继续

前四篇研究的是「簇」：由多项式方程定义的、$k$ 上不可约的几何对象。但簇有几个根本性的局限：它假设底域是代数闭的（整数环上的算术几何无处安放）；它把"点"限制为极大理想（素理想这种"更小的点"没有位置）；它还无法区分"同一个簇的不同嵌入"。Grothendieck 的**概形（scheme）**理论一举解决全部问题，而它的两个支柱就是本节的主角：**层（sheaf）**与**仿射概形**。

层把"局部数据如何拼成全局数据"彻底形式化——这个概念在拓扑学、微分几何（流形的连续/光滑函数层）、以及现代深度学习（"局部特征如何融合成全局表示"）中都是核心；概形则把"簇"升级成"带函数环的空间"（环化空间），让代数的点（素理想）直接成为空间的点。这是从 1950 年代 Serre 的工作到 Grothendieck 1958 年后革命的转折点。

## 1 预层与层：局部数据的拼图

**核心概念：预层（presheaf）**：拓扑空间 $X$ 上的**预层** $\mathcal{F}$ 把每个开集 $U \subseteq X$ 映到一个集合（或群/环/模）$\mathcal{F}(U)$，并对每个包含 $V \subseteq U$ 给出限制映射 $\rho_{UV}: \mathcal{F}(U) \to \mathcal{F}(V)$，满足：$\rho_{UU} = \mathrm{id}$，且 $\rho_{VW} \circ \rho_{UV} = \rho_{UW}$。<span class="marginnote">思想：$\mathcal{F}(U)$ 是"定义在 $U$ 上的可允许对象"，限制映射是"把函数限制到小开集"。预层 = 这个分配体系的全部规则。类比：深度学习中一个特征图在不同感受野上的"局部化"。</span>

**核心概念：层（sheaf）**：预层 $\mathcal{F}$ 是**层**，如果它对任意开覆盖 $\{U_i\}$ 满足两条公理：<span class="marginnote">两条公理分别叫「唯一性」与「胶合」：全局对象被局部限制唯一确定；局部一致的数据必能拼成全局对象。这是"局部决定整体"的严格化——现代几何与代数的基石。</span>

1. **唯一性**：若 $s, t \in \mathcal{F}(U)$ 且对每个 $i$ 有 $s|_{U_i} = t|_{U_i}$，则 $s = t$。
2. **胶合（gluing）**：若对每个 $i$ 有 $s_i \in \mathcal{F}(U_i)$，且 $s_i|_{U_i \cap U_j} = s_j|_{U_i \cap U_j}$ 对所有 $i, j$ 成立，则存在 $s \in \mathcal{F}(U)$ 使得 $s|_{U_i} = s_i$ 对所有 $i$ 成立。

例子：流形上的连续函数层 $\mathcal{C}^0$、光滑函数层 $\mathcal{C}^\infty$、仿射簇 $X$ 上的**正则函数层** $\mathcal{O}_X$（$\mathcal{O}_X(U)$ = $U$ 上的正则函数）都是层。<span class="marginnote">非层的典型例子："有界函数"不是层——胶合公理失败：各局部有界不代表整体有界（把区间切成小片，每片上函数有界，整体可能无界）。这个反例帮助理解"层"与"随便什么分配"的区别。</span>

**核心概念：茎（stalk）**：点 $P \in X$ 处的**茎**定义为直接极限

$$\mathcal{F}_P = \varinjlim_{U \ni P} \mathcal{F}(U)$$

直观上，$\mathcal{F}_P$ 是"在 $P$ 的无穷小邻域上"的对象。对正则函数层，茎 $\mathcal{O}_{P,X}$ 正是第 4 篇定义的局部环。<span class="marginnote">「茎」这个译名很妙：纤维在点处"扎进土壤"的根。层论里"层 → 茎"是取局部，反过来"茎 → 层"需要额外的黏合结构（etale space），两条路构成层论的第一个对偶。</span>

**预层与层的例子对照。**

| 分配规则（$U \mapsto \mathcal{F}(U)$） | 是层？ | 说明 |
| --- | --- | --- |
| 连续函数 $\mathcal{C}^0$ | 是 | 连续函数局部连续则整体连续 |
| 光滑函数 $\mathcal{C}^\infty$ | 是 | 同上 |
| 有界函数 | 否 | 各片有界不保证整体有界 |
| 局部常值函数 | 否 | 胶合后可能不再是局部常值 |
| 正则函数 $\mathcal{O}_X$ | 是 | 簇上的核心例子，本节的出发层 |

这张表说明"层"不是任意分配规则，而是恰好满足"唯一性 + 胶合"的那些——判断一个候选对象是不是层，最稳妥的办法是逐条检查两条公理。

## 2 环化空间与坐标化

**核心概念：环化空间（ringed space）**：配对 $(X, \mathcal{O}_X)$，其中 $X$ 是拓扑空间，$\mathcal{O}_X$ 是 $X$ 上的**环层**（每个 $\mathcal{O}_X(U)$ 是环，限制映射是环同态）。若每个茎 $\mathcal{O}_{X,P}$ 都是**局部环**（有唯一极大理想 $\mathfrak{m}_{X,P}$），则称为**局部环化空间（locally ringed space）**。<span class="marginnote">局部环的极大理想记住了"点处消失的函数"：$\mathfrak{m}_{X,P}$ 是在 $P$ 处取零值的函数。它使"值"的概念良定——函数在点处取值 = 模极大理想的余像。这就是为什么层论里的环必须局部，而不是任意环。</span>

**核心概念：仿射概形（affine scheme）**：对任意交换环 $A$（不必是 $k$ 代数、不必是整环！），定义其**谱**

$$\operatorname{Spec} A = \{ \text{所有素理想 } \mathfrak{p} \subsetneq A \}$$

并装备 **Zariski 拓扑**（闭集由"含某理想的素理想全体" $V(\mathfrak{a})$ 构成），以及**结构层** $\mathcal{O}_{\operatorname{Spec} A}$（在每个主开集 $D(f)$ 上取局部化 $\mathcal{O}(D(f)) = A_f$）。这个环化空间 $(\operatorname{Spec} A, \mathcal{O})$ 称为**仿射概形**。<span class="marginnote">关键飞跃：$A$ 里每个素理想 $\mathfrak{p}$ 都是一个<strong>点</strong>。极大理想是"通常的点"（如 $k[x]$ 里的 $(x-a)$ 对应 $x=a$），素理想如 $(0)$ 则是"通有点"——它不在任何极大理想里，代表"整个空间的默认位置"。正是这点把算术（$\operatorname{Spec} \mathbb{Z}$）和几何统一。</span>

**重点：仿射概形的点不都是闭点。** $\operatorname{Spec} k[x]$：极大理想 $(x-a)$ 是闭点；素理想 $(0)$（一般点）的闭包是整个 $\operatorname{Spec} k[x]$，故它不是闭点。<span class="marginnote">一般点（generic point）是代数几何独有的概念：簇 $X$ 的"一般点" $\eta$ 满足 $\overline{\{\eta\}} = X$。它的存在使得"在一个开集上成立的性质"可以翻译成"在一般点上成立"——这是代数几何大量命题的证明技巧（"一般性论证"）。</span>

## 3 概形：粘起来的仿射概形

**核心概念：概形（scheme）**：**局部环化空间 $(X, \mathcal{O}_X)$ 称为概形**，如果 $X$ 有开覆盖 $\{U_i\}$，使得每个 $(U_i, \mathcal{O}_X|_{U_i})$ 同构于某个仿射概形 $\operatorname{Spec} A_i$。<span class="marginnote">概形 = "粘起来的仿射概形"，正如流形 = "粘起来的 $\mathbb{R}^n$ 开集"。这个定义把簇推广到：底域不必代数闭、环不必整环、可以有幂零元、可以有"点"不是极大理想……一切"太严格"的假设都被去掉。</span>

**例子：$k$ 上的仿射空间** $\mathbb{A}^n_k = \operatorname{Spec} k[x_1, \dots, x_n]$；**射影空间** $\mathbb{P}^n_k$ 由 $n+1$ 个仿射图覆盖：

$$\mathbb{P}^n_k = U_0 \cup \cdots \cup U_n, \qquad U_i = \operatorname{Spec} k[x_0/x_i, \dots, x_n/x_i]$$

粘合条件由坐标比之间的关系给出。<span class="marginnote">这是"射影空间 = $n+1$ 张仿射图拼成"的精确版本，与拓扑学里"球面 = 两张图拼成"是同一手法。每张图对应一个齐次坐标 $x_i \neq 0$ 的仿射片。</span>

**簇 → 概形**：一个仿射簇（$k$ 代数闭、$A(X)$ 整环）对应概形 $\operatorname{Spec} A(X)$；射影簇对应粘合出的射影概形。但概形有簇没有的"额外点"：$\operatorname{Spec} A(X)$ 里除闭点（对应原簇的点）外，还有一般点 $\eta$（对应素理想 $(0)$），以及中间维度的素理想点。<span class="marginnote">这些"非闭点"初看多余，实则不可或缺：它们使"在子簇上成立"可以被表示为"在子簇的一般点上成立"，从而使理想理论、局部化理论直接成为几何理论。这就是为什么概形语言最终胜出。</span>

**辨析｜易错点：** 概形不要求环是整环或约化环。$\operatorname{Spec} k[\varepsilon]/(\varepsilon^2)$ 是"双点重叠"的无穷小邻域（幂零元 $\varepsilon \neq 0$ 但 $\varepsilon^2 = 0$），它是几何的"一阶射影"的代数化身——在第 4 篇切空间的意义下，$\varepsilon$ 正是"无穷小方向"。初学者常误以为幂零元必须被禁止，事实上它们正是"箭头与切向量"的家。

## 4 从正则函数层到坐标化公式

把前面所有概念收束成一条公式。**核心事实：**

$$\mathcal{O}_{\operatorname{Spec} A}(D(f)) = A_f, \qquad \mathcal{O}_{\operatorname{Spec} A, \, \mathfrak{p}} = A_{\mathfrak{p}}$$

其中 $A_f$ 是对元素 $f$ 的局部化，$A_{\mathfrak{p}}$ 是对素理想 $\mathfrak{p}$ 的局部化。<span class="marginnote">这条公式是"结构层在仿射概形上由局部化确定"的精髓：开集 $D(f)$ 上的函数 = 允许 $f$ 作分母的有理式；点 $\mathfrak{p}$ 处的茎 = 以 $\mathfrak{p}$ 为分母允许域的有理式。</span>它把第 1 节预层/层、第 2 节环化空间与"局部化"这个纯代数操作焊在一起：**仿射概形的结构层，本质上就是环的局部化系统**。

## 5 公式解析：茎 = 局部化

$$
\mathcal{O}_{\operatorname{Spec} A, \, \mathfrak{p}} = A_{\mathfrak{p}} = \left\{ \frac{a}{s} \;\middle|\; a \in A,\ s \notin \mathfrak{p} \right\}
$$

分三步拆解：

- **第一步，左边的含义**：茎 $\mathcal{O}_{\operatorname{Spec} A, \mathfrak{p}}$ 是"定义在 $\mathfrak{p}$ 的任意小邻域上的函数"的直接极限。由于 $\mathfrak{p}$ 的邻域由"不含 $\mathfrak{p}$ 的主开集" $D(f)$ 构成，茎 = 这些局部环的并。
- **第二步，局部化的定义**：$A_{\mathfrak{p}}$ 允许除以"不在 $\mathfrak{p}$ 里的元素"。为什么是 $s \notin \mathfrak{p}$？因为在主开集 $D(s)$ 上 $s$ 处处非零，可作分母；而"$\mathfrak{p}$ 的邻域"恰是"$s$ 不含 $\mathfrak{p}$"的那些 $D(s)$。<span class="marginnote">这里的直觉：分母 $s$ 不能使点 $\mathfrak{p}$ 处"消失"。$s \in \mathfrak{p}$ 意味着 $s(\mathfrak{p}) = 0$，作为函数在 $\mathfrak{p}$ 处取零——当然不能作分母。局部化的分母集 $\{\text{不在 } \mathfrak{p} \text{ 里的元素}\}$ 正是"在 $\mathfrak{p}$ 处非零的函数"。</span>
- **第三步，为什么这给出了正则函数**：对 $A = k[x_1, \dots, x_n]$、$\mathfrak{p} = (x_1 - a_1, \dots, x_n - a_n)$，$A_{\mathfrak{p}}$ 是"在点 $a = (a_1, \dots, a_n)$ 附近良定的有理函数"——正是第 4 篇局部环 $\mathcal{O}_{a, \mathbb{A}^n}$。整条链闭合：**局部环 = 结构层在闭点处的茎 = 坐标环的局部化**。

一句话直觉：**概形的"函数"在点 $\mathfrak{p}$ 处允许的分母，恰好是"在该点不消失"的函数**；层、茎、局部化这三个概念在这条公式里合而为一。

## 6 对照表与算例：从簇到概形

| 结构 | 簇（Ch. I） | 概形（Ch. II） |
| --- | --- | --- |
| 底层空间 | 不可约代数集 | 素理想谱 / 粘合 |
| 点 | 闭点（极大理想） | 闭点 + 一般点 + 中间点 |
| 底域 | 要求 $k$ 代数闭 | 任意环（$\mathbb{Z}$ 也行） |
| 环 | 整环（坐标环） | 任意交换环（可有幂零元） |
| 正则函数 | 多项式函数 | 结构层截面（局部化） |
| 嵌入 | 由定义方程给定 | 由理想层给定（可换） |

**算例：$\operatorname{Spec} \mathbb{Z}$ 的"算术几何"。** $\operatorname{Spec} \mathbb{Z}$ 的点是素理想 $(p)$（闭点）与一般点 $(0)$；$\mathcal{O}_{\operatorname{Spec}\mathbb{Z}}$ 在主开集 $D(p)$ 上取 $\mathbb{Z}[1/p]$。几何语言在这里全部成立：$(p)$ 是"点"、$(0)$ 是"一般点"、闭包 $\overline{\{(0)\}} = \operatorname{Spec}\mathbb{Z}$。于是"整数环的代数"第一次有了"空间"的表述——这就是算术几何（Qing Liu 教材的主线）：把 $\mathbb{Q}$ 上的方程放到 $\mathbb{Z}$ 上、再放到 $\mathbb{F}_p$ 上，全用同一个概形语言。<span class="marginnote">对比：作为拓扑空间，$\operatorname{Spec}\mathbb{Z}$ 的闭集是有限个 $(p)$ 并（可能含 $(0)$）——这与 $\mathbb{A}^1$ 的 Zariski 拓扑结构如出一辙。Hartshorne Ch. II 从"仿射概形"起步，正是要让读者先在 $\operatorname{Spec} k[x]$ 上建立全部直觉，再把它平移给 $\operatorname{Spec} \mathbb{Z}$。</span>

**算例：双点与幂零元。** $\operatorname{Spec} k[\varepsilon]/(\varepsilon^2)$ 只有一个闭点 $(\varepsilon)$，但结构层带幂零元 $\varepsilon$（$\varepsilon^2 = 0$）。它的几何是"一个点 + 一个无穷小方向"——正是切向量栖居的地方。第 4 篇的切空间 $T_P X = (\mathfrak{m}_P/\mathfrak{m}_P^2)^*$ 在这里表现为"从 $\operatorname{Spec} k[\varepsilon]/(\varepsilon^2)$ 到 $X$ 的态射"的全体。幂零元不是"错误"，而是"箭头"。

**辨析｜易错点：** 概形不是"带拓扑的环"。$\operatorname{Spec} A$ 的底层集合是素理想，但结构层才是主角：同一个底层集合可以带不同的结构层（如 $\mathbb{A}^n$ 的约化结构与带幂零元的非约化结构）。初学者常只盯集合/拓扑而忽略层——在概形论里，$(X, \mathcal{O}_X)$ 是一个整体，层与空间一样重要。

## 7 小结

- **层**：把局部数据粘成全局对象的公理化（唯一性 + 胶合）；**茎**是点处无穷小邻域的对象。
- **环化空间 / 局部环化空间**：$(X, \mathcal{O}_X)$，茎是局部环，极大理想记住"在点处消失"。
- **仿射概形** $\operatorname{Spec} A$：素理想全体 + Zariski 拓扑 + 结构层（局部化系统）；点不必闭（一般点、幂零元都可以）。
- **概形**：粘起来的仿射概形；$\mathbb{A}^n_k = \operatorname{Spec} k[x]$，$\mathbb{P}^n_k$ 由 $n+1$ 张仿射图覆盖。
- **核心公式**：$\mathcal{O}_{\operatorname{Spec} A,\mathfrak{p}} = A_{\mathfrak{p}}$——层论与交换代数的第一次深度融合。
- **概形的定位**：底域不必代数闭、环不必整环、点不必闭——"簇"的每一种限制都被解除，换来的是算术几何与幂零元几何的统一。
- **层与深度学习（回顾）**：第 1 节的"局部数据拼成全局数据"在注意力机制里同样出现——局部特征如何融合成全局表示；层公理（唯一性 + 胶合）正是"融合必须无歧义"的严格版。

在下一节，我们给概形之间装上"合法映射"：**概形的态射与纤维积**——从基变换（base change）这个在算术几何里无处不在的操作开始。
