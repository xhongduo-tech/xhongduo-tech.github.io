---
title: 模层与凝聚层
date: 2026-08-07
---

# 模层与凝聚层

<div class="epigraph">
<p>向量丛不是别的东西，就是局部自由的模层。</p>
<footer>—— 由让-皮埃尔 · 塞尔（Jean-Pierre Serre）思想转述</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数几何 ｜ Hartshorne, Algebraic Geometry (GTM 52) Ch. II §5 ｜ 2026-08-07</p>
</div>

## 为什么从模层继续

概形 $(X, \mathcal{O}_X)$ 的结构层 $\mathcal{O}_X$ 是"函数环"。但几何对象上不只是函数，还有"函数作用的对象"：向量场的切丛、子簇的理想、以及任意"在 $\mathcal{O}_X$ 上线性"的东西。把它们统一成**模层（sheaf of modules）**，就有了一个可以做线性代数的世界：可以讨论"局部自由"（= 向量丛）、"凝聚"（= 有限生成 + 有限关系）、以及"拉回与推出"（base change 时代的线性代数）。

本节是层论从"函数"到"函数的系数系统"的关键一跳。模层理论是第 10 篇层上同调、第 11 篇 Serre 对偶、第 12 篇 Riemann-Roch 的直接土壤——Riemann-Roch 定理说到底是"算一个模层的 Euler 示性数"。同时它也呼应第一级《线性代数》：那里的"向量空间上的线性映射"在这里升级为"概形上的模层态射"，维数不变量升级为"秩"与"Euler 特征"。

## 1 模层：概形上的"线性代数"

**核心概念：$\mathcal{O}_X$-模层（sheaf of $\mathcal{O}_X$-modules）**：概形 $X$ 上的**模层** $\mathcal{F}$ 是满足条件的层：每个 $\mathcal{F}(U)$ 是 $\mathcal{O}_X(U)$-模，且限制映射与环限制相容（$s \in \mathcal{F}(U)$，$a \in \mathcal{O}_X(U)$，则 $(a s)|_V = a|_V \, s|_V$）。<span class="marginnote">直观：$\mathcal{F}$ 是"定义在 $X$ 上、每个局部都承载 $\mathcal{O}_X$-线性结构的系统"。正如流形上的向量丛是"每个纤维是一个向量空间、光滑依赖于底点"，模层是"每个茎是局部环的模、代数地依赖于点"。</span>

模层之间的**态射**是层的态射，且在每点诱导茎的 $\mathcal{O}_{X,P}$-模同态。全体 $\mathcal{O}_X$-模层构成 Abel 范畴——于是核、余核、直和、正合列这些线性代数概念全部升级到层论：

$$0 \longrightarrow \mathcal{F}' \longrightarrow \mathcal{F} \longrightarrow \mathcal{F}'' \longrightarrow 0$$

的**正合**由每个茎上的正合性定义。**这是后续上同调理论的唯一入口**：上同调就是"正合列被取截面后破损"的度量。

**重点例子：理想层（ideal sheaf）**。对闭子概形 $Y \subseteq X$，定义

$$\mathcal{I}_Y(U) = \{ f \in \mathcal{O}_X(U) \mid f \text{ 在 } Y \cap U \text{ 上为零} \}$$

$\mathcal{I}_Y$ 是 $\mathcal{O}_X$-模层，且有正合列 $0 \to \mathcal{I}_Y \to \mathcal{O}_X \to \mathcal{O}_Y \to 0$（$\mathcal{O}_Y = i_* \mathcal{O}_Y$ 是 $Y$ 的结构层，这里 $i: Y \to X$ 是闭浸入）。<span class="marginnote">理想层是"$Y$ 如何嵌在 $X$ 里"的代数记录——$Y$ 的坐标环 $A(Y)$ 对应 $A(X)/I(Y)$，层论版本即 $\mathcal{O}_Y = \mathcal{O}_X / \mathcal{I}_Y$。整条嵌入信息被压缩进 $\mathcal{I}_Y$ 这个模层里。</span>

## 2 仿射情形的完全分类：准凝聚层

在仿射概形上，模层理论有一个**决定性定理**，它把模层完全翻译回交换代数：

**重点：仿射概形上的模层 = 模。** 设 $X = \operatorname{Spec} A$。对任意 $A$-模 $M$，定义其**相伴模层** $\widetilde{M}$：

$$\widetilde{M}(D(f)) = M_f = M \otimes_A A_f$$

则 $M \mapsto \widetilde{M}$ 给出范畴等价

$$\{A\text{-模}\} \longleftrightarrow \{\text{拟凝聚 } \mathcal{O}_X\text{-模层}\}$$

并且 $\widetilde{M}$ 在点 $\mathfrak{p}$ 处的茎是 $\widetilde{M}_{\mathfrak{p}} = M_{\mathfrak{p}}$。<span class="marginnote">这一句把交换代数整个搬进几何：$A$-模理论（生成、关系、张量、局部化）= 仿射概形上的模层理论。上同调、维数、正合性的许多证明因此可以"先在仿射图上做代数、再拼回几何"。</span>

**核心概念：拟凝聚层（quasi-coherent sheaf）**：$\mathcal{O}_X$-模层 $\mathcal{F}$ 称为**拟凝聚**的，如果在每个仿射开子集 $U = \operatorname{Spec} A$ 上，$\mathcal{F}|_U \cong \widetilde{M}$ 对某个 $A$-模 $M$ 成立。仿射版本是完备分类：拟凝聚层 ⟺ 由 $A$-模生成。<span class="marginnote">这个定义"把可生成性局部化"：局部上总能找到一个（可能很大的）模来生成它。它与"$A$-模有生成元集"对应——不要求有限，所以叫"拟"凝聚。</span>

**核心概念：凝聚层（coherent sheaf）**：若 $\mathcal{F}$ 拟凝聚，且局部上对应的 $A$-模 $M$ 是**有限生成**的，则称 $\mathcal{F}$ **凝聚**。<span class="marginnote">有限生成 = "只有有限多个'生成元'"。在 $X$ 是 Noether 概形时，"凝聚"等价于"局部上由有限生成 $A$-模给出"，这个条件保证了下文层的良好行为（紧致性：$\operatorname{Hom}$ 与张量可交换极限等）。</span>

**辨析｜易错点：** "拟凝聚"与"凝聚"的区别在有限性。拟凝聚层允许局部"无穷生成"，凝聚层要求局部"有限生成"。在 Noether 概形上两者差一层"有限性"，而有限性是上同调理论能"算出有限维数"的前提——**Riemann-Roch 能给出数字，正是因为它只处理凝聚层**。初学者常把两者混用，但"凝聚"的有限性正是"可以计数"的代数根源。

## 3 局部自由层与向量丛

**核心概念：局部自由层（locally free sheaf）**：$\mathcal{O}_X$-模层 $\mathcal{F}$ 称为**秩 $n$ 的局部自由层**，如果存在开覆盖 $\{U_i\}$ 使 $\mathcal{F}|_{U_i} \cong \mathcal{O}_X^{\oplus n}|_{U_i}$。秩 1 的局部自由层称为**可逆层（invertible sheaf）**，也称**线丛（line bundle）**。<span class="marginnote">局部自由 = "局部像自由模 $\mathcal{O}_X^{\oplus n}$"。与向量丛的对应：每个向量丛 $E \to X$ 的截面层 $\mathcal{O}(E)$ 是局部自由层；反过来每个局部自由层给出向量丛。Serre 的这个对应是"代数几何 = 层论几何"的枢纽之一。</span>

**重点：可逆层与 Picard 群。** 可逆层的张量积仍是可逆层，且每个可逆层有逆（对偶层）。于是

$$\operatorname{Pic} X = \{ \text{可逆层的同构类} \} / \otimes \text{ 构成 Abel 群}$$

称为 $X$ 的 **Picard 群**。$\operatorname{Pic} \mathbb{P}^n_k = \mathbb{Z}$（由 $\mathcal{O}(1)$ 生成）——这是"线丛由次数分类"的精确表述。<span class="marginnote">$\mathcal{O}(1)$ 是 $\mathbb{P}^n$ 上的<strong>挠层（tautological line bundle）</strong>，其整体截面是齐次坐标多项式（次数 1 者），$d$ 次齐次多项式恰是 $\mathcal{O}(d) = \mathcal{O}(1)^{\otimes d}$ 的截面。Picard 群 = 次数分类 = $\mathbb{Z}$，这与"齐次多项式的次数决定射影对象"完全一致。</span>

**关键代数事实：局部自由 ⟹ 平坦。** 局部自由层的张量积与拉回保持正合性（"平坦"），这使第 6 篇的基变换与这里的模层运算无缝衔接：**"向量丛拉回仍是向量丛"**。

## 4 拉回与推出：模层的"搬迁"

态射 $f: X \to Y$ 给模层两个方向的操作：

**核心概念：推出（pushforward）** $f_*$：对 $Y$-... 对 $X$ 上的模层 $\mathcal{F}$，定义 $(f_* \mathcal{F})(V) = \mathcal{F}(f^{-1}V)$。直观："把定义在 $X$ 上的数据搬到 $Y$ 上"——但可能"维度膨胀"（纤维的维数被塞进截面）。

**核心概念：拉回（pullback）** $f^*$：对 $Y$ 上的模层 $\mathcal{G}$，定义 $f^* \mathcal{G} = f^{-1} \mathcal{G} \otimes_{f^{-1} \mathcal{O}_Y} \mathcal{O}_X$。<span class="marginnote">拉回与"纤维"一致：$f^*\mathcal{G}$ 在点 $x$ 的纤维 = $\mathcal{G}$ 在 $f(x)$ 的纤维。它是基变换在层论里的对应物——第 6 篇几何的"搬基"，这里给出代数数据的"搬基"。</span>拉回是右正合的，推出一般只左正合——这个不对称正是上同调（第 10 篇）要补偿的。

**重点：射影态射的推出凝聚。** 若 $f: X \to Y$ 是真态射、$\mathcal{F}$ 是 $X$ 上的凝聚层，则 $f_* \mathcal{F}$ 是 $Y$ 上的凝聚层。<span class="marginnote">"真 + 凝聚 ⟹ 推出凝聚"是"紧空间上的有限性不丢"的代数版：紧致性保证推出去的层还是"有限生成"的。这是后文上同调有限性定理（$H^i$ 有限维）的胚胎。</span>

## 5 公式解析：拟凝聚 ⟺ 局部生成

$$
\mathcal{F} \text{ 拟凝聚} \iff \text{ 对仿射开集 } U = \operatorname{Spec} A,\ \mathcal{F}|_U \cong \widetilde{M} = \left( M_f \right)_{f \in A}
$$

分三步拆解：

- **第一步，$\widetilde{M}$ 是什么**：$\widetilde{M}(D(f)) = M_f$，即"$M$ 对元素 $f$ 的局部化"。几何上：$D(f)$ 上的截面 = "允许 $f$ 作分母"的模元素——与结构层 $\mathcal{O}(D(f)) = A_f$ 完全同构的模版本。<span class="marginnote">它把"模"这个纯代数对象变成"在概形上按局部化分布"的层：每片仿射图上的截面由"允许除以 $f$"确定，这就是"模被展开在空间上"。</span>
- **第二步，为什么需要"局部上"**：一般概形不能整体写成 $\operatorname{Spec} A$，只能由仿射图覆盖。拟凝聚的定义要求在**每一张**仿射图上由某个 $A$-模生成，且覆盖重叠处胶合相容。这是"把 $\{A$-模$\}$ 从单张图推广到粘合图"的标准手法。
- **第三步，凝聚 = 加有限性**：$M$ 有限生成 ⟹ 拟凝聚层升格为凝聚层。有限性使"层的数据"在张量、$\operatorname{Hom}$、上同调中保持可控，从而 Riemann-Roch 得以输出整数。

一句话直觉：**拟凝聚层 = "由某个模在每张仿射图上生成"的层；凝聚 = 再要求该模局部有限生成**。模层的世界由此成为"概形上的线性代数"。

## 6 对照表：模层家族

| 概念 | 局部条件 | 对应代数对象 | 上同调表现 |
| --- | --- | --- | --- |
| 任意 $\mathcal{O}_X$-模 | 无要求 | 一般 $A$-模 | 可能"病态" |
| 拟凝聚层 | 局部由 $A$-模生成 | 任意 $A$-模 $M$ | 仿射上 $H^i = 0$（$i \ge 1$） |
| 凝聚层 | 局部有限生成 | 有限生成 $A$-模 | 射影上有限维、高次为零 |
| 局部自由层 | 局部 $\cong \mathcal{O}_X^{\oplus n}$ | 有限生成射影 $A$-模 | 对应向量丛 |
| 可逆层 | 局部 $\cong \mathcal{O}_X$ | 秩 1 射影模 | 对应线丛 |

**算例：$\mathbb{P}^1$ 上的 $\mathcal{O}(d)$。** 它们是可逆层（线丛）。$d \ge 0$ 时 $H^0(\mathbb{P}^1, \mathcal{O}(d)) = k[x_0,x_1]_d$（$d+1$ 维）；$d \le -2$ 时 $H^1(\mathbb{P}^1, \mathcal{O}(d))$ 非零（对偶于 $H^0(\mathbb{P}^1, \mathcal{O}(-d-2))$）。"上同调随次数翻转"的规律正是 Serre 对偶（第 11 篇）的雏形，也决定了 Riemann-Roch 在 $\mathbb{P}^1$ 上的全部内容。<span class="marginnote">特别注意 $d = -1$ 这个"中间地带"：$H^0$ 与 $H^1$ 都为零，线丛 $\mathcal{O}(-1)$ 没有任何整体截面——它是最简单的"没有函数"的可逆层，却作为 $\mathbb{P}^n$ 的<strong>扭层</strong>在投影与 blow-up 里反复出现。</span>

**辨析｜易错点：** 拟凝聚与凝聚的边界在 Noether 概形上只差一层"有限生成"，但这不是形式区别：凝聚层的有限性正是上同调"能算出有限维数"的前提，Riemann-Roch 的 $\ell(D)$ 是整数而非无穷大，全靠它兜底。判断一个层是否凝聚，最稳妥的办法是回到仿射图上看对应的 $A$-模是否有限生成——不要凭直觉猜测。

## 7 小结

- **模层**：每个开集上的 $\mathcal{O}_X$-模、相容于限制；核、余核、正合列全线升级。
- **理想层** $\mathcal{I}_Y$：闭子概形的代数记录，正合列 $0 \to \mathcal{I}_Y \to \mathcal{O}_X \to \mathcal{O}_Y \to 0$。
- **拟凝聚层**：局部由 $A$-模 $\widetilde{M}$ 生成；**凝聚层**：再要求有限生成。仿射情形 $\{A\text{-模}\} \cong \{$拟凝聚$\}$。
- **局部自由层**：局部 $\cong \mathcal{O}_X^{\oplus n}$；秩 1 即可逆层/线丛；$\operatorname{Pic} X$ 是线丛的分类群。
- **拉回 / 推出**：$f^*$ 右正合、$f_*$ 左正合；真态射保持凝聚性。
- **本节的地位**：模层理论是"概形上的线性代数"，它让向量丛、理想层、以及所有"函数作用的对象"都有了统一的家——上同调、Serre 对偶、Riemann-Roch 全部从这里起步。

**辨析｜易错点（续）：** 拟凝聚层在非 Noether 概形上会"失控"：$X$ 不是 Noether 时，"拟凝聚"与"由有限生成 $A$-模给出"不再等价，甚至"凝聚层在拟凝聚层中的核"可能不再是凝聚层。本专题的几何对象（Noether 的射影簇与概形）足够好，这些问题不会出现——但初学阶段应知道"凝聚 = 有限性"是 Noether 假设下的稳定版本。

在下一节，我们把"个别的"模层升级为"系统的"研究对象：**除子、线性系与微分形式**——用"切了哪些零点、留了哪些极点"给线丛分类，并定义曲线的典范类。
