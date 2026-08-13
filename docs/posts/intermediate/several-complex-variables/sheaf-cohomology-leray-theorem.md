---
title: 层上同调与 Leray 定理
date: 2026-08-07
---

# 层上同调与 Leray 定理

<div class="epigraph">
<p>上同调是测量「局部正确的东西无法拼成整体」的尺子——当它为零，一切都各得其所。</p>
<footer>—— 仿 让-皮埃尔 · 塞尔（Jean-Pierre Serre），《凝聚层上同调》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第4章；Krantz 第4章 ｜ 2026-08-07</p>
</div>

## 为什么从层上同调开始

上一节的层论给了我们「局部对象」的织法。但分析问题的核心是**把局部解拼成整体解**——这一步往往会遇到障碍，障碍的大小、形状、层级，正是**层上同调（sheaf cohomology）**要度量的。$H^0$ 度量整体截面，$H^1$ 度量「一对一的粘合障碍」，$H^q$ 度量「$q$ 重覆盖的粘合障碍」。上同调为零（$\bar\partial$ 方程、Cousin 问题）意味着「局部数据总是能拼成整体」——这是多复变几乎一切存在性定理的共同骨架。<span class="marginnote">为什么要专门学 Leray 定理？因为直接定义上同调需要全空间的无穷细化覆盖，算不动。Leray 定理说：只要覆盖的每个有限交的上同调为零，就可以用这<strong>一个覆盖</strong>（有限个开集）计算上同调——把不可算的问题变成有限可算。这是「好的覆盖」思想的开端，也是 Čech 上同调的用武之地。</span>

## 1 Čech 上同调：用覆盖计算

设 $\mathcal U = \{ U_i \}_{i \in I}$ 是 $X$ 的开覆盖，$\mathcal F$ 是层。**$q$-上链（cochain）**是定义在 $q+1$ 重交 $U_{i_0} \cap \cdots \cap U_{i_q}$ 上的截面族 $f_{i_0 \cdots i_q}$，满足交错性。上链全体构成 $C^q(\mathcal U, \mathcal F)$。**上边缘算子** $d: C^q \to C^{q+1}$：

$$
(df)_{i_0 \cdots i_{q+1}} = \sum_{k=0}^{q+1} (-1)^k f_{i_0 \cdots \hat i_k \cdots i_{q+1}}\big|_{U_{i_0}\cap\cdots\cap U_{i_{q+1}}}
$$

易验证 $d^2 = 0$。于是得**上同调群**：

$$
H^q(\mathcal U, \mathcal F) = \frac{\ker(d: C^q \to C^{q+1})}{\mathrm{im}(d: C^{q-1} \to C^q)}
$$

**层上同调**定义为**一切**覆盖的极限：$H^q(X, \mathcal F) = \varinjlim_{\mathcal U} H^q(\mathcal U, \mathcal F)$（细化方向取极限）。<span class="marginnote">$q=0$ 时的意义：$\ker d = $ 在重叠区一致的截面族 $= \mathcal F(X)$（粘合公理）。所以 $H^0(X,\mathcal F) = \mathcal F(X)$ 是整体截面。$H^1$ 的直观：$H^1 = 0$ 意味着「在重叠区差一个上边缘」的局部截面族总能调整成全局一致——这就是粘合无障碍。</span>

## 2 层上同调的基本性质

**性质 1（长正合列）**：对短正合列 $0 \to \mathcal F' \to \mathcal F \to \mathcal F'' \to 0$，有长正合列

$$
0 \to H^0(\mathcal F') \to H^0(\mathcal F) \to H^0(\mathcal F'') \to H^1(\mathcal F') \to H^1(\mathcal F) \to H^1(\mathcal F'') \to H^2(\mathcal F') \to \cdots
$$

这是「局部精确性如何传达给整体」的精确刻画——$H^0$ 满 ⟺ 边界映射 $H^0(\mathcal F'') \to H^1(\mathcal F')$ 为零。<span class="marginnote">长正合列是上同调理论的心脏。多复变里最常见的用法：$0 \to \mathbb Z \to \mathcal O \to \mathcal O^* \to 0$（指数层序列），它把「全纯函数的零点/极点结构」（$\mathcal O^*$）与「整值上同调」（$\mathbb Z$）与「可解性」（$\mathcal O$）联系起来——除子理论的根基。</span>

**性质 2（消失与可缩）**：若 $X$ 是光滑流形且 $\mathcal F$ 是光滑层，$H^q(X, \mathcal F) = 0$ 对 $q > \dim_{\mathbb R} X$；若 $X$ 可缩（如球、多圆柱），实系数上同调 $H^q(X,\mathbb R) = 0$（$q \geq 1$）。

## 3 Leray 定理：好覆盖与计算

**Leray 定理**：设 $\mathcal U$ 是 $X$ 的开覆盖，且对每个有限交 $U_{i_0} \cap \cdots \cap U_{i_q}$ 与所有 $q \geq 1$ 有 $H^q(U_{i_0}\cap\cdots\cap U_{i_q}, \mathcal F) = 0$（覆盖是 **Leray 覆盖**，对 $\mathcal F$）。则

$$
H^q(X, \mathcal F) \cong H^q(\mathcal U, \mathcal F) \qquad \forall q \geq 0
$$

即：**一个好覆盖的上同调 = 全空间的上同调**。计算时只需这一个覆盖。<span class="marginnote">证明用谱序列（Leray 谱序列）或「全实轴消去」方法。一个经典例子：对可缩空间 $X$，覆盖成单点邻域就是 Leray 覆盖；对多圆柱 $U$，覆盖成小多圆柱，则 $H^q(U, \mathcal O) = 0$（$q \geq 1$）——这本身是 Dolbeault 定理（下节）与 Cauchy 积分的结合。</span>

**为什么 Leray 定理对多复变是救星**：一般域 $D$ 的层上同调定义要求取所有覆盖的极限，几乎无法直接计算。而若 $D$ 是全纯凸域，我们可以找到**由多圆柱组成的 Leray 覆盖**（多圆柱的 $\mathcal O$-上同调为零），于是 $H^q(D, \mathcal O)$ 可用有限覆盖算出——分析问题变成**有限线性代数**。

## 4 公式解析：上边缘算子与 $\bar\partial$ 的联系

$$
(df)_{i_0\cdots i_{q+1}} = \sum_{k=0}^{q+1} (-1)^k f_{i_0\cdots\hat i_k\cdots i_{q+1}}
$$

- **第一步，符号的规律**：第 $k$ 项「删掉第 $k$ 个指标」，符号为 $(-1)^k$。交错性保证 $d^2 = 0$：删两次指标，两种顺序差一个符号，恰好抵消。这是「边界算子」的公理形态——与拓扑中单纯复形的边界算子、De Rham 微分算子同构同构。
- **第二步，$q=1$ 的显式写**：对 $f_{ij}$（在 $U_i\cap U_j$ 上，$f_{ij} = -f_{ji}$），$(df)_{ijk} = f_{jk} - f_{ik} + f_{ij}$。$f$ 是上闭链 ⇔ $f_{jk} - f_{ik} + f_{ij} = 0$（**上闭链条件**），即三交上的「协调性」。$H^1$ 把「差一个边界」的上闭链等同起来。
- **第三步，与分析微分算子共鸣**：对 $\mathcal O$ 层，粘合障碍可写成 $\bar\partial$ 型方程 $g_i - g_j = f_{ij}$ 的障碍；$H^1(D, \mathcal O) = 0$ ⟺ 对每个上闭链 $f_{ij}$，存在 $g_i$ 使 $f_{ij} = g_i - g_j$。这正是下一节 Cousin I 问题的层论翻译——**上同调与偏微分方程在此合流**。

## 5 辨析与延伸：上同调的五个要点

**辨析 1：$H^1$ 是「粘合障碍」的精确度量**。$H^1(\mathcal U, \mathcal F) = 0$ 意味着「重叠区一致的数据总能拼成整体截面」。Cousin I 问题、$\bar\partial$ 方程、线丛的平凡性，全部是 $H^1$ 消失的不同版本。<span class="marginnote">$H^q$ 的直观层级：$H^0$ = 整体截面；$H^1$ = 一对一粘合障碍；$H^2$ = 三重覆盖的障碍……$H^q$ 度量「$q+1$ 重覆盖的协调性」。</span>

**辨析 2：Čech 上同调 vs 导出上同调**。Čech 用覆盖直接算；导出上同调用内射消解算。两者在好条件下同构（如 Stein 空间、好覆盖）。**学习时先掌握 Čech（可算），再理解导出（抽象）**——它们是同一台机器的两个操作界面。

**辨析 3：Leray 定理的价值**。Leray 定理让上同调从「不可算的极限」变成「可算的有限覆盖」。没有它，$H^q(D,\mathcal O)$ 几乎无法手算。**「找好覆盖」是上同调计算的第一技能**。

**辨析 4：长正合列是「流水线」**。$0 \to H^0(\mathcal F') \to H^0(\mathcal F) \to H^0(\mathcal F'') \to H^1(\mathcal F') \to \cdots$：每一段的满性/消失性决定下一段的障碍。**读长正合列要像读生产线**——一个环节卡住，下游全部受影响。

**误区清单**：

- **误区 1**：以为「$H^q$ 只在 $q=1$ 有用」。
  正解：$H^2$ 及更高在上同调消失定理、谱序列中至关重要。
- **误区 2**：以为「Čech 上同调 = 导出上同调」恒成立。
  正解：只在好覆盖/良好空间下同构。
- **误区 3**：以为「上同调是纯代数」。
  正解：多复变中 $H^q(D,\mathcal O)$ 直接对应 $\bar\partial$ 方程可解性——分析与代数合流。
- **误区 4**：以为「Leray 覆盖总是存在」。
  正解：需要空间有足够好的局部结构（如多圆柱覆盖全纯凸域）。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| Čech 上同调 | Čech cohomology | 覆盖定义 |
| 上闭链 | cocycle | 协调条件 |
| 上边缘 | coboundary | 精确的差 |
| 上同调群 | cohomology group | 闭/精确商 |
| Leray 定理 | Leray's theorem | 好覆盖计算 |
| 长正合列 | long exact sequence | 局部传整体 |

## 6 历史注记与知识树

**历史**：Leray（1945）在战俘营中发明层与谱序列；Cartan 讨论班将其引入多复变；Serre（1955）建立 $H^q(X,\mathcal F)$ 的完整理论并证明定理 A/B（Stein 空间上凝聚层的上同调消失）。Dolbeault（1953）给出 $\mathcal O$-上同调的微分形式模型——上同调由此成为多复变的标准语言。

**知识树**：

- 向后：凝聚层（本组第 13 篇）、全纯域（第 1 组）。
- 向前：Cousin I/II（本组第 15–16 篇）、Dolbeault 上同调（本组第 17 篇）。
- 横向：代数拓扑的奇异上同调、De Rham 上同调（第三级《代数拓扑》）——同一思想的三个载体。

**一句话记忆**：$H^q$ 度量「$q+1$ 重粘合障碍」；Leray 定理说好覆盖即可算全空间；$H^1(D,\mathcal O)=0$ 是多复变一切可解性的统一判据。

## 7 小结

- **Čech 上同调**：$H^q(\mathcal U, \mathcal F) = \ker d / \mathrm{im}\, d$；$H^0 = \mathcal F(X)$ 是整体截面。
- **长正合列**：局部精确性传达给整体的精确工具；指数层序列 $0 \to \mathbb Z \to \mathcal O \to \mathcal O^* \to 0$ 连接除子与上同调。
- **Leray 定理**：好覆盖（有限交上同调为零）的上同调 = 全局上同调，把不可算变为可算。
- 多圆柱的 $\mathcal O$