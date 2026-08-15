---
title: Hilbert 零点定理与 Zariski 拓扑
date: 2026-08-07
---

# Hilbert 零点定理与 Zariski 拓扑

<div class="epigraph">
<p>代数几何的基本词典：几何与代数，其实是一枚硬币的两面。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）与他的后继者们</footer>
</div>

<div class="article-byline">
<p>第二级 · 交换代数 ｜ Atiyah–Macdonald Ch. 7 ｜ 2026-08-07</p>
</div>

## 为什么从零点定理开始

多项式方程组的解集（代数集）与多项式理想，是两个表面上不同的世界。**Hilbert 零点定理（Nullstellensatz）**把它们焊在一起：`求零点` 与 `取根理想` 互为反演。这是代数几何的基石，也是「交换代数是几何的坐标化语言」这句话最实质的依据。<span class="marginnote">Nullstellensatz 直译为「零点位置定理」，1893 年由 Hilbert 证明。它回答了老问题：给出多项式集合，能否判断它们在 k 的 n 次空间中是否「恰在同一些点消失」——答案藏在根理想里。</span>

这一篇顺着上一节 Noether 链条件的势能推进：先用**弱零点定理**定位极大理想，再用**强零点定理**给出 $V$ 与 $I$ 的对应，最后把这些几何对象组织成 **Zariski 拓扑**——一张点不多、但结构极其深刻的图。第1篇《准素分解》的分支思想将在这里变成拓扑语言。

## 1 弱零点定理：极大理想就是「点」

设 $k$ 是代数闭域，$A = k[x_1, \dots, x_n]$。

**弱零点定理（Weak Nullstellensatz）**：$A$ 的每个极大理想都有形式

$$\mathfrak{m}_{\mathbf{a}} = (x_1 - a_1, \dots, x_n - a_n), \qquad \mathbf{a} = (a_1, \dots, a_n) \in k^n.$$

**重点：代数闭域上，$k^n$ 的点与 $k[x]$ 的极大理想一一对应。** 方向「点 → 极大理想」平凡（$(x_i - a_i)$ 显然极大）；反方向是定理内容：任何极大理想都来自一个点。<span class="marginnote">注意「代数闭」不可省：$\mathbb{R}[x]$ 中 $(x^2+1)$ 是极大理想，但它在 $\mathbb{R}$ 上没有零点——复数域 $\mathbb{C}$ 才补齐这条对应。这正是「代数闭」这个前提的全部意义。</span>

证明的核心是一句惊人的代数学事实：**若 $k$ 是代数闭域、$B$ 是 $A$ 的有限生成域扩张且 $B$ 是域，则 $B$ 是 $k$ 的有限扩张**。由 Zariski 引理可证此事实，再用它把 $A/\mathfrak{m}$ 的生成元映回 $k^n$ 中的点。细节留给教材，直觉记一条：**代数闭域上，「$k[x]/\mathfrak{m}$ 是域」逼出「$\mathfrak{m}$ 是某点的极大理想」。**

这条定理立刻可用 $\mathbb{C}[x]$ 验证：它的极大理想是 $(x-a)$，一一对应 $\mathbb{C}$ 的点 $a$；而 $\mathbb{R}[x]$ 中 $(x^2+1)$ 是极大理想却不对应任何实点——**「代数闭」的前提不是装饰，它保证了「极大理想 = 点」的对应完整成立**。这也把第1篇《理想、模与环同态》里「商是域 ⇔ 极大理想」与「核是理想」两条线索收进同一个图景：几何的点就是代数上「不能再约」的商环。

## 2 强零点定理：$I(V(\mathfrak{a})) = \sqrt{\mathfrak{a}}$

对子集 $S \subseteq k^n$ 定义**消逝理想**

$$I(S) = \{f \in A \mid f(\mathbf{x}) = 0,\ \forall \mathbf{x} \in S\},$$

对理想 $\mathfrak{a}$ 定义**零点集** $V(\mathfrak{a}) = \{\mathbf{x} \in k^n \mid f(\mathbf{x}) = 0,\ \forall f \in \mathfrak{a}\}$。

**强零点定理（Hilbert's Nullstellensatz）**：设 $k$ 代数闭，$\mathfrak{a} \subseteq k[x]$ 是理想，则

$$I(V(\mathfrak{a})) = \sqrt{\mathfrak{a}}.$$

**重点：几何上的「消失条件」恰好对应代数上的「幂落在理想里」。** 左到右的包含平凡：若 $f^n \in \mathfrak{a}$，在 $V(\mathfrak{a})$ 上 $f^n = 0$ 故 $f = 0$。反向需要**Rabinowitsch 技巧**：在 $k[x, t]$ 中给 $f$ 配一个新变量 $t$，令 $t f - 1$ 加入理想，若它无处为零则由弱零点定理矛盾，从而 $(t f - 1, \mathfrak{a})$ 是整环 $k[x,t]$ 的扩环的核……细节交教材。<span class="marginnote">这个技巧后来成为「把无零点问题化为有解问题」的模板，被广泛应用在代数与逻辑的紧致性证明里。</span>

**辨析｜易错点：** $I(V(\mathfrak{a})) = \sqrt{\mathfrak{a}}$ 里根号不可去掉。$k[x]$ 中 $\mathfrak{a} = (x^2)$，则 $V(\mathfrak{a}) = \{0\}$，而 $I(\{0\}) = (x) \neq (x^2)$。零点定理告诉你「从零点集读回的信息」只能是根理想——**几何本身看不见幂次，幂次（重数）是准素分解才记录的信息**。

把字典用具体数字校准一遍。$k = \mathbb{C}$，$A = \mathbb{C}[x]$：$\mathfrak{a} = (x^2 - 1)$，$V(\mathfrak{a}) = \{\pm 1\}$，$I(\{\pm 1\}) = (x^2-1) = \sqrt{(x^2-1)}$；$\mathfrak{a} = (x^2)$，$V = \{0\}$，$I(\{0\}) = (x) = \sqrt{(x^2)}$——两回都验证 $I(V(\mathfrak{a})) = \sqrt{\mathfrak{a}}$。再看 $\mathbb{C}[x,y]$ 的 $\mathfrak{a} = (xy)$：$V$ 是两坐标轴之并，$I = (xy)$，极小素因子 $(x)$、$(y)$ 各自对应一条轴——**几何的分支与代数的极小素因子逐条对齐**。

由强零点定理立刻得到关键的**字典对应**：

**核心对照表：几何与代数词典**

| 几何（$k^n$ / $\operatorname{Spec}$） | 代数（理想 / 环） |
| --- | --- |
| 点 $\mathbf{a} = (a_1,\dots,a_n)$ | 极大理想 $\mathfrak{m}_{\mathbf{a}} = (x_i - a_i)$ |
| 代数子集 $S \subseteq k^n$ | 消逝理想 $I(S)$ |
| 零点集 $V(\mathfrak{a})$ | 理想 $\mathfrak{a}$ |
| 不可约分支 | 极小素因子 / 准素分量 |
| 一般点 $\eta$ | 素理想 $\mathfrak{p}$ |
| 闭集 $V(\mathfrak{a})$ / 开集 $D(f)$ | 理想 $\mathfrak{a}$ / 元素 $f$ |

这张词典贯穿整篇：零点定理保证「$V$ 与 $I$」互为反演（差一个根号），Zariski 拓扑把它变成点的语言，准素分解把它细化到分支与重数。

## 3 Zariski 拓扑：$\operatorname{Spec}$ 的地图

把「$k^n$ 的点」扩充为「$\operatorname{Spec} A$ 的素理想」，代数几何的画布就铺开了。

**Zariski 拓扑**：$\operatorname{Spec} A$ 上以 $V(\mathfrak{a}) = \{\mathfrak{p} \supseteq \mathfrak{a}\}$ 为闭集的拓扑。基开集是

$$D(f) = \{\mathfrak{p} \in \operatorname{Spec} A \mid f \notin \mathfrak{p}\}, \qquad f \in A.$$

**重点：Zariski 拓扑与 Euclidean 拓扑完全不同——它的开集「大而稀」。** $D(f) \neq \emptyset$ 时通常稠密；$\operatorname{Spec} \mathbb{Z}$ 的开集只在去掉有限个素数后才是闭补……开集少到连「分离性」都无法保证，$\operatorname{Spec}$ 一般不是 Hausdorff 空间。<span class="marginnote">代数几何最终靠「层」（sheaf）在 Zariski 拓扑上重建局部理论——正因开集稀少，层条件反而变简单。这一章后面《局部上同调》等主题都默认 Zariski 拓扑。</span>

$\operatorname{Spec} A$ 还比 $k^n$ 多一类点：每个**不可约子簇**的「**一般点（generic point）**」——一个素理想 $\mathfrak{p}$，它的闭包 $V(\mathfrak{p})$ 恰是整个子簇。代数闭域上 $k^n$ 没有一般点，这正说明素理想比极大理想承载更多信息。<span class="marginnote">一般点是交换代数对「无穷小/通性」概念的最早贡献：不可约簇上的「几乎处处成立」可以在它的一般点处统一验证。数域情形的直观类比：$\operatorname{Spec} \mathbb{Z}$ 中 $(0)$ 是「一般点」，它看不见具体的整除。</span>

$\operatorname{Spec} \mathbb{Z}$ 是体会 Zariski 拓扑最好的练习场：它的点是 $(0), (2), (3), (5), \dots$。闭集是有限个 $(p)$ 的并（加上全集）；$(0)$ 的闭包是全体，因为 $(0)$ 含于每个素理想，$V((0)) = \operatorname{Spec} \mathbb{Z}$。于是每个非空开集都「挖掉有限个素数」，既稠密又到处可见——Hausdorff 性彻底无从谈起，这正是「开集大而稀」的活标本。代数闭域上 $k^n$ 没有一般点，正说明素理想比极大理想承载更多信息：**$(0)$ 用「不整除任何数」的视角俯视全部素数**。

![零点定理的词典：几何与代数对照](/images/commutative-algebra/nullstellensatz-dictionary.svg)

## 4 公式解析：从分解到分支再回到根

把零点定理与准素分解合并，可以得到代数几何的**分解公式**：若 $\mathfrak{a} = \mathfrak{q}_1 \cap \cdots \cap \mathfrak{q}_r$ 是最简准素分解、$\mathfrak{p}_i = \sqrt{\mathfrak{q}_i}$，则

$$V(\mathfrak{a}) = V(\mathfrak{p}_1) \cup \cdots \cup V(\mathfrak{p}_r), \qquad \sqrt{\mathfrak{a}} = \mathfrak{p}_1 \cap \cdots \cap \mathfrak{p}_r.$$

拆解这条公式：

- **第一步，几何**：$V(\mathfrak{a})$ 被分解成 $r$ 个不可约闭集 $V(\mathfrak{p}_i)$ 的并——代数簇的不可约分支，见第1篇《准素分解》的几何翻译。
- **第二步，代数**：对两边取根并用根的交集公式 $\sqrt{\mathfrak{q}_1 \cap \cdots} = \sqrt{\mathfrak{q}_1} \cap \cdots$，根的极小分量（相异者）正是素理想。
- **第三步，合流**：把零点定理 $I(V(\mathfrak{a})) = \sqrt{\mathfrak{a}}$ 接上：**几何的分支 = 根的极小素因子**。嵌入素理想对应的分支消失在 $V(\mathfrak{a})$ 中——它们只住在代数侧，住不进几何地图。

**辨析｜易错点：** $V(\mathfrak{a}) = V(\sqrt{\mathfrak{a}})$（几何不认幂次），但 $I(V(\mathfrak{a}))$ 用根表示的是 $I$ 的极大元；而 $V(\mathfrak{q}_i)$ 与 $V(\mathfrak{p}_i)$ 相等，却要写 $\mathfrak{p}_i$ 才能得到不可约分支——分清楚「$V$ 忽略幂次」与「$I$ 只取根」这枚硬币的两面。

**辨析｜不可约性：** Zariski 闭集的「不可约」与拓扑直观并不总是一致。$V(xy)$ 是两坐标轴的并，可约；$V(y - x^2)$ 是抛物线，不可约。判据是代数化的：$V(\mathfrak{a})$ 不可约当且仅当 $\sqrt{\mathfrak{a}}$ 是素理想。几何上「不能写成两个真闭集之并」，代数上「根是素理想」，两条完全等价——「几何不可约」这个词第一次在交换代数里被精确化。

**术语速查表**

| 术语 | 一句话含义 |
| --- | --- |
| 弱零点定理 | $k^n$ 的点 ↔ $k[x]$ 的极大理想 |
| 强零点定理 | $I(V(\mathfrak{a})) = \sqrt{\mathfrak{a}}$ |
| 消逝理想 $I(S)$ | 在 $S$ 上处处消失的多项式全体 |
| 零点集 $V(\mathfrak{a})$ | $\mathfrak{a}$ 的公共零点 |
| Zariski 拓扑 | 以 $V(\mathfrak{a})$ 为闭集，开集少而大 |
| 一般点 | 闭包为整个不可约子簇的素理想 |

## 5 小结

- **弱零点定理**：代数闭域上，$k^n$ 的点 ↔ $k[x]$ 的极大理想 $\mathfrak{m}_{\mathbf{a}} = (x_i - a_i)$。
- **强零点定理**：$I(V(\mathfrak{a})) = \sqrt{\mathfrak{a}}$；几何与根理想互为反演，幂次信息被几何丢掉。
- **Zariski 拓扑**：$\operatorname{Spec} A$ 以 $V(\mathfrak{a})$ 为闭集，基开集 $D(f)$；不可约子簇有一**般点**，素理想比极大理想信息更多。
- 零点定理 + 准素分解：分支 = 根的极小素因子，嵌入素理想只在代数侧。

在下一节，我们聚焦一类「完全被素理想管住」的整环：**离散赋值环与 Dedekind 整环**——那里每个理想唯一分解为素理想之积，算术基本定理以最纯粹的形式再生。
