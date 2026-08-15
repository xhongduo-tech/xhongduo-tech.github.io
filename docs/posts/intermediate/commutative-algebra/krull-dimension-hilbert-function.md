---
title: 维数理论：Krull 维数与 Hilbert 函数
date: 2026-08-07
---

# 维数理论：Krull 维数与 Hilbert 函数

<div class="epigraph">
<p>维数不是预先给定的，它是环自身的结构数据。</p>
<footer>—— 克鲁尔（Wolfgang Krull）精神下的交换代数传统</footer>
</div>

<div class="article-byline">
<p>第二级 · 交换代数 ｜ Atiyah–Macdonald Ch. 11 ｜ 2026-08-07</p>
</div>

## 为什么从维数开始

几何上，「维数」是画布的大小；代数的挑战在于：**不给几何，纯从环定义维数**。Krull 的答案只用一条链：维数 = 素理想链的最大长度。这把「曲线是一维、曲面是二维」的直观，翻译成了纯代数的语言，并且出人意料地好用——它是正则性、深度、Cohen–Macaulay 性所有理论的地平线。<span class="marginnote">Wolfgang Krull（1899—1971），德国代数学家，1928 年在哥廷根提出素理想链维数，还引入了理想的根、完备化等一整套语言。他的维数概念把「几何维数」变成「组合的链长度」，简单到令人怀疑，却深刻到贯穿整个学科。</span>

另一个视角来自**计数**：$k[x,y]$ 里次数 ≤ n 的单项式个数以 $\sim n^2/2$ 增长，这个「体积增长速度」也给出维数 2。Hilbert 函数把「数数」精确化为多项式，而 **Hilbert–Samuel 多项式**把「数数」推广到任何 Noether 局部环——两条思路殊途同归：**维数 = 增长多项式的次数**。这一篇建立维数的两种来源及其一致性。

## 1 Krull 维数与高度

**Krull 维数**：环 $A$ 的维数

$$\dim A = \sup\{\, n \mid \mathfrak{p}_0 \subsetneq \mathfrak{p}_1 \subsetneq \cdots \subsetneq \mathfrak{p}_n \text{ 是 } A \text{ 的素理想链}\,\}.$$

**高度（height）**：素理想 $\mathfrak{p}$ 的维数

$$\operatorname{ht} \mathfrak{p} = \dim A_{\mathfrak{p}} = \sup\{\, n \mid \mathfrak{p}_0 \subsetneq \cdots \subsetneq \mathfrak{p}_n = \mathfrak{p}\,\}$$

（向下看的链长）。于是 $\dim A = \sup_{\mathfrak{p}} \operatorname{ht} \mathfrak{p} = \sup_{\mathfrak{m} \text{ 极大}} \dim A_{\mathfrak{m}}$。

标准值：
- $\dim k[x_1, \dots, x_n] = n$（最长的链 $(0) \subset (x_1) \subset \cdots \subset (x_1,\dots,x_n)$）。
- $\dim \mathbb{Z} = 1$：链 $(0) \subset (p)$。
- $\dim A = 0$ 当且仅当 $A$ 是 Artin 环（第1篇《链条件》已经见过）——素理想全是极大的。

**常见环的 Krull 维数一览：**

| 环 | 维数 | 说明 |
| --- | --- | --- |
| $k$（域） | 0 | 唯一素理想 $(0)$ |
| $k[x]$、$\mathbb{Z}$ | 1 | 主理想整环而非域 |
| $k[x,y]$ | 2 | 平面 |
| $k[x_1,\dots,x_n]$ | $n$ | $n$ 维仿射空间 |
| $k[x,y]/(xy)$ | 1 | 节点曲线 |
| $\mathbb{Z}[\sqrt{-5}]$ | 1 | Dedekind 整环（第1篇） |

竖着读这张表：维数从 0 到 $n$，靠的是「多加一个自由变量」；而「取商掉一个非零因子」往往降一维（超曲面 $k[x,y]/(xy)$ 就是 2→1）——这条「降维」直觉正是下一节高度定理与《正则局部环》参数系理论的前奏。

**重点：维数由「素理想链」定义，而素理想链在局部化、整扩张下有确定的行踪。** 例如 $\dim S^{-1}A \leq \dim A$；再如第1篇讲过 $A_{\mathfrak{p}}$ 的素理想恰是包含于 $\mathfrak{p}$ 的那些，故 $\operatorname{ht}\mathfrak{p}$ 就是 $\dim A_{\mathfrak{p}}$。<span class="marginnote">几何直觉：链 $V(\mathfrak{p}_0) \supsetneq V(\mathfrak{p}_1) \supsetneq \cdots$ 是「降维」的闭集序列，每条不可约闭集压进更小闭集一次就少一维——链越长，画布越大。</span>

**辨析｜易错点：** $\dim A$ 不一定等于「极大理想的高度」，除非 $A$ 是**维数均匀**的环（所有极大理想高度相同，如局部环、多项式环、域）。$A = k[x] \times k[x,y]$ 这样的积环里，不同极大理想高度不同——碰到「求 $\dim$」先找清楚有没有高度不同极大理想。

## 2 高度与重要不等式

素理想的高度不是随便取的。Noether 环境里有著名的限制：

**Krull 高度定理（Hauptidealsatz）**：Noether 环中，$\mathfrak{p}$ 是 $(x)$ 的极小素因子（$x \in A$）时，$\operatorname{ht} \mathfrak{p} \leq 1$；对 $n$ 个元素生成的理想 $\mathfrak{a} = (x_1, \dots, x_n)$，其每个极小素因子 $\mathfrak{p} \supseteq \mathfrak{a}$ 都有 $\operatorname{ht} \mathfrak{p} \leq n$。

**重点：「少生成元 ⇒ 低高度」。** 几何翻译：一条方程（一个函数）的零点集，每个不可约分支余维 ≤ 1——「一个方程在 $n$ 维空间里切出的东西维数 ≥ $n-1$」正是这个代数学事实。<span class="marginnote">Hauptidealsatz 是「主理想定理」，德语 Hauptideal = 主理想。它解释了为什么代数几何里「余维」比「绝对维数」更常出现：生成元个数天然给出高度的上界。</span>

另一个基石是**维数下界**：

$$\dim k[x_1,\dots,x_n]/\mathfrak{a} \;\geq\; n - r, \quad \text{若 } \mathfrak{a} \text{ 由 } r \text{ 个元素生成}.$$

两条合起来说明：**维数的变化被生成元个数牢牢框住**——这正是《维数理论深化与正则局部环》里「参数系」理论的前奏。

## 3 Hilbert 函数：数数与多项式

对标准分次环 $A = \bigoplus_{n \geq 0} A_n$（如 $A_0 = k$、由 $A_1$ 生成），定义 **Hilbert 函数**

$$H_A(n) = \dim_k A_n.$$

**Hilbert 定理**：若 $A$ 是 Noether 标准分次环，则存在多项式 $P \in \mathbb{Q}[t]$（**Hilbert 多项式**）使 $H_A(n) = P(n)$ 对所有足够大的 $n$ 成立，且

$$\deg P = \dim A - 1.$$

算例：$A = k[x_1,\dots,x_r]$ 标准分次，$H_A(n) = \binom{n + r - 1}{r - 1}$（$r-1$ 次多项式）；$A = k[x,y]/(xy)$ 时 $H_A(n) = n + 1$，一次多项式，而 $\dim A = 2$——**跳过一些分支，增长只降一度**。<span class="marginnote">「最终是多项式」这一事实的证明套路：对生成元 $x$ 造短正合列 $0 \to A(-1) \xrightarrow{\cdot x} A \to A/(x) \to 0$，把 $H_A$ 化归到维数更低的环，用归纳——即 Hilbert 用「消灭一个生成元」逐层剥。</span>

**重点：Hilbert 函数在有限项之后被多项式接管，首项系数 $e$（重数）与次数 $\dim - 1$ 都是不变量。** 次数差 1 的细节：标准分次环的 $A_0 = k$ 把「点」（维数 0）记为常数，所以从 $\dim$ 减一。

## 4 Hilbert–Samuel 多项式：局部环的维数

局部情形不需要分次结构，改用「余长度」。设 $(R, \mathfrak{m})$ Noether 局部环，$M$ 有限生成 $R$-模，$\lambda$ 表 $R$-模的**长度**（Artin 模的合成列长度）。

**Hilbert–Samuel 定理**：存在多项式 $P_{M}(t) \in \mathbb{Q}[t]$，使

$$\lambda(M / \mathfrak{m}^{n+1} M) = P_{M}(n) \qquad (n \gg 0),$$

且 $\deg P_{M} = \dim M$（$\dim M$ 为 $\operatorname{Supp} M$ 的维数），首项系数为 $e(M)/d!$，$e(M)$ 叫**重数（multiplicity）**。<span class="marginnote">余长度 $\lambda(N)$ 就是「$N$ 有多长」，对 $\mathbb{Z}$-模即阶数大小。Hilbert–Samuel 多项式把「$\mathfrak{m}$ 的幂逐渐杀死模」时的体积增长记录下来——这是「局部环的体积」概念，$e(M)$ 是体积的规范化首项。</span>

**重点：维数 = Hilbert–Samuel 多项式的次数；重数 = 规范化的首项系数。** 对 $R = k[x,y]_{(x,y)}$，$\mathfrak{m} = (x,y)$，$R/\mathfrak{m}^{n+1}$ 以单项式 $x^a y^b$（$a+b \leq n$）为基，故

$$\lambda(R/\mathfrak{m}^{n+1}) = \frac{(n+1)(n+2)}{2} = \frac{n^2 + 3n + 2}{2},$$

二次多项式，$\deg = 2 = \dim R$，重数 $e = 1$——与图画一致。

![Krull 维数：素理想链与 Hilbert–Samuel 多项式](/images/commutative-algebra/krull-dimension-hilbert.svg)

## 5 公式解析：两种维数为何一致

把两条路放在一起看。设 $(R, \mathfrak{m})$ Noether 局部环，维数 $d = \dim R$，则

$$\lambda(R/\mathfrak{m}^{n+1}) \;=\; \frac{e(R)}{d!}\, n^{d} \;+\; (\text{低次项}), \qquad n \to \infty.$$

- **第一步，为什么是多项式**：$\mathfrak{m}^{n+1}$ 的幂构成降链，用「短正合列 + 生成元剥层」的归纳（与 Hilbert 函数同套路），把 $\lambda(R/\mathfrak{m}^{n+1})$ 归结为更小维数环的同类量。
- **第二步，为什么次数恰是 $d$**：链的几何观——$\operatorname{Spec} R$ 是「一个点附近的 $d$ 维空间」，$\lambda$ 数的是「以 $\mathfrak{m}$ 为原点、幂为半径的体积」，半径 $n$ 的 $d$ 维体积当然是 $n^d$ 量级。$k[x,y]_{(x,y)}$ 的 $\sim n^2/2$ 就是平面里「$\frac12 n^2$」这块三角形的面积。
- **第三步，重数即体积**：首项 $e(R)/d!$ 在局部化下行为良好（$e(R) = e(R_{\mathfrak{p}})$ 在合适条件下），它是环「在该点有多厚」的度量——正则局部环处处 $e = 1$，奇点则 $e > 1$。<span class="marginnote">「重数 > 1 就是奇点」这条线在代数几何里极重要：$k[x,y]/(y^2 - x^3)$ 在原点的重数 2，反映「尖端」；《维数理论深化》中会用 $\dim_k \mathfrak{m}/\mathfrak{m}^2$ 定义正则性，与重数互相印证。</span>

**辨析｜易错点：** 两种 Hilbert 多项式的次数差 1，别混。**标准分次环**（$A_0 = k$）的 $H_A(n) = \dim_k A_n$ 渐近次数是 $\dim A - 1$；**局部环**的 Hilbert–Samuel $\lambda(R/\mathfrak{m}^{n+1})$ 渐近次数是 $\dim R$。原因在索引：分次按「度」数、局部按「幂」数，前者把常数项（维数 0）算作次数 0，后者把「点」算作常数——对照用图表的标题即可记住。

## 6 小结

- **Krull 维数** = 素理想链最大长度；**高度** $\operatorname{ht}\mathfrak{p} = \dim A_{\mathfrak{p}}$；$\dim k[x_1,\dots,x_n] = n$，$\dim \mathbb{Z} = 1$。
- **高度定理**：$n$ 个元素生成的理想，其极小素因子高度 ≤ $n$——生成元个数框住维数。
- **Hilbert 函数**渐近为多项式，次数 = $\dim - 1$（标准分次环）；**Hilbert–Samuel** 次数 = $\dim$（局部环）。
- **重数** $e(M)$ 是规范化的首项系数，正则点重数 1、奇点 > 1。
- 维数 = 链长度 = 增长多项式次数，两种定义殊途同归。

在下一节，维数成为主角：**系统参数、嵌入维数、正则局部环**——「维数最小可能的环」就是几何里的光滑点。
