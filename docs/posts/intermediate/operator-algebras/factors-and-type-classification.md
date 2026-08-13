---
title: 因子与类型的分类
date: 2026-08-07
---

# 因子与类型的分类

<div class="epigraph">
<p>与一个深刻真理相对立的，很可能是另一个深刻真理。</p>
<footer>—— 尼尔斯 · 玻尔（Niels Bohr）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Kadison & Ringrose《Fundamentals of the Theory of Operator Algebras》第8章 ｜ 2026-08-07</p>
</div>

## 为什么从因子分类开始

前两篇把 von Neumann 代数装进了拓扑与对偶的框架。但 von Neumann 代数实在太多——$B(\mathcal{H})$ 的弱闭子代数「海了去」。分类从哪里下手？冯·诺依曼与默里（Murray）1936 年给出革命性的回答：**先按「中心」切碎**，von Neumann 代数被切成「原子」——**因子（factor）**——然后再按「投影的维数函数」给因子分型。这就是本文的主角：**因子的类型分类**。

这个分类之所以震撼，是因为它把看似无穷多样的 von Neumann 代数，压缩进一个从「离散」到「连续」到「超连续」的谱系：I 型（矩阵式的）、II₁ 型（连续有限维）、II∞ 型（连续无穷维）、III 型（没有迹、最「量子」）。理解因子分类，就是拿到 von Neumann 代数世界的地图——四十年后 Connes 对 III 型的精细分类，仍是当代算子代数与量子场论交汇处最深刻的成就之一。

## 1 因子：von Neumann 代数的原子

**中心（center）**：$\mathcal{Z}(\mathcal{M})=\mathcal{M}\cap\mathcal{M}'$，即「与 $\mathcal{M}$ 所有元素都交换」的元素构成的可交换 von Neumann 代数。

**因子（factor）**：中心只含标量（$\mathcal{Z}(\mathcal{M})=\mathbb{C}1$）的 von Neumann 代数。

**定理（因子分解）**：von Neumann 代数可以「按中心分解」成因子的直积分（第 24 篇的直接积分）。**因子是研究的原子**——先懂因子，再拼出一般 von Neumann 代数。<span class="marginnote">因子对量子系统意味着「不可再约化的观察代数」：中心平凡 = 没有「守恒超选择荷」能把它劈开。物理里一个因子的 von Neumann 代数对应一个「不可约相」；第 24 篇的中心分解正是「把混合相拆成纯相」的数学操作。</span>

**例子**：
$B(\mathcal{H})$ 是因子（中心 $\mathbb{C}1$）；
$M_n$ 是因子；
- $L^\infty(X,\mu)$（作为乘法算子代数）**不是**因子（中心 = 自己，很大）；
- 第 21 篇的 $W^*(T)$（$T$ 正规）交换，一般不是因子。

## 2 投影的比较：维数函数的由来

在因子里，投影（orthogonal projections）扮演「子空间」的角色。**Murray–von Neumann 等价**（第 17 篇的 $p\sim q$）给投影分层，于是可以比较：

**偏序**：$p\preceq q$ 若 $p$ 等价于 $q$ 的某个子投影。

**命题（比较定理）**：在因子中，任意两个投影 $p,q$ **可比**：要么 $p\preceq q$，要么 $q\preceq p$。<span class="marginnote">「任意两个投影可比」是因子独有的性质（一般 von Neumann 代数不成立），它让「投影大小」成为一个完全有序的量。这个性质来自中心平凡：若 $p,q$ 不可比，它们的「尺寸差」会造出中心元素，矛盾于因子定义。比较定理是整个类型分类的根基。</span>

**维数函数（dimension function）**：在因子 $\mathcal{M}$ 上，存在映射 $d:\mathrm{Proj}(\mathcal{M})\to[0,\infty]$（或 $[0,1]$），满足 $d(p\sim q)\Rightarrow d(p)=d(q)$、$d(p\oplus q)=d(p)+d(q)$（正交直和），且 $d$ 唯一（差一个常数）。**$d(p)$ 就是「投影的维数」**，是有限维秩的自然推广。

**投影的有限性**：$p$ **有限（finite）**若 $p\sim q\le p$ 推出 $q=p$（不能「真包含却等价」）；否则**无穷（infinite）**。

## 3 类型的谱系：I、II、III

按「有限投影的维数函数取值」，因子被分为四大类型：

| 类型 | 维数函数取值 | 标志性质 | 典型例子 |
| --- | --- | --- | --- |
| I$_n$ | $\{0,1,\dots,n\}$ | 有原子投影，离散 | $M_n$ |
| I$_\infty$ | $\{0,1,2,\dots,\infty\}$ | 有原子投影，无穷 | $B(\mathcal{H})$ |
| II$_1$ | $[0,1]$ | 有限迹（$\tau(1)=1$）连续维数 | 超有限 II$_1$ 因子 $\mathcal{R}$ |
| II$_\infty$ | $[0,\infty]$ | 无原子，有迹但无穷 | $\mathcal{R}\otimes B(\mathcal{H})$ |
| III | $\{0,\infty\}$ | 无迹态，投影全无穷 | Powers 因子、量子场论真空代数 |

**I 型**：有**原子投影**（$p$ 不可再分：$q\le p\Rightarrow q=0$ 或 $q=p$）。I$_n$ 因子 ≅ $M_n$，I$_\infty$ 因子 ≅ $B(\mathcal{H})$（可分离时）。**I 型 = 矩阵世界**。<span class="marginnote">I 型因子恰好是「有纯正常态、有极小投影」的那些——量子力学里有限系统（$M_n$）与单粒子（$B(\mathcal{H})$）都是 I 型。它们对应「可分解成特征空间直和」的可观测量，即第 13 篇谱定理适用的完整范围。</span>

**II$_1$ 型**：有有限的**迹态** $\tau$（$\tau(1)=1$，$\tau(ab)=\tau(ba)$）且无原子投影，维数函数连续取 $[0,1]$。**II$_1$ = 连续维数的有限世界**。最著名的 II$_1$ 因子是**超有限 II$_1$ 因子** $\mathcal{R}$（可由有限维逼近，第 17 篇 AF 语言的 von Neumann 版）。

**II∞ 型**：II$_1$ 因子与 $B(\mathcal{H})$（无穷维）的张量积；有迹（可取值 $\infty$）但无原子。

**III 型**：**没有**任何非零有限投影，也没有迹态——投影要么 0 要么无穷。III 型因子最「神秘」、最非交换，是量子场论里局部可观测代数的自然类型。

## 4 公式解析：维数函数 $d$

$$
d:\ \mathrm{Proj}(\mathcal{M})\to[0,\infty], \qquad d(p\oplus q)=d(p)+d(q), \qquad p\sim q\Rightarrow d(p)=d(q)
$$

- **第一步，看定义域**：$d$ 只活在「投影」上，不活在一般算子上。投影 = 闭子空间 = 「态空间的区域」，$d$ 给每个区域一个「体积/维数」。
- **第二步，看可加性**：正交直和 $p\oplus q$ 的维数 = 两维数之和——「不重叠区域的体积相加」，这是维数最本色的公理。
- **第三步，看等价不变量**：$p\sim q$（Murray–von Neumann 等价）时 $d(p)=d(q)$——维数只认「形状等价类」，不认「具体位置」。这让 $d$ 在因子中（比较定理）完全确定。
- **第四步，看它如何切分类型**：$d$ 的值域是 $\{0,\dots,n\}$ → I$_n$；$[0,1]$ → II$_1$；$[0,\infty]$ → II$_\infty$；$\{0,\infty\}$ → III。**类型的差异 = 维数函数值域的形状差异**：离散、连续有界、连续无界、退化。一条公理（$d$）切出整个世界。

## 5 类型分类的现代回响

**Connes 对 III 型的分类**：1970 年代，Connes 用 **Tomita–Takesaki 模理论**（给定态 $\varphi$，其模算子 $\Delta_\varphi$ 给出「时间演化」$\sigma_t^\varphi$）把 III 型因子再细分：III$_0$、III$_\lambda$（$0\lt \lambda\lt 1$）、III$_1$，并证明 **超有限 III 型因子可由模谱（$\Delta_\varphi$ 的谱）与 Flow 分类**。<span class="marginnote">III 型因子「没有迹」，Connes 用「模的自同构群」替代迹来分类：每个忠实正常态 $\varphi$ 给出模自同构 $\sigma_t^\varphi$，其「固定点的行为」区分 III$_0$/III$_\lambda$/III$_1$。量子场论真空态对应的因子几乎总是 III$_1$——这解释了为何量子场论里没有「绝对粒子数守恒」，一切都是相对论性真空涨落。</span>

**超有限因子分类的完成**：Connes（1976）证明**超有限 II$_1$ 因子唯一**（$\mathcal{R}$，Murray–von Neumann 早已猜测），**超有限 II∞、III$_\lambda$（$\lambda\ne1$）也各唯一**；Haagerup 完成 III$_1$ 的超有限分类。四十年一役，超有限因子的分类全部收官——**分类理论最辉煌的篇章之一**。

**应用（量子场论与凝聚态）**：相对论性 QFT 的局部代数在真空态下是 III$_1$ 因子（Haag–Kastler 公理）；拓扑序、任意子的数学描述也进入因子语言。II$_1$ 因子与 $\mathcal{R}$ 是量子信息「子因素」理论（Jones 指数、畴壁）的家园。

**辨析｜易错点：**类型分类是对**因子**而言的，一般 von Neumann 代数要先按中心分解成因子（第 24 篇）再谈类型；且**类型不依赖具体表示**（抽象 $W^*$-性质）。另一个易错点：II$_1$ 有「有限迹」但 III 没有——「有没有迹」是 II/III 的分水岭，把「III 型 = 没有迹」记牢，就能绕开一半误区。

## 6 例：类型判断速成

把「这是哪一型」的判断流程走一遍，分类就不再是名词堆砌。

**$A=M_n$（矩阵）**：有极小投影（秩一投影 $E_{ii}$）、有限迹 $\frac1n\mathrm{Tr}$、维数函数取 $\{0,1,\dots,n\}$。I$_n$ 型。

**$A=B(\mathcal{H})$（$\mathcal{H}$ 无穷维）**：有极小投影（秩一投影）、维数函数取 $\{0,1,2,\dots,\infty\}$。I$_\infty$ 型。

**$A=\mathcal{R}$（超有限 II$_1$）**：无极小投影（每个投影还能再分）、有有限迹 $\tau(1)=1$、维数连续取 $[0,1]$。II$_1$ 型。

**$A=\mathcal{R}\otimes B(\mathcal{H})$**：无原子、有迹（可取 $\infty$）、维数取 $[0,\infty]$。II$_\infty$ 型。

**$A=$ 量子场论真空代数**：无迹态、所有非零投影都无穷、维数退化为 $\{0,\infty\}$。III 型（通常 III$_1$）。

**一句话总结**：看三点——有无原子投影、有无有限迹、维数函数值域——类型就定了。I/II/III = 离散/连续有限/连续无穷/退化。

## 7 延伸：Murray–von Neumann 等价与维数

类型分类的引擎是投影的 Murray–von Neumann 等价，值得单独加固。

**等价 $\sim$**：$p\sim q$ ⟺ 存在部分等距 $v$，$v^*v=p$、$vv^*=q$。这推广了「子空间同构」：$p\mathcal{H}\cong q\mathcal{H}$。

**为什么需要「部分等距」**：等价不是「$p,q$ 相等」而是「$p\mathcal{H}$ 与 $q\mathcal{H}$ 等距同构」。无穷维子空间可以和真子空间同构——这就是「无穷投影」与「$p\sim q<p$」的来源。

**比较定理**：因子中任意两投影可比（$p\preceq q$ 或 $q\preceq p$）。「尺寸」在因子中是全序——维数函数 $d$ 由此良定义。

**维数函数的构造**：先给极小投影定义 $d=1$（I 型），再用「分拆成 $n$ 份」定义有理数维数（II$_1$），最后取极限得连续维数。II$_1$ 的「$d(p)\in[0,1]$」是「连续维数」的精确含义。

**物理直觉**：II$_1$ 因子是「没有原子的连续维数世界」——量子信息里，Jones 指数的无理数维数（$\mathcal{R}$ 子因素）是「非整数维」的物理实现。

## 8 延伸：超有限因子与分类的完成

超有限（hyperfinite）因子是分类理论最辉煌的篇章。

**超有限（AFD）**：von Neumann 代数可由有限维子代数强逼近（C\* 里 AF 的 von Neumann 版）。$\mathcal{R}$、$\mathcal{R}\otimes B(\mathcal{H})$、以及超有限 III 型因子都超有限。

**Murray–von Neumann 猜想**：超有限 II$_1$ 因子唯一（就是 $\mathcal{R}$）。1930 年代提出，直到 Connes 1976 年用「injectivity」才证明——半个世纪的悬案。

**Connes 的证明**：injectivity（单射性）⟺ 超有限（AFD）；injective II$_1$ 因子唯一。Connes 同时证明超有限 II$_\infty$ 唯一、超有限 III$_\lambda$（$\lambda\neq1$）唯一。

**Haagerup 收官**：超有限 III$_1$ 因子唯一（Haagerup）。至此，所有超有限因子的分类全部完成——一个漂亮而完整的谱系。

**意义**：超有限因子出现在量子场论（III 型）、量子信息（$\mathcal{R}$ 子因素）、遍历理论（group measure space）。分类完成意味着「这些物理对象终于数清了」。

**一句话总结**：超有限因子分类 = 四十年一役——Murray–von Neumann 猜想、Connes 的 injectivity、Haagerup 收官，超有限世界从此全部入列。

## 9 小结

- **因子**：中心 $\mathbb{C}1$ 的 von Neumann 代数；比较定理：因子中任意两投影可比。
- **维数函数** $d$：投影上的可加、等价不变、几乎唯一的「维数标尺」。
- **四大类型**：I（离散，矩阵式）、II$_1$（连续有限、有迹）、II∞（连续无穷、有迹）、III（无迹、投影全无穷）。
- **旗舰例子**：$M_n$、$B(\mathcal{H})$、超有限 II$_1$ 因子 $\mathcal{R}$、$\mathcal{R}\otimes B(\mathcal{H})$、量子场论 III$_1$