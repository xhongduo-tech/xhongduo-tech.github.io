---
title: 局部指标公式（Connes–Moscovici 公式、JLO 上循环）
date: 2026-08-17
---

# 局部指标公式

<div class="epigraph">
<p>数学是一种特别适合处理任何一类抽象概念的工具，在这个领域中没有其他工具能与之相比。</p>
<footer>—— 保罗 · 狄拉克（Paul Dirac）</footer>
</div>

<div class="article-byline">
<p>第二级 · 非交换几何 ｜ Khalkhali《Basic Noncommutative Geometry》Ch.4; Connes《Noncommutative Geometry》Ch.IV ｜ 2026-08-17</p>
</div>

## 为什么从局部指标公式开始

指标定理是非交换几何的试金石。经典的 Atiyah–Singer 指标定理（1963）说：椭圆算子 $D$ 的**解析指标**（$\dim\ker D - \dim\operatorname{coker} D$，依赖分析）等于**拓扑指标**（由符号的拓扑类计算，依赖拓扑）。当空间变得非交换，拓扑指标仍由 K-理论定义，但「局部化」成了问题——经典证明依赖流形上的微局部分析，非交换空间没有坐标图。

Connes 与 Moscovici 在 1995 年的里程碑论文《The local index formula in noncommutative geometry》解决了这一问题：**用谱三元组 $D$ 的谱性质（维数谱、$\zeta$ 函数的留数）显式写出指标公式**，无需任何坐标。它的副产品——JLO 上循环与周期性循环上同调中的 Chern 特征——把第 5 节的循环上同调从「存在性」变成「可计算性」。本节是整个专题的技术顶点：前五篇的每一条线索（K-理论、循环上同调、谱三元组、微分结构）在这里交汇。

## 1 从 Atiyah–Singer 到非交换指标

### 1.1 经典指标

对紧自旋流形 $M$ 上的 Dirac 算子 $D$ 与向量丛 $E$，Atiyah–Singer 指标定理给出

$$
\operatorname{ind} D_E = \langle \operatorname{ch}(E)\, \hat{A}(M), [M] \rangle \in \mathbb{Z}
$$

其中 $\operatorname{ch}(E)$ 是 $E$ 的 Chern 特征，$\hat{A}(M)$ 是 $A$-罗赫类。<span class="marginnote">Atiyah–Singer 指标定理 1963 年由两人证明，被公认为二十世纪数学的重大成就之一；它统一了 Riemann–Roch 定理、Gauss–Bonnet 定理、符号差定理等一大族经典结果。Connes 非交换化它的动机之一正是把指标定理推广到叶子空间、轨道空间这类非交换空间。</span>

### 1.2 非交换版

在非交换框架中，指标变成 K-同调类（$D$ 的类）与 K-理论类（$E$ 的类）的配对：

$$
\operatorname{ind} D_E = \langle [D], [E] \rangle = \langle \operatorname{ch}_*([D]), \operatorname{ch}^*([E]) \rangle
$$

**Connes–Chern 特征（偶/奇）** 把 K-同调类映射到周期循环上同调：

$$
\operatorname{ch}_*: K_*(A) \longrightarrow HP^*(A)
$$

于是指标 = 循环上同调中的显式配对。问题只剩：**如何算出 $\operatorname{ch}_*([D])$？**

## 2 热核与 JLO 上循环

### 2.1 θ-可和与有限可和

谱三元组的正则性可以用热核刻画。若 $e^{-t D^2}$ 对 $t>0$ 是迹类算子，则称三元组为 **θ-可和（theta-summable）**；若 $(1+D^2)^{-p/2}$ 对某个 $p$ 是迹类，则为 **p-可和（p-summable）**。有限可和情形对应「有限维数」的谱三元组，是局部指标公式的适用前提。

### 2.2 JLO 上循环

1988 年，Jaffe、Lesniewski 与 Osterwalder（在超对称量子场论的研究中）构造了一个从 θ-可和谱三元组到周期循环上同调的 Chern 特征——**JLO 上循环（JLO cocycle）**：

$$
\operatorname{ch}^{\mathrm{JLO}}_n(D)(a_0, \ldots, a_n)
= \int_{\Delta_n} \operatorname{Tr}\left( a_0\, e^{-s_0 D^2}\, [D, a_1]\, e^{-s_1 D^2} \cdots [D, a_n]\, e^{-s_n D^2} \right) ds_1 \cdots ds_n
$$

其中 $\Delta_n$ 是 $n$-单形，$s_0 + \cdots + s_n = 1$。<span class="marginnote">JLO 上循环的每一项都是「热核传播子夹着交换子」的积分——这正是量子场论里 Feynman 图的解析对应。Jaffe–Lesniewski–Osterwalder 构造它是为了研究 $\theta$-可和 Fredholm 模的指标；Connes 随即认识到它就是循环上同调中的 Chern 特征。</span>

JLO 上循环给出了完整的 Chern 特征（所有维数），但对有限可和的谱三元组，它涉及无穷和，不便直接计算指标。

## 3 Connes–Moscovici 局部指标公式

### 3.1 维数谱与 $\zeta$ 函数

设 $(A, \mathcal{H}, D)$ 是正则有限可和谱三元组。对 $a \in A$、$n \in \mathbb{Z}$，定义

$$
\zeta_{a, n}(z) = \operatorname{Tr}\left( a\, [D, a_1]^{(k)} \cdots |D|^{-2z} \right)
$$

这些 $\zeta$ 函数在 $z=0$ 附近解析（除可数个极点），其极点的集合称为**维数谱（dimension spectrum）** $Sd$。**局部指标公式用这些 $\zeta$ 函数在 $z=0$ 的留数（即 Dixmier 迹）来计算指标**——这就是「局部」二字的含义：不需要全局拓扑信息，只看谱的渐近行为。

### 3.2 公式（一阶情形）

对一阶算子（$[D, a]$ 无高阶项）与 $n \ge 1$，Connes–Moscovici 定义

$$
\varphi^{(n)}(a_0, \ldots, a_n) = c_n\, \operatorname{Res}_{z=0} \operatorname{Tr}\left( a_0\, [D, a_1]^{(n)} \cdots [D, a_n]^{(n)}\, |D|^{-2z} \right)
$$

其中 $c_n$ 是组合常数，$[D, a]^{(n)}$ 表示 $\nabla^{(n)}$ 阶算子（涉及迭代交换子）。**定理（Connes–Moscovici 局部指标公式）**：存在有限支撑的上循环 $\varphi = \sum_n \varphi^{(n)} \in HC^\bullet(A)$，使得

$$
\operatorname{ind} D_E = \langle \varphi, [E] \rangle \qquad \text{对所有 } [E] \in K_*(A)
$$

并且 $\operatorname{ch}_*([D]) = [\varphi]$ 在周期循环上同调中成立。

### 3.3 公式解析：留数公式如何算指标

- **第一步**，指标 = 配对 $\langle \operatorname{ch}_*([D]), [E]\rangle$：这是第 5 节建立的框架；问题化为找 $\operatorname{ch}_*([D])$ 的显式代表元。
- **第二步**，用热核/$\zeta$ 函数重写：对经典 Dirac 算子，$\operatorname{Tr}(a |D|^{-2z})$ 在 $z=0$ 的留数与热核展开的系数成比例——这正是 Mellin 变换把「热核积分」变成「$\zeta$ 留数」。
- **第三步**，留数 = 局部不变量：Weyl 定律与热核展开（Gilkey 定理）保证这些留数只依赖 $D$ 的符号的**局部**数据。因此公式完全在谱三元组内部可算。
- **第四步**，组合验证：上循环条件（轮换不变性与 $b\varphi = 0$）由 $\zeta$ 函数的函数方程保证；$c_n$ 的选择使配对回到整数。

**一句话**：局部指标公式把「指标」写成「谱三元组的 $\zeta$ 留数的有限线性组合」，从而把 Atiyah–Singer 指标定理提升为完全非交换的陈述。

## 4 意义与影响

### 4.1 非交换空间的指标

局部指标公式直接应用于**叶子空间**（foliation）的纵向指标定理：Connes 用它重新证明了测度叶状结构的指标公式，并给出 Godbillon–Vey 不变量（三阶循环上同调类）的谱解释。<span class="marginnote">Godbillon–Vey 不变量是叶状结构理论里著名的（反常）示性类。Connes–Moscovici 公式把它实现为谱三元组（叶状结构的横截三元组）上的循环上同调类——这是局部指标公式最早的深刻应用之一，见 Connes《Noncommutative Geometry》Ch.III/IV。</span>

### 4.2 进一步的里程碑

- **Hopf 循环上同调**：Connes–Moscovici（1998）把局部指标公式推广到 Hopf 代数作用的横截几何（Hopf cyclic cohomology），Khalkhali 第 2 版新增了它的简介。
- **标量曲率**：局部指标公式的「二阶项」给出非交换流形的标量曲率；Connes–Moscovici 用之证明非交换环面的 Gauss–Bonnet 定理（结合 Tretkoff 等的工作），Khalkhali 第 2 版专门新增了弯曲非交换环面的标量曲率一节。
- **标准模型**：谱作用在标准模型中的应用（下一节）本质上是局部指标公式的物理化——把 $\operatorname{Tr}(f(D_A^2/\Lambda^2))$ 展开成几何项。

## 5 小结

- **非交换指标** = K-同调类与 K-理论类的配对 $\langle [D], [E]\rangle$；Chern 特征 $\operatorname{ch}_*: K_* \to HP^*$ 是核心工具。
- **JLO 上循环**（Jaffe–Lesniewski–Osterwalder 1988）：θ-可和谱三元组的 Chern 特征，由热核积分给出，连接量子场论与循环上同调。
- **Connes–Moscovici 局部指标公式**（1995）：用维数谱与 $\zeta$ 函数留数（Dixmier 迹）显式计算指标，无需坐标。
- 指标公式把循环上同调变成可计算工具，并给出叶子空间指标定理与 Godbillon–Vey 类的谱实现。
- 它同时催生了 Hopf 循环上同调与非交换环面的标量曲率/Gauss–Bonnet 定理。

在下一节，我们将从抽象的指标公式回到具体的非交换空间实例——**非交换环面与量子群**，看看前面建立的整套机器在 $A_\theta$ 与 $SU_q(2)$ 上如何运转。

<span class="marginnote">本文参考：Khalkhali《Basic Noncommutative Geometry》Ch.4; Connes–Moscovici《The local index formula in noncommutative geometry》(GAFA, 1995); Connes《Noncommutative Geometry》Ch.IV; JLO 原始文献见 Commun. Math. Phys. 118 (1988)。</span>