---
pageClass: plain-doc
---

# 实变函数与测度论

实变函数论是数学分析的深化：以集合论与测度论为工具，重建积分理论（Lebesgue 积分），并研究函数的可测性、可积性与微分结构。本篇对标周民强《实变函数论》与那汤松《实变函数论》的章节体系，从集合与基数讲起，经测度、可测函数、Lebesgue 积分，直至微分与不定积分、L^p 空间。

## 主题规划

<ProgressGrid cat="intermediate/real-analysis" />


### 第一篇 集合与基数

- [x] [集合及其运算：并、交、差、补与德摩根律](./set-operations)
- [x] [集合列的极限：上限集与下限集](./set-sequence-limits)
- [x] [映射与对等：集合间的一一对应](./mapping-and-equivalence)
- [x] [可数集：定义、性质与典型例子（有理数集、代数数集）](./countable-sets)
- [x] [不可数集：实数集的不可数性与康托尔对角线法](./uncountable-sets)
- [x] [基数（势）的概念与比较：伯恩斯坦定理](./cardinality)
- [x] [连续基数：c 与 ℵ₀ 的关系，无最大基数定理](./continuum-cardinality)
- [x] [连续统假设简介](./continuum-hypothesis)

### 第二篇 点集拓扑初步

- [x] [n 维欧氏空间 Rⁿ：距离、邻域与收敛点列](./euclidean-space-rn)
- [x] [内点、聚点与边界点](./interior-cluster-boundary-points)
- [x] [开集：定义、性质与直线上开集的构造定理](./open-sets)
- [x] [闭集：定义、性质及与开集的对偶关系](./closed-sets)
- [x] [开核、导集与闭包](./interior-derived-set-closure)
- [x] [康托尔（Cantor）集：构造、性质与康托尔函数](./cantor-set)
- [x] [完备集与疏朗集（无处稠密集）](./perfect-nowhere-dense-sets)
- [x] [Borel 集类与 Fσ、Gδ 型集](./borel-sets-fsigma-gdelta)
- [x] [覆盖定理：有限覆盖定理与可数覆盖](./covering-theorems)
- [x] [距离空间中的连续映射与点集间距离](./continuity-in-metric-spaces)

### 第三篇 Lebesgue 测度

- [x] [外测度：定义（开覆盖下确界）与基本性质](./outer-measure)
- [x] [有界集外测度的等价刻画](./outer-measure-bounded-sets)
- [x] [可测集：Carathéodory 条件的定义](./measurable-sets-caratheodory)
- [x] [可测集的运算封闭性：σ-代数的建立](./sigma-algebra-measurable-sets)
- [x] [开集、闭集与 Borel 集的可测性](./measurability-open-closed-borel)
- [x] [测度的可数可加性与连续性](./countable-additivity-continuity)
- [x] [可测集的逼近：用开集、闭集与 Gδ、Fσ 集逼近](./approximation-measurable-sets)
- [x] [不可测集：Vitali 集的构造](./vitali-set)
- [x] [乘积空间的测度：Rⁿ 中测度与低维截面的关系](./product-measure-rn)
- [x] [正测度集与区间：Steinhaus 定理](./steinhaus-theorem)

### 第四篇 可测函数

- [x] [可测函数：定义与等价条件](./measurable-functions)
- [x] [可测函数的运算：四则运算与格运算的封闭性](./operations-measurable-functions)
- [x] [可测函数列的极限：上确界、下确界与上下极限函数](./limits-of-measurable-functions)
- [x] [简单函数及其对可测函数的逼近](./simple-functions-approximation)
- [x] [几乎处处（a.e.）概念与几乎处处收敛](./almost-everywhere)
- [x] [叶戈罗夫（Egorov）定理：a.e. 收敛与近一致收敛](./egorov-theorem)
- [x] [依测度收敛：定义与性质](./convergence-in-measure)
- [x] [依测度收敛与几乎处处收敛的关系：Riesz 定理](./riesz-theorem-convergence)
- [x] [鲁津（Luzin）定理：可测函数与连续函数的关系](./luzin-theorem)
- [x] [鲁津定理的另一形式与推论](./luzin-theorem-variants)

### 第五篇 Lebesgue 积分

- [x] [非负简单函数的 Lebesgue 积分](./integral-simple-functions)
- [x] [非负可测函数的积分：定义与初等性质](./integral-nonnegative-functions)
- [x] [一般可测函数的积分：正部、负部与可积性](./integral-general-functions)
- [x] [Lebesgue 积分的基本性质：线性、单调性与绝对连续性](./properties-lebesgue-integral)
- [x] [积分的可数可加性（对积分区域）](./countable-additivity-integral)
- [x] [与黎曼积分的关系：R 可积必 L 可积且相等](./riemann-lebesgue-integral)
- [x] [黎曼可积的充要条件：间断点集为零测集](./riemann-integrable-iff)
- [x] [L 积分与广义（反常）R 积分的比较](./improper-riemann-lebesgue)
- [x] [重积分与累次积分：Fubini 定理与 Tonelli 定理](./fubini-tonelli)
- [x] [积分的几何意义：可测函数的下方图形](./integral-geometric-meaning)

### 第六篇 积分号下取极限（积分极限定理）

- [x] [Levi 单调收敛定理（Lebesgue 逐项积分定理）](./levi-monotone-convergence)
- [x] [逐项积分与级数的积分](./termwise-integration-series)
- [x] [Fatou 引理：叙述、证明与不可改进之处](./fatou-lemma)
- [x] [Lebesgue 控制收敛定理](./dominated-convergence)
- [x] [控制收敛定理的推论：有界收敛定理与积分号下求极限、求导](./dominated-convergence-consequences)
- [x] [三大定理的相互关系与典型应用](./three-theorems-applications)

### 第七篇 微分与不定积分

- [x] [单调函数的可微性：Lebesgue 定理](./monotone-functions-differentiability)
- [x] [Dini 导数与单调函数导数的存在性](./dini-derivatives)
- [x] [Vitali 覆盖引理及其应用](./vitali-covering-lemma)
- [x] [有界变差函数：定义与基本性质](./bounded-variation)
- [x] [有界变差函数的 Jordan 分解定理](./jordan-decomposition)
- [x] [绝对连续函数：定义与性质](./absolutely-continuous-functions)
- [x] [不定积分：L 可积函数的不定积分是绝对连续的](./indefinite-integral)
- [x] [Lebesgue 微分定理：不定积分的导数几乎处处等于被积函数](./lebesgue-differentiation-theorem)
- [x] [牛顿-莱布尼茨公式成立的充要条件](./newton-leibniz-ftc)
- [x] [分部积分与换元积分在 L 积分中的推广](./integration-by-parts-substitution)
- [x] [奇异函数与 Lebesgue 分解简介](./singular-functions-lebesgue-decomposition)

### 第八篇 Lᵖ 空间

- [x] [Lᵖ 空间的定义：p 次可积函数类](./lp-spaces-definition)
- [x] [Hölder 不等式](./holder-inequality)
- [x] [Minkowski 不等式与 Lᵖ 的线性空间结构](./minkowski-inequality)
- [x] [Lᵖ 范数与依范数收敛（p 次平均收敛）](./lp-norm-convergence)
- [x] [Lᵖ 空间的完备性：Riesz–Fischer 定理](./riesz-fischer-theorem)
- [x] [依范数收敛与依测度收敛、a.e. 收敛的关系](./convergences-relations)
- [x] [Lᵖ 中的稠密性：连续函数与简单函数的稠密性](./density-in-lp)
- [x] [L² 空间：内积、正交性与 Riesz–Fischer 理论](./l2-space)
- [x] [L∞ 空间与本性有界函数](./l-infinity-space)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

### 第1篇

- [x] [集合与点集（基数、Rn 中开闭集与博雷尔集）](./intermediate-real-analysis-rn-b769e69d.md)
- [x] [Lebesgue 测度（外测度、可测集、不可测集）](./intermediate-real-analysis-lebesgue-18e631d6.md)
- [x] [可测函数（收敛模式：几乎处处/依测度/一致）](./intermediate-real-analysis-a0924d83.md)
- [x] [Lebesgue 积分（定义、三大收敛定理、与黎曼积分比较）](./intermediate-real-analysis-lebesgue-2471f001.md)
- [x] [微分与不定积分（单调函数、有界变差、绝对连续）](./intermediate-real-analysis-67b97a12.md)
- [x] [Lp 空间（Hölder/Minkowski 不等式、完备性、对偶）](./intermediate-real-analysis-lp-hlder-minkowski-7a30834b.md)
- [x] [抽象测度与积分（符号测度、Radon-Nikodym 定理）](./intermediate-real-analysis-radon-nikodym-e9c39073.md)
- [x] [乘积测度与 Fubini 定理（乘积空间、卷积应用）](./intermediate-real-analysis-fubini-7f927008.md)
