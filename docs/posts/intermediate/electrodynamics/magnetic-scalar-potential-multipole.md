---
title: 磁标势与磁多极展开
date: 2026-08-07
---

# 磁标势与磁多极展开

<div class="epigraph">
<p>在没有电流的地方，磁场也能像电场一样拥有标量势。</p>
<footer>—— 西缅 · 德尼 · 泊松（Siméon Denis Poisson）</footer>
</div>

<div class="article-byline">
<p>第二级 · 电动力学 ｜ 郭硕鸿《电动力学》第三章 §3.3 ｜ 2026-08-07</p>
</div>

## 为什么磁场也能有「标势」

上一节我们用矢势 $\mathbf{A}$ 描述磁场，代价是三个分量和规范自由度。但有一个重要的特例可以大幅简化：**在电流密度为零的区域**（$\mathbf{J} = 0$），安培环路定理退化为 $\nabla\times\mathbf{H} = 0$——磁场无旋，于是像静电场一样存在**磁标势（magnetic scalar potential）$\varphi_m$**：

$$\mathbf{H} = -\nabla\varphi_m$$

磁标势在无源静磁问题（永磁体、磁介质、无电流包围的空间）中把问题重新拉回熟悉的标量泊松/拉普拉斯方程，整套静电解法（分离变量、镜像、多极展开）都能平移使用。<span class="marginnote">「无电流区域」在实际中极其常见：永磁体外部、电磁铁气隙、地球磁场的磁层内部。甚至电流本身产生的磁场，在导线<strong>之外</strong>的区域也是无旋的——只要不穿过电流线，磁标势就能用。</span>

## 1 磁标势的引入与限制

**引入**：$\mathbf{J} = 0$ 区域中 $\nabla\times\mathbf{H} = 0$，定义 $\mathbf{H} = -\nabla\varphi_m$（负号沿用电场的习惯）。由 $\nabla\cdot\mathbf{B} = \mu\nabla\cdot\mathbf{H} = 0$（均匀介质中）得

$$\nabla^2\varphi_m = 0$$

——磁标势满足**拉普拉斯方程**。于是无源静磁问题与无源静电问题数学上完全同构，静电里练熟的分离变量法可直接套用。

**磁标势的多值性（关键限制）**：静电标势单值（保守场），但磁标势有一个致命问题——**它可能是多值的**。考虑一根载流无限长直导线，在导线外取 $\varphi_m$，沿导线绕一圈回到原点：磁场 $\mathbf{H}$ 沿回路环量为 $I$（安培环路定理），但 $\varphi_m$ 的净变化为 $\oint \nabla\varphi_m\cdot\mathrm{d}\mathbf{l} = -\oint\mathbf{H}\cdot\mathrm{d}\mathbf{l} = -I \neq 0$——绕一圈，$\varphi_m$ 少了 $I$。<span class="marginnote">处理办法：把「绕导线一圈」的路径禁掉——引入一个<strong>割面（barrier surface）</strong>，规定 $\varphi_m$ 在割面上有跳跃 $I$，禁止路径穿过割面。这是数学上「多值函数的分支切割」在物理中的直接出现，与复变函数中 $\ln z$ 沿分支切割跳跃 $2\pi i$ 完全同构。</span>

**辨析｜易错点：** 磁标势只在**不含电流**的单连通区域成立。区域若被电流「穿洞」（如环电流的环形区域），要么用割面修补多值性，要么干脆退回矢势。判断「能不能用磁标势」的唯一标准是：**所研究的区域内 $\mathbf{J}$ 是否处处为零**。

## 2 磁多极展开

远场区（观测距离远大于电流分布线度），矢势可做**多极展开（multipole expansion）**。对 $\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi}\int\frac{\mathbf{J}(\mathbf{r}')}{r}\,\mathrm{d}V'$ 把 $1/|\mathbf{r}-\mathbf{r}'|$ 展开，最低阶非零项是磁偶极子项：

$$\mathbf{A}(\mathbf{r}) \approx \frac{\mu_0}{4\pi}\frac{\mathbf{m}\times\hat{\mathbf{r}}}{r^2}$$

其中**磁偶极矩（magnetic dipole moment）**

$$\mathbf{m} = \frac{1}{2}\int \mathbf{r}'\times\mathbf{J}(\mathbf{r}')\,\mathrm{d}V'$$

对平面闭合线圈，$\mathbf{m} = I\mathbf{S}$（$I$ 为电流，$\mathbf{S}$ 为线圈面积矢量，方向由右手定则）。<span class="marginnote">与静电类比：静电多极展开的最低非零阶是电荷（单极子）；磁场因无磁单极子，最低阶是偶极子。所以「电流环的远场像磁偶极子」，而「带电点的远场像电荷」。磁偶极子是磁场的「原子」，永磁体、地球磁场、电子自旋的宏观表现全是偶极子叠加。</span>

**磁偶极子的磁场**：由 $\mathbf{B} = \nabla\times\mathbf{A}$ 得

$$\mathbf{B} = \frac{\mu_0}{4\pi r^3}\left[3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{m}\right]$$

——与电偶极子的远场公式形式完全相同。沿偶极矩方向（$\theta = 0$）$B = \dfrac{\mu_0 m}{2\pi r^3}$，垂直方向（$\theta = \pi/2$）$B = \dfrac{\mu_0 m}{4\pi r^3}$。

## 3 公式解析：磁偶极矩 $\mathbf{m} = \frac{1}{2}\int\mathbf{r}'\times\mathbf{J}\,\mathrm{d}V'$

这条定义式初看没头没脑，拆开看：

- **第一步，回顾展开起点**：远场展开 $\mathbf{A} \approx \frac{\mu_0}{4\pi}\frac{1}{r}\int\mathbf{J}\,\mathrm{d}V'$ 的零阶项 $\int\mathbf{J}\,\mathrm{d}V'$ 恰好为零——因为稳恒电流无散，$\int J_i\,\mathrm{d}V = 0$（可用分部积分 + $\nabla\cdot\mathbf{J}=0$ 验证）。所以磁场的多极展开**没有单极子项**，最低阶必须展开到 $1/r^2$。
- **第二步，展开到 $1/r^2$ 阶**：把 $1/|\mathbf{r}-\mathbf{r}'| \approx 1/r + \mathbf{r}\cdot\mathbf{r}'/r^3$ 代入，第二项给出 $\mathbf{A}_1 = \frac{\mu_0}{4\pi r^3}\int \mathbf{J}(\mathbf{r}\cdot\mathbf{r}')\,\mathrm{d}V'$。这个积分用恒等式 $\int J_i r_j'\,\mathrm{d}V' = -\frac{1}{2}\epsilon_{ijk}m_k$（$m_k$ 即偶极矩）化简——叉乘项就是这里跑出来的。<span class="marginnote">这一步是全篇唯一硬核的数学：把一个「$\mathbf{J}$ 与坐标乘积」的积分，用分部积分 + 无散条件改写成反对称组合，恰好凑出 $\mathbf{r}'\times\mathbf{J}$。这个技巧在电四极矩、以及后面辐射场的多极展开中会反复出现。</span>
**第三步，读出方向与大小**：对平面线圈，$\frac{1}{2}\int\mathbf{r}'\times\mathbf{J}\,\mathrm{d}V' = I\mathbf{S}$：大小 = 电流 × 面积，方向沿右手螺旋。这给出「电流环 = 磁偶极子」的经典结论，也解释了为什么条形磁铁的磁偶极矩可视为无数分子环流（安培分子环流假说）的叠加。

**辨析｜易错点：** 磁偶极矩是**轴矢量**（axial vector），由右手定则确定方向，与电偶极矩（极矢量，从负指向正）性质不同。在镜像反射下轴矢量反转、极矢量不反转——这导致「磁偶极子的镜像」问题（如镜像法在磁偶极矩上的应用）必须小心处理，不能照搬电荷镜像。

## 4 磁偶极子在磁场中的能量与受力

类比电偶极子，磁偶极子在外磁场中的行为为：

**能量** $W = -\mathbf{m}\cdot\mathbf{B}$——偶极子趋向与磁场同向（罗盘指针的原理）。
**力矩** $\mathbf{\tau} = \mathbf{m}\times\mathbf{B}$——同向时力矩为零（稳定平衡）。
**力**（非均匀场中）$F = \nabla(\mathbf{m}\cdot\mathbf{B})$——偶极子被推向场强增大处。<span class="marginnote">这解释了经典电磁学里最反直觉的现象之一：为什么磁铁会「吸」铁钉。不是铁钉本身带磁，而是外磁场把铁钉中的磁畴极化，感应出磁偶极矩，然后非均匀场把它们拉向磁铁。条形磁铁吸引铁屑、MRI 吸引铁磁性造影剂，都是这条力的体现。</span>

**原子与微观磁矩**：原子的磁矩来自电子轨道运动与电子自旋。轨道磁矩 $\mathbf{m} = -\dfrac{e}{2m_e}\mathbf{L}$（$\mathbf{L}$ 为轨道角动量，负号因电子带负电），比值 $e/2m_e$ 称为**旋磁比**。电子自旋磁矩与自旋角动量的旋磁比是轨道值的两倍（$g$ 因子 ≈ 2），这一「反常」数值后来成为量子电动力学最精确验证的实验基础之一。

## 5 磁标势的应用

磁标势在实际问题中有重要用途：

**永磁体外部场**：永磁体外部无电流，用磁标势 + 界面条件（$B_n$ 连续）求解，可算磁体对铁磁物体的作用力。
**磁路分析**：电机、变压器磁路中，把磁场类比电路（磁动势、磁阻、磁通），用磁标势建立磁路方程。
**与静电类比**：由于数学同构，静电学的一切解法（分离变量、镜像、格林函数）都能搬到无源磁场。例如「均匀外场中的超导球」用磁标势解出球外偶极场，结果与「均匀外场中介质球」完全同形。<span class="marginnote">注意「磁标势」的名称虽带「标势」，它的单位与物理意义和电标势完全不同——磁标势不是「磁场的势能」，而是「磁场强度 $\mathbf{H}$ 的势」。它不能直接给出磁能，磁能要靠 $\mathbf{B}$ 与 $\mathbf{H}$（见下一节）。名字像，物理完全不同，这是术语最容易骗人的地方。</span>

## 6 磁多极展开的完整例题与检验

多极展开的威力在「只取最低阶就能把握远场」。用一道题把展开到偶极阶的流程走通，并学会检验截断的合理性。

**例题：任意平面电流环的远场矢势。** 一个任意形状的平面闭合线圈载流 $I$，面积矢量 $\mathbf{S}$。磁偶极矩 $\mathbf{m} = I\mathbf{S}$。远场矢势取偶极项：

$$\mathbf{A}(\mathbf{r}) \approx \frac{\mu_0}{4\pi}\frac{\mathbf{m}\times\hat{\mathbf{r}}}{r^2}$$

**验证——圆环的特例**：半径为 $R$ 的圆电流，$m = I\pi R^2$。若直接对圆环用毕奥-萨伐尔积分求轴线上磁场，得 $B(z) = \dfrac{\mu_0 I R^2}{2(R^2+z^2)^{3/2}}$；用偶极近似（$z \gg R$）：$B(z) \approx \dfrac{\mu_0 m}{2\pi z^3} = \dfrac{\mu_0 I\pi R^2}{2\pi z^3} = \dfrac{\mu_0 I R^2}{2z^3}$。两者在 $z \gg R$ 时完全一致——**偶极近似在远场是精确的，这就是「多极展开」的检验方式。**

**截断的判据**：偶极近似在「观测距离 ≫ 电流分布线度」时成立。若电流环尺寸与观测距离同量级，必须保留更高阶项（电四极、磁四极等）。**判断「远场」的标准不是直觉，而是 $r \gg d$（$d$ 为源的线度）。**

**多极展开的物理层级**：电场的最低阶是多极子（电荷）；磁场因无磁单极子，最低阶是偶极子。这决定了「点电荷是静电的基本单元，电流环是静磁的基本单元」——所以永磁体、磁偶极子模型能用「小电流环」来等价，而带电体能用「点电荷」来等价。

**辨析｜易错点：** ① 偶极近似的矢势按 $1/r^2$ 衰减，磁场按 $1/r^3$ 衰减——比点电荷的静电场（$1/r^2$）快一个幂次，磁偶极子「更局域」。② $\mathbf{m} = I\mathbf{S}$ 中 $\mathbf{S}$ 的方向由右手定则，是轴矢量；两个同向电流环并排，偶极矩相加——**磁偶极矩是矢量，遵守矢量叠加**。③ 多极展开的每一项都满足 $\nabla\cdot\mathbf{B} = 0$ 与 $\nabla\times\mathbf{B} = \mu_0\mathbf{J}$，截断不会破坏方程——截断只损失精度，不破坏自洽。

**地球磁场的偶极模型**：地球磁场近似为一个磁偶极子，磁偶极矩约 $8\times10^{22}\ \mathrm{A\cdot m^2}$，轴线与地理轴约偏离 $11°$。地表磁场强度约 $25$–$65\ \mu\mathrm{T}$。**用偶极子模型能解释磁倾角、磁偏角、以及磁极迁移的大部分观测**——尽管地磁的真正来源是地核液体的发电机效应（不是一块永磁体），但远场近似下偶极模型足够好。这也是「磁多极展开」在天文中的直接应用。

**磁偶极子与电偶极子的对照表**：把本章的「电-磁平行」列成一张表，是复习的捷径。

| 量 | 电偶极子 | 磁偶极子 |
| --- | --- | --- |
| 源 | 正负电荷分离 | 电流环 / 自旋 |
| 偶极矩 | $\mathbf{p} = q\mathbf{d}$ | $\mathbf{m} = I\mathbf{S}$ |
| 远场势 | $\varphi \propto \dfrac{\mathbf{p}\cdot\hat{\mathbf{r}}}{r^2}$ | $\mathbf{A} \propto \dfrac{\mathbf{m}\times\hat{\mathbf{r}}}{r^2}$ |
| 远场场强 | $E \propto \dfrac{p}{r^3}$ | $B \propto \dfrac{m}{r^3}$ |
| 外场中的能量 | $W = -\mathbf{p}\cdot\mathbf{E}$ | $W = -\mathbf{m}\cdot\mathbf{B}$ |

**辨识关键**：电场有单极子（电荷），磁场无单极子——所以电偶极矩是「第二阶」展开，磁偶极矩是「第一阶」展开，两者的「地位」不对等，这是对照表里最容易被忽略的结构差异。

## 7 小结

- **磁标势**：$\mathbf{J} = 0$ 区域中 $\mathbf{H} = -\nabla\varphi_m$，满足拉普拉斯方程；**多值性**需用割面修补。
- 磁多极展开最低非零阶是**偶极子**（无单极子），磁偶极矩 $\mathbf{m} = \frac{1}{2}\int\mathbf{r}'\times\mathbf{J}\,\mathrm{d}V'$，线圈 $\mathbf{m} = I\mathbf{S}$。
- 偶极子远场 $\mathbf{B} = \dfrac{\mu_0}{4\pi r^3}[3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{m}]$。
- 偶极子在磁场中：能量 $W = -\mathbf{m}\cdot\mathbf{B}$、力矩 $\mathbf{\tau} = \mathbf{m}\times\mathbf{B}$。
- 磁标势让无源静磁问题复用整套静电解法。

在下一节，我们把磁场能量与物质的磁响应连起来：磁能如何储存、磁场如何极化物质——**磁场能量与磁化**。
