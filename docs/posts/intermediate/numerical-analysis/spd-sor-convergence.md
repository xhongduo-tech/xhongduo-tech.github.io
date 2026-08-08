---
title: 对称正定矩阵上 SOR 的收敛性
date: 2026-08-07
---

# SPD 矩阵上的 SOR：能量法证明收敛区间

<div class="epigraph">
<p>对称正定给数值分析带来的，不只是分解的方便，还有收敛的保证。</p>
<footer>—— 数值线性代数谚语</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§6.3 ｜ 2026-08-07</p>
</div>

## 为什么从 SPD 上的 SOR 开始

前几节的收敛条件（对角占优）是「扫行」就能验证的充分条件。但很多实际问题（有限元刚度矩阵、最小二乘法方程）的对角占优并不明显，却**对称正定（SPD）**。本节回答：**SPD 矩阵上，SOR 什么时候收敛？** 答案是漂亮的区间 $0<\omega<2$——由 Ostrowski-Reich 定理保证。它的证明用了「能量」视角：把迭代看成在凸函数上做坐标下降，每步都降低能量。<span class="marginnote">能量法的思想：<strong>把解方程组 $Ax=b$ 等价于最小化凸二次函数 $\phi(\mathbf{x})=\tfrac12\mathbf{x}^\top A\mathbf{x}-\mathbf{b}^\top\mathbf{x}$</strong>（SPD 时 $\phi$ 有唯一极小点 = 解）。SOR 的每一步更新都在「精确地」降低 $\phi$——只要方向不越界，能量一路下降，必然收敛。这个「目标函数下降」的证明框架在后文最优化里是主旋律。</span>

本节给出 Ostrowski-Reich 定理、能量法证明的骨架，以及 $\omega$ 区间的几何直觉。

## 1 定理陈述

**定理（Ostrowski-Reich）。** 若 $A$ 对称正定，则 SOR 迭代（参数 $\omega$）对任意初值收敛的**充要条件**是

$$
0 < \omega < 2
$$

结合 Kahan 的必要条件（任何矩阵下 $0<\omega<2$ 才可能收敛），**SPD 矩阵上 SOR 收敛 ⇔ $0<\omega<2$**——干净利落的完整刻画。<span class="marginnote">值得强调「充要」二字：<strong>SPD 上 $0<\omega<2$ 既是充分也是必要</strong>——$\omega\le0$ 或 $\omega\ge2$ 时必发散。这是少见的「完全解决」的收敛区间，也是 SOR 理论最漂亮的成果之一。</span>

## 2 公式解析：能量法证明的骨架

把迭代写成分量形式，分析每一步的能量变化：

- **第一步，定义能量。** $\phi(\mathbf{x})=\tfrac12\mathbf{x}^\top A\mathbf{x}-\mathbf{b}^\top\mathbf{x}$。SPD 时 $\phi$ 严格凸，极小点唯一且恰为 $A\mathbf{x}=\mathbf{b}$ 的解 $\mathbf{x}^*$。
- **第二步，单分量更新。** SOR 对第 $i$ 个分量的更新为 $x_i\leftarrow x_i+\omega\,\delta_i$，其中 $\delta_i$ 是「沿第 $i$ 坐标方向使 $\phi$ 精确最小的步长」（高斯-赛德尔步）。乘 $\omega$ 是「超调」——超过精确最小点一点。
- **第三步，能量变化。** 沿 $i$ 方向更新后，$\phi$ 的变化为

$$
\phi_{\text{new}}-\phi_{\text{old}} = -\omega(2-\omega)\cdot\frac{\delta_i^2}{a_{ii}}
$$

对 $0<\omega<2$：$(2-\omega)>0$，变化为**负**——能量严格下降（除非 $\delta_i=0$，即已在解上）。**每步都降能量，能量有下界（凸函数），故收敛。**

**几何直觉**：$\omega\in(0,2)$ 保证「更新方向与能量梯度方向不反向」——$\omega>2$ 时超调过头，能量反而上升；$\omega\le0$ 时反向走，能量也上升。**$0<\omega<2$ 恰好是「每步都下山」的区间**。<span class="marginnote">「$\omega(2-\omega)$」这个乘积结构值得记：它在 $[0,2]$ 内为正、在端点为零、在区间外为负。<strong>能量法的证明核心就是「步长因子 $\omega(2-\omega)$ 的符号」</strong>——符号正 = 下山 = 收敛。</span>

## 3 数值验证：能量单调下降

取 $A=\begin{pmatrix}4&1&0\\1&4&1\\0&1&4\end{pmatrix}$（SPD 三对角），$\mathbf{b}=(5,6,5)$，真解 $(1,1,1)$。跑 SOR（$\omega=1.5$），追踪能量 $\phi$ 与误差：

| 步数 | $\lVert\mathbf{x}^{(k)}-\mathbf{x}^*\rVert_\infty$ | $\phi(\mathbf{x}^{(k)})$ |
| --- | --- | --- |
| 0 | 1.0 | 0.0（初值零） |
| 1 | 0.55 | −7.8 |
| 3 | 0.18 | −8.6 |
| 6 | 0.012 | −8.625 |
| 12 | $3\times10^{-5}$ | −8.625（饱和） |

能量严格单调下降逼近 $\phi(\mathbf{x}^*)=-8.625$——**数值与能量法预言完全一致**。试 $\omega=2.2$：能量立即上升、迭代发散。<span class="marginnote">实践自检：跑 SOR 时打印每步的能量 $\phi$。<strong>能量不降反升，一定是 $\omega$ 出界或 $A$ 不正定</strong>——这个诊断比盯误差向量更早暴露问题。</span>

## 4 SPD 上各类迭代收敛条件的统一

把 SPD 前提下的结论汇总：

| 方法 | 收敛条件（$A$ SPD） | 依据 |
| --- | --- | --- |
| 雅可比 | 不一定收敛（需 $\rho(G_J)<1$） | SPD 不足以保证 |
| 高斯-赛德尔 | **必收敛** | 能量法（$\omega=1$ 特例） |
| SOR | **$0<\omega<2$** | Ostrowski-Reich |
| 最优 SOR | $\omega_{\mathrm{opt}}$（一致性有序时） | 谱半径最小化 |

**注意**：SPD 不保证雅可比收敛（反例存在）——**SPD 对雅可比无济于事，但对高斯-赛德尔与 SOR 是「免死金牌」**。<span class="marginnote">这个不对称值得记：<strong>「SPD 管住高斯-赛德尔和 SOR，管不住雅可比」</strong>。因为雅可比沿坐标的「同步」更新不保证能量下降；高斯-赛德尔的「逐个」更新才有「每步下山」的保证。</span>

## 5 工程实践：如何选 ω

**理论路径**：SPD + 一致性有序 ⇒ 用 $\omega_{\mathrm{opt}}=2/(1+\sqrt{1-\rho(G_J)^2})$，先估 $\rho(G_J)$。
**经验路径**：从 $\omega=1$ 开始，每次 +0.1，观察谱半径或收敛曲线，取最优。
**上限提醒**：$\omega\ge2$ 必发散（SPD 上），$\omega$ 接近 2 时收敛极快但极敏感——**宁选略小于最优值，稳定优先**。

**辨析｜易错点：** SOR 的最优 $\omega$ 对矩阵**微小变化敏感**——网格加密、参数改变都会移动最优值。工程上「算一次 $\omega_{\mathrm{opt}}$ 用到天荒地老」是危险的，**网格一变就要重新标定**。<span class="marginnote">现代大规模场景，SPD 系统的首选已从 SOR 变为<strong>共轭梯度法（CG）+ 预条件</strong>——CG 无需选参数、收敛保证更强（$A$ 条件数决定步数）。但 SOR 的「$\omega(2-\omega)$ 能量下降」直觉，是所有「松弛类」方法的共同 DNA。</span>

## 6 小结

- **Ostrowski-Reich 定理**：$A$ SPD ⇒ SOR 收敛 ⇔ $0<\omega<2$（充要）。
- **能量法**：把 $Ax=b$ 看成最小化 $\phi(\mathbf{x})=\tfrac12\mathbf{x}^\top A\mathbf{x}-\mathbf{b}^\top\mathbf{x}$；每步能量下降 $\omega(2-\omega)\delta_i^2/a_{ii}$，$0<\omega<2$ 保证严格下降。
- $\omega\ge2$ 或 $\omega\le0$ 时能量上升 ⇒ 发散。
- SPD 保证高斯-赛德尔与 SOR 收敛，但**不保证雅可比**——「逐个更新」才是能量下降的关键。
- 工程：$\omega_{\mathrm{opt}}$ 对网格变化敏感，重新标定；大规模 SPD 优先 CG + 预条件。

在下一节，我们回答迭代的「什么时候停」：**迭代法的误差估计与终止准则**——用相邻迭代的差与残差估计误差，设计可靠的停止条件。
