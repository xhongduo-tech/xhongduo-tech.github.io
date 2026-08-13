---
title: 多元线性回归模型与矩阵表示
date: 2026-08-07
---

# 多元线性回归模型与矩阵表示

<div class="epigraph">
<p>凡是线性之处，皆可用矩阵统一之；凡是有矩阵之处，皆可求逆而解之。</p>
<footer>—— 依线性代数精神改写（paraphrase of linear algebra lore）</footer>
</div>

<div class="article-byline">
<p>第二级 · 回归分析与线性模型 ｜ Montgomery《线性回归分析导论》第3章 ｜ 2026-08-07</p>
</div>

## 为什么从矩阵开始

现实问题很少只有一个自变量：房价不只由面积决定，还看地段、楼层、朝向。**多元线性回归（multiple linear regression）** 把 $p$ 个自变量一并纳入 $y = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p + \varepsilon$。$p$ 一旦超过 2，标量公式立刻臃肿到无法书写。幸运的是，线性代数给了一个完美的压缩：**把所有观测、所有变量塞进矩阵，一条公式统一处理**。本课建立矩阵表示与矩阵最小二乘解，这是你进入统计建模「高维世界」的通行证。

## 1 从标量到矩阵：三块积木

$n$ 组观测、每个含 $p$ 个自变量，模型 $y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \varepsilon_i$ 可以整体写成：

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

三块积木分别是：

- **$\mathbf{y}$**：$n \times 1$ 的观测向量，第 $i$ 个分量是 $y_i$；
- **$\mathbf{X}$**：$n \times (p+1)$ 的**设计矩阵（design matrix）**，第一列全 1（对应截距），其余第 $j$ 列是第 $j$ 个自变量的 $n$ 个取值；
- **$\boldsymbol{\beta}$**：$(p+1) \times 1$ 的系数向量 $(\beta_0, \beta_1, \ldots, \beta_p)'$；
- **$\boldsymbol{\varepsilon}$**：$n \times 1$ 的误差向量。

<span class="marginnote">把截距伪装成「第 0 个自变量的系数」，设计矩阵第一列全 1——这个小小的技巧让截距与斜率享受同等待遇，后面的公式无需任何特判。多项式、样条、指示变量最终都只是往 $\mathbf{X}$ 里加列。</span>

误差假设浓缩为向量形式：

$$
E(\boldsymbol{\varepsilon}) = \mathbf{0}, \qquad \mathrm{Var}(\boldsymbol{\varepsilon}) = \sigma^2 \mathbf{I}_n
$$

第二个式子表示误差**等方差**（对角元素全为 $\sigma^2$）且**不相关**（非对角为 0）——这正是标量假设的矩阵翻译。

## 2 矩阵最小二乘：求导的优雅解

最小二乘目标是最小化残差平方和：

$$
S(\boldsymbol{\beta}) = \sum_{i=1}^{n} e_i^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})'(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
$$

展开并配方：

$$
S(\boldsymbol{\beta}) = \mathbf{y}'\mathbf{y} - 2\boldsymbol{\beta}'\mathbf{X}'\mathbf{y} + \boldsymbol{\beta}'\mathbf{X}'\mathbf{X}\boldsymbol{\beta}
$$

对 $\boldsymbol{\beta}$ 求导并令其为零，得到矩阵形式的**正规方程**：

$$
\mathbf{X}'\mathbf{X}\boldsymbol{\beta} = \mathbf{X}'\mathbf{y}
$$

**公式解析：最小二乘解**——当 $\mathbf{X}'\mathbf{X}$ 可逆时：

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}
$$

- **$\mathbf{X}'\mathbf{X}$**：$p+1$ 阶对称矩阵，其元素是自变量间的交叉积；可逆要求设计矩阵列满秩（没有冗余变量、样本足够多）。
- **$(\mathbf{X}'\mathbf{X})^{-1}$**：把「信息量」矩阵求逆，体现「数据越分散，估计越精确」——与简单回归中 $\hat{\beta}_1 = S_{xy}/S_{xx}$ 的分母同构。
- **$\mathbf{X}'\mathbf{y}$**：自变量与因变量的交叉积向量。
- **几何解读**：$\hat{\boldsymbol{\beta}}$ 是把 $\mathbf{y}$ 正交投影到 $\mathbf{X}$ 的列空间 $C(\mathbf{X})$ 上的坐标；预测 $\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{H}\mathbf{y}$，其中 $\mathbf{H} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ 是**帽子矩阵（hat matrix）**，把 $y$ 变成帽子 $\hat{y}$。

## 3 帽子矩阵：投影的算术

**帽子矩阵（hat matrix）** $\mathbf{H} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ 是整个多元回归的暗线角色：

- $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$：预测值是观测值经 $\mathbf{H}$ 的线性变换；
- $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\mathbf{y}$：残差是另一个投影的结果；
- $\mathbf{H}$ 的**对角元素 $h_{ii}$** 称为第 $i$ 个观测的**杠杆值（leverage）**，度量该观测对自身预测值的影响力，取值在 $[1/n, 1]$，总和 $= p+1$。

<span class="marginnote">$h_{ii}$ 是把「第 $i$ 个观测对自身拟合的牵引力」量化：杠杆值高的点在 $\mathbf{X}$ 空间中离群体很远，即使 $y$ 值正常也可能主导回归。第 3 篇《杠杆点、异常值与影响诊断》专门剖析它。</span>

**辨析｜易错点：** 可逆条件 $\mathbf{X}'\mathbf{X}$ 满秩不是自动满足的。当两个自变量完全共线（一个恰好是另一个的倍数）时，$\mathbf{X}'\mathbf{X}$ 奇异，最小二乘解不存在——这是**多重共线性**的极端情形，第 3 篇会系统处理。

## 4 核心对比：简单回归与多元回归

| 维度 | 简单线性回归 | 多元线性回归 |
| --- | --- | --- |
| 自变量个数 | 1 | $p \ge 2$ |
| 系数 | $\beta_0, \beta_1$ | $\boldsymbol{\beta} = (\beta_0,\ldots,\beta_p)'$ |
| 估计公式 | 标量 $S_{xy}/S_{xx}$ | 矩阵 $(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}$ |
| 自由度 | $n-2$ | $n-p-1$ |
| 系数解读 | $x$ 增 1 单位 | **控制其他变量**下，$x_j$ 增 1 单位 |

多元回归最微妙的语义变化是系数解读：$\beta_j$ 不再是「$x_j$ 单独的影响」，而是**在其余 $p-1$ 个自变量保持不变的前提下**，$x_j$ 每增一个单位时 $y$ 期望的变化——这叫**偏效应（partial effect）**。<span class="marginnote">「控制其他变量」这一短语是多元回归的灵魂，也是它区别于简单回归的实质。正因如此，多元系数与简单回归的系数通常不相等——其余变量的解释力被剥离了。</span>

## 5 方差估计与拟合优度的多元版本

误差方差的无偏估计推广为：

$$
\hat{\sigma}^2 = \frac{\mathrm{SSE}}{n - p - 1}, \qquad \mathrm{SSE} = \mathbf{y}'\mathbf{y} - \hat{\boldsymbol{\beta}}'\mathbf{X}'\mathbf{y}
$$

自由度从 $n-2$ 变成 $n-p-1$：每多估一个系数就少一个自由度。

$R^2$ 的朴素定义 $\mathrm{SSR}/\mathrm{SST}$ 在多元下有个陷阱：**加任何变量（哪怕是纯噪声）$R^2$ 都只升不降**。于是引入**调整 $R^2$**：

$$
R_{\text{adj}}^2 = 1 - \frac{\mathrm{SSE}/(n-p-1)}{\mathrm{SST}/(n-1)}
$$

它用自由度惩罚变量个数，只有新变量「解释力超过其成本」时才上升。<span class="marginnote">$R^2_{\text{adj}}$ 是变量选择（第 3 篇）的第一个启蒙老师：它让你学会「解释力要扣除自由度的成本」。AIC、BIC 都是这个思想的不同版本。</span>

## 6 小结

- 多元回归 $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$，设计矩阵第一列全 1 收纳截距。
- 最小二乘解 $\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}$：投影到列空间，预测由帽子矩阵 $\mathbf{H}$ 给出。
- 系数解读为**偏效应**：「控制其他变量」下 $x_j$ 的单位变化效应。
- 方差估计 $\hat{\sigma}^2 = \mathrm{SSE}/(n-p-1)$；多元用**调整 $R^2$** 惩罚变量个数。
- $\mathbf{X}'\mathbf{X}$ 必须满秩；共线性会导致奇异或数值不稳。

在下一节，我们给 $\hat{\boldsymbol{\beta}}$