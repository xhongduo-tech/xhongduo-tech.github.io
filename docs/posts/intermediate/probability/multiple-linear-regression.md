---
title: 多元线性回归简介
date: 2026-08-07
---

# 多元线性回归简介

<div class="epigraph">
<p>一个自变量能解释一部分，多个自变量各解释一部分——多元回归，是「多因一果」的统计拼图。</p>
<footer>—— 罗纳德 · 费希尔（Ronald Fisher）</footer>
</div>

<div class="article-byline">
<p>第二级 · 概率论与数理统计 ｜ 盛骤《概率论与数理统计》§9.5 ｜ 2026-08-07</p>
</div>

## 为什么从多元线性回归开始

现实里 $y$ 几乎从不只受一个因素影响：房价受面积、地段、楼层共同决定；考试成绩受复习时间、基础、睡眠共同影响。**多元线性回归（multiple linear regression）**让 $p$ 个自变量同时进入模型——它是「一元回归」的自然推广，也是现代统计建模、机器学习线性模型（线性回归、岭回归、Lasso 的原型）的通用框架。

多元回归引入的新问题是一元没有的：多个自变量之间可能**相关**（多重共线性），各自的贡献需要「控制其他变量后」才能看清。它是第九章回归的收官，也是从「统计课本」通往「数据科学建模」的桥梁。<span class="marginnote">多元回归的矩阵形式是它优雅的关键：$y = X\beta + \varepsilon$ 一行公式装下 $p$ 个自变量。最小二乘解 $\hat\beta = (X^\top X)^{-1}X^\top y$ 是线性代数（矩阵求逆）与概率论（误差正态）的第一次大规模联姻——第二级《线性代数》里学的「四个基本子空间」在这里迎来实战。</span>

## 1 模型与矩阵形式

设 $y$ 与 $p$ 个自变量 $x_1, \ldots, x_p$ 的关系为

$$y = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p + \varepsilon, \qquad \varepsilon \sim N(0, \sigma^2)$$

矩阵形式：$\boldsymbol{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}$，其中 $X$ 是 $n \times (p+1)$ 设计矩阵（第一列全 1），$\boldsymbol{\beta} = (\beta_0, \beta_1, \ldots, \beta_p)^\top$。<span class="marginnote">「多元」与「一元」的关键差别：多元里每个回归系数 $\beta_j$ 的含义是「<strong>在其他自变量不变时</strong>，$x_j$ 每增加一单位 $y$ 平均变化多少」——这叫「偏回归系数」。因为其他变量被「控制」住了，$\beta_j$ 不再是「$x_j$ 单独与 $y$ 的关系」（那可能被混淆变量污染）。这个「控制其他」的视角是多元回归最深刻的价值。</span>

**例**：房价 $y$（万元）对面积 $x_1$（m²）与楼层 $x_2$ 回归：$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon$。$\beta_1$ 是「楼层相同时，面积每增 1m² 房价平均增加多少」。

## 2 最小二乘估计

多元最小二乘仍是「残差平方和最小」：

$$\min_{\boldsymbol\beta}\ \sum_{i=1}^n \left(y_i - \beta_0 - \beta_1 x_{i1} - \cdots - \beta_p x_{ip}\right)^2$$

求偏导令零，得正规方程 $X^\top X\boldsymbol\beta = X^\top \boldsymbol y$，解：

$$\hat{\boldsymbol\beta} = (X^\top X)^{-1} X^\top \boldsymbol y$$

**基本性质**（与一元同构）：$\hat{\boldsymbol\beta}$ 无偏（$E[\hat{\boldsymbol\beta}] = \boldsymbol\beta$）、正态，且

$$\mathrm{Cov}(\hat{\boldsymbol\beta}) = \sigma^2 (X^\top X)^{-1}$$

——系数向量的协方差矩阵由设计矩阵与 $\sigma^2$ 决定。<span class="marginnote">$(X^\top X)^{-1}$ 是多元回归的「信息量」：$X^\top X$ 越大（数据越丰富），逆矩阵越小，估计越准。若 $X$ 的列近似线性相关（多重共线性），$X^\top X$ 接近奇异，$(X^\top X)^{-1}$ 爆炸——系数估计方差暴涨、极不稳定。这就是「共线性危害估计」的数学本质：$X^\top X$ 的病态直接传导给估计精度。</span>

## 3 检验与选择

**整体显著性（F 检验）**：$H_0: \beta_1 = \cdots = \beta_p = 0$（所有自变量都无预测力），统计量

$$F = \frac{S_R/p}{S_E/(n-p-1)} \sim F(p,\ n-p-1)$$

**单个系数显著性（t 检验）**：$H_0: \beta_j = 0$（控制其他后 $x_j$ 无贡献），

$$T_j = \frac{\hat\beta_j}{S_{\hat\beta_j}} \sim t(n-p-1)$$

**决定系数（$R^2$）**：$R^2 = \frac{S_R}{S_{yy}} = 1 - \frac{S_E}{S_{yy}}$——$x$ 解释掉的 $y$ 波动比例，$0 \le R^2 \le 1$。**调整 $R^2$** 对自变量个数做惩罚：$\bar R^2 = 1 - \frac{S_E/(n-p-1)}{S_{yy}/(n-1)}$。<span class="marginnote">「F 检验整体、t 检验单个」的分工在多元里彻底清晰：F 回答「这组变量有没有用」，t 回答「单个变量在控制其他后还有没有用」。自由度 $n-p-1$ 是「估计 $p+1$ 个参数吃掉 $p+1$ 个自由度」——一元时 $p=1$ 正是 $n-2$。调整 $R^2$ 的存在是因为原始 $R^2$ 随自变量增多只升不降（噪声也能被拟合），调整版惩罚「多余变量」。</span>

**易错点｜辨析：** ① 多元里 t 与 F 不再等价（一元才 $t^2 = F$）；② $R^2$ 大 ≠ 模型好——噪声也能推高 $R^2$，且 $R^2$ 随变量增多单调上升，要用调整 $R^2$ 或信息准则；③ 共线性会掩盖单个系数的显著性——两个高度相关的自变量可能各自都不显著，但一起放进模型整体显著。

## 4 公式解析：多元最小二乘的几何

多元最小二乘解 $\hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top \boldsymbol y$ 有一个优美的几何含义，拆开：

$$

\hat{\boldsymbol\beta} = (X^\top X)^{-1} X^\top \boldsymbol y, \qquad \hat{\boldsymbol y} = X\hat{\boldsymbol\beta} = P\boldsymbol y

$$

- **第一步，投影矩阵**：$P = X(X^\top X)^{-1}X^\top$——$\hat{\boldsymbol y} = P\boldsymbol y$ 是 $\boldsymbol y$ 到「$X$ 的列空间」上的**正交投影**。
- **第二步，最小化的几何**：残差 $\boldsymbol y - \hat{\boldsymbol y}$ 正交于 $X$ 的每一列（$X^\top(\boldsymbol y - \hat{\boldsymbol y}) = 0$，正是正规方程）——「残差平方和最小」⇔「残差向量与列空间正交」。
- **第三步，$p+1$ 维视角**：$X$ 的 $p+1$ 列张成 $\mathbb{R}^n$ 里的一个 $p+1$ 维子空间，$\hat{\boldsymbol y}$ 是 $\boldsymbol y$ 在这个子空间上的投影，残差是投影后的「垂直距离」。

「最小二乘 = 正交投影」是多元回归最深刻的几何图像：拟合就是投影，残差就是垂直偏差。这个视角也解释了为什么共线性危险——列空间「退化」时投影变得不稳定。线性代数的四个基本子空间在这里完整登场。

## 5 多元回归的深入应用与实例

多元回归从一元到多变量，引入了共线性、模型选择等新问题，值得完整展开。

### 例：多元与一元的对比

| | 一元 | 多元 |
| --- | --- | --- |
| 模型 | $y = a + bx$ | $y = \beta_0 + \sum\beta_j x_j$ |
| 系数含义 | 斜率 | 偏回归（控制其他） |
| 检验 | t = F | t 单系数、F 整体 |
| 自由度 | $n-2$ | $n-p-1$ |

**「多元的关键是『控制其他变量』的偏回归视角」**——系数含义从「单独关联」变为「净关联」。

### 例：多重共线性的后果

**例**：$x_1$、$x_2$ 高度相关（如身高与体重）。两者各自入模型的 t 检验可能都不显著（信息重叠），但一起放 F 检验整体显著——**「共线性让单个系数不显著、整体显著」**是多元回归的经典困惑。

### 模型选择：调整 R² 与信息准则

| 准则 | 公式 | 惩罚 |
| --- | --- | --- |
| $R^2$ | $S_R/S_{yy}$ | 无（随变量增） |
| 调整 $\bar R^2$ | 修正自由度 | 有 |
| AIC | $-2\ln L + 2p$ | 有 |
| BIC | $-2\ln L + p\ln n$ | 更严 |

**「选模型看调整 R² 或信息准则，别只看 $R^2$」**——变量越多 $R^2$ 越高，但噪声也被拟合。

### 例：虚拟变量

分类变量（性别、地区）用 0/1 虚拟变量入模型——**「多元回归能处理分类预测变量」**是它的重要扩展（与方差分析的联系）。

### 例：正规方程与计算

$\hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top\boldsymbol y$——**「矩阵求逆解最小二乘」**是多元回归的计算核心，软件（lm、regression）都走这条路。

**易错点｜辨析：** ① 多元里 t 与 F 不再等价——t 管单个、F 管整体；② 共线性让 $(X^\top X)^{-1}$ 病态、估计不稳——用 VIF 或岭回归缓解；③ 「$R^2$ 高 ≠ 模型好」——可能过拟合或共线性虚高。

## 6 小结

- **多元线性回归** $y = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p + \varepsilon$，矩阵形式 $\boldsymbol y = X\boldsymbol\beta + \boldsymbol\varepsilon$。
- **最小二乘解** $\hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top\boldsymbol y$，无偏、正态，协方差 $\sigma^2(X^\top X)^{-1}$。
- **偏回归系数**：控制其他变量后 $x_j$ 每增一单位的平均效应。
- **F 检验整体、t 检验单个**；$R^2$ 度量解释比例，调整 $R^2$ 惩罚多余变量。
- **共线性**让 $X^\top X$ 病态、估计不稳；最小二乘 = 正交投影是它的几何本质。

在下一节，我们开启第十章——**非参数 bootstrap：模拟与再抽样**。
