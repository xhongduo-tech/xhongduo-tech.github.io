---
title: 非线性回归简介
date: 2026-08-07
---

# 非线性回归简介

<div class="epigraph">
<p>当模型对数据不再线性，就得让算法在曲面上摸索着下山。</p>
<footer>—— 依非线性优化精神改写（paraphrase of nonlinear optimization）</footer>
</div>

<div class="article-byline">
<p>第二级 · 回归分析与线性模型 ｜ Montgomery《线性回归分析导论》第12章 ｜ 2026-08-07</p>
</div>

## 为什么从非线性回归开始

本专题至今的「线性」一直指**参数线性**——即使样条、指示变量也是参数的线性组合。但当模型本身对参数非线性（如指数衰减、Michaelis–Menten 饱和曲线、Logistic 生长曲线）时，最小二乘没有闭式解，OLS 公式彻底失效。**非线性回归（nonlinear regression）** 用**迭代优化算法**在参数空间中搜索最小值。本课讲清非线性模型与线性模型的分界、高斯–牛顿与 Levenberg–Marquardt 算法、以及初始值这个「看不见的陷阱」。

## 1 什么算非线性模型

模型 $y = f(\mathbf{x}, \boldsymbol{\theta}) + \varepsilon$ 对参数 $\boldsymbol{\theta}$ 非线性，当且仅当 $f$ 对 $\boldsymbol{\theta}$ 的**导数仍含 $\boldsymbol{\theta}$**。经典例子：

- **指数衰减**：$y = \theta_1 e^{-\theta_2 x}$——对 $\theta_2$ 非线性；
- **Michaelis–Menten**：$y = \dfrac{\theta_1 x}{\theta_2 + x}$——酶促动力学饱和曲线；
- **Logistic 生长**：$y = \dfrac{\theta_1}{1 + e^{\theta_2 + \theta_3 x}}$——S 形生长。

**辨析｜易错点：** 区分「对变量非线性」与「对参数非线性」。$y = \beta_0 + \beta_1 e^x$ 对变量 $x$ 非线性、但对参数 $\beta$ 线性——仍是线性模型，OLS 可解；$y = \beta_0 e^{\beta_1 x}$ 对参数 $\beta_1$ 非线性，才是真正的非线性模型。<span class="marginnote">判断口诀：看「参数是否出现在非线性函数里（如指数、分母、嵌套）」。参数只在线性组合里出现 → 线性模型；参数钻进指数/分式/嵌套 → 非线性模型。前者 OLS 闭式解，后者迭代求解。</span>

## 2 非线性最小二乘：目标与梯度

非线性回归的估计仍是最小化残差平方和：

$$
S(\boldsymbol{\theta}) = \sum_{i=1}^{n}\big(y_i - f(\mathbf{x}_i, \boldsymbol{\theta})\big)^2 \longrightarrow \min_{\boldsymbol{\theta}}
$$

但 $S(\boldsymbol{\theta})$ 对 $\boldsymbol{\theta}$ 不再是二次函数，无法一步求导解出。需要**迭代**：从初始值 $\boldsymbol{\theta}^{(0)}$ 出发，反复更新参数使 $S$ 下降。

**核心思想：局部线性化**。在当前参数点 $\boldsymbol{\theta}^{(0)}$ 附近，把 $f$ 对 $\boldsymbol{\theta}$ 做一阶泰勒展开：

$$
f(\mathbf{x}, \boldsymbol{\theta}) \approx f(\mathbf{x}, \boldsymbol{\theta}^{(0)}) + \sum_{j}\frac{\partial f}{\partial \theta_j}\Big|_{\boldsymbol{\theta}^{(0)}}(\theta_j - \theta_j^{(0)})
$$

展开后，问题在局部变成「以导数矩阵为设计矩阵的线性回归」——每步解一个加权线性最小二乘，迭代推进。<span class="marginnote">「非线性问题的每一步都线性化」是数值方法的通用策略：局部看是线性的，走一步、再局部、再走一步。这也解释了为什么非线性回归的推断工具（近似标准误）来自线性化的设计矩阵。</span>

## 3 公式解析：高斯–牛顿迭代

**高斯–牛顿法（Gauss–Newton）** 把每一步写成线性最小二乘。记 $\mathbf{f}$ 为预测向量，$\mathbf{J}$ 为雅可比矩阵（元素 $J_{ij} = \partial f_i/\partial\theta_j$），更新步为：

$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} + (\mathbf{J}'\mathbf{J})^{-1}\mathbf{J}'\,\big(\mathbf{y} - \mathbf{f}(\boldsymbol{\theta}^{(t)})\big)
$$

逐步拆解：

- **$\mathbf{J}$**：当前参数下的导数矩阵，扮演线性回归里 $\mathbf{X}$ 的角色；
- **$\mathbf{y} - \mathbf{f}$**：当前残差向量，扮演「观测」；
- **$(\mathbf{J}'\mathbf{J})^{-1}\mathbf{J}'(\cdots)$**：形式上与 OLS 解 $\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}$ 完全同构——每一步都在解一个「局部线性化」的回归；
- **迭代**：重复至收敛（参数变化足够小或残差平方和不再下降）。

**缺陷**：$\mathbf{J}'\mathbf{J}$ 病态时步长失控。**Levenberg–Marquardt 算法**在 $\mathbf{J}'\mathbf{J}$ 上加阻尼项 $\lambda\mathbf{I}$（与岭回归同构），在「高斯–牛顿」（快）与「梯度下降」（稳）之间自适应——这是非线性最小二乘的工业标准。

## 4 初始值：看不见的胜负手

非线性最小二乘是局部优化——**不同初始值可能收敛到不同（局部）最优**。初始值选不好，算法可能发散或停在不合理的结果。

**选初始值的实用方法**：

1. **物理/领域含义**：参数有实际含义（最大生长率、半饱和常数），按其合理量级设初值；
2. **变换后线性回归**：对可变换的模型（如 $\ln y$ 线性化），先解线性回归得到初值；
3. **网格搜索**：粗网格上扫初始值，取使 $S$ 最小的作为起点；
4. **多起点**：从多个初始值跑，比较收敛结果，确认全局最小值。

<span class="marginnote">「多起点」是应对局部最优的标准保险：跑 10–20 个随机/网格起点，若都收敛到同一参数，可信度高；若不同，说明目标面崎岖，需要更小心。这也是现代优化（如随机初始化 + 多次重跑）的祖先。</span>

**辨析｜易错点：** 非线性回归的**标准误与 $p$ 值**来自局部线性化近似，样本小时可能严重不准。小样本 + 强非线性时，bootstrap 是更可靠的推断方式。

## 5 核心对比：线性模型 vs 非线性模型

| 维度 | 线性模型 | 非线性模型 |
| --- | --- | --- |
| 参数线性 | 是 | 否 |
| 解 | 闭式 $(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}$ | 迭代（Gauss–Newton 等） |
| 全局最优 | 唯一（二次目标） | 可能多个局部最优 |
| 初始值 | 不需要 | 至关重要 |
| 推断 | 精确（正态误差下） | 近似（线性化） |
| 典型模型 | 多元回归、样条 | 指数衰减、生长曲线 |

<span class="marginnote">用一句话记住分界线：<strong>线性模型有闭式解、结果唯一、推断精确；非线性模型要迭代、结果依赖初始值、推断靠近似</strong>。能用线性模型解决的问题，不必冒非线性模型的险。</span>

## 6 一个非线性拟合的完整示例

用 Michaelis–Menten 酶动力学模型走一遍非线性回归的完整流程。模型为 $y = \dfrac{\theta_1 x}{\theta_2 + x}$，其中 $y$ 是反应速率，$x$ 是底物浓度，$\theta_1$ 是最大速率 $V_{\max}$，$\theta_2$ 是半饱和常数 $K_m$。

**第 1 步，初值**：从散点图读出——$y$ 随 $x$ 趋于平缓的平台约 10，故 $\theta_1^{(0)} = 10$；半最大速率对应的 $x$ 约 3，故 $\theta_2^{(0)} = 3$。

**第 2 步，迭代**：用 Levenberg–Marquardt 从 $(\theta_1,\theta_2) = (10, 3)$ 出发迭代，约 5 轮后收敛到 $\hat{\theta}_1 = 9.8$、$\hat{\theta}_2 = 2.7$。

**第 3 步，诊断**：画残差对拟合值，确认无明显模式；计算近似标准误（来自线性化的雅可比）。

**第 4 步，多起点验证**：再从 $(8, 2)$、$(12, 5)$ 等初值重跑，均收敛到同一参数——确认是全局最优而非局部陷阱。

**第 5 步，解释**：$\hat{V}_{\max} = 9.8$ 表示底物无穷时反应速率趋于 9.8；$\hat{K}_m = 2.7$ 表示速率达到最大一半所需的底物浓度。

<span class="marginnote">注意第 4 步的价值：非线性优化的结果依赖初值，多起点收敛到同一解是「可信」的重要证据。若不同初值给出不同解，说明目标函数崎岖、需要更稳健的策略。</span>

**能否先变换再线性拟合？** 对某些「可线性化」的非线性模型（如 $y = \theta_1 e^{\theta_2 x}$ 取 $\log$ 后变线性），直接变换拟合简单，但要注意：变换同时改变了误差结构（$\log$ 后误差变为乘性），两个途径给出的参数估计与推断语义不同。以建模目的为准——描述性分析可用变换，严格的统计推断更推荐直接做非线性最小二乘。

## 7 小结

- 非线性模型：参数出现在非线性函数中，OLS 闭式解失效。
- 非线性最小二乘迭代最小化 $S(\boldsymbol{\theta})$，每步**局部线性化**。
- **高斯–牛顿**：$(\mathbf{J}'\mathbf{J})^{-1}\mathbf{J}'$