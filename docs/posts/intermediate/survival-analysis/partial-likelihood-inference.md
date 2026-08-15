---
title: 部分似然推断
date: 2026-08-07
---

# 部分似然推断

<div class="epigraph">
<p>真正的发现之旅不在于寻找新的风景，而在于拥有新的眼睛。</p>
<footer>—— 马塞尔 · 普鲁斯特（Marcel Proust）</footer>
</div>

<div class="article-byline">
<p>第二级 · 生存分析 ｜ Kalbfleisch & Prentice, The Statistical Analysis of Failure Time Data, Ch.4 ｜ 2026-08-07</p>
</div>

## 为什么从部分似然开始

上一课的 Cox 模型给出 $h(t \mid \mathbf{X}) = h_0(t) e^{\boldsymbol{\beta}'\mathbf{X}}$，其中 $h_0(t)$ 完全未知。这就带来一个尖锐的问题：**常规的似然函数里每一项都含 $h_0(t)$，都不知道基线，怎么估计 $\boldsymbol{\beta}$？**<span class="marginnote">完整似然需要同时估计无穷维的 $h_0(t)$ 与有限维的 $\boldsymbol{\beta}$，数学上极其笨重。Cox 1972 年的论文用一句「我想把 $h_0$ 约掉」开出了解药，1975 年他在后续论文中把这种构造正式命名为<strong>部分似然（partial likelihood）</strong>。</span>

Cox 的天才在于换了一个视角：**不要用「每个个体在多长时间后出事」来构造似然，而要用「在每个事件时刻，出事的是谁」来构造似然**。后者只涉及协变量之间的相对大小，$h_0(t)$ 在比值中自行消去。一个「谁先出事」的顺序，居然承载了 $\boldsymbol{\beta}$ 的全部信息——这就是部分似然。

## 1 完整似然的困境：基线危险挡在路中央

先看清完整似然长什么样。沿用删失数据的似然结构 $L = \prod h(t_i)^{\delta_i} S(t_i)$，代入 Cox 模型：

$$L(\boldsymbol{\beta}, H_0) = \prod_{i} \left[ h_0(t_i) e^{\boldsymbol{\beta}'\mathbf{X}_i} \right]^{\delta_i} \exp\!\left\{ -e^{\boldsymbol{\beta}'\mathbf{X}_i} H_0(t_i) \right\}$$

这里 $H_0(t) = \int_0^t h_0(u)\,du$ 是基线累积危险。要让 $L$ 关于 $\boldsymbol{\beta}$ 最大化，就必须同时处理 $h_0$ 或 $H_0$——它是一个**未知函数**，不是几个参数。<span class="marginnote">若强行给 $h_0(t)$ 套一个形状（指数、Weibull），就回到了参数模型；但那样就把「分布假设」这个包袱背了回来。部分似然要的是：既不背参数形状，又能对 $\boldsymbol{\beta}$ 做推断。</span>

**关键观察**：$L$ 可以分解为「关于 $h_0$ 的部分」与「只关于 $\boldsymbol{\beta}$ 的部分」的乘积，而后一部分与 $h_0$ 完全无关。把后者单独拎出来，就是部分似然。

## 2 条件化：每个事件时刻都是一次「谁出事」的条件概率

设 $t_{(1)} \lt  t_{(2)} \lt  \cdots \lt  t_{(m)}$ 为所有**事件时刻**（删失时刻不在此列），$R(t_j)$ 为时刻 $t_j$ 的风险集，$X_{(j)}$ 为在 $t_j$ 发生事件的个体的协变量。

在「$t_j$ 时刻恰有一人出事，且此人来自 $R(t_j)$」的条件下，**这个人恰好是个体 $i$** 的条件概率为：

$$\frac{h_0(t_j) e^{\boldsymbol{\beta}'\mathbf{X}_i}}{\sum_{\ell \in R(t_j)} h_0(t_j) e^{\boldsymbol{\beta}'\mathbf{X}_\ell}} = \frac{e^{\boldsymbol{\beta}'\mathbf{X}_i}}{\sum_{\ell \in R(t_j)} e^{\boldsymbol{\beta}'\mathbf{X}_\ell}}$$

$h_0(t_j)$ 在分子分母中同时出现、**直接约掉**。这正是 Cox 洞察的核心：**基线危险对「谁在哪个时刻出事」的排序不产生任何影响，因为它对所有在场者施加同一个倍数**。<span class="marginnote">对比 log-rank 检验：那里我们用超几何分布描述「事件在两组间的分配」，这里用「指数权重按比例分配」描述「事件在众多协变量个体间的分配」——同一个「观测减期望」的思想，从两组推广到了任意协变量。</span>

## 3 部分似然的显式形式

把所有事件时刻的条件概率连乘，得到**部分似然函数（partial likelihood）**：

$$\mathrm{PL}(\boldsymbol{\beta}) = \prod_{j=1}^{m} \frac{e^{\boldsymbol{\beta}'\mathbf{X}_{(j)}}}{\sum_{\ell \in R(t_j)} e^{\boldsymbol{\beta}'\mathbf{X}_\ell}}$$

对 $\mathrm{PL}$ 取对数，得部分对数似然：

$$\ell(\boldsymbol{\beta}) = \sum_{j=1}^{m} \left[ \boldsymbol{\beta}'\mathbf{X}_{(j)} - \ln\!\Big( \sum_{\ell \in R(t_j)} e^{\boldsymbol{\beta}'\mathbf{X}_\ell} \Big) \right]$$

**估计量** $\hat{\boldsymbol{\beta}}$ 是使 $\ell(\boldsymbol{\beta})$ 最大的点，通常用 Newton–Raphson 迭代求解。<span class="marginnote">严格地说，$\mathrm{PL}$ 不是一个真正的似然函数——它略去了关于 $h_0$ 与事件时刻本身的信息。但在计数过程与鞅的理论框架下（本专题最后一课），可以证明 $\hat{\boldsymbol{\beta}}$ 拥有与 MLE 相同的渐近性质：一致性、渐近正态、渐近有效性。这就是「部分似然可以当似然用」的数学根基。</span>

## 4 公式解析：PL 的每一项在做什么

把部分似然拆开，理解每一块的意义。

**第一步，看分子 $e^{\boldsymbol{\beta}'\mathbf{X}_{(j)}}$**：它是「出事者的风险权重」。协变量越大、风险越高，其指数权重就越大，分子越大。

**第二步，看分母 $\sum_{\ell \in R(t_j)} e^{\boldsymbol{\beta}'\mathbf{X}_\ell}$**：它是「当时所有在场者的风险权重之和」，即**风险集内的总权重**。分母越大（在场者多、且普遍风险高），某个特定个体出事的条件概率就越小。

**第三步，读整体比值**：$e^{\boldsymbol{\beta}'\mathbf{X}_{(j)}} / \sum_{\ell} e^{\boldsymbol{\beta}'\mathbf{X}_\ell}$ 是「出事者权重占总权重之比」——它是 $[0,1]$ 之间的一个「概率碎片」，描述「在那么多高风险的候选者里，为什么偏偏是这位出事」。若 $\boldsymbol{\beta}$ 很大、出事者又是风险最高的，这个比值接近 1，模型很「满意」；若 $\boldsymbol{\beta}$ 完全背离（把高风险者估计为低风险），比值很小，模型很「不满意」。

**第四步，连乘所有事件时刻**：把每个时刻的「概率碎片」乘起来。优化 $\boldsymbol{\beta}$ 就是让「每个时刻出事者恰是风险最高者」这件事尽可能可信——**部分似然最大化 = 让事件顺序在模型下尽量「不意外」**。

## 5 估计与推断：从得分函数到渐近正态

对 $\ell(\boldsymbol{\beta})$ 求导得**得分函数** $U(\boldsymbol{\beta})$，令其为零即得分方程。$U$ 的每一项是「观测协变量 − 期望协变量」的形式：

$$U(\boldsymbol{\beta}) = \sum_{j=1}^{m} \left[ \mathbf{X}_{(j)} - \frac{\sum_{\ell \in R(t_j)} \mathbf{X}_\ell e^{\boldsymbol{\beta}'\mathbf{X}_\ell}}{\sum_{\ell \in R(t_j)} e^{\boldsymbol{\beta}'\mathbf{X}_\ell}} \right]$$

方括号第二项是「以指数权重加权的风险集内平均协变量」，即在 $H_0$ 下「出事者协变量的期望」。**得分函数 = 观测值减期望值之和**——与 log-rank 检验的 $O - E$ 结构完全同构。<span class="marginnote">把得分函数在 $\boldsymbol{\beta} = \mathbf{0}$ 处取值，并对「单变量分组」情形展开，恰好回到 log-rank 检验——这就在数学上印证了上一课「log-rank 是 Cox 的 score 检验」的说法。</span>

渐近性质由二阶导（观测信息矩阵 $I(\boldsymbol{\beta})$）支撑：在大样本下 $\hat{\boldsymbol{\beta}}$ 近似服从 $N(\boldsymbol{\beta}, I(\hat{\boldsymbol{\beta}})^{-1})$，由此得到每个系数的标准误、置信区间与 Wald 检验。

## 6 平局处理：Breslow 与 Efron 近似

现实数据常出现同一时刻多人事件（如以「天」为单位记录），此时「恰有一人出事」的条件概率失效。两种常用近似：

**Breslow 近似**：把 $d_j$ 个平局事件当作 $d_j$ 个「几乎同时」的事件，分母用原始风险集重复计算，实现简单、软件默认（R 的 `coxph` 即 Breslow）；
**Efron 近似**：分母按平局内事件的先后依次剔除已「出事者」的权重，更精确，计算略贵。<span class="marginnote">平局比例很大时（如所有生存时间都是整数天），两种近似的系数差异仍很小，但标准误可能被略微低估；若平局极端密集，更严格的做法是「精确部分似然」或改用离散时间模型（如互补 log-log）。</span>

**重点：平局处理只改变分母的构造，不改变部分似然「比值」的本质**。理解 Breslow 与 Efron 的差别，比背诵公式更重要——它们的差异在极端平局下才值得担心。

## 7 一个数值例子：两个事件时刻的部分似然

用极简数据感受部分似然如何「工作」。设 4 个个体，事件时间与协变量如下：

| 个体 | 时间 $t$ | 事件 $\delta$ | 协变量 $X$ |
| --- | --- | --- | --- |
| A | 3 | 1 | 1 |
| B | 5 | 1 | 0 |
| C | 8 | 0（删失） | 1 |
| D | 12 | 0（删失） | 0 |

事件时刻有两个：$t = 3$（A 出事）与 $t = 5$（B 出事）。计算部分对数似然 $\ell(\beta)$。

**时刻 3**：风险集 $R(3) = \{A, B, C, D\}$，出事者是 A，贡献为 $\ln\big(e^{\beta} / (e^{\beta} + 1 + e^{\beta} + 1)\big) = \ln\big(e^{\beta} / (2e^{\beta} + 2)\big)$。

**时刻 5**：风险集 $R(5) = \{B, C, D\}$（A 已出事离场），出事者是 B，贡献为 $\ln\big(1 / (1 + e^{\beta} + 1)\big) = \ln\big(1 / (e^{\beta} + 2)\big)$。

**总目标**：$\ell(\beta) = \beta - \ln(2e^{\beta} + 2) - \ln(e^{\beta} + 2)$。求导并令其为零：

$$\ell'(\beta) = 1 - \frac{2e^{\beta}}{2e^{\beta} + 2} - \frac{e^{\beta}}{e^{\beta} + 2} = 0$$

解得 $\beta = 0$。**直觉验证**：A 先出事（$X=1$），B 后出事（$X=0$），顺序与「$X$ 大者风险高」一致，但 C、D 两个删失者中 C 的 $X=1$ 却活得更久——证据相互抵消，故估计为无效应。若把 C 也改成在 $t=6$ 出事，则「高 $X$ 者几乎都早出事」，$\hat{\beta}$ 会明显为正。

**这个例子说明**：部分似然的每一个数都来自「风险集的比例」，删失者 C、D 只以分母身份参与，与 KM 估计「删失只进分母」完全一致。<span class="marginnote">这里的解析解恰好等于 0 是演示数据精心安排的巧合；真实数据下 $\ell(\beta)$ 通常没有闭式解，Newton–Raphson 迭代几步即可收敛——软件替你做了这些。</span>

## 8 小结

- **完整似然**含未知基线 $H_0$，难以直接优化；部分似然通过「每个事件时刻条件化」把 $h_0$ 约掉。
- **部分似然** $\mathrm{PL} = \prod_j e^{\boldsymbol{\beta}'\mathbf{X}_{(j)}} / \sum_{\ell \in R(t_j)} e^{\boldsymbol{\beta}'\mathbf{X}_\ell}$：只依赖事件顺序与风险集。
- 最大化部分似然 = 让「每个时刻出事者恰是风险最高者」尽量可信；得分函数 = 观测减期望。
- 在鞅理论下，$\hat{\boldsymbol{\beta}}$ 拥有 MLE 的全部渐近性质——这是「部分似然当似然用」的合法性来源。
- 平局用 **Breslow / Efron** 近似处理；单变量分组时部分似然退化为 log-rank 检验。
- 部分似然把「谁先出事」的顺序视为全部信息——删失只进分母，事件顺序承载 $\boldsymbol{\beta}$