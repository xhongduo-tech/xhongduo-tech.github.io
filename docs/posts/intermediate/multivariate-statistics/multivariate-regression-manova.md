---
title: 多元线性回归与多元方差分析（MANOVA）
date: 2026-08-07
---

# 多元线性回归与多元方差分析（MANOVA）

<div class="epigraph">
<p>与其把每个响应变量单独建模、事后拼凑，不如承认它们共享同一批预测变量、同一批分组效应，把整张响应矩阵一次拟合完。</p>
<footer>—— 特德·安德森（Theodore W. Anderson）</footer>
</div>

<div class="article-byline">
<p>第二级 · 多元统计分析 ｜ Anderson《An Introduction to Multivariate Statistical Analysis》Ch.8–9 · Johnson & Wichern Ch.6–7 ｜ 2026-08-07</p>
</div>

## 为什么把响应变量也向量化

一元回归研究「一个预测变量解释一个响应」。但现实中一次实验往往同时记录好几个响应：施肥试验同时测株高、穗重、叶绿素；临床试验同时测血压、心率、炎症指标。把几个响应**分开建模再各说各话**会犯两个错：一是忽略响应之间的相关结构，二是把多个检验叠在一起时第一类错误膨胀。**多元回归与 MANOVA 正是「让整张响应矩阵一起建模」的框架**——上一节的两样本 $T^2$ 检验在此推广为「任意多个响应 + 任意多个组」的统一语言。<span class="marginnote">多元统计的一条方法论主线在这里立住：<strong>凡是标量能做的事，多元版本都要求「一次做完、整体控制错误率」</strong>。一元 ANOVA 拆开做 $m$ 次，与一次 MANOVA 整体做，结果并不等价。</span>

## 1 多元线性回归模型

设每个个体有 $q$ 个响应与 $p$ 个预测变量。对第 $i$ 个观测，模型为

$$
\mathbf{y}_i = \mathbf{B}'\mathbf{x}_i + \boldsymbol{\varepsilon}_i, \qquad \boldsymbol{\varepsilon}_i \sim \mathcal{N}_q(\mathbf{0}, \boldsymbol{\Sigma})
$$

其中 $\mathbf{y}_i$ 是 $q \times 1$ 响应向量，$\mathbf{x}_i$ 是 $p \times 1$（含截距项），$\mathbf{B}$ 是 $p \times q$ 系数矩阵，误差项彼此独立。把 $n$ 个观测堆叠，得到

$$
\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{E}
$$

$\mathbf{Y}$ 是 $n \times q$ 响应矩阵，$\mathbf{X}$ 是 $n \times p$ 设计矩阵，$\mathbf{E}$ 是 $n \times q$ 误差矩阵。**与一元的关键差异：$\mathbf{B}$ 有 $p \times q$ 个参数，而误差协方差 $\boldsymbol{\Sigma}$ 又是 $q\times q$ 矩阵**——参数一下子多了几个数量级，也因此必须珍惜样本。

最小二乘解与一元形式完全平行：$\hat{\mathbf{B}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$。把它逐列拆开看，第 $j$ 列 $\hat{\mathbf{b}}_j = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}_j$ 恰好是「第 $j$ 个响应单独做一元回归」的系数——**最小二乘估计每个响应各自独立，但误差结构（进而检验与推断）共享同一个 $\boldsymbol{\Sigma}$**。<span class="marginnote">这解释了为什么多元回归不改变点估计、却改变推断：$\hat{\mathbf{B}}$ 只依赖 $\mathbf{Y}$ 的列，$\boldsymbol{\Sigma}$ 却把所有响应的不确定性拧在一起。预测区间、假设检验都必须考虑响应间的协方差。</span>

## 2 回归系数的检验：从一元 F 到 Wilks Λ

有了 $\hat{\mathbf{B}}$ 与残差矩阵 $\mathbf{E} = \mathbf{Y} - \mathbf{X}\hat{\mathbf{B}}$，可以检验「预测变量是否真的有效」。一元回归里检验整体显著性用 $F = \frac{SS_R/p}{SS_E/(n-p-1)}$；多元版本把 $SS_R$、$SS_E$ 换成矩阵：

$$
\mathbf{H} = \hat{\mathbf{B}}'\mathbf{X}'\mathbf{X}\hat{\mathbf{B}} \ (\text{回归平方和矩阵}), \qquad \mathbf{E} = \mathbf{E}'\mathbf{E} \ (\text{残差平方和矩阵})
$$

检验 $H_0: \mathbf{B} = \mathbf{0}$ 的 Wilks Λ 是

$$
\Lambda = \frac{|\mathbf{E}|}{|\mathbf{E} + \mathbf{H}|}
$$

结构上与 MANOVA 的 $\Lambda = |\mathbf{W}|/|\mathbf{T}|$ 同构——**回归与方差分析在多元框架里是同一台机器**：回归的「残差」就是方差分析的「组内」，回归的「模型」就是方差分析的「组间」。这正是线性模型统一理论的雏形。<span class="marginnote">更一般的<strong>线性假设</strong> $\mathbf{C}\mathbf{B}\mathbf{D} = \mathbf{0}$（$\mathbf{C}$ 挑预测变量、$\mathbf{D}$ 挑响应）可以覆盖几乎所有实际问题：检验某几个预测变量是否有用、某几个响应是否可合并。Wilks Λ 只是这种一般框架的特例。</span>

检验「某个预测变量子集是否多余」时，比较带子集与不带子集的残差矩阵：$\Lambda = |\mathbf{E}_{\text{full}}|/|\mathbf{E}_{\text{reduced}}|$，再查近似 $F$ 分布——这与一元回归里的「偏 $F$ 检验」一一对应。**口诀：一元里比两个 $SS$，多元里比两个行列式。**

## 3 MANOVA：一元方差分析的多元推广

MANOVA（多元方差分析）回答：**分组变量是否同时影响 $q$ 个响应？** 设 $k$ 个组，第 $i$ 组第 $j$ 个观测 $\mathbf{y}_{ij}$，模型

$$
\mathbf{y}_{ij} = \boldsymbol{\mu} + \boldsymbol{\tau}_i + \boldsymbol{\varepsilon}_{ij}, \qquad i=1,\ldots,k,\ j=1,\ldots,n_i
$$

$\boldsymbol{\tau}_i$ 是 $q$ 维组效应向量。检验 $H_0: \boldsymbol{\tau}_1 = \cdots = \boldsymbol{\tau}_k = \mathbf{0}$，需要把方差分解推广成**矩阵分解**：

$$
\underbrace{\mathbf{T}}_{\text{总平方和}} = \underbrace{\mathbf{B}}_{\text{组间}} + \underbrace{\mathbf{W}}_{\text{组内}}
$$

$$
\mathbf{T} = \sum_{ij}(\mathbf{y}_{ij}-\bar{\mathbf{y}})(\mathbf{y}_{ij}-\bar{\mathbf{y}})', \quad
\mathbf{W} = \sum_{ij}(\mathbf{y}_{ij}-\bar{\mathbf{y}}_{i})(\mathbf{y}_{ij}-\bar{\mathbf{y}}_{i})', \quad
\mathbf{B} = \sum_i n_i(\bar{\mathbf{y}}_{i}-\bar{\mathbf{y}})(\bar{\mathbf{y}}_{i}-\bar{\mathbf{y}})'
$$

一元 ANOVA 比较的是标量 $SS_B$ 与 $SS_W$ 之比；多元里 $\mathbf{B}$ 与 $\mathbf{W}$ 都是矩阵，「比值」不再唯一，于是派生出一族检验统计量。**矩阵的比不能直接相除，是 MANOVA 与一元 ANOVA 最本质的分水岭**。

## 4 检验统计量：Wilks Λ 与三大近似

最经典的统计量是 **Wilks 的 Λ（lambda）**：

$$
\Lambda = \frac{|\mathbf{W}|}{|\mathbf{T}|} = \frac{|\mathbf{W}|}{|\mathbf{B}+\mathbf{W}|}
$$

$\Lambda$ 衡量「组内散布占总散布的份额」：取值越小，说明组间差异占比越大，越支持拒绝 $H_0$。注意 $\Lambda$ 与一元 $F$ 的相似性：一元时 $\Lambda = \frac{SS_W}{SS_T} = \frac{1}{1+F \cdot df_B/df_W}$——**Λ 是「不拒绝方向」的度量**，因此拒绝域是 $\Lambda$ 小于某临界值。

$\Lambda$ 的精确分布只有少数简单情形（如 $q=1$ 或 $k=2$）是已知的，一般情形用三条渐近近似：

- **Bartlett 的 $\chi^2$ 近似**：$-\left[n - \frac{q-k}{2} - 1\right]\ln\Lambda \sim \chi^2_{q(k-1)}$（大样本）。
- **Rao 的 $F$ 近似**：把 $\Lambda$ 变换成 $F$ 分布，多数软件默认输出它。
- **Pillai 迹** $V = \operatorname{tr}\!\big(\mathbf{B}(\mathbf{B}+\mathbf{W})^{-1}\big)$：对协方差不等与样本量失衡最稳健，常作为交叉验证的备选。<span class="marginnote">四大统计量（Wilks Λ、Pillai、Lawley–Hotelling、Roy's 最大根）在总体满足假定时结论通常一致；分歧明显时，<strong>Pillai 迹最稳、Roy's 最大根最灵敏</strong>——这是实战选统计量的经验口诀。</span>

### 一元 ANOVA 与 MANOVA 对照

| 概念 | 一元 ANOVA | MANOVA |
| --- | --- | --- |
| 平方和 | $SS_T, SS_B, SS_W$ 标量 | $\mathbf{T},\mathbf{B},\mathbf{W}$ 矩阵 |
| 统计量 | $F = \frac{SS_B/(k-1)}{SS_W/(n-k)}$ | $\Lambda = \|\mathbf{W}\|/\|\mathbf{T}\|$ 等 |
| 零分布 | $F_{k-1, n-k}$ | $\Lambda$ 近似 $F$ 或 $\chi^2$ |
| 错误率控制 | 单次比较 | 整体同时控制 $q$ 个响应 |

## 5 公式解析：Wilks Λ 的直觉与自由度

**为什么用行列式之比？** 这要回到第一节的几何直觉：$|\mathbf{W}|$ 是组内数据云的「体积」，$|\mathbf{T}|$ 是全部数据的「体积」。

- **第一步，行列式即体积**：$p=2$ 时 $|\mathbf{W}|$ 正比于组内椭圆面积。体积越小，说明各组数据围绕各自中心挤得越紧。
- **第二步，Λ 是「未解释份额」**：$\Lambda = |\mathbf{W}|/|\mathbf{T}|$。若分组完全无效，$\mathbf{B} \approx \mathbf{0}$，$\mathbf{T} \approx \mathbf{W}$，Λ 接近 1；分组越有效，$\mathbf{W}$ 相对越小，Λ 趋于 0。
- **第三步，看自由度结构**：Bartlett 近似里 $n - \frac{q-k}{2} - 1$ 是样本量扣除参数后的「有效样本」，$\chi^2_{q(k-1)}$ 的自由度 $q(k-1)$ 是「$q$ 个响应 × $k-1$ 个独立组差」——每个响应贡献 $k-1$ 个自由度。
- **第四步，联系两样本 $T^2$**：当 $k=2$ 时，$q=1$ 退化为一元 $F$，$q>1$ 退化为两样本 Hotelling $T^2$。Λ 是一个把 $T^2$ 与 $F$ 都包含在内的统一框架。

**核心结论：MANOVA 把「组内对组间的体积比」整体压缩成一个数**，用矩阵行列式完成「矩阵除法」，再用近似分布把 Λ 拉回可查表的 $F$ 或 $\chi^2$。

## 6 小结

- **多元回归** $Y = XB + E$ 让 $q$ 个响应共享设计矩阵与误差协方差；最小二乘解 $\hat{\mathbf{B}} = (X'X)^{-1}X'Y$ 逐列独立，但推断共享 $\boldsymbol{\Sigma}$。
- **MANOVA** 把一元方差分解升级为矩阵分解 $\mathbf{T} = \mathbf{B} + \mathbf{W}$；「矩阵之比」不唯一，派生出一族统计量。
- **Wilks Λ** $=|\mathbf{W}|/|\mathbf{T}|$ 度量组内体积占比，越小越拒绝 $H_0$；精确分布少见，用 Bartlett $\chi^2$ 或 Rao $F$ 近似。
- **Pillai 迹**对协方差不等最稳健，Roy's 最大根最灵敏——实战按此取舍。
- 两样本 $T^2$ 是 $k=2$ 的 MANOVA，MANOVA 是一元 ANOVA 的自然升维。

在下一节，我们进入多元统计最常用的降维工具——**主成分分析（PCA）**：不关心「分组」而关心「如何用最少的新轴保留最多的散布」，把 $\mathbf{S}$