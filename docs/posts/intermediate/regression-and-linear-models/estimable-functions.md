---
title: 可估函数与线性假设检验
date: 2026-08-07
---

# 可估函数与线性假设检验

<div class="epigraph">
<p>并非每一个关于参数的命题都有资格被数据回答——可估性先于可检验性。</p>
<footer>—— 依线性模型理论精神改写（paraphrase of linear-model theory）</footer>
</div>

<div class="article-byline">
<p>第二级 · 回归分析与线性模型 ｜ 王松桂、杨虎《线性模型引论》第3章 ｜ 2026-08-07</p>
</div>

## 为什么从可估函数开始

多元回归的检验看似完备：整体 $F$、单个 $t$、子集 $F$。但这一切都建立在一个隐含前提上——**你想检验的东西确实能从数据中识别出来**。当设计矩阵列不满秩时（如编码化方案导致冗余、或数据缺失产生结构），某些参数组合根本无法被无偏估计。这就是**可估函数（estimable function）** 理论登场的时刻。它回答一个先于一切检验的问题：**哪些参数的线性组合有资格被估计与检验？** 这是线性模型理论中最精细、也最常被教材带过的部分。

## 1 什么是可估函数

设 $\mathbf{c}$ 是一个常向量，$\mathbf{c}'\boldsymbol{\beta}$ 是参数的线性组合（如 $\beta_1 - \beta_2$、$2\beta_1 + \beta_3$）。我们说 $\mathbf{c}'\boldsymbol{\beta}$ 是**可估的（estimable）**，如果存在 $\mathbf{y}$ 的线性函数可以无偏估计它：

$$
\exists\ \mathbf{a},\ \text{s.t.}\ E(\mathbf{a}'\mathbf{y}) = \mathbf{c}'\boldsymbol{\beta},\ \forall\ \boldsymbol{\beta}
$$

把 $E(\mathbf{a}'\mathbf{y}) = \mathbf{a}'\mathbf{X}\boldsymbol{\beta}$ 代入，可估性条件简化为：

$$
\mathbf{c}' \in \text{行空间}\ C(\mathbf{X}') \qquad \Longleftrightarrow \qquad \mathbf{c}' = \mathbf{t}'\mathbf{X}'\ \text{对某个}\ \mathbf{t}
$$

<span class="marginnote">核心判据：$\mathbf{c}'\boldsymbol{\beta}$ 可估，当且仅当 $\mathbf{c}$ 落在设计矩阵的行空间里（等价地，$\mathbf{c}$ 正交于 $\mathbf{X}$ 的零空间）。当 $\mathbf{X}$ 满秩时<strong>一切</strong>线性组合都可估；不满秩时才出现「不可估」的悬念。</span>

**直觉**：可估函数是「数据里确实带着信息的参数组合」。$\mathbf{c}$ 若与零空间不正交，意味着它的一部分「落在数据观测不到的方向上」——那种组合只能瞎猜，谈不上检验。

## 2 为什么会有不可估的情形

不可估性几乎总是源于**设计矩阵列不满秩**。三种典型来源：

1. **过参数化编码**：方差分析里把 $p$ 个水平编码成 $p$ 个 0/1 指示变量而不去截距，导致列线性相关；
2. **冗余变量**：某变量恰好是另几个变量的线性组合；
3. **数据缺失**：某些变量组合从未同时出现，导致对应列无法辨识。

<span class="marginnote">在 ANOVA 语境里，「处理效应之和为零」的约束（sum-to-zero 编码）就是为保证可估性而加的可识别条件；不同编码方案可估的线性组合不同，但可估函数本身不随编码改变——这就是「可估函数是模型的客观性质」的含义。</span>

## 3 公式解析：可估性的等价判定

设 $E(\boldsymbol{\varepsilon})=\mathbf{0}$、$\mathrm{Var}(\boldsymbol{\varepsilon})=\sigma^2\mathbf{I}$。以下三条等价：

$$
\text{(1)}\ \mathbf{c}' = \mathbf{t}'\mathbf{X}' \ \text{有解}; \qquad
\text{(2)}\ \mathbf{c}' \perp \text{零空间}\ \mathcal{N}(\mathbf{X}); \qquad
\text{(3)}\ \mathbf{c}'(\mathbf{I} - \mathbf{P}_{\mathbf{X}'})\mathbf{c} = 0
$$

拆解：

- **（1）可表示性**：$\mathbf{c}'$ 能写成 $\mathbf{X}'$ 各行的线性组合——这是定义的直接翻译；
- **（2）正交性**：$\mathbf{c}'$ 与 $\mathbf{X}$ 的零空间正交。零空间里的方向是「数据完全测不到」的方向，可估组合必须避开它们；
- **（3）投影判据**：$\mathbf{P}_{\mathbf{X}'}$ 是到行空间的投影矩阵，$\mathbf{c}$ 到行空间的投影残差长度为 0 ⇔ $\mathbf{c}$ 已在行空间内。这是数值上最方便的实现。

**重点结论**：当 $\mathbf{X}$ 列满秩时，$\mathcal{N}(\mathbf{X}) = \{\mathbf{0}\}$，零空间里没有非零向量，**一切 $\mathbf{c}'\boldsymbol{\beta}$ 都可估**。可估性理论因此主要是不满秩情形的「安全网」。

## 4 可估函数的检验：一般的线性假设

一旦确认 $\mathbf{c}'\boldsymbol{\beta}$ 可估，就能检验线性假设 $H_0: \mathbf{c}'\boldsymbol{\beta} = m$。其 $t$ 检验为：

$$
t_0 = \frac{\mathbf{c}'\hat{\boldsymbol{\beta}} - m}{\sqrt{\hat{\sigma}^2\, \mathbf{c}'(\mathbf{X}'\mathbf{X})^{-}\mathbf{c}}} \sim t_{n-r}
$$

其中 $(\mathbf{X}'\mathbf{X})^{-}$ 是广义逆，$r$ 是 $\mathbf{X}$ 的秩（不满秩时自由度为 $n-r$ 而非 $n-p-1$）。<span class="marginnote">广义逆 $(\mathbf{X}'\mathbf{X})^{-}$ 的出现是「不满秩也能做推断」的关键技术：虽然 $\mathbf{X}'\mathbf{X}$ 不可逆，但对可估组合 $\mathbf{c}'$，量 $\mathbf{c}'(\mathbf{X}'\mathbf{X})^{-}\mathbf{c}$ 与广义逆的具体选择无关——可估性保证了推断结果的唯一性。</span>

**公式解析的要点**：分子是「估计值减假设值」，分母是估计量的标准误；关键在于**分母里的广义逆**在可估组合下给出唯一、正确的标准误。这保证了即使设计不满秩，只要检验的是可估函数，$t$ 统计量依然精确。

## 5 一般线性假设的 $F$ 检验

把多个约束放在一起（如同时检验 $\beta_1=\beta_2$ 且 $\beta_3=0$），得到一般线性假设：

$$
H_0: \mathbf{C}\boldsymbol{\beta} = \mathbf{m}
$$

其中 $\mathbf{C}$ 是 $q \times (p+1)$ 的行可估矩阵。检验统计量推广为：

$$
F_0 = \frac{(\mathbf{C}\hat{\boldsymbol{\beta}} - \mathbf{m})'[\mathbf{C}(\mathbf{X}'\mathbf{X})^{-}\mathbf{C}']^{-1}(\mathbf{C}\hat{\boldsymbol{\beta}} - \mathbf{m})/q}{\mathrm{SSE}/(n-r)} \sim F_{q,\, n-r}
$$

- 分子是「假设离数据多远」的二次型（马氏距离）；
- 分母是误差均方；
- 自由度 $q$ 是约束个数，$n-r$ 是残差自由度。

<span class="marginnote">这条公式是线性模型检验的「总纲」：整体 $F$（令 $\mathbf{C}$ 为斜率选择矩阵）、子集检验、均值比较、可估函数 $t$ 检验，全是它的特例。学会它，等于掌握了整个线性假设检验家族。</span>

## 6 核心对比：满秩与不满秩的推断

| 维度 | 满秩设计（$r = p+1$） | 不满秩设计（$r \lt  p+1$） |
| --- | --- | --- |
| $\mathbf{X}'\mathbf{X}$ | 可逆 | 奇异，用广义逆 |
| 可估性 | 一切 $\mathbf{c}'\boldsymbol{\beta}$ | 仅行空间内的组合 |
| 参数可识别 | 每个 $\beta_j$ 唯一 | $\beta_j$ 不唯一，但可估函数唯一 |
| 残差自由度 | $n-p-1$ | $n-r$ |
| 典型场合 | 多元回归、正交设计 | 过参数化 ANOVA、缺失设计 |

**辨析｜易错点：** 不满秩时，**个别 $\beta_j$ 可能不可估**，软件却仍会打印「估计值」——那些值依赖特殊的广义逆选择，没有独立含义。唯一可靠的解读对象是**可估的线性组合**。解释 ANOVA 系数前，先确认编码方案与可估函数。

**一点历史注记**：可估函数理论在 20 世纪中叶随方差分析与实验设计的系统化而成熟，其核心工具（广义逆、行空间、零空间）后来成为一般线性模型（含混合模型、缺失数据）推断的公共语言。今天软件里「可估性检查」多由底层自动完成，但理解它，你才能在「模型打印了奇怪系数」时不慌。

## 7 小结

- **可估函数**：$\mathbf{c}'\boldsymbol{\beta}$ 可估 ⇔ $\mathbf{c}'$ 在行空间 ⇔ $\mathbf{c}' \perp \mathcal{N}(\mathbf{X})$。
- 满秩设计下一切线性组合可估；不可估性源于列不满秩（过参数化、冗余、缺失）。
- 可估函数检验用广义逆 $(\mathbf{X}'\mathbf{X})^{-}$，可估性保证结果与广义逆选择无关。
- 一般线性假设 $H_0:\mathbf{C}\boldsymbol{\beta}=\mathbf{m}$ 用 $F_0 \sim F_{q,n-r}$ 检验，是全部线性检验的总纲。
- 不满秩时个别参数不可解释，只信可估组合。

在下一节，我们松开「等方差」这只隐形的手——**广义最小二乘与加权最小二乘**，当误差不再是 $\sigma^2\mathbf{I}$