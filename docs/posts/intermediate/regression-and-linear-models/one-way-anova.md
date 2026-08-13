---
title: 单因素方差分析模型
date: 2026-08-07
---

# 单因素方差分析模型

<div class="epigraph">
<p>比较多个均值，不是把两两差异放大，而是看组间是否盖过了组内。</p>
<footer>—— 依费希尔方差分析思想改写（paraphrase of Fisher's ANOVA）</footer>
</div>

<div class="article-byline">
<p>第二级 · 回归分析与线性模型 ｜ 王松桂、杨虎《线性模型引论》第4章 ｜ 2026-08-07</p>
</div>

## 为什么从方差分析开始

回归比较「连续变量对 $y$ 的影响」，方差分析（ANOVA）则比较「**类别变量的各水平是否让 $y$ 的均值不同**」——比如四种肥料是否产量不同、三种工艺是否良率不同。第 3 篇《指示变量》已经暗示：类别变量用 0/1 编码后，方差分析不过是回归的特例。本课建立单因素方差分析的完整框架：效应模型、平方和分解、$F$ 检验与多重比较，并始终用「回归视角」贯穿——你会发现它与第 2 篇的 $F$ 检验是同一条血脉。

## 1 模型设定：从 $t$ 检验到方差分析

两组均值比较用独立样本 $t$ 检验；但三组以上两两 $t$ 检验会累积错误率（$k$ 组要比较 $k(k-1)/2$ 对，错误率暴涨）。**单因素方差分析（one-way ANOVA）** 一次检验所有组均值是否相等。

模型（效应模型）：

$$
y_{ij} = \mu + \tau_i + \varepsilon_{ij}, \qquad i = 1,\ldots,a;\ j = 1,\ldots,n_i
$$

其中 $y_{ij}$ 是第 $i$ 组的第 $j$ 个观测，$\mu$ 是总均值，$\tau_i$ 是第 $i$ 组的**处理效应（treatment effect）**，$\varepsilon_{ij}$ 是独立同分布的 $N(0,\sigma^2)$ 误差。检验：

$$
H_0: \tau_1 = \tau_2 = \cdots = \tau_a = 0 \qquad \text{vs} \qquad H_1: \text{至少一个}\ \tau_i \neq 0
$$

<span class="marginnote">模型的自由度说明：$\mu$ 与 $a$ 个 $\tau_i$ 共 $a+1$ 个参数，但只有 $a$ 个组均值——需要一条可识别约束，通常取 $\sum_i n_i\tau_i = 0$ 或 $\sum_i\tau_i=0$。这正是第 2 篇《可估函数》里过参数化问题的重现。</span>

## 2 平方和分解：组间与组内

把总变动拆成「组间」与「组内」两部分：

$$
\underbrace{\sum_{i=1}^{a}\sum_{j=1}^{n_i}(y_{ij} - \bar{y}_{\cdot\cdot})^2}_{\mathrm{SS}_T}
= \underbrace{\sum_{i=1}^{a}n_i(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2}_{\mathrm{SS}_{\text{Treatments}}}
+ \underbrace{\sum_{i=1}^{a}\sum_{j=1}^{n_i}(y_{ij} - \bar{y}_{i\cdot})^2}_{\mathrm{SS}_E}
$$

- **总平方和 $\mathrm{SS}_T$**：所有观测相对总均值的离差，自由度 $N-1$（$N = \sum n_i$）；
- **处理平方和 $\mathrm{SS}_{\text{Treatments}}$**：各组均值相对总均值的离差，衡量「组间差异」，自由度 $a-1$；
- **误差平方和 $\mathrm{SS}_E$**：组内观测相对组内均值的离差，衡量「组内噪声」，自由度 $N-a$。

<span class="marginnote">平方和分解是 ANOVA 的心脏：组间大而组内小 → 组别真的影响 $y$；组间小 → 差异只是噪声。它正是回归里 $\mathrm{SST} = \mathrm{SSR} + \mathrm{SSE}$ 的孪生结构，只是把「被解释」换成了「被组别解释」。</span>

## 3 公式解析：$F$ 检验统计量

ANOVA 的检验统计量是两个「均方」之比：

$$
F_0 = \frac{\mathrm{SS}_{\text{Treatments}}/(a-1)}{\mathrm{SS}_E/(N-a)} = \frac{\mathrm{MS}_{\text{Treatments}}}{\mathrm{MS}_E}
$$

逐项拆解：

- **$\mathrm{MS}_{\text{Treatments}} = \mathrm{SS}_{\text{Treatments}}/(a-1)$**：处理均方，组间差异「每自由度」的平均；
- **$\mathrm{MS}_E = \mathrm{SS}_E/(N-a)$**：误差均方，是 $\sigma^2$ 的无偏估计；
- **$H_0$ 下**：$F_0 \sim F_{a-1,\,N-a}$；$F_0$ 大说明组间远大于组内；
- **直觉**：$F_0$ 是「组别信号 / 噪声」之比，比值大于临界值即拒绝 $H_0$。

**重点结论**：$F$ 检验与回归的整体 $F$ 检验同构——事实上把 ANOVA 写成指示变量回归后，两处 $F_0$ 是**同一个数**。这就是「ANOVA 是回归特例」的数学证据。

## 4 方差分析表：标准汇报格式

ANOVA 的结果总是汇总成一张标准表：

| 来源 | 平方和 | 自由度 | 均方 | $F_0$ |
| --- | --- | --- | --- | --- |
| 处理 | $\mathrm{SS}_{\text{Treatments}}$ | $a-1$ | $\mathrm{MS}_{\text{Treatments}}$ | $\frac{\mathrm{MS}_{\text{Treatments}}}{\mathrm{MS}_E}$ |
| 误差 | $\mathrm{SS}_E$ | $N-a$ | $\mathrm{MS}_E$ | — |
| 总计 | $\mathrm{SS}_T$ | $N-1$ | — | — |

<span class="marginnote">这张表的信息浓缩度极高：平方和、自由度、均方、$F$ 值一次到位。学会读它，就掌握了所有实验设计与回归软件输出的通用语言。</span>

**辨析｜易错点：** ANOVA 的假设必须检查：各组方差**齐性**（最大组方差与最小组方差比 $\lt  3$ 左右）、误差近似正态、观测独立。方差不齐时 $F$ 检验失真，可用 Welch 校正或数据变换。

## 5 多重比较：$F$ 显著之后怎么办

$F$ 检验只告诉你「至少一组不同」，不告诉你「哪几组不同」。后续的**多重比较（multiple comparisons）** 才回答细节。常用方法：

**Fisher LSD**：两两 $t$ 检验（不校正，仅 $F$ 显著后使用）；
**Tukey HSD**：校正所有两两比较的错误率，最常用；
- **Bonferroni 校正**：把显著性水平除以比较次数，保守；
- **Dunnett**：专门比较「各处理组 vs 对照组」。

<span class="marginnote">多重比较的本质是「控制总错误率」：比较 $m$ 次时，每次用 $\alpha$ 会让整体错误率膨胀到 $1-(1-\alpha)^m$。Tukey 等方法把整体错误率压回 $\alpha$——代价是每次比较更保守、功效更低。</span>

**辨析｜易错点：** 未经校正的两两比较（尤其是「看哪个显著就报告哪个」）是科研中的重灾区。规则：$F$ 显著后再做多重比较，且必须报告用了什么校正方法。

## 6 回归视角：ANOVA 与指示变量

把 ANOVA 写成回归：对 $a$ 个组造 $a-1$ 个指示变量 $D_2,\ldots,D_a$（基准为第 1 组）：

$$
y = \beta_0 + \beta_2 D_2 + \cdots + \beta_a D_a + \varepsilon
$$

- $\beta_0$ 是基准组均值，$\beta_i$ 是第 $i$ 组与基准组的均值差；
- 检验「所有 $\beta_i = 0$」等价于 ANOVA 的 $F$ 检验——**同一次检验，两种表述**。

<span class="marginnote">这个等价是理解整部线性模型的钥匙：回归、ANOVA、ANCOVA（下一课）都是「线性模型 = 设计矩阵 + 误差」这一母模型的特例。变量类型（连续/类别）只决定设计矩阵长什么样，推断引擎完全相同。</span>

## 7 小结

- 单因素 ANOVA 检验 $H_0: \tau_1 = \cdots = \tau_a = 0$，模型 $y_{ij} = \mu + \tau_i + \varepsilon_{ij}$。
- 平方和分解 $\mathrm{SS}_T = \mathrm{SS}_{\text{Treatments}} + \mathrm{SS}_E$，自由度分别为 $N-1, a-1, N-a$。
- $F_0 = \mathrm{MS}_{\text{Treatments}}/\mathrm{MS}_E \sim F_{a-1,N-a}$：组间信号与组内噪声之比。
- $F$