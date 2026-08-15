---
title: 多项式混沌展开（PCE）
date: 2026-08-07
---

# 多项式混沌展开（PCE）

<div class="epigraph">
<p>混沌为秩序之母。</p>
<footer>—— 诺伯特 · 维纳（Norbert Wiener）</footer>
</div>

<div class="article-byline">
<p>第二级 · 不确定性量化 ｜ Smith《Uncertainty Quantification》Ch.7 ｜ 2026-08-07</p>
</div>

## 为什么从多项式混沌展开开始

蒙特卡洛是正向传播的「笨办法」：抽一万个样本，跑一万次仿真，直白但昂贵——一次仿真可能是几小时的气象模式或几天的有限元计算，一万次不可承受。**多项式混沌展开（Polynomial Chaos Expansion, PCE）**提供一条「聪明」的路：不再逐点逼近，而是把随机输出看成一个无穷维空间里的函数，用一组正交多项式把它展开成级数。就像傅里叶级数用正弦余弦分解周期函数，**PCE 用正交多项式分解随机函数**。一旦展开完成，均值、方差乃至灵敏度指数都能从系数里**解析读出**，不再需要任何额外仿真。这是谱方法（spectral method）在随机空间的投影。<span class="marginnote">名字里的「混沌」来自维纳 1938 年的直觉：高斯过程的混沌分解可以推广到一般随机过程。今天 PCE 早已脱离这个起源，成了 UQ 工具箱里最锋利的一把刀——它对随机输入输出建立了「泰勒展开式」级别的解析表示。</span>

## 1 谱方法的核心思想

设模型输出 $Y$ 是 $d$ 个随机输入 $\boldsymbol\xi = (\xi_1, \dots, \xi_d)$ 的函数：$Y = f(\boldsymbol\xi)$。如果 $f$ 足够光滑，我们可以把它写成：

$$
Y(\boldsymbol\xi) = \sum_{\boldsymbol\alpha \in \mathbb{N}^d} c_{\boldsymbol\alpha}\, \Psi_{\boldsymbol\alpha}(\boldsymbol\xi)
$$

这里 $\boldsymbol\alpha$ 是一个多重指标（multi-index）——每个分量的非负整数次幂的组合，$\Psi_{\boldsymbol\alpha}$ 是**多元正交多项式**，$c_{\boldsymbol\alpha}$ 是展开系数。三个要点：

**正交性是灵魂**：$\Psi_{\boldsymbol\alpha}$ 关于输入分布是正交的，即 $\mathbb{E}[\Psi_{\boldsymbol\alpha}(\boldsymbol\xi)\Psi_{\boldsymbol\beta}(\boldsymbol\xi)] = \delta_{\boldsymbol\alpha\boldsymbol\beta}$。正交性让「求一个系数」和「求其他系数」互不干扰，也让均值、方差变成系数的简单函数。

**输入分布决定多项式族**：高斯输入配埃尔米特多项式、均匀输入配勒让德多项式——每个分布都有一族「天生契合」的正交多项式，这就是 Wiener–Askey 对应表。

**无穷级数，实际截断**：理论上是无穷和，实际只保留有限项。截断就是选择「保留哪些多重指标」，通常按总阶数 $p$ 截断：$\lvert\boldsymbol\alpha\rvert = \alpha_1 + \dots + \alpha_d \le p$。

**辨析｜易错点：** 初学者容易把 PCE 和「对输入做多项式拟合」搞混。PCE 的巧妙在于：**正交多项式是按输入的分布构造的，所以系数天然解耦，统计量不依赖仿真重跑就能算出。** 普通多项式拟合没有这个性质，只能逐点拟合。

## 2 正交多项式家族：Wiener–Askey 对应

不同分布对应的正交多项式如下表。这张表是 PCE 的「字典」，查表即可：

| 输入分布 | 正交多项式 | 支撑区间 | 备注 |
| --- | --- | --- | --- |
| 标准正态 $\mathcal{N}(0,1)$ | 埃尔米特 $He_n$ | $(-\infty,\infty)$ | 最常用，经典 Wiener 混沌 |
| 均匀 $\mathcal{U}(-1,1)$ | 勒让德 $P_n$ | $[-1,1]$ | 工程参数建模常用 |
| 伽马 $\mathrm{Ga}(a)$ | 拉盖尔 $L_n$ | $[0,\infty)$ | 非负量（强度、寿命） |
| 贝塔 $\mathrm{Be}(a,b)$ | 雅可比 $P_n^{(a,b)}$ | $[0,1]$ | 概率类参数（如存活率） |
| 泊松 / 二项 | 恰比雪夫离散 | 离散点集 | 计数型输入 |

一维多项式满足三项递推关系 $P_{n+1}(\xi) = (a_n \xi + b_n) P_n(\xi) - c_n P_{n-1}(\xi)$，高阶多项式由低阶递推生成——这是数值实现的核心算法，也是为什么 PCE 库能快速生成任意阶多项式的原因。<span class="marginnote">三项递推与第二级《数值分析》里求正交多项式的高斯求积节点是同一条原理。事实上 PCE 的系数计算大量复用高斯积分——「随机配点与稀疏网格」（第 5 篇）正是从这里长出来的。</span>

**独立性约定**：多数教科书默认输入各分量相互独立，多元多项式退化成各维多项式的乘积 $\Psi_{\boldsymbol\alpha}(\boldsymbol\xi) = \prod_{j=1}^{d} P_{\alpha_j}(\xi_j)$。相关输入需要先做独立化变换（如 Nataf / Rosenblatt 变换）——这是工程应用里最常见的预处理步骤。

## 3 两种求系数的方法

展开式好写，系数 $c_{\boldsymbol\alpha}$ 怎么求？两条主流路线：

**谱投影法（spectral projection）**：利用正交性，把两边乘以 $\Psi_{\boldsymbol\beta}$ 再取期望：

$$
c_{\boldsymbol\beta} = \frac{\mathbb{E}\big[Y(\boldsymbol\xi)\, \Psi_{\boldsymbol\beta}(\boldsymbol\xi)\big]}{\mathbb{E}\big[\Psi_{\boldsymbol\beta}^2(\boldsymbol\xi)\big]}
$$

分子是 $d$ 重积分，用高斯求积（张量积）近似。优点是稳定、非侵入式（只需调用仿真器），缺点是维数升高后积分点数爆炸。

**回归法（regression / point collocation）**：在随机空间里选 $N$ 个样本点 $\boldsymbol\xi^{(i)}$，逐点跑仿真得 $Y^{(i)} = f(\boldsymbol\xi^{(i)})$，然后解最小二乘：

$$
\min_c \; \sum_{i=1}^{N} \Big( Y^{(i)} - \sum_{\boldsymbol\alpha} c_{\boldsymbol\alpha} \Psi_{\boldsymbol\alpha}(\boldsymbol\xi^{(i)}) \Big)^2
$$

系数由一次矩阵求解得到。样本点可以比系数个数多一点（经验是 2–3 倍），配合稀疏技巧可应对高维。**回归法把「跑仿真」压缩成「一次实验设计 + 一次最小二乘」**，是工业界默认选择。

## 4 公式解析：统计量从系数直接读出

PCE 最迷人的回报在这里：**一旦系数求出来，均值与方差就是几个平方和。** 设截断后的展开为 $Y = \sum_{\boldsymbol\alpha} c_{\boldsymbol\alpha}\Psi_{\boldsymbol\alpha}$，利用正交性 $\mathbb{E}[\Psi_{\boldsymbol\alpha}\Psi_{\boldsymbol\beta}] = \delta_{\boldsymbol\alpha\boldsymbol\beta}$：

- **均值**：$\mu_Y = \mathbb{E}[Y] = c_{\boldsymbol 0}$。常数项就是均值——因为所有非常数多项式期望为 0。
- **方差**：$\sigma^2_Y = \mathbb{E}\big[(Y - \mu_Y)^2\big] = \sum_{\boldsymbol\alpha \neq \boldsymbol 0} c_{\boldsymbol\alpha}^2 \,\mathbb{E}[\Psi_{\boldsymbol\alpha}^2]$。方差是**所有非零系数的加权平方和**——每一项代表一个随机模态的贡献。
- **各维贡献**：把多重指标按「哪些维非零」分组，某组平方和就是该组输入对总方差的贡献。这正是第 7 篇 **Sobol 指数**的来源——PCE 系数天然给出灵敏度分析。

为了更直观，看一个一维例子。设 $Y = a + b\,\xi + c\,(\xi^2 - 1)$，其中 $\xi\sim\mathcal{N}(0,1)$，用的是埃尔米特多项式 $He_0=1, He_1=\xi, He_2=\xi^2-1$。则有：

- 均值 $\mu_Y = a$（零阶系数）；
- 方差 $\sigma^2_Y = b^2 + 2c^2$（注意二阶埃尔米特的范数是 2）。

**这个例子揭示了一个深刻事实：均值与方差是完全正交的两个信息通道，互不干扰。** 你调整系数 $b$ 只改变方差不改均值，调整 $c$ 同理——这在蒙特卡洛里是不可能的（它只能给你一堆样本去数）。

## 5 代价与维数灾难

PCE 不是免费的午餐。截断到总阶 $p$、输入维数 $d$ 时，系数个数为：

$$
M = \binom{p+d}{d}
$$

当 $d$ 增大时 $M$ 爆炸——这就是**维数灾难（curse of dimensionality）**。$p=3, d=10$ 时 $M=286$；$d=50$ 时 $M\approx 23{,}400$。对应地，求系数需要成比例地跑仿真或积分点。工程界的应对三板斧：

**稀疏化**：绝大多数系数实际接近 0，用 L1 正则化（LASSO）只保留显著项——这呼应了第 3 篇「L1 先验给稀疏解」的思想。

**降维先行**：先用第 7 篇的灵敏度分析筛出影响大的输入，把 $d$ 压到个位数再建 PCE。

**稀疏网格**：用第 5 篇的稀疏网格替代张量积网格做谱投影，积分点从指数级降到多项式级。

<span class="marginnote">维数灾难不是 PCE 独有，它是所有谱方法的天花板。理解 $M=\binom{p+d}{d}$ 这个组合数的增长，是你评估「这套 PCE 到底可行不可行」的第一判断工具。</span>

## 6 核心术语速查表与常见误区

| 术语 | 英文 | 一句话含义 | 本章对应 |
| --- | --- | --- | --- |
| 多项式混沌展开 | PCE | 用正交多项式级数表示随机输出 | 第 1 节 |
| 多重指标 | multi-index | 记录各维幂次的向量 $\boldsymbol\alpha$ | 第 1 节 |
| 正交性 | orthogonality | 不同基函数期望为 0，系数解耦 | 第 1 节 |
| Wiener–Askey 对应 | Wiener–Askey scheme | 分布与正交多项式族的对应表 | 第 2 节 |
| 谱投影 | spectral projection | 用积分正交性求系数 | 第 3 节 |
| 回归法 | point collocation | 随机点 + 最小二乘求系数 | 第 3 节 |
| 维数灾难 | curse of dimensionality | 系数个数 $\binom{p+d}{d}$ 爆炸 | 第 5 节 |

**问：PCE 能处理有相关性的输入吗？** 不能直接用。PCE 的方差分解依赖输入独立；输入相关时需要先做独立化变换（Nataf 或 Rosenblatt 变换）把相关输入映射到独立高斯/均匀变量，再做展开。**跳过独立化直接展开，会得到虚假的灵敏度结论**。

**问：函数不光滑时 PCE 还行吗？** 收敛会明显变慢。PCE 本质是多项式近似，对间断（如失效指示函数）会出现吉布斯式振荡。此时要么对输出做平滑变换，要么退回蒙特卡洛或专门处理间断的稀疏方法。

## 7 小结

- **PCE** 把随机输出展开成正交多项式级数 $Y=\sum c_{\boldsymbol\alpha}\Psi_{\boldsymbol\alpha}$，正交性让系数解耦、统计量解析。
- 输入分布决定多项式族：**高斯 → 埃尔米特、均匀 → 勒让德**等（Wiener–Askey 对应）。
- 求系数两条路：**谱投影**（高斯积分近似期望）与**回归法**（随机点 + 最小二乘）。
- 展开后**均值 = 常数项**，**方差 = 非零系数平方和**，且各维贡献直接给出 Sobol 指数。
- 代价是维数灾难：系数个数 $\binom{p+d}{d}$