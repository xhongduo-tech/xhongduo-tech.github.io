---
title: 广义最小二乘与加权最小二乘
date: 2026-08-07
---

# 广义最小二乘与加权最小二乘

<div class="epigraph">
<p>当噪声不再一视同仁，聪明的人会给更可靠的数据更大的发言权。</p>
<footer>—— 依统计加权思想改写（paraphrase of weighted estimation）</footer>
</div>

<div class="article-byline">
<p>第二级 · 回归分析与线性模型 ｜ Seber & Lee《线性回归分析》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从广义最小二乘开始

到目前为止，我们一直假设误差协方差是 $\sigma^2\mathbf{I}$——所有观测等方差且不相关。但真实数据常常打破这条假设：收入数据的方差随水平上升而增大（异方差），时间序列的误差前后相关（自相关），分组抽样导致组内相关。Gauss-Markov 定理的前提一破，OLS 就不再是最优的。**广义最小二乘（GLS）** 用一次「坐标变换」把一般协方差结构变回 $\sigma^2\mathbf{I}$，从而找回最优性；**加权最小二乘（WLS）** 是它的对角线特例，也是最常用的落地形式。

## 1 一般模型设定：误差不再白

更一般的线性模型把误差假设放宽为：

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \qquad
E(\boldsymbol{\varepsilon}) = \mathbf{0}, \qquad \mathrm{Var}(\boldsymbol{\varepsilon}) = \sigma^2 \mathbf{V}
$$

其中 $\mathbf{V}$ 是已知的**正定矩阵**，刻画误差的方差与相关结构。$\mathbf{V} = \mathbf{I}$ 回到普通 OLS；$\mathbf{V}$ 为对角阵时是加权情形；$\mathbf{V}$ 非对角时含相关性。

**核心思想：白化（whitening）**。因为 $\mathbf{V}$ 正定，存在可逆矩阵 $\mathbf{K}$ 使 $\mathbf{V} = \mathbf{K}\mathbf{K}'$（如 Cholesky 分解）。用 $\mathbf{K}^{-1}$ 左乘模型两边：

$$
\mathbf{y}^* = \mathbf{X}^*\boldsymbol{\beta} + \boldsymbol{\varepsilon}^*, \qquad
\mathbf{y}^* = \mathbf{K}^{-1}\mathbf{y},\quad
\mathbf{X}^* = \mathbf{K}^{-1}\mathbf{X},\quad
\boldsymbol{\varepsilon}^* = \mathbf{K}^{-1}\boldsymbol{\varepsilon}
$$

变换后的误差满足 $\mathrm{Var}(\boldsymbol{\varepsilon}^*) = \sigma^2 \mathbf{I}$——**问题被还原成了 OLS**。<span class="marginnote">「变换数据回到标准形，再做 OLS」是 GLS 的全部秘密。这也解释了为什么许多教科书称 GLS 为「加权 OLS」：变换的本质是给不同观测重新加权。</span>

## 2 公式解析：GLS 估计量

对变换后的模型做 OLS，得到 GLS 估计量：

$$
\hat{\boldsymbol{\beta}}_{\text{GLS}} = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{y}
$$

逐步拆解：

- **$\mathbf{V}^{-1}$**：精度矩阵，是 $\mathbf{V}$ 的逆，决定每个观测的「权重」。$\mathbf{V}^{-1}$ 在公式里出现两次（一次在 $\mathbf{X}$ 前、一次在 $\mathbf{y}$ 前），共同实现「白化」；
- **$(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$**：加权后的信息矩阵之逆，取代 OLS 里的 $(\mathbf{X}'\mathbf{X})^{-1}$；
- **若 $\mathbf{V} = \mathbf{I}$**：退化为 $(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}$，GLS 是 OLS 的推广；
- **最优性**：在 $\mathrm{Var}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{V}$ 下，GLS 是 BLUE——它是 Gauss-Markov 定理在一般协方差下的推广（常称 Aitken 定理）。

它的协方差矩阵为：

$$
\mathrm{Var}(\hat{\boldsymbol{\beta}}_{\text{GLS}}) = \sigma^2(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}
$$

**重点结论**：GLS 的公式结构处处是把 $\mathbf{V}^{-1}$ 作为「权重」插入。$\mathbf{V}$ 已知时，GLS 无偏、最优，一切检验照旧。

## 3 加权最小二乘：对角阵特例

当误差只异方差、不相关时，$\mathbf{V} = \mathrm{diag}(w_1^{-1}, \ldots, w_n^{-1})$。GLS 退化为 **WLS**：

$$
\hat{\boldsymbol{\beta}}_{\text{WLS}} = \arg\min \sum_{i=1}^{n} w_i\, (y_i - \mathbf{x}_i'\boldsymbol{\beta})^2
$$

目标函数变成**加权残差平方和**：方差小（$w_i$ 大）的观测被赋予更大权重。<span class="marginnote">权重 $w_i \propto 1/\sigma_i^2$：波动小的数据可信度高，话语权大；波动大的数据被压低。这符合直觉——同一尺子量出的数，精度高者更值得信任。</span>

**典型场景**：

各观测是**均值**而非个体（第 $i$ 组的观测是 $n_i$ 个数据的平均），则 $w_i = n_i$；
方差随某个变量成比例（如 $\sigma_i^2 \propto x_i$），则 $w_i = 1/x_i$；
- 财务数据：大公司波动大，按其规模倒数加权。

**辨析｜易错点：** 权重 $w_i$ 应该反映**已知的相对精度**，不能把 $w_i$ 当自由参数去「试出好看的 $R^2$」——那会严重扭曲推断。权重选择必须来自数据生成机制或先验知识。

## 4 异方差的诊断与补救流程

WLS 的前提是知道 $\mathbf{V}$。实际中 $\mathbf{V}$ 常未知，需要「估计」它——这就是**可行的广义最小二乘（FGLS）** 的思路：

**第 1 步**：跑 OLS，取残差 $e_i$；
**第 2 步**：诊断异方差——残差对拟合值画图（喇叭形）、或做检验（如 Breusch–Pagan 检验）；
**第 3 步**：建模方差结构，估计权重 $\hat{w}_i$（如 $\log \hat{\sigma}_i^2$ 对 $x$ 回归）；
**第 4 步**：用估计权重跑 WLS，再检查新残差是否同方差。

<span class="marginnote">注意 FGLS 的两阶段特性：权重本身是估计的，故推断理论上需修正（这里存在 bootstrap 或小样本校正的空间）。实践中，WLS 常显著改善效率，即便权重只是近似。</span>

**辨析｜易错点：** 面对异方差，并非只有 WLS 一条路。**稳健标准误（robust / sandwich standard errors）** 保持 OLS 系数不变、只修正标准误，在系数一致但标准误不可靠时是更省事的选择；而 WLS 会改变系数本身。两者的取舍取决于「你是否相信方差结构模型」。

## 5 核心对比：OLS、WLS、GLS

| 维度 | OLS | WLS | GLS |
| --- | --- | --- | --- |
| 误差协方差 | $\sigma^2\mathbf{I}$ | $\sigma^2\mathrm{diag}(w^{-1})$ | $\sigma^2\mathbf{V}$ 任意正定 |
| 目标函数 | 平方和 | 加权平方和 | 白化后平方和 |
| 权重位置 | 无 | 每观测 $w_i$ | 矩阵 $\mathbf{V}^{-1}$ |
| 最优性 | BLUE（$\mathbf{V}=\mathbf{I}$ 时） | 异方差下的 BLUE | 一般下的 BLUE |
| 复杂度 | 低 | 中 | 高 |

<span class="marginnote">一句话串起三兄弟：GLS 是家族大哥，WLS 是「只处理异方差」的特例，OLS 是「等方差白噪声」的最简情形。熟悉任何一者，都可通过 $\mathbf{V}$ 的设定滑向另外两者。</span>

**何时该用 WLS 而非稳健标准误**：如果误差方差结构可建模、且你关心**系数本身**的估计（而非仅标准误），WLS 更合适——它通过加权直接改善估计效率；如果你对权重结构没有把握、只想要「可信的检验」，稳健标准误更省事。实践中常先跑 OLS 看稳健标准误，再视诊断决定是否升级到 WLS。两类方法的共同前提是误差**均值仍为零**——若存在系统性偏差，加权再多也无济于事。此外，权重一旦从数据估计，推断就应视为近似，样本小时尤须谨慎。

**一个落地提醒**：GLS/WLS 的收益在「效率」而非「无偏」——当 $n$ 很大时，OLS 即使不是最优也足够好；当 $n$ 小且异方差明显时，WLS 的改进才真正可见。先诊断，再决定要不要「上装备」。

## 6 小结

- 一般模型 $\mathrm{Var}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{V}$；GLS 用白化变换 $\mathbf{K}^{-1}$ 还原为 OLS。
- GLS 估计量 $\hat{\boldsymbol{\beta}}_{\text{GLS}} = (\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}\mathbf{y}$，方差 $\sigma^2(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}$。
- **WLS** 是 $\mathbf{V}$ 对角时的特例：加权残差平方和最小化，权重 $\propto 1/\sigma_i^2$。
- 权重必须来自已知精度，不能自由调参；$\mathbf{V}$