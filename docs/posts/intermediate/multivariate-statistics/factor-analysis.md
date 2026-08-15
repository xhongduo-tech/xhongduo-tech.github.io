---
title: 因子分析
date: 2026-08-07
---

# 因子分析

<div class="epigraph">
<p>如果一群变量彼此纠缠，那么最简洁的解释往往是：背后有一两个看不见的共同原因，其余都是噪音。</p>
<footer>—— 查尔斯·斯皮尔曼（Charles Spearman）</footer>
</div>

<div class="article-byline">
<p>第二级 · 多元统计分析 ｜ Anderson《An Introduction to Multivariate Statistical Analysis》Ch.14 · Johnson & Wichern Ch.9 ｜ 2026-08-07</p>
</div>

## 为什么需要看不见的因子

PCA 把 $p$ 个变量转成 $p$ 个新变量，方差按轴重新分配——但主成分仍然是「原始变量的混叠」，且没有误差概念。**因子分析（factor analysis, FA）**问一个更深的问题：变量之间的相关，是不是源于少数几个**不可观测的共同原因**？斯皮尔曼 1904 年测学生成绩时发现：语文、数学、历史成绩两两都正相关。与其用「三门课都难」这种事后解释，不如假设背后有一个共同的「一般智力」因子。**FA 的雄心是：用少数潜在因子 + 每个变量自己的独特噪音，重现整个相关矩阵**。<span class="marginnote">因子分析最早来自心理学，今天的应用横跨市场调研（满意度量表）、金融（风险因子）与基因表达（潜在调控因子）。它与 PCA 的关系见上一节的对照表：PCA 是几何旋转，FA 是带误差的潜在变量模型。</span>

## 1 正交因子模型

**正交因子模型（orthogonal factor model）**假定可观测向量 $\mathbf{X}$ 由 $m$ 个公共因子与 $p$ 个特殊因子叠加而成：

$$
\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{F} + \boldsymbol{\varepsilon}
$$

逐行写开更直观：对第 $j$ 个变量

$$
X_j = \mu_j + \ell_{j1}F_1 + \ell_{j2}F_2 + \cdots + \ell_{jm}F_m + \varepsilon_j
$$

其中 $\mathbf{F} = (F_1,\ldots,F_m)'$ 是**公共因子（common factors）**，$\boldsymbol{\varepsilon} = (\varepsilon_1,\ldots,\varepsilon_p)'$ 是**特殊因子（specific factors）**，$\mathbf{L} = (\ell_{jk})$ 是 $p \times m$ 的**载荷矩阵（loading matrix）**。假设公共因子不相关、特殊因子彼此不相关也不与公共因子相关：$E(\mathbf{F})=\mathbf{0}$，$\operatorname{Cov}(\mathbf{F}) = \mathbf{I}_m$，$\operatorname{Cov}(\boldsymbol{\varepsilon}) = \boldsymbol{\Psi} = \operatorname{diag}(\psi_1,\ldots,\psi_p)$，$\operatorname{Cov}(\mathbf{F},\boldsymbol{\varepsilon})=\mathbf{0}$。<span class="marginnote">「正交」指公共因子彼此不相关（$\operatorname{Cov}(\mathbf{F})=\mathbf{I}$），这与 PCA 的「主成分彼此不相关」是同一句诺言；放松它（允许因子相关）就是斜交因子模型，解释更自由但数学更繁。</span>**模型写完后必须问一句：$m$ 个因子够不够？** 这是 FA 每一步决策的总纲，答案在后面的第 5 节。



## 2 协方差结构的分解：FA 的签名公式

由模型可以直接推出 $\mathbf{X}$ 的协方差矩阵：

$$
\operatorname{Cov}(\mathbf{X}) = \mathbf{L}\mathbf{L}' + \boldsymbol{\Psi}
$$

这是因子分析**最核心的等式**：**变量间的相关，全部来自公共因子的共享载荷 $\mathbf{L}\mathbf{L}'$；变量各自的多余方差，来自 $\boldsymbol{\Psi}$**。逐元展开，第 $j$ 个变量的方差被拆成两部分：

$$
\operatorname{Var}(X_j) = \underbrace{\ell_{j1}^2 + \ell_{j2}^2 + \cdots + \ell_{jm}^2}_{\text{共同度（communality）}} + \underbrace{\psi_j}_{\text{特殊方差}}
$$

其中共同度 $h_j^2 = \sum_{k=1}^m \ell_{jk}^2$ 是「第 $j$ 个变量的方差被公共因子解释的份额」，特殊方差 $\psi_j$ 是「只属于它自己的份额」。$h_j^2$ 越接近 $\operatorname{Var}(X_j)$，这个变量与公共因子的联系越紧密——**共同度是 FA 中衡量「这个变量有多少由潜在结构解释」的标尺**。<span class="marginnote">注意 $\operatorname{Cov}(X_j, X_k) = \ell_{j1}\ell_{k1} + \cdots + \ell_{jm}\ell_{km}$：两个变量的协方差完全由它们对公共因子的共同依赖决定，特殊因子只贡献自己的方差、不贡献任何协方差。这就是「公共因子制造相关」的数学表达。</span>

## 3 估计：主成分法与最大似然法

模型写好了，怎么从样本协方差矩阵 $\mathbf{S}$ 反推出 $\mathbf{L}$ 与 $\boldsymbol{\Psi}$？两条主流路线：

**主成分法（principal factor method）**：对 $\mathbf{S}$（或相关矩阵 $\mathbf{R}$）做谱分解 $\mathbf{S} = \sum_{k=1}^p \lambda_k\mathbf{e}_k\mathbf{e}_k'$，取前 $m$ 个特征值与特征向量构造

$$
\hat{\mathbf{L}} = \Bigl(\sqrt{\lambda_1}\,\mathbf{e}_1,\ \sqrt{\lambda_2}\,\mathbf{e}_2,\ \ldots,\ \sqrt{\lambda_m}\,\mathbf{e}_m\Bigr), \qquad \hat{\psi}_j = s_{jj} - \sum_{k=1}^m \hat{\ell}_{jk}^2
$$

直觉：用 $m$ 个最大的主轴近似协方差矩阵，剩下的方差全算作特殊方差。<span class="marginnote">主成分法快速但不保证 $\hat{\psi}_j \geq 0$（共同度可能算出负数，叫 Heywood 情形）。出现负特殊方差时说明 $m$ 取得太小或模型不合适——这是 FA 里最著名的警告信号。</span>

**最大似然法（ML）**：假设 $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \mathbf{L}\mathbf{L}'+\boldsymbol{\Psi})$，用数值优化极大化似然，得到 $\hat{\mathbf{L}}, \hat{\boldsymbol{\Psi}}$。它需要正态假设，但给出了检验「$m$ 个因子是否足够」的似然比检验——这是主成分法给不了的。

## 4 因子旋转：让载荷讲故事

初始解在数学上最优，在解释上常常一团糟：每个变量的载荷分布均匀、因子含义模糊。**因子旋转（factor rotation）**是 FA 独有的后处理：把载荷矩阵乘一个正交矩阵 $\mathbf{T}$（$\mathbf{T}'\mathbf{T} = \mathbf{I}$），得到新载荷 $\mathbf{L}^* = \mathbf{L}\mathbf{T}$，模型不变（$\mathbf{L}\mathbf{L}' = \mathbf{L}^*\mathbf{L}^{*\prime}$），但载荷结构被重排成更「稀疏」的形状——每个变量尽量只在一个因子上有大载荷。<span class="marginnote">旋转不改变共同度、不改变 $\Psi$、不改变拟合——它只改变「因子如何被解释」。最常用的是 <strong>varimax（方差极大）旋转</strong>，让每个因子上的载荷平方方差最大，倾向得到「一个变量主要属于一个因子」的清晰结构。这也是 FA 与 PCA 最大的体验差异：PCA 的轴是固定的，FA 的因子可以转动。</span>

旋转后，变量归因变得可读：比如 20 个满意度题目最后落在「服务」「价格」「质量」三个因子上，每个题目归入载荷最大的那个因子——**这就是从数据里「长出」一份量表的经典流程**。

### 主成分法 vs 最大似然法速查

| 对比项 | 主成分法 | 最大似然法 |
| --- | --- | --- |
| 分布假设 | 无 | 多元正态 |
| 计算 | 谱分解，快 | 数值迭代，慢 |
| 检验因子个数 | 无 | 似然比检验 |
| 缺陷 | 可能 Heywood 负方差 | 对假设敏感，可能不收敛 |

## 5 选几个因子与实战流程

因子个数 $m$ 的选择比 PCA 的「主成分个数」更微妙，因为 FA 还受可识别性约束。惯用判据：

**累计共同度**：前 $m$ 个因子解释的总共同度 $\sum_j h_j^2 / p$ 达到 70% 上下（对相关矩阵）。
**特征值 ≥ 1（Kaiser）**：对「约化相关矩阵」（对角线换共同度初值的矩阵）做谱分解，保留特征值 ≥ 1 的因子。
**似然比检验**：用 ML 估计时，检验 $H_0$:「$m$ 个因子足够」对「$m+1$ 个因子更好」，逐步加因子直到不显著。<span class="marginnote">三种判据常常给出不同答案。经验做法是「宁可少选、保证可解释」：多一个因子就多一整套载荷要命名，因子数超过变量数的一半时模型通常已经失去意义。</span>

标准流程与 PCA 高度平行，多出两个 FA 专属步骤：

1. 计算相关矩阵 $\mathbf{R}$（或 $\mathbf{S}$），估计共同度初值（常用 $R^2$ 与其余变量的复相关平方）。
2. 用主成分法或 ML 提取 $m$ 个因子，得到初始载荷 $\hat{\mathbf{L}}$。
3. **旋转**（varimax 或斜交 promax）得到 $\mathbf{L}^*$。
4. 解释因子：每个因子上载荷绝对值最大的若干变量定义其含义。
5. 算**因子得分**（factor scores）：把每个样本投影到因子上，用于后续回归、聚类。

**最后一步提醒**：因子得分不是主成分得分——因子是潜在变量，得分只是它的估计，不同方法（回归法、Bartlett 法）给出不同数值。**用因子得分做正式统计推断时要格外小心**，它带有一层估计误差，p 值偏乐观。

## 6 公式解析：为什么协方差 = LL′ + Ψ

把模型 $\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{F} + \boldsymbol{\varepsilon}$ 翻译成协方差，是理解 FA 全部直觉的一步：

- **第一步，写协方差定义**：$\operatorname{Cov}(\mathbf{X}) = E[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})'] = E[(\mathbf{L}\mathbf{F}+\boldsymbol{\varepsilon})(\mathbf{L}\mathbf{F}+\boldsymbol{\varepsilon})']$。
- **第二步，展开并利用独立性**：交叉项 $E(\mathbf{F}\boldsymbol{\varepsilon}') = \mathbf{0}$（特殊因子与公共因子不相关），于是只剩两项 $E(\mathbf{L}\mathbf{F}\mathbf{F}'\mathbf{L}') + E(\boldsymbol{\varepsilon}\boldsymbol{\varepsilon}')$。
- **第三步，代期望**：$E(\mathbf{F}\mathbf{F}') = \operatorname{Cov}(\mathbf{F}) = \mathbf{I}_m$，$E(\boldsymbol{\varepsilon}\boldsymbol{\varepsilon}') = \boldsymbol{\Psi}$，得 $\mathbf{L}\mathbf{I}_m\mathbf{L}' + \boldsymbol{\Psi} = \mathbf{L}\mathbf{L}' + \boldsymbol{\Psi}$。
- **第四步，数自由度**：$\mathbf{L}\mathbf{L}'+\boldsymbol{\Psi}$ 有 $p(p+1)/2$ 个独立参数待拟合，模型参数 $pm + p$ 个；可识别性要求 $p(p+1)/2 \geq pm+p$。**因子数 $m$ 不能贪多**——这是「选几个因子」的代数硬约束。

**核心结论：FA 的整个数学，就是「把协方差矩阵拆成低秩部分 $\mathbf{L}\mathbf{L}'$ 与对角部分 $\boldsymbol{\Psi}$」**。低秩部分承载变量间的相关，对角部分承载各自的独特性。比 PCA 多出的这一项 $\boldsymbol{\Psi}$，正是「潜在原因 + 个体噪音」这个故事的数学化身。

## 7 小结

- **正交因子模型** $X_j = \mu_j + \ell_{j1}F_1 + \cdots + \ell_{jm}F_m + \varepsilon_j$：公共因子制造相关，特殊因子只贡献个体方差。
- **签名公式** $\operatorname{Cov}(\mathbf{X}) = \mathbf{L}\mathbf{L}' + \boldsymbol{\Psi}$；第 $j$ 个变量的方差 = 共同度 $h_j^2$ + 特殊方差 $\psi_j$。
- 估计用**主成分法**（谱分解，快，可能负方差）或**最大似然法**（要正态，可检验因子个数）。
- **因子旋转**（尤其 varimax）不改变拟合、只改变可解释性，让每个变量尽量归属单一因子。
- FA 是**带误差的潜在变量模型**，PCA 是无误差的几何旋转——两者别混用。

在下一节，我们把「降维」换成「分类」：**判别分析**——已知样本属于哪一组，如何利用 $p$