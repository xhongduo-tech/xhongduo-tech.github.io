---
title: 判别分析：Fisher 判别与贝叶斯判别
date: 2026-08-07
---

# 判别分析：Fisher 判别与贝叶斯判别

<div class="epigraph">
<p>分类问题在统计里已经有一百岁了：把每个新个体放回它最该属于的那一堆，代价要最小、错分要最少。</p>
<footer>—— 罗纳德·费希尔（Ronald A. Fisher）</footer>
</div>

<div class="article-byline">
<p>第二级 · 多元统计分析 ｜ Anderson《An Introduction to Multivariate Statistical Analysis》Ch.6 · Johnson & Wichern Ch.11 ｜ 2026-08-07</p>
</div>

## 为什么需要判别分析

前面的方法都在「无监督」地探索结构。现在换一个问题：**已经有 $k$ 个已知分组（如健康人 / 病人），来了一个新样本，该把它分到哪组？** 医生靠化验指标判断是否患病，银行靠客户特征判断是否违约，质检靠多项测量判断产品合格与否——这些都需要用已知类别的样本学出一条**分类规则（classification rule）**，再对未知样本做预测。这就是**判别分析（discriminant analysis, DA）**。<span class="marginnote">判别分析是「监督学习」在统计里的经典形态，与第三级《机器学习》里的分类器一脉相承：给定标签学决策边界。它的两大流派——贝叶斯判别与 Fisher 判别——分别从「概率模型」和「投影优化」两个角度给出答案，殊途同归时彼此印证。</span>

## 1 贝叶斯判别：让期望错分代价最小

把类别视为随机事件。第 $i$ 个类别出现的先验概率记 $\pi_i$，新样本 $\mathbf{x}$ 在类别 $i$ 里的条件密度记 $f_i(\mathbf{x})$。若把 $\mathbf{x}$ 分到类别 $j$，由贝叶斯公式，后验概率为

$$
P(\text{类别 } j \mid \mathbf{x}) = \frac{\pi_j f_j(\mathbf{x})}{\sum_{i=1}^k \pi_i f_i(\mathbf{x})}
$$

**贝叶斯决策规则：把 $\mathbf{x}$ 分到后验概率最大的那一类**。它等价于最大化 $\pi_j f_j(\mathbf{x})$，因为分母对所有 $j$ 相同。加入代价函数（把「类别 $j$ 的人误判成类别 $i$」的代价 $c(i|j)$）后，规则改为最小化期望代价——**先验概率、密度、代价三者齐备，错分率在理论上达到最小**，这是任何其他分类器的天花板。<span class="marginnote">贝叶斯判别的「最优性」是统计决策理论的核心定理：当且仅当按后验概率（或加权代价）分类时，期望错分代价最小。现实中先验 $\pi_i$ 与密度 $f_i$ 都靠估计，所以「理论上最优」落到「实践上接近最优」。</span>

### 两类情形与似然比

$k=2$ 时，规则退化成一条似然比不等式：分到类别 1 当且仅当

$$
\frac{f_1(\mathbf{x})}{f_2(\mathbf{x})} > \frac{\pi_2\, c(1|2)}{\pi_1\, c(2|1)}
$$

左边是**似然比**，右边是阈值。取对数后，比较 $\ln f_1(\mathbf{x}) - \ln f_2(\mathbf{x})$ 与一个常数。**把「分类」变成「两个密度之比的比较」**——这是统计分类最干净的形式。

## 2 正态假设下的线性与二次判别

若假设每类都服从多元正态，$f_i(\mathbf{x}) = \mathcal{N}_p(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$，则判别规则有封闭形式。

**协方差相等时——线性判别分析（LDA）**：设 $\boldsymbol{\Sigma}_1 = \cdots = \boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$，对数后验之差的二次项消去，只剩下线性函数。两类情形的判别规则是

$$
\mathbf{x} \ \text{分到类 1，当且仅当} \quad
\mathbf{a}'\mathbf{x} + a_0 > 0, \qquad
\mathbf{a} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)
$$

决策边界是**超平面** $\mathbf{a}'\mathbf{x} + a_0 = 0$，垂直于 $\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)$。<span class="marginnote">把 $\mathbf{a} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)$ 换成马氏距离的语言：边界位于到两类中心的马氏距离相等的点集上。<strong>马氏距离贯穿本章</strong>——前面均值检验用它，这里分类也用它。</span>

**协方差不等时——二次判别分析（QDA）**：二次项 $-\frac{1}{2}\mathbf{x}'(\boldsymbol{\Sigma}_1^{-1}-\boldsymbol{\Sigma}_2^{-1})\mathbf{x}$ 不再为零，决策边界变成**二次曲面**（超椭球或双曲面）。QDA 更灵活，但每个类都要估计整个 $\boldsymbol{\Sigma}_i$，参数暴增，小样本极易过拟合——**样本量不够时，宁可牺牲一点灵活性用 LDA**。

多类情形把判别写成**判别得分函数（discriminant score）**：对每个类别 $i$ 计算

$$
d_i(\mathbf{x}) = -\frac{1}{2}\ln|\boldsymbol{\Sigma}_i| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)'\boldsymbol{\Sigma}_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i) + \ln\pi_i
$$

然后分到得分最大的类。等协方差时 $-\frac{1}{2}\ln|\boldsymbol{\Sigma}|$ 是常数，$(\mathbf{x}-\boldsymbol{\mu}_i)'\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)$ 展开后二次项 $\mathbf{x}'\boldsymbol{\Sigma}^{-1}\mathbf{x}$ 与 $i$ 无关，于是得分退化为线性函数——**同一套得分公式，协方差等与不等自动给出线性或二次边界**，这是软件内部统一实现 LDA/QDA 的方式。

## 3 Fisher 判别：找最好的投影轴

贝叶斯判别需要一个概率模型。Fisher 的路线完全不用分布假设：**把 $p$ 维数据投影到一条直线 $y = \mathbf{a}'\mathbf{x}$ 上，让两类在这条线上分得最开**。度量「分得开」的准则是组间方差与组内方差之比：

$$
\max_{\mathbf{a}} \ \frac{\mathbf{a}'\mathbf{B}\mathbf{a}}{\mathbf{a}'\mathbf{W}\mathbf{a}}
$$

其中 $\mathbf{B}$ 是组间散布矩阵、$\mathbf{W}$ 是组内散布矩阵（与 MANOVA 的定义一致）。令导数为零得到广义特征方程：

$$
\mathbf{B}\mathbf{a} = \lambda \mathbf{W}\mathbf{a}, \qquad \text{即} \quad \mathbf{W}^{-1}\mathbf{B}\mathbf{a} = \lambda\mathbf{a}
$$

最大特征值对应的 $\mathbf{a}$ 就是最优投影方向。**Fisher 判别把「分类」转成「找一个方向」，把分类问题化成了特征值问题**——与 PCA 的谱分解是同一类数学，只是目标矩阵从「总散布」换成了「组间/组内之比」。<span class="marginnote">一个漂亮的巧合：两类且协方差相等时，Fisher 判别轴与贝叶斯 LDA 的判别方向 $\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)$ 只差一个常数——两条路线在此汇合。多类时 Fisher 给出 $k-1$ 个判别轴，张成的空间叫判别空间，可直接用于降维可视化。</span>

## 4 评估与实战：错分率与交叉验证

分类规则再好，也要回答「错得多不多」。评估工具：

**混淆矩阵（confusion matrix）**：行是真实类别、列是预测类别；对角线是对的分类数。二类时四个格子衍生出**敏感度**（正类被召回的比例）、**特异度**（负类被判对的占比）。
**错分率估计**：用训练数据自身的错分率（表观错误率）会乐观偏差严重——**正确做法是交叉验证（cross-validation）**：留一法或 K 折，把每个样本在「没见过它的规则」上预测一次。<span class="marginnote">留一法交叉验证与 LDA 之间有著名的捷径（Duda–Hart 公式）：删去一个样本对 $\bar{\mathbf{x}}_i$ 和 $\mathbf{S}_{\text{pooled}}$ 的影响可以解析算出，不必真的重拟合 $n$ 次。这是「统计量可解析更新」的经典例子，也解释了为什么 LDA 的留一验证特别便宜。</span>

**实战要点**：先检验两类协方差是否大致相等（Box's M 或 Q–Q 目检），相等用 LDA、不等用 QDA；先验概率不相等时，把 $\pi_i$ 写进规则（贝叶斯判别）；样本量小而变量多时，先做 PCA 降维或用正则化判别分析，避免 $\mathbf{W}$ 不可逆。

## 5 辨析｜判别分析与聚类分析

判别分析常被拿来和聚类分析比较——它们都「分组」，但方向完全相反：

| 对比项 | 判别分析（DA） | 聚类分析（Cluster） |
| --- | --- | --- |
| 类别是否已知 | 已知，有标签 | 未知，无标签 |
| 学习类型 | 监督学习 | 无监督学习 |
| 输出 | 分类规则 / 决策边界 | 分组结果 / 树状图 |
| 评估 | 错分率、混淆矩阵 | 组内凝聚度、轮廓系数 |
| 典型用途 | 辅助诊断、信用评分 | 市场细分、基因分型 |

一句话概括：**判别分析用「已知分组」学「怎么分」，聚类分析替「未知分组」找「分在哪里」**。两类方法常联用——先用聚类发现结构、打上标签，再用判别分析学习可复用的分类规则。<span class="marginnote">判别分析的「训练」还需要小心一个问题：类别不平衡时，多数类会压过少数类。对策是调整先验 $\pi_i$、对少数类重采样、或在代价函数里给少数类加权重——这和现代机器学习里的类不平衡处理是同一条思路。</span>

## 6 公式解析：Fisher 判别方向为什么是特征向量

一步步把「最大化组间/组内比」翻译成特征方程：

- **第一步，写目标函数**：$J(\mathbf{a}) = \mathbf{a}'\mathbf{B}\mathbf{a}/\mathbf{a}'\mathbf{W}\mathbf{a}$。它是齐次的：$\mathbf{a}$ 乘任意常数 $J$ 不变，所以可加约束 $\mathbf{a}'\mathbf{W}\mathbf{a} = 1$。
- **第二步，拉格朗日**：$\mathcal{L} = \mathbf{a}'\mathbf{B}\mathbf{a} - \lambda(\mathbf{a}'\mathbf{W}\mathbf{a} - 1)$，对 $\mathbf{a}$ 求导置零：$2\mathbf{B}\mathbf{a} - 2\lambda\mathbf{W}\mathbf{a} = \mathbf{0}$。
- **第三步，广义特征方程**：$\mathbf{B}\mathbf{a} = \lambda\mathbf{W}\mathbf{a}$；$\mathbf{W}$ 可逆时化为 $\mathbf{W}^{-1}\mathbf{B}\mathbf{a} = \lambda\mathbf{a}$。
- **第四步，读解**：最大特征值 $\lambda_{\max}$ 就是最大的判别准则值，对应特征向量是最优投影。$k$ 类时前 $k-1$ 个广义特征向量给出 $k-1$ 条判别轴。

**核心结论：Fisher 判别的最优方向 = 矩阵 $\mathbf{W}^{-1}\mathbf{B}$ 的最大特征向量**。把 PCA 的「总散布特征分解」换成「组间/组内特征分解」，无监督降维就变成了有监督降维——这一切换在后面的典型相关分析里还会以同样的配方出现。

## 7 小结

- **贝叶斯判别**按后验概率最大分类，理论上使期望错分代价最小；$k=2$ 时化为似然比比较。
- **正态假设下**：协方差相等得线性边界（LDA，用 $\mathbf{a} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)$），不等得二次边界（QDA）。
- **Fisher 判别**最大化组间/组内方差比，方向是 $\mathbf{W}^{-1}\mathbf{B}$ 的特征向量；与 LDA 在两类等协方差时重合。
- 评估用**混淆矩阵 + 交叉验证**；表观错误率偏乐观，别用训练数据自评。
- 先验不相等写入 $\pi_i$