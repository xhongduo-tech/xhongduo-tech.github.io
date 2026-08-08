---
title: 线性判别分析
date: 2026-08-07
---

# 线性判别分析

<div class="epigraph">
<p>把事物区分开来的本领，是一切智慧的开始。</p>
<footer>—— 仿柏拉图（Plato）《理想国》中「划分」的思想，通行说法</footer>
</div>

<div class="article-byline">
<p>第四级 · 机器学习 ｜ 周志华《机器学习》第3章 §3.4 ｜ 2026-08-07</p>
</div>

## 为什么要有第二种分类思路

前两节讲的是「给线性输出加概率外衣」（对数几率回归），但那只是**一种**构造分类器的思路。本节换一条完全不同的路线：**不先假设一个概率模型，而是直接在几何上找一个「能把两类分开」的方向。**

这个思路来自罗纳德 · 费希尔（Ronald Fisher），他在 1936 年提出：把样本点投影到一条直线上，让**同类样本在投影后尽量聚拢、不同类样本尽量分开**，再用这条线上的一个阈值做分类。这就是**线性判别分析（Linear Discriminant Analysis，LDA）**。<span class="marginnote">费希尔是现代统计学的奠基人之一，也是「方差分析」「Fisher 信息」「Fisher 检验」的提出者。LDA 是他研究鸢尾花数据集时的经典作品——这个 150 个样本、4 个特征的鸢尾花数据，至今仍是机器学习入门最常见的基准数据集。</span>

**重点：LDA 是「监督式降维」与「分类」的合体。** 它先学一个投影方向 $\mathbf{w}$，把高维数据压到一条直线（或多条直线组成的低维子空间）上，投影之后再做分类。因此 LDA 既可以被当成分类器，也可以被当成**降维方法**——这两重身份在后面的第 10 章会再次相遇。

与我们已学的两类线性模型对比，LDA 的独特性一望即知：线性回归的最小二乘与对数几率回归的极大似然，都在「拟合一个输出的映射」；LDA 则完全跳过映射，直接问一个几何问题——**往哪个方向看，两类最容易被分开？** 这个「几何优先」的姿态，让它天然适合可视化：把高维样本投影到一两条直线上，人眼就能直接看出类别是否可分。

## 1 投影后的两类距离怎么度量设给定数据集 $D = \{(\mathbf{x}_i, y_i)\}$，$y_i \in \{0, 1\}$。记第 $i$ 类的样本集合为 $\mathbf{X}_i$、均值向量为 $\boldsymbol{\mu}_i$、协方差矩阵为 $\boldsymbol{\Sigma}_i$。把全部样本投影到方向 $\mathbf{w}$ 上：

- 两类**中心**投影后分别落在 $\mathbf{w}^{\mathrm{T}}\boldsymbol{\mu}_0$ 与 $\mathbf{w}^{\mathrm{T}}\boldsymbol{\mu}_1$；
- 两类投影后的**分散程度**（协方差）分别是 $\mathbf{w}^{\mathrm{T}}\boldsymbol{\Sigma}_0\mathbf{w}$ 与 $\mathbf{w}^{\mathrm{T}}\boldsymbol{\Sigma}_1\mathbf{w}$。

费希尔的思想是：**「类间距离尽可能大」意味着两个投影中心 $\mathbf{w}^{\mathrm{T}}\boldsymbol{\mu}_0$ 与 $\mathbf{w}^{\mathrm{T}}\boldsymbol{\mu}_1$ 离得越远越好；「类内距离尽可能小」意味着投影后的协方差越小越好。** 注意这里必须同时照顾两者：只追求中心距离，则投影方向会沿着「两类整体最分散」的方向走，类内也可能同样分散，两类照样重叠。两者合在一起，就得到一个比值目标：

$$J = \frac{\left\| \mathbf{w}^{\mathrm{T}}\boldsymbol{\mu}_0 - \mathbf{w}^{\mathrm{T}}\boldsymbol{\mu}_1 \right\|_2^2}{\mathbf{w}^{\mathrm{T}}\boldsymbol{\Sigma}_0\mathbf{w} + \mathbf{w}^{\mathrm{T}}\boldsymbol{\Sigma}_1\mathbf{w}}$$

**核心概念：广义瑞利商（generalized Rayleigh quotient）**：目标 $J$ 是「类间距离的平方」除以「类内散度之和」，是一个**比值**。分子要最大，分母要最小——LDA 的全部算法，就是最大化这个比值。<span class="marginnote">瑞利商（Rayleigh quotient）本是线性代数里 $\frac{\mathbf{x}^{\mathrm{T}}A\mathbf{x}}{\mathbf{x}^{\mathrm{T}}\mathbf{x}}$ 的经典对象，其最大值等于 $A$ 的最大特征值。「广义」指分母从 $\mathbf{x}^{\mathrm{T}}\mathbf{x}$ 换成 $\mathbf{x}^{\mathrm{T}}B\mathbf{x}$——它把「找最大方向」的问题转成了「求广义特征向量」的问题，这就是矩阵视角的入口。</span>

**重点：比值形式的目标有一个隐蔽但极有用的性质——它关于 $\mathbf{w}$ 的缩放不变。** 把 $\mathbf{w}$ 放大 10 倍，分子分母同时放大 100 倍，比值 $J$ 纹丝不动。这意味着「方向」才重要，「长度」无所谓——我们可以自由地给 $\mathbf{w}$ 加一个归一化约束来方便求解。

## 2 公式解析：从瑞利商到特征向量把分子用散度矩阵写开。定义**类内散度矩阵（within-class scatter matrix）**与**类间散度矩阵（between-class scatter matrix）**：

$$\mathbf{S}_w = \boldsymbol{\Sigma}_0 + \boldsymbol{\Sigma}_1 = \sum_{\mathbf{x} \in \mathbf{X}_0} (\mathbf{x} - \boldsymbol{\mu}_0)(\mathbf{x} - \boldsymbol{\mu}_0)^{\mathrm{T}} + \sum_{\mathbf{x} \in \mathbf{X}_1} (\mathbf{x} - \boldsymbol{\mu}_1)(\mathbf{x} - \boldsymbol{\mu}_1)^{\mathrm{T}}$$

$$\mathbf{S}_b = (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^{\mathrm{T}}$$

于是目标函数压缩成紧凑的矩阵形式：

$$J = \frac{\mathbf{w}^{\mathrm{T}}\mathbf{S}_b \mathbf{w}}{\mathbf{w}^{\mathrm{T}}\mathbf{S}_w \mathbf{w}}$$

求解过程是拉格朗日乘子法的经典应用，分四步：

- **第一步，加约束**：因为 $J$ 缩放不变，令 $\mathbf{w}^{\mathrm{T}}\mathbf{S}_w\mathbf{w} = 1$，最大化分子 $\mathbf{w}^{\mathrm{T}}\mathbf{S}_b\mathbf{w}$。
- **第二步，写拉格朗日函数**：$L = \mathbf{w}^{\mathrm{T}}\mathbf{S}_b\mathbf{w} - \lambda(\mathbf{w}^{\mathrm{T}}\mathbf{S}_w\mathbf{w} - 1)$，对 $\mathbf{w}$ 求导并令其为零，得广义特征方程 $\mathbf{S}_b\mathbf{w} = \lambda \mathbf{S}_w\mathbf{w}$。
- **第三步，化简**：因为 $\mathbf{S}_b\mathbf{w} = (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^{\mathrm{T}}\mathbf{w}$ 的方向总是沿着 $\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1$（一个数乘上这个向量），所以直接得到解
$$\mathbf{w} = \mathbf{S}_w^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$$
- **第四步，读出几何**：最优投影方向 = **类内散度的逆 × 两类中心的差**。$\mathbf{S}_w^{-1}$ 起到「白化」作用——先压缩类内椭球的形状，再沿中心连线方向取投影，这样投影后两类最容易被阈值切开。

**辨析｜易错点：** 一个高频误会是「LDA 就是 PCA」。两者确实都做投影，但**目标完全不同**：PCA（主成分分析，第 10 章）找的是「数据方差最大」的方向，**完全不看类别标签**；LDA 找的是「类间/类内散度最大」的方向，**必须使用标签**。所以 PCA 是无监督降维，LDA 是监督降维——一个「哪里分散往哪投」，一个「哪里可分往哪投」。<span class="marginnote">费希尔 1936 年的论文标题是《The Use of Multiple Measurements in Taxonomic Problems》，LDA 的诞生比「机器学习」这个词早了几十年。它的思想后来启发了一整族「投影后判类」的方法，包括线性判别式、二次判别式（QDA），以及核化的变体。</span>

## 3 多分类情形：求一组投影方向当类别数 $N > 2$ 时，一条投影线不够用，需要投影到一个**低维子空间**。此时定义**全局散度矩阵** $\mathbf{S}_t = \mathbf{S}_b + \mathbf{S}_w$，目标变成最大化一个矩阵版本的瑞利商：

$$\max_{\mathbf{W}} \frac{\operatorname{tr}\!\left(\mathbf{W}^{\mathrm{T}}\mathbf{S}_b\mathbf{W}\right)}{\operatorname{tr}\!\left(\mathbf{W}^{\mathrm{T}}\mathbf{S}_w\mathbf{W}\right)}$$

其中 $\mathbf{W}$ 的每一列是一个投影方向。解是：**对矩阵 $\mathbf{S}_w^{-1}\mathbf{S}_b$ 做特征值分解，取前 $d'$ 个最大特征值对应的特征向量组成 $\mathbf{W}$**。理论上的极值 $d'$ 最多为 $N-1$（因为 $\mathbf{S}_b$ 的秩至多是 $N-1$），所以多分类 LDA 最多把数据降到 $N-1$ 维。<span class="marginnote">「$N-1$」这个上界是 LDA 与 PCA 的一个关键差异：PCA 可以降到任意维，LDA 受限于类别数。这也解释了为什么在类别数少的任务里，LDA 降维后信息损失小；而类别很多的任务里，LDA 的方向数就不够用了。</span>

投影完成后，多分类 LDA 的判类方式是把测试样本投影到子空间，再与各类中心比较距离（最近邻思想）——这为第 9 章《聚类》与第 10 章《k 近邻学习》埋下了伏笔。

## 4 用代码看 LDA 的投影LDA 的实现只需一行，但它的输出值得仔细读：

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis().fit(X, y)
w = lda.coef_[0]   # 等价于 S_w^{-1}(μ0 - μ1) 方向上的权重向量
```

`w` 就是 $\mathbf{S}_w^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$ 方向上的权重向量。把每个样本点「砸」到这根线上，两类就变成一维直线上两个分离的区间——**分类从二维问题降成了一维问题，阈值切分从此变得容易**。这就是 LDA 作为降维工具最直观的演示。<span class="marginnote">scikit-learn 的 `LinearDiscriminantAnalysis` 默认用的是精确求解（直接解广义特征值问题），当样本量远大于特征数时稳定高效；遇到高维小样本时，则可用 `shrinkage` 参数做收缩估计，这与第 11 章讲 L1/L2 正则化的「用正则换取稳定」是同一种思想。</span>

## 5 LDA 在全书中的坐标LDA 看似是第 3 章的一个小节，其实它串起了半本书：

**与线性回归 / 对数几率回归**：三者都是「线性投影 + 阈值」，区别只在目标函数——最小二乘、负对数似然、瑞利商；
**与 PCA / 度量学习**：第 10 章里 PCA 是无监督投影、LDA 是监督投影，两者并列为「线性降维」的两大支柱；
**与 SVM**：SVM 追求「最大间隔」，LDA 追求「最大类间/类内比」，是「找最优超平面」的两种不同判据；
**与贝叶斯分类器**：第 7 章会看到，LDA 在「两类协方差相等 + 数据服从高斯分布」的假设下，恰好等价于一个贝叶斯最优分类器——**统计派与几何派在这里殊途同归**。<span class="marginnote">这个「殊途同归」值得单独品味：从几何目标（瑞利商）出发求得的 $\mathbf{w}$，与从概率假设（高斯、同协方差）出发做贝叶斯决策得到的边界，在数学上是同一个对象。它提醒我们，机器学习的不同「学派」往往只是同一座山的几条登山路。</span>

**重点：LDA 的深层启示是「降维与分类可以是一件事」。** 与其在高维空间里硬找一条边界，不如先找一个「让类别分得最开」的低维子空间，把问题变简单再做分类。这个「先投影、后决策」的思想，会一路贯穿到深度学习里「表征学习」的核心主张——**好的特征是让类别天然分开的特征**。

## 6 小结- **LDA 的目标**：最大化广义瑞利商 $J = \frac{\mathbf{w}^{\mathrm{T}}\mathbf{S}_b\mathbf{w}}{\mathbf{w}^{\mathrm{T}}\mathbf{S}_w\mathbf{w}}$，即类间散度最大、类内散度最小。
- **两个散度矩阵**：$\mathbf{S}_w$（类内，各样本相对类中心的散布）与 $\mathbf{S}_b = (\boldsymbol{\mu}_0-\boldsymbol{\mu}_1)(\boldsymbol{\mu}_0-\boldsymbol{\mu}_1)^{\mathrm{T}}$（类间，两中心的外积）。
- **二分类解**：$\mathbf{w} = \mathbf{S}_w^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$，来自拉格朗日乘子法解广义特征方程。
- **多分类**：对 $\mathbf{S}_w^{-1}\mathbf{S}_b$ 做特征分解，取前 $d' \le N-1$ 个特征向量构成投影矩阵 $\mathbf{W}$。
- **易错**：LDA ≠ PCA——LDA 用标签、找「可分」方向；PCA 不用标签、找「分散」方向。
- **地位**：监督降维的标杆，也是分类器；与贝叶斯分类器在高斯同协方差假设下等价。

最后补一句实践的提醒：LDA 假设各类的协方差矩阵大致相同，当这个假设明显不成立（例如一类分布极扁、另一类接近正圆）时，投影效果会打折，此时可考虑二次判别分析（QDA）或直接换用非线性的降维方法（第 10 章）。**它是「假设美好、边界清晰」的线性时代代表，理解它的代价，也理解它的适用边界。**

## 本节路线图

- **第1节**：投影后的两类距离怎么度量
- **第2节**：公式解析：从瑞利商到特征向量
- **第3节**：多分类情形：求一组投影方向
- **第4节**：用代码看 LDA 的投影
- **第5节**：LDA 在全书中的坐标
- **小结**：要点复盘与下一课衔接

## 复习自查清单

读完后，试着不翻书复述以下各点：

- [ ] **LDA 的目标**：最大化广义瑞利商 $J = \frac{\mathbf{w}^{\mathrm{T}}\mathbf{S}_b\mathbf{w}}{\mathbf{w}^{\mathrm{T}}\mathbf{S}_w\mathbf{w}}$，即类间散度最大、类内散度最小。
- [ ] **两个散度矩阵**：$\mathbf{S}_w$（类内，各样本相对类中心的散布）与 $\mathbf{S}_b = (\boldsymbol{\mu}_0-\boldsymbol{\mu}_1)(\boldsymbol{\mu}_0-\boldsymbol{\mu}_1)^{\mathrm{T}}$（类间，两中心的外积）。
- [ ] **二分类解**：$\mathbf{w} = \mathbf{S}_w^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$，来自拉格朗日乘子法解广义特征方程。
- [ ] **多分类**：对 $\mathbf{S}_w^{-1}\mathbf{S}_b$ 做特征分解，取前 $d' \le N-1$ 个特征向量构成投影矩阵 $\mathbf{W}$。
- [ ] **易错**：LDA ≠ PCA——LDA 用标签、找「可分」方向；PCA 不用标签、找「分散」方向。
- [ ] **地位**：监督降维的标杆，也是分类器；与贝叶斯分类器在高斯同协方差假设下等价。
- [ ] **第1节**：投影后的两类距离怎么度量
- [ ] **第2节**：公式解析：从瑞利商到特征向量
- [ ] **第3节**：多分类情形：求一组投影方向
- [ ] **第4节**：用代码看 LDA 的投影
- [ ] **第5节**：LDA 在全书中的坐标
- [ ] **小结**：要点复盘与下一课衔接

在下一节，我们把目光从「两类」移向「多类」：当类别数超过两个时，怎么把一个多分类问题拆成多个二分类问题来解？——这就是**多分类学习**。
