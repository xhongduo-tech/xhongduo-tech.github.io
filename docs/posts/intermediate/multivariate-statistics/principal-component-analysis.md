---
title: 主成分分析（PCA）
date: 2026-08-07
---

# 主成分分析（PCA）

<div class="epigraph">
<p>如果你只能用一个数描述一堆高维数据，你会选哪个？——那个携带最大方差的方向。</p>
<footer>—— 哈罗德·霍特林（Harold Hotelling）</footer>
</div>

<div class="article-byline">
<p>第二级 · 多元统计分析 ｜ Anderson《An Introduction to Multivariate Statistical Analysis》Ch.11 · Johnson & Wichern Ch.8 ｜ 2026-08-07</p>
</div>

## 为什么从方差开始降维

多元数据的麻烦在于 $p$ 大：30 个财务指标、几百个基因表达量、几千个词频特征。变量多了，人看不过来、算法也容易过拟合。**主成分分析（principal component analysis, PCA）**回答一个朴素的问题：能不能造出少数几个「新变量」，既尽量保留原始数据的差异，又彼此不相关？霍特林给出的答案同样朴素——**差异就是方差，保留最多的方差就是保留最多的信息**。<span class="marginnote">「方差 = 信息」是个大胆但有效的约定：PCA 只在乎数据散开的程度，不在乎均值高低。换一个目标函数（如保留距离结构），就得到后面的多维标度（MDS）；目标函数不同，方法就不同。</span>

## 1 总体主成分：从最大化方差出发

设 $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，要找第一个新变量 $Y_1 = \mathbf{e}_1'\mathbf{X}$（$\mathbf{e}_1$ 是单位向量），使其方差最大：

$$
\operatorname{Var}(Y_1) = \mathbf{e}_1' \boldsymbol{\Sigma} \mathbf{e}_1 \ \to \ \max, \qquad \text{s.t. } \mathbf{e}_1'\mathbf{e}_1 = 1
$$

用拉格朗日乘子解：$\mathcal{L} = \mathbf{e}_1'\boldsymbol{\Sigma}\mathbf{e}_1 - \lambda(\mathbf{e}_1'\mathbf{e}_1 - 1)$，对 $\mathbf{e}_1$ 求导得

$$
\boldsymbol{\Sigma}\mathbf{e}_1 = \lambda \mathbf{e}_1
$$

**这是一个特征方程**。最大的特征值 $\lambda_1$ 就是 $\operatorname{Var}(Y_1)$，对应特征向量 $\mathbf{e}_1$ 就是第一主成分方向。继续下去，第 $k$ 个主成分 $Y_k = \mathbf{e}_k'\mathbf{X}$ 在与前 $k-1$ 个不相关的约束下方差最大，$\boldsymbol{\Sigma}\mathbf{e}_k = \lambda_k\mathbf{e}_k$——**主成分就是协方差矩阵按特征值降序排列的特征向量方向**。<span class="marginnote">约束「不相关」落到几何上就是特征向量彼此正交（$\boldsymbol{\Sigma}$ 对称保证 $\mathbf{e}_i \perp \mathbf{e}_j$）。所以主成分是一组互相垂直的新坐标轴，就像把数据云的椭圆「转正」后的长短轴。</span>

### 一张图看懂 PCA 的几何

![PCA 几何示意：椭圆数据云的主轴即主成分方向](/images/multivariate-statistics/principal-component-analysis-1.svg)

椭圆的长轴方向是第一主成分 $\mathbf{e}_1$，短轴是 $\mathbf{e}_2$；原始坐标 $(X_1, X_2)$ 旋转后得到新坐标 $(Y_1, Y_2)$，$Y_1$ 携带大部分方差。

## 2 样本主成分：把 Σ 换成 S

总体未知，用样本协方差矩阵 $\mathbf{S}$（或相关矩阵 $\mathbf{R}$）做谱分解：

$$
\mathbf{S} = \hat{\mathbf{\Gamma}} \hat{\boldsymbol{\Lambda}} \hat{\boldsymbol{\Gamma}}', \qquad
\hat{\boldsymbol{\Lambda}} = \operatorname{diag}(\hat{\lambda}_1 \geq \cdots \geq \hat{\lambda}_p \geq 0)
$$

第 $k$ 个样本主成分是 $y_k = \hat{\mathbf{e}}_k'\mathbf{x}$，样本方差 $\hat{\lambda}_k$。**样本主成分就是「让数据云转正、轴按长度排序」**：$\hat{\lambda}_k$ 是第 $k$ 个主轴方向的散布量。<span class="marginnote">数值细节：用 SVD 对中心化数据矩阵 $\tilde{\mathbf{X}}$ 做分解比直接求 $\mathbf{S}$ 的特征分解更稳——$\tilde{\mathbf{X}} = \mathbf{U}\mathbf{D}\mathbf{V}'$ 的右奇异向量 $\mathbf{V}$ 就是特征向量，奇异值平方除以 $n-1$ 就是特征值。这是现代软件内部的默认做法。</span>

**标准化选择**：若变量量纲悬殊（如一个变量是身高、另一个是体重×10³），直接对 $\mathbf{S}$ 做主成分等于暗地里给大方差变量加权重。此时应改用相关矩阵 $\mathbf{R}$（等价于先把每个变量标准化），结果与测量单位无关。<span class="marginnote">对 $\mathbf{R}$ 做主成分时总方差固定为 $p$，特征值直接解读为「该主成分解释的比例」；对 $\mathbf{S}$ 做则特征值带量纲，比例要看总方差 $\operatorname{tr}\mathbf{S}$。绝大多数应用默认对 $\mathbf{R}$ 做。</span>

### 主成分的载荷与得分

主成分方向 $\hat{\mathbf{e}}_k$ 的坐标称为**载荷（loadings）**：第 $j$ 个坐标的绝对值越大，第 $j$ 个原始变量对第 $k$ 个主成分的贡献越大。把每个样本投影到第 $k$ 个主成分轴上得到的数叫**得分（score）**。载荷告诉你「新轴怎么由旧变量构成」，得分告诉你「每个样本在新坐标里的位置」——前者用于解释，后者用于画图、聚类、回归。

## 3 选几个主成分：碎石图与累计贡献率

第 $k$ 个主成分的**贡献率**是 $\hat{\lambda}_k / \sum_{j=1}^p \hat{\lambda}_j$，前 $m$ 个的**累计贡献率**是 $\sum_{k=1}^m \hat{\lambda}_k / \sum_{j=1}^p \hat{\lambda}_j$。选 $m$ 的经验法则：

**累计贡献率达到 70%–80%**：多数教科书建议的标准，但只是一个起点。
**Kaiser 准则**：对 $\mathbf{R}$ 做主成分时，只保留特征值 $\geq 1$ 的成分——平均每个主成分至少要顶一个原始变量的方差。
**碎石图（scree plot）**：画 $\hat{\lambda}_k$ 随 $k$ 的折线，找「肘部」——特征值从陡降转为平缓的拐点，拐点之前的个数即保留数。<span class="marginnote">没有「唯一正确」的个数。交叉验证与置换检验（如 Horn 平行分析）能给出更客观的答案，但对探索性分析，碎石图的肘部往往就够了。记住 PCA 是描述工具，不是检验工具。</span>

## 4 辨析｜易错点：PCA 的边界与常见误用

PCA 如此常用，反而最容易被误用。四条边界必须记住：

**第一，方差不等于一切**。PCA 选「方差最大的方向」，但任务需要的可能是「最能分开两类的方向」——判别分析里 Fisher 判别选的是「组间方差与组内方差之比最大」的方向，与 PCA 不同。**降维目标与任务目标错配时，PCA 会丢信息**。<span class="marginnote">经典的例子：两组数据沿长轴方向方差很大但两组沿长轴几乎重叠，PCA 保留长轴却丢掉了两组唯一可分的短轴方向——此时先用判别分析或带标签的降维才是对的。</span>

**第二，尺度的选择决定一切**。对 $\mathbf{S}$ 做还是对 $\mathbf{R}$ 做，结果可以完全不同。单位选错了，主成分就「背叛」了你的意图。**先想清楚变量是否可比较，再决定是否标准化**。

**第三，PCA 是描述不是检验**。它不回答「降维到 $m$ 维是否显著」，没有 p 值；载荷的解释依赖样本，换一批数据主轴可能旋转。**把 PCA 当结论而非探索，是研究里最常见的过度解读**。

**第四，主成分未必可解释**。载荷通常是所有原始变量的混叠，极少恰好对应「体重因子」「压力因子」这类干净概念。想要可解释的潜在变量，得等下一节——因子分析允许旋转、允许误差项，模型不同。

| 对比项 | 主成分分析（PCA） | 因子分析（FA） |
| --- | --- | --- |
| 目标 | 保留最大方差 | 解释变量间的相关结构 |
| 新变量 | 主成分 = 原始变量的线性组合 | 因子 = 潜在不可测原因 |
| 误差项 | 无（方差全部被解释） | 有（特殊方差 $\Psi$） |
| 旋转 | 一般不旋转 | 常旋转以增强可解释性 |
| 用途 | 降维、压缩、可视化 | 构造量表、发现潜在结构 |

这张表是「下一节预告 + 本节辨析」：**PCA 与 FA 经常被混为一谈，但一个是几何旋转、一个是因果模型**。

## 5 实战流程：从原始数据到得分图

把前面所有零件装成一条流水线，PCA 的标准流程是：

1. **数据准备**：中心化（必要时标准化），检查缺失值与量纲。
2. **分解**：对 $\mathbf{S}$ 或 $\mathbf{R}$ 做特征分解，或对中心化数据做 SVD，得到特征值 $\hat{\lambda}_1 \geq \cdots \geq \hat{\lambda}_p$ 与特征向量。
3. **定个数**：画碎石图找肘部，参考累计贡献率与 Kaiser 准则，定 $m$。
4. **解释载荷**：看前 $m$ 个特征向量的坐标，给每个主成分起个名字（如「规模因子」「风险因子」）。
5. **算得分并可视化**：把每个样本投影到前两个主成分上，画二维散点——相似的样本在得分图上聚在一起。

第 5 步是 PCA 在探索性分析里最出彩的地方：**$p$ 维数据云被压到一张能「看」的平面上**。得分图上点与点的距离近似反映原始空间的距离（在前两个主成分张成的子空间内），于是聚类结构、离群点、分组边界一眼可见。<span class="marginnote">把载荷和得分画在同一张图上就是<strong>双标图（biplot）</strong>：箭头的长短与方向告诉你每个原始变量「往哪个主成分方向贡献」，点告诉你样本落在哪——一张图同时讲变量和样本两个故事。</span>

还有一个常被忽略的用途：**PCA 作预处理器**。把数据降到 $m$ 维再喂给回归或聚类，可以显著抑制高维带来的过拟合与共线性；PCA 回归（PCR）就是「先降维再回归」的正式名称。当然，这牺牲了解释性——降维后的变量是混叠的，回归系数不再对应任何原始变量。

## 6 公式解析：为什么最大化方差给出特征向量

把「最大化 $\mathbf{e}'\boldsymbol{\Sigma}\mathbf{e}$」一步步翻译成特征方程：

- **第一步，拉格朗日函数**：带单位范数约束的优化写成 $\mathbf{e}'\boldsymbol{\Sigma}\mathbf{e} - \lambda(\mathbf{e}'\mathbf{e}-1)$，$\lambda$ 是乘子。
- **第二步，对 $\mathbf{e}$ 求梯度并置零**：$\frac{\partial}{\partial\mathbf{e}}\left[\mathbf{e}'\boldsymbol{\Sigma}\mathbf{e} - \lambda(\mathbf{e}'\mathbf{e}-1)\right] = 2\boldsymbol{\Sigma}\mathbf{e} - 2\lambda\mathbf{e} = \mathbf{0}$，即 $\boldsymbol{\Sigma}\mathbf{e} = \lambda\mathbf{e}$。
- **第三步，读特征值即方差**：把特征方程左乘 $\mathbf{e}'$，得 $\mathbf{e}'\boldsymbol{\Sigma}\mathbf{e} = \lambda\mathbf{e}'\mathbf{e} = \lambda$。**拉格朗日乘子恰好等于被最大化的方差**——这就是为什么特征值 $\lambda_k$ 可以直接解读为第 $k$ 个主成分的方差。
- **第四步，递推**：第二个主成分加约束 $\mathbf{e}_2 \perp \mathbf{e}_1$，解出来是第二大特征值的特征向量；依次类推。

**核心结论：PCA 的最优化问题与「协方差矩阵的特征分解」是同一件事**。这也是为什么第一节花了那么多篇幅讨论 $\mathbf{S}$：它的特征向量就是数据云的主轴，它的特征值就是轴上的散布量。

## 7 小结

- **主成分**是协方差矩阵（或相关矩阵）特征向量的方向；$Y_k = \mathbf{e}_k'\mathbf{X}$，方差 $\lambda_k$，彼此不相关。
- **样本 PCA** 用 $\mathbf{S}$ 或 $\mathbf{R}$ 的谱分解；$\mathbf{S}$ 保留原始量纲，$\mathbf{R}$ 消除量纲影响，默认多用 $\mathbf{R}$。
- **载荷**解释新轴如何由旧变量构成，**得分**给每个样本在新坐标里的位置。
- 选主成分个数看**累计贡献率、Kaiser 准则（$\lambda\geq 1$