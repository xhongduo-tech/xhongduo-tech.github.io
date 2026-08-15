---
title: 光滑样条
date: 2026-08-07
---

# 光滑样条

<div class="epigraph">
<p>在拟合数据与保持平滑之间，统计学需要的不是二选一，而是一条连续的滑杆。</p>
<footer>—— 卡尔 · 德布尔（Carl de Boor）与格蕾丝 · 沃巴（Grace Wahba）对样条光滑的刻画</footer>
</div>

<div class="article-byline">
<p>第二级 · 非参数统计 ｜ Wasserman, <em>All of Nonparametric Statistics</em> §5.5 ｜ 2026-08-07</p>
</div>

## 为什么从光滑样条开始

局部多项式回归是「在每个点的小邻域里做加权回归」，曲线由无数次局部拟合拼成。光滑样条走另一条路：**全局找一个函数 $g$，让它同时「贴住数据」和「保持光滑」**——前者用拟合误差衡量，后者用曲率积分衡量，两者由平滑参数 $\lambda$ 平衡。<span class="marginnote">区别一句话：核/局部多项式是「局部操作、全局拼接」，样条是「全局优化、自动拼接」——样条的光滑性与形状由单一目标函数一次性决定，而非逐点拟合。</span>样条方法在统计、数值分析、以及今天的神经场与插值学习里无处不在，值得单独一篇。

## 1 惩罚最小二乘

给定数据 $(X_i, Y_i)$，$i=1,\dots,n$，**光滑样条（smoothing spline）**定义为如下变分问题的解：

$$\hat g = \arg\min_{g} \sum_{i=1}^{n} \big(Y_i - g(X_i)\big)^2 + \lambda \int \big[g''(t)\big]^2 \,\mathrm{d}t$$

第一项是**拟合误差**，惩罚 $g$ 偏离观测点；第二项是**曲率惩罚**，惩罚 $g$ 剧烈弯曲；$\lambda \ge 0$ 是**平滑参数**。<span class="marginnote">$\lambda$ 的角色与带宽 $h$ 正好相反：$\lambda \to 0$ 时不罚曲率，$\hat g$ 会穿过每个点（插值）；$\lambda \to \infty$ 时曲率罚无限重，唯一解是直线。上一节说「$h$ 是光滑旋钮」，这里 $\lambda$ 就是同一个旋钮的镜像。</span>

关键结论（Schoenberg, 1964）：这个无穷维变分问题的解是**自然三次样条（natural cubic spline）**——分段三次多项式，节点取在 $X_i$ 上，且在 $[X_{(1)}, X_{(n)}]$ 之外为直线。也就是说，**最优解永远落在有限维样条空间里**，无穷维问题被自动降维。

## 2 自然三次样条与有效自由度

为什么最优解偏偏是样条？三类理由：

- **函数空间**：解在相邻节点间是三次多项式，节点处函数值、一阶、二阶导数连续，两端外推为线性（自然边界），于是 $g''$ 平方可积且边界条件自然。
- **可计算性**：记节点处取值 $\mathbf g = (g(X_1),\dots,g(X_n))$，曲率积分可写成二次型 $\int g''^2 = \mathbf g^\top \mathbf K \mathbf g$（$\mathbf K$ 为带号的三对角矩阵），于是目标退化为普通二次优化，解为

$$\hat{\mathbf g} = (\mathbf I + \lambda \mathbf K)^{-1}\mathbf Y$$

这是光滑样条最漂亮的闭合形式：估计 $\hat{\mathbf g}$ 是观测 $\mathbf Y$ 的一个线性变换。<span class="marginnote">$\hat{\mathbf g} = \mathbf S_\lambda \mathbf Y$ 中 $\mathbf S_\lambda = (\mathbf I+\lambda\mathbf K)^{-1}$ 是<strong>帽子矩阵（smoother matrix）</strong>。它的迹 $\mathrm{tr}(\mathbf S_\lambda)$ 是<strong>有效自由度</strong>：$\lambda$ 小则 $\mathrm{tr}(\mathbf S_\lambda) \approx n$（几乎一个点一个参数），$\lambda$ 大则 $\mathrm{tr}(\mathbf S_\lambda) \to 2$（逼近直线，两个参数）。样条与回归于是进了同一套自由度语言。</span>

## 3 平滑参数的选择

$\lambda$ 选得对不对，直接决定曲线过拟合还是欠拟合。标准做法是交叉验证。

**留一交叉验证（CV）**：$\mathrm{CV}(\lambda) = \frac1n \sum_i \big(Y_i - \hat g_{(-i)}(X_i)\big)^2$。对线性光滑器，留一误差有闭式捷径

$$\mathrm{CV}(\lambda) = \frac{1}{n}\sum_{i=1}^{n}\left( \frac{Y_i - \hat g(X_i)}{1 - S_{\lambda, ii}} \right)^2$$

其中 $S_{\lambda,ii}$ 是帽子矩阵第 $i$ 个对角元。<span class="marginnote">这条式子的妙处：$S_{\lambda,ii}$ 度量「第 $i$ 个观测对自己预测的影响」，除以 $1-S_{\lambda,ii}$ 精确模拟「去掉自己再预测」——不必真的重拟合 $n$ 次，一次拟合即可得全部留一误差。</span>

**广义交叉验证（GCV）**：用对角元平均 $\bar S = \mathrm{tr}(\mathbf S_\lambda)/n$ 替换逐元，得

$$\mathrm{GCV}(\lambda) = \frac{1}{n}\sum_{i=1}^{n}\left( \frac{Y_i - \hat g(X_i)}{1 - \bar S} \right)^2$$

GCV 计算更稳，且与「$C_p$ / AIC 型准则」在渐近上等价，因此是大多数软件（R 的 `smooth.spline`、Python 的 `splines`）的默认选择。<strong>选择 $\lambda$ 的本质，是在有效自由度曲线上挑一个「复杂度合理」的点</strong>——这一节与上节讲的带宽选择、有效自由度，构成了非参数建模的同一套调参哲学。

### 有效自由度的一个数值直觉

设 $n=50$ 个等距观测，拟合光滑样条。帽子矩阵 $\mathbf S_\lambda = (\mathbf I+\lambda\mathbf K)^{-1}$ 的特征分解显示：$\mathrm{tr}(\mathbf S_\lambda) = \sum_{j=1}^{50}\tfrac{1}{1+\lambda d_j}$，其中 $d_j$ 是 $\mathbf K$ 的特征值（对应不同频率的弯曲模式）。$d_j$ 小的低频模式几乎不被压缩（系数接近 1），$d_j$ 大的高频模式被强烈收缩（系数接近 0）——样条的本质是**对高频弯曲征税、对低频趋势放行**。<span class="marginnote">这与岭回归完全同构：$\lambda$ 越大，有效自由度越小，曲线越直。把「复杂度」听成「有效参数个数」而不是「节点个数」，是理解一切惩罚估计的关键认知升级。</span>

若调参后 $\mathrm{tr}(\mathbf S_\lambda) = 12$，就可把它当成「一个 12 参数的回归」去读：AIC、BIC、$R^2$ 校正版全部可以用有效自由度重写，模型选择因此统一。

### 样条与混合模型的联系

光滑样条还有一个漂亮的等价表述（Henderson 1950；Speed 1991）：**惩罚样条等价于一个线性混合模型，其中 $\lambda$ 对应方差比**。把 $\lambda\int g''^2$ 看成高斯先验 $\mathbf g \sim \mathcal{N}(0, \sigma^2\lambda^{-1}\mathbf K^{-})$，则 $\hat{\mathbf g}$ 恰是该先验下的后验众数——惩罚最小二乘 = 贝叶斯 MAP 估计。<span class="marginnote">这条等价让样条搭上了方差分量估计（REML）的顺风车：$\lambda$ 不再靠网格搜索，而可以当作方差比用最大似然一步估出；这页纸也预告了广义可加模型（GAM）与高斯过程回归的谱系。</span>

### 辨析｜节点与平滑是两回事

**「节点越多越复杂」是初学者的误区。** 在惩罚样条里，节点通常直接放在每个观测处（或按分位数铺在数据范围里），真正控制复杂度的是 $\lambda$；节点只负责决定「曲线的表达力上限」。用少量节点 + 大 $\lambda$ 与用全部节点 + 中 $\lambda$ 可能给出几乎相同的曲线——但前者是**回归样条（regression spline）**、节点是硬参数，后者是**光滑样条**、$\lambda$ 才是旋钮。读懂这个区别，才算读懂了样条家族。

### 算例：$\lambda$ 的两个极端

用 $n=50$、等距 $x$ 的数据演示两个极端行为。取 $\lambda = 0$：$\hat{\mathbf g} = \mathbf Y$，曲线穿过每一个观测点，$\mathrm{tr}(\mathbf S_0) = 50$——有效自由度等于样本量，曲线完全过拟合，虽有「拟合误差为零」的美名，预测新点的误差却最大。取 $\lambda = 10^6$：$\mathrm{tr}(\mathbf S_\lambda) \to 2$，曲线退化为最小二乘直线，自由度只剩截距与斜率——欠拟合。中间的某个 $\lambda$ 会让留一预测误差最小，这正是 CV/GCV 要找的位置。

### 从光滑样条到 GAM

把光滑样条装进广义线性模型的框架，就得到**广义可加模型（GAM）**：

$$g(\mu_i) = \beta_0 + f_1(X_{i1}) + \cdots + f_p(X_{ip})$$

每个协变量配一条样条 $f_j$，用各自的平滑参数 $\lambda_j$ 控制复杂度，再以迭代加权最小二乘（IRLS）求解。<span class="marginnote">GAM 把「一个协变量一条光滑曲线」变成工程惯例：预测房价时，面积、房龄、区位各配一条曲线，既保留了非线性，又让每个变量的效应可解释。它是「半参数哲学」在生产中的最大规模实现。</span>在 Python 的 `statsmodels`、R 的 `mgcv` 里，GAM 都是开箱即用的标准件——理解光滑样条，就等于拿到了打开它们的钥匙。

### 辨析｜样条 vs 核回归怎么选

**样条**：全局优化、闭合解、自由度可解析、自动处理边界；适合数据量中等、要报「平滑度/自由度」的场景。
**核/局部多项式**：局部直观、边界行为可控、易扩展到高维权重；适合大数据、需要逐点控制平滑的场景。
两者在合理调参下结果往往接近，不必纠结选谁；真正要紧的是**带宽/平滑参数选对**——参数选错，方法再精致也白搭。

### 变分视角：为什么是曲率惩罚

曲率惩罚 $\int g''^2\,\mathrm{d}t$ 不是唯一的平滑度量，却是最自然的一个：它是「曲线弯曲程度」的 $L_2$ 度量，只对弯折征税，直线（$g''=0$）完全免税。换成 $\int g'^2$（总斜率）会迫使曲线趋于水平，扭曲了水平方向的趋势；换成 $\int g'''^2$ 则过度惩罚尖点，曲线易抖。一阶导是「走势」、二阶导是「弯曲」、三阶导是「抖动」，惩罚二阶导恰好卡在「允许走势、抑制抖动」的中间位置。<span class="marginnote">这也是为什么平滑样条总选「三次样条 + 二阶导数惩罚」：三次多项式保证二阶导数存在且连续，惩罚 $g''$ 既不损失表达能力，又天然排斥高频抖动。选对惩罚阶数，比选对节点更本质。</span>

### 实用提示：报告自由度

拟合样条后，把有效自由度 $\mathrm{tr}(\mathbf S_\lambda)$ 写进结果，是评审最看重的规范：它让读者立刻知道曲线「相当于几个参数」，比只报 $\lambda$ 直观得多。对照示例：$\mathrm{tr}=4.2$ 意味着这条光滑曲线大致等价于一个 4 参数模型——远远低于「逐段插值」的 50 个自由度，过拟合风险一目了然。<span class="marginnote">画图时配合 rug plot（把观测位置画成底部小短线）能直观看出数据密度与曲线复杂度是否匹配；密度高的地方曲线细、自由度分布也更合理；数据稀疏区若出现异常的摆动，多半是 $\lambda$ 全局单一所致，可考虑逐段平滑参数。</span>

### 计算效率提示

$\mathbf K$ 是带状的稀疏矩阵，$(\mathbf I+\lambda\mathbf K)^{-1}\mathbf Y$ 可用带状 Cholesky 分解在 $O(n)$ 内求解，样条因此能轻松处理数十万观测；GCV 需要的 $\mathrm{tr}(\mathbf S_\lambda)$ 也可借特征分解一次算齐。工程上，样条「又快又稳又透明」的组合，让它成为大数据非参数光滑的默认首选之一。

## 4 公式解析：$\hat{\mathbf g} = (\mathbf I + \lambda\mathbf K)^{-1}\mathbf Y$

$$
\min_{\mathbf g}\; \|\mathbf Y - \mathbf g\|^2 + \lambda\, \mathbf g^\top \mathbf K \mathbf g
  \;\;\Longrightarrow\;\; \hat{\mathbf g} = (\mathbf I + \lambda \mathbf K)^{-1}\mathbf Y
$$

- **第一步，把目标写成矩阵形式**：拟合误差 $\sum (Y_i - g(X_i))^2 = \|\mathbf Y - \mathbf g\|^2$；曲率积分借助自然样条的结构写成二次型 $\lambda\,\mathbf g^\top \mathbf K \mathbf g$，$\mathbf K$ 是固定的半正定矩阵（与数据无关）。
- **第二步，对 $\mathbf g$ 求梯度**：$\frac{\partial}{\partial \mathbf g}\big[\|\mathbf Y-\mathbf g\|^2 + \lambda \mathbf g^\top\mathbf K\mathbf g\big] = -2(\mathbf Y - \mathbf g) + 2\lambda \mathbf K \mathbf g = 0$。
- **第三步，整理**：$(\mathbf I + \lambda \mathbf K)\mathbf g = \mathbf Y$，两端左乘逆矩阵即得 $\hat{\mathbf g} = (\mathbf I+\lambda\mathbf K)^{-1}\mathbf Y$。
- **第四步，为什么这是一条「收缩」而非「拟合」**：$\mathbf K$ 把相邻点的差加权，$\lambda\mathbf K$ 是对弯曲的线性惩罚；$(\mathbf I+\lambda\mathbf K)^{-1}$ 把 $\mathbf Y$ 朝「更平」的方向收缩。$\lambda=0$ 时 $\hat{\mathbf g}=\mathbf Y$（过拟合），$\lambda\to\infty$ 时 $\hat{\mathbf g}$ 逼近一条直线（欠拟合）。解的线性性意味着方差、偏差、自由度全部可解析计算——这是样条相对核方法最大的工程优势。

## 5 小结

- **光滑样条**通过惩罚最小二乘全局求解：拟合误差 $+ \lambda\int g''^2$，$\lambda$ 平衡「贴合」与「平滑」。
- 最优解是**自然三次样条**：分段三次、节点在 $X_i$、边界线性；无穷维问题自动落入有限维样条空间。
- 解闭合形式 $\hat{\mathbf g} = (\mathbf I+\lambda\mathbf K)^{-1}\mathbf Y$，是线性光滑器；**有效自由度** $\mathrm{tr}(\mathbf S_\lambda)$ 量化复杂度，从 $2$（直线）到 $n$（插值）。
- $\lambda$ 用留一交叉验证或 GCV 选择；报告**有效自由度** $\mathrm{tr}(\mathbf S_\lambda)$，让读者一眼看出曲线「相当于几个参数」。
- 样条的谱系很宽：惩罚样条 ≈ 线性混合模型（REML 一步估 $\lambda$）、≈ 贝叶斯 MAP、装进 GLM 即 GAM——理解 $\hat{\mathbf g}=(\mathbf I+\lambda\mathbf K)^{-1}\mathbf Y$，是一把钥匙开多扇门。

在下一节，我们将回到检验的效率问题：不同非参数检验谁更「省样本」？这正是**渐近相对效率（Pitman ARE）**要正面回答的。
