---
title: 梯度、雅可比矩阵与黑塞矩阵
date: 2026-08-07
---

# 梯度、雅可比矩阵与黑塞矩阵

<div class="epigraph">
<p>上坡的路和下坡的路是同一条路。</p>
<footer>—— 赫拉克利特（Heraclitus）</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§4.3 ｜ 2026-08-07</p>
</div>

## 为什么从梯度开始

深度学习的所有训练，最终都归结为同一个动作：**在参数空间里反复寻找让损失函数更小的点**。而「朝哪个方向走一步，损失下降得最快」这个问题的答案，就是**梯度**。它是「学习」二字最直接的数学化身——没有梯度，就没有反向传播，也就没有后面的一切。

但单个函数的梯度只是起点。真实模型往往是**多输入多输出**的：一个网络层把 $n$ 维输入映射到 $m$ 维输出，我们想同时知道「输出的每一个分量对输入每一个分量的敏感度」；损失函数还可能对参数**弯曲**，下降时曲率是正是负、是陡是缓，都会影响优化步长。这两个问题分别由**雅可比矩阵**与**黑塞矩阵**回答。本节把三件工具放在一起讲，因为它们是三个层次的递进：单输出一阶（梯度）$\to$ 多输出一阶（雅可比）$\to$ 二阶（黑塞）。<span class="marginnote">这条递进链在后面几乎处处复用：反向传播的本质就是「用链式法则把雅可比矩阵逐层乘起来」；黑塞矩阵是牛顿法、鞍点分析与二阶优化的入场券。本节把地基打牢，后续《反向传播》《优化问题》都会回来调用它。</span>

## 1 方向导数与梯度

**偏导数（partial derivative）**：对多元函数 $f: \mathbb{R}^n \to \mathbb{R}$，在点 $\boldsymbol{x}_0$ 处只让第 $i$ 个分量变动、其余固定，得到的导数 $\frac{\partial f}{\partial x_i}(\boldsymbol{x}_0)$ 就是沿坐标轴方向的瞬时变化率。

把 $n$ 个偏导数竖着排成一列，就得到**梯度（gradient）**：

$$
\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

梯度是一个**向量**，它的每个分量告诉你「沿这根坐标轴，函数上升多快」。把这些分量合成一个方向，就得到整座「函数山」在 $\boldsymbol{x}_0$ 处最陡峭的朝向。

**方向导数（directional derivative）**回答更一般的问题：沿任意单位方向 $\boldsymbol{u}$（$\|\boldsymbol{u}\|=1$）走，函数的变化率是多少？答案是内积

$$
\frac{\partial f}{\partial \boldsymbol{u}} = \boldsymbol{u}^{\top} \nabla_{\boldsymbol{x}} f = \sum_i u_i \frac{\partial f}{\partial x_i}
$$

由柯西-施瓦茨不等式 $|\boldsymbol{u}^{\top}\nabla f| \le \|\boldsymbol{u}\|\,\|\nabla f\| = \|\nabla f\|$，等号恰在 $\boldsymbol{u}$ 与梯度同向时取得。于是有本节最重要的一条结论：

**梯度方向是函数上升最快的方向，负梯度方向是函数下降最快的方向**。

这就是梯度下降的全部几何直觉：站在山上，闭眼感受脚下哪个方向最陡、朝下迈步。它不需要知道整座山的形状，只需要当前位置的局部信息。<span class="marginnote">「最陡」是<strong>局部</strong>意义下的最陡：只保证朝这个方向迈出一小步下降最多。真正的下山路要一步步重算方向——这正是随机梯度下降每轮迭代都要重新算梯度的原因。想回顾多元函数的极限与连续，可回看第二级《高等数学》偏导与方向导数一节。</span>

**辨析｜易错点：** 初学者常把「梯度」读成「往下走的方向」。准确说法是**梯度指向上坡**，**负梯度**才指下坡。代码里写 `w -= lr * grad`，减号不是装饰，而是把梯度翻成下降方向的那一步。这个符号一旦写反，训练立刻发散——「符号写反 + 训练发散」是调参新手最经典的故障之一。

## 2 雅可比矩阵：多输出的一阶全景

当函数把向量映成向量时，梯度就不够用了。考虑 $\boldsymbol{f}: \mathbb{R}^n \to \mathbb{R}^m$，输出是 $m$ 个分量 $f_1(\boldsymbol{x}), \dots, f_m(\boldsymbol{x})$。把每个输出的偏导**按行**排起来，就得到**雅可比矩阵（Jacobian matrix）**：

$$
\boldsymbol{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

它的第 $i$ 行就是第 $i$ 个输出分量的梯度（转置成行向量）。当 $m = 1$ 时，雅可比退化为梯度的转置——**梯度是雅可比的单输出特例**。

雅可比矩阵刻画了 $\boldsymbol{f}$ 在一点处的**局部线性化**：对足够小的扰动 $\boldsymbol{\epsilon}$，

$$
\boldsymbol{f}(\boldsymbol{x}_0 + \boldsymbol{\epsilon}) \approx \boldsymbol{f}(\boldsymbol{x}_0) + \boldsymbol{J}(\boldsymbol{x}_0)\,\boldsymbol{\epsilon}
$$

这是一个从「增量 $\boldsymbol{\epsilon}$」到「输出增量」的线性映射：输入空间里的一个方向向量，被矩阵 $\boldsymbol{J}$ 旋转并伸缩成输出空间里的一个方向向量。<span class="marginnote">这个「矩阵即线性化」的视角是微分学的核心心法，与第一级《线性代数》中「矩阵 = 线性变换」完全同构：求导不过是逐点把非线性函数替换成它的最佳线性近似。</span>

链式法则在雅可比语言下变得格外优雅：若有 $\boldsymbol{g}: \mathbb{R}^p \to \mathbb{R}^n$、$\boldsymbol{f}: \mathbb{R}^n \to \mathbb{R}^m$，则复合 $\boldsymbol{f} \circ \boldsymbol{g}$ 的雅可比就是**两个雅可比的矩阵乘积**

$$
\boldsymbol{J}_{\boldsymbol{f} \circ \boldsymbol{g}} = \boldsymbol{J}_{\boldsymbol{f}} \cdot \boldsymbol{J}_{\boldsymbol{g}}
$$

反向传播就是这条规则沿着网络从输出层往输入层**一遍遍左乘**的过程：每穿过一层，就把该层雅可比乘到当前累积的结果前面。这就是为什么说「反向传播 = 链式法则 + 高效计算图求值」。

## 3 黑塞矩阵：二阶曲率

梯度只告诉你「往哪走最快」，不告诉你「路有多弯」。二阶信息由**黑塞矩阵（Hessian matrix）**承载。对 $f: \mathbb{R}^n \to \mathbb{R}$，定义

$$
\boldsymbol{H}_{i,j} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

即所有二阶偏导排成的 $n \times n$ 矩阵。在二阶偏导连续的条件下（Clairaut 定理），**混合偏导与求导次序无关**，于是 $\boldsymbol{H}_{i,j} = \boldsymbol{H}_{j,i}$——**黑塞矩阵是对称矩阵**，这个性质在后面讨论特征值时至关重要。

**黑塞矩阵告诉我们函数在 $\boldsymbol{x}_0$ 附近的弯曲形态**。对梯度为零的驻点，由对称矩阵 $\boldsymbol{H}$ 的特征值即可分类：

| $\boldsymbol{H}$ 的特征值 | 驻点的类型 | 直觉 |
| --- | --- | --- |
| 全部为正 | 局部极小值 | 谷底，任何方向都向上弯 |
| 全部为负 | 局部极大值 | 峰顶，任何方向都向下弯 |
| 有正有负 | 鞍点 | 马鞍：有些方向向上，有些方向向下 |
| 含零 | 退化 | 需要用更高阶信息判断 |

**鞍点（saddle point）**是高维优化里比局部极小值更普遍的障碍：在千维、万维参数空间里，驻点几乎都是鞍点——沿大部分方向是下坡、少数方向是上坡。这对直觉的冲击很大：一维里「平地」多半是极值，高维里「平地」几乎必然是鞍点。<span class="marginnote">这也是「深度学习为什么能优化」的经典争论之一：若损失曲面全是鞍点与盆地，一阶方法依旧能沿着那些下坡方向稳步前进，未必需要精确逃出极小值——见第三篇《优化问题：病态条件、局部极小值与鞍点》。</span>

黑塞矩阵还直接度量了优化的**病态条件（conditioning）**：令 $\lambda_{\max}$ 与 $\lambda_{\min}$ 为最大、最小特征值，则比值 $\frac{\lambda_{\max}}{\lambda_{\min}}$ 越大，损失曲面越「拉长」，梯度下降沿陡方向震荡、沿缓方向龟爬，收敛越慢。这正是自适应学习率方法（RMSProp、Adam）要解决的痛点。

## 4 公式解析：多元泰勒展开的三个层次

把梯度、雅可比、黑塞三件工具一次性串起来的，是多元函数的**泰勒展开**：

$$
f(\boldsymbol{x}) \approx f(\boldsymbol{x}_0) + \underbrace{\nabla f(\boldsymbol{x}_0)^{\top}(\boldsymbol{x}-\boldsymbol{x}_0)}_{\text{一阶：梯度}} + \underbrace{\tfrac{1}{2}(\boldsymbol{x}-\boldsymbol{x}_0)^{\top}\boldsymbol{H}(\boldsymbol{x}-\boldsymbol{x}_0)}_{\text{二阶：黑塞}}
$$

对这条式子做三步拆解：

- **第一步，看清三个层次**：第 0 项是常数 $f(\boldsymbol{x}_0)$；第 1 项是一阶修正，由梯度向量 $\nabla f$ 驱动，是**线性**的；第 2 项是二阶修正，由黑塞矩阵 $\boldsymbol{H}$ 驱动，是**二次**的。
- **第二步，验证维度**：$\nabla f$ 是 $n$ 维列向量，$\boldsymbol{x}-\boldsymbol{x}_0$ 是 $n$ 维列向量，两者转置相乘得标量；$\boldsymbol{H}$ 是 $n\times n$ 矩阵，夹在 $(\boldsymbol{x}-\boldsymbol{x}_0)^{\top}$ 与 $(\boldsymbol{x}-\boldsymbol{x}_0)$ 之间，二次型同样得标量。**每一项的维度都严丝合缝地凑成 $1\times n \times n\times 1 = 1$**——矩阵乘法的维度检查是最便宜的防错手段。
- **第三步，读出几何**：一阶项说「沿梯度方向走高，函数近似直线上升」；二阶项说「曲面还在弯」。若 $\boldsymbol{H}$ 的特征值很大，二次项贡献显著，一阶近似很快就会失效——这正是固定学习率容易震荡的数学根源。

## 5 三者的辨析与实用惯例

把三件工具放在同一张表里对照，是理解它们关系的最快方式：

| 工具 | 输入→输出 | 形状 | 含义 |
| --- | --- | --- | --- |
| 梯度 $\nabla f$ | $\mathbb{R}^n \to \mathbb{R}$ | $n$ 维列向量 | 单输出对每个输入的一阶敏感度 |
| 雅可比 $\boldsymbol{J}$ | $\mathbb{R}^n \to \mathbb{R}^m$ | $m \times n$ 矩阵 | 每个输出对每个输入的一阶敏感度 |
| 黑塞 $\boldsymbol{H}$ | $\mathbb{R}^n \to \mathbb{R}$ | $n \times n$ 对称矩阵 | 单输出对输入的二阶弯曲 |

**易错点一：行向量还是列向量。** 不同教材把梯度写为行向量（$1 \times n$）或列向量（$n \times 1$）。花书与大部分优化文献采用**列向量**惯例，于是「梯度下降」写作 $\boldsymbol{w} \leftarrow \boldsymbol{w} - \eta \nabla_{\boldsymbol{w}} L$。混用惯例是矩阵求导错误的头号来源——写代码前先想清楚自己手里是行还是列。

**易错点二：黑塞矩阵为什么「贵」。** 一个 $n$ 参数模型的黑塞矩阵有 $n^2$ 个元素。现代模型动辄数十亿参数，$n^2$ 是天文数字，直接构造黑塞矩阵在存储上就不可行。这解释了为什么工程上用的是它的替身：对角近似（AdaGrad、RMSProp 每维只存一个尺度）、低秩近似（K-FAC），或干脆只用一阶信息——这是第三篇《自适应学习率》《二阶优化近似》的主题。

**易错点三：把「梯度消失」怪罪给梯度本身。** 深度网络的「梯度消失」不是梯度算错了，而是链式法则连乘多个雅可比矩阵后，谱范数小于 1 的部分被反复压缩——问题出在**雅可比的连乘**，而非梯度定义。<span class="marginnote">这一点把本节与第五篇《通过时间的反向传播》直接接通：RNN 里梯度爆炸/消失，正是同一机制在时间维上的重演。</span>

## 6 小结

- **梯度** $\nabla f$：单输出函数的偏导列向量，指向最陡上升方向，**负梯度**是最陡下降方向。
- **雅可比矩阵** $\boldsymbol{J}$：多输出函数的一阶导数矩阵，$m=1$ 时退化为梯度转置；链式法则在雅可比语言下就是**矩阵连乘**，反向传播由此而来。
- **黑塞矩阵** $\boldsymbol{H}$：二阶偏导的对称矩阵，特征值符号决定驻点是极小、极大还是鞍点，特征值比值度量病态条件。
- 三者统一在**多元泰勒展开**的框架里：一阶项由梯度驱动，二阶项由黑塞驱动。
- 工程上黑塞矩阵因 $O(n^2)$ 存储而不可直接构造，催生了各类近似。

在下一节，我们将从「算梯度」走到「怎么高效算出梯度」——这就是**微积分与自动微分**：把求导从手推公式变成程序在计算图上自动完成的流水线。
