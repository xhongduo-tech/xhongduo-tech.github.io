---
title: 范数、迹运算与伪逆
date: 2026-08-07
---

# 范数、迹运算与伪逆

<div class="epigraph">
<p>并非所有能计数的都重要，也并非所有重要的都能计数。</p>
<footer>—— 威廉 · 布鲁斯 · 卡梅伦（William Bruce Cameron）</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§2.5、§2.9–2.10 ｜ 2026-08-07</p>
</div>

## 为什么从范数、迹与伪逆开始

向量和矩阵能表达「方向」，但要谈论「多大」「多接近」「误差多少」，还需要一把**尺子**。梯度下降要比较参数更新前后差多少，正则化要惩罚过大的权重，评估模型要看预测与真值的距离——这些「量大小、量距离」的活，全部由**范数**承担。**迹运算**则是矩阵世界里的「求和号」：它把矩阵压成一个数，是定义矩阵范数、联系特征值、推导矩阵导数的万能钥匙。而**伪逆**回答的是一个更实际的问题：方程 $\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$ 在 $\boldsymbol{A}$ 不可逆甚至不是方阵时，该怎么「尽量解」？

上一节我们用 SVD 把矩阵拆成了「旋转—伸缩—旋转」。本节的三样工具都站在同一块地基上：**范数量化大小，迹运算提取总和，伪逆用奇异值构造「广义的逆」**——三者合起来，就是深度学习里「度量误差、惩罚参数、求解最小二乘」的全部数学设施。<span class="marginnote">这一节与后面几篇的接口非常直接：L2 范数平方就是权重衰减（weight decay）惩罚项，L1 范数引出稀疏化，伪逆则是线性回归「从零实现」里正规方程 $\boldsymbol{x}=(\boldsymbol{A}^{\top}\boldsymbol{A})^{-1}\boldsymbol{A}^{\top}\boldsymbol{b}$ 的一般化版本。</span>

## 1 范数：测量向量的大小

**范数（norm）**：把向量（或矩阵）映射成非负实数的函数，衡量其「大小」。机器学习中最常用的是 $L^p$ 族，对向量 $\boldsymbol{x} \in \mathbb{R}^n$ 定义为

$$
\|\boldsymbol{x}\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}, \qquad p \ge 1
$$

其中 $p$ 是范数的阶数。几个常见特例：

- **$L^2$ 范数（欧氏范数）**：$\|\boldsymbol{x}\|_2 = \sqrt{\sum_i x_i^2} = \sqrt{\boldsymbol{x}^{\top}\boldsymbol{x}}$，就是几何里的「长度」。它的**平方** $\|\boldsymbol{x}\|_2^2 = \boldsymbol{x}^{\top}\boldsymbol{x}$ 在梯度计算里更常用，因为求导时平方刚好消去根号。
- **$L^1$ 范数**：$\|\boldsymbol{x}\|_1 = \sum_i |x_i|$，把分量绝对值加起来，对稀疏向量（很多分量为 0）极其敏感。
- **$L^\infty$ 范数（max 范数）**：$\|\boldsymbol{x}\|_\infty = \max_i |x_i|$，只看绝对值最大的分量。

任意范数都满足三条公理：**非负性**（$\|\boldsymbol{x}\| \ge 0$，且 $\|\boldsymbol{x}\|=0 \iff \boldsymbol{x}=\boldsymbol{0}$）、**齐次性**（$\|c\boldsymbol{x}\| = |c|\,\|\boldsymbol{x}\|$）、**三角不等式**（$\|\boldsymbol{x}+\boldsymbol{y}\| \le \|\boldsymbol{x}\| + \|\boldsymbol{y}\|$）。三角不等式保证了「走直路不绕远」，正是距离感成立的根基。<span class="marginnote">把三个范数的单位球画在平面里：$L^2$ 是圆，$L^1$ 是菱形，$L^\infty$ 是正方形。为什么 $L^1$ 正则化偏爱稀疏解？因为菱形在坐标轴上有「尖角」，最优点容易落在角上——这就是第一级《解析几何》的图形直觉与第三篇正则化的第一次握手。</span>

**矩阵的范数**最常用的是 **Frobenius 范数**，把矩阵当长向量数平方根：

$$
\|\boldsymbol{A}\|_F = \sqrt{\sum_{i,j} A_{i,j}^2}
$$

它衡量「这个矩阵整体有多大」，也是低秩逼近（上一节截断 SVD）里衡量逼近误差的标准尺子。

## 2 迹运算：矩阵的求和号

**迹（trace）**：方阵主对角线元素之和，记作

$$
\mathrm{tr}(\boldsymbol{A}) = \sum_{i=1}^{n} A_{i,i}
$$

迹有三条几乎天天用的性质：

- **转置不变**：$\mathrm{tr}(\boldsymbol{A}^{\top}) = \mathrm{tr}(\boldsymbol{A})$。
- **循环性（cyclicity）**：乘积的迹与乘积顺序的循环移位无关——
  $$
  \mathrm{tr}(\boldsymbol{A}\boldsymbol{B}\boldsymbol{C}) = \mathrm{tr}(\boldsymbol{C}\boldsymbol{A}\boldsymbol{B}) = \mathrm{tr}(\boldsymbol{B}\boldsymbol{C}\boldsymbol{A})
  $$
- **迹 = 特征值之和**：对 $n \times n$ 方阵，$\mathrm{tr}(\boldsymbol{A}) = \sum_{i=1}^n \lambda_i$。

第一条性质说明迹只关心「沿主对角线的总和」，与坐标系无关（相似变换不改变迹）；第二条是推导矩阵恒等式的大杀器——很多矩阵乘积的表达式看着没法化简，循环移位一次就豁然开朗。第三条则把迹与上一节的特征分解接通：知道特征值之和，就能不展开矩阵直接读出迹。

迹还回馈了范数：Frobenius 范数可以写成

$$
\|\boldsymbol{A}\|_F = \sqrt{\mathrm{tr}(\boldsymbol{A}^{\top}\boldsymbol{A})}
$$

因为 $\boldsymbol{A}^{\top}\boldsymbol{A}$ 的主对角线恰好是 $\boldsymbol{A}$ 各列元素的平方和。又因 $\mathrm{tr}(\boldsymbol{A}^{\top}\boldsymbol{A}) = \sum_i \sigma_i^2$（奇异值平方和），Frobenius 范数于是也与 SVD 的奇异值挂钩——**范数、迹、奇异值三者在这一点汇合**。

**辨析｜易错点：循环性不是「任意换序」。** $\mathrm{tr}(\boldsymbol{A}\boldsymbol{B}\boldsymbol{C}) = \mathrm{tr}(\boldsymbol{C}\boldsymbol{A}\boldsymbol{B})$ 允许把最右的因子**循环**搬到最左，但不允许随意交换相邻因子：一般地 $\mathrm{tr}(\boldsymbol{A}\boldsymbol{B}\boldsymbol{C}) \neq \mathrm{tr}(\boldsymbol{A}\boldsymbol{C}\boldsymbol{B})$。一句话口诀：**只能转圈，不能乱换**。

## 3 Moore–Penrose 伪逆：解「解不了」的方程

线性方程组 $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$，当 $\boldsymbol{A}$ 是 $n \times n$ 可逆矩阵时，答案干净利落：$\boldsymbol{x} = \boldsymbol{A}^{-1}\boldsymbol{b}$。但深度学习里几乎遇不到这么理想的状况——要么 $\boldsymbol{A}$ 是**长方形**（数据矩阵 $n$ 样本 × $d$ 特征，一般 $n \neq d$），要么**不可逆**（有相关性导致秩亏）。这时人们退而求其次，问两个问题：

1. 没有精确解时，能否找**使 $\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2$ 最小**的 $\boldsymbol{x}$（最小二乘）？
2. 最小二乘解不止一个时，能否选**范数最小**的那个？

**Moore–Penrose 伪逆（pseudoinverse）** $\boldsymbol{A}^{+}$ 一次性回答两者：定义

$$
\boldsymbol{A}^{+} = \boldsymbol{V}\boldsymbol{D}^{+}\boldsymbol{U}^{\top}
$$

其中 $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{\top}$ 是上一节的 SVD，$\boldsymbol{D}^{+}$ 由 $\boldsymbol{D}$ 转置并把非零奇异值取倒数得到。而

$$
\boldsymbol{x} = \boldsymbol{A}^{+}\boldsymbol{b}
$$

正是**最小范数最小二乘解**：在所有使 $\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2$ 最小的 $\boldsymbol{x}$ 里，它的欧氏范数最小。<span class="marginnote">伪逆满足四个「逆式的」恒等式：$\boldsymbol{A}\boldsymbol{A}^{+}\boldsymbol{A}=\boldsymbol{A}$、$\boldsymbol{A}^{+}\boldsymbol{A}\boldsymbol{A}^{+}=\boldsymbol{A}^{+}$，以及 $\boldsymbol{A}\boldsymbol{A}^{+}$、$\boldsymbol{A}^{+}\boldsymbol{A}$ 都是对称矩阵。直觉：$\boldsymbol{A}^{+}$ 在 $\boldsymbol{A}$ 的值域（列空间）上扮演逆矩阵，在零空间上则「什么也不做」。</span>

**辨析｜易错点：伪逆不是逆矩阵，也不是「硬凑的」。** 当 $\boldsymbol{A}$ 可逆时 $\boldsymbol{A}^{+}=\boldsymbol{A}^{-1}$，伪逆确实退化为普通逆；但当 $\boldsymbol{A}$ 列满秩时，常用公式 $\boldsymbol{A}^{+}=(\boldsymbol{A}^{\top}\boldsymbol{A})^{-1}\boldsymbol{A}^{\top}$ 要求 $\boldsymbol{A}^{\top}\boldsymbol{A}$ 可逆——一旦 $\boldsymbol{A}$ 存在线性相关的列，$\boldsymbol{A}^{\top}\boldsymbol{A}$ 奇异，这个公式直接失效，而 SVD 定义下的伪逆依然健在。这正是伪逆优于「先凑逆再套公式」的根本原因。

## 4 公式解析：伪逆 $\boldsymbol{A}^{+} = \boldsymbol{V}\boldsymbol{D}^{+}\boldsymbol{U}^{\top}$ 的构造逻辑

伪逆的式子短，但里面藏着整套 SVD 思想。逐块拆开：

- **第一步，写出 SVD**。$\boldsymbol{A} = \boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{\top}$，$\boldsymbol{D}$ 是 $m \times n$ 对角阵，主对角线是奇异值 $\sigma_1 \ge \cdots \ge \sigma_r > 0$（$r=\text{rank}\,\boldsymbol{A}$），其余为 0。
- **第二步，构造 $\boldsymbol{D}^{+}$**。把 $\boldsymbol{D}$ **转置**（从 $m \times n$ 变 $n \times m$），再把非零对角元 $\sigma_i$ 换成 $1/\sigma_i$。零元素保持为零——对应零奇异值的那些方向「没有逆」，直接作废。
- **第三步，转置回绕**。$\boldsymbol{A}^{+} = \boldsymbol{V}\boldsymbol{D}^{+}\boldsymbol{U}^{\top}$ 的形状：$n \times n$ 乘 $n \times m$ 乘 $m \times m$，得到 $n \times m$——正好把 $\boldsymbol{A}$ 的 $m \times n$「反着」织回来。方向上的解读是：$\boldsymbol{U}^{\top}$ 把 $\boldsymbol{b}$ 旋到左奇异向量基，$\boldsymbol{D}^{+}$ 沿非零方向按 $1/\sigma_i$ 反伸缩，$\boldsymbol{V}$ 再旋回输出空间。
- **第四步，验证最小二乘**。对 $\boldsymbol{b}$ 在列空间内的部分，$\boldsymbol{A}\boldsymbol{A}^{+}\boldsymbol{b} = \boldsymbol{b}$ 精确还原；对垂直于列空间的分量，$\boldsymbol{A}^{+}$ 把它们清零，因此 $\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2$ 被压到最小。

看一个具体例子。设

$$
\boldsymbol{A} = \begin{bmatrix} 1 & 0 \end{bmatrix} \in \mathbb{R}^{1 \times 2}
$$

它的 SVD 中 $\sigma_1 = 1$，$\boldsymbol{D}^{+} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$，于是 $\boldsymbol{A}^{+} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$。对任意 $\boldsymbol{b}$，$\boldsymbol{x} = \boldsymbol{A}^{+}\boldsymbol{b} = \begin{bmatrix} b \\ 0 \end{bmatrix}$：方程 $x_1 = b$ 的解是 $(b, t)$（$t$ 任意），其中范数最小者恰好是 $t=0$ 的那个——伪逆自动挑出了最小范数解。

## 5 小结

- **范数**衡量向量/矩阵大小：$L^p$ 范数 $\|\boldsymbol{x}\|_p=(\sum_i|x_i|^p)^{1/p}$，常用 $L^2$、$L^1$、$L^\infty$；矩阵用 Frobenius 范数 $\|\boldsymbol{A}\|_F=\sqrt{\sum_{i,j}A_{i,j}^2}$。
- **迹** $\mathrm{tr}(\boldsymbol{A})=\sum_i A_{i,i}$：转置不变、**只能循环移位**、等于特征值之和；Frobenius 范数可写成 $\sqrt{\mathrm{tr}(\boldsymbol{A}^{\top}\boldsymbol{A})}$。
- **伪逆** $\boldsymbol{A}^{+}=\boldsymbol{V}\boldsymbol{D}^{+}\boldsymbol{U}^{\top}$：对任意矩阵存在，$\boldsymbol{x}=\boldsymbol{A}^{+}\boldsymbol{b}$ 是最小范数最小二乘解；可逆时 $\boldsymbol{A}^{+}=\boldsymbol{A}^{-1}$。
- **易错**：$L^0$（非零元个数）不是真范数；迹只能循环换位不能任意换序；伪逆在列相关时仍有效，而 $(\boldsymbol{A}^{\top}\boldsymbol{A})^{-1}\boldsymbol{A}^{\top}$ 会失效。
- **汇合点**：Frobenius 范数 = 奇异值平方和 = $\sqrt{\mathrm{tr}(\boldsymbol{A}^{\top}\boldsymbol{A})}$，范数、迹、SVD 在此连成一体。

在下一节，我们将离开线性代数的几何舞台，进入深度学习的第二块数学基石——**概率论**。从随机变量与概率分布开始，回答「不确定的世界里如何定量推理」。
