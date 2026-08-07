---
title: 角点检测：Harris 角点与 Shi-Tomasi 角点
date: 2026-08-07
---

# 角点检测：Harris 角点与 Shi-Tomasi 角点

<div class="epigraph">
<p>计算机视觉最难的部分不是看清，而是知道往哪里看。</p>
<footer>—— 大卫 · 马尔（David Marr），《视觉》</footer>
</div>

<div class="article-byline">
<p>第四级 · 计算机视觉 ｜ 《计算机视觉：算法与应用》（Szeliski）§4.1 ｜ 2026-08-07</p>
</div>

## 为什么从角点检测开始

上一节的边缘检测告诉我们图像里「线」在哪，但线在拼接、配准、跟踪中还不够用：一条直线上的任意一点都长得一样，无法唯一地对齐两幅图像。我们需要的是**在图像中能被唯一辨认的小区域**——图像拼接时要在两张图里找到「同一块」，跟踪时要在下一帧里找到「同一个点」。<span class="marginnote">把直觉翻译成数学：我们希望找到这样的像素，当窗口在它周围移动时，图像内容变化显著到「独一无二」。这正是 Harris 在 1988 年给出的定义——「角点 = 局部窗口移动时灰度变化都很剧烈的位置」。</span>

角点是「特征点」里最简单的一类，是后文 SIFT、SURF、ORB 等更复杂特征点的起点。这一节先掌握两个最经典的基于微分的角点响应：**Harris 角点**与它的改进 **Shi-Tomasi 角点**。

## 1 从自相关函数出发：窗口移动与灰度变化

设图像 $I$，考虑一个以 $(x,y)$ 为中心的小窗口 $W$，把它移动 $(\Delta x, \Delta y)$ 后，窗口内灰度变化的加权平方和（SSD）是：

$$
E(\Delta x, \Delta y) = \sum_{(u,v) \in W} w(u,v) \left[ I(u+\Delta x, v+\Delta y) - I(u,v) \right]^2
$$

其中 $w(u,v)$ 是窗口权重（通常取高斯，中心权重高）。对**微小移动**，用一阶泰勒展开 $I(u+\Delta x, v+\Delta y) \approx I(u,v) + I_x \Delta x + I_y \Delta y$，代入得：

$$
E(\Delta x, \Delta y) \approx \begin{pmatrix} \Delta x & \Delta y \end{pmatrix}
M \begin{pmatrix} \Delta x \\ \Delta y \end{pmatrix}
$$

这里的 $M$ 是**结构张量（structure tensor）**或**二阶矩矩阵**：

$$
M = \sum_{(u,v) \in W} w(u,v)
\begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
$$

$M$ 是对称正半定矩阵，它总结了窗口内的梯度分布。<span class="marginnote">矩阵 $M$ 把「窗口内梯度的情况」压缩成三个量：两个特征值 $\lambda_1, \lambda_2$ 与特征方向。特征值的大小反映「沿该方向的梯度能量」——这正是判断角点/边缘/平坦区的全部信息。特征值分解在第二级《线性代数》中已系统学过。</span>

## 2 特征值的几何解读：三种区域的判别

对对称矩阵 $M$，特征值分解给出 $M = R^{-1} \mathrm{diag}(\lambda_1, \lambda_2) R$。设 $\lambda_1 \geq \lambda_2 \geq 0$，三个区域分别对应：

- **平坦区**：$\lambda_1$、$\lambda_2$ 都很小。任意方向移动，$E$ 都不变——没有可定位的特征。
- **边缘**：$\lambda_1 \gg \lambda_2$（或反之）。沿边缘方向移动，灰度几乎不变；垂直方向变化剧烈。只能在「垂直边缘方向」被定位，这就是上一节提到的**孔径问题**的雏形。
- **角点**：$\lambda_1 \approx \lambda_2$ 且都较大。往任何方向移动，灰度都显著变化——位置可被唯一确定。

**角点判定的核心标准：结构张量两个特征值都大，且量级相近。** 于是问题变成「如何用一个标量刻画『两个特征值都大且接近』」，Harris 给出的是不带特征值分解的近似。

## 3 Harris 角点响应函数

直接做特征值分解计算量大，Harris 用一个判别式近似。定义**角点响应值（cornerness）**：

$$
R = \det(M) - k \, \operatorname{trace}(M)^2
$$

其中 $\det(M) = \lambda_1 \lambda_2$，$\operatorname{trace}(M) = \lambda_1 + \lambda_2$，$k$ 是经验常数，通常取 $0.04 \sim 0.06$。改写为特征值形式：

$$
R = \lambda_1 \lambda_2 - k (\lambda_1 + \lambda_2)^2
$$

**辨析｜易错点：** $R$ 的符号在不同文献中定义相反。Harris 原文对**角点定义 $R > 0$**：两个特征值都大时，$\det$ 大而 $\operatorname{trace}^2$ 相对受控，$R$ 为正；边缘区 $\det \approx 0$，$R$ 为负；平坦区 $R$ 很小。因此角点检测实际是找 $R$ 的**局部极大值**（先对 $R$ 图做非极大值抑制），再叠加阈值过滤噪声。

Harris 的贡献在于用一个判别式避免显式求特征值，但代价是 $k$ 需要人工调。这也引出下一个改进。

## 4 Shi-Tomasi 角点：直接取最小特征值

Shi-Tomasi（1994）做了一个极简而优雅的替换：**不再用近似判别式，直接取结构张量的较小特征值作为角点响应**：

$$
R = \min(\lambda_1, \lambda_2)
$$

理由是：角点要求「两个方向都能定位」，而「最弱的方向」决定了定位能力。$\min(\lambda_1, \lambda_2)$ 大，说明任何方向都有足够的梯度——这比 Harris 判别式更直接、更稳定。<span class="marginnote">Shi-Tomasi 在光流跟踪（KLT，即 Kanade-Lucas-Tomasi）中提出：跟踪要找「最容易跟丢时也跟得准」的点，即两个方向都被约束的点。所以它又被叫做「Good Features to Track」。后文 Lucas-Kanade 光流一节会再见到这组名字。</span>

实践中 Harris 与 Shi-Tomasi 检出的角点位置往往高度重合，差异主要在排序与阈值行为：Shi-Tomasi 的响应值有清晰的几何含义（最小特征值），阈值更易设定；Harris 的判别式计算更快（无需特征值分解）。

## 5 公式解析：从 $E(\Delta x,\Delta y)$ 到 Harris 响应

把整条推理链用一条不等式串起来：

$$
E(\Delta x, \Delta y) \approx (\Delta x, \Delta y)\, M \begin{pmatrix}\Delta x \\ \Delta y\end{pmatrix}
\;\Longrightarrow\;
M \text{ 的特征值 } \lambda_1, \lambda_2
\;\Longrightarrow\;
R = \lambda_1\lambda_2 - k(\lambda_1+\lambda_2)^2
$$

逐步拆解：

- **第一步，泰勒展开**：微小移动下 $E$ 变成 $\Delta x, \Delta y$ 的二次型。二次型的「形状」由 $M$ 决定，$M$ 的特征值就是沿主轴方向的弯曲程度。
- **第二步，几何含义**：$\lambda_1, \lambda_2$ 大且接近 → $E$ 的等值线近似圆形且半径小 → 任何方向的移动都引起大变化 → 角点。$\lambda_1 \gg \lambda_2$ → 等值线拉成长条 → 边缘。
- **第三步，判别式近似**：$\det = \lambda_1\lambda_2$ 惩罚「一个特征值为零」的情况（$\det=0$），$k\,\operatorname{trace}^2$ 惩罚「特征值都偏小」的情况。两者相减恰好区分三种区域。

**一句话记忆：角点检测 = 对每个像素算结构张量 → 用特征值（或其近似判别式）打分 → 非极大值抑制 + 阈值。** 这条「打分 + 抑制 + 阈值」的流程，与 Canny 的后三步结构同构，也与后文所有特征点检测器共享骨架。

## 6 尺度问题与下一步

Harris 有一个固有缺陷：**固定窗口没有尺度概念**。同一角点在小尺度下是尖锐折角，在大尺度下可能只是圆弧。上一节我们提到过「尺度空间」这个名词——角点要应对不同大小的物体，就必须在不同尺度下检测。这直接引出下一节：**尺度空间理论与 DoG 关键点检测**，也是 SIFT 的第一块基石。<span class="marginnote">工程上 Harris 的变形还有 Harris-Laplace：在多个尺度上检测 Harris 响应，再选尺度方向响应最强的点，相当于给角点加上了「自适应尺度」能力。但真正让尺度选择成为系统方法的是下一节的 DoG。</span>

## 7 小结

- **角点**是「窗口任何方向移动灰度都大变」的位置，数学上用结构张量 $M$ 的特征值刻画。
- 三种区域判别：**平坦区**（$\lambda_1,\lambda_2$ 都小）、**边缘**（一大一小）、**角点**（都大且接近）。
- **Harris 响应** $R = \det(M) - k\,\operatorname{trace}(M)^2$，$R>0$ 判角点，用近似避免特征值分解。
- **Shi-Tomasi** 直接取 $R = \min(\lambda_1,\lambda_2)$，几何意义清晰，源自光流跟踪。
- 检测流程统一为「打分 → 非极大值抑制 → 阈值」，与 Canny 后段同构。
- Harris 无尺度概念，是下一节尺度空间与 DoG 的动机。

在下一节，我们将把「窗口大小」升级为「尺度」，介绍如何在不同尺度下稳定地找到关键点——这是 SIFT 的起点。
