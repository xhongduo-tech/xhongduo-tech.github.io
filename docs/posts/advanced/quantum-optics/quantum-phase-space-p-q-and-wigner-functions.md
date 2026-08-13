---
title: 量子相空间：P、Q 表示与 Wigner 函数
date: 2026-08-07
---

# 量子相空间：P、Q 表示与 Wigner 函数

<div class="epigraph">
<p>Wigner 函数并不是一个真正的概率分布，它有时候会取负值——而这正是量子性的标志。</p>
<footer>—— 尤金·维格纳（Eugene P. Wigner），1932 年</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子光学 ｜ D. F. Walls & G. J. Milburn, Quantum Optics 第3章 ｜ 2026-08-07</p>
</div>

## 为什么从量子相空间开始

经典力学里，一个粒子的状态是相空间里的一个点 $(q, p)$；
统计力学里，一群粒子是相空间上的一个分布。
量子力学没有这么干净的图像——不确定关系禁止「精确的点」。但 
1932 年维格纳发现，
量子态仍然可以画成相空间上的**准概率分布**，只是它可能取负值。
量子光学继承了这个想法，
发展出三种相互联系的表示：**P 表示、Q 表示与 Wigner 函数**。
它们是理解压缩态、
零差探测与量子态断层成像的几何语言。<span class="marginnote">本节的相空间概念直接对接第一级《向量与张量初步》里的坐标变换思想，
以及第二级《量子力学》里的量子化规则。</span>

## 1 三种表示：一个家族

三种表示都是把密度算符 $\rho$ 映射为相空间上的函数，
区别在于用哪组基展开：

**P 表示（对角相干态展开）**：把 $\rho$ 
写成相干态 $|\alpha\rangle$ 的「加权叠加」

$$\rho = \int d^2\alpha\, P(\alpha)\,|\alpha\rangle\langle\alpha|$$

$P(\alpha)$ 可以是高度奇异的分布（甚至不是普通函数），
但它让正规排序算符的期望变成经典平均：$\langle \hat{a}^{\dagger k}\hat{a}^l\rangle = \int d^2\alpha\, P(\alpha)\,\alpha^{*k}\alpha^l$。

**Q 表示（相干态期望）**：

$$Q(\alpha) = \frac{1}{\pi}\langle\alpha|\rho|\alpha\rangle$$

Q 函数总是光滑非负的，
因为 $\langle\alpha|\rho|\alpha\rangle \geq 0$；
它对应反正规排序算符的期望。

**Wigner 函数（对称排序）**：

$$W(\alpha) = \frac{1}{\pi^2}\int d^2\beta\, e^{\alpha\beta^* - \alpha^*\beta}\,\mathrm{Tr}\left[\rho\, e^{\beta\hat{a}^\dagger - \beta^*\hat{a}}\right]$$

Wigner 
函数可以是负的，**负值区域 = 非经典性**。<span class="marginnote">三种表示对应三种「排序规则」：
正规（P）、反正规（Q）、对称/维格纳（W）。
算符排序并非纯技术问题——它决定哪个量是「经典可类比」的。</span>

## 2 三种表示的对比

| 表示 | 定义核心 | 是否非负 | 对应排序 | 主要用途 |
| --- | --- | --- | --- | --- |
| P 表示 | $\rho = \int d^2\alpha\,P(\alpha)\vert\alpha\rangle\langle\alpha\vert$ | 可负/奇异 | 正规 | 半经典展开、非线性光学 |
| Q 表示 | $Q(\alpha) = \frac{1}{\pi}\langle\alpha\vert\rho\vert\alpha\rangle$ | 恒非负 | 反正规 | 直观图像、相空间截断 |
| Wigner 函数 | 对称排序的特征函数傅里叶变换 | 可负 | 对称 | 断层成像、非经典性判据 |

**重点：Q 函数总非负，但它的非负性并不代表「经典性」**——因为 
Q 是反正规排序，它天然抹掉了一些量子信息。判读非经典性必须用 
Wigner 函数（或其负值），而不是 Q。
这是初学者最容易踩的坑。

## 3 以相干态为例：三种画法

设 $\rho = |\alpha_0\rangle\langle\alpha_0|$ 
是相干态，三种表示分别是：

$$P(\alpha) = \delta^{(2)}(\alpha - \alpha_0), \qquad Q(\alpha) = \frac{1}{\pi}e^{-|\alpha-\alpha_0|^2}, \qquad W(\alpha) = \frac{2}{\pi}e^{-2|\alpha-\alpha_0|^2}$$

三者都是「中心在 $\alpha_0$ 的斑点」，但宽度不同：P 
是无穷窄的尖峰，Q 半径 $\sqrt{1/2}$，W 
半径 $1/2$。
几何图像与上一节「相干态是相空间里最小圆斑」完全一致。<span class="marginnote">量子噪声在这个图像里一目了然：
不是粒子在抖动，
而是态本身在相空间里是「糊」的——糊的大小由不确定关系决定。</span>

**辨析｜易错点：** 不要把 $W(\alpha)$ 
与 $Q(\alpha)$ 的归一化搞混。
两者都归一（$\int d^2\alpha\,W = \int d^2\alpha\,Q = 1$），
但形状不同；只有 $Q$ 恒非负。若论文中 Wigner 
函数出现负值，那是在告诉你「这个态没有经典对应」。

## 4 公式解析：Wigner 函数与相干态的正交分量

Wigner 
函数最常用的等价形式是用正交分量 $X_1, X_2$（$\hat{a} = X_1 + iX_2$）写出的边缘分布性质：


$$\int dx_2\, W(x_1, x_2) = \langle x_1|\rho|x_1\rangle, \qquad \int dx_1\, W(x_1, x_2) = \langle x_2|\rho|x_2\rangle$$

拆成三步：

- **第一步，边缘积分 = 概率**：把 Wigner 函数对 $x_2$ 积分，得到测量 $X_1$ 的概率密度。这是它被称为「准概率」的依据——边缘是真正的概率，但联合分布本身可负。
- **第二步，物理后果**：量子态断层成像正是靠这个性质工作的：转动相空间、测一串边缘分布，再逆 Radon 变换重构 $W(x_1, x_2)$。这与医学 CT 的层析原理相同——只是被断层的对象是量子态。
- **第三步，负值的意义**：$W \lt  0$ 的区域没有经典概率解释，是纯粹干涉/纠缠的体现。对 $|0\rangle$，$W \geq 0$；对 Fock 态 $|1\rangle$，$W$ 在原点取负值 $\langle 0|W|1\rangle \lt  0$