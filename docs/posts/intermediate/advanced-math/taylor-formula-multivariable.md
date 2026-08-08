---
title: 二元函数的泰勒公式
date: 2026-08-07
---

# 二元函数的泰勒公式

<div class="epigraph">
<p>用多项式逼近函数，用局部拼出全局——泰勒的艺术在多维世界依旧奏效。</p>
<footer>—— 布鲁克 · 泰勒（Brook Taylor）</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等数学 ｜ 同济《高等数学》下册 §9.9 ｜ 2026-08-07</p>
</div>

## 为什么从二元函数的泰勒公式开始

一元泰勒公式把函数展开成「$1, x, x^2, \ldots$」的线性组合；二元函数的输入有两个变量，展开的「基底」变成二元多项式——$1, x, y, x^2, xy, y^2, \ldots$。**二元泰勒公式**把「多项式逼近」从一维推广到二维：在点附近，$f(x,y)$ 可用「常数 + 线性项 + 二次项 + …」逼近。它的意义远超公式本身：二元泰勒的二次截断正是 Hessian 判别法的理论来源，也是机器学习里二阶优化（牛顿法）与误差分析的基础。<span class="marginnote">二元泰勒公式的一阶截断就是全微分/切平面（局部线性化），二阶截断是「抛物面近似」（局部二次化）。<strong>线性化是一阶泰勒，二次化是二阶泰勒</strong>——机器学习的优化算法分水岭（一阶梯度下降 vs 二阶牛顿法）正是截断阶数的选择。</span>

## 1 二元函数的泰勒公式

设 $f(x,y)$ 在点 $(x_0, y_0)$ 的某邻域内具有直到 $n+1$ 阶的连续偏导数，则对邻域内任意 $(x,y)$，有**二元泰勒公式**（记 $h = x - x_0$、$k = y - y_0$）：

$$f(x,y) = f(x_0,y_0) + \left(f_x\,h + f_y\,k\right) + \frac{1}{2!}\left(f_{xx}h^2 + 2f_{xy}hk + f_{yy}k^2\right) + \cdots + \frac{1}{n!}\left(h\frac{\partial}{\partial x} + k\frac{\partial}{\partial y}\right)^{(n)} f(x_0,y_0) + R_n$$

其中余项（拉格朗日型）$R_n = \frac{1}{(n+1)!}\left(h\frac{\partial}{\partial x} + k\frac{\partial}{\partial y}\right)^{(n+1)} f(\xi, \eta)$，$(\xi,\eta)$ 在 $(x_0,y_0)$ 与 $(x,y)$ 的连线段上。

**重点：符号算子 $\left(h\frac{\partial}{\partial x} + k\frac{\partial}{\partial y}\right)$**——展开时按二项式定理展开，每一「次方」对应相应阶的偏导组合。<span class="marginnote">「算子」写法是二元泰勒的浓缩记法：$(h\partial_x + k\partial_y)^2 = h^2\partial_{xx} + 2hk\partial_{xy} + k^2\partial_{yy}$——二项式系数 $1,2,1$ 与 $(h+k)^2$ 完全一致。这个「把偏导算子当变量」的符号技巧，让多元泰勒的写法极其紧凑，也是傅里叶分析中平移算子的前奏。</span>

**一阶截断（线性近似）**：$f \approx f_0 + f_x h + f_y k$——正是全微分/切平面。

**二阶截断（二次近似）**：$f \approx f_0 + f_x h + f_y k + \frac{1}{2}\left(f_{xx}h^2 + 2f_{xy}hk + f_{yy}k^2\right)$——抛物面近似，Hessian 判别法就来自这一步。

## 2 二元泰勒与一元泰勒的联系

二元泰勒不是凭空新理论，而是「把一元泰勒分别沿两条路走」的组合：固定 $y=y_0$，$f(x,y_0)$ 是 $x$ 的一元函数，可沿 $x$ 展开；固定 $x$，$f(x_0,y)$ 沿 $y$ 展开。二元泰勒正是这两种一元展开的「混合」——交叉项 $f_{xy}hk$ 来自「先 $x$ 后 $y$」或「先 $y$ 后 $x$」的混合二阶导。<span class="marginnote">理解交叉项的来源：若 $f(x,y) = g(x)h(y)$，则 $f_{xy} = g'(x)h'(y)$——混合二阶导来自「两个方向的独立变化」耦合。Clairaut 定理保证 $f_{xy}=f_{yx}$，所以展开里 $hk$ 项系数合并为 $2f_{xy}$，与 $(h+k)^2$ 的交叉项系数 2 对应。</span>

**公式解析：一元到二元的展开**

$$f(x,y) \approx f_0 + f_x h + f_y k + \frac{1}{2}\left(f_{xx}h^2 + 2f_{xy}hk + f_{yy}k^2\right)$$

- **第一步，对照一元**：一元二阶泰勒 $f(x) \approx f_0 + f'h + \frac{1}{2}f''h^2$——每一项从「导数 × $h^n$」推广为「$n$ 阶偏导组合 × $h^{n-i}k^i$」。
- **第二步，看系数**：二项式系数 $1, 2, 1$ 出现在 $h^2, hk, k^2$ 前（$hk$ 的 2 来自两项合并）。
- **第三步，识别结构**：一次项是梯度点积 $\nabla f\cdot(h,k)$，二次项是 $\frac12(h,k)H(h,k)^T$（Hessian 二次型）。

**关键**：二元泰勒的每一项都是「$n$ 阶偏导 × $n$ 次齐次多项式」，系数由二项式定理决定——与一元泰勒同构，只是「导数」变成「偏导组合」。

## 3 二元泰勒的应用：极值判别的再推导

二元泰勒公式为 Hessian 判别法提供了严格依据。在驻点 $f_x=f_y=0$ 处：

$$f(x,y) - f_0 \approx \frac{1}{2}\left(f_{xx}h^2 + 2f_{xy}hk + f_{yy}k^2\right)$$

- 若 Hessian 正定（$AC-B^2>0$ 且 $A>0$）：右边恒正 ⇒ $f > f_0$ ⇒ **极小值**；
- 若 Hessian 负定：右边恒负 ⇒ **极大值**；
- 若 Hessian 不定（$AC-B^2<0$）：右边可正可负 ⇒ **鞍点**。

**极值判别的本质，就是看二阶截断这一项的符号**——这就是二元泰勒公式对多元极值理论的直接贡献。<span class="marginnote">这个推导揭示了一条普适方法：<strong>判别局部极值 = 考察泰勒展开的最低阶「未消失项」</strong>。一元看 $f''h^2/2$，二元看 Hessian 二次型。到《最优化理论》，这一思想发展为「二阶充分条件」与凸函数理论——「局部行为由最低阶非零项决定」。</span>

## 4 二元泰勒的现代应用

**数值误差分析**：多元函数的误差传播——若 $x,y$ 有误差 $\delta_x,\delta_y$，$f$ 的误差约为一阶截断 $|f_x|\delta_x + |f_y|\delta_y$，二阶截断给出更高精度估计。<span class="marginnote">机器学习里损失函数对参数的二阶泰勒近似是牛顿法的基础：$L(\theta+\Delta) \approx L(\theta) + \nabla L\cdot\Delta + \frac12\Delta^T H \Delta$，牛顿法令导数为零解 $\Delta = -H^{-1}\nabla L$——二阶信息加速收敛。你在这里学的二元泰勒，正是二阶优化器的理论源头。</span>
- **Hessian 矩阵与曲率**：$f_{xx}, f_{yy}, f_{xy}$ 组成的 Hessian 描述曲面局部曲率，特征值定曲率主方向。
- **计算机视觉**：图像局部的二阶结构（Hessian）用于角点检测（Harris 角点）、尺度空间分析。
- **经济学**：多元生产函数的二阶展开用于比较静态与二阶条件验证。

## 5 小结

- **二元泰勒公式**：$f \approx f_0 + (f_xh+f_yk) + \frac{1}{2}(f_{xx}h^2+2f_{xy}hk+f_{yy}k^2) + \cdots$，用算子 $(h\partial_x+k\partial_y)$ 紧凑表示。
- 一阶截断 = 全微分/切平面；二阶截断 = 抛物面近似（Hessian 二次型）。
- 系数由**二项式定理**决定，交叉项 $2f_{xy}hk$ 来自 Clairaut 合并。
- 极值判别 = 看二阶截断的符号——Hessian 正定/负定/不定。
- 应用：误差分析、牛顿法、Hessian 曲率、角点检测。

在下一节，我们将应用多元微积分解决统计中的经典问题——**最小二乘法**。
