---
title: 复数的乘幂与方根
date: 2026-08-07
---

# 复数的乘幂与方根

<div class="epigraph">
<p>上帝创造了整数，其余一切都是人的作品。</p>
<footer>—— 利奥波德 · 克罗内克（Leopold Kronecker）</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数》§1.3 ｜ 2026-08-07</p>
</div>

## 为什么从乘幂与方根开始

上一节我们把复数写成 $z = re^{i\theta}$，并发现乘法就是「旋转 + 缩放」。这一节把这句话用到极致：**连乘 $n$ 次**会怎样？反过来，**开 $n$ 次方**——即解方程 $w^n = z$——又会有几个解？

答案出乎意料地漂亮：$z^n$ 就是把辐角放大 $n$ 倍、把模取 $n$ 次方；而 $z$ 的 $n$ 次方根**恰好有 $n$ 个**，它们均匀地分布在以原点为圆心的圆周上，构成一个正 $n$ 边形。这个「均匀分布」的结构，就是后面快速傅里叶变换（FFT）与大模型里旋转位置编码共享的代数骨架。

## 1 整数次幂：旋转的迭加

设 $z = re^{i\theta}$。由乘法规则 $z_1 z_2 = r_1 r_2 e^{i(\theta_1+\theta_2)}$，把 $z$ 连乘 $n$ 次，得到

**核心概念：复数的整数次幂**：

$$z^n = r^n e^{in\theta} = r^n\bigl(\cos n\theta + i\sin n\theta\bigr)$$

把它写回三角形式，就是

**核心概念：棣莫弗公式（De Moivre's formula）**：

$$(\cos\theta + i\sin\theta)^n = \cos n\theta + i\sin n\theta$$

即 $r = 1$ 时的特例。几何含义：每乘一次 $z$，就旋转 $\theta$ 角、缩放 $r$ 倍；乘 $n$ 次，就是旋转 $n\theta$ 角、缩放 $r^n$ 倍。

**辨析｜易错点：棣莫弗公式的 $n$ 是整数。** 把 $n$ 换成 $\frac{1}{2}$ 之类分数时，式子右边不再唯一——因为方根是多值的。公式只对整数幂直接用，方根另有本章的方法（下一节）。很多同学把「$(e^{i\theta})^{1/2} = e^{i\theta/2}$」当成单值等式，这正是对多值性认识不清的第一个雷区。

利用棣莫弗公式，还能把「倍角公式」用三行写出来。例如取 $n = 2$：

$$(\cos\theta + i\sin\theta)^2 = \cos 2\theta + i\sin 2\theta$$

左端展开得 $\cos^2\theta - \sin^2\theta + 2i\sin\theta\cos\theta$，实部虚部分别相等，立即得到 $\cos 2\theta = \cos^2\theta - \sin^2\theta$，$\sin 2\theta = 2\sin\theta\cos\theta$。<span class="marginnote">这是「复数把三角恒等式变成代数计算」的第一个样本：<strong>一个复等式等于实部、虚部两个实等式</strong>。第七章做频谱分析时，把信号的实部虚部分开看，仍是这一招。</span>

## 2 公式解析：n 次方根是怎样「解」出来的

现在解方程 $w^n = z$。设 $z = re^{i\theta}$（$r \ge 0$），未知量 $w$ 写成 $w = \rho e^{i\varphi}$。代入：

$$(\rho e^{i\varphi})^n = \rho^n e^{in\varphi} = re^{i\theta}$$

分三步拆解：

- **第一步，模相等。** 两边取模，$\rho^n = r$。因为 $\rho, r$ 都是非负实数，且 $\rho^n = r$ 在实数里恰有一个非负解，所以 $\rho = \sqrt[n]{r}$（这是实数的算术根，单值）。
- **第二步，辐角相差 $2\pi$ 的整数倍。** 指数部分的相等条件是「辐角差为 $2\pi$ 的整数倍」：$n\varphi = \theta + 2k\pi$，即

$$\varphi = \frac{\theta + 2k\pi}{n}, \qquad k \in \mathbb{Z}$$

- **第三步，数出不同解。** 让 $k$ 遍历整数，$\varphi$ 会出现无穷多个值，但它们每 $n$ 个重复一次：$k$ 与 $k+n$ 给出的 $\varphi$ 相差 $2\pi$，对应同一个复数。因此真正不同的解只有 $k = 0, 1, \ldots, n-1$ 这 $n$ 个。

**核心概念：复数的 $n$ 次方根**：方程 $w^n = z$ 恰有 $n$ 个解

$$w_k = \sqrt[n]{r}\, e^{i\frac{\theta + 2k\pi}{n}}, \qquad k = 0, 1, \ldots, n-1$$

**重点：非零复数的 $n$ 次方根恰好有 $n$ 个。** 这与实数情形完全不同——在实数里，$x^2 = 4$ 只有 $\pm 2$ 两个解，$x^3 = 8$ 只有 $1$ 个解；在复数里，任何非零复数的 $n$ 次方根都有**整整 $n$ 个**。这是复数「代数封闭」特质的第一道风景：代数基本定理保证 $n$ 次方程在复数域里总有 $n$ 个根（计重数）。<span class="marginnote">当 $z = 0$ 时，$w^n = 0$ 只有 $w = 0$ 一个解（重根），所以「$n$ 个根」的结论只对<strong>非零</strong>复数成立。</span>

## 3 方根的正多边形几何

$n$ 个方根 $w_0, w_1, \ldots, w_{n-1}$ 都落在以原点为圆心、半径为 $\sqrt[n]{r}$ 的圆周上，相邻两个辐角相差 $\frac{2\pi}{n}$。因此它们是一个正 $n$ 边形的顶点。

**例子一：$z = 1$ 的三次方根。** $1 = 1 \cdot e^{i\cdot 0}$，$r = 1$，$\theta = 0$，$n = 3$：

$$w_k = e^{i\frac{2k\pi}{3}}, \qquad k = 0, 1, 2$$

即 $w_0 = 1$，$w_1 = e^{i\frac{2\pi}{3}} = -\frac{1}{2} + \frac{\sqrt{3}}{2}i$，$w_2 = e^{i\frac{4\pi}{3}} = -\frac{1}{2} - \frac{\sqrt{3}}{2}i$。记 $w_1 = \omega$，则三个根是 $\{1, \omega, \omega^2\}$——以原点为中心的正三角形。注意 $\omega^2 + \omega + 1 = 0$，且 $1 + \omega + \omega^2 = 0$：三个根的和为零。

![单位圆上 1 的三个立方根 1、ω、ω² 构成正三角形](/images/complex-analysis/powers-and-roots-1.svg)

**例子二：$z = i$ 的三次方根。** $i = e^{i\frac{\pi}{2}}$，$\theta = \frac{\pi}{2}$，于是

$$w_k = e^{i\frac{\frac{\pi}{2} + 2k\pi}{3}}, \qquad k = 0, 1, 2$$

得 $w_0 = e^{i\frac{\pi}{6}} = \frac{\sqrt{3}}{2} + \frac{1}{2}i$，$w_1 = e^{i\frac{5\pi}{6}} = -\frac{\sqrt{3}}{2} + \frac{1}{2}i$，$w_2 = e^{i\frac{3\pi}{2}} = -i$。仍是正三角形的三个顶点，且三个根的和为零。

**例子三：$z = -1$ 的四次方根。** $-1 = e^{i\pi}$，$n = 4$：

$$w_k = e^{i\frac{\pi + 2k\pi}{4}}, \qquad k = 0, 1, 2, 3$$

得 $e^{i\frac{\pi}{4}}$、$e^{i\frac{3\pi}{4}}$、$e^{i\frac{5\pi}{4}}$、$e^{i\frac{7\pi}{4}}$，即 $\frac{\sqrt{2}}{2}(1+i)$、$\frac{\sqrt{2}}{2}(-1+i)$、$\frac{\sqrt{2}}{2}(-1-i)$、$\frac{\sqrt{2}}{2}(1-i)$——分布在四个象限，构成一个正方形。

**辨析｜易错点：三个例子里所有方根的和都是零，这是巧合吗？** 不是。$n$ 个方根 $w_0, \ldots, w_{n-1}$ 是方程 $w^n - z = 0$ 的全部根，由韦达定理，$n$ 次方程所有根之和等于一次项系数除以最高次系数的相反数——这里一次项系数为 $0$，所以根之和恒为 $0$（$n \ge 2$ 时）。

## 4 单位根：旋转对称的代数骨架

$z = 1$ 的 $n$ 次方根称为 **$n$ 次单位根（roots of unity）**：

$$\varepsilon_k = e^{i\frac{2k\pi}{n}} = \cos\frac{2k\pi}{n} + i\sin\frac{2k\pi}{n}, \qquad k = 0, 1, \ldots, n-1$$

它们把单位圆均分成 $n$ 段。四个最重要的性质：

- $\varepsilon_k^n = 1$（它们是 $1$ 的方根，天经地义）；
- $\varepsilon_k = \varepsilon_1^k$：所有单位根都是 $\varepsilon_1 = e^{i\frac{2\pi}{n}}$ 的幂——**一个根生成全部**；
- $\sum_{k=0}^{n-1} \varepsilon_k = 0$（$n \ge 2$ 时，正多边形中心对称）；
- $\prod_{k=0}^{n-1} \varepsilon_k = (-1)^{n+1}$。

**单位根就是旋转对称性的代数签名。** 它出现在一切「把圆均分」的地方：离散傅里叶变换（DFT）矩阵的每一项正是 $\varepsilon_k$ 的幂，快速傅里叶变换（FFT）利用单位根的高次幂关系把 $O(n^2)$ 降到 $O(n\log n)$；信号里「$n$ 点循环卷积」的周期结构也由单位根刻画。而在大模型的位置编码中，把位置 $m$ 映射为旋转 $m\theta$，本质上就是单位根在「长度为周期」的空间里旋转。<span class="marginnote">单位根的「一个根生成全部」（$\varepsilon_k = \varepsilon_1^k$）对应群论里的<strong>循环群</strong>：$n$ 个元素由单个生成元反复作用而成。这个结构到第二级《抽象代数》会正式登场，这里先种下一颗种子。</span>

**例题：求 $1$ 的四次方根。** 由 $\varepsilon_k = e^{i\frac{2k\pi}{4}} = e^{i\frac{k\pi}{2}}$，$k = 0, 1, 2, 3$，得

$$\{1,\; i,\; -1,\; -i\}$$

四个根均匀分布在单位圆上，正好落在四个象限，构成一个正方形；它们的和 $1 + i - 1 - i = 0$，与「$n \ge 2$ 时单位根之和为零」完全吻合。这正是快速傅里叶变换里「$n=4$ 点蝶形运算」用到的那组数——四个单位根把四个采样点旋到对称位置，对称性就是速度的来源。

## 5 小结

- **整数次幂**：$z^n = r^n e^{in\theta}$，即「模取 $n$ 次方、辐角放大 $n$ 倍」；棣莫弗公式 $(\cos\theta + i\sin\theta)^n = \cos n\theta + i\sin n\theta$。
- **$n$ 次方根**：$w^n = z$ 对非零 $z$ 恰有 $n$ 个解 $w_k = \sqrt[n]{r}\,e^{i\frac{\theta + 2k\pi}{n}}$，$k = 0, 1, \ldots, n-1$。
- **几何形状**：$n$ 个方根均匀分布在半径 $\sqrt[n]{r}$ 的圆周上，构成正 $n$ 边形；$n \ge 2$ 时所有方根之和为零。
- **单位根**：$1$ 的 $n$ 次方根 $\varepsilon_k = e^{i\frac{2k\pi}{n}}$，满足 $\varepsilon_k = \varepsilon_1^k$、$\sum \varepsilon_k = 0$。
- **陷阱**：棣莫弗公式只对整数幂直接用；$z = 0$ 的方根只有一个；分数次幂是多值的。

在下一节，我们走出平面，登上**复球面**——把复数平面「卷」到一个球面上，并邀请一位新成员「无穷远点 $\infty$」加入复数世界。这个扩充会让我们在第五章留数、第六章共形映射里，能对「在无穷远处取什么值」这种问题给出干净的回答。
