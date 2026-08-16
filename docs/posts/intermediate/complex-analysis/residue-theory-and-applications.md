---
title: 留数理论及其应用（实积分计算、辐角原理）
date: 2026-08-07
---

# 留数理论及其应用（实积分计算、辐角原理）

<div class="epigraph">
<p>任何一处小小的遗留之数，都维系着全局的和——这是解析函数论对数学慷慨赠予的一件礼物。</p>
<footer>—— 奥古斯丁-路易 · 柯西（Augustin-Louis Cauchy），《关于留数的报告》（1826–1830 系列论著）</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数》第5章 ｜ 2026-08-07</p>
</div>

## 为什么从「绕圈的代价」开始

洛朗级数告诉我们，孤立奇点周围的信息全部藏在主部里。而主部里最特别的一项，是 $\frac{1}{z-z_0}$ 的系数——因为第三章我们已经算过：$\oint\frac{dz}{z-z_0}=2\pi i$，其余负幂次（$\frac{1}{(z-z_0)^2}$ 等）的积分为零。**于是整个围道积分的值，最终只由这一个系数决定**。这个系数就叫**留数（residue）**。留数理论把「计算复杂围道积分」降维成「找出几个奇点、各取一个系数」，是复分析回报率最高的工具之一。<span class="marginnote">「留数」一词来自拉丁语 residuum，原意是「剩下的东西」。柯西在一篇篇关于积分路径变形的报告里发现：把积分路径搬来搬去，真正「剩下来」决定积分数值的，只有这些孤立奇点的贡献。</span>

## 1 留数的定义与留数定理

**核心概念：留数（residue）**：设 $z_0$ 是 $f$ 的孤立奇点，$f$ 在去心邻域内的洛朗展开为 $\sum_{n=-\infty}^{\infty}a_n(z-z_0)^n$，则称 $a_{-1}$ 为 $f$ 在 $z_0$ 的留数，记作

$$\mathrm{Res}(f, z_0) = a_{-1}$$

**核心概念：留数定理（residue theorem）**：设 $f$ 在简单闭曲线 $C$ 及其内部解析，仅在 $C$ 内部有有限个孤立奇点 $z_1,\dots,z_n$（均不在 $C$ 上），则

$$\oint_C f(z)\,dz = 2\pi i \sum_{k=1}^{n}\mathrm{Res}(f, z_k)$$

**证明思路**：用复合闭路定理把 $C$ 变形到每个奇点周围的小圆，每圈贡献 $2\pi i\,\mathrm{Res}(f,z_k)$，其余负幂项积分为零，相加即得。<span class="marginnote">留数定理的威力在「化整为零」：一个绕大圈、看似无解的积分，被拆成绕每个奇点的小圈的求和，而每个小圈只需要一个洛朗系数。计算 $\oint_C f\,dz$ 从此变成「找奇点 → 算留数 → 求和」的流水线。</span>

## 2 留数的计算规则

算留数不必每次展开洛朗级数。常用规则：

**规则一：简单极点的留数**。若 $z_0$ 是 $f$ 的一阶极点（简单极点），则

$$\mathrm{Res}(f,z_0) = \lim_{z\to z_0}(z-z_0)f(z)$$

**规则二：分式的留数**。若 $f(z)=\frac{P(z)}{Q(z)}$，$z_0$ 是 $Q$ 的一阶零点且 $P(z_0)\ne0$，则

$$\mathrm{Res}(f,z_0) = \frac{P(z_0)}{Q'(z_0)}$$

这一条在实积分应用里几乎天天用。

**规则三：$m$ 阶极点的留数**。

$$\mathrm{Res}(f,z_0) = \frac{1}{(m-1)!}\lim_{z\to z_0}\frac{d^{m-1}}{dz^{m-1}}\left[(z-z_0)^m f(z)\right]$$

**规则四：无穷远点的留数**。设 $f$ 在扩充复平面上只有有限个奇点，则

$$\mathrm{Res}(f,\infty) = -\mathrm{Res}\left(\frac{1}{z^2}f\left(\frac1z\right), 0\right)$$

**重要事实（全留数定理）**：$f$ 在所有有限奇点与无穷远点的留数之和为零。这条守恒律常常把「绕大圆」的积分与「内部奇点」的留数联系起来，是实积分应用的关键一环。

## 3 公式解析：留数定理如何算实积分

### 类型一：$\int_0^{2\pi} R(\cos\theta,\sin\theta)\,d\theta$

令 $z=e^{i\theta}$，则 $dz=iz\,d\theta$，且

$$\cos\theta=\frac{z+z^{-1}}{2}, \qquad \sin\theta=\frac{z-z^{-1}}{2i}$$

代入后原积分化为绕单位圆的围道积分 $\oint_{|z|=1} g(z)\,dz$，再用留数定理。**核心例子**（西交教材 §5.7 经典题）：

$$I=\int_0^{2\pi}\frac{d\theta}{1+\varepsilon\cos\theta}, \qquad 0<\varepsilon<1$$

令 $z=e^{i\theta}$，化为

$$I=\oint_{|z|=1}\frac{1}{1+\varepsilon\frac{z+z^{-1}}{2}}\frac{dz}{iz}=\frac{2}{i\varepsilon}\oint_{|z|=1}\frac{dz}{z^2+\frac{2}{\varepsilon}z+1}$$

分母两根为 $z_{1,2}=-\frac1\varepsilon\pm\sqrt{\frac{1}{\varepsilon^2}-1}$。因为 $0<\varepsilon<1$，$z_1$（取 $+$）在圆内、$z_2$ 在圆外。圆内仅一个简单极点，留数为 $\frac{1}{z_1-z_2}=\frac{1}{2\sqrt{1/\varepsilon^2-1}}=\frac{\varepsilon}{2\sqrt{1-\varepsilon^2}}$。故

$$I=\frac{2}{i\varepsilon}\cdot 2\pi i\cdot\frac{\varepsilon}{2\sqrt{1-\varepsilon^2}}=\frac{2\pi}{\sqrt{1-\varepsilon^2}}$$

这个结果在信号处理里就是「瑞利散射/谐振曲线」的积分原型。<span class="marginnote">验证一下特例：$\varepsilon=0$ 时 $I=\int_0^{2\pi}d\theta=2\pi$，公式给出 $2\pi/\sqrt{1}=2\pi$，吻合。这个「代换 $z=e^{i\theta}$ → 单位圆围道积分 → 找内部极点」的模板，是把三角积分变成代数问题的标准把戏。</span>

### 类型二：$\int_{-\infty}^{\infty}R(x)\,dx$

设 $R(x)=\frac{P(x)}{Q(x)}$，$Q$ 无实零点，且 $\deg Q\ge\deg P+2$。考虑上半平面的大半圆围道（半径 $\rho$ 的实轴线段 + 上半圆弧）。由 $\deg$ 条件，圆弧上的积分随 $\rho\to\infty$ 趋于 $0$（ML 估计），故

$$\int_{-\infty}^{\infty}R(x)\,dx = 2\pi i\sum_{\text{上半平面奇点}}\mathrm{Res}(R, z_k)$$

### 类型三：$\int_{-\infty}^{\infty}R(x)e^{iax}\,dx$ 与若尔当引理

当 $R$ 满足 $\deg Q\ge\deg P+1$ 时，实轴积分不一定绝对收敛，但 $e^{iax}$ 的振荡挽救它。**若尔当引理**：设 $R(z)$ 在上半平面（含实轴）除有限奇点外解析，$\lim_{R\to\infty}\max_{|z|=R,\mathrm{Im}\,z\ge0}|R(z)|=0$，则对 $a>0$，

$$\lim_{R\to\infty}\int_{\text{上半圆弧}}R(z)e^{iaz}\,dz = 0$$

据此

$$\int_{-\infty}^{\infty}R(x)e^{iax}\,dx = 2\pi i\sum_{\text{上半平面}}\mathrm{Res}(R(z)e^{iaz})$$

**直觉**：$e^{iaz}=e^{iax}e^{-ay}$ 在上半平面（$y>0$）指数衰减，圆弧越远贡献越小，所以不用像类型二那样苛求 $\deg$ 条件。<span class="marginnote">类型三是 Fourier 变换的桥梁：傅里叶变换（本专题第七章）正是 $\int R(x)e^{-i\omega x}dx$ 型的实积分，留数定理给出了一大批常见频谱函数的闭式解。两条主线在这里第一次握手。</span>

## 4 辐角原理与儒歇定理：零点的计数

留数定理还有一个「几何版」，专门回答「区域内有多少个零点、多少个极点」。

**核心概念：辐角原理（argument principle）**：设 $f$ 在简单闭曲线 $C$ 上及其内部解析，$C$ 上无零点，$C$ 内部有 $N$ 个零点（按重数计）、$P$ 个极点（按阶数计），则

$$\frac{1}{2\pi i}\oint_C\frac{f'(z)}{f(z)}\,dz = N - P$$

**几何解释**：$\frac{f'(z)}{f(z)}dz$ 是 $f$ 的对数微分 $d\log f$，其围道积分 $2\pi i(N-P)$ 正比于 $f(z)$ 沿 $C$ 转动时辐角的总变化量——「$f$ 把 $C$ 卷了 $N-P$ 圈」。这就是「辐角原理」名字的来源。<span class="marginnote">辐角原理是「零点计数」的利器：想知道一个区域内方程的根有多少个，不必解方程，只需看 $f$ 绕边界转了几圈。它在控制论里判断特征根在左半平面的个数（Nyquist 判据的思想同源），是复分析渗入工程的最深脉络之一。</span>

**核心概念：儒歇定理（Rouché's theorem）**：设 $f$ 与 $g$ 在 $C$ 上及其内部解析，且在 $C$ 上 $|g(z)|<|f(z)|$，则 $f$ 与 $f+g$ 在 $C$ 内部有同样多的零点（按重数计）。

**推论（代数基本定理的又一证明）**：令 $f(z)=a_nz^n$，$g(z)=a_{n-1}z^{n-1}+\cdots+a_0$。在足够大的圆 $|z|=R$ 上 $|g|<|f|$，故多项式 $f+g$ 与 $z^n$ 有同样多的零点，即恰有 $n$ 个——代数基本定理再次被复分析一句话拿下。<span class="marginnote">儒歇定理（Jean Rouché, 1862）的直观：$f$ 是「主项」，$g$ 是小扰动，只要扰动在边界上严格小于主项，零点个数就稳定不变。这与动力系统里「结构稳定」的思想一脉相承。</span>

## 5 核心对比表：三种实积分套路

| 类型 | 被积对象 | 代换/围道 | 圆弧上如何处理 |
| --- | --- | --- | --- |
| $\int_0^{2\pi}R(\cos\theta,\sin\theta)d\theta$ | 三角有理式 | $z=e^{i\theta}$ → 单位圆 | 无需圆弧 |
| $\int_{-\infty}^{\infty}R(x)dx$ | 有理式 | 上半圆 + 实轴 | 用 $\deg$ 条件使圆弧贡献为 $0$ |
| $\int_{-\infty}^{\infty}R(x)e^{iax}dx$ | 有理式 × 指数 | 上半圆 + 实轴 | 若尔当引理（$a>0$） |
| $\frac{1}{2\pi i}\oint\frac{f'}f dz$ | 对数微分 | 任意闭路 | $=N-P$（辐角原理） |

这四行就是留数理论的「工具箱」。前两行解决工程师最常遇到的定积分，第三行把触角伸向傅里叶变换，第四行则服务于零点计数——**同一个 $2\pi i$，四副面孔**。

## 6 小结

- **留数** $\mathrm{Res}(f,z_0)=a_{-1}$：洛朗主部里 $\frac{1}{z-z_0}$ 的系数，是绕该奇点一圈积分的价签。
- **留数定理** $\oint_C f\,dz=2\pi i\sum\mathrm{Res}$：围道积分化归为奇点处留数之和。
- 留数计算规则：简单极点 $\lim(z-z_0)f$、分式 $\frac{P}{Q'}$、$m$ 阶极点公式、无穷远点留数。
- **实积分三类**：三角型（$z=e^{i\theta}$）、$\int R(x)dx$（半圆+$\deg$ 条件）、$\int R(x)e^{iax}dx$（若尔当引理）。
- **辐角原理** $\frac{1}{2\pi i}\oint\frac{f'}f=N-P$ 与**儒歇定理**给出零点的几何计数法。

在下一节，我们从「角度被怎样映射」出发，回答复分析最直观的一问：**什么样的映射保角？** 那将引出分式线性变换与黎曼映射定理——**共形映射**。
