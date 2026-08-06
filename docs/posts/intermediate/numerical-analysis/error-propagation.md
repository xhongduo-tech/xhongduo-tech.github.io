---
title: 函数运算的误差估计
date: 2026-08-07
---

# 函数运算的误差估计：误差如何在算式中旅行

<div class="epigraph">
<p>垃圾进，垃圾出。</p>
<footer>—— 乔治 · 富切尔（George Fuechsel，IBM 程序员，1950 年代）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§1.2.3 ｜ 2026-08-07</p>
</div>

## 为什么从函数运算的误差估计开始

前两节我们有了误差的分类（§1.2.1）和度量（§1.2.2），但还缺最要紧的一环：**误差不是静止的，它会在算式里流动。** 输入带误差，经 $y=f(x)$ 一变，输出误差是变大还是变小？加、减、乘、除四则运算各自把误差放大多少？如果不能回答这些问题，任何「误差限」都只是入口处的标签，算不到出口。<span class="marginnote">「垃圾进，垃圾出」（Garbage in, garbage out）是计算机界的名言，常被归功于 IBM 程序员乔治 · 富切尔。数值分析把这句话变成了定量命题：输入误差以什么倍率传播到输出，这个倍率叫条件数；本节先把一元与多元函数的情形算清楚。</span>

本节的方法论一句话：**误差传播由泰勒展开的一阶项统治。** 只要误差足够小，二阶以上的项都可以忽略，于是每个输入变量的误差都按各自的偏导数独立放大，最后叠加。这也是为什么上一节《[数值分析的研究对象与特点](./what-is-numerical-analysis)》里那条 $|\Delta y|\approx |f'(x)||\Delta x|$ 会反复出现——本节把它推广到多元函数与四则运算。

## 1 一元函数：f'(x) 是误差放大系数

设 $y=f(x)$ 可微，输入 $x$ 带有误差 $\Delta x$，输出误差 $\Delta y = f(x+\Delta x)-f(x)$。对 $f$ 做一阶泰勒展开：

$$
f(x+\Delta x) = f(x) + f'(x)\Delta x + o(\Delta x)
$$

忽略高阶小量，得到输出误差与输入误差的线性关系

$$
\Delta y \approx f'(x)\,\Delta x, \qquad |\Delta y| \approx |f'(x)|\,|\Delta x|
$$

$|f'(x)|$ 就是**误差放大系数**：大于 1，误差被放大；小于 1，误差被压缩。再写成相对误差：

$$
\frac{|\Delta y|}{|y|} \;\approx\; \underbrace{\frac{|x f'(x)|}{|f(x)|}}_{\text{条件数 } c(x)} \cdot \frac{|\Delta x|}{|x|}
$$

系数 $c(x)=\dfrac{|xf'(x)|}{|f(x)|}$ 是该点处的**条件数（condition number）**，它回答：输入相对误差被放大了多少倍。下一节《病态问题与条件数》会对它作系统讨论，这里先记住它是一元误差传播的核心。<span class="marginnote">两个熟悉的例子：$f(x)=x^n$ 的条件数是 $n$，10 次方运算把输入误差放大 10 倍；$f(x)=\sqrt{x}$ 的条件数是 $\tfrac12$，开方把相对误差压缩一半——所以稳定的算法里常能看到开方。</span>

## 2 多元函数：偏导数接力，误差各自放大再相加

实际问题几乎都是多元的。设 $y = f(x_1,x_2,\dots,x_n)$，各输入 $x_i$ 带有误差 $\Delta x_i$。多元泰勒一阶展开给出

$$
\Delta y \;\approx\; \sum_{i=1}^{n} \frac{\partial f}{\partial x_i}\,\Delta x_i
$$

取绝对值，用三角不等式放缩成最坏情形：

$$
|\Delta y| \;\le\; \sum_{i=1}^{n} \left|\frac{\partial f}{\partial x_i}\right|\,|\Delta x_i|
$$

这就是**多元误差传播公式**：每个输入误差先按各自偏导数的绝对值放大，再求和。写成相对误差更常用：

$$
\varepsilon_r(y) \;\approx\; \sum_{i=1}^{n} \left|\frac{\partial f}{\partial x_i}\frac{x_i}{f}\right| \varepsilon_r(x_i)
$$

每个系数 $\left|\dfrac{\partial f}{\partial x_i}\dfrac{x_i}{f}\right|$ 是第 $i$ 个变量的条件数，可称为**偏弹性**。它回答：输入 $x_i$ 的相对误差以多大幅度贡献到输出。<span class="marginnote">「各自放大再相加」暗含一个保守假设：所有输入的误差同时朝最坏方向走。现实中独立测量的误差常常相互抵消，统计上更合理的合成方式是「平方和开方」（RSS），这在第二级《概率论与数理统计》里会严格讨论。数值分析先用最坏情形，因为它给出的是「保证不超过」的界。</span>

## 3 四则运算的误差限：最坏情形的算术

把上节公式套到四则运算上，得到一组可以直接背下的误差限。设 $x,y$ 的绝对误差限分别为 $\varepsilon(x), \varepsilon(y)$：

| 运算 | 绝对误差限 | 相对误差限 |
| --- | --- | --- |
| 加 $x\pm y$ | $\varepsilon(x)+\varepsilon(y)$ | — |
| 乘 $x\cdot y$ | $\lvert y\rvert\varepsilon(x)+\lvert x\rvert\varepsilon(y)$ | $\varepsilon_r(x)+\varepsilon_r(y)$ |
| 除 $x/y$ | $\dfrac{\lvert y\rvert\varepsilon(x)+\lvert x\rvert\varepsilon(y)}{y^2}$ | $\varepsilon_r(x)+\varepsilon_r(y)$ |
| 幂 $x^p$ | — | $\approx p\,\varepsilon_r(x)$ |

三条规律值得记住：

- **加减法：绝对误差限相加。** 两个近似数相加，绝对误差至多等于两者绝对误差之和——所以「大数加小数」时，小数的误差常常淹没在大数的误差里，这正是「避免大数吃小数」原则的误差根源。
- **乘除法：相对误差限相加。** 相乘除时，相对误差限直接相加，条件数在最坏情形下按因子累加。这也解释了为什么「除以一个很小的数」很危险——$y$ 很小会让 $\varepsilon(x)/y^2$ 项爆炸，表现为绝对误差限公式里的 $1/y^2$ 放大。
- **幂运算：指数是放大倍数。** $x^p$ 的相对误差限约为 $p$ 倍输入相对误差；$p$ 越大越危险，$p=\tfrac12$（开方）反而压缩。

**辨析｜易错点：** 许多人以为乘除法误差「随机抵消」就没事，但数值分析给的误差限是**确定性上界**——它承诺「无论误差方向如何，都不会超过」。随机抵消是统计层面的事，不能用来推翻确定性上界；两者用途不同，一个保底、一个平均。

## 4 公式解析：乘积的相对误差等于各因子相对误差之和

把「乘除法相对误差相加」这条推导一步步拆开，体会误差传播公式的用法。设 $z = xy$，输入带误差 $x+\Delta x$、$y+\Delta y$：

- **第一步，写乘积的误差。** 

$$
\Delta z = (x+\Delta x)(y+\Delta y) - xy = x\Delta y + y\Delta x + \Delta x\Delta y
$$

- **第二步，忽略二阶小量。** 当 $\Delta x,\Delta y$ 都很小时，$\Delta x\Delta y$ 是更高阶无穷小，舍去，得 $\Delta z \approx y\Delta x + x\Delta y$。
- **第三步，转成相对误差。** 两边同除以 $z=xy$：

$$
\frac{\Delta z}{z} \;\approx\; \frac{y\Delta x + x\Delta y}{xy} \;=\; \frac{\Delta x}{x} + \frac{\Delta y}{y}
$$

- **第四步，取绝对值。** 得到 $\varepsilon_r(z) \le \varepsilon_r(x) + \varepsilon_r(y)$。

这条结论立刻可以推广：$n$ 个因子连乘，相对误差限就是各因子相对误差限之和；幂次 $p$ 相当于 $p$ 个因子，所以是 $p\,\varepsilon_r(x)$。**相对误差在乘法里做加法，绝对误差在加法里做加法**——这是误差传播里最常用的两句话。

## 5 一个完整的例题

测量一个圆的半径 $r = 5.00\ \text{cm}$，误差限 $\varepsilon(r)=0.01\ \text{cm}$（相对误差限 0.2%），用 $\pi\approx 3.1416$ 计算面积 $A=\pi r^2$，求面积的误差限。

- 面积值：$A = \pi r^2 \approx 3.1416\times 25.0 \approx 78.54\ \text{cm}^2$。
- 相对误差：$A=\pi r^2$ 是 $\pi$ 的一次方、$r$ 的平方，故 $\varepsilon_r(A) \approx \varepsilon_r(\pi) + 2\varepsilon_r(r)$。$\varepsilon_r(\pi)\approx 2.3\times10^{-6}$ 可忽略，$2\varepsilon_r(r) = 2\times 0.002 = 0.004$。
- 绝对误差限：$\varepsilon(A) \approx \varepsilon_r(A)\cdot A \approx 0.004\times 78.54 \approx 0.31\ \text{cm}^2$。

所以面积应写作 $A = 78.5 \pm 0.3\ \text{cm}^2$。**注意半径误差被平方放大成两倍贡献**——若要求面积三位有效数字，半径就得量到误差限 $0.0005\ \text{cm}$ 以下。这就是误差传播倒逼测量精度的例子。

## 6 Python：把误差界算出来

用 Python 把上面的估计程序化，并顺手验证除法情形：

```python
import math

# 例 1：圆的面积，r = 5.00 ± 0.01 cm
r, dr = 5.00, 0.01
A = math.pi * r**2
dA = 2 * math.pi * r * dr          # 2πr·Δr
print(f"A = {A:.3f} ± {dA:.3f} cm²，相对误差约 {dA/A:.2%}")

# 例 2：速度 v = s/t，s = 100.0 ± 0.5 m，t = 9.58 ± 0.01 s
s, ds = 100.0, 0.5
t, dt = 9.58, 0.01
v = s / t
rel_v = ds/s + dt/t                # 除法：相对误差相加
print(f"v = {v:.3f} ± {v*rel_v:.3f} m/s，相对误差约 {rel_v:.2%}")
```

第二例里，$t$ 的相对误差约 0.1%，$s$ 约 0.5%，加起来约 0.6%——**最快的单项相对误差决定了结果的相对误差下限，而加法让最坏情形累加。** 这也解释了为何物理实验里把最差的那个量测准，往往比追求所有量都精确更有效。

## 7 小结

- 误差传播由**一阶泰勒项统治**：$\Delta y \approx \sum_i \dfrac{\partial f}{\partial x_i}\Delta x_i$，二阶以上可忽略。
- 一元情形的放大倍率是 $|f'(x)|$，相对形式的放大倍率是条件数 $c(x)=\dfrac{|xf'(x)|}{|f(x)|}$。
- 四则运算口诀：**加减看绝对误差（相加），乘除看相对误差（相加），幂次是指数倍放大**。
- 误差限是**确定性最坏上界**，与统计意义下的随机抵消（RSS）不是一回事。
- 误差传播可以反向指导测量：要达到给定的输出精度，输入精度必须满足什么要求。

在下一节，我们把「放大倍率」提升为学科级概念：**病态问题与条件数**。当一个函数或方程组某处的条件数巨大时，任何算法都救不了它——这是「问题本身」的难度，与算法无关。
