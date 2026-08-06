---
title: 病态问题与条件数
date: 2026-08-07
---

# 病态问题与条件数：问题的固有难度

<div class="epigraph">
<p>使用不充分的数据所造成的错误，远小于完全没有数据的错误。</p>
<footer>—— 查尔斯 · 巴贝奇（Charles Babbage）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§1.3.2 ｜ 2026-08-07</p>
</div>

## 为什么从病态问题开始

前三节我们把误差的来源、度量和传播都讲清楚了。误差传播公式给出的放大倍率——条件数——如今要「升级」成一个独立的主角：**病态问题（ill-conditioned problem）**。它是数值分析里最反直觉也最重要的概念之一：某些问题，无论你用多好的算法、多高的精度，只要输入有一丁点扰动，输出就会面目全非。这是**问题本身**的固有难度，不怪算法，也不怪计算机。<span class="marginnote">巴贝奇（1791—1871）是差分机的发明者、现代计算机的先驱，这句话今天听来依然贴切：数据不精确会带来误差，但没有数据根本无法计算。病态问题提醒我们另一面——当数据本身很不可靠时，就算算法完美，输出也照样不可信。</span>

上一节《[函数运算的误差估计](./error-propagation)》已经出现过一元函数的条件数 $c(x)=\dfrac{|xf'(x)|}{|f(x)|}$。本节把它放到三个层次：先看直觉，再看一元函数的条件数，最后看数值分析里最经典的病态舞台——**线性方程组**，并引入矩阵条件数 $\mathrm{cond}(A)$。它是判断「一个方程组好不好解」的第一指标。

## 1 输入的小扰动，输出的大地震

先看一个一元函数的例子。设 $f(x) = \log x$，在 $x=1$ 附近做一次计算。条件数 $c(x)=\dfrac{|x\cdot \frac1x|}{|\log x|}=\dfrac{1}{|\log x|}$，在 $x\to 1$ 时趋于无穷——**$x=1$ 处是病态的**。

具体地，若 $x$ 有 $10^{-6}$ 的相对误差，即 $x = 1 + 10^{-6}$，则 $f(x)=\log(1+10^{-6}) \approx 10^{-6}$，输出的相对误差接近 100%。输入相对误差 $10^{-6}$，输出相对误差却接近 1——放大了约一百万倍。用任何算法都救不了：$\log x$ 在 $x=1$ 附近本来就「几乎等于零」，任何输入的微小晃动都会让输出翻天覆地。

再回看绪论《[数值分析的研究对象与特点](./what-is-numerical-analysis)》里那个线性方程组的例子：

$$
\begin{cases}
x + 1000y = 1000 \\
x + 999y = 999
\end{cases}
$$

把第二个方程右端从 `999` 改成 `999.5`，解从 $(x,y)=(0,1)$ 跳到 $(500,0.5)$——**输入扰动 $5\times10^{-4}$，解的变化接近无穷大比例**。这类问题就是病态问题：系数矩阵本身把误差放大了。

**定义。** **病态问题（ill-conditioned problem）**：输入数据发生微小扰动，就引起解的数值产生巨大变化的问题。与之相对的是**良态问题（well-conditioned problem）**，输入的小扰动只会引起解的小变化。

## 2 一元函数的条件数：c(x) 怎么读

判断一元问题 $y=f(x)$ 在 $x$ 处是否病态，看条件数 $c(x)=\dfrac{|xf'(x)|}{|f(x)|}$。它可以直接从误差传播公式里「读」出来：$c(x)$ 是输入相对误差到输出相对误差的放大倍率。

几个常见函数的条件数：

| 函数 | 条件数 $c(x)$ | 解读 |
| --- | --- | --- |
| $f(x)=x^p$ | $\lvert p\rvert$ | 幂次是放大倍数，开方（$p=\tfrac12$）压缩 |
| $f(x)=e^x$ | $\lvert x\rvert$ | $\lvert x\rvert$ 不大时很温和 |
| $f(x)=\log x$ | $1/\lvert\log x\rvert$ | $x\to 1$ 时趋于无穷，$x=1$ 处病态 |
| $f(x)=\sin x$ | $\lvert x\cot x\rvert$ | 在 $\sin x=0$ 的零点附近病态 |

**辨析｜易错点：** 条件数大 ≠ 函数值大。$c(x)$ 度量的是**相对变化的放大**，不是函数值本身。$\log x$ 在 $x=1$ 处函数值很小，于是「小输入相对误差 + 微小输出」造成巨大相对放大；而 $f(x)=10^6 x$ 函数值很大，但 $c(x)=1$，完全良态。**判断病态只看放大倍率，不看函数值大小。**

## 3 线性方程组的条件数：cond(A) 的定义

数值分析里最经典的病态舞台是线性方程组 $Ax = b$。当右端项 $b$ 带有扰动 $\Delta b$（例如测量误差），解 $x$ 会随之扰动 $\Delta x$。方程变为

$$
A(x+\Delta x) = b + \Delta b
$$

两式相减得 $A\Delta x = \Delta b$，即 $\Delta x = A^{-1}\Delta b$。直觉上，解对扰动的敏感程度由 $A^{-1}$ 的「大小」决定——$A^{-1}$ 越大，同样的 $\Delta b$ 造成的 $\Delta x$ 越大。

为此需要度量「矩阵的大小」。**矩阵范数（matrix norm）** $\lVert A\rVert$ 是向量范数在矩阵上的推广，直观理解为「$A$ 能把单位向量放大到多长」。本专题《向量范数与矩阵范数》一节会严格定义，这里先用 2-范数（谱范数）的直觉。<span class="marginnote">向量范数给出向量的「长度」，矩阵范数给出矩阵的「最大放大率」。两个常用事实：$\lVert I\rVert=1$，且 $\lVert AB\rVert\le\lVert A\rVert\lVert B\rVert$——这两条性质正好用来推导下面的误差界。</span>

**矩阵条件数（condition number of a matrix）** 定义为

$$
\mathrm{cond}(A) = \lVert A\rVert\,\lVert A^{-1}\rVert
$$

它满足 $\mathrm{cond}(A) \ge 1$，因为 $I = AA^{-1}$ 两边取范数得 $1=\lVert I\rVert\le\lVert A\rVert\lVert A^{-1}\rVert=\mathrm{cond}(A)$。$\mathrm{cond}(A)$ 越大，方程组越病态。

## 4 公式解析：相对误差界 cond(A)·‖Δb‖/‖b‖

为什么 $\mathrm{cond}(A)$ 是「放大倍率」？推导一条著名的不等式：

$$
\frac{\lVert \Delta x\rVert}{\lVert x\rVert} \;\le\; \mathrm{cond}(A)\,\frac{\lVert \Delta b\rVert}{\lVert b\rVert}
$$

- **第一步，由 $A\Delta x=\Delta b$ 放大。** 取范数并利用 $\lVert AB\rVert\le\lVert A\rVert\lVert B\rVert$：

$$
\lVert \Delta x\rVert = \lVert A^{-1}\Delta b\rVert \le \lVert A^{-1}\rVert\,\lVert\Delta b\rVert
$$

- **第二步，把「分母」放过来。** 由 $b=Ax$ 得 $\lVert b\rVert\le\lVert A\rVert\lVert x\rVert$，即 $\dfrac{1}{\lVert x\rVert}\le\dfrac{\lVert A\rVert}{\lVert b\rVert}$。
- **第三步，两式相乘。** 

$$
\frac{\lVert \Delta x\rVert}{\lVert x\rVert} \;\le\; \lVert A^{-1}\rVert\lVert\Delta b\rVert\cdot\frac{\lVert A\rVert}{\lVert b\rVert} \;=\; \mathrm{cond}(A)\,\frac{\lVert\Delta b\rVert}{\lVert b\rVert}
$$

这条不等式把「解的相对误差」和「输入的相对误差」用 $\mathrm{cond}(A)$ 直接挂钩：**右端项相对误差每有 $10^{-k}$，解的相对误差最多达到 $\mathrm{cond}(A)\times 10^{-k}$。** 反过来，若要求解有 $s$ 位有效数字，就要求 $\mathrm{cond}(A)\times$ 输入相对误差 $< 10^{-s}$——这是判断「这道题在双精度下还做不做得动」的标尺。

需要强调的是：这条界是**最坏情形的上界**，实际误差通常比它小；但它是「保证不会超过」的界，工程上据此分配精度预算。<span class="marginnote">如果扰动同时发生在 $A$ 与 $b$，界的形式会稍复杂，但思想相同：$\mathrm{cond}(A)$ 依然是主导因子。判断实际问题时，通常先估 $\mathrm{cond}(A)$ 的数量级——它超过 $10^{15}$ 时，双精度下解基本上已经「全是噪声」。</span>

## 5 希尔伯特矩阵实验：病态不是传说

光看公式还不够，来做一个数值实验。**希尔伯特矩阵（Hilbert matrix）** $H_n$ 的元素是 $h_{ij}=\dfrac{1}{i+j-1}$，是教科书里最著名的病态矩阵。用 Python 看它的条件数随阶数增长多快：

```python
import numpy as np

def hilbert(n):
    return np.array([[1.0 / (i + j - 1) for j in range(1, n + 1)]
                     for i in range(1, n + 1)])

for n in (3, 5, 8, 10, 12):
    print(f"n = {n:2d}: cond(H) = {np.linalg.cond(hilbert(n)):.3e}")
```

运行结果（2-范数条件数）：

| n | 3 | 5 | 8 | 10 | 12 |
| --- | --- | --- | --- | --- | --- |
| cond(H) | ≈ 5×10² | ≈ 4.8×10⁵ | ≈ 1.5×10¹⁰ | ≈ 1.6×10¹³ | ≈ 1.6×10¹⁶ |

再看实际求解：取 $n=10$，令真解 $x=(1,\dots,1)^\top$，右端 $b=Hx$。把 $b$ 的第一个分量扰动 $10^{-7}$，再解一次：

```python
n = 10
H = hilbert(n)
x_true = np.ones(n)
b = H @ x_true
x1 = np.linalg.solve(H, b)

b2 = b.copy(); b2[0] += 1e-7
x2 = np.linalg.solve(H, b2)

print("未扰动解的误差：", np.linalg.norm(x1 - x_true) / np.linalg.norm(x_true))
print("扰动 1e-7 后解的误差：", np.linalg.norm(x2 - x_true) / np.linalg.norm(x_true))
```

$n=10$ 时 $\mathrm{cond}(H)\approx 1.6\times10^{13}$，输入的相对扰动约 $10^{-7}$，按误差界，解的相对误差最多可达 $10^6$ 量级——实际跑出来，扰动后的解与真解已是「毫无关系」的数值。**这不是算法不行，而是问题本身病态：数据稍微抖一下，解就散架。** 数值分析对这类问题的答案是：先算条件数，若病态严重，就换问题表述（如预条件）或接受低精度——这在第三级《深度学习》里对应「损失函数病态时训练震荡」的现象。

## 6 辨析：病态是问题的，不稳定是算法的

这一对概念最容易混淆，必须钉死：

- **病态（ill-conditioned）** 是**问题本身**的性质，用条件数衡量，与用什么算法无关。病态问题配上最完美的算法，解对输入扰动依然敏感。
- **不稳定（unstable）** 是**算法**的性质，用误差是否在计算中被放大来衡量。良态问题配上不稳定的算法，照样算出一堆垃圾——例如教材里著名的「不稳定递推公式」$I_n = 1 - nI_{n-1}$：正向递推时，初始舍入误差每步放大 $n$ 倍，几步就面目全非。<span class="marginnote">判断口诀：<strong>「问题病不病」看条件数，「算法稳不稳」看误差放大。</strong> 绪论《数值分析的研究对象与特点》里的 2×2 分类值得自己重画一遍：只有「良态 + 稳定」的组合才能期待精确结果。</span>

此外，条件数的取值依赖范数选择（1-、2-、∞-范数之间最多差一个与维数有关的常数因子），所以工程上更关注**数量级**而非精确值。$\mathrm{cond}(A)\sim 10^{2}$ 是「很良态」，$10^{13}$ 是「病态」，$10^{16}$ 在双精度下基本是「算不出来了」。

## 7 小结

- **病态问题**是问题本身的固有难度：输入微小扰动导致解巨大变化，用条件数衡量。
- 一元函数条件数 $c(x)=\dfrac{|xf'(x)|}{|f(x)|}$：大于 1 放大误差，小于 1 压缩误差；$\log x$ 在 $x=1$ 处、$\sin x$ 在零点处病态。
- 线性方程组的条件数 $\mathrm{cond}(A)=\lVert A\rVert\lVert A^{-1}\rVert\ge 1$，并有误差界 $\dfrac{\lVert\Delta x\rVert}{\lVert x\rVert}\le \mathrm{cond}(A)\dfrac{\lVert\Delta b\rVert}{\lVert b\rVert}$。
- 希尔伯特矩阵 $H_n$ 的条件数随 $n$ 爆炸式增长，$n=10$ 时约 $10^{13}$，是经典病态实例。
- **病态 ≠ 不稳定**：病态是问题的（看条件数），不稳定是算法的（看误差放大）；只有「良态 + 稳定」才可靠。

在下一节，我们把目光从「问题有多难」转向「算法怎么选」：**算法的数值稳定性**——同样的病态程度下，为什么有些算法误差越传越大，另一些却能把误差压住，以及避免误差危害的若干工程原则。
