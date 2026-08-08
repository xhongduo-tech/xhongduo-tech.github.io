---
title: 三对角方程组的追赶法（Thomas algorithm）
date: 2026-08-07
---

# 追赶法：O(n) 解三对角方程组

<div class="epigraph">
<p>结构的价值，在于把一般问题变成特殊问题的叠加。</p>
<footer>—— 数值线性代数的结构主义</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§5.4 ｜ 2026-08-07</p>
</div>

## 为什么从追赶法开始

现实中大量方程组是**三对角**的：有限差分解常微分/偏微分方程、三次样条的三弯矩方程组、隐式欧拉方法解热传导方程——它们的矩阵只在主对角线与两条次对角线上有非零元素。对这种结构，通用 LU 分解的 $O(n^3)$ 是巨大浪费；**追赶法（Thomas algorithm）** 利用三对角结构把复杂度压到 $O(n)$，是数值分析里「结构换效率」的典范。<span class="marginnote">三对角方程组出现的地方，往往就是「离散化之后的微分方程」——后文常微分方程数值解法里的隐式格式、偏微分方程的隐式离散，都会反复遇见它。<strong>追赶法也是 LU 分解的三对角特例：$L$ 双对角、$U$ 双对角，加起来还是 $O(n)$ 存贮</strong>。</span>

本节给出追赶法的推导、实现与稳定性条件。

## 1 问题与分解思想

三对角方程组：

$$
\begin{cases}
b_1x_1 + c_1x_2 = d_1 \\
a_ix_{i-1} + b_ix_i + c_ix_{i+1} = d_i, \quad i=2,\dots,n-1 \\
a_nx_{n-1} + b_nx_n = d_n
\end{cases}
$$

矩阵 $A=\mathrm{tridiag}(a,b,c)$。追赶法的思路：**对 $A$ 做 LU 分解，但只分解到双对角**。因为 $A$ 三对角，其 LU 因子保持带状：$L$ 是下双对角（主对角 1 + 一条次对角 $\gamma_i$），$U$ 是上双对角（主对角 $\beta_i$ + 一条次对角 $c_i$）。

$$
A = \underbrace{\begin{pmatrix}1&&&&\\ \gamma_2&1&&&\\ &\gamma_3&1&&\\ &&\ddots&\ddots&\\ &&&\gamma_n&1\end{pmatrix}}_{L} \underbrace{\begin{pmatrix}\beta_1&c_1&&&\\ &\beta_2&c_2&&\\ &&\ddots&\ddots&\\ &&&\beta_{n-1}&c_{n-1}\\ &&&&\beta_n\end{pmatrix}}_{U}
$$

**带状结构被 LU 保持**——这是追赶法 $O(n)$ 的结构根源。

## 2 公式解析：追赶法的两趟扫描

由矩阵乘法匹配 $A=LU$，逐行得到：

**第 1 趟：追（forward elimination，消去）**——算 $\beta_i$ 与 $\gamma_i$：

$$
\beta_1 = b_1
$$

$$
\gamma_i = \frac{a_i}{\beta_{i-1}}, \qquad \beta_i = b_i - \gamma_i c_{i-1}, \qquad i=2,\dots,n
$$

同时把右端 $d_i$ 同步消去：$\tilde{d}_1=d_1$，$\tilde{d}_i=d_i-\gamma_i\tilde{d}_{i-1}$。

**第 2 趟：赶（back substitution，回代）**——解 $U\mathbf{x}=\tilde{\mathbf{d}}$：

$$
x_n = \frac{\tilde{d}_n}{\beta_n}, \qquad x_i = \frac{\tilde{d}_i - c_ix_{i+1}}{\beta_i}, \quad i=n-1,\dots,1
$$

**为什么叫「追赶」**：追 = 从前往后消，把系数 $\gamma_i,\beta_i$ 和右端更新一遍；赶 = 从后往前代，把解逐个赶出来。两趟各 $O(n)$，总复杂度 $O(n)$，存贮只三个向量。<span class="marginnote">对照通用 LU：追赶法的 $\beta_i,\gamma_i$ 就是 LU 的 $u_{ii}$ 与 $\ell_{i,i-1}$——只是三对角结构让每行只需一个乘子。理解它 = 理解 LU 的「带状版本」。</span>

```python
import numpy as np

def thomas(a, b, c, d):
    """追赶法：O(n) 解三对角方程组。b 主对角，a、c 两条次对角（长 n-1）。"""
    n = len(b)
    beta, d_tilde, x = np.zeros(n), np.zeros(n), np.zeros(n)
    # 追：前向消去
    beta[0], d_tilde[0] = b[0], d[0]
    for i in range(1, n):
        gamma = a[i-1] / beta[i-1]
        beta[i] = b[i] - gamma * c[i-1]
        d_tilde[i] = d[i] - gamma * d_tilde[i-1]
    # 赶：后向回代
    x[-1] = d_tilde[-1] / beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d_tilde[i] - c[i] * x[i+1]) / beta[i]
    return x

# 例：2x1-x2=1, -x1+2x2-x3=0, -x2+2x3=1 → (1,1,1)
print(thomas([-1, -1], [2, 2, 2], [-1, -1], [1, 0, 1]))
```

**数值例子**：解
$$
\begin{cases}
2x_1 - x_2 = 1 \\
-x_1 + 2x_2 - x_3 = 0 \\
-x_2 + 2x_3 = 1
\end{cases}
$$
$a=(-1,-1)$，$b=(2,2,2)$，$c=(-1,-1)$，$d=(1,0,1)$。追：$\beta_1=2$，$\gamma_2=-1/2$，$\beta_2=2-(-1/2)(-1)=3/2$，$\tilde d_2=0-(-1/2)(1)=1/2$；$\gamma_3=-1/(3/2)=-2/3$，$\beta_3=2-(-2/3)(-1)=4/3$，$\tilde d_3=1-(-2/3)(1/2)=4/3$。赶：$x_3=1$，$x_2=(1/2-(-1)(1))/(3/2)=1$，$x_1=(1-(-1)(1))/2=1$。**解 $(1,1,1)$**，代入验证正确。<span class="marginnote">这个例子是「对称正定三对角」的标准形态：追赶法里 $\beta_i$ 全部保持正数且不接近零——<strong>对角占优的三对角矩阵是追赶法数值稳定的天然温床</strong>。</span>

## 3 稳定性条件：对角占优

追赶法何时数值稳定？关键在于除法 $\beta_{i-1}$ 不能太小。保证稳定的充分条件是：

**严格对角占优**：$|b_i|>|a_i|+|c_i|$（对全部 $i$，端点放宽）。
或 $A$ **对称正定**（此时 $\beta_i>0$ 且消去不放大误差）。

对角占优保证 $\beta_i$ 不为零且有界，追赶法稳定。物理问题离散化（热传导、梁弯曲）天然满足这个条件——所以追赶法在工程里「基本不会翻车」。<span class="marginnote">若矩阵不满足条件，追赶法仍可能算得通，但无法保证稳定——可能发生「$\beta_i$ 接近零导致误差爆炸」。工程上先检查对角占优；不满足就换列主元 LU。</span>

## 4 追赶法与通用 LU 的对照

| 判据 | 追赶法 | 通用 LU |
| --- | --- | --- |
| 复杂度 | $O(n)$ | $O(n^3)$ |
| 存贮 | 三个向量 | 满矩阵 |
| 适用 | 三对角 | 一般矩阵 |
| 稳定性 | 对角占优下稳定 | 列主元下稳定 |
| 典型来源 | 样条、差分、隐式 ODE | 一切 |

**辨析｜易错点：** 追赶法**不是**独立的算法，而是 **LU 分解的三对角特化**。把追赶法当成「另一种方法」来背会失去理解；记住它是「LU 在带状结构上的收缩」，一切性质（存在性看主子式、稳定性看主元）都能从 LU 推出来。<span class="marginnote">更大的视野：追赶法是「带状 LU 分解」最简特例。带宽更大（五对角等）有对应的带状 LU 算法，复杂度 $O(n\times\text{带宽}^2)$。<strong>「利用结构」是高性能计算的第一原则</strong>——第三级《高性能计算》还会反复回到这个母题。</span>

## 5 小结

- **三对角方程组**：$Ax=d$，$A=\mathrm{tridiag}(a,b,c)$；有限差分、三次样条、隐式 ODE 的标配。
- **追赶法**：追（前向消去算 $\gamma_i,\beta_i$）+ 赶（后向回代解 $x_i$），两趟各 $O(n)$。
- **追赶法 = 三对角 LU 分解**：$L$、$U$ 都是双对角，结构被保持。
- 稳定性：严格对角占优或对称正定保证稳定；不满足时换列主元。
- 结构换效率：$O(n^3)\to O(n)$，是「带状 LU」的最简代表。

在下一节，我们把 LU 分解应用到另一类特殊矩阵：**对称正定矩阵的平方根法（Cholesky 分解）**——利用对称性把分解成本减半，$A=LL^\top$。
