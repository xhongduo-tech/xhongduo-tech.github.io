---
title: 用正交多项式作最小二乘拟合
date: 2026-08-07
---

# 用正交多项式作最小二乘拟合：让法方程对角化

<div class="epigraph">
<p>正交性不是一种奢侈，而是一种刚需——它把复杂问题变成一系列独立的小问题。</p>
<footer>—— 数值分析工程直觉</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§3.8 ｜ 2026-08-07</p>
</div>

## 为什么从正交多项式拟合开始

上一节反复提醒：朴素幂基 $\{1,x,\dots,x^n\}$ 下，最小二乘的 $A^\top A$ 病态，$n$ 稍大就崩。解法之一是换**正交多项式基**——离散最小二乘里的「正交」定义成**加权和为内积**。正交基下，法方程的矩阵变成对角阵，**每个系数独立求出，病态彻底消失，还能自由决定「拟合到第几阶」**。这一节把「换基」从救急招数升格为体系化的方法。<span class="marginnote">离散正交多项式（如切比雪夫多项式在等距点上的取值）与连续正交多项式是「兄弟」：连续定义用积分 $\int\rho fg$，离散定义用加权和 $\sum_i \rho_i f(x_i)g(x_i)$。换内积定义，正交性依旧成立——<strong>同一套理论在两个世界里同时有效</strong>。</span>

本节引入离散内积、用格施密特（或三项递推）构造离散正交多项式，并展示「正交基 + 最小二乘」如何把系数逐个解出。

## 1 离散内积：把「正交」定义在数据点上

给定数据点 $x_1,\dots,x_m$（未必等距），定义**离散内积**

$$
(f, g) = \sum_{i=1}^{m} \rho_i\, f(x_i)\, g(x_i)
$$

其中 $\rho_i>0$ 是点权重。诱导的 2-范数 $\lVert f\rVert_2=\sqrt{(f,f)}$ 正是「加权平方误差」。**离散内积与连续内积唯一区别：积分换成求和。** 它让「正交」这个概念落在离散数据上。

多项式族 $\{\varphi_0,\dots,\varphi_n\}$（$\varphi_k$ 次数恰为 $k$）在数据点上**两两正交**，指

$$
(\varphi_j, \varphi_k) = \sum_{i=1}^{m}\rho_i\,\varphi_j(x_i)\,\varphi_k(x_i) = 0, \qquad j\neq k
$$

注意：要求 $m\ge n+1$ 且数据点不能退化（否则离散内积可能半正定）。<span class="marginnote">与连续情形的区别：离散正交性是「在采样点上的正交」，不是「在整个区间上的正交」。但最小二乘关心的恰恰是采样点上的平方误差——<strong>离散正交性正好服务离散目标</strong>，天作之合。</span>

## 2 构造：三项递推生成离散正交多项式

与连续情形一样，离散正交多项式满足三项递推。可以像格施密特一样逐阶正交化，但更优雅的是直接递推。设 $\varphi_k$ 是次数为 $k$ 的离散正交多项式，则

$$
\varphi_{k+1}(x) = (x - \alpha_k)\,\varphi_k(x) - \beta_k\,\varphi_{k-1}(x)
$$

其中系数由正交性确定：$\alpha_k=\dfrac{(x\varphi_k,\varphi_k)}{(\varphi_k,\varphi_k)}$，$\beta_k=\dfrac{(\varphi_k,\varphi_k)}{(\varphi_{k-1},\varphi_{k-1})}$。

**关键事实：$\alpha_k$ 是「正交多项式自身不动点」的均值，$\beta_k$ 是相邻两阶范数的比值**——它们只要算内积就能得到。由 $\varphi_{-1}=0,\varphi_0=1$ 出发，迭代生成全部 $\varphi_k$。<span class="marginnote">这个递推对等距点 + 权 $\rho_i=1$ 就给出「离散切比雪夫多项式」；对连续区间 $[-1,1]$ 给勒让德。同一套递推模板、不同内积，生成不同家族——<strong>「三项递推 + 正交性」是正交多项式世界的宪法</strong>。</span>

Python 示意（离散正交基生成后求系数）：

```python
import numpy as np

def discrete_ortho_basis(x, rho, deg):
    """三项递推生成离散正交多项式：返回 φ_0..φ_deg 在各数据点上的取值。"""
    phis = []
    phi_prev = np.zeros(len(x))          # φ_{-1} = 0
    phi = np.ones(len(x))                # φ_0 = 1
    for k in range(deg + 1):
        phis.append(phi)
        alpha = np.sum(rho * x * phi * phi) / np.sum(rho * phi * phi)
        beta = (np.sum(rho * phi * phi) / np.sum(rho * phi_prev * phi_prev)) if k > 0 else 0.0
        phi_next = (x - alpha) * phi - beta * phi_prev
        phi_prev, phi = phi, phi_next
    return phis

# 数据点：y = 1 + 2x（无噪声），正交基下应精确恢复 a_0 = 4, a_1 = 2
x = np.array([0., 1., 2., 3.])
y = np.array([1., 3., 5., 7.])
rho = np.ones_like(x)
for k, phi in enumerate(discrete_ortho_basis(x, rho, 1)):
    a = np.sum(rho * y * phi) / np.sum(rho * phi * phi)   # 独立除法
    print(f"a_{k} = {a}")                 # 4.0, 2.0
```

## 3 公式解析：正交基下最小二乘的系数

设 $\{\varphi_0,\dots,\varphi_n\}$ 是数据点上的离散正交多项式，拟合函数写为

$$
\varphi(x) = a_0\varphi_0(x) + a_1\varphi_1(x) + \cdots + a_n\varphi_n(x)
$$

最小二乘目标 $J=\sum_i\rho_i[\varphi(x_i)-y_i]^2$。由正交性，法方程矩阵对角化：

- **第一步，写正交条件。** 残差与每个基函数正交：$\sum_i\rho_i[\varphi(x_i)-y_i]\varphi_j(x_i)=0$，即 $(a_j\varphi_j-y,\varphi_j)=0$——求和展开后，$k\neq j$ 项全部消失。
- **第二步，解出系数。** $a_j=\dfrac{(y,\varphi_j)}{(\varphi_j,\varphi_j)}=\dfrac{\sum_i\rho_i y_i\varphi_j(x_i)}{\sum_i\rho_i\varphi_j^2(x_i)}$。**每个系数独立，一次除法搞定。**
- **第三步，误差可分解。** 拟合后平方误差

$$
J_{\min} = (y,y) - \sum_{k=0}^{n} a_k^2\,(\varphi_k,\varphi_k)
$$

**每加入一阶，误差就减少 $a_k^2(\varphi_k,\varphi_k)$——误差「逐阶贡献」清晰可见。** 这给了你决定「拟合到几阶」的判据：加一阶，若误差减少可忽略，就停。

**这就是「正交多项式拟合」的全部优势：$O(n)$ 个独立除法替代 $O(n^3)$ 的方程组求解，且无病态。** 你甚至可以先拟到 3 阶看效果，再追加到 5 阶——旧的 $a_0,a_1,a_2$ 一个都不用重算。

## 4 离散正交 vs 连续正交：选谁

| 判据 | 连续正交多项式（勒让德等） | 离散正交多项式（离散切比雪夫等） |
| --- | --- | --- |
| 内积 | $\int_a^b\rho fg\,dx$ | $\sum_i\rho_if(x_i)g(x_i)$ |
| 服务对象 | 连续函数逼近、高斯求积 | 离散数据拟合 |
| 节点要求 | 无 | $m\ge n+1$，数据点不退化 |
| 典型用途 | 最佳平方逼近、谱方法 | 等距采样的最小二乘 |

**辨析｜易错点：** 在等距数据点上，用「连续勒让德多项式取值」作拟合基，**不保证离散正交**——离散内积下它们不正交，法方程又病态了。**离散拟合必须用离散正交多项式，或用 QR/SVD 直接解**。连续正交多项式用于连续逼近与求积，二者不能随意互换。<span class="marginnote">一句提醒：<strong>「连续的归连续，离散的归离散」</strong>——内积定义换世界，正交性不自动迁移。查到「勒让德拟合」时要先确认实现里用的是离散正交基还是连续取值，两者结果不同。</span>

## 5 小结

- **离散内积** $(f,g)=\sum_i\rho_if(x_i)g(x_i)$：把正交性定义在数据点上，服务离散最小二乘。
- 离散正交多项式满足**三项递推**，系数 $\alpha_k,\beta_k$ 由内积确定，$\varphi_{-1}=0,\varphi_0=1$ 起步。
- **正交基下最小二乘系数解耦**：$a_j=\dfrac{(y,\varphi_j)}{(\varphi_j,\varphi_j)}$，一次除法一个系数，无病态。
- 平方误差**逐阶可分解**：$J_{\min}=(y,y)-\sum_ka_k^2(\varphi_k,\varphi_k)$，据此决定拟合阶数。
- 连续正交与离散正交**内积不同、不可互换**：离散拟合用离散正交基或 QR/SVD。

在下一节，我们把最小二乘的「基」抽象到最一般的形式：**矛盾方程组与线性最小二乘问题**——用矩阵语言统一处理「方程比未知数多」的所有情形，并引入 QR 分解与正规方程的正规解。
