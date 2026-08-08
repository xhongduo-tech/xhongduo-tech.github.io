---
title: 反幂法（inverse power method）
date: 2026-08-07
---

# 反幂法：用逆矩阵找最小与指定特征值

<div class="epigraph">
<p>当正向找不到时，反向往往豁然开朗。</p>
<footer>—— 反幂法的智慧</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§7.2 ｜ 2026-08-07</p>
</div>

## 为什么从反幂法开始

幂法找到的是「模最大」特征值。可工程里常要**最小特征值**（条件数 $\mathrm{cond}=\lambda_{\max}/\lambda_{\min}$！谱范数、稳定性、奇异程度都靠它）或**指定附近的特征值**。**反幂法（inverse power method）** 用 $A^{-1}$ 做幂法——因为 $A^{-1}$ 的特征值是 $\lambda_i^{-1}$，模最大的 $1/\lambda_i$ 对应 $A$ 的**模最小**特征值。把「找最小」变成「找最大」，幂法全套理论直接复用。<span class="marginnote">直觉：<strong>$A$ 把 $\mathbf{v}_i$ 放大 $\lambda_i$ 倍，$A^{-1}$ 把它放大 $1/\lambda_i$ 倍</strong>——最小的 $\lambda_i$ 在 $A^{-1}$ 眼里是最大的，于是被幂法筛出来。条件数 $\mathrm{cond}=\lambda_{\max}/\lambda_{\min}$ 需要的正是最小特征值。</span>

本节给出反幂法算法、位移版本（找指定特征值）、以及收敛性质。

## 1 算法：对 A⁻¹ 用幂法

**反幂法**：对 $A^{-1}$ 做幂法迭代：

$$
A\,\mathbf{y}^{(k)} = \mathbf{x}^{(k)} \quad\text{（解方程组，等价于 }\mathbf{y}^{(k)}=A^{-1}\mathbf{x}^{(k)}）
$$

$$
\mathbf{x}^{(k+1)} = \frac{\mathbf{y}^{(k)}}{\lVert\mathbf{y}^{(k)}\rVert}
$$

主特征值的近似 $\mu\approx\dfrac{(\mathbf{x}^{(k)})^\top A^{-1}\mathbf{x}^{(k)}}{(\mathbf{x}^{(k)})^\top\mathbf{x}^{(k)}}$，则 $A$ 的最小特征值 $\lambda_{\min}\approx\dfrac{1}{\mu}$。

**关键实现细节：绝不显式构造 $A^{-1}$**——每步解一次方程组 $A\mathbf{y}=\mathbf{x}$。做法：**先 LU 分解 $A$ 一次，每步用两次三角回代**。成本：分解 $O(n^3)$ 一次 + 每步 $O(n^2)$。

**数值例子**：$A=\begin{pmatrix}3&1\\1&3\end{pmatrix}$，特征值 $4,2$。反幂法收敛到 $\lambda_{\min}=2$。<span class="marginnote">实现要点：<strong>「用 LU 分解代替求逆」——反幂法的每步是「解方程」不是「乘逆」</strong>。这是数值线性代数的黄金规则：解方程比显式求逆又便宜又稳（见高斯-若尔当一节）。</span>

## 2 位移反幂法：找指定特征值

更强大的版本是**位移反幂法（shifted inverse iteration）**：对 $A-\sigma I$ 用反幂法。$A-\sigma I$ 的特征值是 $\lambda_i-\sigma$，其逆的特征值是 $1/(\lambda_i-\sigma)$——**模最大者对应离 $\sigma$ 最近的特征值**。

$$
(A-\sigma I)\,\mathbf{y}^{(k)} = \mathbf{x}^{(k)}, \qquad \mathbf{x}^{(k+1)} = \frac{\mathbf{y}^{(k)}}{\lVert\mathbf{y}^{(k)}\rVert}
$$

收敛到离 $\sigma$ **最近**的特征值 $\lambda$，且收敛比 $\approx\dfrac{|\lambda-\sigma|}{\text{次近}}$——**$\sigma$ 越靠近 $\lambda$，收敛越快**。这给了我们「探针」：想找哪里的特征值，就把 $\sigma$ 放在那儿。

**收敛速度**：位移反幂法对「离 $\sigma$ 最近的特征值」的收敛比是

$$
\frac{|\lambda-\sigma|}{|\lambda'-\sigma|}
$$

其中 $\lambda'$ 是次近的特征值。$\sigma\to\lambda$ 时收敛比 $\to0$——**只要位移点够近，收敛极快**（这就是瑞利商迭代三次收敛的机制）。<span class="marginnote">反幂法 + 位移是「精确定位特征值」的标准武器：<strong>先用格什戈林圆盘或粗估计知道特征值大致位置，把 $\sigma$ 放在附近，反幂法精确定位</strong>。配合瑞利商动态更新 $\sigma$，就是上一节的三次收敛瑞利商迭代。</span>

## 3 公式解析：反幂法的收敛分析

设 $A$ 特征值 $\lambda_1,\dots,\lambda_n$，要找最小 $\lambda_n$（模最小，假设 $|\lambda_n|<|\lambda_{n-1}|$）：

- **第一步，逆的特征值。** $A^{-1}$ 的特征值为 $\lambda_i^{-1}$，模最大的是 $1/|\lambda_n|$（因为 $|\lambda_n|$ 最小）。
- **第二步，收敛比。** 对 $A^{-1}$ 的幂法收敛比为 $\dfrac{|1/\lambda_{n-1}|}{|1/\lambda_n|}=\dfrac{|\lambda_n|}{|\lambda_{n-1}|}$。
- **第三步，主特征值恢复。** 幂法得 $A^{-1}$ 的主特征值 $\mu\approx1/\lambda_n$，故 $\lambda_n\approx1/\mu$。**条件数 $\mathrm{cond}(A)=\lambda_{\max}/\lambda_n$ 由此可得。**

**注意**：反幂法要求 $A$ **可逆**（$\lambda_n\neq0$）。若 $A$ 奇异（$\lambda_n=0$），$A^{-1}$ 不存在——此时「最小特征值为 0」本身就是答案，且说明 $A$ 奇异（这在「检测接近奇异」里正是我们要的）。

## 4 实现与验证

```python
import numpy as np
from scipy.linalg import lu_factor, lu_solve

def inverse_power(A, sigma=0.0, x0=None, tol=1e-10, max_iter=100):
    """反幂法（位移版）：收敛到离 σ 最近的特征值，每步解方程不显式求逆。"""
    n = A.shape[0]
    x = np.ones(n) if x0 is None else x0 / np.linalg.norm(x0)
    lu = lu_factor(A - sigma * np.eye(n))     # LU 分解一次；σ 变则需重新分解
    for _ in range(max_iter):
        y = lu_solve(lu, x)
        x_new = y / np.linalg.norm(y)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    lam = x @ A @ x / (x @ x)                 # 瑞利商恢复特征值
    return lam, x

A = np.array([[3., 1], [1, 3]])
print(inverse_power(A, sigma=0.0, x0=[1., 0.])[0])   # λ_min = 2
print(inverse_power(A, sigma=3.5, x0=[1., 0.])[0])   # 离 3.5 最近的 4
```

**验证**：$A=\begin{pmatrix}3&1\\1&3\end{pmatrix}$，$\lambda\approx2$ 得 $\lambda\approx2$（最小）；$\lambda\approx4$ 得 $\lambda\approx4$（最大，离 3.5 最近的是 4）。

**工程注意**：$\sigma$ 恰好等于某特征值时，$A-\sigma I$ 奇异，LU 分解失败——**数值上表现为分解出问题**。处理：给 $\sigma$ 加个微小扰动，或用带小位移的「反幂法修正」。

## 5 反幂法 vs 幂法：互为镜像

| 判据 | 幂法 | 反幂法 |
| --- | --- | --- |
| 迭代矩阵 | $A$ | $A^{-1}$（解方程实现） |
| 求的特征值 | 模最大 | 模最小 / 离 $\sigma$ 最近 |
| 每步成本 | $O(n^2)$（矩阵-向量乘） | $O(n^2)$（三角回代，LU 一次） |
| 用途 | 谱半径、PageRank | 条件数、稳定性、指定特征值 |

**辨析｜易错点：** 反幂法每步的「解方程」用的是**同一个 LU 分解**（$A$ 不变）——只要分解一次。但**位移反幂法**里 $\sigma$ 变化时，$A-\sigma I$ 每变一次就要**重新分解**。**「$\sigma$ 固定 = 分解一次；$\sigma$ 每步变 = 每步分解」**——这正是瑞利商迭代贵的原因。

## 6 小结

- **反幂法**：对 $A^{-1}$ 用幂法（每步解方程组而非求逆），找模最小特征值。
- **位移反幂法**：对 $A-\sigma I$ 求逆迭代，找离 $\sigma$ 最近的特征值，收敛比 $\left|\dfrac{\lambda-\sigma}{\lambda'-\sigma}\right|$。
- 收敛：$\sigma$ 越靠近目标特征值越快；$\sigma\to\lambda$ 时趋于三次（瑞利商迭代）。
- 实现铁律：**LU 分解一次 + 每步三角回代**，不显式求逆。
- 用途：最小特征值（条件数）、指定区域特征值定位、接近奇异的检测。

在下一节，我们开始为 QR 算法铺路：**豪斯霍尔德变换与约化矩阵为三对角形**——用正交反射把对称矩阵化成三对角，让特征值计算又快又稳。
