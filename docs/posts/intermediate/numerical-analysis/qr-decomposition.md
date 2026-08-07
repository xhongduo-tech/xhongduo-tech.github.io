---
title: 矩阵的 QR 分解
date: 2026-08-07
---

# QR 分解：正交化在数值世界的完全体

<div class="epigraph">
<p>把一个矩阵分解成正交部分与三角部分，无数问题从此迎刃而解。</p>
<footer>—— QR 分解的工程哲学</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§7.3 ｜ 2026-08-07</p>
</div>

## 为什么从 QR 分解开始

**QR 分解** 把矩阵分解为

$$
A = Q\,R
$$

其中 $Q$ 是**列正交**矩阵（$Q^\top Q=I$）、$R$ 是**上三角**矩阵。它是数值线性代数的「万金油」：解最小二乘（最稳定的路径）、求特征值的 QR 算法（下一节）、以及正交化（Gram-Schmidt 的数值稳定版）。理解 QR 分解 = 理解「正交化」与「三角形化」的统一。<span class="marginnote">QR 分解的意义：<strong>$Q$ 承载「方向」（正交、稳定），$R$ 承载「尺度」（三角、可回代）</strong>。最小二乘里 $A^\top A$ 的病态被拆开——条件数不再平方恶化。特征值算法里 $A_k=Q_kR_k$ 的迭代最终把 $A$ 逼近对角/三角——QR 分解是那个迭代的引擎。</span>

本节给出 QR 分解的三种算法、数值性质与最小二乘应用。

## 1 QR 分解的三种算法

**（1）经典 Gram-Schmidt（不推荐）**：对列做正交化，$q_i=\dfrac{v_i}{\lVert v_i\rVert}$，$v_i=a_i-\sum_{j<i}(a_i,q_j)q_j$。**数值不稳定**——列向量接近相关时，正交性迅速丢失（经典的「两次正交化反而更差」）。理论上 $O(n^3)$，数值上是坑。

**（2）修正 Gram-Schmidt（MGS）**：逐列更新剩余向量，每步正交化一次。**数值上明显更稳**，仍是 $O(n^3)$，但工程上可作为轻量实现。

**（3）豪斯霍尔德 QR（Householder QR，标准）**：用豪斯霍尔德反射逐列「杀零」。$H_1,H_2,\dots,H_n$ 序列作用：

$$
H_n\cdots H_1 A = R, \qquad Q = (H_n\cdots H_1)^\top
$$

**数值最稳**，是 LAPACK/`numpy.linalg.qr` 的标准实现。<span class="marginnote">三种算法的对比是「教科书陷阱」的经典：<strong>理论等价的算法，数值差异天壤之别</strong>。经典 Gram-Schmidt 会丢失正交性，MGS 好些，豪斯霍尔德最稳——因为豪斯霍尔德每一步都是正交反射（条件数 1），不放大误差。工程上无条件选豪斯霍尔德。</span>

## 2 公式解析：豪斯霍尔德 QR 的流程

对 $A\in\mathbb{R}^{m\times n}$（$m\ge n$）：

- **第一步，第 1 列。** 构造 $H_1$ 使 $H_1\mathbf{a}_1=\pm\lVert\mathbf{a}_1\rVert\mathbf{e}_1$（上一节的本领），于是 $H_1A$ 的第一列只剩首个非零。
- **第二步，收缩子矩阵。** 对右下 $(m-1)\times(n-1)$ 子矩阵重复，构造 $H_2$……共 $n$ 步（或 $n-1$ 步）。
- **第三步，组装。** $H_n\cdots H_1A=R$（上三角），$Q=H_1^\top\cdots H_n^\top$。注意 $H$ 对称，$Q=H_1\cdots H_n$。

**数值性质**：

$$
\lVert A - \hat{Q}\hat{R}\rVert_F \le c\,mn\,\epsilon_{\mathrm{mach}}\,\lVert A\rVert_F
$$

QR 分解的误差被控制在 $A$ 规模的比例内——**不需要主元、无条件稳定**。这是它相对 LU（需选主元）的理论优势，也是最小二乘里「QR 路径优于法方程」的根源。

## 3 QR 在最小二乘中的应用

回顾矛盾方程组一章：$A\mathbf{x}\approx\mathbf{b}$（$m>n$）。QR 路径：

$$
A=QR \Rightarrow \lVert\mathbf{b}-A\mathbf{x}\rVert_2^2 = \lVert Q^\top\mathbf{b}-R\mathbf{x}\rVert_2^2
$$

由于 $Q$ 正交，范数不变；$\lVert Q^\top\mathbf{b}-R\mathbf{x}\rVert^2$ 分解成「可最小化部分」与「不可变部分」。最优解满足 $R\mathbf{x}^*=Q^\top\mathbf{b}$——**上三角回代即可**。

**对比**：

| 路径 | 条件数 | 稳定性 |
| --- | --- | --- |
| 法方程 $A^\top A$ | $\mathrm{cond}(A)^2$ | 病态恶化 |
| **QR 路径** | $\mathrm{cond}(A)$ | 稳定，标准选择 |
| SVD | $\mathrm{cond}(A)$ | 最稳，可诊断秩亏 |

**工程结论：最小二乘默认 QR 或 SVD，永远别手写法方程**（前面已强调，QR 是那个「正确选择」的具体实现）。

## 4 QR 与 LU：两种分解的对照

| 判据 | LU 分解 | QR 分解 |
| --- | --- | --- |
| 分解 | $PA=LU$ | $A=QR$ |
| 因子性质 | 三角（非正交） | $Q$ 正交 + 上三角 |
| 稳定性 | 需列主元 | **无条件稳定** |
| 求解成本 | $O(n^3)$ | $O(n^3)$（常数稍大） |
| 用途 | 求解方程组 | 最小二乘、特征值、正交化 |

**QR 比 LU 稳，但 LU 比 QR 快**（常数因子）——纯求解方程组用 LU，最小二乘/特征值用 QR。<span class="marginnote">一句话：<strong>「解方程找 LU，最小二乘与特征值找 QR」</strong>——稳定性与速度的权衡，让两种分解各有领地。`numpy.linalg.solve` 走 LU，`numpy.linalg.lstsq` 走 QR/SVD，正是这个分工。</span>

## 5 实现

```python
import numpy as np

A = np.array([[1., 1.], [1., 2.], [1., 3.]])   # 线性拟合设计矩阵
b = np.array([1., 2.5, 3.8])

Q, R = np.linalg.qr(A)                          # 豪斯霍尔德 QR
x = np.linalg.solve(R, Q.T @ b)                 # 上三角回代
print(x)                                        # 拟合系数 a0, a1

# 对比 lstsq（内部也是 QR/SVD）
print(np.linalg.lstsq(A, b, rcond=None)[0])
```

**工程注意**：`np.linalg.qr` 有 `mode='reduced'`（默认，$Q\in\mathbb{R}^{m\times n}$）与 `mode='complete'` 之分；最小二乘用 reduced 即可。

## 6 小结

- **QR 分解** $A=QR$：$Q$ 列正交、$R$ 上三角；三种算法中**豪斯霍尔德版最稳**（正交反射，条件数 1）。
- 流程：逐列豪斯霍尔德杀零 → $R$ 成形 → $Q$ 由反射的转置组装。
- QR 无条件稳定（无需主元），误差被控制在 $A$ 规模比例内。
- 最小二乘的 QR 路径：$R\mathbf{x}=Q^\top\mathbf{b}$ 回代，条件数 $\mathrm{cond}(A)$（非平方）。
- LU 快、QR 稳：解方程用 LU，最小二乘/特征值用 QR。

在下一节，我们把 QR 分解变成特征值算法：**基本 QR 算法及其收敛性**——反复 $A=QR$、$A\leftarrow RQ$，让矩阵逼近 Schur 形，特征值浮出水面。
