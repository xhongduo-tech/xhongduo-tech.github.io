---
title: 幂法（power method）求主特征值与主特征向量
date: 2026-08-07
---

# 幂法：反复乘矩阵，筛出主特征方向

<div class="epigraph">
<p>反复作用于同一个向量，最终留下的只有最强势的方向。</p>
<footer>—— 幂法的朴素真理</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§7.2 ｜ 2026-08-07</p>
</div>

## 为什么从幂法开始

求特征值的通用算法（QR）很贵很复杂；但很多场景只需**一个**特征值——**主特征值**（模最大的那个，spectral radius 的来源）。**幂法（power method）** 用最朴素的迭代回答这个问题：从任意向量出发，反复乘以 $A$ 并归一化。每乘一次，主特征方向的分量相对放大，其他方向被「筛掉」。它是理解特征值迭代的起点，也是 Google PageRank（主特征向量）与许多图算法的核心引擎。<span class="marginnote">幂法的直觉：把初始向量按特征向量展开 $\mathbf{x}^{(0)}=\sum_i c_i\mathbf{v}_i$，则 $A^k\mathbf{x}^{(0)}=\sum_i c_i\lambda_i^k\mathbf{v}_i$。若 $|\lambda_1|>|\lambda_2|$，则 $\lambda_1^k$ 项相对爆炸，其他项相对消失——<strong>反复乘矩阵，就是反复放大主特征方向</strong>。</span>

本节给出幂法算法、收敛分析、以及它的局限（退化情形）。

## 1 算法：迭代 + 归一化

**幂法**：给定初始向量 $\mathbf{x}^{(0)}$（通常随机，避开与 $\mathbf{v}_1$ 正交）：

$$
\mathbf{y}^{(k)} = A\mathbf{x}^{(k)}, \qquad \mathbf{x}^{(k+1)} = \frac{\mathbf{y}^{(k)}}{\lVert\mathbf{y}^{(k)}\rVert}
$$

主特征值的近似取「瑞利商」或「相邻迭代的缩放比」：

$$
\lambda_1 \approx \frac{(\mathbf{x}^{(k)})^\top A\mathbf{x}^{(k)}}{(\mathbf{x}^{(k)})^\top\mathbf{x}^{(k)}} \quad \text{（瑞利商，更稳）}
$$

或简单形式 $\lambda_1\approx\dfrac{\mathbf{y}^{(k)}_m}{\mathbf{x}^{(k)}_m}$（取某个分量比）。

**为什么归一化？** 因为不归一化，$A^k\mathbf{x}^{(0)}$ 的范数要么爆炸（$|\lambda_1|>1$）要么萎缩（$|\lambda_1|<1$）。归一化把迭代向量钉在单位球上，只让**方向**演化，数值稳定。

**数值例子**：$A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$，主特征值 3（特征向量 $(1,1)^\top$）。取 $\mathbf{x}^{(0)}=(0,1)^\top$：

- $A\mathbf{x}^{(0)}=(1,2)^\top$，归一化 $\mathbf{x}^{(1)}=(0.447,0.894)$。
- $A\mathbf{x}^{(1)}=(1.789,2.236)$，归一化 $\mathbf{x}^{(2)}=(0.625,0.781)$。
- 继续：$\mathbf{x}^{(k)}\to(0.707,0.707)=\tfrac{1}{\sqrt2}(1,1)^\top$——**方向收敛到主特征向量**。<span class="marginnote">每步方向转一点、逐步「摆正」到主特征方向。瑞利商 $\lambda^{(k)}=(\mathbf{x}^{(k)})^\top A\mathbf{x}^{(k)}$ 给出 $\lambda_1$ 的近似：$k=1$ 时 $\approx2.6$，$k=3$ 时 $\approx2.98$——<strong>方向收敛了，瑞利商也逼近主特征值</strong>。</span>

## 2 公式解析：收敛速度

设特征值 $|\lambda_1|>|\lambda_2|\ge\cdots\ge|\lambda_n|$，且初始向量含 $\mathbf{v}_1$ 分量（$c_1\neq0$）。展开 $\mathbf{x}^{(0)}=\sum_i c_i\mathbf{v}_i$：

**第一步，展开 $A^k$ 作用。** $A^k\mathbf{x}^{(0)}=\sum_i c_i\lambda_i^k\mathbf{v}_i=\lambda_1^k\left(c_1\mathbf{v}_1+\sum_{i\ge2}c_i\left(\frac{\lambda_i}{\lambda_1}\right)^k\mathbf{v}_i\right)$。
**第二步，读出收敛比。** 归一化后方向误差按 $\left|\dfrac{\lambda_2}{\lambda_1}\right|^k$ 衰减——**收敛速度由「主次特征值比」决定**。
**第三步，主特征值误差。** 瑞利商的误差 $O\left(\left|\dfrac{\lambda_2}{\lambda_1}\right|^{2k}\right)$——**瑞利商收敛快一倍**（平方速率）。

**关键数字**：$\left|\dfrac{\lambda_2}{\lambda_1}\right|=0.9$ 时，每步误差压到 90%，到 $10^{-6}$ 需约 130 步；$=0.5$ 时只需约 20 步。**主次特征值靠得越近，幂法越慢**——这就是下一节「原点平移加速」的动机。

## 3 幂法的局限与应对

**局限一：需要 $|\lambda_1|>|\lambda_2|$（严格主特征值）**。若 $|\lambda_1|=|\lambda_2|$（如 $\pm$ 一对），幂法不收敛到单一方向，会振荡。

**局限二：初始向量不得与 $\mathbf{v}_1$ 正交**（$c_1=0$）。随机初值几乎必然避免，但理论上存在风险。

**局限三：复特征值**。实矩阵若有复特征值且模最大，幂法在实数域内振荡。

| 情形 | 幂法行为 | 应对 |
| --- | --- | --- |
| $|\lambda_1|>|\lambda_2|$ | 收敛 | 正常使用 |
| $|\lambda_1|=|\lambda_2|$（$\pm$ 对） | 振荡 | 原点平移、反幂法 |
| $c_1=0$ | 不收敛到 $\mathbf{v}_1$ | 随机初值、重启 |
| 复主特征值 | 实域振荡 | 复域幂法、QR |

<span class="marginnote">工程实践：幂法通常配「随机重启」——跑若干步若瑞利商不稳，换随机初值重跑。<strong>幂法不需要存储矩阵（只要矩阵-向量乘），对稀疏大规模矩阵极友好</strong>——这是它在 PageRank 等场景不可替代的原因。</span>

## 4 实现

```python
import numpy as np

def power_method(A, x0, tol=1e-10, max_iter=1000):
    """幂法：迭代 x ← Ax/‖Ax‖，瑞利商给出主特征值。"""
    x = x0 / np.linalg.norm(x0)
    lam_old = 0.0
    for _ in range(max_iter):
        y = A @ x
        x = y / np.linalg.norm(y)           # 归一化，只让方向演化
        lam = x @ A @ x                     # 瑞利商 λ ≈ xᵀAx
        if abs(lam - lam_old) < tol:
            break
        lam_old = lam
    return x, lam

# 例：A = [[2,1],[1,2]]，主特征值 3，主特征向量 ∝ (1,1)
A = np.array([[2., 1.], [1., 2.]])
v, lam = power_method(A, np.array([0., 1.]))
print(v, lam)                               # ≈ [0.707, 0.707], 3.0
```

**工程注意**：幂法每步 $O(n^2)$（稠密）或 $O(n)$（稀疏矩阵-向量乘）。**它不碰矩阵结构，只碰矩阵-向量乘**——这是它能处理超大规模矩阵的根本原因。

## 5 幂法与后续加速的定位

幂法是「最朴素」的特征值迭代，后面都是它的升级：

**原点平移 / 瑞利商加速**：改变迭代矩阵让主次比更优（下一节）。
**反幂法**：用 $A^{-1}$ 求最小特征值（下下节）。
**QR 算法**：把幂法思想推广到全部特征值（后文）。

**辨析｜易错点：** 幂法求的是**模最大**特征值，不是「最大」（实轴上最大）特征值。若谱半径由**负特征值**决定（$|\lambda_1|=|\lambda_n|$ 但 $\lambda_n<0$），幂法收敛到的可能是负的 $\lambda_n$ 而非正的 $\lambda_1$。<span class="marginnote">一句话：<strong>「幂法对模最大、瑞利商对速度、平移对间隔」</strong>——理解幂法 = 理解「$A^k$ 筛方向」这个机制，其余全是它的工程变体。</span>

## 6 小结

- **幂法**：$\mathbf{x}\leftarrow A\mathbf{x}$ 再归一化，方向收敛到主特征向量，瑞利商给出主特征值。
- 收敛速度由 $\left|\dfrac{\lambda_2}{\lambda_1}\right|$ 决定；瑞利商收敛平方级快一倍。
- 局限：需 $|\lambda_1|>|\lambda_2|$、初值不得与 $\mathbf{v}_1$ 正交、复特征值需复域处理。
- 只做矩阵-向量乘——稀疏大规模矩阵的天然之选（PageRank 等）。
- 求的是「模最大」特征值；加速靠原点平移、反幂法、QR。

在下一节，我们给幂法提速：**幂法的加速：原点平移法与瑞利商加速**——改造迭代矩阵，让主次特征值之比更有利。
