---
title: 对称正定矩阵的平方根法（Cholesky 分解）
date: 2026-08-07
---

# Cholesky 分解：对称性白送的半价分解

<div class="epigraph">
<p>结构是免费的午餐——对称性让你付一半钱，拿两倍效率。</p>
<footer>—— 数值线性代数谚语</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§5.4 ｜ 2026-08-07</p>
</div>

## 为什么从 Cholesky 分解开始

很多矩阵天然**对称正定（symmetric positive definite，SPD）**：最小二乘的 $A^\top A$、协方差矩阵、刚度矩阵（有限元）、以及扩散/热传导离散。对 SPD 矩阵，存在一个比 LU 更优雅的分解——**Cholesky 分解 $A=LL^\top$**，其中 $L$ 是**下三角**（对角线不必为 1）。它把 LU 的成本减半，且数值上**天生稳定**（无需选主元）。<span class="marginnote">「平方根法」这个名字来自 $L$ 是 $A$ 的「矩阵平方根」（$A=LL^\top$）。安德烈-路易 · 肖莱（André-Louis Cholesky）1910 年前后在地籍测量中发明此方法。它是最小二乘、蒙特卡洛采样（马氏距离）与许多优化算法的心脏。</span>

本节给出 Cholesky 分解的推导、计算、成本与稳定性。

## 1 定理：SPD 矩阵的平方根分解

**定理（Cholesky 分解）。** 若 $A$ 是 $n\times n$ 对称正定矩阵，则存在唯一的下三角矩阵 $L$（对角线元素为正）使

$$
A = L\,L^\top
$$

**为什么只有 SPD 才有这种分解？** 因为 $A=LL^\top$ 蕴含两个必要条件：

- **对称**：$(LL^\top)^\top=LL^\top=A$。
- **正定**：对任意 $\mathbf{x}\neq0$，$\mathbf{x}^\top A\mathbf{x}=\lVert L^\top\mathbf{x}\rVert_2^2>0$（$L^\top\mathbf{x}\neq0$ 因 $L$ 可逆）。

反过来，SPD 保证分解存在——这是「对称正定 ⇔ 有 Cholesky」的完整循环。

**为什么「无需选主元」？** Cholesky 过程里的对角元素 $\ell_{kk}$ 满足 $\ell_{kk}^2>0$，主元恒为正且不会小到爆炸（除非矩阵本身病态）。**SPD 结构自带数值稳定**，省去主元选择的全部开销。

## 2 公式解析：Cholesky 的计算

由 $A=LL^\top$，写出元素等式 $a_{ij}=\sum_{k=1}^{\min(i,j)}\ell_{ik}\ell_{jk}$。按列推进：

**第 $j$ 列**：

$$
\ell_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1}\ell_{jk}^2}
$$

$$
\ell_{ij} = \frac{a_{ij} - \sum_{k=1}^{j-1}\ell_{ik}\ell_{jk}}{\ell_{jj}}, \qquad i=j+1,\dots,n
$$

- **第一步，对角线。** $\ell_{jj}$ 是「$\sqrt{A_{jj}}$ 减去前面已算列的平方和」。开方要求被开方数为正——**SPD 保证它恒正**；若算出负的（数值上），说明 $A$ 并非正定。
- **第二步，下方元素。** $\ell_{ij}$ 是「$a_{ij}$ 减去前面内积」再除以 $\ell_{jj}$。
- **第三步，逐列推进。** 与 LU 类似但**只用 $A$ 的下三角部分**（对称性省一半数据）。

**示例**：$A=\begin{pmatrix}4&2&2\\2&5&3\\2&3&6\end{pmatrix}$。

- 第 1 列：$\ell_{11}=\sqrt4=2$，$\ell_{21}=2/2=1$，$\ell_{31}=2/2=1$。
- 第 2 列：$\ell_{22}=\sqrt{5-1^2}=2$，$\ell_{32}=(3-1\cdot1)/2=1$。
- 第 3 列：$\ell_{33}=\sqrt{6-1^2-1^2}=2$。

$L=\begin{pmatrix}2&0&0\\1&2&0\\1&1&2\end{pmatrix}$，验证 $LL^\top=A$ ✓。<span class="marginnote">手算 Cholesky 的验证很愉悦：<strong>每算一个 $\ell_{ij}$，回代进 $a_{ij}=\sum\ell_{ik}\ell_{jk}$ 检查</strong>。上面的矩阵是「完美平方」的结构——$L$ 对角线全 2、次对角全 1，这不是巧合而是刻意选的干净例子。</span>

## 3 计算量与稳定性

**成本**：Cholesky 只需约 $\dfrac{n^3}{6}$ 次乘加——**比 LU 的 $n^3/3$ 又省一半**。省在两点：

1. **只处理下三角**（对称性省一半数据读写）。
2. **无选主元**（结构自带稳定）。

**稳定性**：Cholesky 的舍入误差界满足

$$
\lVert A-\hat{L}\hat{L}^\top\rVert \le c\,n^2\epsilon_{\mathrm{mach}}\lVert A\rVert
$$

误差被**严格控制在 $A$ 规模的比例内**——即便 $A$ 病态，Cholesky 也「诚实反映」问题难度，不会额外放大。**SPD + Cholesky 是数值线性代数里最稳的组合之一**。<span class="marginnote">对比：LU 的误差界含「增长因子」的放大，而 Cholesky 没有——<strong>对称正定结构让分解的稳定性理论变得干净</strong>。这也是为什么「把问题化成 SPD」（如最小二乘的法方程、谱方法的对称化）是工程里的标准套路。</span>

```python
import numpy as np

def cholesky(A):
    """对对称正定矩阵 A 求下三角 L，使 A = L @ L.T。"""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    for j in range(n):
        L[j, j] = np.sqrt(A[j, j] - np.sum(L[j, :j] ** 2))
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L

A = np.array([[4., 2., 2.],
              [2., 5., 3.],
              [2., 3., 6.]])
L = cholesky(A)
print(L)
print(np.allclose(A, L @ L.T))   # True
```

## 4 应用：从最小二乘到采样

Cholesky 无处不在：

| 场景 | 用法 |
| --- | --- |
| 最小二乘法方程 | 解 $(A^\top A)\mathbf{x}=A^\top\mathbf{b}$：先 Cholesky 分解 $A^\top A=LL^\top$，再两次三角回代 |
| 马氏距离 / 高斯采样 | 协方差 $\Sigma=LL^\top$，生成 $\mathbf{y}=L\mathbf{z}$ 使 $\mathrm{Cov}(\mathbf{y})=\Sigma$ |
| 有限元刚度方程 | $K=LL^\top$，$K$ SPD 天然适配 |
| 判定正定性 | 试做 Cholesky：中途开方失败 ⇔ 不正定 |

**辨析｜易错点：** Cholesky 要求 $A$ **对称且正定**。最小二乘里 $A^\top A$ 正定 ⇔ $A$ 列满秩；**秩亏时 $A^\top A$ 半正定，Cholesky 中途 $a_{jj}-\sum\ell^2$ 变零或负**——此时应改用 QR/SVD（秩亏诊断更强）。「分解失败」是「矩阵不正定」的天然报警器。<span class="marginnote">一句话：<strong>「Cholesky 是 SPD 的试金石」</strong>——能分解到底就是正定，中途开方出问题就不是。协方差估计里用这个特性快速检测数值不正定（常见于浮点误差导致的半正定退化）。</span>

## 5 Cholesky vs LU 对照

| 判据 | LU | Cholesky |
| --- | --- | --- |
| 适用 | 一般非奇异 | SPD |
| 分解 | $PA=LU$ | $A=LL^\top$ |
| 成本 | $n^3/3$ | $n^3/6$ |
| 选主元 | 需要 | 不需要 |
| 稳定性 | 列主元下良好 | 天生稳定 |
| 存贮 | 满 | 下三角（一半） |

**工程结论：遇 SPD 无脑上 Cholesky**——快一倍、稳一点、存一半。判断 SPD 的快速方法：对称 + 主对角为正 + Cholesky 能完成。

## 6 小结

- **Cholesky 分解** $A=LL^\top$：$L$ 下三角、对角为正，SPD 矩阵存在唯一。
- 对称正定 ⇔ 有 Cholesky；SPD 结构**无需选主元**、天生稳定。
- 成本 $n^3/6$——比 LU 省一半；误差界干净（无增长因子）。
- 计算：对角线开方 + 下方元素除以对角，逐列推进；开方出问题即不正定报警。
- 应用：最小二乘法方程、高斯采样、刚度方程；秩亏用 QR/SVD 替代。

在下一节，我们把 Cholesky 的「开方」去掉：**改进的平方根法（LDLᵀ 分解）**——避开开方运算，速度更快、对半正定更宽容。
