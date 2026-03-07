## 核心结论

Cholesky 分解解决的是对称正定矩阵的“平方根”分解问题。结论可以直接写成：

$$
A = LL^\top
$$

其中：

- $A$ 是实对称正定矩阵
- $L$ 是下三角矩阵
- $L$ 的对角线元素全部为正

这个分解称为 **Cholesky 分解**。

这里的“正定”不是口头描述，而是一个明确条件：

$$
x^\top A x > 0,\quad \forall x\neq 0
$$

意思是：只要 $x$ 不是零向量，把它代入二次型 $x^\top A x$，结果一定严格大于零。这个条件保证了 Cholesky 递推过程中每一步都能取到合法的正平方根。

Cholesky 的价值不在于“把矩阵换个写法”，而在于它专门利用了“对称正定”这一结构，因此比一般方阵的 LU 分解更省计算。对于 $n\times n$ 矩阵：

$$
\text{Cholesky 分解成本} \approx O(n^3/3)
$$

而一般 LU 分解通常是：

$$
O(2n^3/3)
$$

因此，当矩阵已经确定是对称正定时，Cholesky 往往是默认方案。典型场景包括：

- 求解线性系统 $Ax=b$
- 多次求解同一个矩阵配不同右端项
- 协方差矩阵上的高斯采样
- 核方法、高斯过程、卡尔曼滤波
- 某些优化算法中的 Hessian 近似求解

先看一个最小可算例子：

$$
A=\begin{pmatrix}4&2\\2&3\end{pmatrix}
$$

尝试写成

$$
A = LL^\top,\quad
L=\begin{pmatrix}2&0\\1&\sqrt{2}\end{pmatrix}
$$

直接验证：

$$
LL^\top=
\begin{pmatrix}2&0\\1&\sqrt{2}\end{pmatrix}
\begin{pmatrix}2&1\\0&\sqrt{2}\end{pmatrix}
=
\begin{pmatrix}
4 & 2\\
2 & 1+2
\end{pmatrix}
=
\begin{pmatrix}
4 & 2\\
2 & 3
\end{pmatrix}
=A
$$

这个例子已经包含了 Cholesky 的三个核心事实：

- 不是任意矩阵都能这样分解，输入必须是对称正定矩阵
- 下三角结构使得分解可以按列递推
- 在“对角线元素为正”的约束下，分解结果唯一

如果只记一句话，可以记成：

$$
\text{对称正定矩阵 }A\text{ 的 Cholesky 分解，就是唯一的 }A=LL^\top
$$

---

## 问题定义与边界

本文解决三个问题：

1. 什么样的矩阵可以做 Cholesky 分解
2. 分解矩阵 $L$ 如何计算
3. 为什么它在线性系统求解和高斯采样中更划算

先把边界说清楚。

### 1. 输入必须是实对称矩阵

实对称指：

$$
A=A^\top
$$

这不是附带条件，而是结构前提。因为任意矩阵 $LL^\top$ 一定对称：

$$
(LL^\top)^\top = LL^\top
$$

所以如果输入矩阵本身不是对称矩阵，它不可能被写成某个实矩阵的 $LL^\top$。

### 2. 输入必须是严格正定矩阵

严格正定要求：

$$
x^\top A x > 0,\quad \forall x\neq 0
$$

“严格”两个字很关键。因为 Cholesky 递推时需要不断计算对角元：

$$
L_{ii}=\sqrt{A_{ii}-\sum_{k<i}L_{ik}^2}
$$

被开方的量必须严格大于 $0$。如果等于 $0$，分解会退化；如果小于 $0$，分解直接失败。

### 3. 半正定、不定、非对称矩阵都不属于标准 Cholesky 的适用范围

非标准情形可以有别的处理方法，但那已经不是本文讨论的“标准 Cholesky 分解”。

把边界整理成表：

| 条件 | 是否可做标准 Cholesky | 直接原因 |
|---|---|---|
| 实对称且严格正定 | 可以 | 每一步对角项都能开正平方根 |
| 实对称但半正定 | 通常不行 | 可能出现 $L_{ii}=0$ |
| 实对称但不正定 | 不行 | 可能出现负数平方根 |
| 非对称矩阵 | 不行 | $LL^\top$ 必然对称 |
| 奇异矩阵 | 通常不行 | 严格正定矩阵一定可逆，奇异说明条件已破坏 |

这里还可以补一条常见等价条件。对实对称矩阵 $A$，下面几个说法是等价的：

| 说法 | 是否等价于正定 |
|---|---|
| $x^\top A x>0,\ \forall x\neq 0$ | 是 |
| 所有特征值都大于 0 | 是 |
| 所有顺序主子式都大于 0 | 是 |
| 存在唯一 $L$ 使 $A=LL^\top$ 且对角线为正 | 是 |

这张表的作用是帮助新手建立“同一个条件的不同观察方式”：

- 从定义看，是二次型严格为正
- 从谱看，是全部特征值正
- 从算法看，是 Cholesky 能顺利做完
- 从代数判别看，是顺序主子式全部为正

下面看一个失败案例：

$$
A=\begin{pmatrix}1&1\\1&1\end{pmatrix}
$$

按递推公式：

$$
L_{11}=\sqrt{1}=1
$$

$$
L_{21}=\frac{1}{1}=1
$$

最后：

$$
L_{22}=\sqrt{1-1^2}=\sqrt{0}=0
$$

这说明它不是严格正定，只是半正定。它的两个特征值分别是 $2$ 和 $0$，其中一个已经掉到零，因此不能做标准 Cholesky。

再看一个更直接的反例：

$$
B=\begin{pmatrix}1&2\\2&1\end{pmatrix}
$$

若按递推：

$$
L_{11}=1,\quad L_{21}=2
$$

则

$$
L_{22}=\sqrt{1-2^2}=\sqrt{-3}
$$

这里已经出现负数平方根，说明 $B$ 不是正定矩阵，而是不定矩阵。

到这里可以把判断逻辑压缩成一句话：

$$
\text{标准 Cholesky} = \text{对称} + \text{严格正定}
$$

少任何一个条件都不成立。

---

## 核心机制与推导

这一节做两件事：

1. 通过一个 $2\times 2$ 例子把递推公式算出来
2. 说明为什么分解后求解线性系统会更高效

### 1. 从按元素比较得到递推

设

$$
A=
\begin{pmatrix}
a_{11} & a_{12}\\
a_{12} & a_{22}
\end{pmatrix},\quad
L=
\begin{pmatrix}
l_{11} & 0\\
l_{21} & l_{22}
\end{pmatrix}
$$

则

$$
LL^\top=
\begin{pmatrix}
l_{11}^2 & l_{11}l_{21}\\
l_{11}l_{21} & l_{21}^2+l_{22}^2
\end{pmatrix}
$$

将其与 $A$ 对比，可得：

$$
l_{11}^2=a_{11}
$$

$$
l_{11}l_{21}=a_{12}
$$

$$
l_{21}^2+l_{22}^2=a_{22}
$$

于是：

$$
l_{11}=\sqrt{a_{11}}
$$

$$
l_{21}=\frac{a_{12}}{l_{11}}
$$

$$
l_{22}=\sqrt{a_{22}-l_{21}^2}
$$

这就是 $2\times 2$ 情况下的完整计算过程。高维情况本质一样，只是把“前面已经算出的列”累加进去。

一般情形的递推公式为：

$$
L_{ii}=\sqrt{A_{ii}-\sum_{k=1}^{i-1}L_{ik}^2}
$$

以及对 $j>i$，

$$
L_{ji}=\frac{A_{ji}-\sum_{k=1}^{i-1}L_{jk}L_{ik}}{L_{ii}}
$$

这个公式可以这样理解：

- 先算第 $i$ 列的对角元 $L_{ii}$
- 再用它去归一化这一列下面的元素 $L_{ji}$
- 每一列都只依赖于前面已经算完的列

所以 Cholesky 是天然适合“逐列构造”的。

### 2. 代入具体例子

取

$$
A=\begin{pmatrix}4&2\\2&3\end{pmatrix}
$$

设

$$
L=
\begin{pmatrix}
l_{11} & 0\\
l_{21} & l_{22}
\end{pmatrix}
$$

比较对应元素：

$$
l_{11}^2=4 \Rightarrow l_{11}=2
$$

这里必须取正号，因为 Cholesky 约定对角线元素为正，这也是唯一性的来源。

接着：

$$
l_{11}l_{21}=2 \Rightarrow 2l_{21}=2 \Rightarrow l_{21}=1
$$

最后：

$$
l_{21}^2+l_{22}^2=3
$$

即：

$$
1+l_{22}^2=3 \Rightarrow l_{22}=\sqrt{2}
$$

因此：

$$
L=\begin{pmatrix}
2&0\\
1&\sqrt{2}
\end{pmatrix}
$$

### 3. 为什么它能加速解方程组

如果要求解

$$
Ax=b
$$

而 $A=LL^\top$，则可改写为：

$$
LL^\top x=b
$$

引入中间变量 $y$：

$$
Ly=b,\quad L^\top x=y
$$

这就把一个原始方程组拆成了两个三角方程组。

三角方程组的好处是能直接顺序求解：

- 下三角方程 $Ly=b$ 用前向替代，从上往下解
- 上三角方程 $L^\top x=y$ 用后向替代，从下往上解

下面把前面的例子完整走一遍。取：

$$
b=\begin{pmatrix}6\\5\end{pmatrix}
$$

先解：

$$
\begin{pmatrix}
2&0\\
1&\sqrt{2}
\end{pmatrix}
\begin{pmatrix}
y_1\\
y_2
\end{pmatrix}
=
\begin{pmatrix}
6\\
5
\end{pmatrix}
$$

第一行：

$$
2y_1=6 \Rightarrow y_1=3
$$

第二行：

$$
y_1+\sqrt{2}y_2=5
$$

代入 $y_1=3$：

$$
\sqrt{2}y_2=2 \Rightarrow y_2=\sqrt{2}
$$

于是：

$$
y=\begin{pmatrix}3\\ \sqrt{2}\end{pmatrix}
$$

再解：

$$
\begin{pmatrix}
2&1\\
0&\sqrt{2}
\end{pmatrix}
\begin{pmatrix}
x_1\\
x_2
\end{pmatrix}
=
\begin{pmatrix}
3\\
\sqrt{2}
\end{pmatrix}
$$

第二行先解：

$$
\sqrt{2}x_2=\sqrt{2} \Rightarrow x_2=1
$$

第一行：

$$
2x_1+x_2=3
$$

代入 $x_2=1$：

$$
2x_1=2 \Rightarrow x_1=1
$$

因此：

$$
x=\begin{pmatrix}1\\1\end{pmatrix}
$$

检查：

$$
Ax=
\begin{pmatrix}4&2\\2&3\end{pmatrix}
\begin{pmatrix}1\\1\end{pmatrix}
=
\begin{pmatrix}6\\5\end{pmatrix}=b
$$

### 4. 复杂度和存储优势

把常见分解放在一起对比：

| 任务 | Cholesky | LU |
|---|---|---|
| 输入要求 | 对称正定 | 一般方阵 |
| 分解 FLOP 量级 | $n^3/3$ | 约 $2n^3/3$ |
| 单次求解 | 两次 $O(n^2)$ 三角替代 | 两次 $O(n^2)$ 三角替代 |
| 因子存储 | 只存一个三角区 | 通常存两个因子 |
| 主元选取 | 通常不需要 | 常需 pivoting |

因此 Cholesky 的优势不是“理论上更优雅”，而是：

- 分解阶段更便宜
- 存储更省
- 对称正定结构让算法更直接
- 数值表现通常很好

### 5. 高斯采样中的作用

若协方差矩阵 $\Sigma$ 是对称正定矩阵，先做：

$$
\Sigma = LL^\top
$$

再取标准正态向量：

$$
z\sim \mathcal N(0,I)
$$

定义：

$$
x=\mu+Lz
$$

则有：

$$
\mathbb E[x]=\mu+\mathbb E[Lz]=\mu
$$

并且：

$$
\mathrm{Cov}(x)=\mathrm{Cov}(Lz)=L\,\mathrm{Cov}(z)\,L^\top = LIL^\top = \Sigma
$$

因此：

$$
x\sim \mathcal N(\mu,\Sigma)
$$

这就是多元高斯采样里最常见的一步。它的实用意义是：只要 $\Sigma$ 不变，Cholesky 分解只做一次，之后可以重复生成很多样本。

---

## 代码实现

下面给出一份可直接运行的 Python 示例，完成四件事：

1. 手工实现 Cholesky 分解
2. 验证 $LL^\top=A$
3. 利用分解结果求解线性系统
4. 用同一个分解做高斯采样变换

代码只依赖 `numpy`，可以直接保存为 `cholesky_demo.py` 运行。

```python
import math
import numpy as np


def cholesky_manual(A: np.ndarray) -> np.ndarray:
    """
    Compute the Cholesky factor L of a symmetric positive definite matrix A,
    such that A = L @ L.T.
    """
    A = np.array(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    n = A.shape[0]

    if not np.allclose(A, A.T, atol=1e-12):
        raise ValueError("A must be symmetric")

    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))

            if i == j:
                value = A[i, i] - s
                if value <= 0:
                    raise ValueError(
                        "A is not strictly positive definite: "
                        f"diagonal update became {value}"
                    )
                L[i, j] = math.sqrt(value)
            else:
                if L[j, j] == 0:
                    raise ZeroDivisionError("Encountered zero pivot in Cholesky factor")
                L[i, j] = (A[i, j] - s) / L[j, j]

    return L


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ly = b for lower-triangular L.
    """
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)

    n = L.shape[0]
    y = np.zeros(n, dtype=float)

    for i in range(n):
        rhs = b[i] - np.dot(L[i, :i], y[:i])
        y[i] = rhs / L[i, i]

    return y


def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve Ux = y for upper-triangular U.
    """
    U = np.array(U, dtype=float)
    y = np.array(y, dtype=float)

    n = U.shape[0]
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        rhs = y[i] - np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = rhs / U[i, i]

    return x


def solve_by_cholesky(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve Ax = b using Cholesky factorization.
    Returns (L, x).
    """
    L = cholesky_manual(A)
    y = forward_substitution(L, b)
    x = backward_substitution(L.T, y)
    return L, x


def gaussian_transform(mu: np.ndarray, Sigma: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Transform a standard normal sample z into a sample with mean mu and covariance Sigma.
    """
    L = cholesky_manual(Sigma)
    return mu + L @ z


def main() -> None:
    A = np.array([
        [4.0, 2.0],
        [2.0, 3.0],
    ])
    b = np.array([6.0, 5.0])

    L, x = solve_by_cholesky(A, b)

    print("A =")
    print(A)
    print()

    print("L =")
    print(L)
    print()

    print("L @ L.T =")
    print(L @ L.T)
    print()

    print("solution x =")
    print(x)
    print()

    # Verify against the exact result
    assert np.allclose(L @ L.T, A)
    assert np.allclose(A @ x, b)
    assert np.allclose(x, np.array([1.0, 1.0]))

    # Compare with NumPy's built-in implementation
    L_np = np.linalg.cholesky(A)
    assert np.allclose(L, L_np)

    # Gaussian sampling transform
    Sigma = np.array([
        [2.0, 0.8],
        [0.8, 1.0],
    ])
    mu = np.array([1.0, -1.0])
    z = np.array([1.5, -0.5])  # a fixed standard-normal-like sample for demonstration

    sample = gaussian_transform(mu, Sigma, z)

    print("Sigma =")
    print(Sigma)
    print()

    print("transformed sample =")
    print(sample)
    print()

    # Monte Carlo check: empirical covariance should approach Sigma
    rng = np.random.default_rng(0)
    Z = rng.standard_normal(size=(200000, 2))
    L_sigma = cholesky_manual(Sigma)
    X = mu + Z @ L_sigma.T

    empirical_mean = X.mean(axis=0)
    empirical_cov = np.cov(X, rowvar=False)

    print("empirical mean (approx mu) =")
    print(empirical_mean)
    print()

    print("empirical covariance (approx Sigma) =")
    print(empirical_cov)
    print()

    assert np.allclose(empirical_mean, mu, atol=2e-2)
    assert np.allclose(empirical_cov, Sigma, atol=2e-2)


if __name__ == "__main__":
    main()
```

### 如何运行

```bash
python3 cholesky_demo.py
```

如果环境没有 `numpy`，先安装：

```bash
python3 -m pip install numpy
```

### 运行后你应该看到什么

核心输出应包含三部分：

1. 手工算出的下三角矩阵 `L`
2. 检查 `L @ L.T` 与原矩阵 `A` 一致
3. 线性系统的解 `x = [1, 1]`

Monte Carlo 部分还会输出一个经验均值和经验协方差，它们会逼近：

$$
\mu=\begin{pmatrix}1\\-1\end{pmatrix},\quad
\Sigma=
\begin{pmatrix}
2&0.8\\
0.8&1
\end{pmatrix}
$$

这里加 Monte Carlo 检查的目的不是“数学上证明公式成立”，而是让新手看到：`x = mu + Lz` 确实会生成目标协方差结构。

### 为什么工程里常直接调用库函数

自己实现一遍的意义是理解机制；真正工程里通常直接用库函数：

```python
L = np.linalg.cholesky(A)
```

原因很简单：

- 库实现更快
- 边界处理更完善
- 底层通常调用优化过的 BLAS/LAPACK

如果任务是批量求解：

$$
AX=B
$$

其中 $B$ 有多列，那么同一个 $L$ 可以重复利用。计算流程是：

1. 先分解一次 $A=LL^\top$
2. 对每一列右端项做一次前向替代
3. 再做一次后向替代

这也是 Cholesky 在高斯过程、核岭回归、协方差估计中非常常见的原因。

---

## 工程权衡与常见坑

Cholesky 理论上很干净，但工程里最常见的问题不是公式写错，而是输入矩阵“理论上应该正定，数值上却不够正定”。

### 1. 条件数大时容易出数值问题

**条件数** 可以粗略理解为“问题对误差的放大倍数”。若矩阵条件数很大，说明它接近奇异。即使理论上正定，浮点误差也可能让某一步出现：

- 极小的负数
- 本来该正的对角更新变成零
- 分解报错或结果极不稳定

在实现中，这通常表现为：

- `LinAlgError`
- 某一步对角项接近 `0`
- 解对微小扰动异常敏感

### 2. 典型问题与处理方式

| 问题来源 | 典型表现 | 常见处理 |
|---|---|---|
| 非严格正定 | 对角更新出现 $0$ 或负数 | 改用别的分解，或重新检查模型 |
| 数值上接近奇异 | 分解失败、结果敏感 | 加对角小量 $A+\lambda I$ |
| 输入理论上对称但受舍入误差污染 | `A` 与 `A^\top` 差一点点 | 先做 $(A+A^\top)/2$ |
| 精度不够 | 单精度下不稳定 | 改用 `float64` |
| 重复低秩更新后失去正定性 | 某次迭代突然崩溃 | 增加正定性监控 |

### 3. 对角加小量是什么

工程中最常见的修正是：

$$
A_{\text{reg}} = A + \lambda I,\quad \lambda>0
$$

这叫：

- 对角调节
- 加 `jitter`
- Tikhonov regularization
- ridge 型稳定化

它的作用是把矩阵从“接近奇异”往“更稳的正定区域”拉开一点。

如果从特征值角度看，若 $A$ 的特征值为 $\lambda_1,\dots,\lambda_n$，那么 $A+\lambda I$ 的特征值会变成：

$$
\lambda_1+\lambda,\dots,\lambda_n+\lambda
$$

因此对角加小量会把全部特征值整体往右推，提升最小特征值。

### 4. 但加小量不是“无代价修复”

还是看前面的半正定例子：

$$
A=\begin{pmatrix}1&1\\1&1\end{pmatrix}
$$

它无法做标准 Cholesky。若改成：

$$
A_\varepsilon=
\begin{pmatrix}
1+\varepsilon & 1\\
1 & 1+\varepsilon
\end{pmatrix},\quad \varepsilon>0
$$

则它变成严格正定，因为它的特征值是：

$$
2+\varepsilon,\quad \varepsilon
$$

两者都严格大于零，因此 Cholesky 可以执行。

但这里必须说明清楚：你已经不再求原矩阵对应的问题，而是在求一个正则化后的近似问题。这个近似可能是必要的，也可能改变结果解释，不能把它当成“纯技术修复”。

### 5. 新手最容易混淆的三个点

| 容易混淆的问题 | 正确理解 |
|---|---|
| 对称就够了吗 | 不够，还必须严格正定 |
| 半正定能不能做 | 标准 Cholesky 通常不行 |
| 加了 jitter 还是原问题吗 | 严格说不是，是稳定化后的近似问题 |

### 6. 实际工程中的检查顺序

如果你拿到一个“理论上应该能分解”的矩阵，但代码失败，可以按这个顺序排查：

1. 检查是否近似对称：`np.allclose(A, A.T)`
2. 检查数据类型是否为 `float64`
3. 检查最小特征值是否接近零或为负
4. 尝试对称化：$(A+A^\top)/2$
5. 尝试加小量：$A+\lambda I$
6. 如果问题本质上不是正定，改用别的分解，不要硬套 Cholesky

---

## 替代方案与适用边界

Cholesky 很高效，但它只服务于一个狭窄而重要的结构：对称正定矩阵。超出这个范围，就该换方法。

先做总表：

| 方法 | 适用条件 | 稳定性 | 典型复杂度 | 典型场景 |
|---|---|---|---|---|
| Cholesky | 对称正定 | 高，常数小 | 分解 $O(n^3/3)$ | 协方差矩阵、核矩阵、SPD 系统 |
| LU | 一般方阵 | 较好，常配主元选取 | 约 $O(2n^3/3)$ | 一般线性系统 |
| QR | 一般矩阵，尤其最小二乘 | 通常更稳 | 约 $O(n^3)$ | 最小二乘、回归 |
| SVD | 任意矩阵 | 最稳但代价高 | 更高 | 病态问题、降维、伪逆 |
| 特征分解 | 对称矩阵 | 取决于任务 | $O(n^3)$ | 谱分析、PCA |

### 1. Cholesky 和 LU 的关系

如果一个矩阵是一般方阵，不保证对称正定，那么最常用的是 LU 分解。它把矩阵写成：

$$
A=LU
$$

这里的 $L$ 和 $U$ 与 Cholesky 中的 $L$ 不是同一回事：

- LU 的 $L$ 通常是单位下三角
- Cholesky 的 $L$ 是真正参与平方根分解的下三角因子
- LU 为了稳定性常要做主元选取
- Cholesky 在正定条件下通常不需要 pivoting

### 2. Cholesky 和 QR 的关系

在最小二乘问题中，很多人先写法方程：

$$
X^\top X w = X^\top y
$$

由于 $X^\top X$ 对称，似乎就可以直接做 Cholesky。但这里有一个关键前提：

$$
X^\top X \text{ 必须严格正定}
$$

这等价于 $X$ 的列满秩。如果列之间高度相关，或者样本不足导致列不独立，那么 $X^\top X$ 可能只是半正定甚至严重病态。此时虽然“形式上”可以想用 Cholesky，但数值上往往不稳。

因此在回归问题里常见的经验是：

- 如果只追求速度，且矩阵条件较好，可以考虑法方程加 Cholesky
- 如果更看重稳定性，优先 QR
- 如果矩阵接近秩亏，优先 SVD

### 3. 一个简单选择规则

方法选择可以压缩成下面四句：

1. 已知矩阵是对称正定，优先 Cholesky
2. 一般方阵求解，优先带主元的 LU
3. 最小二乘问题，优先 QR
4. 病态、秩亏、需要伪逆或谱信息时，优先 SVD 或特征分解

### 4. 稀疏情形下还要考虑填充问题

如果矩阵很大且稀疏，除了“能不能分解”，还要关心分解后是否会产生大量填充项（fill-in）。这时虽然仍然可以做稀疏 Cholesky，但实际性能会受到：

- 变量重排序方式
- 稀疏结构图
- 填充规模

的影响。也就是说，稀疏问题里“理论复杂度更低”不等于“实际一定更快”。

### 5. 应用边界的一个典型误区

有些教材会简单写：

> 因为 $X^\top X$ 是对称矩阵，所以可用 Cholesky。

这句话不完整。严格说应该写成：

> 因为 $X^\top X$ 是对称半正定矩阵，所以只有在 $X$ 列满秩、从而 $X^\top X$ 严格正定时，标准 Cholesky 才安全适用。

这类边界一旦忽略，初学者很容易在真实数据上遇到“书上能做，代码里报错”的情况。

---

## 参考资料

1. Wikipedia, *Cholesky decomposition*  
   URL: https://en.wikipedia.org/wiki/Cholesky_decomposition  
   对应内容：定义、唯一性、递推公式、分块形式、与正定条件的等价关系。适合作为术语和公式的快速核对入口。

2. Gene H. Golub, Charles F. Van Loan, *Matrix Computations*  
   URL: https://jhupbooks.press.jhu.edu/title/matrix-computations  
   对应内容：数值线性代数标准教材，系统讨论 Cholesky、LU、QR 的复杂度、稳定性与舍入误差分析。适合进一步理解工程实现中的边界和数值问题。

3. Lloyd N. Trefethen, David Bau III, *Numerical Linear Algebra*  
   URL: https://people.maths.ox.ac.uk/trefethen/text.html  
   对应内容：数值线性代数入门经典，对正交分解、三角求解、条件数和稳定性有更适合初学者的讲法。

4. Stanford EE263 / related linear algebra notes  
   URL: https://see.stanford.edu/Course/EE263  
   对应内容：正定矩阵、二次型、最小二乘与矩阵分解的课程材料，适合把 Cholesky 放回“线性代数 + 优化 + 统计”的整体框架里理解。

5. UC Berkeley STAT 243 course materials  
   URL: https://stat243.berkeley.edu/  
   对应内容：多元高斯分布、协方差矩阵和线性变换采样。适合把 $x=\mu+Lz$ 与概率建模、高斯过程、贝叶斯计算联系起来。

6. LAPACK Users' Guide  
   URL: https://www.netlib.org/lapack/lug/  
   对应内容：实际数值库中 Cholesky 相关例程的设计背景与接口约定。适合从“算法定义”进一步走向“高性能实现”。
