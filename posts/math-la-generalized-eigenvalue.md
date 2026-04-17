## 核心结论

广义特征值问题求解的是
$$
Ax=\lambda Bx
$$
中的特征值 $\lambda$ 和特征向量 $x$。它和普通特征值问题
$$
Ax=\lambda x
$$
的差别，不是多了一个矩阵 $B$，而是“长度”“正交”“能量比值”的定义都被 $B$ 改写了。

如果普通欧氏长度用
$$
\|x\|_2^2=x^Tx
$$
衡量，那么广义问题里更自然的长度是
$$
\|x\|_B^2=x^TBx.
$$
当 $B\succ0$ 时，这确实定义了一个合法的长度和内积。此时“两个向量是否正交”，不再看 $x_i^Tx_j$，而看
$$
x_i^TBx_j.
$$

最重要的结论有三条：

1. 当 $B$ 是对称正定矩阵时，广义特征值可以看成广义 Rayleigh 商
   $$
   \rho(x)=\frac{x^TAx}{x^TBx}
   $$
   在约束 $x^TBx=1$ 下的极值。
2. 当 $B\succ 0$ 时，问题可以稳定地化为标准特征值问题，而不是直接计算 $B^{-1}A$。
3. 在工程里，广义特征值问题通常不是“数学装饰”，而是模型本身的一部分。结构振动中的 $Kx=\lambda Mx$、约束优化中的二次型比值、降维中的加权谱分析，本质都在做这件事。

先看一个最小例子。设
$$
A=\begin{bmatrix}2&0\\0&3\end{bmatrix},\quad
B=\begin{bmatrix}1&0\\0&2\end{bmatrix}.
$$
代入
$$
Ax=\lambda Bx
$$
得到
$$
\begin{bmatrix}
2-\lambda & 0\\
0 & 3-2\lambda
\end{bmatrix}x=0.
$$
因此特征值满足
$$
(2-\lambda)(3-2\lambda)=0,
$$
所以
$$
\lambda_1=2,\quad \lambda_2=1.5.
$$

这个结果可以直接解释成“方向上的收益除以方向上的代价”：

- 第一坐标方向上，$A$ 的系数是 $2$，$B$ 的权重是 $1$，所以比值是 $2/1=2$
- 第二坐标方向上，$A$ 的系数是 $3$，$B$ 的权重是 $2$，所以比值是 $3/2=1.5$

这说明同一个方向在 $A$ 下的“强度”，必须结合 $B$ 对这个方向的权重一起比较。白话说：$B$ 不是旁观者，它决定了“什么叫大、什么叫小”。

---

## 问题定义与边界

标准定义是：给定两个同型矩阵
$$
A,B\in\mathbb{R}^{n\times n},
$$
求非零向量 $x$ 和标量 $\lambda$，使得
$$
Ax=\lambda Bx.
$$

把它改写成
$$
(A-\lambda B)x=0,
$$
可以看到只有当
$$
\det(A-\lambda B)=0
$$
时才有非零解。这说明广义特征值问题本质上研究的是矩阵铅笔
$$
A-\lambda B.
$$
“矩阵铅笔”可以直白理解为：不是看单个矩阵，而是看一整族随着 $\lambda$ 变化的矩阵。

对新手来说，最容易混淆的点是：这个问题的难点通常不在 $A$，而在 $B$。因为 $B$ 决定了长度、归一化方式、算法选择，甚至决定问题有没有良好的谱结构。

| $B$ 的类型 | 数学含义 | 常见处理方式 | 风险 |
|---|---|---|---|
| 对称正定 | $x^TBx>0$ 对任意非零 $x$ 成立 | Cholesky 变换后转标准对称特征值问题 | 最稳定 |
| 对称半正定 | 某些方向满足 $x^TBx=0$ | 先识别零空间，必要时降维 | 可能出现退化或无穷特征值 |
| 奇异 | $B$ 不可逆 | 用 QZ / 广义 Schur，或重写模型 | 不能直接用 $B^{-1}A$ |
| 非对称可逆 | 问题仍可定义 | 用 QZ 或通用广义 eig 算法 | 可能出现复特征值，正交结构变差 |
| 病态 | 理论可逆但数值上接近奇异 | 缩放、预处理、条件数分析 | 舍入误差放大 |

为什么很多教材都强调约束
$$
x^TBx=1
$$
？因为广义 Rayleigh 商
$$
\rho(x)=\frac{x^TAx}{x^TBx}
$$
对缩放不敏感：
$$
\rho(\alpha x)=\rho(x),\quad \alpha\neq 0.
$$
所以如果不加约束，目标函数没有唯一尺度。固定
$$
x^TBx=1
$$
的含义是：先按照 $B$ 定义的长度标准，把所有候选向量都缩到“单位长度”，再比较谁让 $x^TAx$ 更大或更小。

这一点和普通特征值问题完全平行。普通问题中常用的单位球面是
$$
x^Tx=1,
$$
而广义问题中单位球面变成
$$
x^TBx=1.
$$

如果 $A,B$ 都对称，且 $B\succ0$，那么问题有最清晰的结构：

- 特征值是实数
- 特征向量可以选成 $B$-正交
- 广义 Rayleigh 商有明确的极值解释
- 可以稳定地转成标准对称特征值问题

但边界必须说清楚：

- 如果 $B$ 奇异，$x^TBx$ 不能定义所有方向上的合法长度
- 如果 $B$ 非对称，$x^TBx$ 也不再是标准内积
- 如果 $A$ 或 $B$ 严重病态，理论上成立的性质在数值实现里会变脆弱

初学时最常见的错误，就是把所有
$$
Ax=\lambda Bx
$$
都当成“普通特征值问题换了个写法”。这会直接导致错误的算法选择。

---

## 核心机制与推导

### 1. 从极值问题到广义特征值方程

考虑广义 Rayleigh 商
$$
\rho(x)=\frac{x^TAx}{x^TBx}.
$$
因为分子分母同次齐次，可以等价地写成约束优化：
$$
\max x^TAx,\quad \text{s.t. } x^TBx=1.
$$

引入拉格朗日乘子 $\lambda$：
$$
\mathcal{L}(x,\lambda)=x^TAx-\lambda(x^TBx-1).
$$

若 $A,B$ 都对称，对 $x$ 求梯度并令其为零：
$$
\nabla_x \mathcal{L}=2Ax-2\lambda Bx=0.
$$
于是得到
$$
Ax=\lambda Bx.
$$

这一步说明：广义特征值不是凭空定义出来的，而是“二次型在加权单位球面上的极值条件”。

把它说得更直接一点：

- $x^TAx$ 是“收益”或“能量”
- $x^TBx$ 是“成本”或“尺度”
- 广义特征值是单位成本下可达到的极值收益

如果取驻点向量 $x$ 满足
$$
x^TBx=1,
$$
那么由
$$
Ax=\lambda Bx
$$
左乘 $x^T$ 可得
$$
x^TAx=\lambda x^TBx=\lambda.
$$
也就是说，在 $B$-归一化条件下，广义特征值本身就等于该向量对应的 Rayleigh 商值。

进一步地，在对称且 $B\succ0$ 的情形下，最小广义特征值满足
$$
\lambda_{\min}=\min_{x\neq 0}\frac{x^TAx}{x^TBx},
$$
最大广义特征值满足
$$
\lambda_{\max}=\max_{x\neq 0}\frac{x^TAx}{x^TBx}.
$$
这就是 Courant-Fischer 极值结构在广义问题里的对应版本。

### 2. 为什么 $B\succ0$ 时能转成标准问题

若 $B$ 是对称正定矩阵，就存在 Cholesky 分解
$$
B=LL^T,
$$
其中 $L$ 是下三角且可逆。

令
$$
y=L^Tx,\qquad x=L^{-T}y.
$$
代回原方程：
$$
A(L^{-T}y)=\lambda B(L^{-T}y)=\lambda LL^TL^{-T}y=\lambda Ly.
$$
左乘 $L^{-1}$，得到
$$
L^{-1}AL^{-T}y=\lambda y.
$$

定义
$$
C=L^{-1}AL^{-T},
$$
于是原问题变成标准特征值问题
$$
Cy=\lambda y.
$$

这三步是最核心的机械推导：

1. 分解 $B=LL^T$
2. 变量替换 $y=L^Tx$
3. 构造 $C=L^{-1}AL^{-T}$

它的价值不只是“形式上能转化”，而是能保住结构。

如果 $A$ 也对称，那么
$$
C^T=(L^{-1}AL^{-T})^T=L^{-1}A^TL^{-T}=C,
$$
所以 $C$ 仍然对称。于是后续可以使用标准对称特征值算法，而不必落到一般非对称问题上。

这里顺便解释一个常见疑问：为什么不直接把原式左乘 $B^{-1}$，写成
$$
B^{-1}Ax=\lambda x
$$
？

因为这样虽然代数上可行，但数值上通常不划算：

- 显式求逆会放大误差
- 即使 $A,B$ 对称，$B^{-1}A$ 通常也不是对称矩阵
- 原本可用的对称结构和 $B$-正交性质会丢失

所以工程里真正做的是“相合变换”，不是“先求逆再乘”。

### 3. 广义正交性的来源

标准对称特征值问题里，不同特征值对应的特征向量满足
$$
x_i^Tx_j=0.
$$
广义对称问题里，对应关系变成
$$
x_i^TBx_j=0,\quad i\neq j.
$$
这叫 $B$-正交。

证明很短，但非常重要。若
$$
Ax_i=\lambda_i Bx_i,\quad Ax_j=\lambda_j Bx_j,
$$
左乘 $x_j^T$，得
$$
x_j^TAx_i=\lambda_i x_j^TBx_i.
$$
同理，
$$
x_i^TAx_j=\lambda_j x_i^TBx_j.
$$
若 $A$ 对称，则
$$
x_j^TAx_i=x_i^TAx_j.
$$
若 $B$ 对称，则
$$
x_j^TBx_i=x_i^TBx_j.
$$
两式相减得
$$
(\lambda_i-\lambda_j)x_i^TBx_j=0.
$$
当 $\lambda_i\neq\lambda_j$ 时，就得到
$$
x_i^TBx_j=0.
$$

对新手来说，这个结论的实际意义是：

- 广义问题里，“正交”不是看普通点积
- 如果你拿普通内积去比较振型、模态、方向，很可能得出错误结论
- 正确的归一化通常是
  $$
  x_i^TBx_i=1
  $$
  并且
  $$
  X^TBX=I
  $$

如果把全部特征向量按列堆成矩阵
$$
X=[x_1,\dots,x_n],
$$
那么在合适归一化下有
$$
X^TBX=I,\qquad X^TAX=\Lambda,
$$
其中
$$
\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n).
$$
这说明在 $X$ 张成的新坐标系里，$B$ 被归一成单位矩阵，而 $A$ 被对角化。

### 4. 玩具例子再推一遍

仍用
$$
A=\begin{bmatrix}2&0\\0&3\end{bmatrix},\quad
B=\begin{bmatrix}1&0\\0&2\end{bmatrix}.
$$

先对 $B$ 做 Cholesky 分解：
$$
B=LL^T,\quad
L=\begin{bmatrix}
1&0\\
0&\sqrt{2}
\end{bmatrix}.
$$

于是
$$
L^{-1}=
\begin{bmatrix}
1&0\\
0&1/\sqrt{2}
\end{bmatrix}.
$$
构造
$$
C=L^{-1}AL^{-T}.
$$
因为这里 $L$ 是对角矩阵，所以计算很直接：
$$
C=
\begin{bmatrix}
1&0\\
0&1/\sqrt{2}
\end{bmatrix}
\begin{bmatrix}
2&0\\
0&3
\end{bmatrix}
\begin{bmatrix}
1&0\\
0&1/\sqrt{2}
\end{bmatrix}
=
\begin{bmatrix}
2&0\\
0&1.5
\end{bmatrix}.
$$
于是标准问题
$$
Cy=\lambda y
$$
的特征值就是
$$
2,\quad 1.5.
$$

再把特征向量映回原变量。标准问题的两个单位特征向量可取
$$
y_1=\begin{bmatrix}1\\0\end{bmatrix},\quad
y_2=\begin{bmatrix}0\\1\end{bmatrix}.
$$
由
$$
x=L^{-T}y
$$
得到
$$
x_1=
\begin{bmatrix}1\\0\end{bmatrix},\quad
x_2=
\begin{bmatrix}
0\\
1/\sqrt{2}
\end{bmatrix}.
$$
检查它们的 $B$-归一化：
$$
x_1^TBx_1=1,\qquad x_2^TBx_2=1,
$$
并且
$$
x_1^TBx_2=0.
$$

这个例子把全部机制串起来了：

- $B$ 决定长度标准
- 变换后问题回到普通特征值分解
- 原坐标下的特征向量满足的是 $B$-正交，而不是普通正交

所以，广义特征值并不是“多了一个矩阵的标准特征值问题”，而是“在另一套几何规则下做谱分析”。

---

## 代码实现

下面给出一个可运行的 Python 示例。代码只依赖 `numpy`，实现的是最常见的情形：$A$ 对称、$B$ 对称正定。它完成四件事：

1. 检查输入是否满足基本条件
2. 用 Cholesky 变换把问题转成标准对称特征值问题
3. 把特征向量映回原坐标
4. 验证残差和 $B$-正交归一化

```python
import numpy as np


def generalized_eigh_spd(A, B, check=True):
    """
    Solve A x = lambda B x for symmetric A and SPD B.

    Parameters
    ----------
    A, B : array_like, shape (n, n)
        Real matrices. A should be symmetric, B should be symmetric positive definite.
    check : bool
        Whether to perform basic symmetry checks.

    Returns
    -------
    w : ndarray, shape (n,)
        Eigenvalues in ascending order.
    X : ndarray, shape (n, n)
        Eigenvectors in columns, normalized so that X.T @ B @ X = I.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays.")
    if A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same shape.")

    if check:
        if not np.allclose(A, A.T, atol=1e-12):
            raise ValueError("A must be symmetric.")
        if not np.allclose(B, B.T, atol=1e-12):
            raise ValueError("B must be symmetric.")

    # B = L L^T, fails automatically if B is not positive definite.
    L = np.linalg.cholesky(B)

    # Build C = L^{-1} A L^{-T} without forming matrix inverses explicitly.
    tmp = np.linalg.solve(L, A)
    C = np.linalg.solve(L, tmp.T).T

    # Numerical cleanup: keep C symmetric up to roundoff.
    C = 0.5 * (C + C.T)

    # Solve the standard symmetric eigenproblem.
    w, Y = np.linalg.eigh(C)

    # Map eigenvectors back: x = L^{-T} y
    X = np.linalg.solve(L.T, Y)

    # Enforce B-orthonormality: X^T B X = I
    gram = X.T @ B @ X
    scale = np.sqrt(np.diag(gram))
    X = X / scale

    return w, X


def residual_norm(A, B, lam, x):
    """Return ||A x - lam B x||_2."""
    r = A @ x - lam * (B @ x)
    return np.linalg.norm(r)


if __name__ == "__main__":
    # Toy example
    A = np.array([[2.0, 0.0],
                  [0.0, 3.0]])
    B = np.array([[1.0, 0.0],
                  [0.0, 2.0]])

    w, X = generalized_eigh_spd(A, B)

    print("eigenvalues:", w)
    print("eigenvectors:")
    print(X)
    print("X^T B X =")
    print(X.T @ B @ X)

    # Expected eigenvalues: 1.5, 2.0
    assert np.allclose(w, np.array([1.5, 2.0]), atol=1e-10)

    # Check residuals
    for i in range(len(w)):
        err = residual_norm(A, B, w[i], X[:, i])
        print(f"residual[{i}] = {err:.3e}")
        assert err < 1e-10

    # Check B-orthonormality
    assert np.allclose(X.T @ B @ X, np.eye(2), atol=1e-10)
```

这段代码里最关键的不是 `np.linalg.eigh`，而是前面的变换过程。

| 步骤 | 数学对象 | 代码含义 |
|---|---|---|
| 1 | $B=LL^T$ | `np.linalg.cholesky(B)` |
| 2 | $C=L^{-1}AL^{-T}$ | 用两次 `solve` 做三角线性方程求解，不显式求逆 |
| 3 | $Cy=\lambda y$ | `np.linalg.eigh(C)` 解对称标准谱问题 |
| 4 | $x=L^{-T}y$ | `np.linalg.solve(L.T, Y)` 把特征向量映回原坐标 |
| 5 | $X^TBX=I$ | 用 Gram 矩阵做 $B$-正交归一 |

这里专门强调“不要显式求逆”：

- `np.linalg.inv(L)` 会引入额外误差和额外开销
- 三角系统最稳定的做法是直接解方程
- 数值线性代数库内部也是按这个思路实现

再给一个稍微贴近工程的例子。结构振动中常见模型是
$$
Kx=\lambda Mx,
$$
其中：

- $K$ 是刚度矩阵，表示结构抵抗变形的能力
- $M$ 是质量矩阵，表示惯性的分布
- $\lambda=\omega^2$，$\omega$ 是圆频率
- $x$ 是振型，描述某个固有模态下各自由度的相对位移

如果 $M\succ0$，就可以直接套上面的代码框架：

1. 对 $M$ 做 Cholesky 分解
2. 把问题转成标准对称特征值问题
3. 解出特征值 $\lambda$
4. 取
   $$
   \omega=\sqrt{\lambda}
   $$
   得到固有频率
5. 对应特征向量做 $M$-正交归一

这也是很多工程库和 LAPACK 例程背后的基本思路。

---

## 工程权衡与常见坑

最常见的误用是直接形成
$$
B^{-1}A
$$
然后把它当成普通特征值问题去解。理论上只要 $B$ 可逆，这样写没有代数错误；但工程上通常不是好主意，原因有三点：

1. 显式求逆会放大数值误差。
2. 即使 $A,B$ 都对称，$B^{-1}A$ 通常也不是对称矩阵。
3. 非对称化以后，会丢掉原本可利用的稳定结构和 $B$-正交性质。

下面把常见坑汇总成表。

| 常见坑 | 表现 | 根因 | 规避策略 |
|---|---|---|---|
| 直接算 $B^{-1}A$ | 特征向量不稳定，误差偏大 | 显式求逆且丢失对称结构 | 用 Cholesky 变换或专用广义例程 |
| $B$ 奇异仍强行 Cholesky | 分解报错或结果异常 | 正定条件不成立 | 先检查秩，再决定是否降维或改用 QZ |
| 忽略归一化方式 | 同一组向量比较时看起来“不一致” | 特征向量天然只确定到比例因子 | 用 $x^TBx=1$ 或 $X^TBX=I$ 归一 |
| 病态 $B$ | 轻微扰动导致结果剧烈变化 | 条件数过大 | 做缩放、预处理、残差检查 |
| 重复特征值处理不当 | 不同软件算出的向量差很多 | 重根对应子空间不唯一 | 比较子空间，不比较单个向量 |
| 稀疏大矩阵用稠密算法 | 内存和时间都不可接受 | 算法复杂度不匹配 | 用 Lanczos / Arnoldi / LOBPCG |

这里展开两个最容易踩坑的地方。

第一，**重复特征值时，特征向量本来就不唯一**。

例如某个重特征值对应二维子空间，那么只要这个子空间不变，里面任选一组 $B$-正交基都可以。于是：

- 两次运行结果的特征向量可能长得不同
- 但它们张成的子空间可能完全相同
- 这不是算法错了，而是问题本身就允许这种自由度

所以在模态分析、降维、谱聚类里，重复或近重复特征值附近应该比较投影矩阵或子空间夹角，而不是逐列硬对齐向量。

第二，**“只关心特征值，不关心特征向量”经常是误判**。

很多工程任务里，真正需要的是特征向量的结构：

- 振型叠加需要模态基
- 稳定性分析需要主方向
- 约束优化需要判定最危险或最优方向
- 谱方法降维需要前几个特征向量构成嵌入空间

如果你为了图省事把问题粗暴改写成 $B^{-1}A$，最后即使特征值还能看，特征向量的结构也可能已经被破坏。

结构动力学里还有一个典型坑：模型写成
$$
Kx=\lambda Mx,
$$
但某些自由度没有质量，导致 $M$ 奇异。此时如果还按“质量矩阵正定”处理，算法要么直接失败，要么给出混乱结果。正确顺序通常是：

1. 检查约束是否正确施加
2. 删除无物理意义或未参与惯性的自由度
3. 在有效自由度空间里重新组装矩阵
4. 再决定使用 SPD 广义特征值解法还是 QZ

最后给一个简单的工程检查清单。对称 SPD 情形下，求解后最好至少检查三件事：

| 检查项 | 公式 | 目的 |
|---|---|---|
| 残差 | $\|Ax_i-\lambda_iBx_i\|_2$ | 确认方程确实被满足 |
| 归一化 | $x_i^TBx_i=1$ | 确认尺度一致 |
| 正交性 | $x_i^TBx_j\approx 0$ | 确认模态基质量 |

这些检查比“软件有没有返回结果”更重要。

---

## 替代方案与适用边界

当 $B\succ0$ 且问题是对称的，Cholesky 变换通常是首选。但它并不覆盖全部情况。算法选择必须跟着问题结构走，而不是反过来。

| 场景 | 推荐方法 | 适用原因 |
|---|---|---|
| $A,B$ 对称，且 $B\succ0$ | Cholesky + 对称特征值例程 | 最稳定，能保留对称结构 |
| $B$ 可逆但不对称 | QZ / generalized Schur | 直接处理矩阵对，不制造对称假象 |
| $B$ 奇异或接近奇异 | QZ、秩揭示分解、降维重构 | 可处理无穷特征值和退化 |
| 稀疏大规模，只求少量特征对 | Lanczos / Arnoldi / LOBPCG | 只需矩阵向量乘，不必稠密化 |
| 只关心极值特征值 | 迭代法 + 预条件 | 计算量更低 |

### 1. QZ 方法

QZ 分解也叫广义 Schur 分解。它做的不是把问题硬改写成单矩阵特征值分解，而是直接对矩阵对 $(A,B)$ 做稳定变换：
$$
Q^TAZ=S,\qquad Q^TBZ=T,
$$
其中 $Q,Z$ 是正交矩阵，$S,T$ 是上三角或准上三角矩阵。

此时广义特征值由对角元比值给出：
$$
\lambda_i=\frac{S_{ii}}{T_{ii}}.
$$
如果某个
$$
T_{ii}=0,
$$
就可能对应无穷特征值。这个现象在“先算 $B^{-1}A$”的思路里通常要么被掩盖，要么被错误处理。

QZ 的适用面更广，尤其适合：

- $B$ 非对称
- $B$ 奇异
- 需要显式处理无穷特征值
- 不想假设问题具有对称谱结构

代价是：它通常比对称 SPD 专用算法更贵，也更难保留漂亮的正交解释。

### 2. Krylov / Arnoldi 方法

当矩阵很大且稀疏时，真正可行的思路不是“完整分解全部特征对”，而是只计算你需要的少数几个。

这类方法的核心不是存整个矩阵，而是提供矩阵向量乘法：
$$
x\mapsto Ax,\qquad x\mapsto Bx.
$$

伪代码可以写成：

```python
# Pseudocode: sparse generalized eigenproblem
def matvec_A(x):
    return A @ x

def matvec_B(x):
    return B @ x

eigvals, eigvecs = sparse_generalized_eigs(
    matvec_A,
    matvec_B,
    k=5,          # only need the first 5 eigenpairs
    which="SM"    # for example: smallest magnitude / smallest value
)
```

对新手来说，最需要建立的直觉是：

- 稠密算法适合中小规模、全谱问题
- 稀疏迭代法适合超大规模、少量特征对问题
- 两类算法不是谁“更高级”，而是目标不同

如果问题是有限元离散后的百万维系统，却还想一次性做完整稠密分解，那不是算法细节问题，而是模型规模和算法类型根本不匹配。

### 3. 何时可以重写为 $Bx=\mu Ax$

有时会把问题写成
$$
Bx=\mu Ax,\qquad \mu=\frac{1}{\lambda}.
$$
这种重写并不是纯代数游戏，而是为了让“主算子”落在更好处理的一侧，或者为了更方便地提取接近零或接近无穷的谱。

但它只有在你明确以下边界时才安全：

- 零特征值和无穷特征值如何对应
- 哪些方向位于 $A$ 或 $B$ 的零空间
- 你关心的是 $\lambda$ 的大值、小值，还是它们的倒数

如果这些边界不清楚，把
$$
\lambda
$$
换成
$$
\mu=1/\lambda
$$
通常只是把困难从一边搬到另一边。

这一节可以压缩成一句实用判断：

如果 $B$ 正定，用 Cholesky；如果结构一般，用 QZ；如果规模巨大，用迭代法；如果 $B$ 奇异，先承认它奇异，再选适配方法，不要假装它可逆。

---

## 参考资料

1. IMSL, *Generalized Eigenvalue Problems*：介绍广义特征值问题的变量替换、矩阵分解与奇异情形处理。<https://help.imsl.com/PyNL/1.0/html/eigen/usage/generalized.eigenvalue.problems.html>
2. LAPACK Users' Guide, *Generalized Symmetric Definite Eigenproblems*：说明如何通过 Cholesky 变换把对称正定广义问题降为标准特征值问题。<https://netlib.org/lapack/lug/node54.html>
3. Intel MKL 文档，*Generalized Symmetric Definite Eigenvalue Problems*：给出工程库中的对应例程与调用边界。<https://www.smcm.iqfr.csic.es/docs/intel/mkl/mkl_manual/lse/lse_crgsdep.htm>
4. Guangliang Chen, San Jose State University, *Rayleigh Quotient* 讲义：说明 Rayleigh 商、约束极值与特征值问题之间的关系。<https://www.sjsu.edu/faculty/guangliang.chen/Math253S20/lec4RayleighQuotient.pdf>
5. AMS Stony Brook 课程讲义，*Generalized Eigenvalue Problems in Vibrations*：包含结构振动中的 $Kx=\lambda Mx$ 背景与推导。<https://www.ams.stonybrook.edu/~jiao/teaching/ams526/lectures/lecture17.pdf>
6. Gene H. Golub, Charles F. Van Loan, *Matrix Computations*：广义 Schur、对称定广义特征值问题、数值稳定性分析的经典教材。
7. Lloyd N. Trefethen, David Bau III, *Numerical Linear Algebra*：特征值问题、相似变换、数值稳定性的入门教材，适合建立算法直觉。
