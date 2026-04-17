## 核心结论

条件数是问题本身的难度指标。它回答的是：输入里一个很小的误差，输出在最坏情况下会被放大多少倍。对可逆矩阵 $A$，常用定义是

$$
\kappa(A)=\|A\|\cdot\|A^{-1}\|
$$

若使用 $2$-范数，则还有一个更直观的形式：

$$
\kappa_2(A)=\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
$$

其中 $\sigma_{\max}$ 和 $\sigma_{\min}$ 分别是最大、最小奇异值。这个式子直接揭示了一点：矩阵一旦接近奇异，$\sigma_{\min}(A)$ 会很小，条件数就会迅速变大。

如果 $\kappa(A)\approx 10^k$，通常意味着结果可能丢失大约 $k$ 位有效数字。这里的“有效数字”可以直白理解为：在最终输出里，你还能信的那几位。比如双精度浮点数大约有 $16$ 位十进制有效数字，如果问题条件数达到 $10^8$，那就意味着即使算法很稳，最终也可能只剩下大约 $8$ 位可信数字。

稳定性是算法的误差控制能力。直白说，稳定算法不会凭空制造远大于舍入误差的额外灾难。在线性方程组 $Ax=b$ 中，常见判断框架是：

| 维度 | 关注对象 | 含义 |
| --- | --- | --- |
| 问题本身 | 条件数 $\kappa(A)$ | 数据扰动会不会被放大 |
| 算法过程 | 稳定性 / 向后稳定性 | 计算过程会不会额外放大误差 |
| 最终效果 | 前向误差 | 算出来的 $\hat x$ 离真解 $x$ 有多远 |

三者的关系可以压缩成一句话：

$$
\text{前向误差} \approx \text{条件数} \times \text{向后误差}
$$

这不是严格等号，而是工程上很有用的思考框架。它说明：同样一个稳定算法，放在条件良好的问题上，结果通常可信；放在病态问题上，即使算法本身没有犯大错，结果也未必精确。

所以不能只问“算法是不是高级”，而要先问“矩阵是不是好矩阵”。稳定算法加上条件良好的矩阵，结果才可信。矩阵病态时，即使算法本身稳定，也通常只能保证“在一个稍微扰动过的问题上是对的”，不能保证对原问题高精度。

---

## 问题定义与边界

我们讨论的对象是线性方程组

$$
Ax=b
$$

重点是有限精度浮点计算下的数值行为，而不是精确算术下的代数性质。也就是说，我们关心的不是“纸面上答案是否存在”，而是“计算机实际算出来的答案是否还可信”。

先固定几个术语。

| 术语 | 直白解释 | 这里关注什么 |
| --- | --- | --- |
| 舍入误差 | 真实数值无法被浮点数精确表示 | 单步计算误差会如何积累 |
| 前向误差 | 算出来的解和真解差多远 | 最终结果准不准 |
| 向后误差 | 当前结果相当于解了哪个轻微扰动后的问题 | 算法过程稳不稳 |
| 病态问题 | 输入微小变化会引起输出巨大变化 | 问题本身是否脆弱 |
| 稳定算法 | 不会额外制造远超舍入误差的误差 | 算法是否可靠 |

“舍入误差”是指计算机只能存有限位数字，真实结果会被截断或四舍五入。比如 `0.1` 在二进制里不能精确表示，很多看起来简单的十进制小数，在机器内部其实只能存近似值。这类误差在加减乘除、消元、迭代过程中都会传播。

一个典型危险场景是“灾难性消去”。例如两个非常接近的数相减：

$$
1.23456789 - 1.23456780 = 0.00000009
$$

如果前面两项本身已经带有舍入误差，那么相减后剩下的有效位会明显减少。数值线性代数里，很多不稳定算法的核心问题就是不断制造这种局面。

“病态问题”是指输入稍微变一点，输出就大幅变化。它是问题自身的属性，不是代码写得差。对线性系统，若只考虑右端项扰动 $b\mapsto b+\delta b$，常见估计是

$$
\frac{\|\delta x\|}{\|x\|}\lesssim \kappa(A)\frac{\|\delta b\|}{\|b\|}
$$

若矩阵本身也有扰动 $A\mapsto A+\delta A$，更完整的形式常写为

$$
\frac{\|\delta x\|}{\|x\|}
\lesssim
\frac{\kappa(A)}{1-\kappa(A)\frac{\|\delta A\|}{\|A\|}}
\left(
\frac{\|\delta A\|}{\|A\|}+\frac{\|\delta b\|}{\|b\|}
\right)
$$

只要 $\kappa(A)\frac{\|\delta A\|}{\|A\|}<1$，这个估计就有意义。它表达的不是精确常数，而是方向：条件数越大，相同量级的输入误差越容易被放大。

一个很小的例子就能看出“病态”和“奇异”不是一回事。设

$$
A_\varepsilon=
\begin{bmatrix}
1 & 1\\
1 & 1+\varepsilon
\end{bmatrix}
$$

则

$$
\det(A_\varepsilon)=\varepsilon
$$

当 $\varepsilon\neq 0$ 时它仍然可逆，但只要 $\varepsilon$ 很小，矩阵就已经非常接近奇异，条件数会很大。这类矩阵在代数上“有解”，在数值上却可能“很难算准”。

迭代法还要额外看收敛边界。把矩阵拆成 $A=D-L-U$ 后，Jacobi 迭代写成

$$
x^{(k+1)}=D^{-1}\bigl(b-(L+U)x^{(k)}\bigr)
$$

对应迭代矩阵

$$
G=D^{-1}(L+U)
$$

只有当谱半径 $\rho(G)<1$ 时才收敛。这里“谱半径”可以白话理解为：矩阵所有特征值绝对值里的最大者，它决定误差分量在迭代中是缩小还是放大。

误差递推式是

$$
e^{(k+1)}=Ge^{(k)}
$$

所以

$$
e^{(k)}=G^k e^{(0)}
$$

若 $\rho(G)<1$，则 $G^k\to 0$，误差才会衰减；若 $\rho(G)>1$，误差至少在某些方向上会放大。

一个玩具例子能直接看出边界。设

$$
A=\begin{bmatrix}
5&5&5\\
5&5&5\\
5&5&5
\end{bmatrix}
$$

这个系统本身就退化，Jacobi 的对角部分 $D=5I$，于是

$$
G=D^{-1}(L+U)=
\begin{bmatrix}
0&1&1\\
1&0&1\\
1&1&0
\end{bmatrix}
$$

这个矩阵的特征值是 $2,-1,-1$，因此

$$
\rho(G)=2>1
$$

误差每轮都会放大，必然发散。这个例子说明：不是“多迭代几轮就会更准”，而是先满足收敛条件才有资格谈精度。

可以把常见边界记成一张表：

| 场景 | 常用方法 | 适用前提 |
| --- | --- | --- |
| 严格对角占优 | Jacobi / Gauss-Seidel | 迭代矩阵谱半径小于 1 |
| 对称正定（SPD） | Cholesky / 共轭梯度 | 结构良好，通常更稳定更高效 |
| 稀疏大规模 SPD | 预条件共轭梯度 | 不能显式分解或分解代价过高 |
| 高共线、近奇异 | QR / SVD | 比法方程和普通消元更稳 |
| 明显病态 | 正则化 / 重构模型 | 仅换算法通常不够 |

再补一张“不要混淆”的表：

| 现象 | 根源 | 典型处理 |
| --- | --- | --- |
| 算法中间量爆炸 | 算法过程不稳 | 主元选取、换分解方式 |
| 对输入极度敏感 | 问题病态 | 缩放、正则化、改模型 |
| 迭代反复震荡 | 收敛条件不满足 | 预条件、换迭代矩阵、换方法 |
| Cholesky 报错 | 非正定或数值上接近非正定 | 抖动、LDL、QR、SVD |

---

## 核心机制与推导

先分清两个概念。前向误差是“答案偏了多少”；向后误差是“这个答案是否等价于某个稍微改动过的输入的精确解”。数值线性代数更偏爱向后误差，因为它更容易分析，也更能反映算法是否稳定。

设计算结果为 $\hat x$。如果存在很小的 $\delta A,\delta b$，使得

$$
(A+\delta A)\hat x=b+\delta b
$$

那么就说 $\hat x$ 是一个小扰动问题的精确解。若 $\delta A,\delta b$ 足够小，算法就可以认为是向后稳定的。之后再结合条件数，才能推出前向误差是否也小。

### 1. 条件数为什么决定误差放大

真解满足 $Ax=b$。如果右端项变成 $b+\delta b$，新解变成 $x+\delta x$，则

$$
A(x+\delta x)=b+\delta b
\Rightarrow A\delta x=\delta b
\Rightarrow \delta x=A^{-1}\delta b
$$

取范数得

$$
\|\delta x\|\le \|A^{-1}\|\|\delta b\|
$$

另一方面，由 $b=Ax$ 可得

$$
\|b\|=\|Ax\|\le \|A\|\|x\|
\Rightarrow
\frac{1}{\|x\|}\le \frac{\|A\|}{\|b\|}
$$

两边结合得到

$$
\frac{\|\delta x\|}{\|x\|}
\le
\|A^{-1}\|\|A\|\frac{\|\delta b\|}{\|b\|}
=
\kappa(A)\frac{\|\delta b\|}{\|b\|}
$$

这就是最常见的相对误差放大界。核心点是：$A^{-1}$ 越“巨大”，系统越脆弱。

如果看 $2$-范数，这个式子还能给出一个几何解释。$\sigma_{\min}(A)$ 很小时，说明矩阵会把某些方向压得很扁；沿这些方向，反过来求解时就会把微小噪声猛烈放大。这就是病态的几何来源。

一个最小例子：

$$
A=
\begin{bmatrix}
1 & 1\\
1 & 1.0001
\end{bmatrix},\quad
b=
\begin{bmatrix}
2\\
2.0001
\end{bmatrix}
$$

真解是 $x=[1,1]^\top$。如果把第二个分量改成 $2.0002$，也就是只动了 $10^{-4}$，解就会明显变化。这不是程序异常，而是矩阵本身几乎把两列压成同一方向，导致系统极其敏感。

### 2. Jacobi 为什么看谱半径

把线性系统写成

$$
Ax=b,\qquad A=D-L-U
$$

Jacobi 迭代是

$$
x^{(k+1)}=D^{-1}\bigl(b-(L+U)x^{(k)}\bigr)
$$

设 $x^\*$ 是真解，误差记为

$$
e^{(k)}=x^{(k)}-x^\*
$$

把真解方程和迭代方程相减，得到

$$
e^{(k+1)}=Ge^{(k)},\qquad G=D^{-1}(L+U)
$$

于是

$$
e^{(k)}=G^k e^{(0)}
$$

所以是否收敛，本质上取决于 $G^k$ 会不会趋于零，而这正由 $\rho(G)<1$ 保证。

为什么是谱半径而不是单纯看某个元素是否小？因为误差不是一个数，而是多个特征方向的叠加。沿某个特征向量方向 $v_i$，有

$$
Gv_i=\lambda_i v_i
$$

那么第 $k$ 轮后该方向的误差会被放大为

$$
\lambda_i^k
$$

最大那个 $|\lambda_i|$ 决定整体趋势，这就是谱半径的作用。

新手容易犯的一个误解是：对角元素很大，就一定适合 Jacobi。其实不够。真正相关的是整个迭代矩阵的谱性质。严格对角占优是常见的充分条件，但不是必要条件；反过来，非对角占优也不等于一定发散，只是你不能再轻易保证。

### 3. 共轭梯度为什么对 SPD 特别有效

“共轭梯度”可以直白理解为：不是沿普通坐标轴反复试，而是构造一组彼此不冲突的搜索方向。对 SPD 矩阵，这些方向在 $A$-内积下正交，即

$$
p_i^\top A p_j = 0\qquad (i\neq j)
$$

这意味着每走一步，都在消去一个新的误差分量，而不会把前一步已经消掉的部分又重新引回来。

它还可以从优化角度理解。若 $A$ 是 SPD，那么解线性方程组 $Ax=b$ 等价于最小化二次函数

$$
\phi(x)=\frac12 x^\top A x - b^\top x
$$

其梯度为

$$
\nabla \phi(x)=Ax-b = -r
$$

其中残差

$$
r=b-Ax
$$

正是负梯度方向。因此 CG 不是“神秘的线性代数技巧”，而是在特殊几何结构下，比普通最速下降更聪明地选方向。

它的基本迭代从残差开始：

$$
r_0=b-Ax_0,\qquad p_0=r_0
$$

后续每步更新

$$
\alpha_k=\frac{r_k^\top r_k}{p_k^\top A p_k},\qquad
x_{k+1}=x_k+\alpha_k p_k
$$

$$
r_{k+1}=r_k-\alpha_k A p_k
$$

$$
\beta_k=\frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k},\qquad
p_{k+1}=r_{k+1}+\beta_k p_k
$$

理论上，在精确算术下，$n$ 维问题最多 $n$ 步到精确解；实际浮点环境中，收敛速度通常受谱分布控制，条件数越小，误差下降越快。常见上界写成

$$
\frac{\|x_k-x^\*\|_A}{\|x_0-x^\*\|_A}
\le
2\left(
\frac{\sqrt{\kappa(A)}-1}{\sqrt{\kappa(A)}+1}
\right)^k
$$

这个式子的含义很直接：$\kappa(A)$ 越小，括号里的比值越小，误差下降越快；$\kappa(A)$ 很大时，CG 也会慢下来，所以预条件才会重要。

玩具例子：

$$
A=\begin{bmatrix}4&1\\1&3\end{bmatrix},\quad
b=\begin{bmatrix}1\\2\end{bmatrix},\quad
x_0=\begin{bmatrix}2\\1\end{bmatrix}
$$

先算残差：

$$
r_0=b-Ax_0=
\begin{bmatrix}1\\2\end{bmatrix}-
\begin{bmatrix}9\\5\end{bmatrix}
=
\begin{bmatrix}-8\\-3\end{bmatrix}
$$

也就是第一条搜索方向 $p_0$。因为这是二维 SPD 问题，理论上最多两步就能到精确解，真解是

$$
x^\*=\begin{bmatrix}1/11\\7/11\end{bmatrix}
$$

第一步可算出

$$
\alpha_0=\frac{r_0^\top r_0}{p_0^\top A p_0}
=\frac{73}{331}
$$

得到

$$
x_1=x_0+\alpha_0 p_0
\approx
\begin{bmatrix}
0.23565\\
0.33837
\end{bmatrix}
$$

第二步后就能到达精确解。这个例子展示了三件事：残差就是当前误差的驱动信号；方向不是随便选，而是保持共轭；维数有限时可有限步终止。

### 4. Cholesky 为什么稳定

若 $A$ 是对称正定矩阵，可分解为

$$
A=LL^\top
$$

这里用下三角形式更符合大多数数值库的实现。之后通过两次三角回代求解：

$$
Ly=b,\qquad L^\top x=y
$$

Cholesky 的优势不是“永不出错”，而是它通常满足较好的向后误差界：计算结果等价于解了一个轻微扰动后的系统

$$
(A+\delta A)\hat x=b
$$

其中

$$
\|\delta A\| = O(u\|A\|)
$$

$u$ 是机器精度。双精度下，$u$ 大约是 $2.22\times 10^{-16}$。这就是为什么工程上会说 Cholesky 对 SPD 系统“很稳”：它利用了对称和正定这两个强结构，避免了很多通用算法里不必要的误差来源。

还可以从计算量看它为什么常被优先选用。对 $n\times n$ 稠密矩阵：

| 方法 | 典型计算量 | 备注 |
| --- | --- | --- |
| LU | 约 $\frac{2}{3}n^3$ | 通用，但不利用 SPD 结构 |
| Cholesky | 约 $\frac{1}{3}n^3$ | 利用对称正定，成本更低 |
| QR | 约 $\frac{2}{3}n^3$ 到更高 | 更稳，但更贵 |

所以对明确的 SPD 系统，Cholesky 同时兼顾了速度、存储和稳定性。

但要注意一个数值边界：理论上 SPD 不代表数值上一定顺利分解。如果矩阵极度病态，或由近似计算得到，分解过程中可能出现负主元，导致库函数报 `not positive definite`。这不一定说明数学模型错了，更常见的是矩阵已经接近半正定，浮点误差把它推到了边界之外。

---

## 代码实现

下面的代码做四件事：

1. 计算条件数，先判断问题是否可能病态。
2. 检查矩阵是否对称，并尽量避免对非 SPD 系统误用 Cholesky。
3. 对 SPD 系统优先用 Cholesky，而不是显式求逆。
4. 给出一个可运行的最小 CG 实现，并验证二维玩具例子两步收敛。

代码中有两个工程性设计需要明确说明：

- “加对角抖动”只是一种数值修复手段，用于把接近半正定的矩阵稍微推回正定区域，不能根治病态。
- 代码里不会调用 `inv(A)`。显式求逆通常更慢，也更容易累积误差；实际求解应优先使用分解加回代。

```python
import numpy as np


def is_symmetric(A, tol=1e-12):
    A = np.asarray(A, dtype=float)
    return np.allclose(A, A.T, atol=tol, rtol=0.0)


def solve_spd_with_jitter(A, b, cond_threshold=1e12, jitter=1e-10, max_tries=6):
    """
    Solve Ax=b for a numerically SPD matrix using Cholesky.
    If Cholesky fails because A is near-semi-definite, add diagonal jitter.

    Returns
    -------
    x : ndarray
        Computed solution.
    info : dict
        Diagnostics including condition number, whether jitter was used,
        and the final diagonal shift.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b must be a 1D vector with length matching A.")
    if not is_symmetric(A):
        raise ValueError("Cholesky requires a symmetric matrix.")

    kappa = np.linalg.cond(A)
    shift = 0.0
    A_work = A.copy()

    # 条件数很大时，先给出风险信号；是否加抖动仍以分解是否失败为准
    if not np.isfinite(kappa):
        kappa = np.inf

    for i in range(max_tries):
        try:
            L = np.linalg.cholesky(A_work)
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
            return x, {
                "cond": kappa,
                "used_jitter": shift > 0.0,
                "jitter": shift,
                "warning": (
                    "matrix is ill-conditioned"
                    if kappa > cond_threshold
                    else None
                ),
            }
        except np.linalg.LinAlgError:
            # 逐步增加对角修正，避免一次性加过大扰动
            shift = jitter if shift == 0.0 else shift * 10.0
            A_work = A + shift * np.eye(A.shape[0])

    raise np.linalg.LinAlgError(
        "Cholesky failed even after diagonal jitter; "
        "matrix may be indefinite or too ill-conditioned."
    )


def cg(A, b, x0=None, max_iter=None, tol=1e-12):
    """
    Minimal Conjugate Gradient solver for SPD matrices.

    Returns
    -------
    x : ndarray
        Approximate solution.
    history : dict
        Residual norms and iteration count.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError("A must be square.")
    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError("b must have shape (n,).")
    if not is_symmetric(A):
        raise ValueError("CG requires a symmetric matrix.")

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    if max_iter is None:
        max_iter = n

    r = b - A @ x
    p = r.copy()
    rr_old = r @ r
    residual_norms = [np.sqrt(rr_old)]

    if residual_norms[-1] < tol:
        return x, {"iterations": 0, "residual_norms": residual_norms}

    for k in range(max_iter):
        Ap = A @ p
        denom = p @ Ap
        if denom <= 0:
            raise ValueError("A does not appear to be positive definite.")

        alpha = rr_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = r @ r
        residual_norms.append(np.sqrt(rr_new))

        if residual_norms[-1] < tol:
            return x, {"iterations": k + 1, "residual_norms": residual_norms}

        beta = rr_new / rr_old
        p = r + beta * p
        rr_old = rr_new

    return x, {"iterations": max_iter, "residual_norms": residual_norms}


if __name__ == "__main__":
    # 玩具例子：2x2 SPD 系统
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x0 = np.array([2.0, 1.0])
    x_star = np.array([1.0 / 11.0, 7.0 / 11.0])

    x_chol, info = solve_spd_with_jitter(A, b)
    x_cg, hist = cg(A, b, x0=x0, max_iter=2, tol=1e-14)

    print("cond(A) =", info["cond"])
    print("Cholesky solution =", x_chol)
    print("CG solution =", x_cg)
    print("CG residual norms =", hist["residual_norms"])

    assert np.allclose(x_chol, x_star, atol=1e-12)
    assert np.allclose(x_cg, x_star, atol=1e-12)
```

这段代码可以直接运行，输出条件数、Cholesky 解、CG 解以及残差历史。对上面的二维例子，CG 在两步内达到精确解，符合精确算术下的理论。

如果想看“病态”对数值结果的影响，可以补一个小实验。下面构造一个接近奇异的矩阵：

```python
import numpy as np

A = np.array([[1.0, 1.0], [1.0, 1.0000001]])
b1 = np.array([2.0, 2.0000001])
b2 = np.array([2.0, 2.0000002])

x1 = np.linalg.solve(A, b1)
x2 = np.linalg.solve(A, b2)

print("cond(A) =", np.linalg.cond(A))
print("x1 =", x1)
print("x2 =", x2)
print("relative change in b =", np.linalg.norm(b2 - b1) / np.linalg.norm(b1))
print("relative change in x =", np.linalg.norm(x2 - x1) / np.linalg.norm(x1))
```

这个例子的重点不是具体数值，而是现象：$b$ 的变化很小，但 $x$ 的相对变化会明显更大。条件数正是在量化这种放大。

真实工程例子是广义最小二乘（GLS）。在统计建模中，常见系统是

$$
(X^\top \Sigma^{-1} X)\beta = X^\top \Sigma^{-1} y
$$

这里 $\Sigma$ 是噪声协方差矩阵。若直接求逆，误差和开销都不划算；更稳的做法是把 $\Sigma$ 或 Gram 矩阵做 Cholesky 分解，再通过三角解法完成。若协方差矩阵接近奇异，比如空间数据里点非常密集、相关长度设置不合适，就可能出现“理论上半正定、数值上不正定”的情况，此时要加对角抖动、改用 LDL 分解，或直接退到 QR / SVD 体系。

再补一个工程判断表：

| 目标 | 不推荐 | 更推荐 |
| --- | --- | --- |
| 解 SPD 线性系统 | `inv(A) @ b` | Cholesky + triangular solve |
| 解一般最小二乘 | 法方程 | QR |
| 解秩亏最小二乘 | 普通消元 | SVD |
| 解大规模稀疏 SPD | 稠密分解 | CG / PCG |
| 修复近半正定协方差 | 直接硬算 | 抖动、LDL、重参数化 |

---

## 工程权衡与常见坑

工程里最常见的误区，是把“算法不稳定”和“问题病态”混为一谈。前者可以换算法缓解，后者往往需要改建模、加正则、变变量尺度。

| 风险 | 现象 | 缓解方式 |
| --- | --- | --- |
| 条件数很大 | 解对微小扰动极敏感 | 先算 `cond`，必要时改用 QR / SVD 或正则化 |
| 直接算逆 | 误差和成本都偏高 | 改成分解加回代 |
| Cholesky 失败 | 报 “not positive definite” | 对角抖动、LDL、pivoted Cholesky |
| Jacobi 不收敛 | 迭代震荡或发散 | 检查谱半径、对角占优、缩放矩阵 |
| CG 很慢 | 残差下降拖沓 | 做预条件，降低 $\kappa(M^{-1}A)$ |

有两个坑尤其需要记住。

第一，部分主元不能拯救病态。它主要控制消元过程中的增长因子，减少中间量爆炸，但无法改变 $\kappa(A)$。如果问题本身已经接近奇异，LU 再精巧也不可能凭空恢复有效数字。它解决的是“算法过程别太糟”，不是“把坏问题变好”。

第二，法方程会平方条件数。最典型的是最小二乘里把问题写成

$$
X^\top X \beta = X^\top y
$$

因为在 $2$-范数下有

$$
\kappa_2(X^\top X)=\kappa_2(X)^2
$$

如果 $X$ 本来就有共线性，法方程会把病态进一步放大。这也是为什么工程上常说：回归求解优先 QR，别默认上 $X^\top X$。

再补三个经常被忽略的坑。

第三，缩放不当会放大病态。假设某一列量级是 $10^6$，另一列是 $10^{-3}$，即使模型本身没错，数值表现也可能很差。常见处理是列归一化、变量无量纲化，或者把模型重写成尺度更平衡的形式。

第四，残差小不等于解一定准。若 $\kappa(A)$ 很大，即使残差

$$
r=b-A\hat x
$$

已经很小，前向误差仍可能不小。常见估计是

$$
\frac{\|x-\hat x\|}{\|x\|}
\lesssim
\kappa(A)\frac{\|r\|}{\|b\|}
$$

也就是说，病态问题里“残差很小”只能说明你把方程代回去看起来差不多成立，不能保证真解也离得近。

第五，预条件不是“随便乘个矩阵”。预条件的目标是把系统变成一个谱分布更集中的等价问题，让迭代更快。好的预条件器要同时满足两件事：

- $M^{-1}A$ 的条件数比原来小。
- 应用 $M^{-1}$ 本身不能太贵。

如果预条件器构造成本过高，或者每步求解 $Mz=r$ 也很重，整体反而可能更慢。

把这些坑压缩成一句工程判断：

$$
\text{可信结果} \approx \text{合理建模} + \text{可控条件数} + \text{稳定算法}
$$

少了任何一项，数值结果都可能看起来正常、实际上不可靠。

---

## 替代方案与适用边界

Cholesky 不是万能解。它非常适合对称正定且条件数可控的问题，但超出这个边界就要切换。

| 方法 | 适用条件 | 优点 | 局限 |
| --- | --- | --- | --- |
| Cholesky | SPD，条件较好 | 快，存储省，向后稳定 | 非正定或近半正定时易失败 |
| QR | 一般稠密最小二乘 | 比法方程稳定 | 比 Cholesky 稍贵 |
| SVD | 秩亏、强共线 | 最稳，可看清秩结构 | 成本最高 |
| Jacobi / GS | 结构简单、对角占优 | 实现简单 | 收敛依赖谱半径 |
| CG | 大规模稀疏 SPD | 不必显式分解 | 需预条件，受谱分布影响 |
| 预条件 CG | 大规模稀疏且病态 | 可显著加速 | 预条件器设计本身是难点 |

如果是中小规模、明确 SPD、条件数不大，Cholesky 往往是第一选择。如果是大规模稀疏 SPD，共轭梯度通常更合适；如果残差降不下来，不是先加迭代轮数，而是先想预条件。如果矩阵接近奇异、列高度相关，QR 或 SVD 更可靠。

还可以从“决策问题”角度再压一层：

| 你面对的系统 | 第一反应 |
| --- | --- |
| 稠密、SPD、规模中等 | 先试 Cholesky |
| 稠密、最小二乘、可能共线 | 先试 QR |
| 可能秩亏，想看数值秩 | 直接上 SVD |
| 稀疏、SPD、规模很大 | 先试 CG / PCG |
| 明显病态，解不稳定 | 先检查建模与正则化，不要只换求解器 |

这里有一个常见误判边界：很多人把“换更复杂的算法”当成默认修复手段。实际上，当问题已经明显病态时，算法升级往往只能减少额外误差，不能突破信息论边界。比如输入数据本身噪声很大、设计矩阵高度共线、协方差接近奇异，这时更有效的动作常常是：

- 重新缩放变量。
- 删除冗余特征。
- 加正则化项。
- 改模型参数化方式。
- 接受“这个问题本来就只能得到有限精度”的事实。

一句话概括边界：条件数决定你最多能信多少，算法决定你会不会白白再丢更多。

---

## 参考资料

- Trefethen, Lloyd N.; Bau, David. *Numerical Linear Algebra*. SIAM, 1997.
- Higham, Nicholas J. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002.
- Demmel, James W. *Applied Numerical Linear Algebra*. SIAM, 1997.
- Golub, Gene H.; Van Loan, Charles F. *Matrix Computations*, 4th ed. Johns Hopkins University Press, 2013.
- Condition number. Wikipedia. https://en.wikipedia.org/wiki/Condition_number
- Jacobi method. Wikipedia. https://en.wikipedia.org/wiki/Jacobi_method
- Conjugate gradient method. Wikipedia. https://en.wikipedia.org/wiki/Conjugate_gradient_method
- Cholesky decomposition. Wikipedia. https://en.wikipedia.org/wiki/Cholesky_decomposition
- Higham, N. “Cholesky Factorization.” MIMS EPrint. https://eprints.maths.manchester.ac.uk/1199/
- Bilman Lecture 22: Stability of Gaussian Elimination. https://bilman.github.io/Lecture-22.html
- TUM Numerical Linear Algebra Lecture 03. https://venkovic.github.io/NLA-for-CS-and-IE/TUM_NLA-for-CS-and-IE_Lecture03.pdf
- Stat 243 Unit 10: Linear Algebra. UC Berkeley. https://stat243.berkeley.edu/fall-2024/units/unit10-linalg.html
- Burkardt, J. Condition Number Notes. https://people.sc.fsu.edu/~jburkardt/m_src/condition/condition.html
