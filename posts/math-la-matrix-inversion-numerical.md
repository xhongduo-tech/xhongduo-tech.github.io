## 核心结论

矩阵求逆的数值方法，本质上是在有限精度计算机里，把一个方阵 $A$ 通过一系列稳定的消元操作变成单位阵 $I$，并把同样的操作同步施加到 $I$ 上，最终得到 $A^{-1}$。这里“有限精度”可以直接理解为：计算机只能保存有限位有效数字，所以算法不仅要“能算”，还要尽量“不放大误差”。

对一般的 $n\times n$ 稠密方阵，直接求逆的主成本是 $O(n^3)$。更细一点看，标准高斯消元或 LU 分解的主导浮点操作量大约是
$$
\frac{2}{3}n^3.
$$
这也是工程里一个非常稳定的经验结论：如果同一个矩阵 $A$ 需要被重复使用，那么“先分解、后复用”几乎总是优于“每次重新求逆”。

最重要的工程结论有五个：

| 结论 | 含义 | 工程影响 |
|---|---|---|
| 高斯消元能直接求逆 | 对增广矩阵 $[A\mid I]$ 做行变换 | 一次性得到整个逆矩阵 |
| LU 分解适合重复求解 | 把 $A$ 写成 $PA=LU$ | 同一个 $A$ 对多个 $b$ 只需重复代回 |
| 分块求逆依赖 Schur 补 | 利用局部块的逆恢复整体逆 | 适合块结构系统和增量更新 |
| 条件数决定精度损失 | $\kappa(A)=\|A\|\|A^{-1}\|$ | 病态矩阵会让结果失真 |
| 奇异或欠定时用伪逆 | $A^+$ 给出最小范数解 | 并非所有问题都该硬求经典逆 |

一个最小玩具例子是
$$
A=\begin{bmatrix}4&7\\2&6\end{bmatrix},
\qquad
A^{-1}=\frac{1}{10}\begin{bmatrix}6&-7\\-2&4\end{bmatrix}.
$$
这个例子适合看清步骤，但真正的工程价值不在手算，而在复用：如果同一个 $A$ 需要反复解不同右端项 $b$，应该复用分解结果，而不是反复显式构造 $A^{-1}$。

再把结论说得更直白一些：

- “求逆”是一个完整矩阵级别的任务。
- “解方程 $Ax=b$”通常只是向量级别的任务。
- 如果你只是想求 $x$，直接解线性系统，通常比先求逆再相乘更快、更稳。

---

## 问题定义与边界

问题定义很直接：给定一个方阵 $A\in\mathbb{R}^{n\times n}$，希望计算出一个矩阵 $A^{-1}$，使得
$$
AA^{-1}=A^{-1}A=I.
$$

这里“逆矩阵”可以先用一句直白的话理解：它是把线性变换“做回去”的那个矩阵。若 $A$ 把一个向量做了拉伸、旋转、剪切，那么 $A^{-1}$ 负责把这些效果撤销，让结果回到原来的坐标位置。

但这个定义只在两个前提下成立：

1. $A$ 必须是方阵。
2. $A$ 必须非奇异，即 $\det(A)\neq 0$，等价地说，它的列向量线性无关。

这两个条件缺一不可。因为只有当线性变换既“输入维度和输出维度一致”，又“没有把某些方向压扁成零”，它才可能被完整地反向恢复。

超出这个边界，经典逆就可能不存在。常见情况可以分成三类：

| 情况 | 是否有经典逆 | 常见处理 |
|---|---|---|
| 方阵且非奇异 | 有 | 高斯消元、LU、QR |
| 方阵但接近奇异 | 理论上可能有，数值上危险 | 主元选取、QR、正则化 |
| 非方阵或欠定/超定 | 无 | 伪逆 $A^+$、最小二乘 |

这一节最重要的边界量是条件数。它定义为
$$
\kappa(A)=\|A\|\|A^{-1}\|.
$$

“条件数”可以理解为：输入里一个很小的误差，经过这个矩阵问题后，最坏情况下会被放大多少倍。$\kappa(A)$ 越大，问题越病态，结果越容易对输入噪声、舍入误差或测量误差敏感。

一个很常用的经验公式是：如果输入大约有 $s$ 位有效数字，那么输出大约只剩
$$
s-\log_{10}\kappa(A)
$$
位有效数字。

例如，当
$$
\kappa(A)=10^8
$$
时，如果你使用双精度浮点数，通常只有约 16 位十进制有效数字，那么结果理论上可能只剩下
$$
16-8=8
$$
位可信数字。换句话说，程序即使没有报错，结果也未必可信。

对新手来说，可以把整个过程看成下面这条流水线：

$$
A \longrightarrow \text{消元操作} \longrightarrow I
$$

高斯消元做的事，就是不断用某一行去消掉其他行的非对角元素，把左边的矩阵一步步加工成单位阵。而逆矩阵是什么？就是把这些加工步骤完整记录下来，然后同步作用到单位阵 $I$ 上得到的结果。

这里还要区分两个经常被混淆的概念：

| 概念 | 含义 |
|---|---|
| 矩阵可逆 | 理论上存在严格的逆矩阵 |
| 问题数值上好解 | 在有限精度下能稳定算出可信结果 |

一个矩阵可以理论上可逆，但数值上非常难解。病态矩阵最典型：定义上有逆，但计算出来的逆可能带有巨大误差。

如果 $A$ 不可逆，或者虽然可逆但病态到误差不可控，就要切换问题定义。此时通常不再追求经典逆，而是使用 Moore-Penrose 伪逆 $A^+$。它满足
$$
AA^+A=A,
$$
并且对方程 $Ax=b$ 给出最小范数解。这里“最小范数解”可以理解为：在所有可行解里，长度最短、最不夸张的那个解。

---

## 核心机制与推导

高斯消元和 LU 分解，其实是在描述同一件事，只是视角不同。

高斯消元强调“操作过程”。把增广矩阵写成
$$
[A\mid I],
$$
然后通过初等行变换把左边的 $A$ 化成 $I$。如果这一步成功，右边自然就变成了 $A^{-1}$。

LU 分解强调“结构结果”。如果把消元过程固化下来，就可以把矩阵写成
$$
PA=LU.
$$
这里：

- $P$ 是置换矩阵，表示交换行，也就是主元选取的结果。
- $L$ 是下三角矩阵，记录每一步消元所用的系数。
- $U$ 是上三角矩阵，记录消元后留下的三角结构。

“主元”可以先用一句白话解释：它是当前这一步拿来做除法基准的元素。如果这个数太小，那么后续的除法会把舍入误差急剧放大，所以实际计算通常不会死守原始行顺序，而是做部分主元选取，也就是在当前列中挑一个绝对值最大的元素换到对角线上。

### 从行变换到逆矩阵

如果把每次行变换都记成一个初等矩阵 $E_k$，那么把 $A$ 化成单位阵的过程可以写成
$$
E_mE_{m-1}\cdots E_1A=I.
$$
于是
$$
A^{-1}=E_mE_{m-1}\cdots E_1.
$$

这正是“为什么把同样的操作作用到 $I$ 上会得到逆矩阵”的数学原因。因为
$$
E_mE_{m-1}\cdots E_1I=E_mE_{m-1}\cdots E_1=A^{-1}.
$$

### 为什么 LU 能复用

一旦有了
$$
PA=LU,
$$
求解线性方程
$$
Ax=b
$$
就等价于
$$
LUx=Pb.
$$
把中间变量记成 $y=Ux$，就变成两步：

$$
Ly=Pb,\qquad Ux=y.
$$

这两步分别叫前向代回和后向代回。它们都是三角系统求解，复杂度是 $O(n^2)$，远小于一次分解的 $O(n^3)$。因此：

- 第一次处理矩阵 $A$ 时，主要成本在分解。
- 之后每换一个新的 $b$，只需再做两次三角代回。

这就是“复用 LU”在工程里极其常见的原因。

### 玩具例子：2×2 直接求逆

取
$$
A=\begin{bmatrix}4&7\\2&6\end{bmatrix}.
$$
先写增广矩阵
$$
\left[
\begin{array}{cc|cc}
4 & 7 & 1 & 0\\
2 & 6 & 0 & 1
\end{array}
\right].
$$

第一步，用第一行消掉第二行第一列的 2。第二行减去第一行的一半，得到
$$
\left[
\begin{array}{cc|cc}
4 & 7 & 1 & 0\\
0 & 2.5 & -0.5 & 1
\end{array}
\right].
$$

第二步，把第二行除以 $2.5$，得到
$$
\left[
\begin{array}{cc|cc}
4 & 7 & 1 & 0\\
0 & 1 & -0.2 & 0.4
\end{array}
\right].
$$

第三步，用第二行消掉第一行第二列的 7，得到
$$
\left[
\begin{array}{cc|cc}
4 & 0 & 2.4 & -2.8\\
0 & 1 & -0.2 & 0.4
\end{array}
\right].
$$

第四步，把第一行除以 4，得到
$$
\left[
\begin{array}{cc|cc}
1 & 0 & 0.6 & -0.7\\
0 & 1 & -0.2 & 0.4
\end{array}
\right].
$$

因此
$$
A^{-1}=
\begin{bmatrix}
0.6 & -0.7\\
-0.2 & 0.4
\end{bmatrix}
=
\frac{1}{10}
\begin{bmatrix}
6 & -7\\
-2 & 4
\end{bmatrix}.
$$

这个例子虽然很小，但已经完整体现了“左边化成 $I$，右边变成 $A^{-1}$”的机制。

还可以立刻做一个正确性检查：
$$
\begin{bmatrix}4&7\\2&6\end{bmatrix}
\begin{bmatrix}0.6&-0.7\\-0.2&0.4\end{bmatrix}
=
\begin{bmatrix}1&0\\0&1\end{bmatrix}.
$$

### LU 的结构推导

对同一个矩阵
$$
A=\begin{bmatrix}4&7\\2&6\end{bmatrix},
$$
它的一个 LU 分解可以写成
$$
L=\begin{bmatrix}1&0\\0.5&1\end{bmatrix},
\qquad
U=\begin{bmatrix}4&7\\0&2.5\end{bmatrix},
\qquad
A=LU.
$$

为什么 $L$ 的左下角是 $0.5$？因为在消元时，第二行减去了第一行的 $\frac{2}{4}=0.5$ 倍。LU 分解的本质，就是把这些“消元倍数”集中记录到 $L$ 中，再把消元后的结果记录到 $U$ 中。

如果右端项是
$$
b=\begin{bmatrix}b_1\\b_2\end{bmatrix},
$$
先解
$$
Ly=b.
$$
按行写开就是
$$
\begin{cases}
y_1=b_1,\\
0.5y_1+y_2=b_2.
\end{cases}
$$
所以
$$
y_2=b_2-0.5b_1.
$$

再解
$$
Ux=y,
$$
也就是
$$
\begin{cases}
4x_1+7x_2=y_1,\\
2.5x_2=y_2.
\end{cases}
$$
先由第二式得到
$$
x_2=\frac{y_2}{2.5},
$$
再代回第一式得到
$$
x_1=\frac{y_1-7x_2}{4}.
$$

这个过程比先显式写出 $A^{-1}$ 再做矩阵乘法更符合工程实际，因为它直接服务于“解方程”这个目标。

对大矩阵，求逆和分解的主要成本都集中在消元阶段，因此复杂度主导项都是立方级。对稠密矩阵，常见数量级如下：

| 操作 | 一次成本 |
|---|---|
| LU 分解 | $O(n^3)$ |
| 显式构造整个逆 | $O(n^3)$ |
| 已有 LU 后解一个 $b$ | $O(n^2)$ |
| 已有 LU 后解多个右端矩阵 $B$ | $O(n^2k)$，其中 $k$ 是列数 |

这张表的关键不是背复杂度，而是理解一个事实：分解是昂贵的一次性成本，代回是便宜的重复成本。

### 分块矩阵与 Schur 补

很多工程矩阵不是普通的大方阵，而是天然带有块结构。例如约束优化、流固耦合、电路方程、卡尔曼滤波中的联合状态估计，都常出现块矩阵：
$$
M=
\begin{bmatrix}
A & B\\
C & D
\end{bmatrix}.
$$

假设 $A$ 可逆，定义 Schur 补
$$
S=D-CA^{-1}B.
$$

“Schur 补”可以用一句直白的话理解：先把左上块 $A$ 的影响消去以后，右下块真正剩下的有效系统。

如果 $A$ 和 $S$ 都可逆，则
$$
M^{-1}=
\begin{bmatrix}
A^{-1}+A^{-1}BS^{-1}CA^{-1} & -A^{-1}BS^{-1}\\
-S^{-1}CA^{-1} & S^{-1}
\end{bmatrix}.
$$

这条公式的价值不在手算，而在结构复用。因为在很多应用里：

- $A$ 对应一个已经能高效求解的子系统；
- $D$ 是约束或耦合块；
- 真正的难点集中在较小的补矩阵 $S$ 上。

一个典型场景是带约束的线性系统：
$$
\begin{bmatrix}
H & C^T\\
C & 0
\end{bmatrix}
\begin{bmatrix}
x\\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
g\\
h
\end{bmatrix},
$$
这里 $H$ 可能来自优化问题的 Hessian，$C$ 是约束矩阵，$\lambda$ 是拉格朗日乘子。直接求整个大矩阵的逆通常不划算，而通过 Schur 补消去 $x$，往往更自然。

---

## 代码实现

下面给出一个可直接运行的 Python 版本，实现带部分主元选取的 LU 分解、线性系统求解，以及“通过逐列求解构造逆矩阵”的做法。代码只依赖 Python 标准库，保存为 `lu_inverse_demo.py` 后可直接运行。

```python
from typing import List, Tuple

Matrix = List[List[float]]
Vector = List[float]

EPS = 1e-12


def copy_matrix(A: Matrix) -> Matrix:
    return [row[:] for row in A]


def identity(n: int) -> Matrix:
    I = [[0.0] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I


def matmul(A: Matrix, B: Matrix) -> Matrix:
    rows = len(A)
    cols = len(B[0])
    inner = len(B)
    C = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = A[i][k]
            for j in range(cols):
                C[i][j] += aik * B[k][j]
    return C


def matvec(A: Matrix, x: Vector) -> Vector:
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def max_abs_diff(A: Matrix, B: Matrix) -> float:
    diff = 0.0
    for i in range(len(A)):
        for j in range(len(A[0])):
            diff = max(diff, abs(A[i][j] - B[i][j]))
    return diff


def permutation_matrix(P: List[int]) -> Matrix:
    n = len(P)
    M = [[0.0] * n for _ in range(n)]
    for i, p in enumerate(P):
        M[i][p] = 1.0
    return M


def apply_permutation(P: List[int], b: Vector) -> Vector:
    return [b[p] for p in P]


def lu_factor(A: Matrix) -> Tuple[Matrix, Matrix, List[int]]:
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A must be a non-empty square matrix")

    U = copy_matrix(A)
    L = identity(n)
    P = list(range(n))

    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(U[i][k]))
        pivot_value = U[pivot][k]
        if abs(pivot_value) < EPS:
            raise ValueError("Matrix is singular or nearly singular")

        if pivot != k:
            U[k], U[pivot] = U[pivot], U[k]
            P[k], P[pivot] = P[pivot], P[k]
            for j in range(k):
                L[k][j], L[pivot][j] = L[pivot][j], L[k][j]

        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            U[i][k] = 0.0
            for j in range(k + 1, n):
                U[i][j] -= factor * U[k][j]

    return L, U, P


def forward_sub(L: Matrix, b: Vector) -> Vector:
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - s
    return y


def back_sub(U: Matrix, y: Vector) -> Vector:
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < EPS:
            raise ValueError("Zero pivot encountered during back substitution")
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]
    return x


def lu_solve(L: Matrix, U: Matrix, P: List[int], b: Vector) -> Vector:
    if len(b) != len(L):
        raise ValueError("Dimension mismatch between A and b")
    pb = apply_permutation(P, b)
    y = forward_sub(L, pb)
    return back_sub(U, y)


def inverse_from_lu(L: Matrix, U: Matrix, P: List[int]) -> Matrix:
    n = len(L)
    cols = []
    for col in range(n):
        e = [0.0] * n
        e[col] = 1.0
        x = lu_solve(L, U, P, e)
        cols.append(x)

    A_inv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        for i in range(n):
            A_inv[i][j] = cols[j][i]
    return A_inv


def print_matrix(name: str, A: Matrix) -> None:
    print(f"{name} =")
    for row in A:
        print("  ", ["{:.6f}".format(v) for v in row])
    print()


def main() -> None:
    A = [
        [4.0, 7.0],
        [2.0, 6.0],
    ]

    L, U, P = lu_factor(A)
    Pm = permutation_matrix(P)

    print_matrix("A", A)
    print_matrix("P", Pm)
    print_matrix("L", L)
    print_matrix("U", U)

    PA = matmul(Pm, A)
    LU = matmul(L, U)
    print("max |PA - LU| =", max_abs_diff(PA, LU))
    print()

    b1 = [1.0, 0.0]
    x1 = lu_solve(L, U, P, b1)
    print("solve A x = e1 -> x =", x1)
    print("A x =", matvec(A, x1))
    print()

    b2 = [0.0, 1.0]
    x2 = lu_solve(L, U, P, b2)
    print("solve A x = e2 -> x =", x2)
    print("A x =", matvec(A, x2))
    print()

    A_inv = inverse_from_lu(L, U, P)
    print_matrix("A_inv", A_inv)

    I = identity(len(A))
    A_Ainv = matmul(A, A_inv)
    print_matrix("A * A_inv", A_Ainv)
    print("max |A*A_inv - I| =", max_abs_diff(A_Ainv, I))


if __name__ == "__main__":
    main()
```

如果运行正常，输出里应当能看到：

- $P A$ 与 $L U$ 的误差非常小；
- 用 $e_1,e_2$ 做右端项解出的两列，正好构成 $A^{-1}$；
- $A A^{-1}$ 与单位阵 $I$ 的最大误差接近机器精度。

这段代码体现了两个核心接口：

```python
L, U, P = lu_factor(A)
x = lu_solve(L, U, P, b)
```

第一次调用 `lu_factor(A)` 支付的是分解成本。只要 $A$ 不变，之后无论右端项 $b$ 怎么变，都只需要重复调用 `lu_solve(...)`。这就是 LU 在多次求解时性能更好的根本原因。

如果一定要构造整个逆矩阵，也不应该直接硬写“代数公式”，而是把单位阵的每一列看成一个右端项。也就是逐列求解
$$
Ax_i=e_i,\qquad i=1,2,\dots,n,
$$
然后把所有解向量拼成
$$
A^{-1}=[x_1,x_2,\dots,x_n].
$$

这件事也解释了一个常见误区：计算逆矩阵，本质上并不是一个完全不同的问题，而是“解 $n$ 个右端项分别为单位向量的线性系统”。

---

## 工程权衡与常见坑

第一条原则：工程里通常不应该“为了求解 $Ax=b$ 而先显式求逆”。

原因并不抽象，可以直接拆成三点：

| 做法 | 额外成本 | 数值风险 | 是否推荐 |
|---|---|---|---|
| 直接解 $Ax=b$ | 只需分解 + 代回 | 较小 | 推荐 |
| 先求 $A^{-1}$ 再算 $x=A^{-1}b$ | 要额外构造整个逆矩阵 | 更容易累计误差 | 通常不推荐 |

为什么误差会更大？因为显式逆矩阵本身就是一个完整的数值对象。你先在有限精度下构造了它，再拿它去乘 $b$，等于把误差经历了“两段传播”：先在求逆时传播一次，再在乘法时传播一次。

### 真实工程例子：隐式时间步进

在偏微分方程离散、有限元、热传导、结构力学或流体模拟中，常见形式是每个时间步都要求解
$$
A_kx_k=b_k.
$$

如果时间步很小，材料参数或边界条件变化不剧烈，经常会出现：

- 多个时间步中矩阵完全相同；
- 或者矩阵只有小幅变化，但右端项频繁变化。

这种场景下，最常见的优化策略是：

1. 对当前的 $A_k$ 做一次 LU 分解。
2. 后续多个时间步只更新 $b_k$。
3. 通过前向代回和后向代回快速得到新解。

这可以理解成“先把机器拆开一次，以后只换输入件”。如果每一步都重新做完整消元，就会反复支付最贵的那部分成本。

### 常见风险与对策

| 风险 | 现象 | 原因 | 对策 |
|---|---|---|---|
| 主元太小 | 除法后数值突然变大 | 小数作分母会放大误差 | 部分主元选取 |
| 条件数很大 | 结果对微小扰动极敏感 | 问题本身病态 | 改用 QR、SVD、正则化 |
| 明明只要求解却显式求逆 | 更慢且更不稳 | 做了不必要的完整矩阵计算 | 直接解 $Ax=b$ |
| 接近奇异仍强行 LU | 解震荡、溢出或完全失真 | 对角主元接近 0 | 检查阈值并降级算法 |
| 分块公式直接硬套 | 推导正确但实现失败 | 子块或 Schur 补不可逆 | 先验证可逆性和维度关系 |
| 把理论可逆当作数值可解 | 程序不报错但答案不可信 | 可逆性不等于稳定性 | 结合残差和条件数判断 |

关于精度，最值得记住的仍是
$$
\text{有效数字} \approx s-\log_{10}\kappa(A).
$$
这不是严格的精确上界，但足够解释很多“代码没报错，结果却不靠谱”的现象。

例如 Hilbert 矩阵
$$
H_n(i,j)=\frac{1}{i+j-1}
$$
是经典病态矩阵。它的每个元素都很普通，但随着维度上升，条件数会迅速增大。你可能用双精度顺利算出一个“逆矩阵”，但乘回去以后：

- 并不真的接近单位阵；
- 或者只有前几位看起来正常；
- 稍微改动输入，输出就发生大幅波动。

这不是实现一定写错，而是问题本身已经接近数值不可解。

工程里还应同时看两个量：

1. 残差：$\|Ax-b\|$
2. 条件数：$\kappa(A)$

原因是：

- 残差小，只说明“算出来的 $x$ 代回去还算说得过去”；
- 条件数大，则说明“即使残差小，解本身也可能对输入极敏感”。

两者不能互相替代。

---

## 替代方案与适用边界

高斯消元和 LU 分解是默认方案，但不是唯一方案。真正的算法选择，取决于矩阵结构、精度要求和问题目标。

| 方法 | 适用对象 | 稳定性 | 性能 | 典型用途 |
|---|---|---|---|---|
| 高斯消元 / LU | 一般非奇异稠密方阵 | 中等，需主元选取 | 快 | 通用线性求解、多右端复用 |
| QR 分解 | 最小二乘、较病态问题 | 更稳定 | 比 LU 略慢 | 回归、过定系统 |
| SVD / 伪逆 | 欠定、超定、近奇异 | 最稳定 | 最慢 | 最小范数解、秩亏问题 |
| 分块逆 / Schur 补 | 结构化块矩阵 | 依赖子块性质 | 结构利用后很高效 | 约束系统、耦合系统 |

### 何时改用 QR

QR 分解把矩阵写成
$$
A=QR,
$$
其中 $Q$ 是正交矩阵，$R$ 是上三角矩阵。

“正交”可以理解为：列向量两两垂直，而且长度保持不变。更具体地说，
$$
Q^TQ=I.
$$
这意味着乘上 $Q$ 或 $Q^T$ 时，不会像一般矩阵那样随意放大向量长度，因此数值稳定性通常更好。

QR 特别适合最小二乘问题。比如对于过定系统
$$
Ax\approx b,
$$
我们不是寻找严格满足的解，而是寻找让误差
$$
\|Ax-b\|_2
$$
最小的解。此时 QR 往往比直接从法方程
$$
A^TAx=A^Tb
$$
出发更稳定，因为法方程会把条件数平方放大。

一个常见经验是：
$$
\kappa(A^TA)\approx \kappa(A)^2.
$$
因此当矩阵已经有点病态时，直接构造法方程通常不是好主意。

### 何时改用伪逆

当矩阵不是方阵，或者秩亏，经典逆不存在，此时应转向伪逆：
$$
x=A^+b.
$$

如果通过 SVD 写成
$$
A=U\Sigma V^T,
$$
则伪逆为
$$
A^+=V\Sigma^+U^T.
$$

这里 $\Sigma^+$ 的做法是：

- 对非零奇异值取倒数；
- 对为零或非常接近零的奇异值，不再直接取倒数，而是按阈值截断。

“奇异值”可以理解为矩阵沿不同方向拉伸空间的强度。某个奇异值非常小，就说明某个方向几乎被压扁了，这正是不可逆或病态的来源。

伪逆的两个典型用途是：

| 场景 | 经典逆是否存在 | 伪逆给出什么 |
|---|---|---|
| 过定系统 | 不存在 | 最小二乘解 |
| 欠定系统 | 不存在 | 最小范数解 |

例如，当未知数比方程多时，满足 $Ax=b$ 的解可能有无穷多个。伪逆给出的那个解，是欧几里得长度最小的解，也就是最小范数解。这在信号重建、参数辨识、冗余机械臂控制里都很常见。

### 何时用 Schur 补

如果系统天然具有块结构，例如
$$
\begin{bmatrix}
A & B\\
C & D
\end{bmatrix},
$$
而且你已经能高效处理 $A$ 或 $D$，那么 Schur 补非常合适。它不是更通用的替代品，而是更结构化的工程工具。

它常见于：

- 稀疏直接法中的消元顺序设计；
- 约束系统中的变量消去；
- 域分解方法中的接口方程；
- 并行预条件器中的局部块求解。

边界也必须说清楚：Schur 补只有在相关子块可逆、补矩阵本身数值性质还能接受时，才是真正高效的方案。否则它只是把难题转移到另一块上，而不是消除难题。

一个实用判断标准是：如果某个子块已经有成熟求解器，且补矩阵维度明显更小，那么 Schur 补往往值得考虑；否则直接在原系统上处理可能更简单。

---

## 参考资料

- Stanford Math 114 Gaussian Elimination Notes: 讨论高斯消元的操作量 $\frac{2}{3}n^3$、消元流程以及数值稳定性的基本背景。https://web.stanford.edu/class/math114/decks/linear_systems/gauss_elim37.html
- CS 357, University of Illinois Condition Number Notes: 说明条件数 $\kappa(A)$ 对误差放大的影响，并给出有效数字损失的直观解释。https://cs357.cs.illinois.edu/textbook/notes/condition.html
- DataScienceBase LU Decomposition Overview: 介绍 $A=LU$、$Ly=Pb$、$Ux=y$ 的计算流程，以及多次求解时复用分解的优势。https://www.datasciencebase.com/intermediate/linear-algebra/lu-decomposition/
- Wikipedia Block Matrix / Schur Complement: 给出块矩阵求逆公式、Schur 补定义及其标准表达式，适合查公式。https://en.wikipedia.org/wiki/Block_matrix
- Wikipedia Moore-Penrose Inverse: 总结伪逆的定义、四个 Penrose 条件与最小范数解的性质。https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
- Trefethen and Bau, *Numerical Linear Algebra*: 数值线性代数的经典教材，适合理解“可逆”和“数值稳定”为什么是两件不同的事。
- Golub and Van Loan, *Matrix Computations*: 系统覆盖 LU、QR、SVD、块矩阵与实际数值算法，是工程实现层面的权威参考。
