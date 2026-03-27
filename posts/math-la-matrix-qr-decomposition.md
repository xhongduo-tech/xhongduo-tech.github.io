## 核心结论

QR 分解是把矩阵 $A$ 拆成“正交部分”与“三角部分”的方法，即
$$
A = QR,\qquad Q^TQ = I
$$
其中 $Q$ 的列彼此正交且长度为 1，白话说就是“方向互相垂直、每个方向都标准化”；$R$ 是上三角矩阵，白话说就是“只保留主对角线及其上方元素”。

对列满秩矩阵 $A\in \mathbb{R}^{m\times n}$，最常用的是薄 QR 分解：
$$
A = Q R,\qquad Q\in \mathbb{R}^{m\times n},\ R\in \mathbb{R}^{n\times n}
$$
其中 $Q$ 的列正交，$R$ 为可逆上三角矩阵。若再规定 $R$ 的对角线元素为正，则分解在可逆情形下可视为唯一。

它的重要性不在“把矩阵拆开”本身，而在两个直接用途：

1. 解最小二乘问题 $\min_x \|Ax-b\|_2$ 时，QR 比正规方程 $A^TAx=A^Tb$ 更稳定。
2. 计算特征值时，QR 迭代是标准路线，最后收敛到 Schur 上三角形式，白话说就是“把矩阵逐步变成几乎能直接读特征值的形状”。

一个玩具例子：二维中有两列向量 $a_1,a_2$，它们一般不是垂直的。QR 做的事，就是先从它们生成一组垂直方向 $q_1,q_2$，再用 $R$ 记录原向量在这些方向上的投影系数。于是“斜着的坐标系”被改写成“标准正交坐标系 + 系数表”。

---

## 问题定义与边界

问题定义很直接：给定矩阵 $A$，希望找到 $Q,R$，使得
$$
A = QR
$$
并满足：

- $Q$ 的列正交，即 $Q^TQ=I$
- $R$ 是上三角矩阵

当 $m\ge n$ 且 $A$ 列满秩时，这是最常见、最干净的场景。这里“列满秩”的意思是各列线性无关，白话说就是“没有哪一列能由其他列拼出来”。

例如
$$
A=\begin{bmatrix}1&1\\1&2\end{bmatrix}
$$
它是满秩矩阵，目标就是把它写成“正交矩阵乘以上三角矩阵”。

下表先把常见情形划清楚：

| 矩阵形状 | 是否列满秩 | 能否做标准薄 QR | 工程含义 |
|---|---:|---:|---|
| $m>n$ | 是 | 可以 | 最小二乘的主力场景 |
| $m=n$ | 是 | 可以 | 可逆矩阵分解、求解线性方程 |
| $m<n$ | 不可能列满秩 | 不能做标准列满秩 QR | 欠定系统，需要别的方法 |
| $m\ge n$ | 否 | 可做带主元 QR 或秩揭示 QR | 需要判断有效秩 |
| 列近线性相关 | 近似满秩 | 理论可做，数值上危险 | 容易失去正交性 |

最小二乘是 QR 最常见的落点。若 $A=QR$，则
$$
\min_x \|Ax-b\|_2
=
\min_x \|QRx-b\|_2
$$
由于正交矩阵不改变 2-范数，也就是“只旋转不拉伸长度”，有
$$
\|QRx-b\|_2 = \|Rx-Q^Tb\|_2
$$
因此问题变成解上三角系统
$$
Rx = Q^Tb
$$
这一步只需要回代，不需要构造 $A^TA$。

边界也必须讲清楚：

- 若矩阵不满秩，$R$ 会出现接近 0 的对角线，普通回代会失稳。
- 若列几乎相关，经典 Gram-Schmidt 虽然公式正确，但浮点误差会让 $Q$ 不再正交。
- 若矩阵很稀疏，直接用 Householder 可能破坏稀疏结构，此时 Givens 旋转往往更合适。

---

## 核心机制与推导

QR 的核心构造有两条主线：Gram-Schmidt 正交化与 Householder 反射。

### 1. Gram-Schmidt：逐列变正交

Gram-Schmidt 的思路是，依次处理每一列，把后面的列减去在前面正交基上的投影。这里“投影”就是“某个方向上的分量”。

设 $A=[a_1,a_2,\dots,a_n]$，构造 $Q=[q_1,q_2,\dots,q_n]$。

第一列：
$$
r_{11}=\|a_1\|_2,\qquad q_1=\frac{a_1}{r_{11}}
$$

第二列先减去在 $q_1$ 上的投影：
$$
r_{12}=q_1^Ta_2,\qquad
u_2=a_2-r_{12}q_1
$$
再归一化：
$$
r_{22}=\|u_2\|_2,\qquad q_2=\frac{u_2}{r_{22}}
$$

一般地，
$$
r_{ij}=q_i^Ta_j\quad (i<j),\qquad
u_j=a_j-\sum_{i=1}^{j-1}r_{ij}q_i,\qquad
q_j=\frac{u_j}{\|u_j\|_2}
$$

玩具例子就用
$$
A=\begin{bmatrix}1&1\\1&2\end{bmatrix}
$$
它的两列分别是
$$
a_1=\begin{bmatrix}1\\1\end{bmatrix},\quad
a_2=\begin{bmatrix}1\\2\end{bmatrix}
$$

先算第一列：
$$
q_1=\frac{1}{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix}
$$

再算第二列投影系数：
$$
r_{12}=q_1^Ta_2=\frac{3}{\sqrt2}
$$

减去投影后的残差：
$$
u_2=a_2-r_{12}q_1
=
\begin{bmatrix}1\\2\end{bmatrix}
-\frac{3}{2}\begin{bmatrix}1\\1\end{bmatrix}
=
\begin{bmatrix}-\frac12\\\frac12\end{bmatrix}
$$

归一化得到
$$
q_2=\frac{1}{\sqrt2}\begin{bmatrix}-1\\1\end{bmatrix}
$$

所以
$$
Q=
\begin{bmatrix}
\frac{1}{\sqrt2}&-\frac{1}{\sqrt2}\\
\frac{1}{\sqrt2}&\frac{1}{\sqrt2}
\end{bmatrix},
\qquad
R=
\begin{bmatrix}
\sqrt2&\frac{3}{\sqrt2}\\
0&\frac{1}{\sqrt2}
\end{bmatrix}
$$

这个推导清楚，但它有一个致命问题：浮点数里“先投影、再相减”会产生消去误差，尤其当两列几乎平行时，残差会非常小，误差就被放大。

### 2. Householder：一次反射消一整列

Householder 的做法不是一列列减投影，而是构造一个反射矩阵
$$
H = I - 2uu^T,\qquad \|u\|_2=1
$$
它表示“绕某个超平面做镜面反射”。白话说，就是“翻一下方向，但长度不变”。

目标是把某个列向量 $x$ 反射成与标准基 $e_1$ 同方向：
$$
Hx = \pm \|x\|_2 e_1
$$
这样，$x$ 除第一项外，其余项全部变成 0，于是当前列的下三角元素一次性被消掉。

第 1 步处理第一列 $x=A_{:,1}$。取
$$
v = x + \operatorname{sign}(x_1)\|x\|_2 e_1,\qquad
u=\frac{v}{\|v\|_2}
$$
然后构造
$$
H_1=I-2uu^T
$$
左乘后，$H_1A$ 的第一列就只剩第一行非零。

接着对右下角子块继续做同样操作，得到
$$
R = H_k\cdots H_2H_1A
$$
由于每个 $H_i$ 都正交，令
$$
Q = H_1H_2\cdots H_k
$$
便得到
$$
A=QR
$$

Householder 的关键优势是：每一步用的是正交变换，误差不会像经典 GS 那样在“投影减法”中积累得那么严重。这就是为什么 LAPACK 这类工业级库默认走 Householder 路线。

### 3. 为什么 QR 比正规方程稳

正规方程是
$$
A^TAx=A^Tb
$$
问题在于 $A^TA$ 的条件数大约是 $A$ 条件数的平方：
$$
\kappa(A^TA)\approx \kappa(A)^2
$$
“条件数”可以理解为“输入误差会被放大多少”的指标。平方意味着本来已经不太稳的问题，会进一步恶化。

QR 不需要显式构造 $A^TA$，只做正交变换和上三角回代，因此通常更稳。这不是理论上的小优势，而是数值线性代数里的标准实践。

---

## 代码实现

下面给出一个可运行的 Python 实现。为了可读性，这里实现经典 Gram-Schmidt 和最小二乘求解，同时用 `assert` 验证分解正确性。工业实现不会这样写，而是直接调用 LAPACK 的 Householder 版本。

```python
import numpy as np

def classical_gs_qr(A: np.ndarray):
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-14:
            raise ValueError("Matrix is rank deficient or nearly rank deficient")
        Q[:, j] = v / R[j, j]
    return Q, R

def back_substitution(R: np.ndarray, y: np.ndarray):
    n = R.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - R[i, i+1:] @ x[i+1:]) / R[i, i]
    return x

def qr_least_squares(A: np.ndarray, b: np.ndarray):
    Q, R = classical_gs_qr(A)
    y = Q.T @ b
    x = back_substitution(R, y)
    return x, Q, R

# 玩具例子
A = np.array([[1.0, 1.0],
              [1.0, 2.0]])
b = np.array([3.0, 5.0])

x, Q, R = qr_least_squares(A, b)

assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-10)
assert np.allclose(Q @ R, A, atol=1e-10)
assert np.allclose(A @ x, b, atol=1e-10)

# 过定方程最小二乘
A2 = np.array([[1.0, 1.0],
               [1.0, 2.0],
               [1.0, 3.0]])
b2 = np.array([1.0, 2.0, 2.0])

x2, Q2, R2 = qr_least_squares(A2, b2)
x2_np, *_ = np.linalg.lstsq(A2, b2, rcond=None)

assert np.allclose(Q2.T @ Q2, np.eye(2), atol=1e-10)
assert np.allclose(A2 @ x2, A2 @ x2_np, atol=1e-10)
print("all tests passed")
```

最小二乘的流程可以概括成：

```python
Q, R = householder_qr(A)
y = Q.T @ b
x = back_substitution(R, y)
```

真实工程里，通常不会显式生成完整的 $Q$，因为那样会多占内存。更常见的是只存 Householder 向量，并在需要时把这些反射依次作用到向量或矩阵上。

下面是常见 LAPACK 调用链：

| 函数 | 作用 | 说明 |
|---|---|---|
| `DGEQRF` | 计算 QR 分解 | 用 Householder 生成 $R$，并隐式存储反射向量 |
| `DORGQR` | 显式生成 $Q$ | 仅在确实需要完整 $Q$ 时调用 |
| `DORMQR` | 应用 $Q$ 或 $Q^T$ | 把正交变换作用到 $b$ 或别的矩阵上 |
| `DTRTRS`/回代例程 | 解上三角系统 | 对 $Rx=y$ 做回代 |
| 高层接口 `lstsq` | 最小二乘封装 | 底层一般仍走 QR 或 SVD |

一个真实工程例子：在线性回归中，设特征矩阵 $A\in\mathbb{R}^{10^6\times 50}$，目标是拟合参数 $x$。如果直接做 $A^TA$，虽然最终只得到 $50\times 50$ 的矩阵，但构造过程中会丢掉数值稳定性，且病态特征会被平方放大。用 QR 的流程是先把 $A$ 正交化成 $QR$，再解 $Rx=Q^Tb$。在科学计算、统计拟合、控制系统辨识里，这都是默认正路。

---

## 工程权衡与常见坑

最重要的坑不是“公式写错”，而是“数值上看起来没错，结果却已经坏了”。

### 1. 经典 GS 容易失去正交性

当列接近线性相关时，经典 GS 的误差很明显。比如
$$
A=\begin{bmatrix}
1 & 1\\
1 & 1+\varepsilon\\
1 & 1-\varepsilon
\end{bmatrix},\qquad \varepsilon \ll 1
$$
两列几乎平行。理论上第二列减去第一列投影后应得到很小但准确的残差；实际浮点计算时，这一步会丢失有效数字，导致算出的 $q_2$ 方向被噪声污染，最终 $Q^TQ$ 偏离单位阵。

Modified Gram-Schmidt 的做法是调整投影更新顺序，白话说就是“每消掉一个方向就立刻更新当前向量”，比经典 GS 稍稳，但在高病态问题上仍不如 Householder。

### 2. Householder 更稳定，但不一定最省结构

Householder 的优点是稳定、适合稠密矩阵、便于调用成熟库。代价是它会对整列或子块做更新，容易破坏稀疏结构。也就是说，原矩阵里很多 0 可能会被“填满”。

### 3. 别显式求逆

有些初学者会先算
$$
x=R^{-1}Q^Tb
$$
这是不推荐的。上三角系统应该直接回代，不该显式构造逆矩阵。原因是求逆更慢，也更不稳。

### 4. 近秩亏时要做秩判定

若 $R_{ii}$ 很小，说明矩阵可能近似秩亏。工程上通常会设置阈值，例如判断
$$
|R_{ii}| < \tau
$$
就把它视为数值 0，并切换到带列主元 QR 或 SVD。否则你会得到“形式上有解、实际上噪声驱动”的结果。

下表总结常见方案的差异：

| 方法 | 正交性保持 | 误差来源 | 适用场景 | 常见问题 |
|---|---|---|---|---|
| 经典 GS | 较差 | 投影后相减的消去误差 | 教学、小规模原型 | 列近相关时明显失稳 |
| Modified GS | 中等 | 仍有累计舍入误差 | 中小规模、需要简单实现 | 高病态时不够稳 |
| Householder | 好 | 主要是反射计算中的舍入误差 | 稠密矩阵、工业库默认 | 可能破坏稀疏性 |
| 带重正交 GS | 较好 | 额外投影次数带来的成本 | 特殊科研代码 | 性能差，仍非主流工业方案 |

一个实际坑是机器学习或数据分析里的特征标准化。如果某些列尺度差异极大，哪怕使用 Householder，问题本身也会更病态。QR 不是万能补丁，前处理仍然重要。

---

## 替代方案与适用边界

QR 不是唯一方案。选择方法取决于矩阵结构、稳定性要求和目标任务。

### 1. Givens 旋转：适合稀疏与局部更新

Givens 旋转只作用于两行或两列，用一个 $2\times2$ 旋转块把单个元素消成 0。白话说，就是“每次只修一个位置，不大面积改动”。

它特别适合：

- 稀疏矩阵，因为不容易引入大规模填充
- 增量更新问题，比如新来一条数据，需要在已有分解上做小修补

但对于一般稠密矩阵，批量效率通常不如 Householder。

### 2. SVD：病态或秩亏问题更稳

奇异值分解 SVD 把矩阵写成
$$
A=U\Sigma V^T
$$
它比 QR 更贵，但在秩亏、近秩亏、病态问题中更稳，也更适合做伪逆、降维和秩判断。若你怀疑数据强相关、噪声大、模型不可辨识，SVD 往往比 QR 更可靠。

### 3. QR 迭代：不是解方程，而是求特征值

QR 还有另一个完全不同的用途：特征值计算。做法是不断迭代
$$
A_k = Q_kR_k,\qquad A_{k+1}=R_kQ_k
$$
由于
$$
A_{k+1}=Q_k^TA_kQ_k
$$
每一步都与原矩阵相似，也就是“特征值不变，只改变表达形式”。合适的移位策略下，它会收敛到实 Schur 形式，即准上三角矩阵，特征值可从对角块读出。这是数值代数中的标准算法族。

下表给出选择边界：

| 方案 | 稠密/稀疏 | 动态更新 | 稳定性 | 主要用途 |
|---|---|---:|---:|---|
| Householder QR | 稠密优先 | 一般 | 高 | 最小二乘、正交化 |
| Givens QR | 稀疏优先 | 好 | 高 | 稀疏消元、在线更新 |
| SVD | 都可 | 一般 | 最高 | 病态问题、秩分析、伪逆 |
| QR 迭代 | 稠密优先 | 否 | 高 | 全部特征值、Schur 分解 |

所以边界可以概括成一句话：

- 解稳定的稠密最小二乘，用 Householder QR。
- 解稀疏或需要局部更新的问题，看 Givens。
- 怀疑秩亏或病态严重，直接考虑 SVD。
- 目标是特征值，不是线性方程，走 QR 迭代。

---

## 参考资料

1. Wikipedia, QR decomposition  
   用途：给出 QR 分解的定义、唯一性约定、薄 QR 与满 QR 的基本区别。  
   链接：https://en.wikipedia.org/wiki/QR_decomposition

2. Wikipedia, Gram-Schmidt process  
   用途：说明 Gram-Schmidt 的投影构造方式，以及经典版与修改版的区别。  
   链接：https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

3. UIC Numerical Linear Algebra Notes  
   用途：推导最小二乘如何通过 QR 化为上三角系统，说明避免 $A^TA$ 的价值。  
   链接：https://homepages.math.uic.edu/~jan/mcs572f16/mcs572notes/lec22.html

4. GNU Scientific Library Reference Manual, QR Decomposition  
   用途：展示工程库如何以 Householder 反射实现 QR，并给出接口语义。  
   链接：https://www.math.utah.edu/software/gsl/gsl-ref_204.html

5. LAPACK Documentation, `DGEQRF` / `DORGQR` / `DORMQR`  
   用途：对应工业实现中的标准调用链，理解“隐式存储 $Q$”的工程做法。  
   链接：https://www.netlib.org/lapack/

6. Golub & Van Loan, Matrix Computations  
   用途：系统讲清 QR、Householder、Givens、最小二乘与特征值算法，是工程实现与理论推导之间的桥梁。  
   链接：可检索该书对应版本目录页或教材资源页
