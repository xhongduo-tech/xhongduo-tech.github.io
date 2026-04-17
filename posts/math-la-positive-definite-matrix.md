## 核心结论

正定矩阵与半正定矩阵，本质上是在判断一个对称矩阵对应的二次型 $x^\top A x$ 在所有方向上的符号。

术语先解释：

- 二次型：就是把向量 $x$ 左右各乘一次矩阵 $A$ 得到的标量，可以理解为“这个方向上的二阶能量”。
- 正定矩阵：对任意非零向量 $x$，都有 $x^\top A x>0$。
- 半正定矩阵：对任意非零向量 $x$，都有 $x^\top A x\ge 0$。

对实对称矩阵 $A$，下面三条对“正定”是等价的，可以互相推出：

| 判定角度 | 条件 |
|---|---|
| 二次型 | 对任意 $x\ne 0$，$x^\top A x>0$ |
| 特征值 | 所有特征值都大于 0 |
| Sylvester 准则 | 所有顺序主子式都大于 0 |
| 分解形式 | 存在非奇异下三角矩阵 $L$，使 $A=LL^\top$ |

这组等价关系很重要，因为它把“定义”“可计算判定”“数值分解”连成了一条线。理论上你可以从特征值理解，手算时可以用主子式，工程实现里通常直接尝试 Cholesky 分解。

半正定矩阵与正定矩阵的差别，在于是否允许某些方向上的能量恰好为 0。协方差矩阵
$$
\Sigma = \mathbb E[(x-\mu)(x-\mu)^\top]
$$
天然是半正定，因为任意方向 $u$ 上都有
$$
u^\top \Sigma u = \mathbb E[(u^\top(x-\mu))^2]\ge 0.
$$
如果某些维度线性相关，某个方向上的方差就会变成 0，此时最小特征值等于 0，矩阵只能是半正定，不能是正定。

---

## 问题定义与边界

本文讨论的对象是**实对称矩阵**。这是边界条件，不满足这个条件，很多结论要么不成立，要么需要改写。

为什么必须强调“对称”？

因为正定矩阵的大部分常用结论，比如“特征值全正”“Cholesky 分解存在”，默认都建立在实对称矩阵上。工程里常见的协方差矩阵、核矩阵、Hessian 矩阵，通常都属于这一类。

一个最小的玩具例子：

设
$$
A=\begin{pmatrix}2&1\\1&2\end{pmatrix}.
$$
则
$$
x^\top A x = 2x_1^2+2x_1x_2+2x_2^2.
$$
把它配方：
$$
2x_1^2+2x_1x_2+2x_2^2
= \left(x_1+x_2\right)^2 + x_1^2+x_2^2.
$$
只要 $(x_1,x_2)\ne (0,0)$，这个值就严格大于 0，所以它是正定矩阵。

对应的特征值是 $3$ 和 $1$，都为正；顺序主子式是 $2$ 和 $3$，也都为正；并且它有 Cholesky 分解。这正好把三种判定方式串起来。

再看边界例子：
$$
B=\begin{pmatrix}1&1\\1&1\end{pmatrix}.
$$
它的两个特征值是 $2$ 和 $0$。此时
$$
x^\top B x=(x_1+x_2)^2\ge 0,
$$
但当 $x=(1,-1)$ 时取到 0，所以它是半正定，不是正定。

这个例子对应真实含义也很直接：如果两个特征完全重复，那么系统实际上只有一个有效方向，另一个方向的信息是冗余的。

| 输入结构 | 最小特征值 | 类型 | Cholesky 状态 | 建议 |
|---|---:|---|---|---|
| 各维独立且有波动 | $>0$ | 正定 | 可直接分解 | 正常使用 |
| 存在线性相关 | $=0$ | 半正定 | 可能失败 | 降维或加 $\epsilon I$ |
| 数值误差较大 | 可能略小于 0 | 理论上 PSD，数值上不稳定 | 常报错 | 对称化并正则化 |

---

## 核心机制与推导

先看“特征值全正”和“正定”的关系。

如果 $A$ 是实对称矩阵，就可以正交对角化：
$$
A = Q\Lambda Q^\top,
$$
其中 $Q$ 是正交矩阵，$\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_n)$ 是特征值对角阵。

令 $y=Q^\top x$，则
$$
x^\top A x = x^\top Q\Lambda Q^\top x = y^\top \Lambda y = \sum_{i=1}^n \lambda_i y_i^2.
$$
于是结论立刻出来：

- 如果所有 $\lambda_i>0$，那么只要 $x\ne 0$，就有 $y\ne 0$，所以 $\sum \lambda_i y_i^2>0$，即 $A$ 正定。
- 如果存在某个 $\lambda_i\le 0$，取 $y$ 只在该坐标非零，就能构造出不满足正定定义的方向。

这就是“特征值判定”的根。

再看 Sylvester 准则。它说的是：对实对称矩阵，正定当且仅当前导主子式全部为正：
$$
\det(A_{1:k,1:k})>0,\quad k=1,\dots,n.
$$
白话解释：不是检查所有可能的子矩阵，只检查从左上角开始逐级扩大的那一串子矩阵，就够了。对低维手算很方便，对理论推导也很常见。

再看 Cholesky 分解：
$$
A=LL^\top,
$$
其中 $L$ 是下三角矩阵。它的意义不是“另一个表达式”这么简单，而是把正定性转成了可计算结构。

因为如果 $A=LL^\top$ 且 $L$ 非奇异，那么
$$
x^\top A x=x^\top LL^\top x = \|L^\top x\|_2^2 >0,\quad x\ne 0.
$$
平方和一定非负，而非奇异保证除零向量外不会压成 0，所以它一定正定。

这也是为什么 Cholesky 既是判定工具，也是求解线性方程组的高效算法。对于 $n\times n$ 正定矩阵：

- Cholesky 计算量约为 $n^3/3$
- 一般 LU 分解约为 $2n^3/3$

也就是前者大约只需要后者一半的浮点运算。

真实工程例子是 Hessian 矩阵。Hessian 就是多元函数二阶偏导组成的矩阵，可以理解为“各方向曲率的总表”。

若 $x^\star$ 是驻点，且 Hessian $H(x^\star)$ 正定，则二阶近似
$$
f(x^\star+\Delta x)\approx f(x^\star)+\frac12 \Delta x^\top H(x^\star)\Delta x
$$
在所有非零方向上都增加，所以 $x^\star$ 是严格局部极小值。

例如
$$
f(x,y)=x^2+y^2,
$$
其 Hessian 为
$$
H=\begin{pmatrix}2&0\\0&2\end{pmatrix},
$$
显然正定，所以原点是严格局部极小值。

但如果 Hessian 只是半正定，例如
$$
f(x,y)=x^4+y^2,
$$
在原点的 Hessian 为
$$
\begin{pmatrix}0&0\\0&2\end{pmatrix},
$$
它只能说明“没有明显向下弯的方向”，不能直接给出严格二阶结论，因为 $x$ 方向二阶项退化了，必须看更高阶项。

---

## 代码实现

工程里最常见的流程是三步：

1. 先检查矩阵是否近似对称。
2. 用特征值或 Sylvester 准则做判定。
3. 如果需要高效求解，再做 Cholesky 分解。

下面给出一个可运行的 Python 例子，覆盖正定、半正定、协方差三个场景。

```python
import numpy as np

def is_symmetric(A, tol=1e-10):
    return np.allclose(A, A.T, atol=tol)

def is_positive_definite_by_eig(A, tol=1e-10):
    eigvals = np.linalg.eigvalsh(A)
    return np.min(eigvals) > tol, eigvals

def is_positive_semidefinite_by_eig(A, tol=1e-10):
    eigvals = np.linalg.eigvalsh(A)
    return np.min(eigvals) >= -tol, eigvals

def sylvester_positive_definite(A, tol=1e-10):
    n = A.shape[0]
    for k in range(1, n + 1):
        minor = np.linalg.det(A[:k, :k])
        if minor <= tol:
            return False
    return True

# 玩具例子：正定矩阵
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])

assert is_symmetric(A)
pd_flag, eigvals_A = is_positive_definite_by_eig(A)
assert pd_flag
assert sylvester_positive_definite(A)

L = np.linalg.cholesky(A)
assert np.allclose(L @ L.T, A)

# 边界例子：半正定但非正定
B = np.array([[1.0, 1.0],
              [1.0, 1.0]])

psd_flag, eigvals_B = is_positive_semidefinite_by_eig(B)
pd_flag_B, _ = is_positive_definite_by_eig(B)

assert psd_flag
assert not pd_flag_B
assert np.isclose(np.min(eigvals_B), 0.0)

# 协方差矩阵一定半正定
X = np.array([
    [1.0, 2.0],
    [2.0, 4.0],
    [3.0, 6.0],
    [4.0, 8.0],
])

Sigma = np.cov(X.T, bias=False)
psd_cov, eigvals_cov = is_positive_semidefinite_by_eig(Sigma)

assert is_symmetric(Sigma)
assert psd_cov
assert np.min(eigvals_cov) <= 1e-10  # 两列完全线性相关，所以最小特征值接近 0

print("A eigenvalues:", eigvals_A)
print("B eigenvalues:", eigvals_B)
print("Sigma:\n", Sigma)
print("Sigma eigenvalues:", eigvals_cov)
print("Cholesky L:\n", L)
```

这个例子说明三件事：

- 对正定矩阵，特征值判定、Sylvester 准则、Cholesky 分解三者是一致的。
- 对半正定矩阵，特征值允许出现 0，但 Cholesky 往往会失败。
- 协方差矩阵在理论上至少是半正定；如果样本维度线性相关，它通常不会是正定。

真实工程里，若矩阵来自浮点计算，通常会先做一次对称化：
```python
A = 0.5 * (A + A.T)
```
这是因为数值误差会把本来对称的矩阵弄成“几乎对称”，而线性代数库通常要求严格对称输入。

---

## 工程权衡与常见坑

正定问题真正难的地方不在定义，而在“理论上对，数值上不稳”。

第一类坑是**理论 PSD，计算时却看起来不是 PSD**。原因通常是浮点误差。比如协方差矩阵按定义应该半正定，但算出来最小特征值可能是 $-10^{-12}$。这不代表模型错了，更可能是数值舍入造成的。

第二类坑是**线性相关导致 Cholesky 失败**。协方差矩阵最典型。若某个特征是另一个特征的线性组合，就会出现零特征值。此时矩阵仍然合法，但不再是正定矩阵。

第三类坑是**把 Hessian 半正定误当成最小值结论**。半正定只能说明“二阶上没有发现下降方向”，不能自动推出严格局部极小值。鞍点、平台区、退化极值都可能出现。

第四类坑来自优化器。Adam 中二阶矩
$$
v_t=\beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$
按元素平方累积，所以每个分量都非负。再加上 $\epsilon$ 后，分母
$$
\sqrt{\hat v_t}+\epsilon
$$
严格为正，因此它对应一个正定的对角预条件器。白话解释：每个参数坐标都分到一个正的缩放系数，不会出现除零。

但这里也有工程边界：

- 如果 $\epsilon$ 太小，数值上仍可能不稳。
- 如果 $\beta_2$ 过大，二阶矩更新太慢，历史信息会压住当前梯度。
- 如果某些参数长期梯度极小，预条件器会非常尖锐，学习率表现会失衡。

| 问题 | 现象 | 常见对策 |
|---|---|---|
| 协方差线性相关 | 最小特征值为 0，Cholesky 失败 | 删除重复特征、PCA 降维、加 $\epsilon I$ |
| Hessian 仅半正定 | 二阶检验不给确定结论 | 看高阶项、换优化视角、检查邻域 |
| 数值误差导致微小负特征值 | 理论正确但实现报错 | 对称化、截断小负值、做正则 |
| Adam 漏掉 $\epsilon$ 或过小 | 更新震荡或除零风险 | 保留 $\epsilon$，用稳定默认超参 |

---

## 替代方案与适用边界

如果矩阵不是严格正定，但任务又需要“像正定一样可分解”，常见处理有三类。

第一类是正则化：
$$
A_{\text{reg}} = A + \epsilon I.
$$
这叫对角加载。白话解释：给每个方向都补一点最小曲率，把原来为 0 的方向抬起来。它最常见于协方差估计、核方法、高斯过程、牛顿法近似。

适用边界：

- 需要 Cholesky。
- 可以接受轻微改变原矩阵。
- 主要目标是数值稳定。

第二类是改用允许半正定或不定矩阵的分解，例如 $LDL^\top$。它不要求所有特征值都严格为正，因此更适合处理退化 Hessian 或边界协方差。

适用边界：

- 你想保留原矩阵结构。
- 你需要知道“到底是零特征值还是负特征值”。
- 你处理的是更一般的对称矩阵。

第三类是直接走谱方法或 SVD。SVD 就是奇异值分解，可以理解为把矩阵拆成“旋转 + 缩放 + 再旋转”。当矩阵退化时，它比直接求逆更稳。

适用边界：

- 需要伪逆而不是普通逆。
- 关心秩缺失问题。
- 数据本身高维且冗余明显。

真实工程中可以这样选：

| 场景 | 首选方法 | 原因 |
|---|---|---|
| 高斯模型采样、核方法 | $A+\epsilon I$ 后做 Cholesky | 快，接口简单 |
| Hessian 分析 | 特征值分解或 $LDL^\top$ | 能看清退化与不定 |
| 高维冗余特征 | PCA/SVD | 先去掉无效方向 |
| 神经网络训练 | Adam/RMSProp 这类对角预条件 | 不显式构造完整 Hessian |

不要把“加 $\epsilon I$”理解成万能修复。它只是把问题从“不可分解”改成“可分解”，但也会改变原问题的几何结构。如果 $\epsilon$ 大到和主特征值同量级，结论就可能失真。更稳妥的做法是先判断问题来源：是数据共线、模型退化，还是单纯数值误差。

---

## 参考资料

- ProofWiki, *Equivalence of Definitions of Positive Definite Matrix*  
- Wikipedia, *Sylvester's criterion*  
- Wikipedia, *Cholesky decomposition*  
- Math StackExchange, 关于 Hessian 第二导数检验的讨论  
- Math StackExchange, 关于协方差矩阵半正定性的讨论  
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015  
- HandWiki, *Cholesky decomposition*
