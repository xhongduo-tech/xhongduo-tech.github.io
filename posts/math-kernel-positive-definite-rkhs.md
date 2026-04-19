## 核心结论

正定核 `k(x,z)` 是一种有数学约束的相似度函数。它的合法性不取决于“看起来像不像相似度”，而取决于：对任意有限样本 $x_1,\dots,x_n$，由

$$
K_{ij}=k(x_i,x_j)
$$

组成的 Gram 矩阵都必须满足

$$
K \succeq 0
$$

这里的 Gram 矩阵是“样本两两相似度组成的矩阵”；半正定是指任意向量 $c$ 都有 $c^\top Kc \ge 0$。

RKHS，即再生核希尔伯特空间，是由核函数生成的函数空间。它的核心价值不是“把数据放进高维空间”这个说法本身，而是再生性质：

$$
f(x)=\langle f,k(x,\cdot)\rangle_{H_k}
$$

这句话的白话含义是：函数在某个点的取值，可以变成函数和一个核函数的内积。于是，训练和预测可以落到核矩阵上的线性代数计算。

| 符号 | 含义 | 工程角色 |
|---|---|---|
| `k(x,z)` | 核函数 | 直接计算两个样本的相似度 |
| `K` | Gram 矩阵 | 训练时的核心矩阵 |
| `H_k` | RKHS | 由核生成的函数空间 |
| `φ(x)` | 特征映射 | 隐式高维坐标 |
| `α` | 核展开系数 | 决定预测函数的权重 |

玩具例子：如果样本只有 `1` 和 `2`，用线性核 `k(x,z)=xz`，则核矩阵是

$$
K=
\begin{bmatrix}
1 & 2\\
2 & 4
\end{bmatrix}
$$

它的特征值是 $5$ 和 $0$，所以 $K \succeq 0$。这个核可以用于标准核方法。

---

## 问题定义与边界

核函数、特征映射和 RKHS 是三个不同层次的概念。核函数定义在输入空间上，直接接收两个样本；特征映射把样本映射到某个向量空间；RKHS 是函数组成的空间。

| 概念 | 定义位置 | 作用 |
|---|---|---|
| 核函数 `k(x,z)` | 输入空间上 | 计算样本间相似度 |
| 特征映射 `φ(x)` | 隐空间中 | 把样本映射到线性可处理空间 |
| Gram 矩阵 `K` | 有限样本上 | 检查核是否合法、支撑训练 |
| RKHS `H_k` | 函数空间中 | 让函数值评价可用内积表示 |

如果存在某个特征映射 `φ`，使得

$$
k(x,z)=\langle \phi(x),\phi(z)\rangle
$$

那么 `k` 就是在隐空间里做内积。核技巧的重点是：可以不显式写出 `φ(x)`，只计算 `k(x,z)`。

“正定核”在机器学习文章里常被宽松地用于表示“半正定核”。严格数学里，正定和半正定可能区分得更细。本文采用机器学习中的常见约定：正定核指 Gram 矩阵总是半正定的核，即 $K\succeq 0$。

不是所有“像相似度”的函数都能当核。例如，某个函数可能让距离近的样本分数更高，但只要它在某组有限样本上产生的 Gram 矩阵有负特征值，就不能直接用于标准核岭回归、核 SVM 或高斯过程。对有限训练集来说，工程上通常先检查当前数据对应的 Gram 矩阵是否半正定。

---

## 核心机制与推导

核技巧的起点是显式特征映射。假设每个输入 $x$ 都能变成隐空间里的向量 $\phi(x)$，线性模型可以写成

$$
f(x)=\langle w,\phi(x)\rangle
$$

如果训练算法只依赖样本之间的内积，那么所有

$$
\langle \phi(x_i),\phi(x_j)\rangle
$$

都可以替换为

$$
k(x_i,x_j)
$$

于是问题从“构造高维特征”变成“构造核矩阵”。

| 方法 | 需要显式 `φ(x)` | 计算核心 | 适用特点 |
|---|---|---|---|
| 显式特征法 | 是 | 向量内积 | 特征维度较低时简单 |
| 核技巧 | 否 | 核矩阵 | 非线性问题、小中样本更方便 |

对于训练样本 $x_1,\dots,x_n$，核矩阵为 $K_{ij}=k(x_i,x_j)$。若 `k` 是合法核，则任意向量 $c$ 满足：

$$
c^\top Kc
=
\sum_i\sum_j c_i c_j k(x_i,x_j)
\ge 0
$$

这保证了很多优化问题仍然是稳定的凸问题或良定义问题。

RKHS 进一步说明：由核生成的函数可以写成核函数的线性组合：

$$
f(x)=\sum_i \alpha_i k(x_i,x)
$$

在核岭回归中，目标是拟合标签 $y$，同时控制函数复杂度。常见解为：

$$
\alpha=(K+\lambda I)^{-1}y
$$

预测新样本 $x^*$ 时：

$$
f(x^*)=\sum_i \alpha_i k(x_i,x^*)
$$

这里 $\lambda$ 是正则项系数，白话解释是“防止模型为了贴合训练数据而变得过于极端”。

高斯过程回归也使用核函数，但解释不同。高斯过程把 `k` 视为协方差函数，也就是描述两个输入点的函数值应该如何一起变化。给定训练集后，后验均值可写成：

$$
\mu(x^*)=k_*^\top (K+\sigma^2 I)^{-1}y
$$

其中 $k_*=[k(x_1,x^*),\dots,k(x_n,x^*)]^\top$，$\sigma^2$ 是噪声方差。形式上它仍然是核展开。

真实工程例子：锂电池健康度 SOH 回归。输入可能包括循环次数、温度、内阻、充电片段统计特征等。样本通常不大，噪声较多，关系明显非线性。用 RBF 核的 `KernelRidge` 可以得到稳定的非线性回归；如果用 `GaussianProcessRegressor`，还能得到预测不确定性，用于维护告警。

---

## 代码实现

实现核岭回归通常分三步：构造核矩阵、求解线性系统、对新样本做核展开预测。

下面是一个可运行的最小 Python 例子。它不用显式高维特征，只用线性核矩阵完成训练和预测。

```python
import numpy as np

def linear_kernel(X, Z):
    return X @ Z.T

X = np.array([[1.0], [2.0]])
y = np.array([1.0, 0.0])
lam = 1.0

K = linear_kernel(X, X)
alpha = np.linalg.solve(K + lam * np.eye(len(X)), y)

x_star = np.array([[3.0]])
k_star = linear_kernel(X, x_star).reshape(-1)
pred = k_star @ alpha

assert np.allclose(K, np.array([[1.0, 2.0], [2.0, 4.0]]))
assert np.all(np.linalg.eigvalsh(K) >= -1e-12)
assert np.allclose(alpha, np.array([5.0 / 6.0, -1.0 / 3.0]))
assert np.allclose(pred, 0.5)

print(pred)
```

对应公式是：

$$
K_{ij}=k(x_i,x_j)
$$

$$
\alpha=(K+\lambda I)^{-1}y
$$

$$
f(x^*)=\sum_i \alpha_i k(x_i,x^*)
$$

使用 scikit-learn 时，可以直接调用 `KernelRidge`：

```python
import numpy as np
from sklearn.kernel_ridge import KernelRidge

X = np.array([[1.0], [2.0]])
y = np.array([1.0, 0.0])

model = KernelRidge(kernel="linear", alpha=1.0)
model.fit(X, y)

x_star = np.array([[3.0]])
pred = model.predict(x_star)

assert pred.shape == (1,)
print(pred[0])
```

| 参数 | 作用 | 常见影响 |
|---|---|---|
| `kernel` | 核类型 | 决定相似度结构 |
| `lambda / alpha` | 正则项 | 控制平滑程度 |
| `gamma` | RBF 等核参数 | 控制局部性 |
| `degree` | 多项式核次数 | 控制非线性强度 |

工程中更重要的是数据预处理和参数选择。对于 RBF 核、多项式核，输入尺度会直接影响核值。如果某个特征范围是 `0~1`，另一个特征范围是 `0~100000`，模型会主要被大尺度特征支配。

---

## 工程权衡与常见坑

核方法的主要成本来自核矩阵。若训练样本数为 $n$，存储 Gram 矩阵需要 $O(n^2)$，直接线性求解常见复杂度为 $O(n^3)$。这意味着几千条样本可能还能接受，几十万条样本通常不能直接使用完整核矩阵。

| 问题 | 现象 | 规避方式 |
|---|---|---|
| 核不是合法正定核 | 训练数值不稳定 | 检查 Gram 矩阵半正定性 |
| 特征尺度不一致 | 核值退化 | 标准化或归一化 |
| 样本量过大 | 计算慢、内存爆 | Nyström / 随机特征 |
| 参数设置不当 | 过拟合或欠拟合 | 交叉验证 |
| 术语混淆 | 文献理解错误 | 明确“正定”是否指半正定 |

RBF 核是常见选择：

$$
k(x,z)=\exp(-\gamma \|x-z\|^2)
$$

其中 $\gamma$ 控制局部性。$\gamma$ 太大时，只有非常接近的点才相似，模型容易过拟合；$\gamma$ 太小时，大量样本看起来都差不多，模型容易欠拟合。

标准化是核方法里的基础步骤。常见做法是用 `StandardScaler` 让每个特征接近零均值、单位方差，再接 `KernelRidge` 或 `GaussianProcessRegressor`。这不是形式问题，而是直接改变核矩阵的数值结构。

还有一个常见误解：核方法不是“自动更高级”。它只是把特定非线性结构编码进核函数，并在样本规模允许时高效利用这种结构。核选错、尺度错、正则项错，效果一样会很差。

---

## 替代方案与适用边界

核方法适合小样本、中等规模、非线性明显、需要稳定训练的场景。它也适合需要不确定性估计的任务，例如用高斯过程做小样本建模、实验设计、异常告警。

当样本数很大时，完整核矩阵会成为瓶颈。这时可以考虑显式近似或换模型。

| 方法 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| 核岭回归 | 简单、稳定、非线性强 | 规模大时慢 | 小中样本回归 |
| GPR | 可给不确定性 | 更重、更慢 | 小样本建模和告警 |
| 随机特征 | 可扩展 | 近似有误差 | 中大规模数据 |
| 神经网络 | 表达能力强 | 训练依赖较多 | 大数据、复杂模式 |
| 树模型 | 易用、鲁棒 | 光滑性差 | 表格数据强基线 |

Nyström 近似的思想是选一部分代表性样本近似整个核矩阵。随机傅里叶特征的思想是把某些平移不变核近似成显式有限维特征，使原来的核方法变成普通线性模型。两者都是在用近似换规模。

可以把完整核方法理解为“精确但重”，把随机特征和 Nyström 理解为“近似但更可扩展”。如果数据量只有几百到几千，完整核方法常常值得尝试；如果数据有几十万条，通常应优先考虑树模型、线性模型加显式特征、神经网络或核近似。

---

## 参考资料

| 类型 | 来源 |
|---|---|
| 理论原始文献 | Aronszajn, 1950. *Theory of Reproducing Kernels* |
| 经典教材 | *Learning with Kernels* |
| 工程实现 | scikit-learn `KernelRidge` |
| 工程实现 | scikit-learn `GaussianProcessRegressor` |
| 工程实现 | GPyTorch kernels |
| 源码对照 | `sklearn/kernel_ridge.py`、`sklearn/gaussian_process/_gpr.py` |

1. [Aronszajn, 1950. Theory of Reproducing Kernels](https://doi.org/10.1090/S0002-9947-1950-0051437-7)
2. [Learning with Kernels](https://mitpress.mit.edu/9780262194754/learning-with-kernels/)
3. [scikit-learn: KernelRidge](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
4. [scikit-learn: GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
5. [GPyTorch: Kernels](https://docs.gpytorch.ai/en/latest/kernels.html)
6. [scikit-learn kernel_ridge.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/kernel_ridge.py)
7. [scikit-learn _gpr.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/gaussian_process/_gpr.py)
