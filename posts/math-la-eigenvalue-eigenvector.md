## 核心结论

特征值与特征向量描述的是一种“方向保持不变的线性变换”。线性变换可以理解成“矩阵把向量拉伸、压缩、翻转或旋转”的规则。如果存在非零向量 $v$，使得

$$
A v = \lambda v
$$

那么 $v$ 就是矩阵 $A$ 的特征向量，$\lambda$ 就是对应的特征值。白话说，向量经过矩阵作用后，方向没有偏离原来的直线，只是长度变了，倍率就是 $\lambda$。

求特征值的入口是特征方程：

$$
\det(A-\lambda I)=0
$$

这里的 $\det$ 是行列式，可以理解成“矩阵是否可逆的判别量”；$I$ 是单位矩阵，也就是“对角线为 1、其他位置为 0 的基准矩阵”。这个方程的根就是特征值。根重复出现几次，叫代数重数；对应解空间的维度，叫几何重数。白话说，代数重数是“公式里这个根出现了几次”，几何重数是“真的能找出多少个彼此独立的方向”。

玩具例子先看：

$$
A=\begin{bmatrix}2&1\\1&2\end{bmatrix}
$$

这个矩阵会把平面上的向量沿两条特殊方向分别拉伸。对应的两条方向是 $[1,1]^T$ 和 $[1,-1]^T$，对应的拉伸倍数分别是 3 和 1。也就是说，一条方向被放大 3 倍，另一条方向长度不变。

对角化是理解特征分解的核心。对角化可以理解成“先换到一组特殊坐标系，在这个坐标系里矩阵只做各轴独立缩放”。如果存在可逆矩阵 $V$ 使得

$$
A=V\Lambda V^{-1}
$$

其中 $\Lambda$ 是对角矩阵，那么 $A$ 可对角化。其充要条件是：矩阵有 $n$ 个线性无关特征向量，也等价于每个特征值的代数重数等于几何重数。

如果矩阵还是实对称矩阵，即 $A=A^T$，结论更强。实对称矩阵一定可以正交对角化：

$$
A=Q\Lambda Q^T
$$

这里 $Q$ 的列向量是一组单位正交特征向量。白话说，这组方向彼此垂直、长度都为 1，所以数值稳定、几何意义也最清楚。

真实工程里，PCA 的核心操作就是对协方差矩阵做特征分解。协方差矩阵可以理解成“各个特征一起变化的统计描述”。第 $k$ 个主成分方向，就是第 $k$ 大特征值对应的特征向量；特征值越大，说明该方向保留的数据方差越多。

---

## 问题定义与边界

问题本身很明确：给定一个方阵 $A$，找出所有满足

$$
A v=\lambda v,\quad v\neq 0
$$

的 $\lambda$ 和 $v$。

为什么一定要求 $v\neq 0$？因为零向量无论乘什么矩阵都还是零，没有信息量，不能区分任何方向。

标准求解流程如下：

| 步骤 | 输入/操作 | 输出 | 作用 |
| --- | --- | --- | --- |
| 1 | 给定方阵 $A$ | 矩阵本体 | 确定分析对象 |
| 2 | 写出 $A-\lambda I$ | 含参数 $\lambda$ 的矩阵 | 为求根做准备 |
| 3 | 计算 $\det(A-\lambda I)=0$ | 特征多项式 | 找特征值 |
| 4 | 对每个 $\lambda$ 解 $(A-\lambda I)v=0$ | 特征向量/特征空间 | 找不变方向 |
| 5 | 验证 $Av=\lambda v$ | 正确性检查 | 排除算错或数值误差 |

对新手，最重要的边界有三条。

第一，只讨论方阵。非方阵一般没有完整的“特征值-特征向量”体系，因为 $Av$ 和 $v$ 甚至不在同一个空间里，无法直接比较是否仍在同一方向上。

第二，非对称矩阵可能出现复数特征值。复数可以理解成“超出实数轴的新数系，包含虚数单位 $i$”。这意味着即使矩阵元素全是实数，特征值也不一定全是实数。

第三，非对称矩阵的特征向量通常不正交。正交就是“彼此垂直”。所以很多教材里那种“取一组正交特征向量”的做法，只能直接用于对称矩阵，不能无条件推广。

看一个最小求解过程。设

$$
A=\begin{bmatrix}4&1\\2&3\end{bmatrix}
$$

先写

$$
A-\lambda I=
\begin{bmatrix}
4-\lambda & 1\\
2 & 3-\lambda
\end{bmatrix}
$$

然后解

$$
\det(A-\lambda I)=(4-\lambda)(3-\lambda)-2=0
$$

得到

$$
\lambda^2-7\lambda+10=0
$$

所以特征值是 $\lambda=5,2$。再分别代回 $(A-\lambda I)v=0$ 求对应方向。

这套流程是“定义层面”的标准答案，但它不意味着所有矩阵都能优雅地分解。真正进入工程时，还要考虑重根、数值误差、矩阵是否对称、是否真的可对角化。

---

## 核心机制与推导

从定义出发：

$$
Av=\lambda v
$$

移项得到：

$$
(A-\lambda I)v=0
$$

这是一个齐次线性方程组。齐次的意思是“右边全是 0”。它要有非零解，系数矩阵 $A-\lambda I$ 必须不可逆，因此必须满足：

$$
\det(A-\lambda I)=0
$$

这就是特征方程。把行列式展开后得到一个关于 $\lambda$ 的 $n$ 次多项式，称为特征多项式。它的根就是特征值。

这里要区分两个非常容易混淆的概念：

| 概念 | 数学定义 | 白话解释 |
| --- | --- | --- |
| 代数重数 | 特征值作为特征多项式根的重复次数 | 这个根在公式里出现几次 |
| 几何重数 | 特征空间 $\ker(A-\lambda I)$ 的维度 | 真能找到多少个独立方向 |

并且始终有：

$$
1 \le \text{几何重数} \le \text{代数重数}
$$

玩具例子用最经典的对称矩阵：

$$
A=\begin{bmatrix}2&1\\1&2\end{bmatrix}
$$

先求特征值：

$$
\det(A-\lambda I)=
\begin{vmatrix}
2-\lambda & 1\\
1 & 2-\lambda
\end{vmatrix}
=(2-\lambda)^2-1
=\lambda^2-4\lambda+3
=(\lambda-3)(\lambda-1)
$$

所以特征值为 $\lambda_1=3,\lambda_2=1$，每个特征值的代数重数都是 1。

再求特征向量。

当 $\lambda=3$ 时：

$$
A-3I=
\begin{bmatrix}
-1&1\\
1&-1
\end{bmatrix}
$$

方程组等价于 $x=y$，所以特征空间由 $[1,1]^T$ 张成。单位化后可写成：

$$
q_1=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}
$$

当 $\lambda=1$ 时：

$$
A-I=
\begin{bmatrix}
1&1\\
1&1
\end{bmatrix}
$$

方程组等价于 $x=-y$，所以特征空间由 $[1,-1]^T$ 张成。单位化后：

$$
q_2=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}
$$

于是

$$
Q=\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{bmatrix},
\quad
\Lambda=\begin{bmatrix}
3&0\\
0&1
\end{bmatrix}
$$

并且满足

$$
A=Q\Lambda Q^T
$$

这就是谱定理在 2 维情形下的具体样子。谱定理可以理解成“对称矩阵一定能被一组正交方向完全拆开”。

为什么对称矩阵这么特殊？因为如果 $A=A^T$，那么属于不同特征值的特征向量一定正交。设

$$
Au=\lambda u,\quad Av=\mu v,\quad \lambda\neq\mu
$$

则有

$$
u^TAv=u^T(\mu v)=\mu u^Tv
$$

另一方面由于 $A=A^T$，

$$
u^TAv=(A^Tu)^Tv=(Au)^Tv=(\lambda u)^Tv=\lambda u^Tv
$$

所以

$$
\lambda u^Tv=\mu u^Tv
\Rightarrow (\lambda-\mu)u^Tv=0
\Rightarrow u^Tv=0
$$

即 $u,v$ 正交。

接着看可对角化条件。矩阵 $A$ 可对角化，当且仅当它有 $n$ 个线性无关特征向量。等价地说，对每个特征值，都有：

$$
\text{代数重数}=\text{几何重数}
$$

如果某个特征值在多项式里重复了 2 次，但只能找到 1 个独立方向，那就不够组成完整基，矩阵就不能对角化。

典型反例是：

$$
A=\begin{bmatrix}1&1\\0&1\end{bmatrix}
$$

它的特征多项式是：

$$
\det(A-\lambda I)=
\begin{vmatrix}
1-\lambda & 1\\
0 & 1-\lambda
\end{vmatrix}
=(1-\lambda)^2
$$

所以特征值只有 $\lambda=1$，代数重数为 2。但解 $(A-I)v=0$ 得到

$$
\begin{bmatrix}
0&1\\
0&0
\end{bmatrix}
\begin{bmatrix}
x\\y
\end{bmatrix}=0
\Rightarrow y=0
$$

只有形如 $[x,0]^T$ 的向量，因此几何重数是 1，不等于代数重数 2。这个矩阵不可对角化。

---

## 代码实现

工程里很少手算高维矩阵的特征分解，通常直接调用数值线性代数库。数值的意思是“在计算机浮点数上近似计算，而不是符号推导”。Python 里最常见的是 `numpy.linalg.eig` 和 `numpy.linalg.eigh`。

规则很简单：

| 场景 | 推荐函数 | 原因 |
| --- | --- | --- |
| 一般方阵 | `np.linalg.eig` | 适用于一般实矩阵或复矩阵 |
| 实对称矩阵 | `np.linalg.eigh` | 专门针对 Hermitian/对称矩阵，更稳定，结果更适合谱分解 |

下面给一个可运行的例子，先验证玩具例子，再给一个真实工程里的 PCA 最小实现。

```python
import numpy as np

# 玩具例子：对称矩阵
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])

# eigh 专门用于实对称矩阵
eigvals, eigvecs = np.linalg.eigh(A)

# eigh 默认升序，翻转成降序，便于和“最大特征值”直觉一致
idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# 验证 Av ≈ λv，并检查正交性
for i in range(len(eigvals)):
    v = eigvecs[:, i]
    lam = eigvals[i]
    assert np.allclose(A @ v, lam * v, atol=1e-10)

assert np.allclose(eigvals, np.array([3.0, 1.0]))
assert np.allclose(eigvecs.T @ eigvecs, np.eye(2), atol=1e-10)

# 真实工程例子：PCA 的最小实现
# 3 个样本，2 个特征
X = np.array([
    [2.0, 0.0],
    [0.0, 2.0],
    [3.0, 1.0],
])

# 去中心化：减去每列均值
X_centered = X - X.mean(axis=0, keepdims=True)

# 协方差矩阵 C = X^T X / (n-1)
n = X_centered.shape[0]
C = X_centered.T @ X_centered / (n - 1)

# 协方差矩阵是对称矩阵，仍用 eigh
pca_vals, pca_vecs = np.linalg.eigh(C)
idx = pca_vals.argsort()[::-1]
pca_vals = pca_vals[idx]
pca_vecs = pca_vecs[:, idx]

# 第一主成分方向 = 最大特征值对应特征向量
first_pc = pca_vecs[:, 0]

# 验证协方差矩阵的特征关系
assert np.allclose(C @ first_pc, pca_vals[0] * first_pc, atol=1e-10)

# 主成分方向应为单位向量
assert np.allclose(np.linalg.norm(first_pc), 1.0, atol=1e-10)

print("A 的特征值:", eigvals)
print("PCA 第一主成分方向:", first_pc)
```

这段代码体现了两个工程事实。

第一，验证比调用函数更重要。不要只相信库函数返回了结果，要至少检查一次 $Av\approx \lambda v$。符号里的 $\approx$ 表示“在数值误差允许范围内近似相等”。

第二，PCA 和特征分解的关系非常直接。设样本矩阵去中心化后为 $X$，则协方差矩阵通常写成：

$$
C=\frac{1}{n-1}X^TX
$$

然后对 $C$ 做特征分解。最大的特征值对应“方差最大的方向”，这就是第一主成分。

真实工程例子可以更具体一些。假设你有一个用户行为数据表，列是“停留时长、点击次数、滚动深度、购买次数”。这些特征可能彼此相关，比如停留时长和滚动深度经常一起变大。PCA 通过协方差矩阵的特征分解，找出“最能解释变化”的几个方向，把原始高维特征压到更低维，用于可视化、聚类或下游模型预处理。

---

## 工程权衡与常见坑

先给一个对比表，很多错误都来自把这些概念混为一谈。

| 对比项 | 对称矩阵 | 非对称矩阵 |
| --- | --- | --- |
| 特征值 | 一定是实数 | 可能是复数 |
| 特征向量是否可取正交基 | 可以 | 通常不能保证 |
| 是否一定可对角化 | 实对称矩阵一定可以正交对角化 | 不一定 |
| 推荐数值方法 | `eigh` | `eig` / Schur 分解 |

| 对比项 | 代数重数 | 几何重数 |
| --- | --- | --- |
| 含义 | 根在特征多项式里重复次数 | 特征空间维度 |
| 来源 | 多项式因式分解 | 解线性方程组 |
| 关系 | 至少不小于几何重数 | 至多不超过代数重数 |

第一个坑：把“有重根”误认为“肯定不可对角化”。这不对。真正的判据不是有没有重根，而是每个特征值是否满足“代数重数 = 几何重数”。例如单位矩阵 $I$ 的唯一特征值是 1，代数重数很高，但几何重数也同样高，所以它完全可对角化。

第二个坑：把“可对角化”误认为“能正交对角化”。这也不对。正交对角化要求更强，一般要矩阵是实对称的。非对称矩阵即使可对角化，分解形式也通常是

$$
A=V\Lambda V^{-1}
$$

而不是 $Q\Lambda Q^T$。

第三个坑：直接用数值结果判断“是否完全相等”。浮点数不是实数本体，而是有限精度近似。因此工程里要用 `allclose` 一类的容差判断，而不是直接判断 `A @ v == lam * v`。

第四个坑：重根附近非常不稳定。如果两个特征值很接近，计算出的特征向量方向可能对微小扰动非常敏感。这不是库坏了，而是问题本身病态。病态可以理解成“输入轻微变化，输出大幅波动”。

第五个坑：忘记归一化。归一化就是“把向量长度调整为 1”。特征向量本身只定义到比例，即如果 $v$ 是特征向量，那么 $cv$ 也是，所以比较或展示时通常要单位化。

对新手最值得记住的失败例子还是 Jordan 块：

$$
A=\begin{bmatrix}1&1\\0&1\end{bmatrix}
$$

它告诉你一件事：只有特征值，不代表就有足够多的特征向量。工程上如果你强行假设所有矩阵都能像对称矩阵那样分解，算法会在这里直接失效。

---

## 替代方案与适用边界

特征分解不是唯一工具，尤其在非对称、不可对角化或只关心低维表示时，替代方案更合适。

| 方法 | 输入类型 | 结果结构 | 适用场景 |
| --- | --- | --- | --- |
| 特征分解 | 方阵 | $A=V\Lambda V^{-1}$ | 研究不变方向、稳定性、PCA 协方差矩阵 |
| 正交特征分解 | 实对称矩阵 | $A=Q\Lambda Q^T$ | 最稳、最常用的对称情形 |
| Schur 分解 | 任意复方阵，实矩阵也可扩展 | $A=QTQ^*$ | 非对称矩阵的稳定数值处理 |
| SVD | 任意矩阵 | $A=U\Sigma V^T$ | 降维、压缩、最小二乘、推荐系统 |

这里的 $Q^*$ 表示共轭转置，可以理解成“复数版本的转置”。

如果矩阵不可对角化，理论上可以谈 Jordan 分解，但工程里很少直接使用。原因是 Jordan 分解对数值误差极不稳定，轻微扰动就可能让 Jordan 结构发生变化。因此实际计算通常更偏向 Schur 分解。Schur 分解不一定把矩阵化成对角，但可以稳定地化成上三角：

$$
A=QTQ^*
$$

对角线上的元素就是特征值。

如果你的目标不是研究“方向保持不变”，而是做降维、压缩、噪声过滤，SVD 往往更通用。SVD 可以理解成“把任意矩阵拆成左正交方向、缩放强度、右正交方向”：

$$
A=U\Sigma V^T
$$

新手常见疑问是：PCA 不是用特征分解吗，为什么又说可以用 SVD？原因是二者本质相关。设去中心化数据矩阵为 $X$，则

$$
X=U\Sigma V^T
$$

那么协方差矩阵

$$
C=\frac{1}{n-1}X^TX
\;=\;
V\left(\frac{\Sigma^T\Sigma}{n-1}\right)V^T
$$

这说明 $V$ 的列向量正是协方差矩阵的特征向量，奇异值平方再除以 $n-1$ 就对应特征值。所以很多 PCA 实现直接做 SVD，而不是显式构造协方差矩阵。

对零基础到初级工程师，一个实用判断规则如下：

| 任务 | 首选方法 |
| --- | --- |
| 对称矩阵分析、协方差矩阵、PCA | 特征分解 `eigh` |
| 一般方阵的稳定数值分解 | Schur |
| 任意矩阵降维、压缩、推荐系统 | SVD |
| 理论上研究不可对角化结构 | Jordan 形式 |

结论不是“特征分解最好”，而是“它解决的是特定类型问题”。你要先确认输入是否方阵、是否对称、目标是解释结构还是做数值计算，然后再选方法。

---

## 参考资料

| 来源名称 | 主题 | 重点 |
| --- | --- | --- |
| djps.github.io | 特征值与特征向量基础定义 | 明确 $Av=\lambda v$ 与 $\det(A-\lambda I)=0$ 的关系 |
| GeeksforGeeks | 代数重数与几何重数 | 适合理解“根的重复次数”和“特征空间维度”的区别 |
| MLWiki | Spectral Theorem | 说明实对称矩阵可正交对角化，且不同特征值特征向量正交 |
| LibreTexts | Diagonalization | 说明可对角化的判据是有足够多线性无关特征向量 |
| DataScienceBase | PCA 数学基础 | 说明 PCA 本质上依赖协方差矩阵的特征分解 |

- djps.github.io: https://djps.github.io/docs/gradcalclinalg24/part1/eigen/
- GeeksforGeeks: https://www.geeksforgeeks.org/algebraic-and-geometric-multiplicity/
- MLWiki Spectral Theorem: https://mlwiki.org/index.php/Spectral_Theorem
- LibreTexts Diagonalization: https://math.libretexts.org/Bookshelves/Linear_Algebra/Interactive_Linear_Algebra_%28Margalit_and_Rabinoff%29/05%253A_Eigenvalues_and_Eigenvectors/5.03%253A_Diagonalization
- DataScienceBase PCA: https://www.datasciencebase.com/intermediate/linear-algebra/principal-component-analysis/
