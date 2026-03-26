## 核心结论

矩阵分解的核心作用，是把一个难直接处理的矩阵 $A$ 改写成几个“结构清晰的矩阵”相乘。结构清晰的意思是：有的矩阵只负责旋转坐标，有的只负责缩放，有的是三角形结构，便于回代求解。常见三类分解分别解决三类典型问题：

| 分解 | 公式 | 直观含义 | 典型任务 |
|---|---|---|---|
| SVD | $A = U\Sigma V^\top$ | 先旋转，再按主方向缩放，再旋转回来 | 降维、PCA、伪逆、低秩压缩、协同过滤 |
| QR | $A = QR$ | 把原矩阵拆成正交基与上三角系数 | 最小二乘、正交投影、线性方程组 |
| LU | $PA = LU$ | 把消元过程显式保存成下三角和上三角 | 解线性系统、行列式、重复求解 |

对初学者最重要的结论有三条。

第一，SVD 是“最通用的低秩工具”。低秩，意思是矩阵的主要信息可以由少量方向表示。它既能做 PCA，也能做推荐系统里的潜在因子建模。若只保留前 $k$ 个奇异值，就得到最优秩-$k$ 近似：
$$
A_k=\sum_{i=1}^k \sigma_i u_i v_i^\top
$$

第二，QR 是“最小二乘的默认安全解法”。最小二乘，意思是方程无精确解时，找误差平方和最小的近似解。相比直接解正规方程 $(A^\top A)x=A^\top b$，QR 数值更稳定。

第三，LU 是“高效重复求解线性系统的工程工具”。当同一个矩阵 $A$ 对应多个右端项 $b$ 时，先做一次 $PA=LU$，后续只做前代与回代，通常比每次重新求逆更快也更稳。

玩具例子：对角矩阵
$$
A=\begin{bmatrix}3&0\\0&2\end{bmatrix}
$$
它的 SVD 就是 $U=V=I,\ \Sigma=\mathrm{diag}(3,2)$。这说明两个坐标方向本身就是主方向，沿第一个方向的信息强度是 3，第二个方向是 2。若只保留第一个奇异值，就把二维信息压到一维。

---

## 问题定义与边界

矩阵分解要解决的问题，不是“把公式写复杂”，而是把原任务变成更容易做的子任务。给定一个矩阵 $A\in\mathbb{R}^{m\times n}$，常见目标包括：

| 任务 | 想解决什么 | 更适合的分解 |
|---|---|---|
| 降维 | 用更少维度近似原数据 | SVD |
| 伪逆 | 非方阵或不可逆时求广义逆 | SVD |
| 最小二乘 | 超定方程 $m \ge n$ 的近似解 | QR |
| 线性系统 | 快速解 $Ax=b$ | LU |
| 行列式 | 快速计算 $\det(A)$ | LU |

边界也很明确：

| 分解 | 要求 | 适用任务 | 常见失败点 |
|---|---|---|---|
| SVD | 任意矩阵 | PCA、伪逆、低秩压缩 | 截断过度会丢主信息 |
| QR | 通常 $m\ge n$ | 最小二乘、投影 | 用正规方程替代会放大误差 |
| LU | 方阵，通常要求可逆；实践中常用带置换 | 解线性系统、行列式 | 不做 pivot 会遇到零对角 |

这里的 pivot 是“主元交换”，白话讲就是在消元前交换行，避免拿 0 或极小数做除数。

一个简单边界例子：

- 若 $A$ 是 $3\times 2$ 满列秩矩阵，SVD 与 QR 都适合，SVD 可做截断，QR 可做最小二乘。
- 若
  $$
  A=\begin{bmatrix}1&2\\2&4\end{bmatrix}
  $$
  第二行是第一行的 2 倍，这是奇异矩阵。直接 LU 消元时第二个对角元会变成 0，必须考虑置换或承认它不可逆。

因此，矩阵分解不是“哪个更高级”，而是“哪个更匹配问题形状”。

---

## 核心机制与推导

### 1. SVD：把信息拆成若干主方向

SVD 写成
$$
A=U\Sigma V^\top
$$
其中：

- $U$ 的列向量是左奇异向量，可理解为输出空间的主方向
- $V$ 的列向量是右奇异向量，可理解为输入空间的主方向
- $\Sigma$ 的对角线是奇异值，表示每个方向的重要程度

若奇异值按 $\sigma_1\ge\sigma_2\ge\cdots$ 排序，则前 $k$ 项构成最优低秩近似：
$$
A_k=U_k\Sigma_kV_k^\top=\sum_{i=1}^k \sigma_i u_i v_i^\top
$$
“最优”指在 Frobenius 范数意义下误差最小。白话说，同样只保留 $k$ 个方向，SVD 的压缩损失最小。

伪逆也直接来自 SVD：
$$
A^+=V_k\Sigma_k^{-1}U_k^\top
$$
它用于非方阵或秩亏矩阵的最小范数解。

### 2. QR：把列空间变成正交基

QR 分解写成
$$
A=QR
$$
其中 $Q$ 的列两两正交，$R$ 是上三角矩阵。正交，意思是向量互相垂直，计算投影时不会相互污染误差。

最小二乘问题
$$
\min_x \|Ax-b\|_2^2
$$
代入 $A=QR$ 得
$$
\min_x \|QRx-b\|_2^2
$$
左乘 $Q^\top$ 后，因 $Q^\top Q=I$，可转为
$$
Rx=Q^\top b
$$
于是问题退化成上三角回代。

玩具例子：令
$$
A=\begin{bmatrix}
1&1\\
1&-1\\
1&0
\end{bmatrix},\quad
b=\begin{bmatrix}2\\0\\1\end{bmatrix}
$$
这类方程通常无精确解，但可用 QR 找到误差最小的 $x$。这比先算 $A^\top A$ 更稳，因为后者会放大条件数。

### 3. LU：把消元步骤保存下来

LU 分解本质上是高斯消元的矩阵形式。更常见的工程写法是
$$
PA=LU
$$
其中 $P$ 是置换矩阵，$L$ 是下三角矩阵，$U$ 是上三角矩阵。

求解 $Ax=b$ 时，先写成
$$
LUx=Pb
$$
再分两步：

1. 先解 $Ly=Pb$
2. 再解 $Ux=y$

若 $A$ 可逆，则行列式可由
$$
\det(A)=\det(P)^{-1}\prod_i U_{ii}
$$
得到。也就是说，考虑置换符号后，$U$ 对角线元素的乘积就给出行列式。

---

## 代码实现

下面给出一个最小可运行示例，覆盖 SVD 低秩近似、QR 最小二乘和带缺失值的推荐系统玩具流程。这里的“潜在因子”是压缩后隐藏的低维表示，可理解为用户兴趣和物品属性的隐式坐标。

```python
import numpy as np

def truncated_svd(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]
    return Uk @ np.diag(sk) @ Vtk, Uk, sk, Vtk

def qr_least_squares(A, b):
    Q, R = np.linalg.qr(A, mode="reduced")
    x = np.linalg.solve(R, Q.T @ b)
    return x

# 1) SVD 玩具例子
A = np.array([[3.0, 0.0], [0.0, 2.0]])
A1, Uk, sk, Vtk = truncated_svd(A, k=1)
assert A1.shape == A.shape
assert np.allclose(A1, np.array([[3.0, 0.0], [0.0, 0.0]]))

# 2) QR 最小二乘
B = np.array([[1.0, 1.0], [1.0, -1.0], [1.0, 0.0]])
b = np.array([2.0, 0.0, 1.0])
x = qr_least_squares(B, b)
residual = np.linalg.norm(B @ x - b)
assert residual < 1.0

# 3) 推荐系统玩具例子：先用 0 填充缺失值，再做低秩重建
R_observed = np.array([
    [5.0, np.nan, 3.0],
    [4.0, 5.0, np.nan],
    [np.nan, 2.0, 1.0]
])
mask = ~np.isnan(R_observed)
R_init = np.nan_to_num(R_observed, nan=0.0)
R_hat, Uk, sk, Vtk = truncated_svd(R_init, k=2)
R_filled = np.where(mask, R_observed, R_hat)

assert R_filled.shape == R_observed.shape
assert not np.isnan(R_filled).any()
```

要点有三个：

- `np.linalg.svd(..., full_matrices=False)` 返回紧凑形式，通常更适合工程实现。
- 低秩近似真正可复用的是 `Uk` 和 `Vtk`。在推荐系统里，它们对应用户向量和物品向量的低维表示。
- `np.nan_to_num` 这里只是演示流程，真实工程会把“补全缺失值”改成显式优化目标，例如只在已观测位置上计算损失，并加入正则化。

真实工程例子：推荐系统里常把评分矩阵 $R$ 近似为
$$
R\approx PQ^\top
$$
其中 $P$ 是用户潜在因子矩阵，$Q$ 是物品潜在因子矩阵。SVD 提供了低秩结构的基础，而工业系统常进一步加入偏置项、正则化、时间漂移和曝光建模。

---

## 工程权衡与常见坑

矩阵分解一旦进入工程环境，问题不再只是“能不能分解”，而是“误差、速度、稳定性是否可接受”。

| 场景 | 收益 | 主要代价 | 关键风险 |
|---|---|---|---|
| 推荐系统召回 | 用低维向量快速召回相似物品 | 需要训练与定期更新 | 稀疏数据下容易过拟合 |
| PCA/SVD 压缩传感器数据 | 降带宽、降延迟 | 需要选择截断维数 | 丢失关键主成分会降精度 |
| 神经网络权重压缩 | 降模型参数量与推理成本 | 可能需要压缩前预训练 | 截断后精度回退 |

常见坑主要有五个。

第一，盲目按“压缩率”选 $k$。正确做法通常是看累计能量比：
$$
\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}
$$
若这个比例不足，说明压缩过猛。

第二，把缺失值当作真实 0。推荐系统的评分矩阵极稀疏，未观测不等于低评分。把 NaN 直接补 0 再 SVD，只适合作为教学玩具例子。

第三，最小二乘直接走正规方程。它形式简单，但数值稳定性更差。工程里优先 QR，条件特别差时进一步考虑 SVD。

第四，LU 不做 pivot。只要对角元可能接近 0，就应使用 $PA=LU$。否则不仅会报错，还会因舍入误差导致结果失真。

第五，忽略数据分布变化。真实工程里，奇异值结构会随时间漂移。推荐系统的用户兴趣变化、工业传感器工况变化，都会让旧分解快速失效。

一个真实工程例子是 Netflix/Movielens 协同过滤：评分矩阵非常稀疏，但用户与电影之间仍存在较稳定的低秩结构，因此矩阵分解能把高维稀疏交互压成少量潜在因子。另一个例子是制造业虚拟传感器：先用 PCA/SVD 压缩有限元高维数据，再把主成分送入小型神经网络，在延迟与精度之间取平衡。

---

## 替代方案与适用边界

矩阵分解不是唯一选择。是否继续使用 SVD，取决于数据是否稠密、是否需要可解释性、是否必须保留稀疏结构。

| 方法 | 保留稀疏性 | 解释性 | 压缩效果 | 适用场景 | 主要缺点 |
|---|---|---|---|---|---|
| SVD 截断 | 弱 | 中 | 强 | 通用低秩压缩、PCA、伪逆 | 基向量通常不直接可解释 |
| CUR/CX | 强 | 强 | 中 | 希望保留原始列/行语义 | 误差通常不如最优低秩 |
| NMF | 中 | 强 | 中 | 非负数据，如词频、计数 | 结果依赖初始化 |
| Sparse PCA | 强 | 强 | 中 | 传感器选择、特征筛选 | 优化更复杂 |
| 低秩预热 + SVD | 弱 | 低 | 强 | 神经网络权重压缩 | 需要额外训练成本 |

选择建议可以概括为：

- 若目标是最优低秩近似，优先 SVD。
- 若要保留原始列或原始行的业务含义，考虑 CUR/CX。
- 若数据天然非负且希望因子有直观解释，考虑 NMF。
- 若需要“少量特征直接落地部署”，例如只保留少数传感器，考虑 Sparse PCA。
- 若是大模型压缩，单纯后处理式 SVD 往往不够，需要低秩预热、Anchored SVD 或 LC-SVD 这类方法，在压缩前或压缩时约束模型保持行为稳定。

因此，矩阵分解的适用边界并不是“数学上能不能做”，而是“分解后的结构是否还能服务你的工程目标”。

---

## 参考资料

- Wikipedia: Matrix decomposition  
  https://en.wikipedia.org/wiki/Matrix_decomposition
- Wikipedia: QR decomposition  
  https://en.wikipedia.org/wiki/QR_decomposition
- Math LibreTexts: Using Singular Value Decompositions  
  https://math.libretexts.org/Bookshelves/Linear_Algebra/Understanding_Linear_Algebra_%28Austin%29/07%3A_The_Spectral_Theorem_and_singular_value_decompositions/7.05%3A_Using_Singular_Value_Decompositions
- Built In: What Is Matrix Factorization?  
  https://builtin.com/articles/matrix-factorization
- Sensors 2024: Implementation of PCA/SVD and Neural Networks in virtual sensing  
  https://www.mdpi.com/1424-8220/24/24/8065
- Mathematics 2023: CUR/CX 与相关降维替代方法  
  https://www.mdpi.com/2227-7390/11/12/2674
- Emergent Mind: Low-Rank Prehab  
  https://www.emergentmind.com/articles/2512.01980
- OpenReview: Anchored SVD / LC-SVD 相关低秩压缩讨论  
  https://openreview.net/forum?id=fIpDd5UlFP
