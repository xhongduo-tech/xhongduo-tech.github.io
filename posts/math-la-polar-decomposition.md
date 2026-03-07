## 核心结论

极分解把一个矩阵拆成“方向变化”和“尺度变化”两部分。更准确地说，对任意非奇异实矩阵 $A$，都存在唯一分解

$$
A = UP = VU
$$

其中：

- $U$ 是正交矩阵。它满足 $U^TU=I$，因此不会改变向量长度，只会改变方向。二维里它对应旋转或反射，三维里对应旋转加可能的镜像翻转。
- $P=(A^TA)^{1/2}$ 是对称正定矩阵。它描述沿一组两两正交方向的拉伸或压缩。
- $V=(AA^T)^{1/2}$ 也是对称正定矩阵，对应左极分解中的尺度因子。

它的价值不在于“再做一次矩阵分解”，而在于把几何含义直接拆开：$U$ 负责姿态，$P$ 或 $V$ 负责形变尺度。这也是它在机器人姿态估计、连续介质形变分析、最近正交矩阵问题中常见的原因。

先看一个玩具例子。设

$$
A=
\begin{bmatrix}
0 & 2\\
1 & 0
\end{bmatrix}
$$

先算

$$
A^TA=
\begin{bmatrix}
1 & 0\\
0 & 4
\end{bmatrix}
\Rightarrow
P=(A^TA)^{1/2}=
\begin{bmatrix}
1 & 0\\
0 & 2
\end{bmatrix}
$$

这里平方根是“矩阵平方根”，意思是找一个矩阵 $P$，使得 $P^2=A^TA$。因为 $A^TA$ 已经是对角矩阵，所以平方根只需要对对角元分别开方。

再算

$$
U=AP^{-1}
=
\begin{bmatrix}
0 & 2\\
1 & 0
\end{bmatrix}
\begin{bmatrix}
1 & 0\\
0 & \frac12
\end{bmatrix}
=
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
$$

于是

$$
UP=
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
\begin{bmatrix}
1 & 0\\
0 & 2
\end{bmatrix}
=
\begin{bmatrix}
0 & 2\\
1 & 0
\end{bmatrix}
=A
$$

并且

$$
U^TU=
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
=
I
$$

这一步验证了 $U$ 确实是正交矩阵。

同理，左极分解中

$$
AA^T=
\begin{bmatrix}
4 & 0\\
0 & 1
\end{bmatrix},
\quad
V=(AA^T)^{1/2}
=
\begin{bmatrix}
2 & 0\\
0 & 1
\end{bmatrix},
\quad A=VU
$$

这个例子已经把核心事实说清楚了：同一个线性变换，可以拆成“先调整方向，再做轴向拉伸”，或者“先做拉伸，再调整方向”。

把它写成对向量的作用更直观。对任意 $x$，

$$
Ax = U(Px)
$$

意思是：

1. 先由 $P$ 把 $x$ 沿某些正交方向拉伸或压缩。
2. 再由 $U$ 把结果整体旋转或反射。

所以极分解本质上是在回答一个几何问题：一个线性变换里，哪些部分是在改长度，哪些部分是在改方向。

---

## 问题定义与边界

极分解解决的问题是：给定一个矩阵 $A$，能否把它表示成“纯方向因子”和“纯尺度因子”的乘积。

对实矩阵，最常见的右极分解写成

$$
A=UP,\quad P=(A^TA)^{1/2}
$$

这里的边界条件很重要：

| 条件 | 结论 | 风险 |
|---|---|---|
| $A$ 非奇异 | $U$ 唯一，$P$ 唯一 | 可稳定解释为姿态 + 尺度 |
| $A$ 奇异但非零 | $P$ 仍唯一，$U$ 一般不唯一 | 不同算法可能给出不同 $U$ |
| $A=0$ | $P=0$ | 任意正交矩阵都可配合，完全失去唯一性 |

“非奇异”的意思是矩阵可逆，不会把某个非零向量压成零。只有在这个条件下，$P$ 可逆，$U=AP^{-1}$ 才能直接写出并保持唯一。

这里还要区分两个层次的“存在”：

| 问题 | 结论 |
|---|---|
| 是否存在某个极分解 | 总体上存在 |
| 是否存在唯一极分解 | 只在非奇异时完整成立 |
| 哪个因子一定唯一 | 对称半正定因子 $P$ 一定唯一 |
| 哪个因子可能不唯一 | 正交因子 $U$ 在奇异情形下可能不唯一 |

退化矩阵最能暴露边界。设

$$
A=
\begin{bmatrix}
1 & 0\\
0 & 0
\end{bmatrix}
$$

则

$$
A^TA=
\begin{bmatrix}
1 & 0\\
0 & 0
\end{bmatrix},
\quad
P=(A^TA)^{1/2}=
\begin{bmatrix}
1 & 0\\
0 & 0
\end{bmatrix}
$$

这里 $P$ 仍然唯一，因为 $A^TA$ 的平方根在“取对称半正定平方根”这个意义下是唯一的。但 $P$ 不可逆，所以不能直接用 $U=AP^{-1}$。这时只要求 $A=UP$，会发现很多不同的正交矩阵都能成立，例如

$$
U_1=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix},
\quad
U_2=
\begin{bmatrix}
1 & 0\\
0 & -1
\end{bmatrix}
$$

都有

$$
U_1P=U_2P=A
$$

原因很直接：第二列被 $P$ 压成了零，所以 $U$ 在那个零空间上的动作不会反映到最终结果里。

把这个现象写成子空间语言会更准确。若 $x \in \ker(P)$，那么 $Px=0$，因此

$$
UPx = U0 = 0
$$

也就是说，只要 $U$ 在 $\ker(P)$ 上怎么变都不影响最终结果，极分解中的正交因子就会出现自由度。

因此，工程上必须先区分两个问题：

1. 你是在处理满秩矩阵，还是退化矩阵。
2. 你要的是唯一的姿态解释，还是只要一个可用分解。

如果要稳定恢复姿态，秩检测和正则化通常不是可选项，而是前置步骤。一个常见判据是查看最小奇异值 $\sigma_{\min}(A)$ 是否过小，或者条件数

$$
\kappa(A)=\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
$$

是否过大。条件数很大时，说明矩阵接近奇异，极分解中的方向因子会对噪声更敏感。

---

## 核心机制与推导

极分解的推导可以从一个最短公式链开始。

第一步，构造

$$
P=(A^TA)^{1/2}
$$

因为 $A^TA$ 总是对称半正定矩阵，所以它存在唯一的对称半正定平方根。这里“半正定”的意思是对任意向量 $x$ 都有

$$
x^TA^TAx = \|Ax\|^2 \ge 0
$$

所以它不会把“平方长度”算成负数。

当 $A$ 非奇异时，$A^TA$ 正定，因此 $P$ 正定且可逆。

第二步，定义

$$
U=AP^{-1}
$$

检查它是否真是正交矩阵：

$$
U^TU
=
(AP^{-1})^T(AP^{-1})
=
P^{-1}A^TAP^{-1}
=
P^{-1}P^2P^{-1}
=
I
$$

于是 $U$ 正交，得到

$$
A=UP
$$

这就是右极分解。

左极分解完全类似。定义

$$
V=(AA^T)^{1/2}
$$

则可写成

$$
A=VU
$$

几何上可理解为两种看法：

- $A=UP$：先沿正交主轴拉伸，再做整体方向变换。
- $A=VU$：先做方向变换，再在输出空间中拉伸。

这两种写法并不矛盾，只是“把尺度放在左边还是右边”的观察角度不同。

如果引入 SVD，也就是奇异值分解，就能更直接看清各因子的来源。设

$$
A=W\Sigma V^T
$$

其中：

- $W,V$ 是正交矩阵，对应输入空间和输出空间的正交基变换。
- $\Sigma$ 是对角矩阵，对角线上的奇异值是各方向上的拉伸倍数。

那么

$$
A^TA = V\Sigma^2V^T
$$

所以

$$
P=(A^TA)^{1/2}=V\Sigma V^T
$$

再代回去：

$$
U=AP^{-1}
=
W\Sigma V^T(V\Sigma V^T)^{-1}
=
WV^T
$$

于是右极分解变成

$$
A=(WV^T)(V\Sigma V^T)
$$

同理，左极分解中

$$
V_L=(AA^T)^{1/2}=W\Sigma W^T
$$

于是

$$
A=(W\Sigma W^T)(WV^T)
$$

可以把这个流程记成：

| 步骤 | 数学对象 | 几何意义 |
|---|---|---|
| 1 | $V^T$ | 把输入坐标转到右奇异向量基底 |
| 2 | $\Sigma$ | 沿正交轴拉伸或压缩 |
| 3 | $W$ | 把结果转到输出坐标方向 |
| 合并后 | $WV^T$ | 纯方向变化 |
| 合并后 | $V\Sigma V^T$ | 纯尺度变化 |

如果只关心极分解，真正保留的是两类信息：

- 方向因子：$U=WV^T$
- 尺度因子：$P=V\Sigma V^T$

这也是它和 SVD 的差异。SVD 保留“每个奇异方向是谁、奇异值按什么顺序排列”；极分解只保留“总体方向”和“总体尺度张量”。

还可以从长度变化角度再看一遍。对任意向量 $x$，

$$
\|Ax\|^2 = x^TA^TAx = x^TP^2x
$$

而另一方面，

$$
\|Ux\|^2 = x^TU^TUx = x^Tx = \|x\|^2
$$

这两式合起来说明：

- 长度变化完全由 $P$ 决定；
- $U$ 不改变长度，只改变方向。

这正是“方向因子 + 尺度因子”这个说法的严格数学版本。

再补一个二维直观例子。设

$$
A=
\begin{bmatrix}
2 & 1\\
0 & 1
\end{bmatrix}
$$

它不是纯旋转，也不是纯拉伸，而是两者混合。极分解做的事情不是“把它改成更简单的矩阵”，而是回答：

- 哪个部分可以解释成旋转/反射？
- 哪个部分可以解释成沿主轴方向的拉伸？

这就是极分解在几何建模里比 QR 更自然的原因。

---

## 代码实现

最直接的实现有两种：平方根法和 SVD 法。

平方根法直接按定义走，适合教学和小规模验证。下面给出一个可运行的 Python 版本：

```python
import numpy as np

def polar_decomposition_via_eig(A, eps=1e-12):
    """
    右极分解: A = U @ P
    使用 A^T A 的特征分解构造 P = (A^T A)^(1/2)。
    eps 用于处理非常小的特征值，避免数值上除零。
    """
    A = np.array(A, dtype=float)
    ATA = A.T @ A

    # A^T A 是对称矩阵，适合用 eigh
    eigvals, Q = np.linalg.eigh(ATA)

    # 数值安全：裁掉微小负值和过小特征值
    eigvals_clipped = np.maximum(eigvals, eps)

    sqrt_eigvals = np.sqrt(eigvals_clipped)
    inv_sqrt_eigvals = 1.0 / sqrt_eigvals

    P = Q @ np.diag(sqrt_eigvals) @ Q.T
    P_inv = Q @ np.diag(inv_sqrt_eigvals) @ Q.T
    U = A @ P_inv

    return U, P

def check_polar(A, U, P, atol=1e-8):
    A = np.array(A, dtype=float)
    n = A.shape[1]
    return {
        "reconstruction_ok": np.allclose(U @ P, A, atol=atol),
        "orthogonal_ok": np.allclose(U.T @ U, np.eye(n), atol=atol),
        "symmetric_ok": np.allclose(P, P.T, atol=atol),
        "psd_eigvals": np.linalg.eigvalsh(P),
    }

# 玩具例子
A = np.array([[0., 2.],
              [1., 0.]])
U, P = polar_decomposition_via_eig(A)
result = check_polar(A, U, P)

print("A =\n", A)
print("U =\n", U)
print("P =\n", P)
print("check =", result)

assert result["reconstruction_ok"]
assert result["orthogonal_ok"]
assert result["symmetric_ok"]
assert np.all(result["psd_eigvals"] >= -1e-10)

# 再测一个一般矩阵
B = np.array([[3., 1.],
              [0., 2.]])
U2, P2 = polar_decomposition_via_eig(B)
result2 = check_polar(B, U2, P2)

print("\nB =\n", B)
print("U2 =\n", U2)
print("P2 =\n", P2)
print("check =", result2)

assert result2["reconstruction_ok"]
assert result2["orthogonal_ok"]
assert result2["symmetric_ok"]
assert np.all(result2["psd_eigvals"] >= -1e-10)
```

如果运行第一组样例，输出会接近：

```text
A =
 [[0. 2.]
 [1. 0.]]
U =
 [[0. 1.]
 [1. 0.]]
P =
 [[1. 0.]
 [0. 2.]]
```

这段代码里，`np.linalg.eigh` 用于对称矩阵特征分解。它适合处理 $A^TA$，因为：

- `eigh` 专门针对实对称矩阵；
- 返回的特征值理论上应非负；
- 用它构造矩阵平方根比对一般矩阵直接开方更稳定。

但这个版本有一个前提：你默认接受对极小特征值做截断。因此它更像“教学实现”或“轻量验证实现”，而不是最稳健的生产实现。

如果你已经在用数值库，SVD 版本通常更稳健，因为库实现更成熟：

```python
import numpy as np

def polar_decomposition_via_svd(A):
    """
    右极分解: A = U_polar @ P
    基于 SVD: A = W @ Sigma @ Vt
    """
    A = np.array(A, dtype=float)
    W, s, Vt = np.linalg.svd(A, full_matrices=False)

    U_polar = W @ Vt
    P = Vt.T @ np.diag(s) @ Vt
    return U_polar, P

def nearest_rotation(A):
    """
    如果只允许纯旋转(det=1)，在 det(U) < 0 时修正最后一个奇异方向。
    常见于三维刚体配准。
    """
    A = np.array(A, dtype=float)
    W, s, Vt = np.linalg.svd(A, full_matrices=False)

    D = np.eye(len(s))
    if np.linalg.det(W @ Vt) < 0:
        D[-1, -1] = -1.0

    R = W @ D @ Vt
    return R

A = np.array([[3., 1.],
              [0., 2.]])
U, P = polar_decomposition_via_svd(A)

assert np.allclose(U @ P, A, atol=1e-8)
assert np.allclose(U.T @ U, np.eye(2), atol=1e-8)
assert np.allclose(P, P.T, atol=1e-8)
assert np.all(np.linalg.eigvalsh(P) >= -1e-10)

# 最近旋转矩阵示例
R = nearest_rotation(A)
assert np.allclose(R.T @ R, np.eye(2), atol=1e-8)
assert np.linalg.det(R) > 0
```

这段代码多补了一个 `nearest_rotation`，原因是很多工程任务不要“正交矩阵”，而要“旋转矩阵”。区别在于：

| 对象 | 条件 | 是否允许反射 |
|---|---|---|
| 正交矩阵 | $U^TU=I$ | 允许 |
| 旋转矩阵 | $R^TR=I,\ \det(R)=1$ | 不允许 |
| 含反射的正交矩阵 | $U^TU=I,\ \det(U)=-1$ | 允许 |

如果只写伪代码，Matlab 版通常是：

```matlab
[W, S, V] = svd(A);
U = W * V';
P = V * S * V';
```

如果要处理奇异或近奇异矩阵，实际代码还应补两件事：

1. 不直接调用显式逆 `inv(P)`。
2. 对很小奇异值做阈值裁剪。

例如：

```python
def polar_decomposition_rank_deficient(A, tol=1e-10):
    """
    对奇异/近奇异矩阵更稳健的版本。
    返回一个可用的 U, P，其中 U 在零空间上的选择由 SVD 决定。
    """
    A = np.array(A, dtype=float)
    W, s, Vt = np.linalg.svd(A, full_matrices=False)

    # 对非常小的奇异值做截断
    s_cut = np.where(s > tol, s, 0.0)

    U_polar = W @ Vt
    P = Vt.T @ np.diag(s_cut) @ Vt
    return U_polar, P
```

真实工程例子可以看 IMU 标定。IMU 是惯性测量单元，通常包括陀螺仪和加速度计。制造误差和安装误差会让标定矩阵不是理想旋转，而是“姿态偏差 + 非正交缩放”混在一起。设标定矩阵为 $M$，可写为

$$
M = UP
$$

其中：

- $U$ 对应安装姿态误差或坐标系错位；
- $P$ 对应轴间非正交、尺度因子耦合等形变误差。

这样做的好处是，后续补偿链路可以分开处理：

| 误差类型 | 对应因子 | 处理方式 |
|---|---|---|
| 姿态错位 | $U$ | 作为坐标系旋转修正 |
| 轴不正交 | $P$ 非对角项 | 作为几何耦合补偿 |
| 比例因子偏差 | $P$ 特征值或对角主量 | 作为尺度校准 |

所以极分解不是为了“把矩阵写漂亮”，而是为了让误差模型可解释、可维护、可校准。

---

## 工程权衡与常见坑

极分解不是“比 SVD 更高级”，而是“问题表述更贴近几何需求”。如果你的目标就是把线性变换拆成姿态和尺度，它往往比直接操作 SVD 更顺手。

先看工程流程：

| 步骤 | 目标 | 注意事项 |
|---|---|---|
| 1. 检查秩或条件数 | 判断矩阵是否接近奇异 | 条件数过大时结果会放大噪声 |
| 2. 计算极分解 | 得到 $U,P$ | 优先用稳定库实现 |
| 3. 解释 $U$ | 当作旋转/反射因子 | 若要求纯旋转，要检查 $\det(U)$ |
| 4. 解释 $P$ | 当作尺度/非正交误差 | $P$ 是对称半正定，不一定对角 |
| 5. 回代验证 | 检查 $UP \approx A$ | 同时检查 $U^TU \approx I$ |

常见坑主要有四类。

第一，把极分解误当成 SVD 的简写版本。这是概念错误。SVD 给出的是完整的奇异方向和奇异值排序；极分解给出的是合并后的方向因子与尺度因子。两者有关，但信息量不同。

第二，忽略奇异矩阵的不唯一性。退化情况下，$P$ 仍唯一，但 $U$ 不唯一。如果你的算法把 $U$ 当成“唯一姿态”，结果会在零空间附近漂移。

第三，误把正交矩阵等同于旋转矩阵。正交矩阵满足 $U^TU=I$，但它可能包含反射，表现为 $\det(U)=-1$。如果任务要求刚体旋转，例如三维姿态估计，通常还要额外强制 $\det(U)=1$。

第四，在数值实现中直接求逆。若 $P$ 接近奇异，直接算 `inv(P)` 会放大误差。实际工程里更常见的做法是：

- 用 SVD 直接组装；
- 或在特征值上加小正则，如 $\lambda_i \leftarrow \max(\lambda_i,\varepsilon)$；
- 或用线性方程求解替代显式求逆。

再补两个容易被忽略的坑。

第五，把 $P$ 误解成“按坐标轴缩放”。这通常不对。$P$ 虽然是对称矩阵，但一般不是对角矩阵。它表示的是“沿某组正交主轴缩放”，这组轴可能和原始坐标轴不对齐。

第六，只检查重构误差，不检查结构约束。很多实现只看

$$
\|A-UP\|
$$

是否足够小，却忘了检查：

$$
U^TU \approx I,\quad P \approx P^T,\quad \lambda_i(P)\ge 0
$$

如果这些结构约束没满足，结果就不是合法的极分解。

以 IMU 校准为例，传感器误差矩阵常含有轴不垂直、比例因子不一致、安装偏角等混合误差。极分解的优势在于把这两类误差分离：

- $P$ 吸收非正交和尺度畸变；
- $U$ 吸收整体姿态错位。

这种分离在高动态导航里很有用，因为“非正交”与“姿态误差”对后续补偿链路的影响并不相同，强行混成一个旋转近似，往往会让校准模型的解释力变差。

如果把工程选项压缩成一个决策表，可以写成：

| 场景 | 推荐做法 | 原因 |
|---|---|---|
| 满秩、规模小、教学验证 | 特征分解法 | 公式直接，对照理论方便 |
| 满秩、规模中大、生产实现 | SVD 法 | 稳定性更好 |
| 接近奇异 | SVD + 阈值/正则 | 避免逆放大误差 |
| 必须输出纯旋转 | SVD 后修正 $\det=1$ | 防止反射混入 |
| 只需要最近正交矩阵 | 直接取极分解中的 $U$ | 几何意义最直接 |

最后补一句工程上很常见但文章里容易漏掉的事实：极分解里的正交因子 $U$，恰好是 Frobenius 范数意义下离 $A$ 最近的正交矩阵之一。也就是

$$
U = \arg\min_{Q^TQ=I}\|A-Q\|_F
$$

这也是它在姿态投影、旋转矩阵纠偏、形变恢复里非常常见的原因。

---

## 替代方案与适用边界

如果问题不只是“分离方向和尺度”，那极分解未必是最佳工具。

| 方法 | 输出 | 信息量 | 数值稳定性 | 适合场景 |
|---|---|---|---|---|
| 极分解 | $A=UP$ 或 $A=VU$ | 中 | 高，常借助 SVD 实现 | 只关心姿态与尺度分离 |
| SVD | $A=W\Sigma V^T$ | 高 | 很高 | 需要奇异值、主方向、低秩分析 |
| QR 分解 | $A=QR$ | 中低 | 高 | 解线性方程、构造正交基 |
| Gram-Schmidt | 正交基 + 投影系数 | 低 | 普通版本较差，改进版较好 | 教学、简单正交化 |

选型可以按需求判断：

- 只想求“离某矩阵最近的正交矩阵”：优先极分解，取 $U$ 即可。
- 需要知道每个主方向上的拉伸倍数，并按大小排序：用 SVD。
- 只是想把列向量正交化，服务于数值解法：用 QR 更直接。
- 数据接近奇异、噪声较大，又要求稳定实现：通常直接调用 SVD 库，再从 SVD 组装极分解。

一个容易混淆的点是：QR 里的 $Q$ 也是正交矩阵，为什么不用它代替极分解里的 $U$？因为 QR 的 $R$ 一般是上三角矩阵，不是对称半正定矩阵，它没有“纯尺度张量”的几何解释。也就是说，QR 不是在做“姿态 + 拉伸”的自然分离，而是在做“正交基变换 + 上三角坐标表示”。

把这几个分解的差异再压缩成一句话：

| 方法 | 你真正得到的东西 |
|---|---|
| 极分解 | 一个最接近几何动作分离的“方向 + 尺度”模型 |
| SVD | 一套完整谱结构，告诉你每个主方向怎么变 |
| QR | 一个适合数值计算流程的正交基和三角系数 |
| Gram-Schmidt | 一种构造正交基的方法，不是完整矩阵分解框架 |

所以边界很清楚：

- 极分解适合几何解释优先的场景。
- SVD 适合完整谱信息优先的场景。
- QR 适合数值线性代数计算流程，不适合替代极分解的几何含义。

还有两个常见应用边界值得明确。

第一，若矩阵来自仿射变换的线性部分，极分解只处理线性部分，不处理平移项。也就是说，对齐点云时，通常要先把平移和线性变换分开。

第二，若矩阵来自非线性系统在某点的雅可比矩阵，极分解解释的是该点附近的局部线性行为，不是全局行为。这个边界在机器人学和连续介质里都很重要。

---

## 参考资料

1. Horn, Roger A.; Johnson, Charles R. *Matrix Analysis*. 书中系统讨论了正定矩阵、矩阵平方根、SVD 与极分解之间的关系，适合确认定理条件和证明链条。  
2. Nicholas J. Higham. *Functions of Matrices: Theory and Computation*. 详细讨论矩阵平方根和数值实现，是理解 $P=(A^TA)^{1/2}$ 数值稳定性的核心资料。  
3. Polar decomposition, Wikipedia。适合快速查看定义、存在唯一性、左右极分解写法以及与最近正交矩阵问题的联系。  
4. Polar decomposition, Encyclopedia of Mathematics。表述更紧凑，适合核对“对称半正定平方根唯一”这类理论结论。  
5. Golub, Gene H.; Van Loan, Charles F. *Matrix Computations*. 说明为何工程实现通常通过 SVD 组装极分解，而不是直接做显式逆。  
6. Research on IMU Calibration Model Based on Polar Decomposition, *Micromachines*, 2023。给出极分解在 IMU 标定中的一个典型工程应用，展示如何分离姿态误差与非正交尺度误差。
