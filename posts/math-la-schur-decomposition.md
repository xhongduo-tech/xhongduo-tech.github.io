## 核心结论

Schur 分解的核心结论是：

对于任意复方阵 $A\in\mathbb{C}^{n\times n}$，都存在一个酉矩阵 $Q$ 和一个上三角矩阵 $T$，使得

$$
A = QTQ^{H}.
$$

其中：

- $Q^{H}$ 表示 $Q$ 的共轭转置；
- 酉矩阵满足 $Q^{H}Q=QQ^{H}=I$，表示换了一组“长度不变、彼此正交”的坐标基；
- 上三角矩阵 $T$ 的对角线元素，正好就是 $A$ 的全部特征值。

这句话的工程含义很直接：即使矩阵不能对角化，通常也仍然可以被稳定地改写成“几乎已经对角化”的三角形态，而三角形态足够读取谱信息。

如果 $A$ 是实矩阵，则常见写法是

$$
A = QTQ^{T},
$$

这里 $Q$ 是正交矩阵，满足 $Q^TQ=I$。但这时 $T$ 一般不一定是实数上三角，而是**准上三角**矩阵。所谓准上三角，是指主对角线上既可能有 $1\times 1$ 块，也可能有 $2\times 2$ 块：

- $1\times 1$ 块对应实特征值；
- $2\times 2$ 块对应一对复共轭特征值。

一个最小例子是二维旋转矩阵

$$
A=\begin{bmatrix}
0 & -1\\
1 & 0
\end{bmatrix}.
$$

它表示平面逆时针旋转 $90^\circ$。它的特征值满足

$$
\det(\lambda I-A)=
\det\begin{bmatrix}
\lambda & 1\\
-1 & \lambda
\end{bmatrix}
=\lambda^2+1,
$$

所以特征值是

$$
\lambda=\pm i.
$$

在复数域上，它的 Schur 形式可以写成对角矩阵

$$
T=\operatorname{diag}(i,-i),
$$

因此对角线直接给出特征值 $i,-i$。但在实数域上，不可能把它正交相似化成实对角矩阵，因为它根本没有实特征值，所以实 Schur 形式只能保留成一个 $2\times2$ 实块，本质上就是这种旋转结构本身。

| 场景 | Schur 形式 | 特征值如何出现 |
|---|---|---|
| 复 Schur | 上三角 | 直接在对角线上 |
| 实 Schur | 准上三角 | 实特征值对应 $1\times1$ 块，复共轭对对应 $2\times2$ 块 |

在数值计算中，Schur 形式不是附属结论，而是很多特征值算法真正追求的终点。QR 算法并不是直接去“解特征方程”，而是通过一连串正交或酉相似变换，把矩阵逐步逼近 Schur 形式。因为这类变换保持范数、不显著放大误差，所以它比直接操作特征多项式稳定得多。

---

## 问题定义与边界

问题可以表述为：

给定一个任意方阵 $A$，希望找到一个变换矩阵 $Q$，把它改写成更容易读取谱信息的形式 $T$。这里的谱信息主要指特征值；如果还保留了变换矩阵 $Q$，那么还可以继续分析不变子空间、模态方向和基变换。

为了避免概念混淆，先明确几个边界。

第一，**输入所在的数域决定输出结构**。

- 输入是复矩阵时，复 Schur 分解总能得到真正的上三角矩阵；
- 输入是实矩阵时，实 Schur 分解保持全程实数运算，但必须允许 $2\times2$ 块出现。

第二，**Schur 分解一般不唯一**。

不唯一主要来自三种自由度：

- 特征值在三角对角线上的顺序可以改变；
- 重复特征值对应的基可以重新选；
- 同一个不变子空间内还可以继续做正交或酉变换。

因此，不同软件、不同参数、甚至同一软件不同版本，得到的 $Q,T$ 可能外观不同，但只要满足分解关系并保留相同谱信息，就是正确结果。

第三，**工程实现通常先做 Hessenberg 化**。

上 Hessenberg 矩阵的定义是：除主对角线和下方第一条副对角线外，其余更低位置全为零。形状是

$$
H=
\begin{bmatrix}
* & * & * & * \\
* & * & * & * \\
0 & * & * & * \\
0 & 0 & * & *
\end{bmatrix}.
$$

它比一般稠密矩阵更有结构，但又保留全部特征值。对一般矩阵先做 Hessenberg 化，再做 QR 迭代，是标准数值线性代数流程，因为这样每一步迭代的代价和常数都会明显更低。

下面用一个结构例子说明实 Schur 的块形态。设某个实矩阵的特征值为

$$
\lambda_1=1,\quad \lambda_2=2,\quad \lambda_{3,4}=1\pm i.
$$

那么它的一个实 Schur 形式通常写成

$$
T=
\begin{bmatrix}
1 & * & * & * \\
0 & 2 & * & * \\
0 & 0 & a & b \\
0 & 0 & c & d
\end{bmatrix},
$$

其中右下角 $2\times2$ 块

$$
B=\begin{bmatrix}a&b\\c&d\end{bmatrix}
$$

的特征值正好是 $1\pm i$。这说明：即使原矩阵完全由实数组成，复特征值也仍会出现，只不过不是以单个对角元素的形式出现，而是编码在一个实二维块中。

| 输入类型 | 允许的 $Q$ | 输出 $T$ 的结构 | 是否能只读对角线 |
|---|---|---|---|
| 复矩阵 | 酉矩阵 | 上三角 | 能 |
| 实矩阵且全实谱 | 正交矩阵 | 可为上三角 | 基本能 |
| 实矩阵且含复共轭对 | 正交矩阵 | 准上三角，含 $2\times2$ 块 | 不能 |
| 大规模数值实现 | 正交/酉迭代 | 常先化到 Hessenberg 再求 Schur | 需结合块结构读取 |

因此，Schur 分解回答的不是“矩阵能不能对角化”，而是：

**不管矩阵能否对角化，都能否稳定地化到足够接近对角的标准形。**

Schur 给出的答案是：可以，复数域上是上三角，实数域上是准上三角。

---

## 核心机制与推导

Schur 分解的证明思路可以压缩成一句话：

**先找一个特征向量，把它放进第一列；再在剩余子空间中重复同样的事情。**

这就是它为什么总能走到上三角形式。

### 1. 复 Schur 的递归机制

先看复数域。由于代数基本定理，任意复方阵 $A\in\mathbb C^{n\times n}$ 至少有一个特征值 $\lambda_1$，因此存在非零特征向量 $\mathbf q_1$，满足

$$
A\mathbf q_1=\lambda_1\mathbf q_1.
$$

把它单位化，使 $\|\mathbf q_1\|_2=1$。然后把它扩展成一组标准正交基

$$
Q_1=[\mathbf q_1,\mathbf q_2,\dots,\mathbf q_n].
$$

因为 $Q_1$ 是酉矩阵，所以做相似变换

$$
Q_1^HAQ_1.
$$

在这组新基下，第一列会出现特殊结构。更具体地说，

$$
Q_1^HAQ_1 e_1
=
Q_1^HA\mathbf q_1
=
Q_1^H(\lambda_1\mathbf q_1)
=
\lambda_1 e_1.
$$

这说明新矩阵作用在第一基向量 $e_1$ 上时，只会落回 $e_1$ 本身，不会跑到下面坐标，因此它必须具有块上三角形状：

$$
Q_1^{H}AQ_1=
\begin{bmatrix}
\lambda_1 & * \\
0 & A_1
\end{bmatrix},
$$

其中 $A_1\in\mathbb C^{(n-1)\times(n-1)}$。

这里左下角是 $0$，本质原因是第一列已经对应一个不变一维子空间。白话说，矩阵对这条方向的作用已经被“锁住”了。

接着，对右下角子块 $A_1$ 再做同样的事情：

- 找到 $A_1$ 的一个特征向量；
- 扩展成新的正交基；
- 再做一次相似变换；
- 递归下去。

最终就得到某个酉矩阵 $Q$ 使得

$$
Q^HAQ=T,
$$

其中 $T$ 是上三角矩阵，于是

$$
A=QTQ^H.
$$

这就是复 Schur 分解。

### 2. 为什么上三角的对角线就是特征值

这一步经常被直接引用，但对新手最好说明原因。

如果

$$
T=
\begin{bmatrix}
t_{11} & * & \cdots & *\\
0 & t_{22} & \cdots & *\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & t_{nn}
\end{bmatrix},
$$

那么

$$
\det(\lambda I-T)=\prod_{k=1}^n (\lambda-t_{kk}),
$$

因为三角矩阵的行列式等于对角线元素之积。于是 $T$ 的特征值就是

$$
t_{11},t_{22},\dots,t_{nn}.
$$

又因为相似变换不改变特征多项式，所以 $A$ 与 $T$ 拥有相同特征值。因此，Schur 分解一旦得到，特征值就直接读出来了。

### 3. 实 Schur 为什么会出现 $2\times2$ 块

实数情形的区别不在于思想，而在于“能不能一直使用实基”。

- 如果当前特征值是实数，那么仍然可以抽出一个实特征向量，构造 $1\times1$ 块；
- 如果当前出现的是复特征值 $\alpha\pm \beta i$，就不能把单个复特征向量直接塞进实正交基中。

设复特征向量为

$$
z=u+iv,\quad u,v\in\mathbb R^n.
$$

若

$$
Az=(\alpha+\beta i)z,
$$

把实部和虚部分开，可得到

$$
Au=\alpha u-\beta v,\qquad
Av=\beta u+\alpha v.
$$

这说明由 $u,v$ 张成的二维实子空间在 $A$ 作用下保持不变。在基 $\{u,v\}$ 下，$A$ 在这个二维子空间上的表示矩阵是

$$
\begin{bmatrix}
\alpha & \beta\\
-\beta & \alpha
\end{bmatrix}
$$

或与之实相似的某个 $2\times2$ 实块。这个块的特征多项式是

$$
\lambda^2-2\alpha\lambda+(\alpha^2+\beta^2),
$$

对应的特征值正是

$$
\alpha\pm \beta i.
$$

所以实 Schur 中的 $2\times2$ 块，不是“没化简干净”，而是“在实数域里能够达到的正确终点”。

### 4. 与 QR 算法的直接关系

Schur 分解不仅是存在性定理，也是 QR 算法的收敛目标。

设第 $k$ 步矩阵为 $A_k$，对它做 QR 分解：

$$
A_k=Q_kR_k.
$$

然后交换乘法顺序，定义下一步

$$
A_{k+1}=R_kQ_k.
$$

把上式代回去：

$$
A_{k+1}=Q_k^HA_kQ_k
$$

在实数情形则写成

$$
A_{k+1}=Q_k^TA_kQ_k.
$$

因此每一步 QR 迭代都是一次正交或酉相似变换。相似变换不改变特征值，所以整个迭代过程一直保留谱信息。对很多矩阵，尤其配合位移策略后，$A_k$ 会逐步逼近 Schur 形式。

可以把理论、算法、实现三层关系概括成：

| 层次 | 核心动作 | 目标 |
|---|---|---|
| 理论 | 递归抽取不变子空间 | 证明 Schur 形式存在 |
| 算法 | QR 迭代 + 相似变换 | 逼近 Schur 形式 |
| 工程 | 先 Hessenberg 化，再做隐式 QR | 降低成本并提升稳定性 |

因此可以把整件事理解成：

1. 理论上，存在一组正交或酉基，使矩阵变成三角或准三角。
2. 数值上，QR 迭代在逐步逼近这组基。
3. 工程上，Hessenberg 预处理让这个过程更快也更稳。

---

## 代码实现

工程里通常不会手写 Schur 分解，而是调用 LAPACK 或其上层封装。

- 实 Schur 的经典底层例程是 `dgees`；
- 复 Schur 的经典底层例程是 `zgees`；
- Python 中最常见的入口是 `scipy.linalg.schur`。

下面先给出一个可直接运行的最小示例，展示复 Schur 和实 Schur 的差别，并补上新手最容易漏掉的验证步骤。

```python
import numpy as np
from scipy.linalg import schur

def print_matrix(name, M):
    print(f"{name} =")
    print(np.array2string(M, precision=4, suppress_small=True))
    print()

# 90 度旋转矩阵
A = np.array([
    [0.0, -1.0],
    [1.0,  0.0]
])

print_matrix("A", A)

# 复 Schur：返回上三角矩阵 Tc 和酉矩阵 Qc
Tc, Qc = schur(A, output="complex")
vals_c = np.diag(Tc)

print_matrix("Tc", Tc)
print_matrix("Qc", Qc)
print("complex eigenvalues from diag(Tc):", vals_c)
print()

# 验证分解关系 A = Q T Q^H
assert np.allclose(A, Qc @ Tc @ Qc.conj().T)

# 理论值是 i 和 -i，顺序可能不同
target = np.array([1j, -1j])
assert np.allclose(np.sort_complex(vals_c), np.sort_complex(target))

# 实 Schur：返回准上三角矩阵 Tr 和正交矩阵 Qr
Tr, Qr = schur(A, output="real")

print_matrix("Tr", Tr)
print_matrix("Qr", Qr)

# 验证分解关系 A = Q T Q^T
assert np.allclose(A, Qr @ Tr @ Qr.T)

# 这个例子在实数域里无法化成实对角，因此保留为 2x2 块
print("diag(Tr):", np.diag(Tr))
print("real Schur block eigenvalues:", np.linalg.eigvals(Tr))
```

这段代码有三个检查点：

| 检查项 | 代码 | 目的 |
|---|---|---|
| 分解关系是否成立 | `A == Q @ T @ Q^H` 或 `Q^T` | 确认不是只调用了 API，而是真满足定义 |
| 复 Schur 是否上三角 | `Tc` | 验证特征值可直接从对角线读取 |
| 实 Schur 是否为块结构 | `Tr` | 验证复共轭对会以 $2\times2$ 块出现 |

### 1. 如何从实 Schur 结果中正确读取特征值

对 `output="real"` 的结果，不能简单把 `np.diag(T)` 当全部特征值。需要识别：

- 若某个位置下方元素接近零，则对应一个 $1\times1$ 块；
- 若某个位置下方元素不接近零，则它与下一行列共同组成一个 $2\times2$ 块。

下面给出一个可运行的辅助函数：

```python
import numpy as np
from scipy.linalg import schur

def eigs_from_real_schur(T, tol=1e-12):
    """从实 Schur 形式 T 中提取全部特征值。"""
    n = T.shape[0]
    eigs = []
    i = 0
    while i < n:
        if i == n - 1 or abs(T[i + 1, i]) < tol:
            eigs.append(T[i, i])
            i += 1
        else:
            B = T[i:i+2, i:i+2]
            eigs.extend(np.linalg.eigvals(B))
            i += 2
    return np.array(eigs)

A = np.array([
    [1.0, -3.0, 0.0, 0.0],
    [3.0,  1.0, 0.0, 0.0],
    [0.0,  0.0, 2.0, 1.0],
    [0.0,  0.0, 0.0, 4.0]
])

T, Q = schur(A, output="real")
eigs = eigs_from_real_schur(T)

assert np.allclose(A, Q @ T @ Q.T)

print("T =")
print(T)
print()
print("eigenvalues from real Schur:", np.sort_complex(eigs))
print("eigenvalues from np.linalg.eigvals:", np.sort_complex(np.linalg.eigvals(A)))
```

如果某个 $2\times2$ 块是

$$
B=\begin{bmatrix}
a & b\\
c & d
\end{bmatrix},
$$

那么它对应的特征值由二次方程

$$
\lambda^2-(a+d)\lambda+(ad-bc)=0
$$

得到，即

$$
\lambda =
\frac{(a+d)\pm\sqrt{(a+d)^2-4(ad-bc)}}{2}.
$$

这就是为什么实 Schur 中不能只看对角线。对角线上的 $a,d$ 只是块内元素，不一定就是最终特征值。

### 2. 一个更贴近工程的示例：稳定性判断

设连续时间线性系统

$$
\dot x = Ax.
$$

判定系统渐近稳定的一个常用条件是：矩阵 $A$ 的所有特征值实部都严格小于 $0$。工程里常见做法是先做复 Schur，再读对角线。

```python
import numpy as np
from scipy.linalg import schur

A = np.array([
    [0.0,  1.0,  0.0],
    [-5.0, -2.0, 1.0],
    [0.0, -3.0, -4.0]
])

T, Q = schur(A, output="complex")
eigs = np.diag(T)

assert np.allclose(A, Q @ T @ Q.conj().T)

stable = np.all(np.real(eigs) < 0)

print("Schur form T =")
print(T)
print()
print("eigenvalues =", eigs)
print("real parts  =", np.real(eigs))
print("stable =", stable)
```

这个流程在控制、结构动力学、模型降阶、振动分析中都非常常见，因为：

- Schur 分解稳定；
- 特征值读取简单；
- 若还保留 $Q$，后续还能继续做模态子空间分析。

### 3. 什么时候优先用复 Schur，什么时候用实 Schur

| 目标 | 推荐形式 | 原因 |
|---|---|---|
| 只想简单读取特征值 | 复 Schur | 上三角，直接读对角线 |
| 必须保持实数链路 | 实 Schur | 不引入复数存储 |
| 后续还要分析复模态 | 复 Schur 更直接 | 省去块识别逻辑 |
| 与传统实数值库兼容 | 实 Schur | 便于和实矩阵算法衔接 |

一个实用判断是：

- 如果你不介意复数数组，优先用复 Schur；
- 如果项目约束要求全流程使用实数，再用实 Schur，但必须实现块读取逻辑。

---

## 工程权衡与常见坑

Schur 分解在工程上很稳，但“理论知道了”不等于“代码就不会错”。以下几个坑最常见。

| 常见坑 | 错误表现 | 正确做法 |
|---|---|---|
| 误以为 Schur 唯一 | 不同机器结果顺序不同就怀疑错误 | 接受特征值和块顺序可重排 |
| 把实 Schur 当普通上三角 | 直接 `np.diag(T)` 读特征值 | 识别 $1\times1$ 与 $2\times2$ 块 |
| 只保存 `T` 不保存 `Q` | 无法回到原坐标解释模态 | 同时保留 `Q` 与 `T` |
| 跳过标准库流程 | 自己写 QR 结果不稳或性能差 | 用 LAPACK/Scipy 等成熟实现 |
| 把 Schur 当 Jordan | 期待看到广义特征向量链 | 明确 Schur 只保证三角化 |
| 忽略数值阈值 | 把很小的舍入误差当成非零块 | 用容差 `tol` 判断块边界 |

### 1. 误把 `diag(T)` 当全部特征值

这是最常见的实现错误。如下写法：

```python
T, _ = schur(A, output="real")
eigs = np.diag(T)   # 错误：含复共轭对时会漏信息
```

如果 $T$ 中含有

$$
\begin{bmatrix}
a & b\\
c & d
\end{bmatrix},
$$

那么真实特征值是这个块的两个根，而不是 $a,d$ 两个对角元素。

### 2. 只保留 `T`，丢掉 `Q`

新手常把 $Q$ 看成中间变量，只保留 $T$。这在只做一次性特征值读取时勉强可以，但在下面这些任务里会直接出问题：

- 模态坐标变换；
- 特征子空间提取；
- 不变子空间投影；
- 平衡截断、模型降阶；
- 稳定/不稳定子空间分离。

原因很简单：$T$ 告诉你“谱结构是什么”，而 $Q$ 告诉你“这些结构在原坐标里朝哪个方向”。

### 3. 忽略 Schur 与对角化的区别

Schur 总存在，但对角化不总存在。若矩阵不可对角化，Schur 仍然成立。这一点对理解算法边界很重要，因为现实中的矩阵未必有完整特征向量基。

### 4. 大规模稀疏矩阵盲目做完整 Schur

完整 Schur 分解适合中小规模、稠密、需要全谱的场景。如果矩阵很大而且稀疏，而你只关心少量特征值，那么完整 Schur 往往是过度计算。

可以把实践建议压缩成以下检查表：

- `TODO:` 需要最直接的谱读取时，优先用复 Schur。
- `TODO:` 必须保持实数链路时，用实 Schur，但实现里必须识别 $2\times2$ 块。
- `TODO:` 后续还要做模态或子空间分析时，`Q` 和 `T` 都要保存。
- `TODO:` 大规模问题不要手写 QR，直接调用成熟库。
- `TODO:` 写块识别逻辑时，一定设置数值容差，不要用“是否严格等于 0”判断。

---

## 替代方案与适用边界

Schur 分解不是唯一的谱工具，但它通常是最稳、最通用的默认方案。要理解它的价值，最有效的方法是把它和几个常见替代方案放在一起比较。

### 1. 与特征值分解比较

如果矩阵可对角化，可以写成

$$
A=X\Lambda X^{-1},
$$

其中 $\Lambda$ 是对角矩阵，$X$ 的列是特征向量。这个形式当然最直观，因为特征值都在对角线上，特征向量也明明白白。

但问题在于，它要求矩阵有足够多线性无关的特征向量。这个要求并不总满足。

例如 Jordan 型矩阵

$$
J=
\begin{bmatrix}
1 & 1\\
0 & 1
\end{bmatrix}
$$

只有一个特征值 $1$。计算

$$
J-I=
\begin{bmatrix}
0 & 1\\
0 & 0
\end{bmatrix},
$$

解 $(J-I)x=0$ 可知只有一维特征向量空间，因此它不可对角化。但它本身已经是上三角矩阵，所以 Schur 分解显然存在。这说明：

**Schur 比普通特征值分解更普适。**

### 2. 与 Jordan 分解比较

Jordan 分解能给出更细的结构信息。它不仅告诉你特征值，还告诉你广义特征向量链，理论上信息最完整。

但数值上它很脆弱。一个很小的扰动，Jordan 块结构就可能改变。也就是说：

- 理论上，Jordan 形式适合证明和分类；
- 工程上，Jordan 形式几乎不是主力工具。

Schur 分解恰好处在“信息够用”和“数值稳定”之间的平衡点上。它不追求暴露最细结构，而追求一个稳定、可计算、可复现的标准形。

### 3. 与 Krylov/Arnoldi 类方法比较

当矩阵非常大且稀疏时，完整 Schur 往往不再合适。此时更常见的方法是 Krylov 子空间方法，如 Arnoldi 或 Lanczos。

它们的核心思想不是求整个谱，而是：

- 在一个逐步扩展的小子空间里工作；
- 只逼近你关心的少数特征值；
- 保留稀疏结构带来的计算优势。

例如一个状态维数几十万的稀疏系统，如果你只关心最慢衰减的前 10 个模态，那么完整 Schur 基本不现实。这时 Arnoldi 通常更合理。

### 4. 适用场景对比

| 方法 | 主要特点 | 适用场景 |
|---|---|---|
| Schur 分解 | 稳定、普适、可处理不可对角化矩阵 | 中小规模稠密矩阵，全谱分析 |
| 特征值分解 | 结果直观，特征向量显式出现 | 可对角化且需要完整模态基 |
| Jordan 分解 | 理论信息最细 | 理论推导，不适合数值计算 |
| Krylov/Arnoldi | 只求少数特征值，适合稀疏大矩阵 | 大规模 PDE、控制、结构分析 |

因此可以用下面这组判断来选方法：

- 需要一个稳定、通用、默认安全的谱表示，用 Schur。
- 需要完整对角基，而且矩阵确实可对角化，可考虑特征值分解。
- 只做理论结构分析，才考虑 Jordan。
- 只关心少量特征值，且矩阵很大很稀疏，转向 Krylov 类方法。

Schur 的定位不是“唯一正确答案”，而是“在绝大多数稠密数值场景中，最稳妥的中间表示”。

---

## 参考资料

- Gene H. Golub, Charles F. Van Loan, *Matrix Computations*, 4th ed.  
  经典数值线性代数教材。Schur 分解、Hessenberg 化、QR 算法的工程视角都非常完整。

- Lloyd N. Trefethen, David Bau III, *Numerical Linear Algebra*.  
  对“为什么正交变换稳定、为什么 QR 以 Schur 为目标”解释得很清楚，适合建立数值直觉。

- Schur decomposition（Wikipedia）  
  概念总览，含复 Schur、实 Schur 和准上三角块结构。  
  https://en.wikipedia.org/wiki/Schur_decomposition

- QR algorithm（Wikipedia）  
  说明 QR 迭代与 Schur 形式之间的直接关系。  
  https://en.wikipedia.org/wiki/QR_algorithm

- Nicholas Hu, UCLA Notes on Schur Decomposition  
  递归构造、证明思路和数值线性代数视角较清晰。  
  https://math.ucla.edu/~njhu/notes/nla/eig/schur/

- LAPACK `dgees` / `zgees` 文档  
  实 Schur 与复 Schur 的底层工程接口来源，适合理解库层行为。  
  https://www.netlib.org/lapack/

- SciPy `scipy.linalg.schur` 文档  
  Python 中最直接的工程入口，参数、返回值和实/复输出形式都有说明。  
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.schur.html
