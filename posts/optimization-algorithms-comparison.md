## 核心结论

一阶方法和二阶方法的差别，不在于“谁更高级”，而在于它们使用了多少局部信息，以及这些信息是否足以支撑更大的、更加可靠的步子。

一阶方法只看梯度。梯度表示当前位置最陡上升的方向，取负号后就是最直接的下降方向。典型更新式为
$$
x_{k+1}=x_k-\eta \nabla f(x_k)
$$
其中 $\eta$ 是学习率，决定每次更新走多远。它的优点是单步便宜、实现简单、适合高维；缺点是步长敏感，遇到病态问题时容易来回震荡，整体通常只有线性收敛。

二阶方法同时看梯度和 Hessian。Hessian 是二阶偏导组成的矩阵，用来描述不同方向上的局部曲率。Newton 法的更新式为
$$
x_{k+1}=x_k-[\nabla^2 f(x_k)]^{-1}\nabla f(x_k)
$$
它不是简单沿梯度方向走，而是先用曲率修正方向和步幅。若目标函数在最优点附近足够光滑且 Hessian 正定，Newton 法可达到局部二次收敛，也就是误差会非常快地缩小。

工程上真正常用的往往不是“纯 Newton”，而是拟牛顿。拟牛顿不直接计算 Hessian，而是从梯度变化中近似曲率，其中 BFGS、L-BFGS 最典型。它们保留二阶信息带来的快收敛，又避免显式求 Hessian 的高代价。

共轭梯度法处在中间地带。它主要用于求解对称正定线性系统 $Ax=b$，也可作为二阶优化中的子步骤。它不显式构造 Hessian，只需要矩阵向量积，但会利用历史方向构造更有效的搜索方向，通常比普通梯度下降更快。

如果把这些方法放到同一张表里，差异会更清楚：

| 方法 | 使用信息 | 单步主要成本 | 典型收敛特征 | 适合场景 |
|---|---|---:|---|---|
| 梯度下降 GD | 梯度 | 低 | 线性收敛 | 超高维、数据流式、便宜迭代 |
| Newton | 梯度 + Hessian | 高，常含 $O(n^3)$ 线性代数 | 局部二次收敛 | 中等维度、凸且光滑、要求高精度 |
| BFGS / L-BFGS | 梯度 + 历史差分 | 中 | 通常快于 GD，接近二阶效果 | 中高维、通用工程优化 |
| 共轭梯度 CG | 矩阵向量积 + 历史方向 | 中低 | 在线性 SPD 系统上很高效 | 稀疏线性系统、Hessian-free 子问题 |

一句话概括：一阶法用更少信息换低单步成本，二阶法用更多信息换更少迭代轮数；工程选择的核心是总成本，而不是单次更新看起来是否复杂。

---

## 问题定义与边界

我们讨论的目标是无约束优化：
$$
\min_{x\in \mathbb{R}^n} f(x)
$$

这里 $f(x)$ 是目标函数，也就是希望尽量小的量。梯度
$$
\nabla f(x)=\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$
给出局部一阶变化；Hessian
$$
\nabla^2 f(x)=\left[\frac{\partial^2 f}{\partial x_i\partial x_j}\right]_{i,j}
$$
给出局部二阶变化。

分析这些方法时，边界条件必须说清楚，否则“快”没有意义。因为优化算法的速度，取决于问题本身的几何结构，而不是算法名字。

第一，梯度下降只要求可微，门槛最低，但速度高度依赖学习率和条件数。条件数常记为
$$
\kappa=\frac{\lambda_{\max}}{\lambda_{\min}}
$$
其中 $\lambda_{\max},\lambda_{\min}$ 是 Hessian 或矩阵的最大、最小特征值。若 $\kappa$ 很大，说明不同方向的尺度差别很大，等高线会被拉成长椭圆，GD 会像在狭长山谷里左右横跳。

第二，Newton 法要求 Hessian 可算、可逆，并且最好正定。正定表示
$$
d^\top \nabla^2 f(x)\, d>0,\qquad \forall d\neq 0
$$
这意味着局部形状像碗，而不是马鞍或山顶。若 Hessian 非正定，$-[\nabla^2 f]^{-1}\nabla f$ 不一定是下降方向，算法可能直接走错。

第三，共轭梯度法标准形式针对 $A$ 为对称正定矩阵的线性系统
$$
Ax=b
$$
它等价于最小化二次函数
$$
\phi(x)=\frac12 x^\top A x-b^\top x
$$
因此常被看作“二次优化的专用快速方法”。这里的关键不是“它像优化”，而是“它本质上就是在解一个特殊优化问题”。

下面这个表格把前提条件列全：

| 方法 | 函数要求 | 是否需要 Hessian | 是否要求正定性 | 维度适用性 |
|---|---|---|---|---|
| GD | 可微 | 否 | 否 | 很高 |
| Newton | 二阶可微 | 是 | 通常需要 | 中等 |
| L-BFGS | 可微，最好较平滑 | 否，近似即可 | 通过更新规则尽量保持 | 高于 Newton |
| CG | 二次型或 SPD 线性系统 | 不显式需要 | 需要 $A$ 为 SPD | 很高，尤其稀疏 |

玩具例子先看一维函数
$$
f(x)=(x-2)^2
$$
它的梯度是
$$
\nabla f(x)=2(x-2)
$$
Hessian 是常数
$$
\nabla^2 f(x)=2
$$
从 $x_0=0$ 出发：

- 若用梯度下降且 $\eta=0.5$，则
  $$
  x_1=0-0.5\times(-4)=2
  $$
  一步到最优点。
- 若用 Newton 法，则
  $$
  x_1=0-\frac{1}{2}\times(-4)=2
  $$
  也是一步到最优点。

这个例子说明两件事。第一，简单二次函数上，选对学习率时 GD 也能很快。第二，Newton 不需要手工调这个学习率，因为曲率已经把步长编码进去了。

但这个例子也有局限。它是一维、强凸、Hessian 恒定的理想情形。真实问题更接近下面这个二维二次函数：
$$
f(x_1,x_2)=100x_1^2+x_2^2
$$
其 Hessian 为
$$
\nabla^2 f(x)=
\begin{bmatrix}
200 & 0\\
0 & 2
\end{bmatrix}
$$
两个方向的曲率相差 $100$ 倍。此时如果学习率太小，GD 在平缓方向推进慢；如果学习率太大，又会在陡峭方向震荡。这正是“一阶信息不够”的典型表现。

---

## 核心机制与推导

### 1. 梯度下降为什么只是一阶法

在点 $x_k$ 附近做一阶泰勒展开：
$$
f(x_k+\Delta x)\approx f(x_k)+\nabla f(x_k)^\top \Delta x
$$
如果只看这一项，要让函数下降最快，就取
$$
\Delta x=-\eta \nabla f(x_k)
$$
于是得到
$$
x_{k+1}=x_k-\eta \nabla f(x_k)
$$

这里的逻辑很直接：一阶展开只告诉我们“局部往哪边下降”，并不告诉我们“这个方向到底弯得多厉害”。所以 GD 必须额外引入学习率 $\eta$，或者通过线搜索来试探合理步长。

更具体地说，若目标函数是二次型
$$
f(x)=\frac12 x^\top A x-b^\top x,\qquad A\succ 0
$$
则梯度下降变成
$$
x_{k+1}=x_k-\eta(Ax_k-b)
$$
定义误差 $e_k=x_k-x^\star$，其中 $x^\star=A^{-1}b$，可得
$$
e_{k+1}=(I-\eta A)e_k
$$
也就是说，GD 的收敛速度由矩阵 $I-\eta A$ 的谱半径决定。若各方向尺度差异大，就很难用单个 $\eta$ 同时兼顾所有方向。

这也是新手最容易误解的地方。GD 慢，不是因为“只用梯度所以落后”，而是因为它把所有方向都交给同一个标量步长处理，而真实问题往往需要方向相关的缩放。

### 2. Newton 法为什么能更快

在点 $x_k$ 附近做二阶泰勒展开：
$$
f(x_k+\Delta x)\approx f(x_k)+\nabla f(x_k)^\top \Delta x+\frac12 \Delta x^\top \nabla^2 f(x_k)\Delta x
$$
把这个近似函数对 $\Delta x$ 求最小值，令导数为 0：
$$
\nabla f(x_k)+\nabla^2 f(x_k)\Delta x=0
$$
解得
$$
\Delta x=-[\nabla^2 f(x_k)]^{-1}\nabla f(x_k)
$$
于是得到 Newton 更新式：
$$
x_{k+1}=x_k-[\nabla^2 f(x_k)]^{-1}\nabla f(x_k)
$$

这一步的含义是：它不再使用“统一步长”，而是让 Hessian 在不同方向上分别决定缩放。曲率大的方向少走一点，曲率小的方向多走一点。

若目标函数本身就是二次函数
$$
f(x)=\frac12 x^\top A x-b^\top x
$$
则
$$
\nabla f(x)=Ax-b,\qquad \nabla^2 f(x)=A
$$
Newton 更新为
$$
x_{k+1}=x_k-A^{-1}(Ax_k-b)=A^{-1}b=x^\star
$$
也就是说，二次函数上 Newton 一步到精确解。这个结论非常重要，因为它解释了为什么二阶信息在“局部近似成二次型”的区域会特别有效。

局部二次收敛通常写成
$$
\|x_{k+1}-x^\star\|\le C\|x_k-x^\star\|^2
$$
这表示误差不是按固定比例缩小，而是按“平方级别”缩小。例如误差从 $10^{-1}$ 变成 $10^{-2}$ 后，下一步可能直接变成 $10^{-4}$、再下一步变成 $10^{-8}$。这就是 Newton 法在最优点附近非常快的原因。

但要注意，这个结论是局部性质，不是全局保证。离最优点很远时，二阶近似可能很差，因此工程上常配合线搜索或信赖域，而不是裸奔使用纯 Newton。

### 3. 拟牛顿为什么是工程上的主力

纯 Newton 的瓶颈是 Hessian。维度为 $n$ 时，存 Hessian 需要 $O(n^2)$ 内存，解线性方程常要 $O(n^3)$ 运算。对于中高维问题，这个成本很快就不可接受。

于是拟牛顿改成维护近似矩阵 $B_k\approx \nabla^2 f(x_k)$，只从相邻两步的信息更新：
$$
s_k=x_{k+1}-x_k,\qquad y_k=\nabla f(x_{k+1})-\nabla f(x_k)
$$
BFGS 的核心要求是满足割线条件
$$
B_{k+1}s_k=y_k
$$
也就是“沿刚走过的方向，曲率近似要匹配真实梯度变化”。

若维护的是逆 Hessian 近似 $H_k\approx B_k^{-1}$，BFGS 常写为
$$
H_{k+1}=(I-\rho_k s_k y_k^\top)H_k(I-\rho_k y_k s_k^\top)+\rho_k s_k s_k^\top
$$
其中
$$
\rho_k=\frac{1}{y_k^\top s_k}
$$

这条更新式看起来复杂，但抓住两个事实就够了：

- 它只使用梯度差分，不显式计算 Hessian。
- 当 $y_k^\top s_k>0$ 时，更新能维持正定性，因而更容易生成下降方向。

L-BFGS 则进一步只保留最近 $m$ 组 $(s_k,y_k)$，不存完整矩阵。这样存储从 $O(n^2)$ 降到 $O(nm)$，计算也从“矩阵级操作”变成“少量向量内积和加法”。

对新手而言，可以把 L-BFGS 理解成一句话：它不记整个曲率矩阵，只记最近几次“走了多远”和“梯度变了多少”，再把这些历史压缩成一个近似二阶方向。

### 4. 共轭梯度为什么比普通梯度更聪明

对于二次函数
$$
\phi(x)=\frac12 x^\top A x-b^\top x,\quad A=A^\top\succ0
$$
其梯度为
$$
\nabla \phi(x)=Ax-b
$$
记残差
$$
r_k=b-Ax_k
$$
它就是负梯度。普通最速下降每次只沿当前残差方向走，而 CG 会构造一组 $A$-共轭方向 $p_0,p_1,\dots$，满足
$$
p_i^\top A p_j=0,\quad i\neq j
$$

“共轭”这个词容易让新手卡住。最实用的理解是：如果普通梯度下降是在欧氏几何里反复修修补补，那么 CG 是按照矩阵 $A$ 的几何结构，故意选一组互不打架的方向。沿一个方向完成的下降，不会被后续方向严重抵消。

标准迭代为
$$
\alpha_k=\frac{r_k^\top r_k}{p_k^\top A p_k},\qquad
x_{k+1}=x_k+\alpha_k p_k
$$
$$
r_{k+1}=r_k-\alpha_k A p_k,\qquad
\beta_k=\frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k}
$$
$$
p_{k+1}=r_{k+1}+\beta_k p_k
$$

对 $n$ 维 SPD 系统，精确算术下 CG 最多 $n$ 步可到精确解。这个结论的根本原因是：每一步都在一个更大的 Krylov 子空间中找最优解，
$$
\mathcal{K}_k(A,r_0)=\mathrm{span}\{r_0,Ar_0,A^2r_0,\dots,A^{k-1}r_0\}
$$
因此它不是简单重复局部下降，而是在逐步扩展可表达的方向空间。

一个 $2\times 2$ 玩具例子可以说明它与历史方向的关系。设
$$
A=\begin{bmatrix}4&1\\1&3\end{bmatrix},\quad b=\begin{bmatrix}1\\2\end{bmatrix}
$$
从 $x_0=0$ 出发时，$r_0=b$，第一步取
$$
p_0=r_0=\begin{bmatrix}1\\2\end{bmatrix}
$$
计算
$$
\alpha_0=\frac{r_0^\top r_0}{p_0^\top A p_0}
=\frac{1^2+2^2}{[1,2]\begin{bmatrix}6\\7\end{bmatrix}}
=\frac{5}{20}
=\frac14
$$
因此
$$
x_1=x_0+\alpha_0 p_0=\begin{bmatrix}0.25\\0.5\end{bmatrix}
$$
新残差为
$$
r_1=b-Ax_1=\begin{bmatrix}1\\2\end{bmatrix}-\begin{bmatrix}1.5\\1.75\end{bmatrix}
=\begin{bmatrix}-0.5\\0.25\end{bmatrix}
$$
第二步不直接取 $p_1=r_1$，而是取
$$
p_1=r_1+\beta_0 p_0,\qquad
\beta_0=\frac{r_1^\top r_1}{r_0^\top r_0}=\frac{0.3125}{5}=0.0625
$$
于是
$$
p_1=\begin{bmatrix}-0.5\\0.25\end{bmatrix}+0.0625\begin{bmatrix}1\\2\end{bmatrix}
=\begin{bmatrix}-0.4375\\0.375\end{bmatrix}
$$
这个新方向不是“当前残差方向本身”，而是“当前残差 + 旧方向修正项”。这正是 CG 能比最速下降更有效的原因。

---

## 代码实现

下面给出三个最小可运行版本，重点看“每步需要什么信息”，并把测试一起写全，避免示例代码只能看不能跑。

先看一维二次函数上的 GD 与 Newton：

```python
import numpy as np

def gd_quadratic_1d(x0, eta=0.1, steps=50):
    x = float(x0)
    history = [x]
    for _ in range(steps):
        grad = 2.0 * (x - 2.0)
        x = x - eta * grad
        history.append(x)
    return x, history

def newton_quadratic_1d(x0, steps=5):
    x = float(x0)
    history = [x]
    for _ in range(steps):
        grad = 2.0 * (x - 2.0)
        hessian = 2.0
        x = x - grad / hessian
        history.append(x)
    return x, history

x_star, hist_newton = newton_quadratic_1d(0.0, steps=1)
assert abs(x_star - 2.0) < 1e-12
assert hist_newton == [0.0, 2.0]

x_gd, hist_gd = gd_quadratic_1d(0.0, eta=0.1, steps=50)
assert abs(x_gd - 2.0) < 1e-4
assert len(hist_gd) == 51
```

上面这段代码里：

- GD 只需要 `grad`
- Newton 需要 `grad` 和 `hessian`
- 一维时 `grad / hessian` 很简单，多维时要解线性系统

多维 Newton 的核心操作通常写成解方程，而不是显式求逆：

```python
import numpy as np

def newton_optimize(x0, grad_fn, hess_fn, steps=10, tol=1e-10):
    x = np.asarray(x0, dtype=float).copy()
    history = [x.copy()]

    for _ in range(steps):
        g = grad_fn(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess_fn(x)

        # 数值计算里更推荐 solve(H, g)，而不是 inv(H) @ g
        delta = np.linalg.solve(H, g)
        x = x - delta
        history.append(x.copy())

    return x, history

def grad_quad(x):
    A = np.array([[4.0, 0.0], [0.0, 2.0]])
    b = np.array([8.0, 4.0])
    return A @ x - b

def hess_quad(x):
    return np.array([[4.0, 0.0], [0.0, 2.0]])

x0 = np.array([0.0, 0.0])
x_star, hist = newton_optimize(x0, grad_quad, hess_quad, steps=3)
assert np.allclose(x_star, np.array([2.0, 2.0]), atol=1e-12)
```

L-BFGS 的完整实现较长，但核心就是“两次循环”。下面给出一个可以直接运行的方向计算函数。这里返回的是近似逆 Hessian 乘梯度后的结果，优化时通常取负号作为搜索方向。

```python
import numpy as np

def lbfgs_two_loop(g, history):
    q = g.astype(float).copy()
    alphas = []
    rhos = []

    for s, y in reversed(history):
        ys = float(y @ s)
        if ys <= 1e-12:
            raise ValueError("Encountered non-positive curvature: y^T s <= 0")
        rho = 1.0 / ys
        alpha = rho * (s @ q)
        q = q - alpha * y
        alphas.append(alpha)
        rhos.append(rho)

    if history:
        s_last, y_last = history[-1]
        gamma = float((s_last @ y_last) / (y_last @ y_last))
    else:
        gamma = 1.0

    r = gamma * q

    for (s, y), alpha, rho in zip(history, reversed(alphas), reversed(rhos)):
        beta = rho * (y @ r)
        r = r + s * (alpha - beta)

    return r

g = np.array([2.0, -1.0])
history = [
    (np.array([1.0, 0.0]), np.array([2.0, 0.0])),
    (np.array([0.0, 1.0]), np.array([0.0, 3.0])),
]
Hg = lbfgs_two_loop(g, history)
assert Hg.shape == (2,)
```

上面这段代码有两个新手容易忽略的点：

- `y @ s` 不能接近 0，否则更新会不稳定。
- L-BFGS 只算“方向”，通常还要配合线搜索决定步长。

共轭梯度法更适合展示“只用矩阵向量积”的优势：

```python
import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    n = b.shape[0]
    if max_iter is None:
        max_iter = n

    x = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float).copy()
    r = b - A @ x
    p = r.copy()
    rs_old = float(r @ r)

    if np.sqrt(rs_old) < tol:
        return x

    for _ in range(max_iter):
        Ap = A @ p
        denom = float(p @ Ap)
        if denom <= 0:
            raise ValueError("A must be symmetric positive definite for standard CG.")

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)

        if np.sqrt(rs_new) < tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x

A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
x = conjugate_gradient(A, b, tol=1e-12, max_iter=10)
assert np.allclose(A @ x, b, atol=1e-10)
assert np.allclose(x, np.array([1.0 / 11.0, 7.0 / 11.0]), atol=1e-10)
```

如果想把三种方法放在同一个可对比的实验里，下面这个二维病态二次函数更有代表性：

```python
import numpy as np

A = np.array([[100.0, 0.0], [0.0, 1.0]])
b = np.array([0.0, 0.0])

def f(x):
    return 0.5 * x @ A @ x - b @ x

def grad(x):
    return A @ x - b

def hess(x):
    return A

x0 = np.array([10.0, 10.0])

# GD: 需要较小步长，否则沿高曲率方向震荡
x = x0.copy()
eta = 0.01
for _ in range(2000):
    x = x - eta * grad(x)
assert np.linalg.norm(x) < 1e-6

# Newton: 二次型上一轮到位
x_newton, _ = newton_optimize(x0, grad, hess, steps=1)
assert np.allclose(x_newton, np.zeros(2), atol=1e-12)

# CG: SPD 二次问题至多 n 步
x_cg = conjugate_gradient(A, b, x0=x0, tol=1e-12, max_iter=2)
assert np.allclose(x_cg, np.zeros(2), atol=1e-10)
```

真实工程例子是逻辑回归。逻辑回归的目标函数通常是光滑凸函数。对二分类数据 $(x_i,y_i)$，其负对数似然可写为
$$
L(w)=\sum_{i=1}^m \log\bigl(1+\exp(-y_i w^\top x_i)\bigr)+\frac{\lambda}{2}\|w\|_2^2
$$
其梯度容易求，Hessian 也存在，但完整 Hessian 在高维下偏贵。因此工程上常见选择是 L-BFGS：保留曲率信息、减少训练轮数，同时避免构造完整 Hessian。

---

## 工程权衡与常见坑

真正落地时，核心不是“理论上谁更快”，而是“总时间、总内存、数值稳定性谁更划算”。

| 方法 | 每步成本 | 迭代轮数 | 内存 | 常见失效原因 | 常见缓解策略 |
|---|---:|---:|---:|---|---|
| GD | 低 | 高 | 低 | 学习率难调、病态震荡 | 动量、Adam、线搜索、预条件 |
| Newton | 高 | 低 | 高 | Hessian 非正定、求解昂贵 | 阻尼、正则化、信赖域 |
| L-BFGS | 中 | 低到中 | 中低 | 历史信息质量差、噪声梯度不稳定 | 线搜索、限制历史长度 |
| CG | 中低 | 中 | 低 | 条件数差、浮点误差累积 | 预条件、重启 |

第一个坑是“不要显式求逆”。实现 Newton 时，很多新手写
$$
\Delta x = H^{-1} g
$$
然后在代码里写 `np.linalg.inv(H) @ g`。这在数值分析里通常不是好做法，因为更慢也更不稳定。正确做法是解线性方程
$$
H\Delta x=g
$$
对应代码是 `np.linalg.solve(H, g)`。

第二个坑是“Newton 不天然稳定”。如果 Hessian 非正定，Newton 方向可能不是下降方向。更稳的做法是使用阻尼版本：
$$
x_{k+1}=x_k-\alpha_k[\nabla^2 f(x_k)]^{-1}\nabla f(x_k),\qquad 0<\alpha_k\le 1
$$
或者做正则化：
$$
(\nabla^2 f(x_k)+\lambda I)d_k=-\nabla f(x_k)
$$
这相当于把局部模型变得更“像碗”。

第三个坑是“CG 很怕条件数差”。经典误差界可写成
$$
\|x_k-x^\star\|_A
\le
2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k
\|x_0-x^\star\|_A
$$
其中 $\kappa$ 是矩阵 $A$ 的条件数。这个式子说明：条件数越差，CG 需要的步数越多。因此大规模线性系统里，预条件器不是可选优化，而是核心组件。

第四个坑是“拟牛顿并不适合所有深度学习问题”。若梯度本身噪声很大，比如超大规模 mini-batch 训练，L-BFGS 保存的历史曲率可能不稳定，此时 SGD、Momentum、Adam 往往更稳。原因不是 L-BFGS 理论差，而是它依赖“梯度差分能反映真实曲率”，而噪声会破坏这一点。

第五个坑是“线搜索并不是装饰”。很多教程把方向和步长分开讲，容易让人误以为方向对了就够了。实际上在 BFGS、Newton 这类方法里，好的步长策略直接决定更新是否稳定，常见条件包括 Armijo 条件与 Wolfe 条件。对新手而言，只要记住一点：二阶方向通常不能脱离步长控制单独使用。

一个真实工程例子是逻辑回归训练。若样本数几十万、特征几千维，完整 Hessian 可能仍然偏贵，但纯 GD 又要很多轮。此时 L-BFGS 很合适：每轮成本可控，通常几轮到十几轮就能把目标值明显压下来。反过来，如果特征达到百万级稀疏维度，存储和线搜索成本会变得敏感，很多系统会改用 SGD、坐标下降或近端方法。

---

## 替代方案与适用边界

如果把优化方法按“是否显式使用二阶信息”排开，可以得到更实用的选择图：

| 方法 | 依赖信息 | 适用范围 | 典型应用 |
|---|---|---|---|
| SGD / Momentum | 随机梯度 | 超大规模、噪声训练 | 深度学习 |
| GD + 线搜索 | 全量梯度 | 中小规模光滑问题 | 教学、基线方法 |
| L-BFGS | 梯度 + 有限历史 | 中高维光滑优化 | 逻辑回归、参数估计 |
| Newton / Trust Region | 梯度 + Hessian | 中等维度高精度优化 | 科学计算、数值优化 |
| Hessian-free + CG | Hessian 向量积 | 大模型二阶近似 | 大规模二阶训练 |
| 预条件 CG | 稀疏矩阵向量积 | SPD 大线性系统 | PDE、图计算、最小二乘 |

L-BFGS 是 Newton 的直接工程替代：不算完整 Hessian，但保留曲率信息，适合“想要比 GD 快，又撑不起 Newton 成本”的场景。

Hessian-free + CG 是另一条路线。它不构造完整 Hessian，而是只需要 Hessian 向量积 $Hv$。这意味着可以通过自动微分或结构化算子，把 Newton 子问题
$$
\nabla^2 f(x_k)\, d = -\nabla f(x_k)
$$
交给 CG 近似求解。这样做的好处是保留二阶方向感，同时避免 $O(n^2)$ 存储。

这里再补一个常见概念。所谓 Hessian 向量积，不是先算完整 Hessian 再乘向量，而是直接求
$$
v \mapsto \nabla^2 f(x)\, v
$$
这类算子。很多自动微分系统都能在接近一次梯度计算的成本下给出这个结果，因此非常适合大模型或大规模科学计算。

这和纯 Newton 的对比很直接：

- Newton：先拿到完整 Hessian，再解线性系统
- Hessian-free + CG：不给完整 Hessian，只提供“给我一个向量 $v$，我能算 $Hv$”

在大模型里，后者更现实，因为完整 Hessian 根本放不下，也没必要显式形成。

最后强调适用边界：

- 目标函数高噪声、非凸且参数极大时，优先考虑 SGD 类方法。
- 目标函数光滑、凸、维度中高时，L-BFGS 往往是性价比最高的默认选项。
- 目标函数维度中等、需要高精度、Hessian 易得时，Newton 或信赖域方法更强。
- 问题本质是 SPD 线性系统或二次型最小化时，CG 往往优于普通梯度下降。
- 如果问题含约束、非光滑项或稀疏正则，仅讨论 GD / Newton / CG 往往不够，需要转向投影法、近端法、ADMM 或内点法。

可以把选择规则压缩成一个判断流程：

1. 先看问题是不是“线性系统 / 二次型 / SPD”。如果是，优先考虑 CG。
2. 若不是，再看是否需要高精度且能承受 Hessian。能承受则考虑 Newton 或信赖域。
3. 承受不了 Hessian，但函数光滑且希望比 GD 快，则优先 L-BFGS。
4. 若问题噪声大、数据流式、参数规模极大，则回到 SGD、Momentum、Adam 这类一阶随机方法。

---

## 参考资料

1. Nocedal, J., Wright, S. J., *Numerical Optimization*, 2nd ed., Springer, 2006. 本书是拟牛顿、线搜索、信赖域方法的经典教材，系统覆盖 GD、Newton、BFGS、L-BFGS、CG 及其收敛理论。
2. Boyd, S., Vandenberghe, L., *Convex Optimization*, Cambridge University Press, 2004. 本书适合系统理解凸优化中的一阶与二阶方法，尤其适合建立“条件数、曲率、收敛速度”之间的联系。
3. Shewchuk, J. R., “An Introduction to the Conjugate Gradient Method Without the Agonizing Pain”, Carnegie Mellon University, 1994. 这是解释 CG 最清楚的入门资料之一，对残差、共轭方向、Krylov 子空间和预条件都有直白说明。
4. Nocedal, J., “Updating Quasi-Newton Matrices with Limited Storage”, *Mathematics of Computation*, 1980. 这是 L-BFGS 的经典论文来源之一，用于理解“有限历史近似曲率”的工程动机。
5. Dembo, R. S., Eisenstat, S. C., Steihaug, T., “Inexact Newton Methods”, *SIAM Journal on Numerical Analysis*, 1982. 该文说明了为什么 Newton 子问题不必精确求解，也为 Hessian-free 与截断 CG 的工程实践提供理论背景。
6. Martens, J., “Deep Learning via Hessian-free Optimization”, *ICML 2010*. 该文展示了 Hessian-free + CG 在大规模模型训练中的使用方式，适合理解“只用 Hessian 向量积”的实际意义。
7. Polyak, B. T., *Introduction to Optimization*, Optimization Software, 1987. 本书适合补足一阶法、二阶法和收敛阶之间的基本概念，读起来比很多教材更直接。
8. Wikipedia, “Conjugate gradient method”, https://en.wikipedia.org/wiki/Conjugate_gradient_method ，访问日期：2026-03-08。用于快速核对 CG 标准迭代与误差界公式。
9. Wikipedia, “Broyden-Fletcher-Goldfarb-Shanno algorithm”, https://en.wikipedia.org/wiki/BFGS_method ，访问日期：2026-03-08。用于快速核对 BFGS 更新式与割线条件。
10. TensorTonic, “Newton's Method: Second-Order Optimization”, https://www.tensortonic.com/ml-math/optimization/newtons-method ，访问日期：2026-03-08。可作为 Newton、拟牛顿与工程代价的通俗补充材料。
