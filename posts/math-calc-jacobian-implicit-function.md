## 核心结论

雅可比矩阵（Jacobian matrix，白话说就是“把多输入函数在某一点附近拉直后得到的斜率表”）是多元函数的一阶线性近似工具。若 $f:\mathbb{R}^n\to\mathbb{R}^m$，则在点 $x$ 附近有

$$
dy \approx J_f(x)\,dx,\qquad (J_f)_{ij}=\frac{\partial f_i}{\partial x_j}.
$$

它回答两个核心问题：

1. 输入发生一个很小的变化 $dx$ 时，输出会沿哪些方向变化、变化多大。
2. 当变换可逆时，局部面积或体积会被放大多少，这个缩放因子由 $|\det J|$ 给出。

直观玩具例子：二维变换
$$
f(x,y)=(2x,\;0.75y)
$$
的雅可比矩阵是
$$
J=\begin{bmatrix}
2 & 0\\
0 & 0.75
\end{bmatrix},
\qquad \det J = 1.5.
$$
这意味着单位小正方形经过变换后，面积变成原来的 $1.5$ 倍。这里“行列式”可以理解为“局部面积缩放倍数”。

隐函数定理（Implicit Function Theorem，白话说就是“虽然方程没把 $y$ 明确写成 $x$ 的函数，但局部上仍然能解出来”）把约束方程 $F(x,y)=0$ 转成局部显式函数 $y=g(x)$。如果某点附近 $\partial F/\partial y$ 可逆，那么局部存在这样的 $g$，并且

$$
\frac{\partial g}{\partial x}
=
-\left(\frac{\partial F}{\partial y}\right)^{-1}\frac{\partial F}{\partial x}.
$$

这条公式说明：隐式约束下的导数，本质上仍然由 Jacobian 控制。

在工程里，Jacobian 不是抽象符号。深度学习里的 JVP（Jacobian-vector product，白话说就是“不显式构造整个 Jacobian，直接算 $Jv$”）和 VJP（vector-Jacobian product，白话说就是“直接算 $v^\top J$”）分别对应前向模式和反向模式自动微分。Flow 模型依赖 $\log|\det J|$ 做密度变换，梯度惩罚依赖输入梯度或 Jacobian 范数控制模型稳定性。

---

## 问题定义与边界

本文讨论两个问题。

第一，Jacobian 如何描述多元函数 $f:\mathbb{R}^n\to\mathbb{R}^m$ 的局部线性行为，以及为什么变量替换时会出现 $|\det J|$。

第二，隐函数定理在什么条件下成立，以及它只能保证“局部可解”，不能保证“全局都能解”。

先把对象分清楚：

| 对象 | 数学形式 | Jacobian 维度 | 可逆性要求 | 影响 |
|---|---|---:|---|---|
| 标量函数 | $f:\mathbb{R}^n\to\mathbb{R}$ | $1\times n$ | 通常不谈整体可逆 | 给出梯度方向 |
| 向量函数 | $f:\mathbb{R}^n\to\mathbb{R}^m$ | $m\times n$ | 只有 $m=n$ 时才能谈 $\det J$ | 描述局部线性映射 |
| 可逆变量变换 | $g:\mathbb{R}^n\to\mathbb{R}^n$ | $n\times n$ | 需要局部可逆，通常要求 $\det J\neq 0$ | 决定积分和密度缩放 |
| 隐式约束 | $F(x,y)=0$ | 按变量分块 | 关键是 $\partial F/\partial y$ 可逆 | 才能局部解出 $y=g(x)$ |

边界也要说清楚。

1. Jacobian 是局部一阶近似，不是远距离行为描述。点附近好用，不代表全局都准确。
2. $\det J$ 只有在方阵 Jacobian 时才有定义，所以只有输入输出维度相同时，才直接谈“面积/体积缩放因子”。
3. 隐函数定理只保证局部存在显式函数。即使某点可以解出 $y=g(x)$，换个区域可能就不行。
4. $\partial F/\partial y$ 可逆是硬条件。若它奇异，定理可能失效。

新手最常见的隐函数例子是单位圆：
$$
F(x,y)=x^2+y^2-1=0.
$$
若 $y\neq 0$，则
$$
\frac{\partial F}{\partial y}=2y\neq 0,
$$
所以在这些点附近可以把 $y$ 看成 $x$ 的函数，得到局部解
$$
y=\pm\sqrt{1-x^2}.
$$
但在 $(1,0)$ 和 $(-1,0)$ 处，$\partial F/\partial y=0$，此时不能把圆局部写成 $y=g(x)$。这不是公式失灵，而是曲线在这里“竖起来了”。

---

## 核心机制与推导

Jacobian 的定义来自多元泰勒展开。若 $f$ 在点 $x$ 可微，则

$$
f(x+\Delta x) \approx f(x) + J_f(x)\Delta x.
$$

把微小变化写成微分记号，就是

$$
dy \approx J\,dx.
$$

这里的意思不是“输出等于 Jacobian”，而是“Jacobian 把输入的小变化线性映射到输出的小变化”。

### 玩具例子：二维小正方形如何被拉伸

设
$$
f(x,y)=(2x+0.5y,\;y).
$$
其 Jacobian 为
$$
J=
\begin{bmatrix}
2 & 0.5\\
0 & 1
\end{bmatrix}.
$$

第一列 $\begin{bmatrix}2\\0\end{bmatrix}$ 表示沿 $x$ 方向走一步，输出大致朝这个方向变化；第二列 $\begin{bmatrix}0.5\\1\end{bmatrix}$ 表示沿 $y$ 方向走一步的输出变化。单位小正方形在输出空间会变成由这两列向量张成的平行四边形，其面积就是

$$
|\det J| = \left|2\cdot 1 - 0.5\cdot 0\right| = 2.
$$

所以它把面积放大为原来的 2 倍。若 $\det J<0$，还会带来局部翻转；做积分和概率密度变换时必须取绝对值。

### 变量替换为什么会出现 $|\det J|$

设 $y=g(x)$ 是可逆可微变换。局部上，一个很小的体积元 $dx$ 经过映射后变成 $dy$，缩放因子是 $|\det J_g(x)|$，因此

$$
dy = |\det J_g(x)|\,dx.
$$

换元公式可写为

$$
\int f(g(x))\,dx
=
\int f(y)\,\left|\det Dg^{-1}(y)\right|\,dy.
$$

极坐标是最标准的例子：
$$
x=r\cos\theta,\qquad y=r\sin\theta.
$$
对应 Jacobian 为
$$
J=
\begin{bmatrix}
\cos\theta & -r\sin\theta\\
\sin\theta & r\cos\theta
\end{bmatrix},
\qquad \det J = r.
$$
所以面积元不是 $dr\,d\theta$，而是
$$
dx\,dy = r\,dr\,d\theta.
$$
这个额外的 $r$ 不是技巧记忆，而是 Jacobian 行列式。

### 隐函数定理的推导

设 $F(x,y)=0$，并且局部上 $y=g(x)$。把它代回去：

$$
F(x,g(x))=0.
$$

对 $x$ 求导，用链式法则：

$$
\frac{d}{dx}F(x,g(x))
=
\frac{\partial F}{\partial x}
+
\frac{\partial F}{\partial y}\frac{dg}{dx}
=0.
$$

移项得到

$$
\frac{dg}{dx}
=
-\left(\frac{\partial F}{\partial y}\right)^{-1}\frac{\partial F}{\partial x}.
$$

这里的关键点是：$\partial F/\partial y$ 必须可逆，否则无法把 $\frac{dg}{dx}$ 单独解出来。换句话说，隐函数定理不是凭空生成函数，而是依赖 Jacobian 的局部可逆性把微分信息“解出来”。

### 详细推导例子：$F(x,y)=xy-1$

令
$$
F(x,y)=xy-1=0.
$$
这条曲线的显式解是 $y=1/x$，但我们先不用它，直接走隐函数公式。

偏导为
$$
\frac{\partial F}{\partial x}=y,\qquad \frac{\partial F}{\partial y}=x.
$$

当 $x\neq 0$ 时，$\partial F/\partial y=x\neq 0$，因此局部可解出 $y=g(x)$，并且

$$
\frac{dy}{dx}
=
-\frac{\partial F/\partial x}{\partial F/\partial y}
=
-\frac{y}{x}.
$$

再代入约束 $y=1/x$，得到
$$
\frac{dy}{dx}=-\frac{1}{x^2},
$$
和显式求导一致。

### 真实工程例子：Flow 模型中的密度变换

Normalizing Flow（归一化流，白话说就是“用一串可逆变换把简单分布变成复杂分布”）要计算样本密度。若 $x=f(z)$ 且 $z=f^{-1}(x)$，那么

$$
\log p_X(x)
=
\log p_Z(z)
+
\log\left|\det \frac{\partial f^{-1}(x)}{\partial x}\right|.
$$

如果每层都显式算一个高维 Jacobian 的行列式，成本会很高，所以工程实现通常把 Jacobian 设计成三角或分块三角结构。这样 $\det J$ 直接等于对角线乘积，$\log|\det J|$ 就是对角线对数之和。

---

## 代码实现

下面先用纯 Python 写一个可运行的玩具例子，验证 Jacobian 的线性近似和隐函数导数。

```python
import math

def f(x, y):
    return (2 * x + 0.5 * y, y)

def jacobian_f(x, y):
    # 对 f(x, y) = (2x + 0.5y, y)
    return ((2.0, 0.5),
            (0.0, 1.0))

# 检查面积缩放：det J = 2
J = jacobian_f(1.0, 2.0)
det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
assert abs(det - 2.0) < 1e-12

# 用有限差分验证 dy ≈ J dx
x, y = 1.0, 2.0
dx = (1e-6, -2e-6)

fx1 = f(x, y)
fx2 = f(x + dx[0], y + dx[1])
actual = (fx2[0] - fx1[0], fx2[1] - fx1[1])

linear = (
    J[0][0] * dx[0] + J[0][1] * dx[1],
    J[1][0] * dx[0] + J[1][1] * dx[1],
)

assert abs(actual[0] - linear[0]) < 1e-12
assert abs(actual[1] - linear[1]) < 1e-12

# 隐函数例子：F(x, y) = xy - 1 = 0, y = 1/x
def implicit_dydx(x):
    y = 1.0 / x
    return -y / x

assert abs(implicit_dydx(2.0) + 0.25) < 1e-12
print("all checks passed")
```

上面这段代码做了两件事：

1. 验证 $dy\approx Jdx$ 在这个线性函数上是精确成立的。
2. 验证隐函数公式 $\frac{dy}{dx}=-\frac{y}{x}$ 与显式解一致。

如果进入自动微分场景，常见写法是用 PyTorch 计算 JVP、VJP 和 Jacobian。下面是最小示意：

```python
import torch
from torch.autograd.functional import jvp, vjp, jacobian

def f(inp):
    x0, x1 = inp[0], inp[1]
    return torch.stack([x0 * x0 + x1, x0 * x1])

x = torch.tensor([2.0, 3.0], requires_grad=True)
v = torch.tensor([1.0, -1.0])

# JVP: 直接算 J(x) @ v
y, jvp_val = jvp(f, (x,), (v,))
assert y.shape == (2,)
assert jvp_val.shape == (2,)

# VJP: 直接算 v_out^T @ J(x)
v_out = torch.tensor([1.0, 2.0])
y2, vjp_fn = vjp(f, x)
vjp_val = vjp_fn(v_out)[0]
assert y2.shape == (2,)
assert vjp_val.shape == (2,)

# 全量 Jacobian
J = jacobian(f, x)
assert J.shape == (2, 2)

print("y =", y)
print("JVP =", jvp_val)
print("VJP =", vjp_val)
print("Jacobian =\n", J)
```

这里要注意张量形状：

- 输入 `x.shape == (2,)`，表示 $\mathbb{R}^2$ 中一个点。
- 输出 `f(x).shape == (2,)`，表示 $\mathbb{R}^2\to\mathbb{R}^2$。
- 因此 `jacobian(f, x).shape == (2, 2)`，第 0 维对应输出分量，第 1 维对应输入分量。

真实工程例子一：WGAN-GP 里的梯度惩罚。判别器 $D(x)$ 对输入的梯度范数要接近 1，常见目标是

$$
\lambda \,\mathbb{E}\left(\|\nabla_x D(x)\|_2 - 1\right)^2.
$$

这里虽然很多实现直接用 `torch.autograd.grad`，但本质上也是在取输入 Jacobian，因为标量函数的 Jacobian 就是梯度行向量。

真实工程例子二：Flow 模型。如果某层是仿射耦合层，Jacobain 常被设计为下三角，那么

$$
\log|\det J| = \sum_i \log|J_{ii}|.
$$

这样就避免了通用行列式的高成本。

---

## 工程权衡与常见坑

理论上 Jacobian 很清楚，工程上真正难的是成本和稳定性。

| 问题 | 后果 | 规避方式 |
|---|---|---|
| 高维全量 Jacobian | 显存和时间迅速爆炸 | 优先用 JVP/VJP，不显式构造整个矩阵 |
| 直接算通用行列式 | 常见为 $O(d^3)$，高维难用 | 设计三角、分块三角、耦合结构 |
| 忽略 $|\det J|$ 绝对值 | 积分或密度方向错误 | 始终检查是否需要绝对值 |
| 把局部定理当全局结论 | 推出错误的全局显式解 | 明确“只在某点附近成立” |
| $\partial F/\partial y$ 奇异 | 隐函数公式无法使用 | 换参数化、换变量、做数值求解 |
| 数值差分步长随便选 | 截断误差或舍入误差大 | 用自动微分，或调试步长敏感性 |

最常见的误区有三个。

第一，把 Jacobian 当作“全局映射矩阵”。这不对。非线性函数只有在一点附近才能被 Jacobian 近似。离开该点后，Jacobian 会变化。

第二，把“可微”直接等同于“可逆”。也不对。局部可逆通常还需要 $\det J\neq 0$。例如二维到一维的函数根本没有方阵 Jacobian，自然也不能谈普通意义下的局部可逆变换。

第三，在隐函数问题里忘了检查条件。比如
$$
F(x,y)=y^3-x=0.
$$
在 $(0,0)$ 处，
$$
\frac{\partial F}{\partial y}=3y^2=0.
$$
此时隐函数定理不能直接用于“把 $y$ 写成 $x$ 的光滑函数”。虽然形式上 $y=x^{1/3}$ 仍然存在，但其导数在 0 附近不再良好，这正是条件失败带来的结果。

对于 Flow 模型，核心权衡是：你想要足够强的表达能力，但又必须让 $\log|\det J|$ 便宜。工程上常见做法是牺牲一部分变换自由度，换取结构化 Jacobian。比如下三角 Jacobian 不需要显式求行列式，只需求对角线和：

$$
\log|\det J|=\sum_i \log|J_{ii}|.
$$

这类结构不是数学上的“近似技巧”，而是模型设计约束。

---

## 替代方案与适用边界

如果目标只是“得到导数或 Jacobian”，不只有解析推导这一条路。

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 解析推导 | 精确、可解释 | 人工推导复杂，易出错 | 低维公式分析、教学 |
| 数值差分 | 实现最简单 | 有截断误差和舍入误差 | 快速验证、小规模原型 |
| 自动微分 | 精度高、适合复杂程序 | 需要框架支持，图构建有开销 | 深度学习、科学计算 |
| 符号推导 | 可输出闭式表达 | 容易爆炸，不适合大程序 | 公式生成、代数分析 |

新手最容易上手的是中心差分。对一元函数，
$$
f'(x)\approx \frac{f(x+h)-f(x-h)}{2h}.
$$

它可以视为 Jacobian 的数值近似版。比如对 $f(x)=x^2$，在 $x=3$ 处：

```python
def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x * x

approx = central_diff(f, 3.0)
exact = 6.0
assert abs(approx - exact) < 1e-3
print(approx)
```

误差来源有两个：

1. $h$ 太大，泰勒展开截断误差大。
2. $h$ 太小，浮点数舍入误差放大。

所以数值差分适合做检查，不适合替代大规模训练中的自动微分。

对隐函数问题，如果 $\partial F/\partial y$ 不可逆，通常有三种处理思路：

1. 换一组变量，把别的量写成显式函数。
2. 不追求显式函数，只做局部数值求解，例如牛顿法。
3. 把问题从“求函数”改成“求曲线或流形上的点”。

这说明隐函数定理不是“所有约束都能解成函数”的万能工具。它适用于局部、光滑、可逆块 Jacobian 存在的情形。

---

## 参考资料

1. https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant  
用途：定义 Jacobian、行列式与变量替换的理论背景。

2. https://en.wikipedia.org/wiki/Implicit_function_theorem  
用途：隐函数定理的条件、结论与导数公式。

3. https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian  
用途：PyTorch `jacobian` API 参考，适合查输入输出形状与参数说明。

4. https://docs.pytorch.org/docs/2.9/generated/torch.autograd.functional.jvp.html  
用途：PyTorch `jvp` 接口说明，理解前向模式自动微分。

5. https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.vjp.html  
用途：PyTorch `vjp` 接口说明，理解反向模式中的向量-Jacobian 积。

6. https://changezakram.github.io/Deep-Generative-Models/flows.html  
用途：Flow 模型中的可逆变换与 $\log|\det J|$ 工程解释。

7. https://oboe.com/learn/openstax-calculus-multiple-integration-vr3iig/variable-changes-and-jacobians-5  
用途：多元积分中的变量替换与极坐标 Jacobian 直观例子。
