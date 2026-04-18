## 核心结论

自动微分是把一个函数拆成一串基础运算，然后在计算函数值的同时，按链式法则传播导数信息。它不是符号求导，也不是数值差分。符号求导试图推导出一个新的解析公式；数值差分用很小的扰动近似斜率；自动微分直接沿着程序实际执行过的计算路径，把每个局部导数组合起来。

设函数为：

$$
f: \mathbb{R}^m \rightarrow \mathbb{R}^n
$$

输入为 $x \in \mathbb{R}^m$，输出为 $y=f(x)\in \mathbb{R}^n$，Jacobian 矩阵为：

$$
J = \frac{\partial y}{\partial x} \in \mathbb{R}^{n \times m}
$$

Jacobian 是“所有输出对所有输入的一阶偏导数组成的矩阵”。如果有 $m$ 个输入、$n$ 个输出，完整 Jacobian 就有 $n \times m$ 个元素：

$$
J =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_m} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_n}{\partial x_1} & \cdots & \frac{\partial y_n}{\partial x_m}
\end{bmatrix}
$$

前向模式计算的是 Jacobian-向量积，即 $Jv$。这里的 $v \in \mathbb{R}^m$ 是输入空间里的一个方向，表示“输入沿这个方向发生微小变化”。输出 $Jv \in \mathbb{R}^n$ 表示所有输出沿这个输入方向的变化率。

反向模式计算的是向量-Jacobian 积，即 $uJ$，也常按列向量习惯写成 $J^T u$。这里的 $u \in \mathbb{R}^n$ 是输出空间里的上游权重，表示“输出端哪些量重要，以及各自权重是多少”。输出 $J^T u \in \mathbb{R}^m$ 表示这个输出加权组合对每个输入的敏感度。

新手版理解：把函数图看成一条流水线。前向模式是“输入变化一路往后传”，看这个变化最后怎么影响输出；反向模式是“输出需要的梯度一路往前分摊”，看输出端的变化应该归因到哪些输入。

| 对比项 | 前向模式 | 反向模式 |
|---|---|---|
| 传播方向 | 从输入到输出 | 从输出到输入 |
| 计算对象 | $Jv$ | $uJ = J^T u$ |
| 种子向量位置 | 输入端 | 输出端 |
| 种子向量维度 | $v\in\mathbb{R}^m$ | $u\in\mathbb{R}^n$ |
| 结果维度 | $Jv\in\mathbb{R}^n$ | $J^T u\in\mathbb{R}^m$ |
| 典型适用场景 | 输入方向少、输出多的敏感性分析 | 标量损失对大量参数求梯度 |
| 完整 Jacobian 代价直觉 | 约需 $O(m)$ 次遍历 | 约需 $O(n)$ 次遍历 |

核心差别不在“能不能求导”，而在“导数信息沿哪个方向传播、适合算什么”。训练神经网络时，损失函数通常只有一个标量输出，所以反向模式很高效；做物理仿真、控制系统或不确定性传播时，如果只关心少数输入扰动如何影响很多输出，前向模式通常更直接。

---

## 问题定义与边界

本文讨论的对象是可微程序里的函数：

$$
x \in \mathbb{R}^m,\quad y=f(x)\in \mathbb{R}^n,\quad J\in \mathbb{R}^{n\times m}
$$

“可微程序”指由可微原语组成的计算过程。原语是自动微分系统能直接知道局部导数的基础操作，例如加法、乘法、除法、`sin`、`exp`、`log`、矩阵乘法、卷积、归一化和激活函数。自动微分系统不需要先把整个程序化成一个大公式；它只需要知道每个原语的局部导数规则。

例如：

$$
a=x_1^2,\quad b=a+x_2,\quad c=x_1x_2
$$

这里有三个基础步骤。自动微分只要知道平方、加法、乘法的导数规则，就能组合出整个函数的导数。

方向导数是“函数沿某个输入方向变化时的变化率”。如果给定方向 $v$，那么 $Jv$ 就表示所有输出沿这个输入方向的变化：

$$
Jv =
\begin{bmatrix}
\nabla y_1(x)^T v \\
\nabla y_2(x)^T v \\
\vdots \\
\nabla y_n(x)^T v
\end{bmatrix}
$$

梯度是标量输出对所有输入的偏导数组成的向量，通常用于 $f:\mathbb{R}^m\rightarrow\mathbb{R}$ 的场景：

$$
\nabla f(x)=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\cdots\\
\frac{\partial f}{\partial x_m}
\end{bmatrix}
$$

当输出是标量时，Jacobian 只有一行。此时反向模式的一次 VJP 就能得到所有输入的梯度：

$$
u=1,\quad uJ=J=\nabla f(x)^T
$$

新手版理解：如果 `f(x)` 的输出是一个数，通常关心的是“这个数对每个输入参数的梯度”；如果输出是很多维，往往更关心“某个输入方向会怎么影响输出”，或者“某些输出组合应该如何回传到输入”。

| 问题类型 | 数学形式 | 常见目标 | 对应计算 |
|---|---|---|---|
| 标量输出 | $f:\mathbb{R}^m\rightarrow\mathbb{R}$ | 求所有输入的梯度 | 反向模式 VJP |
| 向量输出，只关心输入方向 | $f:\mathbb{R}^m\rightarrow\mathbb{R}^n$ | 看方向 $v$ 如何影响输出 | 前向模式 JVP |
| 向量输出，只关心输出加权组合 | $f:\mathbb{R}^m\rightarrow\mathbb{R}^n$ | 将上游向量 $u$ 回传到输入 | 反向模式 VJP |
| 向量输出，全部偏导都要 | $f:\mathbb{R}^m\rightarrow\mathbb{R}^n$ | 得到完整 Jacobian | 多次 JVP 或多次 VJP |

本文边界有三点。

第一，不讨论纯数值差分的截断误差推导，也不展开步长 $h$ 的选择问题。数值差分会在后文作为对照方法出现。

第二，不把所有“求导”都默认成反向传播。反向传播只是反向模式自动微分在神经网络训练中的典型应用，不等于自动微分本身。

第三，不展开高阶导数、稀疏 Jacobian 优化、编译器级自动微分、分布式自动微分和复杂控制流的系统实现。本文重点是前向模式与反向模式的一阶导数机制。

---

## 核心机制与推导

自动微分依赖计算图。计算图是由节点和边组成的有向图，节点表示中间变量或原语操作，边表示数据依赖。对没有循环展开问题的单次执行路径，可以把它理解成一个有向无环图。

例如：

```text
x1 ---- square ----\
                    add ---- y1
x2 ----------------/

x1 ----\
        multiply ---- y2
x2 ----/
```

这个图对应玩具函数：

$$
f(x_1,x_2)=(x_1^2+x_2,\ x_1x_2)
$$

写成中间变量：

$$
a=x_1^2,\quad y_1=a+x_2,\quad y_2=x_1x_2
$$

它的 Jacobian 是：

$$
J =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2}\\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2}
\end{bmatrix}
=
\begin{bmatrix}
2x_1 & 1\\
x_2 & x_1
\end{bmatrix}
$$

前向模式在每个节点上同时维护两个量：`primal` 和 `tangent`。`primal` 是普通函数值；`tangent` 是这个值沿输入方向 $v$ 的变化率。给定种子向量：

$$
v=(\dot{x}_1,\dot{x}_2)\in\mathbb{R}^2
$$

前向模式沿拓扑序传播：

$$
\dot{a}=2x_1\dot{x}_1
$$

$$
\dot{y}_1=\dot{a}+\dot{x}_2=2x_1\dot{x}_1+\dot{x}_2
$$

$$
\dot{y}_2=\dot{x}_1x_2+x_1\dot{x}_2
$$

最终得到：

$$
\dot{y}=
\begin{bmatrix}
\dot{y}_1\\
\dot{y}_2
\end{bmatrix}
=
\begin{bmatrix}
2x_1 & 1\\
x_2 & x_1
\end{bmatrix}
\begin{bmatrix}
\dot{x}_1\\
\dot{x}_2
\end{bmatrix}
=Jv
$$

新手版理解：前向模式像“带着一个小扰动沿路传递”，看这个扰动最后把输出改成多少。它不会一次算出所有方向，只会算出当前这个方向。

反向模式先执行普通前向计算，并记录中间值。这个记录常叫 tape，意思是“计算过程的记录带”。随后它从输出端开始，按逆拓扑序累计 `adjoint`。`adjoint` 是某个中间变量对最终加权输出的贡献，也可理解为“上游梯度”。

给定上游向量：

$$
u=(\bar{y}_1,\bar{y}_2)\in\mathbb{R}^2
$$

反向模式要计算：

$$
uJ=
\begin{bmatrix}
\bar{y}_1 & \bar{y}_2
\end{bmatrix}
\begin{bmatrix}
2x_1 & 1\\
x_2 & x_1
\end{bmatrix}
$$

也就是：

$$
\bar{x}_1=2x_1\bar{y}_1+x_2\bar{y}_2
$$

$$
\bar{x}_2=\bar{y}_1+x_1\bar{y}_2
$$

从计算图看，反向传播规则如下：

$$
y_1=a+x_2
$$

所以：

$$
\bar{a} += \bar{y}_1,\quad \bar{x}_2 += \bar{y}_1
$$

又因为：

$$
a=x_1^2
$$

所以：

$$
\bar{x}_1 += 2x_1\bar{a}
$$

再看：

$$
y_2=x_1x_2
$$

所以：

$$
\bar{x}_1 += x_2\bar{y}_2,\quad \bar{x}_2 += x_1\bar{y}_2
$$

把两条路径累加后，就得到 $uJ$。这里的“累加”很关键：同一个变量可能通过多条路径影响输出，反向模式会把每条路径贡献的梯度加起来。

新手版理解：反向模式像“先算完结果，再倒着问每一步该分担多少责任”。如果一个输入同时影响多个输出，来自多个输出的责任会累加到这个输入上。

| 概念 | 前向模式 | 反向模式 |
|---|---|---|
| 保存的导数信息 | tangent | adjoint |
| 普通值 | primal | primal |
| 初始种子 | 输入方向 `seed vector` $v$ | 输出上游向量 `upstream vector` $u$ |
| 遍历顺序 | 拓扑序 | 先正向记录，再逆拓扑序 |
| 输出结果 | $Jv$ | $uJ$ 或 $J^T u$ |
| 直观问题 | 输入这样变，输出怎么变 | 输出这样加权，输入该承担多少 |

为什么前向对应“输入方向”？因为它一开始就在输入端指定 $v$，之后所有局部导数都只是在回答“这个输入扰动传到这里是多少”。

为什么反向对应“输出权重”？因为它一开始在输出端指定 $u$，之后所有局部导数都只是在回答“这个输出组合应该给每个上游变量分多少权重”。

---

## 代码实现

用玩具例子说明两种模式：

$$
f(x_1,x_2)=(x_1^2+x_2,\ x_1x_2)
$$

在 $x=(2,3)$ 处：

$$
J=
\begin{bmatrix}
4 & 1\\
3 & 2
\end{bmatrix}
$$

给定前向模式种子 $v=(1,2)$：

$$
Jv=
\begin{bmatrix}
4 & 1\\
3 & 2
\end{bmatrix}
\begin{bmatrix}
1\\
2
\end{bmatrix}
=
\begin{bmatrix}
6\\
7
\end{bmatrix}
$$

给定反向模式上游向量 $u=(1,2)$：

$$
uJ=
\begin{bmatrix}
1 & 2
\end{bmatrix}
\begin{bmatrix}
4 & 1\\
3 & 2
\end{bmatrix}
=
\begin{bmatrix}
10 & 5
\end{bmatrix}
$$

下面代码不依赖 PyTorch 或 JAX，只用普通 Python 展示计算路径。它可以直接保存为 `autodiff_demo.py` 后运行。

```python
def f_primal(x1, x2):
    y1 = x1 * x1 + x2
    y2 = x1 * x2
    return y1, y2


def analytic_jacobian(x1, x2):
    return (
        (2 * x1, 1),
        (x2, x1),
    )


def matvec_jvp(J, v):
    return (
        J[0][0] * v[0] + J[0][1] * v[1],
        J[1][0] * v[0] + J[1][1] * v[1],
    )


def vecmat_vjp(u, J):
    return (
        u[0] * J[0][0] + u[1] * J[1][0],
        u[0] * J[0][1] + u[1] * J[1][1],
    )


def forward_mode_jvp(x1, x2, v1, v2):
    # 输入节点：primal 是普通值，dot 是沿输入方向 v 的变化率。
    x1_dot = v1
    x2_dot = v2

    # a = x1 * x1
    a = x1 * x1
    a_dot = x1_dot * x1 + x1 * x1_dot

    # y1 = a + x2
    y1 = a + x2
    y1_dot = a_dot + x2_dot

    # y2 = x1 * x2
    y2 = x1 * x2
    y2_dot = x1_dot * x2 + x1 * x2_dot

    return (y1, y2), (y1_dot, y2_dot)


def reverse_mode_vjp(x1, x2, u1, u2):
    # forward pass：计算普通值，并保留反向传播需要的中间值。
    a = x1 * x1
    y1 = a + x2
    y2 = x1 * x2

    # reverse pass：从输出端给定上游权重。
    y1_bar = u1
    y2_bar = u2

    # 初始化所有中间变量的 adjoint。
    x1_bar = 0
    x2_bar = 0
    a_bar = 0

    # y2 = x1 * x2
    x1_bar += y2_bar * x2
    x2_bar += y2_bar * x1

    # y1 = a + x2
    a_bar += y1_bar
    x2_bar += y1_bar

    # a = x1 * x1
    x1_bar += a_bar * 2 * x1

    return (y1, y2), (x1_bar, x2_bar)


def finite_difference_jvp(x1, x2, v1, v2, eps=1e-6):
    y = f_primal(x1, x2)
    y_eps = f_primal(x1 + eps * v1, x2 + eps * v2)

    return (
        (y_eps[0] - y[0]) / eps,
        (y_eps[1] - y[1]) / eps,
    )


def almost_equal_pair(a, b, tolerance=1e-5):
    return abs(a[0] - b[0]) < tolerance and abs(a[1] - b[1]) < tolerance


if __name__ == "__main__":
    x = (2.0, 3.0)
    v = (1.0, 2.0)
    u = (1.0, 2.0)

    J = analytic_jacobian(*x)

    value, jvp = forward_mode_jvp(*x, *v)
    expected_jvp = matvec_jvp(J, v)
    numerical_jvp = finite_difference_jvp(*x, *v)

    assert value == (7.0, 6.0)
    assert jvp == expected_jvp
    assert almost_equal_pair(jvp, numerical_jvp)

    value, vjp = reverse_mode_vjp(*x, *u)
    expected_vjp = vecmat_vjp(u, J)

    assert value == (7.0, 6.0)
    assert vjp == expected_vjp

    print("value:", value)
    print("J:", J)
    print("forward-mode JVP:", jvp)
    print("finite-difference check:", numerical_jvp)
    print("reverse-mode VJP:", vjp)
```

运行结果应接近：

```text
value: (7.0, 6.0)
J: ((4.0, 1), (3.0, 2.0))
forward-mode JVP: (6.0, 7.0)
finite-difference check: (6.000001000927568, 7.000000000090267)
reverse-mode VJP: (10.0, 5.0)
```

数值差分检查里出现很小误差是正常的。它用的是近似：

$$
Jv \approx \frac{f(x+\epsilon v)-f(x)}{\epsilon}
$$

自动微分不是用这个公式算导数；这里的数值差分只用于验证结果是否合理。

| 步骤 | 前向模式保存什么 | 反向模式保存什么 | 内存直觉 |
|---|---|---|---|
| 输入节点 | primal + tangent | primal | 前向多存一份方向导数 |
| 中间节点 | 当前节点的 primal + tangent | 反向所需中间值 | 反向需要保留 tape |
| 输出节点 | 直接得到 $Jv$ | 给定上游向量后回传 | 反向多一次逆序遍历 |
| 深计算图 | 可边算边传 | 需要缓存或重算 | 反向内存压力更明显 |

真实工程例子是神经网络训练。模型可能有上亿个参数，输入维度巨大，但损失函数 $L$ 通常是一个标量：

$$
L:\mathbb{R}^m\rightarrow\mathbb{R}
$$

此时 $n=1$。反向模式只需一次前向计算和一次反向遍历，就能得到所有参数的梯度 $\nabla L$。如果用前向模式逐个参数方向计算，就需要接近 $m$ 次遍历，通常不可接受。

但反过来，如果函数是：

$$
f:\mathbb{R}^2\rightarrow\mathbb{R}^{10000}
$$

并且只关心一个输入扰动方向 $v$，前向模式一次 JVP 就能得到 $10000$ 个输出在该方向上的变化。此时没有必要构造完整 Jacobian。

---

## 工程权衡与常见坑

完整 Jacobian 的代价可以用遍历次数理解：

$$
\text{前向模式求完整 }J \approx O(m)\text{ 次遍历}
$$

$$
\text{反向模式求完整 }J \approx O(n)\text{ 次遍历}
$$

这里的 $m$ 是输入维度，$n$ 是输出维度。原因是前向模式一次给一个输入方向，得到完整 Jacobian 的一列；反向模式一次给一个输出权重，得到完整 Jacobian 的一行。

更精确地说，若 $e_i$ 是输入空间的标准基向量：

$$
Je_i
$$

就是 Jacobian 的第 $i$ 列。对 $i=1,\dots,m$ 做 $m$ 次 JVP，就能拼出完整 Jacobian。

若 $r_j$ 是输出空间的标准基行向量：

$$
r_jJ
$$

就是 Jacobian 的第 $j$ 行。对 $j=1,\dots,n$ 做 $n$ 次 VJP，就能拼出完整 Jacobian。

但工程决策不能只看“输入多还是输出多”。关键问题是：你要完整 Jacobian，还是只要一个方向上的导数？

| 典型场景 | 推荐模式 | 原因 |
|---|---|---|
| 神经网络标量损失训练 | 反向模式 | 一次回传得到所有参数梯度 |
| 少数输入扰动影响大量输出 | 前向模式 | 直接计算目标方向的 $Jv$ |
| 需要完整 Jacobian 且 $m \ll n$ | 前向模式 | 列数少，前向次数少 |
| 需要完整 Jacobian 且 $n \ll m$ | 反向模式 | 行数少，反向次数少 |
| 需要 Hessian-vector product | 前向和反向组合 | 常用 forward-over-reverse 或 reverse-over-forward |
| 深层网络且显存紧张 | 反向模式 + checkpointing | 用重计算换内存 |

Checkpointing 是“只保存部分中间值，反向时重新计算缺失部分”的技术。它降低内存占用，但会增加计算时间。对深层网络来说，反向模式的主要成本往往不是导数公式本身，而是为了反向传播保存激活值带来的显存压力。

常见坑包括以下几类。

第一，把 `grad` 当成万能入口。很多框架里的 `grad` 默认更适合标量输出。如果函数输出是向量，要先明确自己要的是 $Jv$、$uJ$，还是完整 Jacobian。对向量输出直接调用 `grad`，通常需要额外指定上游梯度，或者先把输出聚合成标量。

第二，混淆 $Jv$ 和 $uJ$。$Jv$ 的输入是输入方向，输出仍在输出空间；$uJ$ 的输入是输出权重，输出回到输入空间。两者维度不同，语义也不同。

| 表达式 | 输入 | 输出 | 语义 |
|---|---|---|---|
| $Jv$ | 输入方向 $v\in\mathbb{R}^m$ | 输出变化 $\in\mathbb{R}^n$ | 输入扰动如何影响输出 |
| $uJ$ | 输出权重 $u\in\mathbb{R}^n$ | 输入梯度 $\in\mathbb{R}^m$ | 输出组合如何归因到输入 |
| $J$ | 无方向压缩 | 矩阵 $\in\mathbb{R}^{n\times m}$ | 所有输出对所有输入的偏导 |

第三，忽略反向模式的内存占用。反向模式通常要缓存前向中间值。图越深、张量越大，tape 的内存压力越明显。真实训练中经常需要 activation checkpointing、梯度累积、混合精度、张量并行或重计算策略。

第四，以“输入多/输出少”机械决策。如果只需要一个 $Jv$，前向模式就是一次传播；如果只需要一个 $uJ$，反向模式也是一次传播。真正的复杂度差异通常出现在“要不要完整 Jacobian”。

第五，把自动微分当成数学上永远存在的导数。程序里可能有不可微点，例如 `abs(x)` 在 $x=0$、`relu(x)` 在 $x=0$、`max(a,b)` 在 $a=b$。框架通常会选择一个约定的次梯度或局部规则，但这不等于函数在经典意义下处处可微。

第六，忽略浮点数和原语稳定性。自动微分会忠实地沿程序执行路径传播导数。如果原函数里有数值不稳定写法，例如直接计算 `log(exp(a) + exp(b))`，导数也会受到影响。工程中通常要用 `logsumexp` 这类稳定原语。

新手版理解：训练神经网络时，损失函数只有一个输出，所以反向模式一次就能把所有参数的梯度算出来；但如果做物理仿真，只想看少数输入扰动怎么影响很多输出，前向模式更合适。选择模式前，先问清楚自己要的是一个方向、一个输出组合，还是完整 Jacobian。

---

## 替代方案与适用边界

自动微分不是唯一求导方案。常见替代方案有数值差分、符号求导和手工推导。

数值差分用函数值近似导数。一维情况下：

$$
f'(x)\approx \frac{f(x+h)-f(x)}{h}
$$

更常用的中心差分是：

$$
f'(x)\approx \frac{f(x+h)-f(x-h)}{2h}
$$

它适合快速验证公式，但结果受步长 $h$ 影响。$h$ 太大，近似误差大；$h$ 太小，浮点舍入误差会变明显。高维输入下，如果要估计完整梯度，数值差分通常需要对每个输入维度分别扰动，成本会随输入维度增长。

符号求导直接对表达式做代数变换。它适合简单公式，例如多项式、三角函数组合。但真实程序包含分支、循环、数组操作和复杂控制流时，表达式可能迅速膨胀。符号结果也未必是数值上最稳定的实现。

手工推导适合关键小模块，例如自定义 CUDA kernel、特殊损失函数、隐式层、数值稳定性要求很高的算子。但它维护成本高，代码改了公式也要同步改；推导错误和实现错误都需要额外测试覆盖。

自动微分适合真实程序里的可微计算过程。它的优势是精度接近解析导数，且能直接作用在程序执行路径上。它的限制是依赖原语规则、控制流记录、内存管理和框架实现。自动微分不会自动修复不稳定的数学表达式，也不会把不可微程序变成处处可微程序。

| 方法 | 精度 | 性能 | 适用对象 | 工程复杂度 |
|---|---|---|---|---|
| 自动微分 | 接近解析导数，受浮点计算影响 | 通常高效 | 真实代码里的可微计算过程 | 中等，依赖框架或实现 |
| 数值差分 | 受步长和舍入误差影响 | 输入维度高时昂贵 | 快速校验、黑箱函数 | 低 |
| 符号求导 | 理论上精确 | 表达式膨胀后可能很差 | 简单闭式表达式 | 中到高 |
| 手工推导 | 取决于推导质量 | 可做到很高 | 小而关键的核心算子 | 高 |

| 场景边界 | 优先方案 | 理由 |
|---|---|---|
| 标量损失训练 | 反向模式自动微分 | 一次得到大量参数梯度 |
| 向量输出敏感性分析 | 前向模式自动微分 | 少数输入方向直接对应 $Jv$ |
| 高内存深图 | 反向模式 + checkpointing | 用重计算缓解缓存压力 |
| 高阶导数需求 | 框架组合 JVP/VJP 或专门工具 | 需要关注数值稳定性和性能 |
| 只想验证某个梯度实现 | 数值差分 | 简单直接，适合测试 |
| 简单数学表达式推导 | 符号求导 | 公式清晰时更直观 |
| 自定义高性能算子 | 手工推导 + 自动微分接口 | 兼顾性能和框架集成 |

如果你的目标是训练一个标量损失模型，优先选反向模式自动微分；如果你的目标是研究某几个输入扰动对大量输出的影响，优先选前向模式；如果你的目标只是验证梯度有没有写错，优先用数值差分做测试；如果你的目标是给简单公式写教材推导，符号求导更清晰。

一个实用判断流程是：

| 问题 | 判断 |
|---|---|
| 输出是不是标量损失 | 是，优先反向模式 |
| 是否只关心少数输入方向 | 是，优先前向模式 |
| 是否只关心少数输出加权组合 | 是，优先反向模式 |
| 是否必须显式拿到完整 Jacobian | 是，再比较 $m$ 和 $n$ |
| 是否只是测试梯度实现 | 是，用数值差分做校验 |
| 是否遇到显存瓶颈 | 考虑 checkpointing、重计算或更小 batch |

---

## 参考资料

先看 JAX 理解 $Jv$ / $uJ$ 的定义，再看 PyTorch 理解工程上的接口差异，最后用综述文章补全算法背景。

| 资料名称 | 适合解决的问题 | 先读理由 |
|---|---|---|
| [JAX: The Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) | 理解 JVP 与 VJP 的数学定义 | 用于本文“核心结论”和“核心机制与推导” |
| [PyTorch: Forward-mode Automatic Differentiation](https://docs.pytorch.org/tutorials/intermediate/forward_ad_usage.html) | 理解前向模式在工程 API 中如何暴露 | 用于本文“代码实现”和“工程权衡” |
| [PyTorch docs: `torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html) | 理解反向自动微分、计算图和梯度接口 | 用于本文“反向模式”和“常见坑” |
| [JMLR: Automatic Differentiation in Machine Learning: a Survey](https://www.jmlr.org/papers/v18/17-468.html) | 系统理解自动微分在机器学习中的位置 | 用于本文“替代方案与适用边界” |

建议阅读顺序是：先读 JAX 文档建立 JVP/VJP 概念，再读 PyTorch 文档看实际接口，最后读 JMLR 综述补齐前向模式、反向模式、计算图和机器学习训练之间的关系。

| 阅读目标 | 推荐入口 |
|---|---|
| 只想分清 JVP 和 VJP | JAX Autodiff Cookbook |
| 想在 PyTorch 里写前向模式例子 | PyTorch Forward-mode AD tutorial |
| 想理解 `backward`、`grad`、计算图 | PyTorch `torch.autograd` 文档 |
| 想系统理解自动微分与机器学习关系 | JMLR survey |
