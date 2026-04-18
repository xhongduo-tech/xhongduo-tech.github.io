## 核心结论

雅可比矩阵是向量值函数的一阶导数矩阵。向量值函数，指的是输入是一个向量、输出也是一个向量的函数。设 $f:\mathbb{R}^n\to\mathbb{R}^m$，它把 $n$ 维输入变成 $m$ 维输出，那么 $J_f(x)$ 就描述了输入 $x$ 的每个分量发生微小变化时，输出 $f(x)$ 的每个分量会怎么变。

核心公式是：

$$
J_h(x) = J_g(f(x)) J_f(x)
$$

其中 $h=g\circ f$，也就是先算 $f$，再算 $g$。这个公式的含义是：复合函数的雅可比矩阵，等于后一个函数在中间结果处的雅可比，乘以前一个函数在输入处的雅可比。

新手可以先把两个函数理解成两台机器。第一台机器 $f$ 先把输入变形，第二台机器 $g$ 再处理第一台机器的输出。你不需要一次性看清整条流水线的所有细节，只要知道每台机器在当前位置附近会怎么放大、缩小、旋转输入的微小变化，最后把这些局部变化按顺序乘起来，就是整体导数。

在反向传播里，常见目标不是求完整输出的变化，而是求一个标量损失对输入或参数的梯度。标量损失，指输出是一个数字的函数，比如训练误差 $L$。这时使用的是：

$$
\nabla_x L(f(x)) = J_f(x)^T \nabla_y L(y)\big|_{y=f(x)}
$$

也就是说，反向传播把上游梯度乘上局部雅可比的转置，然后继续往前一层传。

| 对照项 | 前向计算 | 反向传播 |
|---|---|---|
| 方向 | 从输入到输出 | 从损失到输入或参数 |
| 传递对象 | 激活值、特征、状态 | 梯度 |
| 局部操作 | $y=f(x)$ | $\nabla_x = J_f(x)^T\nabla_y$ |
| 链式法则形式 | $J_g(f(x))J_f(x)$ | 逐层使用局部雅可比转置 |
| 工程含义 | 构建计算图并保存必要中间量 | 利用中间量计算局部梯度并回传 |

结论先记住两句：向量输出看雅可比；标量损失反传看雅可比转置乘上游梯度。

---

## 问题定义与边界

本文固定使用以下约定：如果 $f:\mathbb{R}^n\to\mathbb{R}^m$，输入 $x$ 有 $n$ 个分量，输出 $f(x)$ 有 $m$ 个分量，那么雅可比矩阵 $J_f(x)$ 的形状是 $m\times n$，即“输出维 × 输入维”。

定义为：

$$
J_f(x)_{ij} = \frac{\partial f_i}{\partial x_j}
$$

这里 $f_i$ 是第 $i$ 个输出分量，$x_j$ 是第 $j$ 个输入分量。偏导数，指在其他输入分量固定时，只看某一个输入分量变化对输出的影响。

例如输入是 2 维、输出是 3 维，即 $f:\mathbb{R}^2\to\mathbb{R}^3$，那么 $J_f$ 就是 3 行 2 列。每一行对应一个输出分量，每一列对应一个输入分量。

| 函数类型 | 输出类型 | 导数对象 | 形状 |
|---|---|---|---|
| $f:\mathbb{R}\to\mathbb{R}$ | 标量 | 普通导数 | $1\times 1$ |
| $f:\mathbb{R}^n\to\mathbb{R}$ | 标量 | 梯度 $\nabla f$ | 常写成 $n$ 维向量 |
| $f:\mathbb{R}\to\mathbb{R}^m$ | 向量 | 每个输出对单个输入的导数 | $m\times 1$ |
| $f:\mathbb{R}^n\to\mathbb{R}^m$ | 向量 | 雅可比矩阵 $J_f$ | $m\times n$ |

梯度是标量函数对向量输入的导数。比如 $L:\mathbb{R}^n\to\mathbb{R}$，它的梯度 $\nabla_x L$ 表示每个输入分量对这个标量输出的影响。雅可比矩阵则处理向量输出的情况。

本文只讨论实值函数、有限维空间、局部可微的情况。实值函数，指输入和输出都由实数组成。有限维空间，指输入输出可以用有限个数字表示。局部可微，指函数在某个点附近可以用一阶线性近似描述。

本文不展开一般流形、无限维函数空间、弱导数等更抽象的分析问题。工程里大多数神经网络、数值优化、自动微分问题，都可以先在这个有限维可微框架里理解。

---

## 核心机制与推导

链式法则的本质是局部线性近似的连续组合。局部线性近似，指函数在某个点附近虽然可能是非线性的，但足够小的扰动可以近似看成经过一个线性变换。

设：

$$
f:\mathbb{R}^n\to\mathbb{R}^m,\quad g:\mathbb{R}^m\to\mathbb{R}^k,\quad h=g\circ f
$$

也就是：

$$
y=f(x),\quad z=g(y),\quad h(x)=g(f(x))
$$

给输入一个很小的扰动 $\delta x$。扰动，指输入上的微小变化量。第一层 $f$ 会把这个扰动变成输出空间里的扰动：

$$
\delta y \approx J_f(x)\delta x
$$

第二层 $g$ 接收到 $y$ 的扰动后，再把它变成 $z$ 的扰动：

$$
\delta z \approx J_g(y)\delta y
$$

把第一步代入第二步：

$$
\delta z \approx J_g(f(x))J_f(x)\delta x
$$

因为整体函数 $h$ 也应该满足：

$$
\delta z \approx J_h(x)\delta x
$$

所以得到：

$$
J_h(x)=J_g(f(x))J_f(x)
$$

这个推导说明，矩阵乘法顺序不是记忆规则，而是由扰动流动方向决定的。$\delta x$ 先被 $J_f(x)$ 处理，再被 $J_g(f(x))$ 处理，所以整体矩阵必须是 $J_g(f(x))J_f(x)$。

一个玩具例子：

$$
g(x_1,x_2)=
\begin{bmatrix}
x_1^2\\
x_1x_2
\end{bmatrix},
\quad
f(u,v)=u+2v
$$

这里 $g:\mathbb{R}^2\to\mathbb{R}^2$，$f:\mathbb{R}^2\to\mathbb{R}$。如果构造复合函数 $h=f\circ g$，就是先用 $g$ 产生两个输出，再用 $f$ 把它们合成一个标量：

$$
h(x_1,x_2)=x_1^2+2x_1x_2
$$

分别计算：

$$
J_g(x)=
\begin{bmatrix}
2x_1 & 0\\
x_2 & x_1
\end{bmatrix},
\quad
J_f=
\begin{bmatrix}
1 & 2
\end{bmatrix}
$$

在 $x=(1,3)$ 处：

$$
J_g(1,3)=
\begin{bmatrix}
2 & 0\\
3 & 1
\end{bmatrix}
$$

所以：

$$
J_h(1,3)=
\begin{bmatrix}
1 & 2
\end{bmatrix}
\begin{bmatrix}
2 & 0\\
3 & 1
\end{bmatrix}
=
\begin{bmatrix}
8 & 2
\end{bmatrix}
$$

直接对 $h(x_1,x_2)=x_1^2+2x_1x_2$ 求导，也有：

$$
\nabla h(1,3)=(2x_1+2x_2,\ 2x_1)\big|_{(1,3)}=(8,2)
$$

结果一致。

反向传播处理的是另一个常见问题：最终只关心一个标量损失 $L(y)$，其中 $y=f(x)$。此时不一定需要完整的 $J_f$，只需要知道损失对 $x$ 的梯度。

由链式法则：

$$
\nabla_x L(f(x)) = J_f(x)^T \nabla_y L(y)
$$

这里 $\nabla_y L(y)$ 是上游梯度。上游梯度，指后续计算已经得到的“损失对当前输出的导数”。局部算子只要把它乘上自己的雅可比转置，就能得到“损失对当前输入的导数”。

| 前向传播对象 | 反向传播对象 | 数学形式 | 工程含义 |
|---|---|---|---|
| 输入 $x$ | 输入梯度 $\nabla_x L$ | $J_f(x)^T\nabla_y L$ | 求损失对输入的敏感度 |
| 输出 $y=f(x)$ | 上游梯度 $\nabla_y L$ | 从后续节点传来 | 表示后续损失如何依赖当前输出 |
| 局部函数 $f$ | 局部 VJP | $v^TJ_f$ 或 $J_f^Tv$ | 不显式存完整雅可比 |
| 计算图节点 | 反向节点 | 按拓扑逆序执行 | 从输出层传回输入层 |

VJP 是 vector-Jacobian product，中文可称为“向量-雅可比积”。在许多框架里，它的核心作用是避免构造完整雅可比，只计算当前反传需要的乘积。

---

## 代码实现

下面用 PyTorch 写一个最小可运行例子。第一个函数是二维输入、二维输出：

$$
f(x_1,x_2)=
\begin{bmatrix}
x_1^2\\
x_1x_2
\end{bmatrix}
$$

它的雅可比矩阵为：

$$
J_f(x)=
\begin{bmatrix}
2x_1 & 0\\
x_2 & x_1
\end{bmatrix}
$$

在 $x=(1,3)$ 处，应为：

$$
J_f(1,3)=
\begin{bmatrix}
2 & 0\\
3 & 1
\end{bmatrix}
$$

```python
import torch

def f(x):
    x1, x2 = x[0], x[1]
    return torch.stack([x1**2, x1 * x2])

x = torch.tensor([1.0, 3.0], requires_grad=True)

J = torch.autograd.functional.jacobian(f, x)

expected = torch.tensor([
    [2.0, 0.0],
    [3.0, 1.0],
])

print(J)
assert torch.allclose(J, expected)
```

这个例子显式计算了完整雅可比。它适合教学和小规模验证，但不适合直接照搬到大模型训练中。

下面再看复合函数的矩阵乘法顺序。设：

$$
r(y_1,y_2)=y_1+2y_2
$$

那么 $h(x)=r(f(x))=x_1^2+2x_1x_2$。根据链式法则：

$$
J_h(x)=J_r(f(x))J_f(x)
$$

```python
import torch

def f(x):
    x1, x2 = x[0], x[1]
    return torch.stack([x1**2, x1 * x2])

def r(y):
    return y[0] + 2.0 * y[1]

def h(x):
    return r(f(x))

x = torch.tensor([1.0, 3.0], requires_grad=True)

J_f = torch.autograd.functional.jacobian(f, x)
J_r = torch.autograd.functional.jacobian(r, f(x)).reshape(1, 2)
J_h_by_chain = J_r @ J_f

J_h_direct = torch.autograd.functional.jacobian(h, x).reshape(1, 2)

print(J_h_by_chain)
print(J_h_direct)

assert torch.allclose(J_h_by_chain, torch.tensor([[8.0, 2.0]]))
assert torch.allclose(J_h_by_chain, J_h_direct)
```

再看反向传播。真实训练里通常有标量损失，比如：

$$
L(y)=3y_1+4y_2
$$

上游梯度就是：

$$
\nabla_y L=
\begin{bmatrix}
3\\
4
\end{bmatrix}
$$

那么：

$$
\nabla_x L = J_f(x)^T\nabla_y L
$$

```python
import torch

def f(x):
    x1, x2 = x[0], x[1]
    return torch.stack([x1**2, x1 * x2])

x = torch.tensor([1.0, 3.0], requires_grad=True)
y = f(x)

loss = 3.0 * y[0] + 4.0 * y[1]
loss.backward()

J = torch.autograd.functional.jacobian(f, x)
upstream = torch.tensor([3.0, 4.0])
manual_grad = J.T @ upstream

print(x.grad)
print(manual_grad)

assert torch.allclose(x.grad, torch.tensor([18.0, 4.0]))
assert torch.allclose(x.grad, manual_grad)
```

这个 `backward()` 做的事情不是“重新正向算一遍函数”，而是沿着计算图反向执行局部梯度规则。对于每个局部函数，它把上游梯度乘上局部雅可比的转置，再传给前面的节点。

真实工程例子是神经网络训练。一个 Transformer 模型可以看成由嵌入层、注意力层、MLP、归一化、残差连接组成的计算图。最终损失是一个标量。训练时框架不会显式构造“损失前所有输出对所有参数”的巨大雅可比矩阵，而是从损失开始，逐层计算 VJP，把梯度传给每个参数张量。

如果是标量损失，优先看 `.grad`。如果是向量输出，才需要关注 `jacobian`、`jvp` 或 `vjp`。

---

## 工程权衡与常见坑

真实工程里通常不显式构造大雅可比。原因很直接：矩阵太大。

假设模型输出是 10 万维，输入是 1 万维，那么完整雅可比形状是：

$$
100000\times 10000
$$

也就是 $10^9$ 个元素。如果每个元素用 32 位浮点数存储，仅这个矩阵就需要约 4GB 内存，还不包括计算图、中间激活、参数、优化器状态。更大的模型里，这种做法很快不可行。

所以自动微分框架通常不会默认存完整雅可比，而是提供两类更常用的乘积：

| 名称 | 形式 | 直观含义 | 常见用途 |
|---|---|---|---|
| JVP | $Jv$ | 输入方向 $v$ 经过函数后造成的输出变化 | 前向模式自动微分 |
| VJP | $v^TJ$ 或 $J^Tv$ | 上游梯度经过局部函数传回输入 | 反向传播 |
| 显式雅可比 | $J$ | 全部输出对全部输入的偏导 | 小规模分析、调试、敏感性研究 |

最常见的错误不是不会写公式，而是维度约定不一致。本文固定 $J_f$ 形状是“输出维 × 输入维”。如果有人把它写成“输入维 × 输出维”，那么链式法则的矩阵乘法顺序也会变化。两种约定本身都可以成立，但同一篇推导、同一段代码里不能混用。

| 坑点 | 错误表现 | 正确做法 |
|---|---|---|
| 行列约定混乱 | $J_gJ_f$ 维度对不上，或结果方向反了 | 固定 $J_f$ 为“输出维 × 输入维” |
| 把梯度和雅可比混用 | 向量输出时误以为只有一个梯度向量 | 标量输出看梯度，向量输出看雅可比 |
| 显式构造大雅可比 | 内存暴涨、运行极慢 | 优先使用 JVP 或 VJP |
| 反向时忘记转置 | 用 $J$ 直接乘上游梯度导致形状错误 | 使用 $J^T\nabla_y L$ |
| 忽略中间点 | 把 $J_g(f(x))$ 写成 $J_g(x)$ | 后一层雅可比必须在它自己的输入点处计算 |
| 原地修改张量 | 自动微分报错或梯度异常 | 避免破坏计算图需要的中间值 |

需要特别强调四点：

| 规则 | 含义 |
|---|---|
| 固定约定 | $J_f$ 形状始终是“输出维 × 输入维” |
| 区分对象 | 标量输出看梯度，向量输出看雅可比 |
| 避免大矩阵 | 优先使用 JVP/VJP，不默认显式存完整矩阵 |
| 反向转置 | 反向传播里是 $J^T$ 乘上游梯度，不是 $J$ 直接往回乘 |

一个易错例子是线性层。设 $y=Wx$，其中 $W$ 形状是 $m\times n$，$x$ 是 $n$ 维，$y$ 是 $m$ 维。此时：

$$
J_y(x)=W
$$

如果损失对输出的梯度是 $\nabla_y L$，那么损失对输入的梯度是：

$$
\nabla_x L=W^T\nabla_y L
$$

很多初学者会写成 $W\nabla_y L$，这在维度上通常就已经不成立；即使某些方阵场景下维度碰巧成立，含义也错了。

---

## 替代方案与适用边界

链式法则本身是通用的，但工程实现要根据输入维度、输出维度、是否需要全部导数来选择策略。不是所有场景都应该显式求雅可比，也不是所有场景都只适合反向传播。

| 方法 | 适合场景 | 优点 | 缺点 |
|---|---|---|---|
| 显式雅可比 | 输入输出维度都小，需要完整偏导信息 | 结果完整，便于分析和调试 | 高维时内存和计算成本过高 |
| JVP | 少量输入方向对大量输出的影响 | 适合前向模式，计算 $Jv$ 高效 | 不直接给出标量损失对所有参数的梯度 |
| VJP | 标量损失反传，或给定上游向量 | 适合反向模式，不必存完整 $J$ | 若要完整雅可比，需要多次调用 |
| 反向模式自动微分 | 一个标量损失，对大量参数求梯度 | 深度学习训练的主力方法 | 对大量输出分别求导时可能不划算 |

如果只想知道“一个标量损失对很多参数的梯度”，反向模式最合适。神经网络训练就是这种情况：参数可能有上亿个，但最终 loss 通常是一个标量。一次反向传播就能得到所有参数的梯度。

如果想知道“少量输入方向对很多输出的变化”，前向模式可能更划算。例如你只关心输入沿某个方向 $v$ 扰动后，模型所有输出如何变化，这时直接算 $Jv$，不需要完整 $J$。

如果确实需要完整雅可比，比如小模型的敏感性分析、机器人控制里的局部线性化、小规模数值检查，可以显式构造。但要先估算矩阵大小。只要输出维和输入维都很大，完整雅可比通常就不是默认选项。

还要注意以下边界：

| 边界情况 | 问题 | 常见处理 |
|---|---|---|
| 不可微点 | 标准导数不存在 | 使用次梯度或约定梯度 |
| 离散算子 | 取整、采样、argmax 无法直接求导 | 使用近似梯度、重参数化或替代损失 |
| 超高维输入输出 | 完整雅可比不可承受 | 使用 JVP/VJP、分块计算、随机估计 |
| 批量计算 | 逐样本雅可比可能维度复杂 | 使用向量化、`vmap`、批量 VJP/JVP |
| 控制流分支 | 不同路径导数不同 | 按实际执行路径构建计算图 |

不可微点不是罕见问题。ReLU 在 0 处不可微，绝对值函数在 0 处不可微，`max` 在多个输入相等时也有不可微点。深度学习框架通常会选择一个工程上可用的梯度定义，例如 ReLU 在 0 处取某个约定值。这不等于数学问题消失了，只是训练算法需要一个可执行的规则。

离散操作更需要谨慎。例如 `argmax` 输出的是类别编号，输入发生很小变化时，类别编号大多数时候不变，一旦越过边界又突然跳变。标准链式法则无法直接给出有用梯度。分类模型训练通常不会对 `argmax` 反传，而是对 softmax 前后的连续值构造损失。

---

## 参考资料

| 资料 | 类型 | 适合读者 | 能解决的问题 |
|---|---|---|---|
| PyTorch Autograd mechanics | 框架文档 | 想理解 PyTorch 反向传播行为的读者 | 解释计算图、梯度保存、反向执行机制 |
| `torch.autograd.functional.jacobian` | API 文档 | 需要显式计算雅可比的读者 | 展示如何用 PyTorch 直接求雅可比 |
| `torch.autograd.functional.vjp` | API 文档 | 需要理解 VJP 的读者 | 展示如何计算向量-雅可比积 |
| JAX forward/reverse mode autodiff | 框架文档 | 想比较前向模式和反向模式的读者 | 解释 JVP、VJP 与自动微分模式选择 |
| `jax.jacrev` | API 文档 | 使用 JAX 求雅可比的读者 | 展示反向模式构造雅可比的方法 |
| Baydin et al., 2018, *Automatic differentiation in machine learning: a survey* | 自动微分综述 | 想系统理解自动微分的读者 | 梳理自动微分、符号微分、数值微分的区别 |
| Rumelhart, Hinton, Williams, 1986, *Learning representations by back-propagating errors* | 经典论文 | 想了解反向传播历史来源的读者 | 说明反向传播如何用于多层神经网络学习 |

推荐阅读顺序是先看 PyTorch 或 JAX 文档，理解工具怎么用；再看 Baydin 等人的综述，理解自动微分为什么这样设计；最后看 Rumelhart、Hinton、Williams 1986 年的论文，理解反向传播在神经网络训练中的历史位置。

链接：

| 资料 | 链接 |
|---|---|
| PyTorch Autograd mechanics | https://docs.pytorch.org/docs/stable/notes/autograd |
| PyTorch `jacobian` | https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian |
| PyTorch `vjp` | https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.vjp.html |
| JAX autodiff cookbook | https://docs.jax.dev/en/latest/jacobian-vector-products.html |
| JAX `jax.jacrev` | https://docs.jax.dev/en/latest/_autosummary/jax.jacrev.html |
| Baydin et al., 2018 | https://www.jmlr.org/papers/v18/17-468.html |
| Rumelhart, Hinton, Williams, 1986 | https://www.nature.com/articles/323533a0 |
