## 核心结论

偏导数解决的是“只沿一个坐标轴动一点，函数怎么变”。对多元函数 $f(x_1,\dots,x_n)$，第 $i$ 个偏导数定义为

$$
\frac{\partial f}{\partial x_i}(a)=\lim_{h\to 0}\frac{f(a+h e_i)-f(a)}{h}
$$

其中 $e_i$ 是第 $i$ 个标准基向量，白话就是“只改一个变量，其余变量先固定”。

全微分解决的是“所有变量都发生很小变化时，函数能否被一个线性函数近似”。如果 $f$ 在点 $a$ 可微，那么存在线性映射 $Df(a)$，使得

$$
f(a+h)=f(a)+Df(a)\,h+o(\|h\|)
$$

在坐标形式下，这个线性映射就是

$$
df=\sum_{i=1}^n \frac{\partial f}{\partial x_i}dx_i
$$

白话就是“函数在这一点附近最像一个平面或超平面”，全微分就是这个平面的线性部分。

梯度是把所有偏导数组成的向量：

$$
\nabla f=\left(\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\right)
$$

它指向函数增长最快的方向。若 $v$ 是单位方向向量，方向导数满足

$$
D_v f=\nabla f\cdot v
$$

链式法则是多层复合函数求导的核心。若 $y=f(g(x))$，则总导数满足

$$
D(f\circ g)(x)=Df(g(x))\cdot Dg(x)
$$

这正是神经网络反向传播和 PyTorch `autograd` 的数学基础。

---

## 问题定义与边界

偏导数、全微分、可微，这三个概念相关，但不是一回事。初学时最容易混淆的是：偏导存在，不等于函数可微。

可以先看一个对照表：

| 概念 | 数学对象 | 关注的问题 | 结论强度 |
|---|---|---|---|
| 偏导数 | 单个变量方向的极限 | 固定其他变量后，沿某个坐标轴的瞬时变化率 | 最弱 |
| 梯度 | 所有偏导组成的向量 | 各坐标方向变化率的整体表达 | 中等 |
| 全微分 | 线性映射 $Df(a)$ | 点附近是否能用线性函数统一近似 | 更强 |
| 可微 | 线性近似成立 | 函数局部是否“像线性的” | 最强 |

一个常见误解是：“只要每个偏导数都存在，就能写出 $df$，所以函数可微。”这句话不成立。因为写出形式上的

$$
df=\sum_i \frac{\partial f}{\partial x_i}dx_i
$$

并不自动说明它真的是“最佳线性近似”。可微要求误差项比 $\|h\|$ 更小，即必须满足 $o(\|h\|)$。

玩具例子先看最标准的函数：

$$
f(x,y)=x^2+3y
$$

它的偏导数是

$$
\frac{\partial f}{\partial x}=2x,\qquad \frac{\partial f}{\partial y}=3
$$

在点 $(1,2)$ 处，梯度为

$$
\nabla f(1,2)=(2,3)
$$

这说明：
- 沿 $x$ 轴正方向，每前进一点，函数局部增加约 $2$
- 沿 $y$ 轴正方向，每前进一点，函数局部增加约 $3$

如果方向向量取 $u=\frac{1}{\sqrt 2}(1,-1)$，它是单位向量，那么方向导数是

$$
D_u f=\nabla f\cdot u=(2,3)\cdot \frac{1}{\sqrt 2}(1,-1)= -\frac{1}{\sqrt 2}
$$

结果是负数，表示沿这个方向函数在下降。这里必须强调：标准方向导数公式要求 $u$ 是单位向量。如果直接用 $(1,-1)$ 去点积，得到的是沿该位移向量的线性变化量，不是标准化后的方向导数。

再看边界例子。设

$$
f(x,y)=
\begin{cases}
\dfrac{xy}{\sqrt{x^2+y^2}}, &(x,y)\neq (0,0) \\
0, &(x,y)=(0,0)
\end{cases}
$$

在原点，沿 $x$ 轴和 $y$ 轴计算偏导，都存在且等于 $0$。但它不可微，因为当 $(x,y)=(t,t)$ 时，

$$
f(t,t)=\frac{t^2}{\sqrt{2t^2}}=\frac{|t|}{\sqrt 2}
$$

这个量是一级小量，不是比 $\sqrt{x^2+y^2}$ 更高阶的误差，因此无法被某个固定线性映射统一吸收。结论是：偏导存在，只说明“坐标轴方向上看起来没问题”，不说明“所有方向一起看也线性”。

一个足够实用的判断规则是：若各偏导在某点邻域内存在且连续，那么函数在该点可微。这是常用充分条件，不是必要条件。白话就是：偏导连续通常能保票局部线性近似成立，但不是唯一道路。

---

## 核心机制与推导

全微分的本质是局部线性化。设输入扰动为

$$
h=(h_1,\dots,h_n)
$$

若 $f$ 在点 $a$ 可微，则

$$
f(a+h)-f(a)\approx Df(a)\,h
$$

在坐标表示下，

$$
Df(a)\,h=\sum_{i=1}^n \frac{\partial f}{\partial x_i}(a)h_i
$$

把 $h_i$ 记成 $dx_i$，就得到熟悉的全微分公式

$$
df=\sum_{i=1}^n \frac{\partial f}{\partial x_i}dx_i
$$

所以“全微分”不是多写一个符号，而是把函数局部变化压缩成一个线性模型。

梯度与全微分的关系也可以这样理解：
- 全微分 $Df(a)$ 是线性映射
- 梯度 $\nabla f(a)$ 是这个线性映射在欧氏空间下对应的向量表示

因此任意方向 $v$ 上的一阶变化率都能写成

$$
Df(a)\,v=\nabla f(a)\cdot v
$$

如果 $v$ 是单位向量，这就是方向导数。

链式法则则说明，局部线性化可以逐层传递。设

$$
g:\mathbb{R}^n\to \mathbb{R}^m,\qquad f:\mathbb{R}^m\to \mathbb{R}
$$

复合函数 $y=f(g(x))$ 的导数是

$$
D(f\circ g)(x)=Df(g(x))\cdot Dg(x)
$$

这里：
- $Dg(x)$ 是雅可比矩阵，白话就是“中间变量对输入变量的一阶变化表”
- $Df(g(x))$ 是输出对中间变量的导数
- 两者相乘，就得到输出对原始输入的导数

可以看一个两层玩具例子。设

$$
g(x_1,x_2)=
\begin{bmatrix}
x_1x_2\\
x_1+x_2
\end{bmatrix},
\qquad
f(z_1,z_2)=z_1^2+3z_2
$$

那么

$$
Dg(x)=
\begin{bmatrix}
x_2 & x_1\\
1 & 1
\end{bmatrix},
\qquad
Df(z)=
\begin{bmatrix}
2z_1 & 3
\end{bmatrix}
$$

于是

$$
D(f\circ g)(x)=
\begin{bmatrix}
2z_1 & 3
\end{bmatrix}
\begin{bmatrix}
x_2 & x_1\\
1 & 1
\end{bmatrix}
$$

这不是技巧拼接，而是“先算后层局部斜率，再乘前层局部斜率”的严格线性代数结果。

真实工程里，神经网络就是大规模复合函数：
- 输入层到隐藏层是一层映射
- 激活函数又是一层映射
- 隐藏层到损失函数还是一层映射

反向传播做的事情，就是把这些 Jacobian 按链式法则连乘，并把梯度从输出一路传回输入和参数。

---

## 代码实现

先用纯 Python 写一个可运行的最小例子，验证梯度、方向导数和线性近似。

```python
import math

def f(x, y):
    return x * x + 3 * y

def grad_f(x, y):
    return (2 * x, 3.0)

def directional_derivative(x, y, v):
    vx, vy = v
    norm = math.sqrt(vx * vx + vy * vy)
    assert norm > 0
    ux, uy = vx / norm, vy / norm
    gx, gy = grad_f(x, y)
    return gx * ux + gy * uy

# 梯度
gx, gy = grad_f(1.0, 2.0)
assert gx == 2.0
assert gy == 3.0

# 方向导数：方向 (1, -1) 先单位化
dv = directional_derivative(1.0, 2.0, (1.0, -1.0))
assert abs(dv - (-1 / math.sqrt(2))) < 1e-12

# 全微分给出的线性近似
x0, y0 = 1.0, 2.0
dx, dy = 1e-6, -2e-6
linear_approx = f(x0, y0) + gx * dx + gy * dy
real_value = f(x0 + dx, y0 + dy)

# 对这个二次函数，误差是二阶小量，量级约为 dx^2
assert abs(real_value - linear_approx) < 1e-10
```

这段代码对应三件事：
- `grad_f` 实现偏导数组成的梯度
- `directional_derivative` 说明标准方向导数要先单位化方向向量
- `linear_approx` 展示全微分是局部最佳线性近似

再看真实工程例子。PyTorch 的自动微分系统会为参与求导的张量构建计算图。计算图是“每个运算节点都记录局部导数关系”的有向无环图。调用 `backward()` 时，系统按链式法则做反向累积。

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x[0] ** 2 + 3 * x[1]

assert y.grad_fn is not None  # 非叶子结果张量会记录计算历史

y.backward()

assert torch.allclose(x.grad, torch.tensor([2.0, 3.0]))
```

这里有三个关键点：
- `requires_grad=True` 表示这个张量需要记录梯度路径
- `grad_fn` 表示当前张量由哪些可导运算得到
- `backward()` 会从标量输出出发，按链式法则把梯度回传到叶子节点

如果把它翻译回数学，就是：
- $y=x_0^2+3x_1$
- $\frac{\partial y}{\partial x_0}=2x_0$
- $\frac{\partial y}{\partial x_1}=3$

在 $x=(1,2)$ 处，梯度正好是 $(2,3)$。

---

## 工程权衡与常见坑

工程里最常见的问题不是公式不会写，而是默认了错误前提。

| 常见坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| 偏导存在就当作可微 | 局部线性化失真 | 坐标轴方向成立，不代表所有方向成立 | 检查连续可微条件，或做数值验证 |
| 把非单位向量直接当方向导数 | 数值大小不对 | 方向导数标准定义要求单位方向 | 先归一化向量 |
| ReLU 在 0 点的导数当成严格数学事实 | 不同实现可能不同 | 0 点不可微，只能选次梯度或约定值 | 明确库实现，不要误当定理 |
| `.grad` 不清零重复反传 | 梯度越积越大 | PyTorch 默认累加梯度 | 每轮训练前 `optimizer.zero_grad()` |
| 把自动微分当成数值差分 | 调试方向错误 | 自动微分算的是精确链式导数，不是近似 | 区分 AD 与 finite difference |

ReLU 是一个典型工程坑。它定义为

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

当 $x<0$ 时导数是 $0$，当 $x>0$ 时导数是 $1$，但在 $x=0$ 处左右导数不同，所以数学上不可微。很多深度学习框架在实现时会把 $x=0$ 的梯度约定为 $0$。这能让训练继续，但要知道这是工程约定，不是“导数天然存在”。

真实工程例子是神经元死亡。若某层大量输入长期落在 ReLU 的负半轴，梯度为 $0$，参数更新就会停住；若刚好大量样本把激活压在 $0$ 附近，数值抖动还会让训练不稳定。常见做法包括：
- 用 He 初始化降低早期大面积饱和
- 用较小学习率避免把激活整体推入负区间
- 必要时改用 Leaky ReLU、GELU 等更平滑或保留负区梯度的激活

还有一个常被忽略的点：链式法则只要求各层在需要的位置可导，不要求全局都很漂亮。但一旦你在关键节点引入硬阈值、离散采样、排序、`argmax` 这类不可导操作，梯度路径就可能中断。此时要么改模型结构，要么使用近似梯度、替代损失或 straight-through estimator 之类的技巧。

---

## 替代方案与适用边界

不是所有问题都能直接用解析导数或自动微分。

如果函数只是一个黑盒，例如：
- 你只能调用接口拿到输出
- 函数来自外部仿真器
- 输入来自实验控制系统
- 中间过程不可追踪

那么自动微分就不可用。这时通常改用有限差分。有限差分是“用小步长前后试一下，再用差商近似导数”。

常见方案如下：

| 方案 | 公式 | 误差阶 | 数据需求 | 适用场景 |
|---|---|---|---|---|
| 前向差分 | $\frac{f(x+h)-f(x)}{h}$ | $O(h)$ | 当前点和右侧点 | 只能单侧采样 |
| 后向差分 | $\frac{f(x)-f(x-h)}{h}$ | $O(h)$ | 当前点和左侧点 | 只能单侧采样 |
| 中心差分 | $\frac{f(x+h)-f(x-h)}{2h}$ | $O(h^2)$ | 两侧点 | 离线分析、精度要求更高 |
| 自动微分 | 链式法则精确传播 | 理论上非截断误差 | 需要可追踪计算图 | 神经网络、可导程序 |

玩具例子：若只拿到单变量函数采样值，可以用中心差分估计导数

$$
f'(x)\approx \frac{f(x+h)-f(x-h)}{2h}
$$

例如 $f(x)=x^2$，在 $x=3$ 处取 $h=10^{-4}$，就会得到接近 $6$ 的结果。

真实工程例子：做推荐系统线上实验时，你可能有一个复杂业务指标，它依赖排序、曝光、点击、延迟、风控规则，整个链路未必可导。这时常见做法不是硬上反向传播，而是：
- 在离线可导代理模型上训练
- 对真实黑盒指标做有限差分灵敏度分析
- 或直接用强化学习、贝叶斯优化等黑盒优化方法

这就是适用边界：
- 如果系统是可导程序，自动微分最准确也最高效
- 如果系统只有输入输出采样，没有内部计算图，有限差分更现实
- 如果系统含大量离散决策或高噪声反馈，单纯一阶导数方法可能不够，需要更高层的优化框架

---

## 参考资料

- Wikipedia, Partial derivative: https://en.wikipedia.org/wiki/Partial_derivative
- Wikipedia, Total derivative: https://en.wikipedia.org/wiki/Total_derivative
- Wikipedia, Gradient: https://en.wikipedia.org/wiki/Gradient
- Wikipedia, Chain rule: https://en.wikipedia.org/wiki/Chain_rule
- PyTorch Docs, Automatic Differentiation with `torch.autograd`: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- LibreTexts, Finite Difference Approximation: https://math.libretexts.org/Bookshelves/Scientific_Computing_Simulations_and_Modeling/Scientific_Computing_%28Chasnov%29/I%3A_Numerical_Methods/6%3A_Finite_Difference_Approximation
