## 核心结论

泰勒展开解决的问题很直接：当我们只知道函数在当前位置的信息时，怎样估计它在附近会怎么变。对多元函数 $f:\mathbb{R}^d\to\mathbb{R}$，在点 $x$ 附近走一个小步长 $\delta$，有

$$
f(x+\delta)=f(x)+\nabla f(x)^\top \delta+\frac12\delta^\top \nabla^2 f(x)\delta+R_2(x,\delta),
$$

其中二阶余项 $R_2(x,\delta)$ 在足够光滑时满足

$$
R_2(x,\delta)=O(\|\delta\|^3).
$$

如果只关心主导项，也常写成

$$
f(x+\delta)=f(x)+\nabla f(x)^\top \delta+\frac12\delta^\top \nabla^2 f(x)\delta+O(\|\delta\|^3).
$$

这里的梯度 $\nabla f(x)$ 表示函数在当前位置的局部变化率；它指出上升最快的方向。Hessian $\nabla^2 f(x)$ 表示局部曲率；它告诉你不同方向上的“弯曲”有多强。

这条公式给出三个直接结论：

| 近似层级 | 保留的项 | 描述的现象 | 对应优化含义 |
|---|---:|---|---|
| 一阶近似 | $f(x)+\nabla f(x)^\top\delta$ | 局部线性变化 | 梯度下降的基础 |
| 二阶近似 | 再加 $\frac12\delta^\top H\delta$ | 局部曲率 | 牛顿法的基础 |
| 余项 | $O(\|\delta\|^3)$ | 更高阶误差 | 决定步长能否信任 |

把更新步写成 $\delta=-\eta \nabla f(x)$，就得到梯度下降的一步展开：

$$
f(x-\eta \nabla f(x))
=
f(x)-\eta\|\nabla f(x)\|^2
+\frac12\eta^2 \nabla f(x)^\top H(x)\nabla f(x)
+O(\eta^3\|\nabla f(x)\|^3).
$$

如果把 $\eta$ 视作小量，常简写为

$$
f(x-\eta \nabla f(x))
=
f(x)-\eta\|\nabla f(x)\|^2
+\frac12\eta^2 \nabla f(x)^\top H(x)\nabla f(x)
+O(\eta^3).
$$

第一项 $-\eta\|\nabla f(x)\|^2$ 负责下降，第二项和更高阶项负责把线性模型的乐观估计拉回真实函数。因此步长 $\eta$ 不能只看梯度大小，还要看曲率大小。

先看一维函数 $f(x)=\tfrac12x^2$。在 $x=1$、$\eta=0.1$ 时：

- 梯度 $f'(1)=1$
- Hessian $f''(1)=1$
- 更新后 $x^+=1-0.1=0.9$
- 真值 $f(0.9)=0.405$

用泰勒展开计算：

$$
f(1-0.1)=f(1)-0.1\cdot 1^2+\frac12\cdot 0.1^2\cdot 1
=0.5-0.1+0.005
=0.405.
$$

这个例子里二阶近似就是精确值，因为二次函数没有三阶项。

再把曲率变大。若 $f(x)=2x^2$，则在 $x=1$ 处：

- 梯度 $f'(1)=4$
- Hessian $f''(1)=4$

若仍取 $\eta=0.1$，则

$$
f(1-0.1\cdot 4)=f(0.6)=0.72.
$$

一阶近似给出

$$
f(1)-\eta\|g\|^2 = 2 - 0.1\times 16 = 0.4,
$$

明显过于乐观。二阶修正项为

$$
\frac12\eta^2 g^\top H g
=
\frac12\times 0.01 \times 4 \times 4^2
=
0.32.
$$

把它加回去后得到

$$
0.4+0.32=0.72,
$$

恰好回到真实值。这个例子说明：问题不是“梯度大一点”，而是“曲率一大，线性模型更容易失真”。因此更新幅度要缩小，或者改用裁剪、预条件化、二阶修正。

再看一个二维例子，帮助理解“不同方向曲率不同”是什么意思。设

$$
f(x_1,x_2)=\frac12(x_1^2+100x_2^2).
$$

则

$$
\nabla f(x)=
\begin{bmatrix}
x_1\\
100x_2
\end{bmatrix},
\qquad
\nabla^2 f(x)=
\begin{bmatrix}
1&0\\
0&100
\end{bmatrix}.
$$

这说明第 2 个方向的曲率是第 1 个方向的 100 倍。若对两个方向使用同一个学习率，沿 $x_2$ 方向就更容易过冲。这正是很多优化器要做“按坐标缩放步长”的原因。

---

## 问题定义与边界

泰勒展开不是对所有函数、所有步长都可靠。它成立的前提是：函数在当前点附近足够光滑。最少要能算到你保留的那一阶导数；如果还想给出明确误差界，就要再加更强的平滑性假设。

对初学者，最常见的两个假设是：

| 假设 | 形式 | 直观含义 | 常见用途 |
|---|---|---|---|
| 二阶可导 | $\nabla^2 f(x)$ 存在 | 能写出二阶泰勒式 | 分析牛顿法、二阶近似 |
| $L$-smooth | $\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|$ | 梯度变化不会过猛 | 推出下降引理与步长上界 |
| Hessian Lipschitz | $\|\nabla^2 f(x)-\nabla^2 f(y)\|\le \rho\|x-y\|$ | 曲率本身也不会剧烈跳变 | 控制二阶余项 |

最常见的工程假设是 $L$-smooth。它的白话解释是：梯度不会突然跳变，任意两点之间的梯度差，最多与距离成正比。形式上写作

$$
\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|.
$$

在这个条件下，有标准上界，也叫下降引理：

$$
f(y)\le f(x)+\nabla f(x)^\top (y-x)+\frac{L}{2}\|y-x\|^2.
$$

把 $y=x-\eta \nabla f(x)$ 代入，得到

$$
f(x-\eta \nabla f(x))
\le
f(x)-\eta\|\nabla f(x)\|^2+\frac{L}{2}\eta^2\|\nabla f(x)\|^2.
$$

整理后可得

$$
f(x-\eta \nabla f(x))
\le
f(x)-\eta\left(1-\frac{L\eta}{2}\right)\|\nabla f(x)\|^2.
$$

因此只要

$$
0<\eta<\frac{2}{L},
$$

右侧下降量就是正的，梯度下降在单步意义上是安全的。若取更保守的 $\eta\le \frac1L$，则可直接得到

$$
f(x-\eta \nabla f(x))
\le
f(x)-\frac{\eta}{2}\|\nabla f(x)\|^2.
$$

这比“经验上把学习率调小”更精确：学习率上界本质上来自曲率上界。

一个标准例子是二次函数

$$
f(x)=\frac12 x^\top A x,
$$

其中 $A$ 是对称正定矩阵。此时

$$
\nabla f(x)=Ax,\qquad \nabla^2 f(x)=A.
$$

因为 Hessian 恒定，所以

$$
L=\lambda_{\max}(A).
$$

也就是说，最大特征值直接给出最陡方向的曲率，也直接决定安全步长上限。对二次函数，很多“步长为什么不能太大”的问题都能被这个量解释。

| 误差项 | 来源 | 量级 | 需要的假设 |
|---|---|---|---|
| 一阶截断误差 | 忽略二阶项 | $O(\|\delta\|^2)$ | Hessian 有界 |
| 二阶截断误差 | 忽略三阶及以上项 | $O(\|\delta\|^3)$ | Hessian Lipschitz 或三阶导有界 |
| 步长误差 | 取 $\delta=-\eta g$ 后局部模型失真 | 常表现为 $O(\eta^2)$ 主导修正 | $L$-smooth |
| 数值误差 | 浮点舍入、下溢、上溢 | 与机器精度 $\varepsilon_{\text{mach}}$ 相关 | 数值稳定性分析 |

这里的边界要说清楚：

1. 泰勒展开是局部结论，不是全局结论。你只能在“小邻域内”信它。
2. 误差受 $\|\delta\|$ 控制，所以“步长设计”的本质，就是让局部近似不要失效。
3. 在深度学习里函数常常非凸，Hessian 也可能非正定。泰勒展开仍然能分析单步更新，但不能保证“按二阶信息走就一定更优”。
4. 若函数不光滑，例如 ReLU 在折点处不可导，严格泰勒公式要改成次梯度或分段分析，不能直接照搬。

---

## 核心机制与推导

先看最基本的梯度下降。它等于只信一阶项，把局部模型写成

$$
m_1(\delta)=f(x)+\nabla f(x)^\top \delta.
$$

这个模型本身没有最小值。原因很简单：只要沿着 $-\nabla f(x)$ 方向不断走，线性函数就会一直下降。因此工程上不会直接最小化 $m_1(\delta)$，而是额外加一个步长限制，再取

$$
\delta=-\eta \nabla f(x).
$$

所以梯度下降真正做的事情不是“精确解一个局部优化问题”，而是“在一阶模型可信的小范围内，选一个下降步”。

二阶方法更进一步，直接保留曲率：

$$
m_2(\delta)=f(x)+\nabla f(x)^\top\delta+\frac12\delta^\top H(x)\delta.
$$

若 $H(x)\succ 0$，也就是 Hessian 正定，当前位置附近像一个开口向上的碗，那么这个二次模型有唯一极小点。对 $\delta$ 求导并令其为零：

$$
\nabla_\delta m_2(\delta)=\nabla f(x)+H(x)\delta=0,
$$

解得

$$
\delta^*=-H(x)^{-1}\nabla f(x).
$$

这就是牛顿步。它不是简单地沿着负梯度方向走，而是先根据各方向的曲率做缩放：

- 曲率大的方向，步子自动缩小
- 曲率小的方向，步子自动放大
- 坐标之间存在耦合时，Hessian 的非对角项还会改变方向本身

这一点可以用二维二次函数看得更清楚。设

$$
f(x)=\frac12 x^\top
\begin{bmatrix}
1&0\\
0&100
\end{bmatrix}
x.
$$

在点 $x=(1,1)$ 处，

$$
g=
\begin{bmatrix}
1\\
100
\end{bmatrix},
\qquad
H=
\begin{bmatrix}
1&0\\
0&100
\end{bmatrix}.
$$

此时：

- 梯度下降步：$\delta_{\text{gd}}=-\eta g$
- 牛顿步：$\delta_{\text{nt}}=-H^{-1}g=-(1,1)^\top$

可以看到，梯度下降对第 2 个坐标会给出 100 倍更大的原始驱动力；牛顿法则会用曲率把它重新缩放回来。对这个二次函数，牛顿法一步就到最优点 $x=0$。

再看一个矩阵级别的玩具例子。若 Hessian 为 $H=4I$，对某点的梯度记为 $g$，则

- 梯度下降：$\delta_{\text{gd}}=-\eta g$
- 牛顿法：$\delta_{\text{nt}}=-H^{-1}g=-\tfrac14 g$

若取 $\eta=0.1$，梯度下降步长系数是 $0.1$，牛顿步系数是 $0.25$。牛顿看起来走得更大，但这不是鲁莽，而是因为它已经把曲率显式算进去了。对二次函数而言，它是在直接最小化二阶模型。

Adam 可以理解为不去构造完整 Hessian，而是只做一个便宜得多的对角缩放。它维护一阶矩 $m_t$ 和二阶矩 $v_t$：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t\odot g_t.
$$

偏差修正后，

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
$$

更新写成

$$
\theta_{t+1}
=
\theta_t-\alpha \frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
$$

这里的 $\hat v_t$ 不是 Hessian，但它起到了“按坐标缩放步长”的作用。白话说，某个坐标如果长期梯度大，分母就会变大，这个坐标的更新会变小。它更接近“对角预条件器”，而不是严格意义上的二阶导。

把几种方法放在一个表里更清楚：

| 方法 | 更新 $\delta$ | 用到的信息 | 对泰勒展开的理解 | 主要误差来源 |
|---|---|---|---|---|
| SGD | $-\eta g$ | 梯度 | 只保留一阶项 | 二阶曲率被忽略 |
| Newton | $-H^{-1}g$ | 梯度 + Hessian | 直接最小化二阶模型 | 三阶余项、Hessian 求逆误差 |
| Adam | $-\alpha \hat m/(\sqrt{\hat v}+\varepsilon)$ | 梯度 + 二阶矩 | 用对角缩放模拟曲率 | 非对角曲率被忽略 |
| 梯度裁剪 | $-\eta\operatorname{clip}(g)$ | 梯度 + 阈值 | 限制 $\|\delta\|$ 让局部展开更可信 | 裁剪引入方向或幅值偏差 |

真实工程例子是大模型训练中的全局梯度裁剪。设梯度为 $g$，阈值为 $c$，常见做法是

$$
\operatorname{clip}(g,c)=
g\cdot \min\left(1,\frac{c}{\|g\|}\right).
$$

这件事的本质不是一句“防止爆炸”就讲完了。更准确的理解是：它在直接控制更新半径。因为如果更新写成 $\delta=-\eta g$，那么裁剪后就有

$$
\|\delta\|=\eta\|\operatorname{clip}(g,c)\|\le \eta c.
$$

一旦 $\eta\|g\|$ 太大，二阶项和更高阶项就会迅速放大，线性近似不再可信；裁剪就是把更新强行拉回局部模型还能解释的区域。

---

## 代码实现

下面给出一个可直接运行的 Python 例子，做四件事：

1. 验证一阶、二阶近似和真实值之间的关系
2. 演示高曲率时一阶近似为什么会失真
3. 比较 SGD、梯度裁剪、Newton、Adam 风格更新
4. 用二维二次函数展示“按方向缩放”的意义

代码只依赖 Python 标准库，可直接运行。

```python
import math

def f1(x):
    return 0.5 * x * x

def g1(x):
    return x

def h1(x):
    return 1.0

def f2(x):
    return 2.0 * x * x

def g2(x):
    return 4.0 * x

def h2(x):
    return 4.0

def taylor_first_order(f, grad, x, delta):
    return f(x) + grad(x) * delta

def taylor_second_order(f, grad, hessian, x, delta):
    return f(x) + grad(x) * delta + 0.5 * hessian(x) * delta * delta

def clip_grad_scalar(g, max_norm):
    norm = abs(g)
    if norm <= max_norm:
        return g
    return g * (max_norm / norm)

def sgd_step_scalar(x, grad, lr, max_norm=None):
    g = grad(x)
    if max_norm is not None:
        g = clip_grad_scalar(g, max_norm)
    return x - lr * g

def newton_step_scalar(x, grad, hessian, damping=0.0):
    g = grad(x)
    h = hessian(x) + damping
    if h <= 0:
        raise ValueError("Hessian + damping must be positive in this toy example.")
    return x - g / h

def adam_like_step_scalar(
    x, grad, m, v, t, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8
):
    g = grad(x)
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g * g)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    x = x - lr * m_hat / (math.sqrt(v_hat) + eps)
    return x, m, v

def quad2_value(x1, x2):
    return 0.5 * (x1 * x1 + 100.0 * x2 * x2)

def quad2_grad(x1, x2):
    return (x1, 100.0 * x2)

def quad2_sgd_step(x1, x2, lr):
    g1, g2 = quad2_grad(x1, x2)
    return (x1 - lr * g1, x2 - lr * g2)

def quad2_newton_step(x1, x2):
    # Hessian = diag(1, 100), inverse = diag(1, 0.01)
    g1, g2 = quad2_grad(x1, x2)
    return (x1 - g1, x2 - 0.01 * g2)

# 1) 一维：f(x)=0.5 x^2
x = 1.0
lr = 0.1
delta = -lr * g1(x)

true_value = f1(x + delta)
first_value = taylor_first_order(f1, g1, x, delta)
second_value = taylor_second_order(f1, g1, h1, x, delta)

assert abs(true_value - 0.405) < 1e-12
assert abs(first_value - 0.4) < 1e-12
assert abs(second_value - true_value) < 1e-12

# 2) 高曲率：f(x)=2x^2
x = 1.0
lr = 0.1
delta = -lr * g2(x)

true_value = f2(x + delta)
first_value = taylor_first_order(f2, g2, x, delta)
second_value = taylor_second_order(f2, g2, h2, x, delta)

assert abs(true_value - 0.72) < 1e-12
assert abs(first_value - 0.4) < 1e-12
assert abs(second_value - 0.72) < 1e-12

# 3) 几种一步更新
x_sgd = sgd_step_scalar(1.0, g1, 0.1)
x_clip = sgd_step_scalar(1.0, g2, 0.1, max_norm=2.0)
x_newton = newton_step_scalar(1.0, g1, h1)
x_adam, m, v = adam_like_step_scalar(1.0, g1, 0.0, 0.0, 1)

assert abs(x_sgd - 0.9) < 1e-12
assert abs(x_clip - 0.8) < 1e-12   # 原始梯度 4 被裁到 2
assert abs(x_newton - 0.0) < 1e-12
assert x_adam < 1.0

# 4) 二维：各方向曲率不同
x1, x2 = 1.0, 1.0
sgd_next = quad2_sgd_step(x1, x2, lr=0.01)
newton_next = quad2_newton_step(x1, x2)

assert abs(quad2_value(*newton_next) - 0.0) < 1e-12
assert sgd_next == (0.99, 0.0)

print("All checks passed.")
print("1D quadratic true / 1st / 2nd =", true_value, first_value, second_value)
print("SGD next =", x_sgd)
print("Clipped SGD next =", x_clip)
print("Newton next =", x_newton)
print("Adam-like next =", x_adam)
print("2D SGD next =", sgd_next)
print("2D Newton next =", newton_next)
```

这段代码里有几个工程点值得单独说明。

第一，一阶近似和二阶近似的误差要分开看。对二次函数，二阶近似就是精确值；但一阶近似只要曲率大，就可能严重低估真实函数值。上面的 $f(x)=2x^2$ 例子就是最简单的反例。

第二，裁剪顺序通常是先算梯度，再裁剪，再更新。因为你要限制的是最终更新半径 $\|\delta\|$，而不是更新之后再补救。

第三，Adam 中的 $\varepsilon$ 很关键。它的作用不是“让公式看起来完整”，而是避免某些坐标上 $\hat v_t$ 太小，导致分母接近 0，从而把该坐标的步长异常放大。

第四，Newton 在玩具例子里一步到最优，只因为这里的目标函数是理想二次函数。真实问题里 Hessian 可能：

- 不可逆
- 非正定
- 计算太贵
- 含噪声

所以实际中常用阻尼牛顿法，把 $H$ 换成 $H+\lambda I$，也就是

$$
\delta=-(H+\lambda I)^{-1}g.
$$

这等于在使用二阶信息的同时，给模型加一个保守项，避免走得太激进。

如果把训练循环抽象一下，典型顺序是：

```python
grad = compute_grad(x)
grad = clip(grad, max_norm=c)
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * (grad * grad)
update = -eta * m / (sqrt(v) + eps)
x += update
```

这段顺序对应的泰勒含义是：

1. `compute_grad`：获取一阶项，也就是局部线性变化率。
2. `clip`：限制更新半径，避免离开局部可信区。
3. `v`：估计各坐标上的尺度，用对角缩放近似曲率效应。
4. `eps`：减少数值不稳定带来的额外误差。

| 算法 | 需要的量 | 单步复杂度 | 误差主导项 | 适合场景 |
|---|---|---|---|---|
| SGD | 梯度 | 低 | $O(\eta^2)$ 二阶修正主导 | 大规模、便宜、稳定基线 |
| SGD + Clip | 梯度、阈值 | 低 | 裁剪偏差 + 二阶修正 | 梯度爆炸风险高 |
| Newton | 梯度、Hessian | 高 | $O(\|\delta\|^3)$ 余项 + 求逆误差 | 低维或结构可利用问题 |
| Adam | 梯度、一二阶矩 | 中 | 对角近似误差 | 深度网络常用默认选择 |

---

## 工程权衡与常见坑

梯度裁剪最容易被误解。它不是无成本的稳定器，而是用偏差换稳定性。

如果阈值 $c$ 很小，大量梯度都会被压到相近的范数，方向信息和幅值信息都会被改写；如果阈值 $c$ 很大，裁剪几乎不生效，更新仍可能跑出局部线性区。Koloskova 等人在 2023 年的分析指出：在确定性梯度下降里，裁剪阈值主要影响高阶项；但在随机梯度场景下，裁剪可能引入不可忽略的随机偏差，不能简单理解成“只会更稳定，不会改目标”。

| 情况 | 直接后果 | 泰勒视角下的问题 | 工程现象 |
|---|---|---|---|
| 阈值太小 | 更新长期被压缩 | 一阶项本身被系统性改写 | 收敛慢，甚至停在次优邻域 |
| 阈值太大 | 控幅失效 | 二阶项和高阶项放大 | loss 抖动、训练不稳 |
| 阈值随层不匹配 | 各层有效步长失衡 | 局部模型在不同层同时失效 | 某些层几乎不学，某些层过冲 |

真实工程里，例如大语言模型训练，经常设置一个全局阈值 $c$，目的就是控制

$$
\|\delta\|=\eta\|\operatorname{clip}(g,c)\|\le \eta c.
$$

这等于给泰勒展开中的“可信半径”加了一个硬约束。你可以把它理解成一种很粗但有效的 trust region。

混合精度训练也可以从泰勒展开看。设一次低精度计算引入舍入误差 $e$，实际更新是 $\delta+e$，则

$$
f(x+\delta+e)
=
f(x+\delta)+\nabla f(x+\delta)^\top e + O(\|e\|^2).
$$

如果又叠加截断近似，整体误差可以粗略写成

$$
\text{总误差}
=
O(\|\delta\|^{k+1}) + O(\|\nabla f(x+\delta)\|\cdot\|e\|) + O(\|e\|^2),
$$

其中 $k$ 是泰勒截断阶数。这个式子的意思很直接：你不能只关心“理论上保留到几阶”，还要关心“低精度把更新又扭曲了多少”。

因此工程上常见三件事总是一起出现：

1. 损失缩放，防止 FP16 梯度下溢
2. `eps`，防止分母过小
3. FP32 主权重，避免更新累积误差过大

Micikevicius 等人的混合精度训练工作给出的实践建议也是这个方向：大部分张量可以低精度存储和计算，但主权重保留高精度，并通过 loss scaling 降低下溢风险。

常见坑还有几个。

第一，把 Adam 的二阶矩当成真实 Hessian，这不准确。$\hat v_t$ 只是逐坐标的平方梯度统计量，不包含参数之间的耦合项。它更像“每个坐标自己的尺度估计”，而不是完整曲率矩阵。

第二，把“二阶方法更快”理解成普遍规律，这也不成立。若 Hessian 非正定，牛顿方向甚至可能不是下降方向。此时常见补救是：

- 加阻尼
- 做线搜索
- 加 trust region
- 改用拟牛顿法或共轭梯度近似求解

第三，把“学习率小”误解成“永远更安全”。学习率太小虽然减少了高阶误差，但也可能让有效进展几乎为零。优化真正要平衡的是“模型误差”和“前进速度”，不是单纯地压缩步长。

第四，忽视非对角曲率。很多深度模型里真正困难的不是单个坐标太陡，而是不同参数之间强耦合。对角方法便宜，但看不见这一层结构。

---

## 替代方案与适用边界

当完整 Hessian 太贵时，工程上不会硬算，而是寻找不同程度的近似。

Adam 与 RMSProp 都属于对角预条件化。它们只估计“每个参数自己的尺度”，不估计参数之间的耦合。这种近似之所以常用，不是因为它最准确，而是因为它在“效果、存储、计算”之间给了一个可接受的平衡。

SGD + Momentum 是另一类替代。它没有显式二阶信息，但通过累计历史梯度来平滑方向。它解决的是“噪声和震荡”问题，不是“曲率显式建模”问题。狭长谷底里，Momentum 往往比纯 SGD 更容易沿着主方向前进。

拟牛顿法，例如 BFGS 或 L-BFGS，则处在一阶和二阶之间。它们不显式计算 Hessian，而是根据历史梯度和参数变化来近似 Hessian 或其逆。对中等规模、较光滑的问题，常比纯一阶法更高效。

| 方法 | 泰勒截断/近似策略 | 误差控制机制 | 适用边界 |
|---|---|---|---|
| Adam | 对角二阶近似 | $\sqrt{v_t}+\varepsilon$ 缩放步长 | 高维、噪声大、各层尺度差异大 |
| RMSProp | 对角二阶近似但无一阶矩 | 历史平方梯度平滑 | 非平稳梯度、训练初期 |
| SGD + Momentum | 仍是一阶，但对方向做时间平滑 | 减少震荡、平滑随机噪声 | 数据大、内存敏感、追求简单稳定 |
| Damped Newton | 二阶模型 + 正则化 | 用 $H+\lambda I$ 防止非正定/病态 | 中小规模、结构化问题 |
| Trust Region | 只在可信半径内相信局部模型 | 显式限制 $\|\delta\|$ | 局部模型容易失真时 |
| L-BFGS | 近似二阶但不显式存 Hessian | 用历史曲率信息改善方向 | 中等规模、较平滑目标 |

适用边界可以总结成一句话：你对曲率了解得越准，单步可以走得越大胆；你对曲率了解得越粗糙，就越要依赖小步长、裁剪、阻尼或可信半径控制。

所以：

- 想要最低成本和强基线，用 SGD、Momentum 或 Adam。
- 曲率差异很大，但完整 Hessian 不现实，用 Adam、RMSProp 这类对角方法。
- 明显存在梯度爆炸风险，用裁剪。
- 问题维度不大、目标函数光滑、Hessian 可利用，用阻尼牛顿或拟牛顿法。
- 局部模型经常失真时，用 trust region 或线搜索，而不是盲目把二阶步直接走满。

---

## 参考资料

| 标题 | 作者/机构 | 年份 | 对应章节 |
|---|---|---:|---|
| [Lecture 7: Gradient Descent (and Beyond)](https://www.cs.cornell.edu/courses/cs4780/2023sp/lectures/pdfs/lecturenote07.pdf) | Cornell CS4780 | 2021 讲义，2023 课程页收录 | 问题定义与边界、步长界 |
| [Convex Optimization, Lecture 15-16: Unconstrained Minimization / Newton’s Method](https://see.stanford.edu/Course/EE364A/76) | Stanford EE364A | 2008 课程资料 | 核心机制与推导、牛顿步 |
| *Numerical Optimization* | Jorge Nocedal, Stephen Wright | 2006 | 核心结论、牛顿法、拟牛顿法 |
| [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | Diederik P. Kingma, Jimmy Ba | 2014 | 核心机制与推导、替代方案 |
| [Revisiting Gradient Clipping: Stochastic bias and tight convergence guarantees](https://proceedings.mlr.press/v202/koloskova23a.html) | Anastasia Koloskova, Hadrien Hendrikx, Sebastian U. Stich | 2023 | 工程权衡与常见坑 |
| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | Paulius Micikevicius et al. | 2017 | 工程权衡与常见坑 |
| [Train With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) | NVIDIA Docs | 近年持续更新 | 工程实现、loss scaling |

文中最关键的两个公式分别来自 smoothness 分析与二阶局部模型：

$$
f(x-\eta\nabla f(x))
\le
f(x)-\eta\left(1-\frac{L\eta}{2}\right)\|\nabla f(x)\|^2
$$

以及

$$
\delta^*=-H(x)^{-1}\nabla f(x),\qquad H(x)\succ 0.
$$

前者说明一阶方法为什么需要步长上界，后者说明二阶方法为什么能显式利用曲率。

如果把全文压缩成一句话，就是：泰勒展开把一次参数更新拆成“线性下降收益”和“曲率带来的修正成本”；优化器的差别，本质上是在用不同代价估计这两个量，并设法把余项控制在可接受范围内。
