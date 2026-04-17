## 核心结论

重积分可以理解为“在多维空间里做总和”。如果一元积分是在一条线上累加面积，那么重积分就是在平面、空间或更高维空间里累加面积、体积、质量、概率等总量。它的核心价值有两点。

第一，**Fubini 定理**说明：当函数是**绝对可积**时，也就是 $\int |f| < \infty$，多维积分可以拆成若干个一维积分按顺序做，而且积分顺序可以交换。这件事的重要性在于，复杂问题往往不是“不会算”，而是“直接算太难”。拆成逐维积分后，问题常常立刻变简单。

$$
\int_{A\times B} f(x,y)\,d(x,y)
=
\int_A\left(\int_B f(x,y)\,dy\right)dx
$$

第二，**变量代换**说明：如果把原来的积分区域通过可微可逆变换改写成更规则的区域，积分值本身不会凭空改变，但微元面积或体积会缩放，所以必须乘上 Jacobian 行列式的绝对值。**Jacobian**可以直白理解为“这个变换在局部把小面积、小体积放大或缩小了多少倍”。

$$
\int_D f(x)\,dx
=
\int_U f(g(u))\,\left|\det J_g(u)\right|\,du
$$

一个最小玩具例子是长方形区域上的积分。比如在 $[0,1]\times[0,2]$ 上计算函数 $f(x,y)=xy$ 的总量。你可以先固定 $x$，把每一列沿着 $y$ 方向加起来；也可以先固定 $y$，把每一行沿着 $x$ 方向加起来。只要满足绝对可积，结果相同。这就是“逐层切片再累加”。

更高一层看，很多机器学习公式本质上都在做重积分。变分推断里的期望 $\mathbb{E}_{q(z)}[f(z)]$ 是高维积分；扩散模型训练中的边缘化也是高维积分；Normalizing Flow 则把变量代换直接写进密度公式：

$$
\log p(x)=\log p(z)+\log\left|\det\frac{\partial z}{\partial x}\right|
$$

这不是数学装饰，而是模型能否正确计算概率密度的基础。

---

## 问题定义与边界

重积分研究的是：给定一个函数 $f:\mathbb{R}^n\to\mathbb{R}$ 和一个区域 $D\subseteq \mathbb{R}^n$，如何计算

$$
\int_D f(\mathbf{x})\,d\mathbf{x}
$$

这里 $\mathbf{x}$ 表示一个 $n$ 维向量，直白说就是“坐标点”；$D$ 是积分区域，也就是“在哪块地方做总和”；$f(\mathbf{x})$ 是每个点上的密度、权重或函数值。

如果 $f(\mathbf{x})=1$，重积分常表示区域体积；如果 $f$ 是质量密度，重积分就是总质量；如果 $f$ 是概率密度，重积分就是总概率。

本文的讨论边界有三条。

第一，只讨论可积问题。特别是使用 Fubini 定理时，需要强调绝对可积，因为“条件收敛但不可绝对积分”的函数可能会出现换序后结果失真的问题。

第二，只讨论可微、局部可逆、并且 Jacobian 可计算的变量代换。若变换不可逆，你无法唯一地把新变量映射回旧变量；若不可微，Jacobian 就不存在或不稳定。

第三，工程上重点放在两类常见方法：迭代积分与变量代换。更高维、解析解困难的情形，再转向 Monte Carlo 近似。

一个新手例子是：

$$
\int_{[0,1]\times[0,2]} xy\,dA
$$

这里 $dA$ 表示二维面积微元，直白理解就是“一小块面积”。可以先对 $y$ 积分：

$$
\int_0^1\left(\int_0^2 xy\,dy\right)dx
=
\int_0^1 x\left[\frac{y^2}{2}\right]_0^2 dx
=
\int_0^1 2x\,dx
=1
$$

也可以先对 $x$ 积分，结果仍然是 1。

不同区域常对应不同策略，先把这个映射关系记住会比死背公式更有用。

| 区域类型 | 典型形式 | 优先策略 | 原因 |
|---|---|---|---|
| 矩形/长方体 | $[a,b]\times[c,d]$ | 直接迭代积分 | 上下限固定，最稳定 |
| 三角形区域 | $0\le y\le x\le 1$ | 改变积分顺序或写成分段迭代 | 一层变量的上下限依赖另一层 |
| 圆盘/球体 | $x^2+y^2\le R^2$ | 极坐标/球坐标代换 | 区域对称，Jacobian 简洁 |
| 一般可测集 | 形状复杂 | 分块、变量代换或采样 | 很少能直接解析求解 |

---

## 核心机制与推导

先看 Fubini 机制。它的本质不是“背公式”，而是“把多维累加拆成分层累加”。二维情况下，你可以把区域想成很多竖条，每个竖条先在 $y$ 方向求和，再把所有竖条沿 $x$ 方向加起来。若函数绝对可积，这个分层过程不依赖你先切横条还是先切竖条。

从二维到 $n$ 维只是重复这个想法：

$$
\int_D f(x_1,\dots,x_n)\,dx_1\cdots dx_n
\]

如果区域可写成逐层上下限形式，就能把它改写成嵌套积分。每做完一层，就少一个变量，直到变成一维积分。

再看变量代换。它解决的是“区域或函数写法不方便”的问题。比如圆盘区域用直角坐标难处理，但用极坐标就更自然：

$$
x=r\cos\theta,\quad y=r\sin\theta
$$

这时不是简单把 $x,y$ 替换掉就结束了，因为一个小矩形 $dr\,d\theta$ 映射到平面后，面积不是 $dr\,d\theta$，而是

$$
dx\,dy = r\,dr\,d\theta
$$

这里的 $r$ 就是 Jacobian 产生的缩放因子。它说明：离原点越远，同样的角度变化扫过的弧越长，所以对应的小面积越大。

对极坐标变换，

$$
J=
\begin{pmatrix}
\frac{\partial x}{\partial r} & \frac{\partial x}{\partial \theta}\\
\frac{\partial y}{\partial r} & \frac{\partial y}{\partial \theta}
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta & -r\sin\theta\\
\sin\theta & r\cos\theta
\end{pmatrix}
$$

因此

$$
\det J = r
$$

于是对单位圆盘 $x^2+y^2\le 1$ 上的 $f(x,y)=x^2+y^2$，有

$$
x^2+y^2=r^2
$$

所以

$$
\int_{x^2+y^2\le 1}(x^2+y^2)\,dx\,dy
=
\int_0^{2\pi}\int_0^1 r^2\cdot r\,dr\,d\theta
=
\int_0^{2\pi}\int_0^1 r^3\,dr\,d\theta
=
2\pi\cdot \frac14
=
\frac{\pi}{2}
$$

这个例子同时说明了两件事：
1. 变量代换常让积分区域变简单。
2. Jacobian 不是可选项，而是积分正确性的必要修正。

高斯积分是重积分和变量代换结合后的经典结果。先看一维：

$$
\int_{-\infty}^{\infty} e^{-x^2}\,dx=\sqrt{\pi}
$$

推广到多维，若 $K$ 是正定矩阵，也就是“定义了一个合法椭球尺度的矩阵”，则

$$
\int_{\mathbb{R}^n}\exp\left(-\frac12 x^T K^{-1}x\right)\,dx
=
(2\pi)^{n/2}\sqrt{\det K}
$$

这个式子本质上就是先把一般椭球形二次型通过线性代换变成标准球形，再由 Jacobian 给出 $\sqrt{\det K}$ 的体积缩放。它是多元高斯分布归一化常数的来源，也是概率模型里“为什么密度能积成 1”的核心依据。

真实工程例子可以看变分推断。ELBO 中经常出现

$$
\mathbb{E}_{q(z)}[\log p(x,z)-\log q(z)]
=
\int q(z)\big(\log p(x,z)-\log q(z)\big)\,dz
$$

这就是高维积分。维度稍高时几乎不可能手算，所以工程里通常不做解析积分，而是采样近似。但你要知道：Monte Carlo 不是替代数学原理，它只是用随机样本去近似这个本来就存在的重积分。

---

## 代码实现

先给出一个最基础、可运行的二维数值积分实现。它用的是中点法，思路就是“先固定一个变量，再把另一层加完”。

```python
import math
import numpy as np

def double_integral(f, x_range, y_range, nx=200, ny=200):
    dx = (x_range[1] - x_range[0]) / nx
    dy = (y_range[1] - y_range[0]) / ny
    total = 0.0
    for i in range(nx):
        x = x_range[0] + (i + 0.5) * dx
        inner = 0.0
        for j in range(ny):
            y = y_range[0] + (j + 0.5) * dy
            inner += f(x, y) * dy
        total += inner * dx
    return total

toy_value = double_integral(lambda x, y: x * y, (0.0, 1.0), (0.0, 2.0))
assert abs(toy_value - 1.0) < 1e-2

def polar_integral_unit_disk_for_r2(nr=400, nt=400):
    dr = 1.0 / nr
    dt = 2 * math.pi / nt
    total = 0.0
    for i in range(nr):
        r = (i + 0.5) * dr
        for j in range(nt):
            theta = (j + 0.5) * dt
            value = r**2           # x^2 + y^2 = r^2
            jacobian = r           # dxdy = r dr dtheta
            total += value * jacobian * dr * dt
    return total

polar_value = polar_integral_unit_disk_for_r2()
assert abs(polar_value - math.pi / 2) < 2e-2

def flow_log_density(x, A):
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    z = A @ x
    sign, logabsdet = np.linalg.slogdet(A)
    assert sign != 0
    log_pz = -0.5 * np.dot(z, z) - 0.5 * len(z) * math.log(2 * math.pi)
    return log_pz + logabsdet

A = np.array([[2.0, 0.0], [0.0, 0.5]])
x = np.array([1.0, -1.0])
value = flow_log_density(x, A)
assert np.isfinite(value)
```

上面第一段是玩具例子，验证 $\int_0^1\int_0^2 xy\,dy\,dx=1$。第二段把单位圆盘上的积分改写成极坐标，显式乘上 Jacobian。第三段对应一个最简单的 Normalizing Flow 线性变换。

这里的 Flow 例子要抓住一个点：如果变换是 $z=T(x)$，而基分布 $p(z)$ 已知，那么数据空间中的密度不是直接拿来用，而要加上体积修正项：

$$
\log p(x)=\log p(z)+\log\left|\det\frac{\partial z}{\partial x}\right|
$$

工程上常用 `np.linalg.slogdet`，而不是先算 `det` 再取对数。原因很直接：高维矩阵的行列式可能非常大或非常小，直接算容易数值上溢或下溢，`slogdet` 更稳定。

如果把这套结构抽象一下，迭代积分和 Flow 密度计算其实很像：
1. 先选一个便于处理的坐标系或隐空间。
2. 在那个空间里计算函数值或基密度。
3. 用 Jacobian 修正尺度变化。

---

## 工程权衡与常见坑

重积分一旦进入工程，问题就不再只是“公式对不对”，而是“算得稳不稳、快不快、误差可不可控”。

第一个权衡是精度和计算量。规则网格积分在二维、三维时很直观，但维度一高，网格数会指数增长。比如每维取 100 个点，二维是 $10^4$ 个格子，六维就变成 $10^{12}$ 个格子，基本不可用。这叫**维度灾难**，直白说就是“维度一高，枚举法很快爆炸”。

第二个常见坑是忘记 Jacobian 的绝对值。行列式为负只表示方向翻转，例如镜像变换会把坐标朝向反过来，但面积和体积不能是负的，所以积分公式里必须写 $\left|\det J\right|$。

第三个坑是使用不可逆变换。Normalizing Flow 要计算密度，就必须能从 $x$ 唯一映射到 $z$，否则同一个点的概率质量会混在一起，密度公式无法成立。这也是为什么很多 Flow 结构专门设计成可逆层，而不是随便堆叠一个神经网络。

第四个坑来自 Monte Carlo。它在高维上往往比规则网格更现实，但方差可能很大。以变分推断为例，如果你只用 5 个样本估计梯度，单步更新会抖得很厉害；如果增加到 500 个样本，估计更稳定，但计算成本明显上升。这里没有免费午餐，只能在方差和吞吐量之间折中。

| 常见坑 | 后果 | 规避方法 |
|---|---|---|
| 省略 $\left|\det J\right|$ | 面积/体积修正错误，密度错 | 明确使用绝对值或 `slogdet` |
| 变换不可逆 | 无法正确写出密度 | 选择可逆层或分块可逆结构 |
| 直接算 `det` 再 `log` | 数值上溢/下溢 | 使用稳定的对数行列式实现 |
| 样本太少 | Monte Carlo 方差大，训练震荡 | 增大样本数、batch 或用控制变量 |
| 积分顺序随意改 | 在非绝对可积情形下可能错 | 先确认绝对可积条件 |

现实工程里有两个监控指标值得单独强调。

第一，**Jacobian 的条件数**。条件数可以直白理解为“这个变换把某些方向拉得过长、压得过扁的程度”。条件数很大时，逆变换不稳定，训练容易出现数值问题。

第二，**梯度方差**。尤其在变分推断和扩散模型采样估计里，平均损失下降并不代表训练稳定；如果梯度方差很大，优化器会在噪声中乱跳。工程上通常要同时看 loss 曲线、梯度范数和样本方差估计。

---

## 替代方案与适用边界

不是所有高维积分都适合硬做变量代换，也不是所有问题都该先写成迭代积分。判断策略时，重点看区域形状、维度和变换结构。

如果区域规则、维度低、上下限明确，优先用迭代积分。它最透明，调试最简单，适合教学、验证和小规模数值计算。

如果区域有明显几何结构，比如圆盘、球体、椭球、仿射拉伸区域，优先考虑变量代换。因为这时 Jacobian 往往有漂亮的解析式，能把问题一下变成标准形式。

如果维度高、区域复杂、函数只支持采样或前向计算，Monte Carlo 更实用。它的核心做法很直白：按某个分布随机取样，计算函数值，然后取平均，用样本平均逼近期望。精度收敛慢，但不怕高维到无法离散化。

如果你不仅要算积分，还要学习一个复杂分布的显式密度，那么 Normalizing Flow 是更强的结构化方案。它不是“专门算积分”的工具，而是把变量代换嵌入模型参数化中，让高维密度计算可训练、可微分。

可以把选择过程简化成一条判断链：

问题结构明确且低维 $\rightarrow$ 先试迭代积分。  
区域几何对称明显 $\rightarrow$ 试变量代换。  
维度高且解析解困难 $\rightarrow$ 用 Monte Carlo 或 Quasi-Monte Carlo。  
需要显式密度且要求可逆映射 $\rightarrow$ 用 Normalizing Flow。

| 方法 | 适用条件 | 优点 | 缺点 |
|---|---|---|---|
| 迭代积分 | 低维、区域规则 | 直观、易验证 | 高维爆炸，非规则区域麻烦 |
| 变量代换 | 存在好坐标系 | 可大幅简化区域与函数 | 依赖可逆可微变换和 Jacobian |
| Monte Carlo | 高维、只需近似 | 不怕高维、实现通用 | 方差大、收敛慢 |
| Quasi-Monte Carlo | 中高维、函数较平滑 | 比普通采样更稳定 | 实现和理论门槛稍高 |
| Normalizing Flow | 需要密度建模 | 可训练、可逆、可求密度 | 结构设计复杂，数值稳定性要求高 |

一个新手直观例子是：如果区域长得像一块弯曲的饼干，手工写出积分上下限很痛苦，这时你不一定非要找完美代换。完全可以先在包围盒里随机采样，判断点是否落在区域内，再估计区域积分。这种方法未必最精确，但往往是第一版工程实现的正确起点。

---

## 参考资料

1. Fubini 定理，适合理解“为什么多维积分可以拆成逐维积分，以及何时可以换顺序”：Wikipedia, Fubini's theorem, https://en.wikipedia.org/wiki/Fubini%27s_theorem  
2. 多元积分中的变量代换与 Jacobian，适合建立“为什么要乘行列式绝对值”的几何直觉：LibreTexts, Substitutions in Multiple Integrals, https://math.libretexts.org/Bookshelves/Calculus/Map%3A_University_Calculus_%28Hass_et_al%29/14%3A_Multiple_Integrals/14.8%3A_Substitutions_in_Multiple_Integrals  
3. 多维高斯积分公式，适合理解正态分布归一化常数与 $\sqrt{\det K}$ 的来源：PlanetMath, Multidimensional Gaussian Integral, https://planetmath.org/multidimensionalgaussianintegral  
4. 变分推断与高维期望、Normalizing Flow 与对数 Jacobian，适合理解本文中的真实工程场景：Stan Reference Manual, Variational Inference, https://mc-stan.org/docs/ ; Emergent Mind, Normalizing Flows, https://www.emergentmind.com/topics/normalizing-flows
