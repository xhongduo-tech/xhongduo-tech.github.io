## 核心结论

随机变量的变换，本质上是在问一件事：同一批概率质量，在坐标系被拉伸、压缩、扭曲后，新的密度应该怎样改写。

单变量情形里，若 $Y=g(X)$，且 $g$ 单调可微，那么密度不是“把 $x$ 换成 $y$”这么简单，而是必须乘上一个缩放因子：

$$
f_Y(y)=f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|
$$

这项导数绝对值表示“长度被压缩或拉伸了多少”。白话说，同样的概率，挤进更短的区间，密度就更高；铺到更长的区间，密度就更低。

多变量情形里，长度缩放变成体积缩放，对应 Jacobian 行列式。Jacobian 可以理解为“局部线性近似下的放缩矩阵”。公式变成：

$$
f_Y(y)=f_X(g^{-1}(y))\cdot\left|\det J_{g^{-1}}(y)\right|
$$

正则化流（Normalizing Flow）就是把这个规则做成多层可逆变换链。先从简单分布 $z\sim p_Z$ 出发，再经过可逆映射 $x=f(z)$ 得到复杂分布。每一层都修正一次密度，最终得到：

$$
\log p_X(x)=\log p_Z(z)-\sum_{i=1}^{L}\log\left|\det J_{f_i}(h_{i-1})\right|
$$

若改写成逆变换 $z=f^{-1}(x)$，也常写成：

$$
\log p_X(x)=\log p_Z(z)+\sum_{i=1}^{L}\log\left|\det J_{f_i^{-1}}(h_i)\right|
$$

这两个写法等价，差别只在你沿正向还是逆向记账。

---

## 问题定义与边界

这类问题处理的是“已知 $X$ 的分布，求 $Y=g(X)$ 的分布”。

边界先讲清楚：

| 场景 | 需要的条件 | 结论是否能直接套公式 |
| --- | --- | --- |
| 单变量 | $g$ 单调、可微 | 可以 |
| 单变量但不单调 | 需分段求逆并求和 | 不能直接一行写完 |
| 多变量 | $g$ 可逆、可微、Jacobian 存在 | 可以 |
| 不可逆映射 | 多个点压到同一点 | 不能当普通密度变换处理 |

最小玩具例子是线性变换。设 $X\sim \mathcal U[0,1]$，即在 $[0,1]$ 上均匀分布，密度恒为 1。令

$$
Y=2X
$$

则逆变换为 $g^{-1}(y)=y/2$，导数为 $1/2$，所以

$$
f_Y(y)=f_X(y/2)\cdot \frac{1}{2}
$$

又因为只有当 $y/2\in[0,1]$ 时原密度非零，所以 $y\in[0,2]$。因此：

$$
f_Y(y)=
\begin{cases}
0.5, & y\in[0,2] \\
0, & \text{otherwise}
\end{cases}
$$

这说明区间长度从 1 拉到 2，密度就从 1 降到 0.5，概率总量仍守恒：

$$
\int_0^2 0.5\,dy = 1
$$

线性变换的一般形式也值得记住。若 $Y=a+bX,\ b\neq 0$，则

$$
f_Y(y)=\frac{1}{|b|}f_X\left(\frac{y-a}{b}\right)
$$

其中定义域必须跟着变。很多错误不是公式错，而是忘了支持集，也就是“哪些 $y$ 根本不可能出现”。

---

## 核心机制与推导

为什么会多出导数或行列式？因为密度不是概率本身，而是“单位长度”或“单位体积”上的概率浓度。

单变量里，对一个很小区间 $[y,y+dy]$，对应回原空间是 $[x,x+dx]$。如果局部近似线性，那么

$$
dy \approx g'(x)\,dx
\quad\Rightarrow\quad
dx \approx \left|\frac{d}{dy}g^{-1}(y)\right|dy
$$

概率守恒给出：

$$
f_Y(y)\,dy = f_X(x)\,dx
$$

两边同时除以 $dy$，就得到变量替换公式。

多变量完全同理，只是“小区间长度”换成“小体积元”。设 $y=g(x)$，局部线性近似为：

$$
dy \approx J_g(x)\,dx
$$

体积缩放因子不是导数，而是行列式绝对值：

$$
|dy| \approx |\det J_g(x)|\,|dx|
$$

因此：

$$
f_Y(y)=f_X(x)\cdot\frac{1}{|\det J_g(x)|}
=f_X(g^{-1}(y))\cdot |\det J_{g^{-1}}(y)|
$$

这就是多变量变量替换公式。

正则化流只是把这个规则链式叠起来。设

$$
h_0=z,\quad h_i=f_i(h_{i-1}),\quad x=h_L
$$

则每一步都做一次密度修正：

$$
p_{h_i}(h_i)=p_{h_{i-1}}(h_{i-1})\cdot \left|\det J_{f_i^{-1}}(h_i)\right|
$$

取对数后，乘法变加法：

$$
\log p(x)=\log p(z)+\sum_{i=1}^L \log \left|\det J_{f_i^{-1}}(h_i)\right|
$$

这就是流模型训练时常见的“base log-prob + sum log-det”。

真实工程例子可以看 RealNVP。它把输入拆成两部分 $x=[x_a,x_b]$，只变换后一部分：

$$
y_a=x_a
$$

$$
y_b=x_b\odot \exp(s(x_a)) + t(x_a)
$$

这里 $s,t$ 由神经网络输出，$\odot$ 表示逐元素乘法。因为 $y_a$ 不动，$y_b$ 只依赖 $x_a$ 和自身，所以 Jacobian 是三角矩阵。三角矩阵的行列式，白话说就是“只乘对角线元素”，因此：

$$
\det J = \prod_j \exp(s_j(x_a))
\quad\Rightarrow\quad
\log |\det J|=\sum_j s_j(x_a)
$$

这一步极其关键。一般 $d\times d$ 矩阵行列式代价常是 $O(d^3)$，而三角结构把它降成 $O(d)$。流模型能落地，靠的就是这种结构化设计，而不是单纯把公式抄进代码。

---

## 代码实现

下面先给一个玩具实现：用单变量线性变换验证变量替换公式，再给一个二维仿射耦合层的最小实现。

```python
import math

def uniform01_pdf(x: float) -> float:
    return 1.0 if 0.0 <= x <= 1.0 else 0.0

def transformed_pdf_y_eq_2x(y: float) -> float:
    # Y = 2X, X ~ Uniform[0, 1]
    x = y / 2.0
    jacobian_inv = 0.5  # d/dy (y/2)
    return uniform01_pdf(x) * jacobian_inv

# 点值检查
assert abs(transformed_pdf_y_eq_2x(0.5) - 0.5) < 1e-12
assert abs(transformed_pdf_y_eq_2x(1.5) - 0.5) < 1e-12
assert transformed_pdf_y_eq_2x(-0.1) == 0.0
assert transformed_pdf_y_eq_2x(2.1) == 0.0

# 数值积分检查：积分应约等于 1
step = 0.0005
area = 0.0
y = -1.0
while y <= 3.0:
    area += transformed_pdf_y_eq_2x(y) * step
    y += step

assert abs(area - 1.0) < 1e-2
```

这个例子说明两件事：

1. 变换后的密度值变了。
2. 但积分仍为 1，说明概率没有凭空产生或消失。

再看一个接近工程实现的二维仿射耦合层。这里不用深度学习框架，只用 Python 演示机制：

```python
import math

def net(xa):
    # 假设 xa 是长度为 1 的列表
    s = [0.2 * xa[0]]
    t = [xa[0] + 1.0]
    return s, t

def affine_coupling_forward(x):
    # mask = [1, 0]
    xa = [x[0]]
    xb = [x[1]]

    s, t = net(xa)
    yb = [xb[0] * math.exp(s[0]) + t[0]]
    ya = xa[:]

    log_det = s[0]
    y = [ya[0], yb[0]]
    return y, log_det

def affine_coupling_inverse(y):
    ya = [y[0]]
    yb = [y[1]]

    s, t = net(ya)
    xb = [(yb[0] - t[0]) * math.exp(-s[0])]
    xa = ya[:]

    log_det_inv = -s[0]
    x = [xa[0], xb[0]]
    return x, log_det_inv

x = [2.0, 3.0]
y, log_det = affine_coupling_forward(x)
x_recovered, log_det_inv = affine_coupling_inverse(y)

assert abs(x_recovered[0] - x[0]) < 1e-12
assert abs(x_recovered[1] - x[1]) < 1e-12
assert abs(log_det + log_det_inv) < 1e-12
```

这段代码对应的工程含义是：

| 组件 | 作用 | 为什么重要 |
| --- | --- | --- |
| `mask` | 固定一部分维度不动 | 保证变换易逆 |
| `net(xa)` | 预测缩放 `s` 和平移 `t` | 提供非线性表达能力 |
| `log_det` | 累积密度修正项 | 用于精确似然训练 |
| 交替掩码 | 让不同维度轮流被更新 | 避免有些维度永远不变 |

真实训练时，会把很多层耦合层、置换层、归一化层串起来。前向做采样：从高斯生成数据。反向做似然：把数据压回高斯，再加上所有 log-det。

---

## 工程权衡与常见坑

流模型的优点很明确：可采样、可求显式似然、前后向关系清楚。但代价也很明确：结构必须为了“可逆”和“可计算 Jacobian”服务。

| 因素 | 优点 | 挑战 |
| --- | --- | --- |
| 三角/分块 Jacobian | $\log|\det J|$ 可快速计算 | 单层表达力有限，需要堆叠很多层 |
| 完全自由的可逆映射 | 表达空间更大 | 行列式和逆映射都可能很贵 |
| 显式似然训练 | 目标函数清晰 | 对数行列式数值稳定性要求高 |
| 忽略 log-det | 实现看起来简单 | 密度估计直接有偏 |

常见坑主要有五类。

第一，忘记支持集。  
例如 $Y=X^2$ 且 $X\sim \mathcal U[-1,1]$，如果你直接套单调公式就会错，因为 $x^2$ 在整个区间上不是单调的，必须拆成 $x=\sqrt y$ 和 $x=-\sqrt y$ 两支再求和。

第二，把“可采样”误当成“可算概率”。  
很多生成模型会采样，但未必能给出精确密度。流模型的特点不是只会生成，而是能同时做生成和似然评估。

第三，log-det 符号写反。  
如果定义的是正向映射 $x=f(z)$，通常写：

$$
\log p_X(x)=\log p_Z(z)-\log|\det J_f(z)|
$$

如果写成逆映射 $z=f^{-1}(x)$，则是加号。训练代码里这一点经常导致 loss 全错但程序还能跑。

第四，缩放项失控。  
在仿射耦合层里，$\exp(s)$ 负责缩放。如果 $s$ 太大，会带来数值爆炸；太小则趋近退化。工程里通常会对 `s` 做裁剪、限制范围，或通过特殊参数化稳定训练。

第五，误以为“可逆”就一定好训。  
可逆只解决了数学可定义，不自动解决优化难题。层数过深、底层分布选得太简单、数据预处理不合适，都会让模型虽然理论正确，实际却拟合很差。

---

## 替代方案与适用边界

当目标是“既要能采样，又要有显式似然”，流模型非常合适。但它不是所有生成问题的默认最优解。

| 架构 | 采样 | 似然 | 主要限制 |
| --- | --- | --- | --- |
| 正则化流 | 并行或近并行 | 显式 | 需可逆可微，结构受限 |
| VAE | 容易 | 近似，通常优化 ELBO | 不是精确似然 |
| GAN | 强采样能力 | 通常无显式似然 | 训练可能不稳定 |
| 自回归流（MAF/IAF） | 常需顺序执行 | 显式 | 高维时采样或推理速度受限 |

VAE 是“通过潜变量近似数据分布”的方法，白话说是先学一个压缩和解压系统；GAN 是“生成器和判别器对抗训练”的方法，白话说是让生成器不断骗过评审器。它们都很重要，但和流的取舍不同。

流模型的适用边界主要有三点：

1. 需要精确或至少可追踪的似然，例如密度估计、异常检测、某些贝叶斯推断任务。
2. 能接受可逆结构带来的设计限制，例如通道拆分、耦合层、可逆卷积。
3. 希望基分布简单，复杂性由变换承担。

再看一个真实工程例子。假设要建一个网络流量异常检测器，输入是几十维连续特征。若用流模型训练正常流量的密度，那么新样本到来时可以直接算 $\log p(x)$。概率极低的样本可判为异常。这个场景里，“能算概率”比“生成图像好看”更重要，因此流比 GAN 更贴合目标。

但如果任务是超高分辨率图像生成，且最关注感知质量，不太关心精确似然，那么扩散模型或 GAN 往往更常见。原因不是变量替换公式失效，而是工程目标变了。

---

## 参考资料

- Transformations of Random Variables, LibreTexts: 单变量与多变量变量替换公式  
- Normalizing Flows Explained, Emergent Mind: 流模型定义、链式 log-det 记账  
- Triangular Normalizing Flows Overview, Emergent Mind: 三角 Jacobian 与自回归结构  
- Machine Learning for Mechanical Engineering, UMD: RealNVP 与仿射耦合层讲解  
- Normalizing Flow Models, Change Zakram: RealNVP 的 Jacobian 结构与 log-det 推导  
- UW 课程讲义 Normalizing Flows: 与 VAE 等生成模型的对比
