## 核心结论

泛函导数描述的是“函数对函数的变化率”，可以把它理解成无限维空间里的梯度。普通导数研究一个数变了以后另一个数怎么变，泛函导数研究一整条函数曲线变了以后一个标量目标怎么变。

泛函是“输入为函数、输出为数”的映射。例如

$$
J[u]=\int_0^1 u(x)^2\,dx
$$

这里的输入不是某个数 $x$，而是整条函数 $u(x)$；输出是一个实数 $J[u]$。这个例子可以看成“函数版的平方和”。如果把区间上的 $u(x)$ 看成无穷多个坐标，那么 $J[u]$ 就类似有限维里的 $\sum_i u_i^2$。

泛函导数通过一阶变分定义：

$$
\delta J[u;v]=\left.\frac{d}{d\epsilon}J[u+\epsilon v]\right|_{\epsilon=0}
$$

其中 $v(x)$ 是扰动方向，$\epsilon$ 是一个很小的标量。若一阶变化可以写成

$$
\delta J[u;v]=\int_a^b g(x)v(x)\,dx
$$

则称

$$
g(x)=\frac{\delta J}{\delta u(x)}
$$

为 $J$ 关于 $u$ 的泛函导数。

对玩具例子

$$
J[u]=\int_0^1 u(x)^2\,dx
$$

有

$$
J[u+\epsilon v]=\int_0^1 (u+\epsilon v)^2\,dx
=\int_0^1 (u^2+2\epsilon uv+\epsilon^2v^2)\,dx
$$

所以

$$
\delta J[u;v]=\int_0^1 2u(x)v(x)\,dx
$$

因此

$$
\frac{\delta J}{\delta u}=2u
$$

新手版理解是：如果整条曲线整体抬高一点，目标值的变化量会按每个位置上的 $u(x)$ 大小成比例变化；$u(x)$ 越大，该位置对积分目标越敏感。

普通导数与泛函导数的对照如下：

| 对象 | 普通导数 | 泛函导数 |
|---|---|---|
| 输入 | 数或有限维向量 | 函数 |
| 输出 | 数 | 数 |
| 变化方向 | $\Delta x$ 或向量方向 | 扰动函数 $v(x)$ |
| 梯度形式 | $\nabla f(x)$ | $\frac{\delta J}{\delta u(x)}$ |
| 典型问题 | 参数优化 | 曲线、场、分布、轨迹优化 |

对积分型泛函

$$
J[u]=\int_a^b L(x,u,u')\,dx
$$

做一阶变分并令其为 0，会得到 Euler-Lagrange 方程：

$$
\frac{\partial L}{\partial u}-\frac{d}{dx}\frac{\partial L}{\partial u'}=0
$$

这是变分法的核心结论。连续优化、最优控制、物理中的最小作用量原理、机器学习中的分布优化和梯度流，都可以用这套语言表达。

---

## 问题定义与边界

讨论变分法之前，必须先区分几个对象。很多错误来自把“函数”“函数值”“泛函值”混为一谈。

| 对象 | 含义 | 维度/类型 | 作用 |
|---|---|---|---|
| $x$ | 自变量 | 标量 | 表示位置、时间或空间坐标 |
| $u(x)$ | 待优化函数 | 函数 | 被优化的曲线、轨迹或场 |
| $v(x)$ | 扰动方向 | 函数 | 表示允许怎样改变 $u$ |
| $\epsilon$ | 扰动强度 | 标量 | 把函数变化转成一元变化 |
| $u+\epsilon v$ | 被扰动后的函数 | 函数 | 用于定义一阶变分 |
| $J[u]$ | 泛函 | 函数到数的映射 | 优化目标 |
| $L(x,u,u')$ | 拉格朗日量 | 标量函数 | 描述积分目标的局部代价 |

变分法的基本动作是构造

$$
u_\epsilon(x)=u(x)+\epsilon v(x)
$$

然后研究 $J[u_\epsilon]$ 在 $\epsilon=0$ 附近的一阶变化。这样做的好处是：原本“对函数求导”的问题，被转化成了“对标量参数 $\epsilon$ 求导”的问题。

边界条件必须明确，因为它决定推导中边界项是否保留。

| 边界类型 | 条件 | 对扰动的要求 | 结果 |
|---|---|---|---|
| 固定边界 | $u(a),u(b)$ 已知 | $v(a)=v(b)=0$ | 边界项消失 |
| 自由边界 | $u(a)$ 或 $u(b)$ 未固定 | 对应端点 $v$ 可非零 | 产生自然边界条件 |
| 周期边界 | $u(a)=u(b)$ | 扰动也满足周期性 | 首尾边界项配对抵消或合并 |
| 受约束边界 | 端点满足某个约束 | 扰动必须保持可行 | 需要加入约束或乘子 |

固定端点问题中，$u(a)$ 和 $u(b)$ 不允许变动，所以 $v(a)=v(b)=0$。新手版理解是：只允许中间弯曲，不能动两头。

真实工程例子中，轨迹优化经常有固定边界。例如无人车规划一段速度曲线 $v(t)$，起点速度和终点速度可能由安全规则或道路条件规定。这时优化器只能调整中间的速度变化，不能随便改变起点和终点。若把这些端点也当成可变，就会得到另一个问题。

---

## 核心机制与推导

变分法的核心机制是：通过 $u+\epsilon v$ 的一阶展开，把“函数的变化”转化为“标量对参数的导数”。

设

$$
J[u]=\int_a^b L(x,u,u')\,dx
$$

其中 $u'$ 表示 $u$ 对 $x$ 的导数。考虑扰动函数

$$
u_\epsilon=u+\epsilon v
$$

则

$$
u_\epsilon'=u'+\epsilon v'
$$

代入泛函：

$$
J[u+\epsilon v]=\int_a^b L(x,u+\epsilon v,u'+\epsilon v')\,dx
$$

对 $\epsilon$ 求导，并在 $\epsilon=0$ 处取值：

$$
\delta J[u;v]
=
\left.\frac{d}{d\epsilon}
\int_a^b L(x,u+\epsilon v,u'+\epsilon v')\,dx
\right|_{\epsilon=0}
$$

在足够光滑的条件下，可以把求导放入积分号内：

$$
\delta J[u;v]
=
\int_a^b
\left(
\frac{\partial L}{\partial u}v
+
\frac{\partial L}{\partial u'}v'
\right)dx
$$

这里出现了 $v'$。但泛函导数希望写成 $\int g(x)v(x)\,dx$ 的形式，所以要对第二项做分部积分：

$$
\int_a^b \frac{\partial L}{\partial u'}v'\,dx
=
\left[\frac{\partial L}{\partial u'}v\right]_a^b
-
\int_a^b
\frac{d}{dx}\left(\frac{\partial L}{\partial u'}\right)v\,dx
$$

代回得到完整一阶变分：

$$
\delta J
=
\int_a^b
\left(
\frac{\partial L}{\partial u}
-
\frac{d}{dx}\frac{\partial L}{\partial u'}
\right)v(x)\,dx
+
\left[\frac{\partial L}{\partial u'}v\right]_a^b
$$

这条公式把变化拆成两部分：曲线内部怎么变，以及边界怎么动。

| 步骤 | 操作 | 得到的结果 |
|---|---|---|
| 1 | 写出 $J[u]$ | 明确泛函形式 |
| 2 | 替换 $u$ 为 $u+\epsilon v$ | 得到 $J[u+\epsilon v]$ |
| 3 | 对 $\epsilon$ 求导 | 得到一阶变分 |
| 4 | 对含 $v'$ 的项分部积分 | 分出体项和边界项 |
| 5 | 加入边界条件 | 判断边界项是否消失 |
| 6 | 对所有允许 $v$ 令 $\delta J=0$ | 得到 Euler-Lagrange 方程 |

在固定端点下，$v(a)=v(b)=0$，所以边界项

$$
\left[\frac{\partial L}{\partial u'}v\right]_a^b
$$

消失。若驻值条件要求对任意允许扰动 $v$ 都有 $\delta J=0$，则必须有

$$
\frac{\partial L}{\partial u}
-
\frac{d}{dx}\frac{\partial L}{\partial u'}
=0
$$

这就是 Euler-Lagrange 方程。

一个玩具例子是最短曲线问题。若曲线写成 $u(x)$，长度泛函为

$$
J[u]=\int_a^b \sqrt{1+u'(x)^2}\,dx
$$

此时 $L=\sqrt{1+u'^2}$，它不显含 $u$，所以 $\frac{\partial L}{\partial u}=0$。Euler-Lagrange 方程会推出某个与 $u'$ 相关的量为常数，最后得到直线。这说明“直线最短”不是只靠直觉，也能由变分法推出。

---

## 代码实现

连续泛函在程序里不能直接表示为“无穷维对象”。常见做法是把区间离散成网格，把函数值变成数组。此时泛函导数会近似成普通向量梯度。

以

$$
J[u]=\int_0^1 u(x)^2\,dx
$$

为例，离散化后可写成

$$
J(u)\approx \sum_i u_i^2\Delta x
$$

连续泛函导数是

$$
\frac{\delta J}{\delta u}=2u
$$

而离散向量梯度是

$$
\nabla J_i = 2u_i\Delta x
$$

注意这里多了 $\Delta x$。这是因为离散目标是一个有限维求和，连续目标是一个积分。两者相关，但不是同一个对象。

最小示例表格如下：

| 变量 | 取值 | 含义 |
|---|---:|---|
| $u$ | $[1,2]$ | 两个网格点上的函数值 |
| $v$ | $[1,-1]$ | 扰动方向 |
| $\Delta x$ | $0.5$ | 每段网格宽度 |
| $J(\epsilon)$ | $0.5[(1+\epsilon)^2+(2-\epsilon)^2]$ | 扰动后的目标 |
| $\left.\frac{dJ}{d\epsilon}\right|_0$ | $-1$ | 沿 $v$ 的方向导数 |

可运行 Python 代码如下：

```python
import numpy as np

dx = 0.5
u = np.array([1.0, 2.0])
v = np.array([1.0, -1.0])

def J(arr):
    return np.sum(arr ** 2) * dx

eps = 1e-6

# 数值方向导数：用中心差分近似 d/dε J[u + εv]
numeric = (J(u + eps * v) - J(u - eps * v)) / (2 * eps)

# 连续泛函导数为 2u；离散内积要乘 dx
functional_derivative = 2 * u
theory_directional = np.sum(functional_derivative * v) * dx

# 离散向量梯度为 2u dx；与 v 做普通点积
discrete_gradient = 2 * u * dx
discrete_directional = np.dot(discrete_gradient, v)

assert abs(numeric - theory_directional) < 1e-9
assert abs(numeric - discrete_directional) < 1e-9
assert abs(numeric + 1.0) < 1e-9

print(numeric)
```

这段代码验证的是一阶变分公式，而不是直接求解析解。新手版理解是：先把曲线切成很多小段，再把函数当数组处理；数组梯度可以用来近似连续泛函导数，但必须记住积分权重 $\Delta x$。

真实工程例子是再生制动优化。车辆速度轨迹 $v(t)$ 可以看成一个函数，目标可能是最大化回收电量，同时满足安全距离、舒适性和电机功率约束。目标函数通常包含时间积分，例如能耗、加速度惩罚、制动效率等。若直接在连续轨迹上推导，可以得到 Euler-Lagrange 方程或伴随方程；若落到工程实现，通常会离散成时间网格，然后交给数值优化器求解。

---

## 工程权衡与常见坑

连续理论和离散实现不是同一个对象。连续形式中的

$$
\frac{\delta J}{\delta u}
$$

是一个函数；离散实现中的 $\nabla J$ 是一个向量。它们之间通常通过网格权重、积分格式和内积定义联系起来。直接把离散梯度当成连续泛函导数，会在尺度上出错。

边界项也不能随手丢掉。完整公式里有

$$
\left[\frac{\partial L}{\partial u'}v\right]_a^b
$$

固定端点时它确实为 0，因为 $v(a)=v(b)=0$。但自由边界问题中，端点扰动不一定为 0，这一项会给出自然边界条件。若忽略它，固定端点下可能还能凑出正确结果，自由边界下会直接错掉。新手版理解是：有些题边界本身就是未知量，不能默认它们不重要。

| 错误做法 | 后果 | 正确处理 |
|---|---|---|
| 把泛函导数当普通偏导 | 忽略扰动函数和积分结构 | 先写 $\delta J[u;v]$ |
| 不说明边界条件 | 不知道边界项能否消失 | 明确固定、自由或周期边界 |
| 忽略 $[\frac{\partial L}{\partial u'}v]_a^b$ | 自由边界问题推导错误 | 保留边界项再代入条件 |
| 把离散梯度直接等同于连续导数 | 少掉或多出 $\Delta x$ 权重 | 区分连续内积和离散内积 |
| 不检查光滑性 | 分部积分、求导换积分可能不合法 | 说明可微性、可积性假设 |
| 任意选择扰动 $v$ | 可能走出可行集合 | 扰动必须满足约束和边界 |

函数空间是另一个常见隐含前提。函数空间是“允许哪些函数参与问题”的集合。例如要求 $u$ 可导、平方可积、满足边界条件，这些都会改变问题本身。若 $J[u]$ 中含有 $u'$，通常至少要让 $u$ 具有弱导数或普通导数；若目标里有积分，通常还要保证相关项可积。

在机器学习里也有类似问题。分布优化中，变量可能是概率密度 $\rho(x)$。概率密度不是任意函数，它必须满足

$$
\rho(x)\ge 0,\qquad \int \rho(x)\,dx=1
$$

因此扰动方向也不能随便选，通常要满足质量守恒条件。否则推出来的“梯度方向”可能根本不是合法概率分布的变化方向。

---

## 替代方案与适用边界

变分法不是所有优化问题的首选工具。当问题本来就是有限维参数优化时，普通梯度法更简单；当变量天然是连续函数、轨迹、场或概率分布时，泛函导数更自然。

| 问题类型 | 推荐方法 | 优点 | 限制 |
|---|---|---|---|
| 少量标量参数优化 | 普通微积分或梯度下降 | 简单直接 | 不适合连续场推导 |
| 固定网格上的数组优化 | 自动微分、数值优化 | 工程实现方便 | 依赖离散化质量 |
| 连续曲线或轨迹优化 | 变分法、最优控制 | 能保留连续结构 | 推导要求较高 |
| PDE 约束优化 | 伴随法、有限元优化 | 适合大规模场问题 | 实现复杂 |
| 概率分布优化 | 泛函导数、Wasserstein 梯度流 | 适合分布动态 | 需要概率和测度基础 |
| 深度学习模型训练 | 反向传播、自动微分 | 工具链成熟 | 通常不需要手推变分 |

轨迹优化是典型适用场景。若速度 $v(t)$ 是连续时间上的函数变量，目标是最小化能耗

$$
J[v]=\int_0^T C(t,v(t),v'(t))\,dt
$$

并且还要满足舒适性和动力学约束，那么变分法或最优控制语言更合适。它能直接表达“整条轨迹怎么变”。

但如果问题已经被建模成 100 个离散时间点上的变量

$$
[v_1,v_2,\dots,v_{100}]
$$

并且目标函数也是普通数组计算，那么直接使用普通优化器、自动微分或二次规划通常更直接。新手版理解是：连续问题用连续语言，离散问题用离散语言。

伴随法可以看成变分思想在约束系统中的工程化形式。它常用于 PDE 约束优化，例如流体形状优化、结构优化、天气模型参数反演等。直接对每个设计变量求导成本很高，而伴随法可以用一次或少数几次方程求解得到梯度。

有限元优化则把连续问题先投影到有限维基函数空间中。它比简单网格数组更结构化，适合复杂几何和边界条件。自动微分适合已经离散好的程序，但它不自动替你解决“连续模型是否正确”“边界条件是否合理”“离散化是否收敛”这些问题。

何时不用泛函导数：第一，变量本来就是少量参数；第二，只关心工程实现而不关心连续极限；第三，目标函数不可微或主要由离散逻辑组成；第四，已有成熟优化器可以直接处理当前形式。泛函导数的价值不是让简单问题复杂化，而是在变量本质上是函数时提供准确语言。

---

## 参考资料

基础概念：

1. [MIT OCW: Waves and Imaging, Calculus of Variations, Functional Derivatives](https://ocw.mit.edu/courses/18-325-topics-in-applied-mathematics-waves-and-imaging-fall-2015/resources/mit18_325f15_appendix_a/)
2. [OpenLearn: Introduction to the calculus of variations](https://www.open.edu/openlearn/science-maths-technology/introduction-the-calculus-variations/content-section-0)

推导与数学背景：

3. [Duke: Calculus of Variations and Notions of Convexity](https://sites.math.duke.edu/DOmath/DOmath2021/CalculusOfVariationsReport.pdf)
4. [Wikipedia: Euler-Lagrange equation](https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation)

工程应用：

5. [Penn State: Optimizing Regenerative Braking: A Variational Calculus Approach](https://pure.psu.edu/en/publications/optimizing-regenerative-braking-a-variational-calculus-approach/)
6. [MIT OCW: Variational Methods in Mechanics](https://ocw.mit.edu/courses/2-094-finite-element-analysis-of-solids-and-fluids-ii-spring-2011/pages/lecture-notes/)
