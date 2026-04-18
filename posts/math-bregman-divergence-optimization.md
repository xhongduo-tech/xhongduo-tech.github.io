## 核心结论

Bregman 散度是由凸函数诱导的非欧氏误差度量。它的定义是：

$$
D_\phi(x,y)=\phi(x)-\phi(y)-\langle \nabla\phi(y),x-y\rangle
$$

其中，凸函数是指函数图像上任意两点连线都不低于函数图像本身的函数；直观说，它是“向上弯”的函数。$\nabla\phi(y)$ 是 $\phi$ 在 $y$ 处的梯度，也就是多维函数的一阶变化方向。内积 $\langle a,b\rangle$ 可以理解为两个向量在方向上的乘积累加。

Bregman 散度不是普通距离。它通常不对称，也不一定满足三角不等式。它度量的是：用 $\phi$ 在 $y$ 处的一阶线性近似去估计 $x$ 时，低估了多少。换句话说，它是一阶泰勒展开的剩余误差。

当

$$
\phi(x)=\frac{1}{2}\|x\|_2^2
$$

时，有

$$
D_\phi(x,y)=\frac{1}{2}\|x-y\|_2^2
$$

这说明平方欧氏距离只是 Bregman 散度的一个特例。

| 对象 | 形式 | 是否对称 | 典型用途 |
|---|---:|---:|---|
| 欧氏距离 | $\|x-y\|_2$ | 是 | 普通连续变量优化 |
| 平方欧氏 Bregman 散度 | $\frac{1}{2}\|x-y\|_2^2$ | 是 | 梯度下降、最小二乘 |
| KL 散度 | $\sum_i p_i\log\frac{p_i}{q_i}$ | 否 | 概率分布、信息论、熵正则 |
| 一般 Bregman 散度 | $\phi(x)-\phi(y)-\langle\nabla\phi(y),x-y\rangle$ | 通常否 | 非欧氏优化、镜像下降 |

核心结论是：Bregman 散度的价值不在公式本身，而在几何匹配。选对 $\phi$，优化算法就能更贴合变量结构。概率分布、非负变量、稀疏权重、资源分配这类问题，常常不适合直接套欧氏几何。

---

## 问题定义与边界

讨论 Bregman 散度前，需要先明确三个对象：

| 对象 | 含义 |
|---|---|
| $\Omega$ | 定义域，也就是变量允许出现的区域 |
| $\phi:\Omega\to\mathbb{R}$ | 生成散度的凸函数 |
| $x,y\in\Omega$ | 被比较的两个点 |

如果 $\Omega$、$\phi$、$x,y$ 没有定义清楚，$D_\phi(x,y)$ 就没有明确意义。比如 KL 散度要求概率分布非负且和为 1；如果把任意实数向量丢进去，公式不再对应概率分布之间的差异。

常见条件如下：

| 条件 | 白话解释 | 带来的结论 |
|---|---|---|
| 可微 | 每个内部点都有梯度 | 可以写出 $\nabla\phi(y)$ |
| 严格凸 | 不同点之间函数值严格弯曲 | $D_\phi(x,y)=0$ 通常只在 $x=y$ 时成立 |
| 强凸 | 弯曲程度有统一下界 | 可用于证明更稳定的收敛界 |
| 定义域内部正性 | 点不能落在边界坏位置 | 熵函数、KL 散度中很常见 |

KL 散度是重要例子。对两个离散分布 $p,q$，有：

$$
D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}
$$

如果某个位置 $q_i=0$ 但 $p_i>0$，则 $D_{KL}(p\|q)=\infty$。白话解释是：$q$ 认为某事件完全不可能发生，但 $p$ 认为它确实有正概率，这种解释代价是无穷大。

Bregman 散度还有三个边界：

| 边界 | 说明 |
|---|---|
| 不对称 | 一般 $D_\phi(x,y)\neq D_\phi(y,x)$ |
| 非三角不等式 | 一般不能保证 $D_\phi(x,z)\le D_\phi(x,y)+D_\phi(y,z)$ |
| 可能无穷大 | 特别是在概率分布、熵函数、边界点附近 |

一个玩具例子：令 $\phi(x)=x\log x$，定义域为 $x>0$。如果 $x=1$、$y=0$，公式里的 $\log y$ 没有定义。因此不是所有数值点都能直接代入 Bregman 散度。定义域是公式的一部分，不是附加说明。

---

## 核心机制与推导

Bregman 散度来自凸函数的一阶近似。对可微凸函数 $\phi$，在 $y$ 处的一阶线性近似是：

$$
\phi(y)+\langle\nabla\phi(y),x-y\rangle
$$

凸函数的性质保证：

$$
\phi(x)\ge \phi(y)+\langle\nabla\phi(y),x-y\rangle
$$

两边相减得到：

$$
D_\phi(x,y)=\phi(x)-\phi(y)-\langle\nabla\phi(y),x-y\rangle\ge 0
$$

这说明 Bregman 散度本质上是“真实函数值”和“切线预测值”的差。

取二次函数作为玩具例子。令：

$$
\phi(x)=\frac{1}{2}x^2
$$

则 $\nabla\phi(y)=y$，所以：

$$
D_\phi(x,y)=\frac{1}{2}x^2-\frac{1}{2}y^2-y(x-y)=\frac{1}{2}(x-y)^2
$$

当 $x=3,y=1$ 时：

$$
D_\phi(3,1)=\frac{1}{2}(3-1)^2=2
$$

这就是平方距离的一半。

Bregman 投影把“找最近点”从欧氏几何换成 $\phi$ 定义的几何。给定集合 $C$ 和点 $z$：

$$
\Pi_C^\phi(z)=\arg\min_{x\in C}D_\phi(x,z)
$$

其中 $\arg\min$ 表示让目标函数最小的变量取值。

镜像下降是 Bregman 散度在优化中的核心用法。设 $g_t$ 是当前点 $x_t$ 处的梯度或次梯度，$\eta_t$ 是学习率，则镜像下降写作：

$$
x_{t+1}=\arg\min_{x\in C}\left\{\langle g_t,x\rangle+\frac{1}{\eta_t}D_\phi(x,x_t)\right\}
$$

如果暂时忽略约束，这个更新等价于在对偶空间做一步梯度下降：

$$
\nabla\phi(x_{t+1})=\nabla\phi(x_t)-\eta_t g_t
$$

对偶空间是由 $\nabla\phi(x)$ 表示的空间；白话说，就是先把变量通过 $\nabla\phi$ 换一种坐标，再做更新，最后映回原变量。

机制可以写成：

| 步骤 | 欧氏梯度下降 | 镜像下降 |
|---|---|---|
| 当前变量 | $x_t$ | $x_t$ |
| 更新空间 | 原空间 | 对偶空间 |
| 更新方向 | $x_t-\eta_t g_t$ | $\nabla\phi(x_t)-\eta_t g_t$ |
| 回到可行域 | 欧氏投影 | Bregman 投影或逆映射 |
| 适合结构 | 普通实向量 | 分布、非负、单纯形 |

原空间到对偶空间的流程是：

```text
x_t
  -> grad_phi(x_t)
  -> grad_phi(x_t) - eta_t * g_t
  -> grad_phi_inv(...)
  -> x_{t+1}
```

真实工程例子是分布式资源分配。假设多个节点共同优化一个全局资源比例 $x$，要求 $x_i\ge 0$ 且 $\sum_i x_i=1$。这个约束集合叫概率单纯形，也就是所有分量非负且总和为 1 的向量集合。若直接做欧氏梯度下降，更新后可能出现负数，还需要额外投影。若使用熵函数：

$$
\phi(x)=\sum_i x_i\log x_i
$$

镜像下降会产生指数加权形式的更新，天然保持正性，再通过归一化保证和为 1。这比每轮都解一个欧氏投影子问题更贴合概率变量结构。

---

## 代码实现

实现镜像下降时，重点不是只写一个散度函数，而是把更新过程拆成四个部分：梯度、对偶映射、逆映射、投影。不同 $\phi$ 的差异主要体现在 $\nabla\phi$ 和 $\nabla\phi^{-1}$。

下面代码包含两个版本：二次势能和熵势能。势能函数是这里对 $\phi$ 的一种称呼，表示它决定了优化使用的几何。

```python
import numpy as np

def mirror_descent_step(x_t, g_t, eta, grad_phi, grad_phi_inv, proj=None):
    dual = grad_phi(x_t) - eta * g_t
    x_next = grad_phi_inv(dual)
    if proj is not None:
        x_next = proj(x_next)
    return x_next

def quadratic_grad_phi(x):
    return x

def quadratic_grad_phi_inv(u):
    return u

def simplex_normalize(x):
    x = np.maximum(x, 1e-12)
    return x / x.sum()

def entropy_grad_phi(x):
    # phi(x)=sum_i x_i log x_i, grad_phi(x)=log(x)+1
    x = np.maximum(x, 1e-12)
    return np.log(x) + 1.0

def entropy_grad_phi_inv(u):
    # inverse of log(x)+1 = u is x = exp(u-1)
    x = np.exp(u - 1.0)
    return simplex_normalize(x)

def squared_bregman(x, y):
    return 0.5 * np.sum((x - y) ** 2)

def kl_divergence(p, q):
    eps = 1e-12
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    return np.sum(p * np.log(p / q))

# 1. 二次势能：镜像下降退化为普通梯度下降
x = np.array([1.0, 2.0])
g = np.array([0.2, -0.4])
eta = 0.5

x_next = mirror_descent_step(
    x, g, eta,
    quadratic_grad_phi,
    quadratic_grad_phi_inv
)

expected = x - eta * g
assert np.allclose(x_next, expected)
assert np.isclose(squared_bregman(np.array([3.0]), np.array([1.0])), 2.0)

# 2. 熵势能：在 simplex 上做指数型更新
p = np.array([0.2, 0.5, 0.3])
grad = np.array([1.0, -0.5, 0.2])

p_next = mirror_descent_step(
    p, grad, 0.3,
    entropy_grad_phi,
    entropy_grad_phi_inv
)

assert np.all(p_next > 0)
assert np.isclose(p_next.sum(), 1.0)
assert kl_divergence(p_next, p) >= 0.0
```

二次势能下，$\nabla\phi(x)=x$，所以对偶空间和原空间相同，镜像下降就是普通梯度下降：

$$
x_{t+1}=x_t-\eta_t g_t
$$

熵势能下，$\nabla\phi(x)=\log x+1$，更新发生在对数空间里。简化后可以理解为：

```python
dual = np.log(x_t) - eta * g_t
x_next = np.exp(dual)
x_next = x_next / x_next.sum()
```

这个形式在概率分布优化、在线学习、多分类权重更新中非常常见。它不会轻易把正数权重更新成负数，因此比直接在原空间做加减更适合 simplex。

---

## 工程权衡与常见坑

工程里最常见的问题不是公式写错，而是几何选错。若变量本来是概率分布，却用欧氏更新，每一步都可能跑出可行域。再把它投影回来，会增加计算成本，也可能造成数值不稳定。

| 常见坑 | 后果 | 处理方式 |
|---|---|---|
| 把 Bregman 散度当距离 | 错用对称性、三角不等式 | 明确它通常不是 metric |
| 交换 $D_\phi(x,y)$ 参数 | 结果改变，优化方向错误 | 固定约定并写测试 |
| KL 中 $q_i=0,p_i>0$ | 散度变成无穷大 | 加 epsilon 或限制支撑集 |
| 熵几何出现零值 | $\log 0$ 不可用 | 裁剪到 $[epsilon,1]$ |
| 指数更新下溢 | 权重变成 0 | 在对数域计算，必要时减去最大值 |
| 盲目使用强凸结论 | 收敛证明不成立 | 检查 $\phi$ 和范数条件 |

真实工程例子：联邦学习中的个性化模型混合。假设每个客户端要学习一组模型权重 $x$，表示从多个全局专家模型中各取多少比例。这个 $x$ 是概率向量。若用普通梯度下降，某些权重可能变成负数，随后要投影回 simplex。若用熵型镜像下降，权重通过指数更新后再归一化，天然保持非负和总和为 1。这种更新更像“重新分配比例”，而不是“在实数空间里随意平移”。

稳定实现通常遵守三条规则：

| 场景 | 建议 |
|---|---|
| 需要 $\log x$ | 先做 `np.maximum(x, eps)` |
| 需要 `exp` | 先减去最大值，避免溢出 |
| 需要归一化 | 检查分母不为 0，并写 `assert` |

也要注意，Bregman 投影不一定比欧氏投影便宜。它是否更好，取决于 $\phi$ 是否让更新和约束结构变简单。对于盒约束 $l\le x_i\le u$，欧氏投影只是逐元素裁剪，通常已经足够。

---

## 替代方案与适用边界

Bregman 散度不是所有优化问题的默认答案。选型应先看变量结构，再看算法习惯。

| 方法 | 更新形式 | 适合场景 | 不适合场景 |
|---|---|---|---|
| 标准梯度下降 | $x_{t+1}=x_t-\eta g_t$ | 无约束平滑优化 | 复杂约束、概率变量 |
| 投影梯度下降 | 梯度步后欧氏投影 | 盒约束、球约束、简单凸集 | 投影代价高的集合 |
| 镜像下降 | 对偶空间更新后映回 | simplex、非负变量、熵正则 | $\phi$ 难选或逆映射难算 |
| 自然梯度 | 用信息几何修正梯度 | 统计模型、概率参数 | Fisher 矩阵代价过高 |

自然梯度是另一类非欧氏优化方法。它用模型分布的局部几何修正梯度方向，常见于统计学习和强化学习。它与镜像下降有关，但实现上通常需要 Fisher 信息矩阵，成本可能更高。

适用边界可以按下面原则判断：

| 问题特征 | 优先考虑 |
|---|---|
| 变量在 $\mathbb{R}^d$，无复杂约束 | 标准梯度下降 |
| 变量有简单盒约束 | 投影梯度下降 |
| 变量是概率分布 | 熵型镜像下降 |
| 目标中有 KL 或熵正则 | Bregman / 镜像下降 |
| 分布式节点有本地约束 | 分布式镜像下降 |
| 需要统计流形上的参数更新 | 自然梯度 |

一个简单判断是：如果变量本身就是普通实数向量，欧氏方法通常更直接；如果变量是分布、比例、非负权重、稀疏组合，就应该考虑 Bregman 几何。

分布式优化中，Bregman 散度的作用更明显。多个节点各自持有本地目标 $f_i(x)$，全局目标是：

$$
\min_{x\in C}\sum_{i=1}^n f_i(x)
$$

若 $C$ 是 simplex 或非欧氏约束集合，每个节点用镜像下降更新本地变量，再通过通信做平均或 gossip 交换，就能把约束结构和分布式通信结合起来。这里的 gossip 是指节点只和邻居交换信息，而不是每轮都汇总到中心服务器。

选型原则是：问题结构优先于算法习惯。不要因为熟悉梯度下降就默认使用欧氏更新，也不要因为 Bregman 散度更高级就到处使用。几何、约束、数值成本三者匹配，才是合理选择。

---

## 参考资料

| 参考 | 作用 |
|---|---|
| Beck, A., Teboulle, M. “Mirror descent and nonlinear projected subgradient methods for convex optimization.” Operations Research Letters, 2003. https://doi.org/10.1016/S0167-6377(02)00231-6 | 镜像下降和非线性投影子梯度方法的经典框架 |
| Technion 论文页：https://cris.technion.ac.il/en/publications/mirror-descent-and-nonlinear-projected-subgradient-methods-for-co/ | 上述论文的机构页面，便于查找文献信息 |
| Nemirovski, A. “Mirror Descent and Saddle Point First Order Algorithms.” ICML/COLT 2012 tutorial entry. https://icml.cc/2012/files/handbook.pdf | 从对偶视角理解镜像下降和一阶方法 |
| Lu, Y. et al. “Optimal distributed stochastic mirror descent for strongly convex optimization.” Automatica, 2018. https://www.sciencedirect.com/science/article/abs/pii/S0005109817306404 | 分布式随机镜像下降在强凸优化中的收敛分析 |
| “Gossip-based distributed stochastic mirror descent for constrained optimization.” Neural Networks, 2024. https://www.sciencedirect.com/science/article/abs/pii/S0893608024002156 | gossip 通信和约束分布式镜像下降的应用研究 |
