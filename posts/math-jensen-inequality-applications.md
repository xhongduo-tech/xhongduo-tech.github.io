## 核心结论

Jensen 不等式描述的是“函数作用”和“取平均”之间的顺序关系。对凸函数 $\varphi$，如果随机变量 $X$ 的期望存在，并且 $\varphi(X)$ 的期望也存在，那么：

$$
\varphi(E[X]) \le E[\varphi(X)]
$$

凸函数是指函数图像上任意两点之间的连线都在图像上方。白话说，凸函数会放大输入的波动，所以“先把随机变量送进函数，再取平均”通常更大。

离散形式更直观。给定 $x_1,\dots,x_n$ 和权重 $w_i \ge 0$，且 $\sum_i w_i = 1$，有：

$$
\varphi\left(\sum_i w_i x_i\right) \le \sum_i w_i \varphi(x_i)
$$

这里的 $w_i$ 可以理解为概率，所以 $\sum_i w_i x_i$ 就是离散随机变量的期望。

玩具例子：令 $X \in \{0,2\}$，两个取值概率都是 $1/2$，取 $\varphi(x)=x^2$。则：

$$
E[X] = 1
$$

$$
E[\varphi(X)] = E[X^2] = \frac{0^2 + 2^2}{2} = 2
$$

$$
\varphi(E[X]) = 1^2 = 1
$$

所以 $E[\varphi(X)] = 2 \ge 1 = \varphi(E[X])$。这不是巧合，而是凸函数 Jensen 不等式的直接体现。

如果 $\varphi$ 是凹函数，方向反过来。凹函数是指函数图像上任意两点之间的连线都在图像下方。常见例子是 $\log x$，定义域要求 $x>0$。令 $Y \in \{1,9\}$，两个取值概率都是 $1/2$，则：

$$
E[\log Y] = \frac{\log 1 + \log 9}{2} = \log 3
$$

$$
\log E[Y] = \log 5
$$

因为 $\log 3 < \log 5$，所以：

$$
E[\log Y] \le \log E[Y]
$$

机器学习里，Jensen 不等式最常见的价值不是单独证明一个不等式，而是把难优化的 $\log E[\cdot]$ 变成可优化下界。EM 算法、变分推断、ELBO 推导、KL 散度非负性，都可以放到这条逻辑线上理解。

---

## 问题定义与边界

Jensen 不等式不能脱离前提使用。它依赖函数形状、随机变量定义域、期望是否存在，以及等号条件。

| 概念 | 说明 |
|---|---|
| 凸函数 | 任意两点连线在图像上方，例如 $x^2$、$e^x$ |
| 凹函数 | 任意两点连线在图像下方，例如 $\log x$ |
| 期望 | 随机变量按概率加权后的平均值 |
| 退化随机变量 | 取值几乎处处相同，即没有实际波动 |
| 线性函数 | 形如 $ax+b$，Jensen 两边通常相等 |

凸函数版本是：

$$
E[\varphi(X)] \ge \varphi(E[X])
$$

凹函数版本是：

$$
E[\varphi(X)] \le \varphi(E[X])
$$

等号通常只在两类情况下成立。第一，$X$ 是退化随机变量，也就是 $X$ 几乎总是同一个值。第二，$\varphi$ 在 $X$ 可能取值覆盖的区间上是线性的。比如 $\varphi(x)=2x+1$ 时，不管 $X$ 怎么分布，都有：

$$
E[2X+1] = 2E[X]+1
$$

这时 Jensen 只是等式。

反例也很重要。如果把凸函数的方向写反，通常会错。仍取 $X \in \{0,2\}$，各 $1/2$，$\varphi(x)=x^2$。错误方向会得到：

$$
E[X^2] < (E[X])^2
$$

但真实数值是：

$$
2 < 1
$$

这显然不成立。

定义域也不能忽略。$\log x$ 只有在 $x>0$ 时才有意义。因此在 KL 散度、ELBO 或 EM 推导中，如果要写 $\log \frac{p(x,z)}{q(z)}$，就必须保证分母不为 0，并且比值在相关区域为正。更准确地说，通常要求 $q(z)>0$ 的地方也满足 $p(x,z)>0$，否则对数项可能变成无定义或无穷大。

---

## 核心机制与推导

先看离散权重形式：

$$
\varphi\left(\sum_i w_i x_i\right) \le \sum_i w_i\varphi(x_i)
$$

其中 $w_i \ge 0$ 且 $\sum_i w_i = 1$。因为权重和为 1，所以 $\sum_i w_i x_i$ 是加权平均。如果把 $w_i$ 看成概率 $P(X=x_i)$，那么：

$$
E[X] = \sum_i w_i x_i
$$

$$
E[\varphi(X)] = \sum_i w_i \varphi(x_i)
$$

于是离散形式就变成了：

$$
\varphi(E[X]) \le E[\varphi(X)]
$$

这说明 Jensen 的本质不是概率论专属结论，而是凸性对加权平均的约束。概率论只是把“权重”解释成了“概率”。

在 EM 和变分推断中，关键函数通常是 $\log$。$\log$ 是凹函数，所以 Jensen 方向是：

$$
E[\log Z] \le \log E[Z]
$$

等价地：

$$
\log E[Z] \ge E[\log Z]
$$

这正是构造下界的来源。

设观测变量为 $x$，隐变量为 $z$，目标是计算或优化边缘似然 $p(x)$。边缘似然是对隐变量求和或积分后的结果：

$$
p(x)=\int p(x,z)dz
$$

直接优化 $\log p(x)$ 往往困难，因为对数外面包着求和或积分。引入任意分布 $q(z)$，并要求在 $q(z)>0$ 的地方比值有定义：

$$
p(x)=\int q(z)\frac{p(x,z)}{q(z)}dz
$$

所以：

$$
\log p(x) = \log E_q\left[\frac{p(x,z)}{q(z)}\right]
$$

因为 $\log$ 是凹函数：

$$
\log E_q\left[\frac{p(x,z)}{q(z)}\right]
\ge
E_q\left[\log \frac{p(x,z)}{q(z)}\right]
$$

展开对数：

$$
E_q[\log p(x,z)-\log q(z)]
$$

这个下界就是 ELBO。ELBO 是 evidence lower bound 的缩写，白话说就是“证据 $\log p(x)$ 的一个可优化下界”。

完整链条是：

$$
\log p(x)
=
\log E_q\left[\frac{p(x,z)}{q(z)}\right]
\ge
E_q[\log p(x,z)-\log q(z)]
=
ELBO(q)
$$

它还可以写成 KL 形式。KL 散度是衡量两个概率分布差异的非负量，记为 $D_{KL}(q||p)$。有：

$$
\log p(x) = ELBO(q) + D_{KL}(q(z)||p(z|x))
$$

因此，当 $\log p(x)$ 固定时，最大化 $ELBO(q)$ 等价于最小化 $q(z)$ 和真实后验 $p(z|x)$ 之间的 KL 散度。

KL 非负性也可以用 Jensen 看清楚：

$$
D_{KL}(q||p)=E_q\left[\log\frac{q}{p}\right]
$$

令 $\varphi(t)=-\log t$。因为 $-\log t$ 是凸函数：

$$
E_q\left[-\log\frac{p}{q}\right]
\ge
-\log E_q\left[\frac{p}{q}\right]
$$

而：

$$
E_q\left[\frac{p}{q}\right]=\int q(z)\frac{p(z)}{q(z)}dz=\int p(z)dz=1
$$

所以：

$$
D_{KL}(q||p)
\ge
-\log 1
=
0
$$

推导路径可以概括为：

| 目标 | 关键表达式 | 使用形式 | 得到什么 |
|---|---|---|---|
| Jensen 基础 | $\varphi(E[X]) \le E[\varphi(X)]$ | 凸函数 | 平均与函数顺序关系 |
| ELBO | $\log E_q[p(x,z)/q(z)]$ | 凹函数 $\log$ | $\log p(x)$ 的下界 |
| EM | $\log \sum_z p(x,z)$ | 凹函数 $\log$ | 可迭代优化的辅助函数 |
| KL 非负 | $E_q[\log(q/p)]$ | 凸函数 $-\log$ | $D_{KL}(q||p)\ge0$ |

真实工程例子：高斯混合模型 GMM 中，每个样本可能来自 $K$ 个高斯分量之一，分量编号就是隐变量 $z$。直接最大化 $\log \sum_z p(x,z)$ 不方便，因为 $\log$ 外面套着求和。EM 算法通过 Jensen 构造下界，E 步计算每个样本属于每个分量的责任度，M 步最大化这个下界。若有 $N$ 个样本、$K$ 个分量，单轮责任度计算至少是 $O(NK)$；若每个样本维度是 $d$，且使用 full covariance，高斯密度计算还会带来更高的维度代价，常见实现会接近 $O(NKd^2)$ 或更高，取决于协方差结构和预计算方式。

---

## 代码实现

下面的代码分两部分：第一部分直接验证 Jensen 数值；第二部分给出一个极简 ELBO 骨架，展示“优化下界”的工程形式。

| 代码段 | 输入 | 输出 | 目的 |
|---|---|---|---|
| Jensen 验证 | 离散分布 | 两边数值 | 检查凸函数和凹函数方向 |
| ELBO 骨架 | $q(z)$、$p(x,z)$ | 下界值 | 说明优化目标如何落地 |

```python
import math

def expectation(values, probs):
    assert len(values) == len(probs)
    assert abs(sum(probs) - 1.0) < 1e-12
    return sum(v * p for v, p in zip(values, probs))

# Part 1: Jensen for convex phi(x)=x^2
x_values = [0.0, 2.0]
x_probs = [0.5, 0.5]

ex = expectation(x_values, x_probs)
e_x_square = expectation([x * x for x in x_values], x_probs)
square_ex = ex * ex

print("E[X] =", ex)
print("E[X^2] =", e_x_square)
print("(E[X])^2 =", square_ex)

assert e_x_square >= square_ex
assert e_x_square == 2.0
assert square_ex == 1.0

# Part 2: Jensen for concave phi(x)=log(x)
y_values = [1.0, 9.0]
y_probs = [0.5, 0.5]

e_log_y = expectation([math.log(y) for y in y_values], y_probs)
log_e_y = math.log(expectation(y_values, y_probs))

print("E[log Y] =", e_log_y)
print("log E[Y] =", log_e_y)

assert e_log_y <= log_e_y
assert abs(e_log_y - math.log(3.0)) < 1e-12
assert abs(log_e_y - math.log(5.0)) < 1e-12

# Part 3: A tiny ELBO computation over a discrete latent variable z
# Suppose z has two states. q(z) is a variational distribution.
q = [0.4, 0.6]

# p(x,z) values are unnormalized joint probabilities for a fixed observed x.
p_xz = [0.12, 0.18]

assert all(v > 0 for v in q)
assert all(v > 0 for v in p_xz)
assert abs(sum(q) - 1.0) < 1e-12

def elbo(q, p_xz):
    return sum(qz * (math.log(pxz) - math.log(qz)) for qz, pxz in zip(q, p_xz))

lower_bound = elbo(q, p_xz)
log_px = math.log(sum(p_xz))

print("ELBO =", lower_bound)
print("log p(x) =", log_px)

assert lower_bound <= log_px + 1e-12
```

变分推断的工程骨架通常长这样：

```python
# pseudo-code

initialize q_theta_z

for step in range(max_steps):
    # 1. estimate or compute ELBO(theta)
    current_elbo = E_q_theta[log p(x, z) - log q_theta(z)]

    # 2. update theta by gradient ascent or coordinate update
    theta = optimizer.step(theta, gradient(current_elbo))

    # 3. monitor lower bound
    if converged(current_elbo):
        break
```

这个伪代码里的核心不是某个具体模型，而是目标函数从 $\log p(x)$ 变成了 $ELBO(q)$。原目标难算，下界可算；直接似然难优化，下界可以迭代优化。

---

## 工程权衡与常见坑

Jensen 最常见的工程错误是方向写反。看到 $\log$ 时尤其要谨慎，因为 $\log$ 是凹函数，不是凸函数：

$$
E[\log Z] \le \log E[Z]
$$

所以：

$$
\log E[Z] \ge E[\log Z]
$$

ELBO 正是这个方向给出的下界。如果误写成 $E[\log Z] \ge \log E[Z]$，整个优化目标就会变成错误的上界或无意义表达。

第二个坑是只看均值。Jensen 说明，均值相同不代表函数期望相同。取两个随机变量：

$$
X_1 \equiv 1
$$

$$
X_2 \in \{0,2\}, P(X_2=0)=P(X_2=2)=1/2
$$

它们都有：

$$
E[X_1]=E[X_2]=1
$$

但对 $\varphi(x)=x^2$：

$$
E[X_1^2]=1
$$

$$
E[X_2^2]=2
$$

均值一样，平方期望不同。原因是 $X_2$ 更分散，而凸函数会放大波动。这也是为什么机器学习里的方差、熵、KL、风险函数不能只靠均值判断。

| 常见错误 | 为什么错 | 正确写法 |
|---|---|---|
| 把 Jensen 方向写反 | 凸函数和凹函数方向相反 | 先确认函数形状 |
| 只看均值 | 忽略波动对凸函数期望的影响 | 比较分布而不是只看 $E[X]$ |
| 误用 $\log$ | $\log$ 是凹函数 | $E[\log Z] \le \log E[Z]$ |
| 忽略定义域 | $\log x$ 要求 $x>0$ | 检查概率比值是否为正 |
| 忽略等号条件 | Jensen 通常不是等式 | 只有退化或线性区间才常相等 |

复杂度上，Jensen 本身不增加算法复杂度，它只是给出可优化目标。但由它导出的算法会引入迭代成本。以 GMM 的 EM 为例，每轮要遍历样本和分量，基础责任度计算是 $O(NK)$；如果密度计算涉及高维协方差矩阵，还要乘上维度相关代价。工程上通常用 diagonal covariance、共享协方差、缓存矩阵分解等方式降低成本。

---

## 替代方案与适用边界

Jensen 不等式不是万能工具。它只能在函数具有凸性或凹性时直接给出方向。如果目标函数既不是凸函数，也不是凹函数，机械套用 Jensen 通常得不到正确结论。

| 方法 | 适用对象 | 优点 | 局限 |
|---|---|---|---|
| Jensen | 凸函数或凹函数 | 简洁，通用，推导短 | 只给方向，不一定紧 |
| 泰勒展开 | 局部光滑函数 | 可刻画误差项 | 需要光滑性和局部条件 |
| 切线界 | 凸优化、变分界 | 可构造可优化代理目标 | 依赖具体函数形式 |
| Hoeffding 类界 | 有界随机变量尾部概率 | 给出概率型偏差控制 | 要求有界或次高斯条件 |
| 数值积分/蒙特卡洛 | 黑箱期望 | 易实现，模型无关 | 方差高，成本可能大 |
| 鞅工具 | 序列依赖随机过程 | 适合在线学习和随机过程 | 前提更复杂 |

所谓“反向 Jensen”也要小心。标准 Jensen 不提供一般性的反向结论。对凸函数，一般不能写：

$$
E[\varphi(X)] \le \varphi(E[X])
$$

除非 $\varphi$ 在相关区间线性，或随机变量退化，或额外假设能补出一个误差上界。

某些信息论分析会出现“反向 Jensen”形式，例如在变量有界、函数曲率受控、尾部概率受控时，可以得到类似：

$$
E[\varphi(X)] \le \varphi(E[X]) + \text{error term}
$$

这里的关键是后面的误差项。它不是标准 Jensen 的直接反向，而是额外条件带来的补偿。没有这些条件，反向 Jensen 通常不成立。

一个实用判断流程是：

| 问题 | 判断 |
|---|---|
| 函数是凸的吗 | 用凸 Jensen 得下界或上界关系 |
| 函数是凹的吗 | 方向反过来，$\log$ 属于这一类 |
| 需要紧界吗 | Jensen 可能太松，需要泰勒或模型结构 |
| 目标无显式形式吗 | 考虑蒙特卡洛估计 |
| 要控制尾部风险吗 | 考虑 Hoeffding、Bernstein、鞅不等式 |

因此，Jensen 最适合用来识别和构造“平均与函数顺序”带来的界。它不负责解决所有优化问题，也不保证界足够紧。

---

## 参考资料

| 类型 | 资料 |
|---|---|
| 基础理论 | Stanford SEP: The Jensen Inequality，用于理解 Jensen 的基础定义和离散权重形式 |
| EM | Dempster, Laird, Rubin (1977), Maximum Likelihood from Incomplete Data via the EM Algorithm，用于理解 EM 如何构造辅助函数 |
| 变分推断 | Jordan et al. (1999), An Introduction to Variational Methods for Graphical Models，用于理解图模型中的变分方法 |
| 变分推断 | Blei, Kucukelbir, McAuliffe (2017), Variational Inference: A Review for Statisticians，用于理解 ELBO、KL 和现代 VI 表述 |
| 工程应用 | Arenz, Zhong, Neumann (2020), Trust-Region Variational Inference with Gaussian Mixture Models，用于理解 GMM 与变分推断的工程结合 |
| 特殊扩展 | Reversing Jensen’s Inequality for Information-Theoretic Analyses，用于理解反向 Jensen 需要额外条件 |

1. Stanford SEP: The Jensen Inequality  
   https://sepwww.stanford.edu/sep/prof/pvi/jen/paper_html/node2.html

2. Dempster, A. P., Laird, N. M., Rubin, D. B. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm.  
   https://academic.oup.com/jrsssb/article-abstract/39/1/1/7027539

3. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., Saul, L. K. (1999). An Introduction to Variational Methods for Graphical Models.  
   https://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf

4. Blei, D. M., Kucukelbir, A., McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians.  
   https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2016.pdf

5. Arenz, O., Zhong, M., Neumann, G. (2020). Trust-Region Variational Inference with Gaussian Mixture Models.  
   https://www.jmlr.org/papers/v21/19-524.html

6. Reversing Jensen’s Inequality for Information-Theoretic Analyses.  
   https://www.mdpi.com/2078-2489/13/1/39
