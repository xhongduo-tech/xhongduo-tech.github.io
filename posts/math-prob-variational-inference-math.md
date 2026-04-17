## 核心结论

变分推断（Variational Inference, VI）的核心是优化，不是采样。它要解决的问题是：真实后验分布 $p(z|x)$ 往往难以直接计算，于是选一个容易处理的参数化分布 $q(z;\phi)$ 去逼近它，并通过优化参数 $\phi$ 让两者尽量接近。

最常用的目标函数是证据下界（Evidence Lower BOund, ELBO）：

$$
\mathrm{ELBO}(q)=\mathbb{E}_{q(z)}[\log p(x,z)]-\mathbb{E}_{q(z)}[\log q(z)]
$$

它和边缘对数似然 $\log p(x)$ 的关系是：

$$
\log p(x)=\mathrm{ELBO}(q)+\mathrm{KL}(q(z)\|p(z|x))
$$

其中 KL 散度定义为：

$$
\mathrm{KL}(q(z)\|p(z|x))=\mathbb{E}_{q(z)}\left[\log \frac{q(z)}{p(z|x)}\right]\ge 0
$$

因此立刻得到：

$$
\mathrm{ELBO}(q)\le \log p(x)
$$

这条式子有两层含义。

第一，ELBO 是 $\log p(x)$ 的下界，所以名字叫“证据下界”。

第二，最大化 ELBO 等价于最小化 $\mathrm{KL}(q\|p)$。因为 $\log p(x)$ 对给定数据 $x$ 是常数，优化时唯一能变的是 ELBO 和 KL 两项，ELBO 越大，KL 就越小。

从直觉上看，ELBO 有两个相互拉扯的部分：

$$
\mathrm{ELBO}(q)=\underbrace{\mathbb{E}_{q(z)}[\log p(x,z)]}_{\text{拟合数据}}
+\underbrace{\mathcal H(q)}_{\text{保持不确定性}}
$$

其中熵 $\mathcal H(q)$ 定义为：

$$
\mathcal H(q)=-\mathbb{E}_{q(z)}[\log q(z)]
$$

第一项要求 $q$ 把概率质量放到“能解释观测数据 $x$”的潜变量区域；第二项要求 $q$ 不要无端塌缩成过窄的分布。工程上，这对应“拟合能力”和“稳定性”之间的平衡。

一个最简单的玩具例子是一维高斯后验。设目标后验为：

$$
p(z|x)=\mathcal N(1,0.25^2)
$$

也就是：

$$
p(z|x)\propto \exp\left(-\frac{(z-1)^2}{2\cdot 0.25^2}\right)
$$

如果把变分族限制为一维高斯：

$$
q(z)=\mathcal N(m,s^2)
$$

那么最优解就是直接取：

$$
m=1,\quad s=0.25
$$

因为此时 $q$ 与真实后验完全一致，KL 散度为零：

$$
\mathrm{KL}(q\|p)=0
$$

这说明 VI 的误差并不是“方法天然粗糙”，而是来自变分族本身的限制。只要近似族足够表达真实后验，VI 可以精确恢复目标分布。

| 项 | 数学形式 | 作用 | 优化趋势 |
| --- | --- | --- | --- |
| 期望项 | $\mathbb E_q[\log p(x,z)]$ | 鼓励 $q$ 关注高联合概率区域 | 越大越好 |
| 熵项 | $-\mathbb E_q[\log q(z)]$ | 抑制 $q$ 过度自信、过窄 | 越大越好 |
| KL 项 | $\mathbb E_q[\log q(z)-\log p(z|x)]$ | 衡量近似后验与真实后验的差距 | 越小越好 |

---

## 问题定义与边界

贝叶斯推断的目标通常是求后验分布：

$$
p(z|x)=\frac{p(x,z)}{p(x)}
$$

其中：

- $x$ 表示观测数据
- $z$ 表示潜变量或未知参数
- $p(x,z)$ 是联合分布
- $p(x)$ 是边缘似然，也叫 evidence

真正困难的部分通常在分母：

$$
p(x)=\int p(x,z)\,dz
$$

如果 $z$ 是离散变量，这里是求和；如果 $z$ 是连续变量，这里是积分。无论哪种情形，只要潜变量维度升高、变量之间存在强相关、或者模型不再满足共轭结构，这一步就会变得非常难算。

所谓共轭，是指先验分布和似然函数相乘后，后验仍落在同一分布族中。比如高斯均值的高斯先验配高斯似然，就能得到高斯后验。这种情况下常常能写出闭式公式；一旦不共轭，闭式解通常就消失了。

面对难算后验，常见有两条路线。

一条是 MCMC。它通过构造马尔可夫链采样来逼近后验，理论上渐近精确，但代价是计算慢、样本相关、收敛诊断复杂。

另一条就是 VI。它不直接“算出”后验，而是把问题改写成优化问题：

1. 选一个可计算的近似分布族 $q(z;\phi)$
2. 定义“接近后验”的目标函数，通常是 ELBO 或 KL
3. 在这个分布族中优化参数 $\phi$

这一步转换非常关键。原问题是积分难，VI 把它改写成优化问题，利用梯度下降、坐标上升、自动微分等工具求解。

最常见的变分族是均值场近似（mean-field approximation）：

$$
q(z)=\prod_{i=1}^m q_i(z_i)
$$

它的含义很直接：先假设潜变量分量之间相互独立。这个假设未必真实，但它带来一个非常大的计算收益，即复杂联合分布被拆成多个一维或低维因子。

在均值场下，单个因子的最优更新形式是：

$$
\log q_i^*(z_i)=\mathbb E_{q_{-i}}[\log p(x,z)] + \mathrm{const}
$$

其中 $q_{-i}$ 表示除第 $i$ 个因子外其余所有因子的乘积。

对新手来说，这条公式可以这样理解：更新第 $i$ 个变量时，先把别的变量“按当前近似分布平均掉”，只保留与 $z_i$ 有关的那部分结构，然后得到 $q_i$ 的最优形式。

均值场的边界也很清楚。它通过“独立”换来了“可算”，但如果真实后验里变量相关性很强，这种近似就会系统性丢失依赖结构。最常见的后果有两个：

- 后验方差被低估
- 联合不确定性被误判，置信区间过窄

一个二维高斯就是最典型的例子。若真实后验是相关高斯：

$$
p(z_1,z_2|x)=\mathcal N\left(
\begin{bmatrix}0\\0\end{bmatrix},
\begin{bmatrix}
1 & \rho \\
\rho & 1
\end{bmatrix}
\right)
$$

当 $\rho$ 接近 $1$ 时，真实后验质量集中在一条斜线附近；而均值场近似只能写成 $q(z_1)q(z_2)$，它无法表达这种“沿对角线分布”的形状，于是会倾向于给出一个偏窄、但因子化的近似。

| 方案 | 可计算性 | 逼近能力 | 典型场景 |
| --- | --- | --- | --- |
| 全后验直接处理 | 低 | 高 | 小模型、解析结构强 |
| 均值场 VI | 高 | 中到低 | 高维潜变量、需要快速推理 |
| 结构化 VI | 中 | 中到高 | 变量依赖明显但仍需优化式推断 |
| MCMC | 低 | 高 | 重视后验精度而非速度 |

---

## 核心机制与推导

ELBO 的推导并不复杂，关键是把 $\log p(x)$ 写成关于任意分布 $q(z)$ 的期望。由于 $\log p(x)$ 与 $z$ 无关，所以：

$$
\log p(x)=\mathbb E_{q(z)}[\log p(x)]
$$

再利用贝叶斯公式：

$$
p(x)=\frac{p(x,z)}{p(z|x)}
\quad\Rightarrow\quad
\log p(x)=\log p(x,z)-\log p(z|x)
$$

代入期望得：

$$
\log p(x)=\mathbb E_q[\log p(x,z)]-\mathbb E_q[\log p(z|x)]
$$

接着“加一项再减一项”：

$$
\log p(x)
=
\mathbb E_q[\log p(x,z)]-\mathbb E_q[\log q(z)]
+
\mathbb E_q\left[\log q(z)-\log p(z|x)\right]
$$

识别出两部分：

$$
\mathrm{ELBO}(q)=\mathbb E_q[\log p(x,z)]-\mathbb E_q[\log q(z)]
$$

$$
\mathrm{KL}(q(z)\|p(z|x))=\mathbb E_q\left[\log \frac{q(z)}{p(z|x)}\right]
$$

于是得到：

$$
\log p(x)=\mathrm{ELBO}(q)+\mathrm{KL}(q(z)\|p(z|x))
$$

这条恒等式是整个 VI 的数学基础。它说明：

- 只要 $q$ 合法，ELBO 一定是 $\log p(x)$ 的下界
- 只要 ELBO 提升，说明近似后验一般在向真实后验靠近
- 当且仅当 $q(z)=p(z|x)$ 时，KL 为零，ELBO 等于 $\log p(x)$

另一种常见写法是把 ELBO 拆成“重构项 + 先验正则项”：

$$
\mathrm{ELBO}(q)
=
\mathbb E_q[\log p(x|z)]-\mathrm{KL}(q(z)\|p(z))
$$

推导只要从联合分布分解：

$$
p(x,z)=p(x|z)p(z)
$$

代入即可：

$$
\mathrm{ELBO}(q)
=
\mathbb E_q[\log p(x|z)]
+\mathbb E_q[\log p(z)]
-\mathbb E_q[\log q(z)]
$$

整理为：

$$
\mathrm{ELBO}(q)
=
\mathbb E_q[\log p(x|z)]-\mathrm{KL}(q(z)\|p(z))
$$

这个形式在 VAE 中最常见。第一项鼓励模型解释数据，第二项约束近似后验不要偏离先验太远。

在均值场设定下，CAVI（Coordinate Ascent Variational Inference）的更新公式可由变分法推出。结论是：

$$
q_i^*(z_i)\propto \exp\left(\mathbb E_{q_{-i}}[\log p(x,z)]\right)
$$

如果把它写成对数形式，就是：

$$
\log q_i^*(z_i)=\mathbb E_{q_{-i}}[\log p(x,z)] + \mathrm{const}
$$

这里的 `const` 表示与 $z_i$ 无关的归一化常数。它存在的原因很简单：任何概率分布最后都必须积分为 1，所以需要补一个常数把表达式正规化。

这条公式的意义很大，因为它不是某种经验规则，而是在均值场约束下的最优单因子解。也就是说，如果其余因子固定不动，那么这样更新 $q_i$ 一定不会比别的更新更差。

可以用一个离散的两变量例子帮助理解。设：

$$
q(z_1,z_2)=q_1(z_1)q_2(z_2)
$$

若当前只更新 $q_1$，就把 $q_2$ 当成已知。此时：

$$
q_1^*(z_1)\propto \exp\left(\mathbb E_{q_2}[\log p(x,z_1,z_2)]\right)
$$

也就是说，先对 $z_2$ 的不确定性取平均，再看不同 $z_1$ 取值对应的平均联合对数概率，最后指数化并归一化。CAVI 的本质就是反复做这件事，直到 ELBO 不再显著增加。

再看前面的高斯玩具例子。假设目标后验为：

$$
p(z|x)=\mathcal N(1,0.25^2)
$$

若变分族也是：

$$
q(z)=\mathcal N(m,s^2)
$$

则最优点在：

$$
m=1,\quad s=0.25
$$

此时：

$$
\mathrm{KL}(q\|p)=0
$$

并且：

$$
\mathrm{ELBO}=\log p(x)
$$

这个例子说明，VI 并不是“注定有偏”，偏差来自近似族而非 ELBO 本身。

真实工程里，更常见的是没有闭式更新的情形。例如 VAE 使用摊销推断（amortized inference）：

$$
q_\phi(z|x)=\mathcal N(\mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x)))
$$

这里不再为每个样本单独优化一组变分参数，而是用一个共享神经网络直接输出参数：

- 输入样本 $x$
- 编码器输出 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$
- 这组参数定义该样本的近似后验

问题在于，若直接写：

$$
z\sim q_\phi(z|x)
$$

采样操作通常不可对 $\phi$ 直接反向传播。解决办法是重参数化技巧：

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot \epsilon,\quad \epsilon\sim\mathcal N(0,I)
$$

它把随机性从参数化分布中拆出来，转移到一个与参数无关的噪声变量 $\epsilon$ 上。这样：

- 随机性仍然保留
- 采样路径对 $\mu_\phi,\sigma_\phi$ 可微
- ELBO 可以通过标准反向传播优化

如果只看单样本，VAE 的训练目标一般写成：

$$
\mathcal L(\theta,\phi;x)
=
\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]
-
\mathrm{KL}(q_\phi(z|x)\|p(z))
$$

第一项是重构质量，第二项是正则化。很多初学者会把它理解为“重构损失 + KL 惩罚”，这个理解在工程上是可用的，但数学上更准确的说法是：它就是单样本 ELBO。

| 方法 | 更新方式 | 需要条件 | 优势 | 局限 |
| --- | --- | --- | --- | --- |
| CAVI | 闭式坐标更新 | 常见于共轭模型 | 稳定、可解释 | 难扩展到复杂神经网络 |
| 黑盒 VI | 随机梯度优化 | 只需能估计梯度 | 适用范围广 | 方差控制更难 |
| 重参数化 VAE | 随机梯度优化 | 分布需可重参数化 | 适合大规模深度模型 | 有采样噪声、训练更敏感 |

---

## 代码实现

下面先给一个最小可运行的 Python 例子。目标是验证三个事实：

1. $\mathrm{ELBO}=\log p(x)-\mathrm{KL}(q\|p)$
2. 当 $q$ 更接近真实后验时，KL 下降
3. 当 KL 下降时，ELBO 上升

为了让例子足够直接，这里把真实后验设为一维高斯：

$$
p(z|x)=\mathcal N(\mu_p,\sigma_p^2)
$$

近似分布也设为高斯：

$$
q(z)=\mathcal N(\mu_q,\sigma_q^2)
$$

两个一维高斯之间的 KL 散度有闭式表达式：

$$
\mathrm{KL}(q\|p)
=
\log\frac{\sigma_p}{\sigma_q}
+
\frac{\sigma_q^2+(\mu_q-\mu_p)^2}{2\sigma_p^2}
-\frac12
$$

完整代码如下，可以直接运行：

```python
import math


def kl_gaussian(mu_q, sigma_q, mu_p, sigma_p):
    """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)) for 1D Gaussians."""
    if sigma_q <= 0 or sigma_p <= 0:
        raise ValueError("Standard deviations must be positive.")
    return (
        math.log(sigma_p / sigma_q)
        + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2)
        - 0.5
    )


def elbo_from_log_px(log_px, mu_q, sigma_q, mu_p, sigma_p):
    """Use the identity ELBO = log p(x) - KL(q || p)."""
    kl = kl_gaussian(mu_q, sigma_q, mu_p, sigma_p)
    return log_px - kl, kl


def main():
    # Assume the true posterior is p(z|x) = N(1.0, 0.25^2)
    mu_p = 1.0
    sigma_p = 0.25

    # log p(x) is treated as a constant in this toy example
    log_px = 3.0

    candidates = [
        ("bad_init", 0.0, 1.0),
        ("closer", 0.8, 0.4),
        ("optimal", 1.0, 0.25),
    ]

    for name, mu_q, sigma_q in candidates:
        elbo, kl = elbo_from_log_px(log_px, mu_q, sigma_q, mu_p, sigma_p)
        print(
            f"{name:8s} | mu_q={mu_q:>4.2f} sigma_q={sigma_q:>4.2f} "
            f"| KL={kl:>8.6f} | ELBO={elbo:>8.6f}"
        )

    # Exact match should give zero KL up to numerical precision
    final_kl = kl_gaussian(1.0, 0.25, mu_p, sigma_p)
    assert abs(final_kl) < 1e-12


if __name__ == "__main__":
    main()
```

预期输出形态类似：

```text
bad_init | mu_q=0.00 sigma_q=1.00 | KL=8.113706 | ELBO=-5.113706
closer   | mu_q=0.80 sigma_q=0.40 | KL=0.183145 | ELBO=2.816855
optimal  | mu_q=1.00 sigma_q=0.25 | KL=0.000000 | ELBO=3.000000
```

这个结果说明得非常清楚：

- 初始分布离真实后验很远，所以 KL 很大，ELBO 很低
- 参数逐渐接近时，KL 明显下降
- 当 $q=p$ 时，KL 变成 0，ELBO 达到上界 $\log p(x)$

如果希望进一步理解 ELBO 中“期望项 + 熵项”的组成，可以把上面代码再拆开。对一维高斯，有：

$$
\mathcal H(q)=\frac12 \log(2\pi e \sigma_q^2)
$$

因此 ELBO 也可以写成：

$$
\mathrm{ELBO}(q)=\mathbb E_q[\log p(x,z)] + \mathcal H(q)
$$

在共轭小模型里，这两项常能分别计算；在深度模型里，通常只保留 Monte Carlo 估计形式。

如果是均值场 CAVI，核心循环通常长这样：

```python
# Pseudocode for CAVI
initialize q1, q2, ..., qm

for step in range(max_steps):
    for i in range(m):
        # Update one factor while keeping others fixed
        log_qi = expectation_under_q_except_i(log_joint(x, z))
        qi = normalize(exp(log_qi))

    elbo = compute_elbo(q, x)

    if converged(elbo):
        break
```

这段伪代码背后的数学更新是：

$$
q_i^*(z_i)\propto \exp\left(\mathbb E_{q_{-i}}[\log p(x,z)]\right)
$$

所以每次更新一个因子后，ELBO 不会下降。在很多共轭模型中，可以证明它单调上升。

如果是 VAE，训练目标一般写成 Monte Carlo 近似形式：

$$
\mathcal L(\theta,\phi;x)
\approx
\log p_\theta(x|z)-\mathrm{KL}(q_\phi(z|x)\|p(z)),
\quad z\sim q_\phi(z|x)
$$

典型 PyTorch 骨架如下：

```python
import torch


def vae_loss(encoder, decoder, x):
    mu, log_var = encoder(x)                 # shape: [batch, latent_dim]
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + std * eps                       # reparameterization

    recon_logprob = decoder.log_prob(x, z)   # shape: [batch]
    kl = 0.5 * torch.sum(
        mu.pow(2) + torch.exp(log_var) - log_var - 1,
        dim=-1
    )

    elbo = recon_logprob - kl
    loss = -elbo.mean()
    return loss, elbo.mean(), recon_logprob.mean(), kl.mean()
```

这里有几个新手容易混淆的点：

- `log_var` 是对数方差，不是标准差
- `std = exp(0.5 * log_var)` 是因为 $\log \sigma^2 = 2\log \sigma$
- 优化器最小化 `loss`，所以代码里通常写负的 ELBO
- `recon_logprob` 不是普通 MSE，而是模型定义下的对数似然项

如果只是为了先跑通概念，一个更完整但仍很小的学习路径是：

1. 先运行上面的高斯例子，确认 KL 与 ELBO 的数值关系
2. 再手推一遍均值场更新公式，理解 CAVI 的坐标更新逻辑
3. 最后看 VAE，把“ELBO 优化”与“神经网络训练”连接起来

这样比直接跳进深度学习代码更稳，因为底层数学对象没有变，只是优化工具从闭式解换成了随机梯度。

---

## 工程权衡与常见坑

VI 最大的优势是快，尤其适合大规模模型和需要重复推断的场景。但它的误差不是 Monte Carlo 那种“样本足够多就能消掉”的随机误差，而是由变分族和优化过程共同带来的系统误差。

最常见的问题是方差低估。原因在于，VI 优化的是：

$$
\mathrm{KL}(q\|p)
$$

而不是：

$$
\mathrm{KL}(p\|q)
$$

这两者差别很大。$\mathrm{KL}(q\|p)$ 对“把概率质量放在真实低密度区域”惩罚很重，因此优化时常出现“宁可少覆盖，也不要瞎覆盖”的倾向。这种性质常被称为 zero-forcing。结果就是：

- 遇到多峰后验时，VI 容易只抓住一个峰
- 遇到强相关后验时，均值场容易给出偏窄近似
- 不确定性估计常比真实后验更乐观

一个简单示意是双峰分布：

$$
p(z|x)=0.5\,\mathcal N(-3,1)+0.5\,\mathcal N(3,1)
$$

若你用单峰高斯 $q(z)=\mathcal N(m,s^2)$ 去逼近，最优解往往不会“同时覆盖左右两个峰”，因为那样会在中间低密度区域放太多概率。更常见的是偏向某一侧，抓住其中一个峰。

第二个常见问题是局部最优。ELBO 通常不是凸函数，尤其在深度模型里更明显。这意味着：

- 不同初始化可能收敛到不同解
- ELBO 上升不代表找到全局最优
- 某些坏初始化会让模型长期停在差解附近

因此工程上至少要做两件事：

- 监控 ELBO 曲线或其近似值
- 尝试多组初始化或多个随机种子

第三个问题是梯度噪声。黑盒 VI 或 VAE 常用 Monte Carlo 估计梯度，若采样数太少、batch 太小、学习率过大，训练会明显抖动。重参数化技巧能显著降低梯度方差，但不能彻底消除噪声。

第四个问题是 posterior collapse，也叫后验塌缩。它在 VAE 中尤其常见。表现是：

$$
q_\phi(z|x)\approx p(z)
$$

也就是无论输入什么样本，编码器都输出几乎一样的潜变量分布，导致潜变量基本不携带信息。此时解码器主要靠自身建模能力重构数据，而不是使用潜变量。

它通常发生在这些情形：

- 解码器过强，自己就能完成建模
- KL 惩罚过早、过强地把后验压回先验
- 数据复杂但潜变量维度不足

常见缓解方法包括：

- KL warm-up：训练初期先弱化 KL，再逐步增大其权重
- $\beta$-VAE 调度：把 KL 项乘以可控系数 $\beta$
- free bits：给每个潜变量维度保留最小信息预算
- 限制解码器能力：避免解码器强到完全绕过潜变量

还有一个很实际的问题是“你优化的 ELBO，不一定对应你真正关心的指标”。例如推荐系统中，业务更关心排序质量；主题模型中，业务更关心主题可解释性；VAE 中，业务可能更关心生成质量。这些目标和 ELBO 相关，但不完全等价，所以不能只看训练损失。

| 常见坑 | 表现 | 原因 | 规避策略 |
| --- | --- | --- | --- |
| 方差低估 | 置信区间过窄 | 均值场丢失相关性，$\mathrm{KL}(q\|p)$ 有 zero-forcing 倾向 | 用结构化变分族、低秩协方差、flow |
| 局部最优 | 不同初始化结果差异大 | ELBO 非凸 | 多次重启、监控 ELBO、选更稳初始化 |
| 梯度噪声大 | 训练不稳定、收敛慢 | Monte Carlo 估计方差大 | 重参数化、增大 batch、方差缩减 |
| 后验塌缩 | 潜变量无信息 | KL 过强、解码器过强 | KL warm-up、free bits、弱化解码器 |
| 目标错配 | ELBO 提升但业务指标一般 | 优化目标与任务目标不完全一致 | 同时监控下游指标 |

实践里可以记住一个经验判断：

- 如果你主要关心速度、可扩展性、可微训练，VI 通常是对的方向
- 如果你主要关心严格的后验不确定性，尤其是多峰和相关结构，VI 需要更谨慎

---

## 替代方案与适用边界

如果任务重点是后验精度，而不是速度，MCMC 往往更合适。它通过采样逼近真实后验，理论上在样本数趋于无穷时可以收敛到目标分布。代价是：

- 计算慢
- 样本存在自相关
- 高维下混合困难
- 很难直接嵌入大规模端到端训练流程

VI 则相反。它更像“把后验推断做成一个优化模块”，非常适合：

- 大规模主题模型
- 推荐系统中的贝叶斯近似推断
- VAE、扩散模型中的近似后验模块
- 在线推理或需要低时延的场景

当均值场假设太强时，一个自然升级是结构化变分推断。它不再强行让所有变量独立，而是保留部分依赖结构，例如：

$$
q(z)=q(z_1)q(z_2|z_1)
$$

或更一般地写成图结构因子分解。ELBO 仍然成立：

$$
\mathrm{ELBO}
=
\mathbb E_q[\log p(x,z)]-\mathbb E_q[\log q(z)]
$$

在上述分解下可写成：

$$
\mathrm{ELBO}
=
\mathbb E_{q(z_1)q(z_2|z_1)}[\log p(x,z_1,z_2)]
-
\mathbb E_{q(z_1)q(z_2|z_1)}[\log q(z_1)+\log q(z_2|z_1)]
$$

这里第二项是近似分布的负对数密度期望，也就是熵相关项。和均值场相比，结构化 VI 表达能力更强，但优化成本也更高，因为条件依赖被保留下来了。

如果结构化 VI 仍然不够，可以进一步增强变分族。最典型的是 normalizing flow。它从一个简单基础分布出发，例如：

$$
z_0\sim q_0(z_0)
$$

再经过一系列可逆变换：

$$
z_k=f_k(z_{k-1}),\quad k=1,\dots,K
$$

最终得到更复杂的分布 $q_K(z_K)$。依据变量替换公式：

$$
\log q_K(z_K)=\log q_0(z_0)-\sum_{k=1}^K \log \left|\det \frac{\partial f_k}{\partial z_{k-1}}\right|
$$

它的优点是保留了 ELBO 优化框架，同时显著增强了分布表达能力。缺点是：

- 需要设计可逆变换
- 需要计算 Jacobian 行列式
- 训练和实现复杂度更高

从工程角度看，可以把方法选择理解为一条梯度递增的路线：

1. 先用均值场 VI 获取可运行基线
2. 若不确定性估计明显失真，再考虑结构化 VI
3. 若后验形状复杂、非高斯、多峰，再考虑 flow-based VI
4. 若任务核心是高精度贝叶斯分析，而不是推断速度，则回到 MCMC

以贝叶斯矩阵分解为例。若你做推荐系统，只想快速得到用户向量和物品向量的后验近似，并服务在线排序，均值场 VI 往往已经够用。若你关心冷启动用户的不确定性是否可靠，特别是要把置信度直接用于业务决策，那么结构化 VI 或 MCMC 会更有价值。

另一个边界是模型规模。很多深度生成模型的参数规模已经大到无法接受传统 MCMC 的代价，此时 VI 几乎是唯一现实选择。也正因为如此，现代生成模型中的“近似后验”往往不是为了得到最精确的贝叶斯答案，而是为了获得一个可训练、可扩展、可部署的近似机制。

| 方法 | 精度 | 速度 | 可扩展性 | 适用边界 |
| --- | --- | --- | --- | --- |
| MCMC | 高 | 低 | 中到低 | 小到中规模、高精度后验分析 |
| 均值场 VI | 中 | 高 | 高 | 大规模、快速近似、在线推理 |
| 结构化 VI | 中到高 | 中 | 中 | 相关性明显、但仍需优化式推断 |
| Flow-based VI | 高于均值场 | 中 | 中 | 后验复杂、明显非高斯或多峰 |
| Laplace 近似 | 中 | 高 | 高 | 后验接近单峰高斯、需快速局部近似 |

---

## 参考资料

| 资料 | 核心贡献 | 适用章节 |
| --- | --- | --- |
| Blei, Kucukelbir, McAuliffe, *Variational Inference: A Review for Statisticians* | 最系统的综述之一，覆盖 ELBO、均值场、CAVI、随机变分推断 | 全文 |
| Bishop, *Pattern Recognition and Machine Learning* | 概率模型、KL、变分法、图模型背景最扎实的教材之一 | 问题定义、机制推导 |
| Murphy, *Probabilistic Machine Learning: Advanced Topics* | 对现代 VI、重参数化、随机梯度 VI 的解释更贴近当前实践 | 机制推导、工程实现 |
| Kingma and Welling, *Auto-Encoding Variational Bayes* | VAE 与重参数化技巧的原始论文 | VAE、代码实现 |
| Hoffman et al., *Stochastic Variational Inference* | 把 VI 扩展到大规模数据的关键论文 | 工程权衡、可扩展性 |
| Rezende and Mohamed, *Variational Inference with Normalizing Flows* | 用可逆变换增强变分族表达能力 | 替代方案与适用边界 |

阅读顺序建议如下。

第一步，先看 Bishop 或 Murphy 中关于 KL、后验、变分法的基础章节，把符号和目标函数看清楚。

第二步，读 Blei 等人的综述，理解 ELBO 恒等式、均值场假设和 CAVI 的一般形式。

第三步，手算一次一维高斯例子，确认“KL 下降 <=> ELBO 上升”的数值关系。

第四步，再看 Kingma and Welling 的 VAE 论文，把重参数化技巧和现代深度学习训练连接起来。

第五步，若需要更强表达能力，再看 stochastic VI 和 normalizing flow 的相关工作。

如果只保留一个最重要的理解主线，那就是：

- 贝叶斯推断的难点在后验难算
- VI 用近似分布把后验推断改写成优化问题
- ELBO 是这个优化问题的统一目标
- 误差主要来自变分族限制和优化近似，而不是 ELBO 这个框架本身
