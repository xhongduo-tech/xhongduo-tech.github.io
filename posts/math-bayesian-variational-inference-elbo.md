## 核心结论

变分推断（Variational Inference, VI）处理的是这样一类问题：真实后验分布 $p(z\mid x)$ 难以直接计算，于是不用“把后验精确算出来”，而是选一个可计算、可优化的分布族 $\mathcal Q$，在其中找一个分布 $q(z)$ 去逼近它。这里的“后验”是观测到数据 $x$ 之后，对潜变量 $z$ 的概率更新；“潜变量”是模型里参与生成数据、但没有被直接观测到的变量。

主公式只有一条：

$$
\log p(x)=\mathrm{ELBO}(q)+\mathrm{KL}\bigl(q(z)\|p(z\mid x)\bigr)
$$

它直接推出三个结论：

1. 对固定数据 $x$，$\log p(x)$ 是常数，不随 $q$ 改变。
2. 因为 $\mathrm{KL}(\cdot)\ge 0$，所以 $\mathrm{ELBO}(q)\le \log p(x)$，因此 ELBO 是证据下界。
3. 最大化 ELBO，等价于最小化 $\mathrm{KL}(q\|p)$，也就是让近似分布 $q(z)$ 尽量贴近真实后验 $p(z\mid x)$。

这条分解式的意义不在“名字”，而在它把一个积分问题改写成了一个优化问题。原本难点是：

$$
p(x)=\int p(x,z)\,dz
$$

一旦这个积分高维、无解析解或者成本过高，直接求后验就会卡住。ELBO 的作用是绕开这个积分，转而优化一个可计算目标。

先看最小例子。设先验与观测模型为：

$$
z\sim \mathcal N(0,1),\qquad x\mid z\sim \mathcal N(z,\sigma_x^2)
$$

再假设变分分布是高斯：

$$
q(z)=\mathcal N(\mu,s^2)
$$

那么 ELBO 可以写成：

$$
\mathrm{ELBO}(q)=\mathbb E_q[\log p(x\mid z)]-\mathrm{KL}\bigl(q(z)\|p(z)\bigr)
$$

其中第一项叫重构项或似然项，第二项可以看作正则项。对这个模型，有：

$$
\mathbb E_q[(x-z)^2]=(x-\mu)^2+s^2
$$

以及

$$
\mathrm{KL}\bigl(\mathcal N(\mu,s^2)\|\mathcal N(0,1)\bigr)
=
\frac12\left(\mu^2+s^2-1-\log s^2\right)
$$

所以 ELBO 可以完全写成 $\mu,s^2$ 的显式函数。这说明一件关键事实：即使后验本身不容易直接写出，优化目标仍然可能是可算、可求导、可迭代更新的。

真实工程里，VAE 是 VI 最常见的例子。VAE 的编码器输出 $\mu_\phi(x),\sigma_\phi(x)$，也就是为每个输入 $x$ 构造一个近似后验：

$$
q_\phi(z\mid x)=\mathcal N\bigl(z;\mu_\phi(x),\mathrm{diag}(\sigma_\phi^2(x))\bigr)
$$

训练时最大化 ELBO，本质上是在同时学两件事：

1. 学一个压缩映射，让输入 $x$ 能被编码到潜变量 $z$。
2. 学一个受约束的潜空间，让编码结果既能重构数据，又不要偏离先验太远。

如果只记一句话，那么应该记这句：  
**变分推断的核心不是“猜后验长什么样”，而是“把后验计算转成可优化的下界目标”。**

---

## 问题定义与边界

标准贝叶斯推断要计算：

$$
p(z\mid x)=\frac{p(x,z)}{p(x)}
$$

分子 $p(x,z)$ 往往由模型定义，可以直接写出；真正困难的是分母：

$$
p(x)=\int p(x,z)\,dz
$$

这个量叫证据（evidence）或边缘似然（marginal likelihood）。它表示模型在“把潜变量积分掉之后”，生成观测数据 $x$ 的概率。对简单模型，这个积分可能有闭式解；对高维模型、层级模型、神经网络参数化模型，它通常不可解析，或者数值代价过高。

变分推断把问题改写为：

$$
q^*(z)=\arg\min_{q\in \mathcal Q}\mathrm{KL}\bigl(q(z)\|p(z\mid x)\bigr)
$$

这里有三个对象必须区分清楚：

| 记号 | 含义 | 是否真实存在 | 是否容易计算 |
|---|---|---|---|
| $p(x,z)$ | 联合分布 | 是 | 通常可写出 |
| $p(z\mid x)$ | 真实后验 | 是 | 通常难算 |
| $q(z)$ | 近似后验 | 人为构造 | 设计成易算 |

$\mathcal Q$ 是事先指定的变分族，也就是候选近似分布的集合。它不是从数据里自动长出来的，而是建模者提前选定的。例如：

- 所有对角协方差高斯分布；
- 所有均场分布；
- 保留部分依赖结构的图模型分布；
- 由神经网络参数化、再通过可逆变换增强的流模型分布。

这里的边界非常明确：  
VI 不是在所有可能分布里找最优，而是在你允许的分布族 $\mathcal Q$ 里找最优。于是误差来源天然分成两部分：

| 误差来源 | 原因 | 是否能靠优化完全消除 |
|---|---|---|
| 近似误差 | $\mathcal Q$ 表达能力不够 | 不能 |
| 优化误差 | 参数没训到位、局部最优、数值问题 | 可以部分缓解 |

这也是为什么“ELBO 很高”不等于“后验逼近已经很好”。如果变分族太弱，优化再彻底，也只能得到受限最优。

均场变分（mean-field VI）是最常见的入门设定：

$$
q(z)=\prod_{j=1}^m q_j(z_j)
$$

它做了一个强假设：近似后验中的各个潜变量彼此独立。这个假设的收益是推导和计算都大幅简化，代价是丢失变量之间的相关结构。

可以把真实后验和均场近似的差异压缩成下面这张表：

| 场景 | 真实后验 | 均场近似的能力 | 常见后果 |
|---|---|---|---|
| 各维几乎独立 | 接近轴对齐 | 能较好逼近 | 效果通常不错 |
| 存在线性相关 | 斜椭圆形 | 难表达 | 协方差被低估 |
| 多峰结构 | 多个峰 | 单峰均场难覆盖 | 容易只抓住一个峰 |
| 重尾分布 | 尾部厚 | 高斯均场难表达 | 不确定性失真 |

因此，问题定义不是“VI 能不能近似后验”，而是下面这个更具体的问题：

1. 当前任务需要多高精度的不确定性估计。
2. 当前算力预算允许多复杂的推断。
3. 选定的变分族是否足够表达任务中的相关性、偏态或多峰结构。

常见变分族的工程权衡如下：

| 变分族 | 形式 | 表达能力 | 计算复杂度 | 典型问题 |
|---|---|---:|---:|---|
| 均场 | $\prod_j q_j(z_j)$ | 低 | 低 | 忽略相关性，常低估方差 |
| 分块变分 | $\prod_b q_b(z_b)$ | 中 | 中 | 块划分依赖经验 |
| 结构化变分 | 保留部分依赖结构 | 高 | 高 | 推导和实现更复杂 |
| 正态化流变分 | 可逆变换增强分布 | 很高 | 较高 | 调参复杂，训练不稳 |

一开始就要接受这个边界：  
**VI 追求的不是“无偏恢复真实后验”，而是“在可承受成本下得到足够好的后验近似”。**

---

## 核心机制与推导

从联合分布开始：

$$
p(x,z)=p(x\mid z)p(z)
$$

对于任意满足 $q(z)>0$ 的分布 $q(z)$，有：

$$
\log p(x)=\log \int p(x,z)\,dz
$$

在积分里乘除 $q(z)$：

$$
\log p(x)=\log \int q(z)\frac{p(x,z)}{q(z)}\,dz
$$

把积分写成对 $q$ 的期望：

$$
\log p(x)=\log \mathbb E_{q(z)}\left[\frac{p(x,z)}{q(z)}\right]
$$

由于对数函数是凹函数，应用 Jensen 不等式：

$$
\log \mathbb E_q\left[\frac{p(x,z)}{q(z)}\right]
\ge
\mathbb E_q\left[\log \frac{p(x,z)}{q(z)}\right]
$$

于是得到下界：

$$
\log p(x)\ge \mathbb E_q[\log p(x,z)]-\mathbb E_q[\log q(z)]
$$

定义：

$$
\mathrm{ELBO}(q)=\mathbb E_q[\log p(x,z)]-\mathbb E_q[\log q(z)]
$$

这里第二项也可写成熵：

$$
\mathcal H(q)=-\mathbb E_q[\log q(z)]
$$

因此

$$
\mathrm{ELBO}(q)=\mathbb E_q[\log p(x,z)]+\mathcal H(q)
$$

这个形式很重要，因为它把目标拆成了两股力量：

| 项 | 作用 | 直观意义 |
|---|---|---|
| $\mathbb E_q[\log p(x,z)]$ | 鼓励 $q$ 关注高联合概率区域 | 贴近模型解释数据的区域 |
| $\mathcal H(q)$ | 鼓励分布不要过度塌缩 | 保持一定不确定性 |

再把后验写进去：

$$
p(z\mid x)=\frac{p(x,z)}{p(x)}
\quad\Longrightarrow\quad
\log p(x,z)=\log p(z\mid x)+\log p(x)
$$

代回 ELBO：

$$
\mathrm{ELBO}(q)
=
\mathbb E_q[\log p(z\mid x)+\log p(x)-\log q(z)]
$$

把常数 $\log p(x)$ 提出来：

$$
\mathrm{ELBO}(q)
=
\log p(x)+\mathbb E_q[\log p(z\mid x)-\log q(z)]
$$

再识别出 KL 散度定义：

$$
\mathrm{KL}\bigl(q(z)\|p(z\mid x)\bigr)
=
\mathbb E_q\left[\log \frac{q(z)}{p(z\mid x)}\right]
$$

于是得到主公式：

$$
\mathrm{ELBO}(q)=\log p(x)-\mathrm{KL}\bigl(q(z)\|p(z\mid x)\bigr)
$$

等价地写成：

$$
\log p(x)=\mathrm{ELBO}(q)+\mathrm{KL}\bigl(q(z)\|p(z\mid x)\bigr)
$$

这一步说明 ELBO 不是拍脑袋定义出来的目标，而是由恒等分解直接得到的。

### 玩具例子：单变量高斯模型

考虑模型：

$$
z\sim \mathcal N(0,1),\qquad x\mid z\sim \mathcal N(z,\sigma_x^2)
$$

变分分布取：

$$
q(z)=\mathcal N(\mu,s^2)
$$

先写似然项：

$$
\log p(x\mid z)= -\frac12\log(2\pi\sigma_x^2)-\frac{(x-z)^2}{2\sigma_x^2}
$$

对 $q$ 取期望：

$$
\mathbb E_q[\log p(x\mid z)]
=
-\frac12\log(2\pi\sigma_x^2)-\frac{1}{2\sigma_x^2}\mathbb E_q[(x-z)^2]
$$

而

$$
\mathbb E_q[(x-z)^2]=(x-\mu)^2+s^2
$$

所以：

$$
\mathbb E_q[\log p(x\mid z)]
=
-\frac12\log(2\pi\sigma_x^2)-\frac{(x-\mu)^2+s^2}{2\sigma_x^2}
$$

再写 KL 项。由于先验是标准正态：

$$
\mathrm{KL}\bigl(\mathcal N(\mu,s^2)\|\mathcal N(0,1)\bigr)
=
\frac12\left(\mu^2+s^2-1-\log s^2\right)
$$

于是 ELBO 显式变成：

$$
\mathrm{ELBO}(\mu,s^2)
=
-\frac12\log(2\pi\sigma_x^2)
-\frac{(x-\mu)^2+s^2}{2\sigma_x^2}
-\frac12\left(\mu^2+s^2-1-\log s^2\right)
$$

这说明在这个模型里，优化 VI 只是优化两个标量参数 $\mu,s^2$。它虽然简单，但把 ELBO 的结构完整暴露出来了。

为了帮助新手建立感受，可以把两个参数的作用直接对照：

| 参数 | 影响 | 变大时会怎样 |
|---|---|---|
| $\mu$ | 控制近似后验中心 | 更靠近能解释 $x$ 的区域，但偏离先验会增大 KL |
| $s^2$ | 控制近似后验宽度 | 太小会被 KL 惩罚，太大又会拉低似然项 |

在这个高斯共轭模型中，真实后验其实可以解析写出：

$$
p(z\mid x)=\mathcal N\left(
\frac{x}{1+\sigma_x^2},
\frac{\sigma_x^2}{1+\sigma_x^2}
\right)
$$

因此最优变分分布正好能与真实后验重合。这个例子不是为了说明“VI 总能精确”，而是为了说明“当变分族足够表达真实后验时，ELBO 优化会恢复它”。

### 均场更新公式

若潜变量为 $z=(z_1,\dots,z_m)$，并假设均场分解：

$$
q(z)=\prod_{j=1}^m q_j(z_j)
$$

固定除第 $j$ 个因子外的所有因子，对 ELBO 关于 $q_j$ 做变分优化，可得到：

$$
\log q_j^*(z_j)\propto \mathbb E_{i\ne j}[\log p(x,z)]
$$

更完整地写，存在归一化常数 $C$ 使得：

$$
\log q_j^*(z_j)=\mathbb E_{i\ne j}[\log p(x,z)] + C
$$

指数化后：

$$
q_j^*(z_j)\propto \exp\left(\mathbb E_{i\ne j}[\log p(x,z)]\right)
$$

这就是坐标上升变分推断（Coordinate Ascent Variational Inference, CAVI）。它的工作流程是：

1. 先初始化每个因子 $q_j(z_j)$。
2. 依次更新一个因子，其他因子固定不变。
3. 每次更新都不会降低 ELBO。
4. 重复迭代直到 ELBO 基本不再提升。

这里的“对其他变量取期望”可以理解成：  
更新 $q_j$ 时，不再把其余潜变量当作固定点估计，而是用它们当前的分布平均掉不确定性。

一个两变量例子更直观。若 $z=(z_1,z_2)$，则：

$$
\log q_1^*(z_1)\propto \mathbb E_{q_2}[\log p(x,z_1,z_2)]
$$

$$
\log q_2^*(z_2)\propto \mathbb E_{q_1}[\log p(x,z_1,z_2)]
$$

更新顺序通常写成：

| 轮次 | 操作 |
|---|---|
| 第 1 步 | 用当前 $q_2$ 更新 $q_1$ |
| 第 2 步 | 用新的 $q_1$ 更新 $q_2$ |
| 第 3 步 | 再回到更新 $q_1$ |
| 直到收敛 | 监控 ELBO 或参数变化 |

CAVI 适合共轭结构明显、条件分布容易写成指数族形式的模型。到了深度生成模型里，显式坐标更新通常不再可行，就会转向随机梯度变分推断。

### 重参数化技巧

当变分分布由神经网络参数化，比如：

$$
q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\mathrm{diag}(\sigma_\phi^2(x)))
$$

训练的难点变成：  
如何对

$$
\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
$$

关于 $\phi$ 求梯度。

直接从 $q_\phi(z\mid x)$ 采样会把随机性放在参数里，导致梯度估计方差大。重参数化把采样写成：

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot \epsilon,\qquad \epsilon\sim \mathcal N(0,I)
$$

这样改写后，随机性只来自 $\epsilon$，而 $\epsilon$ 与参数 $\phi$ 无关。于是期望可以改写为：

$$
\mathbb E_{q_\phi(z\mid x)}[f(z)]
=
\mathbb E_{\epsilon\sim \mathcal N(0,I)}
\left[f\bigl(\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon\bigr)\right]
$$

这个改写的收益是梯度能穿过采样步骤：

$$
\nabla_\phi \mathbb E_{q_\phi(z\mid x)}[f(z)]
=
\nabla_\phi \mathbb E_{\epsilon}[f(g_\phi(x,\epsilon))]
$$

其中 $g_\phi(x,\epsilon)=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$。

对新手来说，最容易混淆的是“为什么只是换个写法，梯度就稳定了”。关键不在形式变化本身，而在于：

| 写法 | 随机性位置 | 对参数求导时的问题 |
|---|---|---|
| 直接采样 $z\sim q_\phi(z\mid x)$ | 随机节点依赖参数 $\phi$ | 梯度难传、方差大 |
| 重参数化 $z=g_\phi(x,\epsilon)$ | 随机性移到 $\epsilon$ | 可直接反向传播 |

这一步是 VAE 能高效训练的关键技术之一。没有重参数化也能训练，但通常需要分数函数估计器（score function estimator），方差更大，调试成本更高。

---

## 代码实现

先给一个最小、可直接运行的 Python 例子。它演示单变量高斯模型的 ELBO，并用简单梯度上升验证：最优的变分参数会收敛到解析后验附近。

```python
import math


def elbo(x, mu, log_var, obs_var=0.25):
    """ELBO for:
       z ~ N(0, 1)
       x|z ~ N(z, obs_var)
       q(z) = N(mu, exp(log_var))
    """
    var = math.exp(log_var)

    expected_log_lik = (
        -0.5 * math.log(2.0 * math.pi * obs_var)
        - 0.5 * ((x - mu) ** 2 + var) / obs_var
    )

    kl_q_p = 0.5 * (mu * mu + var - 1.0 - log_var)

    return expected_log_lik - kl_q_p


def grad_elbo(x, mu, log_var, obs_var=0.25):
    """Analytic gradients of ELBO w.r.t. mu and log_var."""
    var = math.exp(log_var)

    # d/dmu
    dmu = (x - mu) / obs_var - mu

    # d/d(log_var) = d/dvar * dvar/d(log_var) = d/dvar * var
    # ELBO = const - 0.5 * var / obs_var - 0.5 * (var - 1 - log_var)
    dlog_var = -0.5 * var / obs_var - 0.5 * var + 0.5

    return dmu, dlog_var


def exact_posterior_params(x, obs_var=0.25):
    """Posterior for prior N(0,1), likelihood x|z ~ N(z, obs_var)."""
    post_var = obs_var / (1.0 + obs_var)
    post_mu = x / (1.0 + obs_var)
    return post_mu, post_var


def optimize_vi(x, obs_var=0.25, lr=0.05, steps=400):
    mu = 0.0
    log_var = 0.0

    for step in range(steps):
        dmu, dlog_var = grad_elbo(x, mu, log_var, obs_var)

        mu += lr * dmu
        log_var += lr * dlog_var

        if step % 50 == 0 or step == steps - 1:
            value = elbo(x, mu, log_var, obs_var)
            print(
                f"step={step:03d} "
                f"elbo={value:.6f} "
                f"mu={mu:.6f} "
                f"var={math.exp(log_var):.6f}"
            )

    return mu, math.exp(log_var)


if __name__ == "__main__":
    x = 1.2
    obs_var = 0.25

    vi_mu, vi_var = optimize_vi(x, obs_var=obs_var)
    post_mu, post_var = exact_posterior_params(x, obs_var=obs_var)

    print("\noptimized variational parameters")
    print("mu =", round(vi_mu, 6))
    print("var =", round(vi_var, 6))

    print("\nexact posterior parameters")
    print("mu =", round(post_mu, 6))
    print("var =", round(post_var, 6))

    assert abs(vi_mu - post_mu) < 1e-3
    assert abs(vi_var - post_var) < 1e-3
```

这段代码有几个初学者容易忽略的点：

| 点 | 说明 |
|---|---|
| 用 `log_var` 而不是 `var` | 方差必须为正，优化对数方差更稳定 |
| 直接优化 ELBO | 所以参数更新是“加上梯度”，即梯度上升 |
| 用解析梯度而不是数值差分 | 便于看清 ELBO 对参数的真实作用 |
| 最后和解析后验对比 | 用来验证实现没有推错符号 |

如果你运行这段代码，会看到 `mu` 收敛到 $\frac{x}{1+\sigma_x^2}$，`var` 收敛到 $\frac{\sigma_x^2}{1+\sigma_x^2}$。这不是巧合，而是因为当前变分族刚好能表示真实后验。

真实工程里，更常见的是 VAE。其 ELBO 一般写成：

$$
\mathrm{ELBO}
=
\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
\mathrm{KL}\bigl(q_\phi(z\mid x)\|p(z)\bigr)
$$

其中：

- 编码器定义 $q_\phi(z\mid x)$；
- 解码器定义 $p_\theta(x\mid z)$；
- 训练时同时优化 $\phi,\theta$；
- 实现里通常最小化 $-\mathrm{ELBO}$。

下面给一个真正可运行的 PyTorch 最小示例。它不是完整训练脚本，但单步前向与损失计算是完整的。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc_out(h))


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=16):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var


def vae_loss(x, x_recon, mu, log_var):
    # x and x_recon: [batch, 784], values in [0, 1]
    recon = F.binary_cross_entropy(x_recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = recon + kl
    return loss, recon, kl


def vae_step(model, optimizer, x):
    model.train()
    optimizer.zero_grad()

    x_recon, mu, log_var = model(x)
    loss, recon, kl = vae_loss(x, x_recon, mu, log_var)

    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "recon": float(recon.detach()),
        "kl": float(kl.detach()),
    }
```

这个实现里有几个容易混淆的工程点：

| 问题 | 正确理解 |
|---|---|
| 为什么 `loss = recon + kl`？ | 因为代码最小化的是负 ELBO |
| 为什么编码器输出 `log_var`？ | 避免方差出现负值，并改善数值稳定性 |
| 为什么 `decoder` 末尾用 `sigmoid`？ | 因为这里假设输入像素归一化到 $[0,1]$，重构分布近似为 Bernoulli |
| `reduction="sum"` 还是 `"mean"`？ | 都能用，但要与 KL 的归一化方式保持一致，否则权重会失衡 |

如果输入不是二值或归一化图像，重构项也要相应变化。例如：

| 数据类型 | 常见重构分布 | 常见损失 |
|---|---|---|
| 二值图像 | Bernoulli | BCE |
| 实值连续数据 | Gaussian | MSE 或高斯负对数似然 |
| 计数数据 | Poisson / NegBin | 对应对数似然 |
| 文本 token | Categorical | Cross-Entropy |

真实工程里，VAE 的优势不是“比 MCMC 更准”，而是“后验推断可以一次前向传播完成”。这使它特别适合：

- 大规模训练；
- 在线推理；
- 需要把潜变量当作下游特征的系统；
- 图像、文本、语音等高维输入建模。

---

## 工程权衡与常见坑

VI 的优势是快、可扩展、容易与 SGD 结合；代价是近似通常有偏，而且偏差受变分族和优化过程双重影响。下面列的是最常见的工程问题。

| 问题 | 表现 | 根因 | 缓解策略 |
|---|---|---|---|
| 均场独立假设过强 | 后验过窄，相关结构丢失 | 变分族太弱 | 分块变分、结构化变分、流模型 |
| 未做重参数化 | 梯度波动大，训练不稳 | 估计器方差高 | 重参数化、控制变量 |
| KL 项过强 | 潜变量塌缩 | 编码器学成接近先验 | KL annealing、free bits、$\beta$ 调整 |
| 局部最优明显 | 不同初始化差异大 | 目标非凸 | 多次初始化、监控 ELBO |
| 数值不稳定 | NaN、方差爆炸 | 指数参数化和极端输入 | clamp、梯度裁剪、标准化 |
| 重构项失衡 | 只顾重构或只顾 KL | 标度不匹配 | 检查 batch reduction 和 loss scaling |

### 均场为什么会低估协方差

真实后验若有强相关，例如：

$$
p(z_1,z_2\mid x)
$$

的高概率区域沿一条斜线分布，那么均场近似

$$
q(z_1,z_2)=q_1(z_1)q_2(z_2)
$$

无法表达这种斜向耦合。它只能用“轴对齐”的形状去逼近。结果通常是：

1. 边缘均值可能还不错。
2. 联合协方差表达不出来。
3. 每个维度的区间往往被压得过窄。

这类误差在只看重构误差时不一定明显，但在依赖不确定性的任务中会暴露出来，例如：

- 风险评估；
- 主动学习；
- 贝叶斯优化；
- 医疗决策和科学计算。

### 梯度噪声过大时怎么办

如果训练中看到这些现象：

- loss 曲线剧烈抖动；
- 同一配置复现实验结果差很多；
- 学习率一大就发散；
- KL 项忽大忽小；

优先检查的不是模型结构，而是梯度估计方式。对连续潜变量，重参数化几乎是默认选项。对离散潜变量，则需要其他近似方法，例如：

| 场景 | 常见方法 |
|---|---|
| 离散类别潜变量 | Gumbel-Softmax |
| 通用高方差梯度估计 | REINFORCE + baseline |
| 更复杂控制变量方法 | RELAX / NVIL |

### KL 项过强与潜变量塌缩

在 VAE 中，如果解码器过强，模型可能学会“忽略潜变量”。表现为：

$$
q_\phi(z\mid x)\approx p(z)
$$

此时 KL 很小，看起来训练很稳定，但潜变量几乎不携带输入信息，这叫 posterior collapse。常见缓解手段：

| 方法 | 思路 |
|---|---|
| KL annealing | 训练初期减弱 KL，让模型先学会重构 |
| Free bits | 给每个 latent 维度一定的信息预算 |
| $\beta$-VAE 调权 | 显式控制 KL 强度 |
| 弱化解码器能力 | 避免解码器完全绕开 latent |

### 新手最容易犯的实现错误

| 错误 | 后果 |
|---|---|
| KL 公式符号写反 | 训练目标完全错误 |
| `log_var` 与 `std` 混用 | 方差尺度错一倍或指数错位 |
| BCE 用在未归一化连续值上 | 重构项失真 |
| batch 上 `sum`/`mean` 不一致 | KL 与重构项相对权重异常 |
| 忽略随机种子和初始化 | 结果难复现 |
| 只看总 loss，不分开看 recon 和 KL | 无法判断训练具体出了什么问题 |

一个实用习惯是训练时同时记录三条曲线：

1. 总 loss 或负 ELBO。
2. 重构项。
3. KL 项。

只看总 loss，经常会把问题掩盖掉。

---

## 替代方案与适用边界

VI 只是后验近似的一条路线，它的特点是“把积分问题改写成优化问题”。如果需求不是速度优先，未必一定该选 VI。常见替代方法如下：

| 方法 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 变分推断 VI | 大数据、在线推理、深度生成模型 | 快，易与 SGD 结合 | 近似有偏，依赖变分族 |
| MCMC / HMC | 精度优先、离线分析、科学计算 | 渐近更准确，可恢复复杂后验 | 慢，调参和混合成本高 |
| Laplace 近似 | 后验单峰、局部近似即可 | 简单，工程成本低 | 多峰或强非高斯时失真大 |
| 期望传播 EP | 局部因子结构明显的模型 | 某些场景近似质量高 | 理论和实现复杂 |
| 粒子方法 / SMC | 时序模型、在线更新 | 能表示更复杂分布 | 粒子退化、计算成本高 |

可以用下面这个判断框架来选方法：

| 你最关心什么 | 更合适的方法 |
|---|---|
| 在线服务速度 | VI |
| 大规模神经网络训练 | VI |
| 高精度不确定性估计 | MCMC / HMC |
| 后验多峰结构 | MCMC、粒子方法 |
| 只要局部高斯近似 | Laplace |
| 共轭图模型局部消息传递 | EP 或 CAVI |

如果给同一模型 30 分钟预算，常见现实是：

- VI 能较快收敛到一个稳定近似；
- HMC 可能还在 warmup 或混合阶段；
- 但如果多给时间，HMC 往往能恢复更真实的相关性和尾部行为。

所以真正的选择标准不是“谁更高级”，而是下面四个问题：

1. 你是否需要实时或大批量推理。
2. 你是否需要可靠的不确定性区间，而不只是点估计。
3. 后验是否强相关、多峰、重尾。
4. 你能否接受“有偏但快”的近似。

一句话压缩适用边界：

**如果目标是大规模训练和快速部署，VI 很合适；如果目标是高精度后验与不确定性分析，尤其在层级贝叶斯、科学建模、药物与因果推断里，MCMC 往往更可靠。**

---

## 参考资料

1. David M. Blei, Alp Kucukelbir, Jon D. McAuliffe, *Variational Inference: A Review for Statisticians*  
   链接：https://arxiv.org/abs/1601.00670  
   价值：最系统的综述之一，适合建立完整框架，包括 ELBO、均场、随机变分推断与应用边界。

2. Michael I. Jordan, Zoubin Ghahramani, Tommi S. Jaakkola, Lawrence K. Saul, *An Introduction to Variational Methods for Graphical Models*  
   链接：https://people.eecs.berkeley.edu/~jordan/variational.html  
   价值：经典文献，适合理解“变分”在概率图模型中的原始语境，以及为什么它本质上是优化近似。

3. Jason Eisner, *Tutorial on Variational Approximation Methods*  
   链接：https://www.cs.jhu.edu/~jason/tutorials/variational  
   价值：适合查基本定义、KL 方向、ELBO 直觉和 Jensen 推导，入门友好。

4. Carl Doersch, *Tutorial on Variational Autoencoders*  
   链接：https://arxiv.org/abs/1606.05908  
   价值：适合理解 VAE、重参数化技巧，以及 ELBO 在神经网络语境中的具体含义。

5. Diederik P. Kingma, Max Welling, *Auto-Encoding Variational Bayes*  
   链接：https://arxiv.org/abs/1312.6114  
   价值：VAE 原始论文，适合核对重参数化形式与训练目标的标准写法。

6. Matthew D. Hoffman, David M. Blei, Chong Wang, John Paisley, *Stochastic Variational Inference*  
   链接：https://jmlr.org/papers/v14/hoffman13a.html  
   价值：适合进一步理解大规模数据下为什么 VI 能和随机优化自然结合。

7. Suzy Ahyah, *Coordinate Ascent Variational Inference*  
   链接：https://suzyahyah.github.io/bayesian%20inference/machine%20learning/2019/03/20/CAVI.html  
   价值：适合配合均场假设看 CAVI 更新公式，把“单因子更新”与 ELBO 推导串起来。

8. Lei Mao, *Introduction to Variational Inference*  
   链接：https://leimao.github.io/article/Introduction-to-Variational-Inference/  
   价值：适合理解均场近似的局限，以及为什么独立性假设会带来系统偏差。

9. Wikipedia, *Reparameterization trick*  
   链接：https://en.wikipedia.org/wiki/Reparameterization_trick  
   价值：适合快速核对重参数化定义、常见写法和相关术语。

查证建议：

- 要核对主公式，优先看 Blei 综述、Jason 教程和 Jordan 的经典综述。
- 要核对均场更新，重点看 CAVI 公式
  $$
  \log q_j^*(z_j)\propto \mathbb E_{-j}[\log p(x,z)]
  $$
  的变分推导。
- 要核对 VAE 与重参数化，优先看 Kingma-Welling 原论文和 Doersch 教程。
- 要核对“VI 快但有偏”这类工程判断，优先看综述类资料，不要只看博客总结。
- 要核对实现细节，特别是 KL 项符号、`log_var` 写法和重构项形式，最好同时对照论文与可运行代码。
