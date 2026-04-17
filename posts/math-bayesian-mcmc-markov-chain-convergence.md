## 核心结论

MCMC，Markov Chain Monte Carlo，本质上是两件事的组合：

1. 用马尔可夫链生成一串相关样本。
2. 让这条链的长期分布等于目标分布 $\pi$。

因此，MCMC 追求的不是“每一步都抽得很准”，而是“跑得足够久之后，样本出现的频率逼近 $\pi$$”。如果链满足合适的转移条件，那么对任意可积函数 $f$，样本均值会逼近期望：
$$
\frac{1}{n}\sum_{t=1}^n f(X_t)\;\longrightarrow\; \mathbb{E}_{\pi}[f(X)]
$$

马尔可夫链收敛到目标分布时，通常会检查三个核心条件：

| 条件 | 数学含义 | 直白解释 | 不满足时的后果 |
|---|---|---|---|
| 平稳分布 | $\pi P=\pi$ | 目标分布在一步转移后不变 | 链会收敛到别的分布 |
| 不可约 | 任意状态之间最终可达 | 不会被困在某个局部子空间 | 只能覆盖部分状态 |
| 非周期 | 不按固定节奏循环 | 不会只在若干步长上来回跳 | 分布可能振荡，不稳定 |

在实际算法里，最常见的构造方式是让转移核 $P$ 满足细致平衡：
$$
\pi(x)P(x,y)=\pi(y)P(y,x)
$$
它比“$\pi$ 是平稳分布”更强，但验证更直接。只要细致平衡成立，再配合不可约和非周期，链就会向 $\pi$ 收敛。

MCMC 最常见的两类方法是 Metropolis-Hastings（MH）和 Gibbs 采样：

| 方法 | 更新方式 | 需要知道什么 | 接受率 | 典型场景 |
|---|---|---|---|---|
| Metropolis-Hastings | 先提候选，再决定是否接受 | 目标密度比值，必要时加 proposal 比值 | $\le 1$ | 连续参数、条件分布难直接采样 |
| Gibbs | 直接从条件分布更新一个变量或一个块 | 条件分布可显式采样 | 恒为 1 | 图模型、层级贝叶斯模型 |
| 随机游走 MH | 在当前点附近小步提议 | 常只需未归一化目标密度 | 依赖步长 | 入门简单，但混合可能慢 |
| 块 Gibbs | 一次更新一组变量 | 联合条件分布可采样 | 恒为 1 | 强相关变量较多时更稳定 |

一个最小直觉例子是标准高斯分布 $\pi(x)\propto e^{-x^2/2}$。如果链当前位置在 $x=0$，每次提议向左或向右移动一点，那么高密度区域附近的候选更容易被接受，低密度区域附近的候选更容易被拒绝。结果不是“链永远停在 0 附近”，而是“链在 0 附近停留更久，在尾部停留更短”。长期统计后，停留比例就会逼近高斯分布本身。

---

## 问题定义与边界

MCMC 解决的核心问题通常不是“把整个分布完整画出来”，而是估计复杂分布下的期望、方差、分位数或边际概率。标准形式是：
$$
\mathbb{E}_{\pi}[f(X)] = \int f(x)\pi(x)\,dx
$$

当下面两类情况出现时，MCMC 就有价值：

| 情况 | 说明 |
|---|---|
| 积分没有解析解 | 比如高维后验期望无法手算 |
| 不能直接独立采样 | 比如只知道未归一化后验 $\tilde{\pi}(x)$，不知道归一化常数 |

这时就构造一条极限分布为 $\pi$ 的马尔可夫链 $\{X_t\}$，再用样本均值近似积分：
$$
\hat{\mu}_n=\frac{1}{n}\sum_{t=1}^{n} f(X_t)
$$

这里必须明确边界。MCMC 的正确性是渐近正确，不是有限步精确。换句话说：

- 你不能指望跑 100 步就“保证正确”。
- 你必须同时关心“会不会收敛对”和“收敛得快不快”。

这两个问题对应两组概念。

### 1. 会不会收敛到正确分布

这部分关注极限分布是否真的是目标分布。常用检查项包括：

| 概念 | 数学对象 | 要回答的问题 |
|---|---|---|
| 平稳分布 | $\pi P=\pi$ | 链长期是否保持目标分布不变 |
| 细致平衡 | $\pi(x)P(x,y)=\pi(y)P(y,x)$ | 转移机制是否与目标分布匹配 |
| 不可约 | 状态空间连通性 | 所有状态是否最终都能访问 |
| 非周期 | 返回步数的周期性 | 是否会被固定节奏锁住 |

如果这部分做错，后面所有诊断都没有意义。因为链即使“看起来很稳定”，也可能稳定在错误分布上。

### 2. 收敛得快不快

即使目标分布是对的，链也可能非常慢。这里要看两个量：混合时间和自相关。

混合时间描述的是链忘掉初始状态的速度。常见定义是
$$
t_{\mathrm{mix}}(\epsilon)=\min\left\{t:\sup_x \|P^t(x,\cdot)-\pi\|_{\mathrm{TV}}\le \epsilon\right\}
$$
其中总变差距离定义为
$$
\|\mu-\nu\|_{\mathrm{TV}}=\sup_A |\mu(A)-\nu(A)|
$$
它衡量当前分布与目标分布的最大事件概率差。

直白地说：

- 如果从很偏的初始值出发，比如 $x_0=100$，前几步明显还带着初始状态痕迹。
- 当分布距离足够小后，链才可以认为“基本混合”。

自相关描述的是样本之间有多相似。对某个统计量 $f(X_t)$，滞后 $k$ 的自相关定义为
$$
\rho_k=\mathrm{Corr}(f(X_0),f(X_k))
$$
如果 $\rho_k$ 衰减很慢，说明样本虽然很多，但重复信息很多。

| 指标 | 数学含义 | 工程含义 | 风险 |
|---|---|---|---|
| 混合时间 | 忘掉初始值所需步数 | 前多少步不该直接用 | burn-in 不足导致偏差 |
| 自相关 | 样本间相关程度 | 相邻样本像不像 | 方差大，信息增长慢 |
| 有效样本数 ESS | 等价独立样本个数 | 真正可用的信息量 | 样本数看起来多，信息量却少 |

ESS 的常见近似式是
$$
\mathrm{ESS}\approx \frac{N}{1+2\sum_{k=1}^{\infty}\rho_k}
$$
这个式子有一个直接结论：总样本数 $N$ 不是有效信息量。若自相关很强，$N=100000$ 也可能只相当于几百个独立样本。

### 3. 一个初学者例子：为什么“跑出来一串数”还不够

假设你想估计标准高斯分布的均值，理论答案是 0。下面三种情况看起来都“有样本”，但可信度完全不同：

| 情况 | 样本现象 | 是否可信 | 原因 |
|---|---|---|---|
| 链从 $x_0=0$ 开始，步长合适 | 样本在 0 附近上下波动 | 较可信 | 混合较快，自相关可接受 |
| 链从 $x_0=50$ 开始，只跑 30 步 | 样本还都在右尾区域 | 不可信 | 还没忘掉初始值 |
| 步长极小，接受率接近 1 | 样本一点点挪动 | 不可信 | 高接受率不等于高效率 |

因此，MCMC 的边界不是“能不能生成数”，而是“这些数是否已经代表目标分布”。

---

## 核心机制与推导

### 1. 为什么细致平衡足够重要

设状态空间是离散的，转移概率是 $P(x,y)$。如果对任意状态对 $(x,y)$ 都有
$$
\pi(x)P(x,y)=\pi(y)P(y,x)
$$
那么对任意状态 $y$，
$$
\sum_x \pi(x)P(x,y)=\sum_x \pi(y)P(y,x)
=\pi(y)\sum_x P(y,x)=\pi(y)
$$
于是得到
$$
(\pi P)(y)=\pi(y)
$$
也就是 $\pi$ 是平稳分布。

这段推导的重要性在于：你不需要直接证明长期极限，只需要先把“一步转移不改变 $\pi$”证明出来。后续再用不可约、非周期等条件保证从任意初值都能收敛过去。

对连续状态空间，求和换成积分：
$$
\int \pi(x)P(x,dy)\,dx=\pi(y)\,dy
$$
思想完全一样，都是“流入概率质量等于流出概率质量”。

### 2. MH 接受率是怎么来的

MH 的结构是：

1. 在当前位置 $x$ 先从 proposal $q(x'|x)$ 采样一个候选点 $x'$。
2. 再用接受率 $\alpha(x,x')$ 决定是否跳到 $x'$。

因此真实转移核是
$$
P(x,x')=q(x'|x)\alpha(x,x'), \quad x'\neq x
$$
而留在原地的概率是
$$
P(x,x)=1-\int q(x'|x)\alpha(x,x')\,dx'
$$

为了让细致平衡成立，希望对所有 $x\neq x'$ 有
$$
\pi(x)q(x'|x)\alpha(x,x')
=
\pi(x')q(x|x')\alpha(x',x)
$$

定义比值
$$
r(x,x')=\frac{\pi(x')q(x|x')}{\pi(x)q(x'|x)}
$$
取
$$
\alpha(x,x')=\min(1,r(x,x'))
$$
就得到标准 MH 接受率：
$$
\alpha(x,x')=
\min\left(
1,\frac{\pi(x')q(x|x')}{\pi(x)q(x'|x)}
\right)
$$

这个构造成立的原因可以分两种情况看：

| 情况 | $r(x,x')$ | 结果 |
|---|---|---|
| 候选更“合理” | $r\ge 1$ | $\alpha(x,x')=1$，直接接受 |
| 候选更“差” | $r<1$ | 以概率 $r$ 接受 |

于是总能满足
$$
\pi(x)q(x'|x)\min(1,r)
=
\pi(x')q(x|x')\min(1,1/r)
$$

如果 proposal 是对称的，即
$$
q(x'|x)=q(x|x')
$$
接受率简化为
$$
\alpha(x,x')=\min\left(1,\frac{\pi(x')}{\pi(x)}\right)
$$
这就是最常见的随机游走 MH。

### 3. 玩具例子：一维高斯上的 MH

目标分布是标准高斯：
$$
\pi(x)\propto e^{-x^2/2}
$$
设当前状态是 $x=0.5$，候选状态是 $x'=1.4$，并且 proposal 对称。则
$$
\alpha
=
\min\left(1,\frac{\pi(1.4)}{\pi(0.5)}\right)
=
\min\left(1,e^{-1.4^2/2+0.5^2/2}\right)
$$
具体计算：
$$
-1.4^2/2+0.5^2/2=-0.98+0.125=-0.855
$$
所以
$$
\alpha=\min(1,e^{-0.855})\approx 0.425
$$

这说明：

- 从 0.5 走到 1.4 并不是不允许。
- 只是因为 1.4 处的密度更低，所以接受概率只有约 42.5%。
- 如果这一步被拒绝，链保持在 0.5，不会强行移动。

再看一个反向例子。如果当前点是 $x=1.4$，候选点是 $x'=0.5$，则
$$
\alpha=\min\left(1,\frac{\pi(0.5)}{\pi(1.4)}\right)=1
$$
也就是从低密度区往高密度区移动时，通常会直接接受。

这个机制保证了两点同时成立：

| 目标 | MH 如何实现 |
|---|---|
| 高密度区更常被访问 | 进入高密度区容易，被接受概率高 |
| 低密度区也能被访问 | 进入低密度区不是绝对禁止，只是概率更低 |

### 4. Gibbs 为什么是特殊 MH

设多维变量为 $x=(x_1,\dots,x_d)$。Gibbs 采样每一步只更新一个条件分布，例如第 $i$ 维：
$$
x_i^{(t+1)} \sim \pi(x_i\mid x_{-i}^{(t)})
$$
其中 $x_{-i}$ 表示除第 $i$ 维外的其余变量。

这相当于 proposal 就是真实条件分布：
$$
q(x_i' \mid x_{-i})=\pi(x_i' \mid x_{-i})
$$
把它代入 MH 接受率，可以得到 proposal 比值与目标密度比值完全抵消，最终
$$
\alpha=1
$$

更细一点地写，设新状态与旧状态只在第 $i$ 维不同，则
$$
\pi(x)=\pi(x_i,x_{-i}), \qquad \pi(x')=\pi(x_i',x_{-i})
$$
proposal 为
$$
q(x'|x)=\pi(x_i'|x_{-i}),\qquad q(x|x')=\pi(x_i|x_{-i})
$$
于是
$$
\frac{\pi(x')q(x|x')}{\pi(x)q(x'|x)}
=
\frac{\pi(x_i',x_{-i})\pi(x_i|x_{-i})}{\pi(x_i,x_{-i})\pi(x_i'|x_{-i})}
=1
$$
因此接受率恒为 1。

所以 Gibbs 不是另一套完全无关的思想，而是 MH 在“条件分布可直接采样”这一特殊场景下的特例。

### 5. 机制小结

| 步骤 | 做了什么 | 为什么需要 |
|---|---|---|
| 定义 proposal $q$ | 先产生候选状态 | 让链能探索状态空间 |
| 设计接受率 $\alpha$ | 决定是否接受候选 | 保证目标分布正确 |
| 检查细致平衡 | 验证 $\pi$ 为平稳分布 | 确保长期目标正确 |
| 检查不可约与非周期 | 确保从任意初值都能到达极限 | 保证真正收敛 |
| 诊断混合与自相关 | 评估效率 | 判断样本是否足够有用 |

---

## 代码实现

下面给出一个可以直接运行的 Python 示例。它做四件事：

1. 用随机游走 MH 采样标准高斯分布。
2. 丢弃 burn-in 阶段。
3. 计算接受率、样本均值、样本方差、滞后自相关和 ESS。
4. 对不同 proposal 步长做对比，让“步长过小”和“步长过大”的差异直接可见。

```python
import math
import random
from typing import List, Tuple


def log_target(x: float) -> float:
    """标准高斯的未归一化对数密度。"""
    return -0.5 * x * x


def mh_sample(
    n_samples: int,
    proposal_std: float = 1.0,
    burn_in: int = 1000,
    thinning: int = 1,
    x0: float = 0.0,
    seed: int | None = None,
) -> Tuple[List[float], float]:
    """
    随机游走 Metropolis-Hastings.

    参数:
        n_samples: 保留多少个样本
        proposal_std: proposal 标准差，必须 > 0
        burn_in: 前多少步作为热身期丢弃
        thinning: 每隔多少步保留一次样本，必须 >= 1
        x0: 初始状态
        seed: 随机种子，便于复现

    返回:
        chain: 长度为 n_samples 的样本序列
        acceptance_rate: 所有提议中的接受比例
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if proposal_std <= 0:
        raise ValueError("proposal_std must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thinning <= 0:
        raise ValueError("thinning must be positive")

    rng = random.Random(seed)
    x = x0
    chain: List[float] = []
    accepted = 0
    total_steps = burn_in + n_samples * thinning

    for step in range(total_steps):
        proposal = rng.gauss(x, proposal_std)

        # 对称 proposal 下，接受率只依赖目标密度比值
        log_alpha = log_target(proposal) - log_target(x)

        # 在对数域比较，避免直接算 exp(log_alpha) 带来的数值问题
        if math.log(rng.random()) < min(0.0, log_alpha):
            x = proposal
            accepted += 1

        if step >= burn_in and (step - burn_in) % thinning == 0:
            chain.append(x)

    acceptance_rate = accepted / total_steps
    return chain, acceptance_rate


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def variance(xs: List[float]) -> float:
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def autocorr(xs: List[float], lag: int) -> float:
    if lag <= 0 or lag >= len(xs):
        raise ValueError("lag must satisfy 1 <= lag < len(xs)")
    m = mean(xs)
    den = sum((x - m) ** 2 for x in xs)
    if den == 0:
        return 0.0
    num = sum((xs[i] - m) * (xs[i + lag] - m) for i in range(len(xs) - lag))
    return num / den


def integrated_autocorr_time(xs: List[float], max_lag: int = 200) -> float:
    """
    用截断和估计积分自相关时间:
        tau = 1 + 2 * sum_{k>=1} rho_k
    当 rho_k <= 0 时提前停止，是常见的简单工程近似。
    """
    max_lag = min(max_lag, len(xs) - 1)
    tau = 1.0
    for lag in range(1, max_lag + 1):
        rho = autocorr(xs, lag)
        if rho <= 0:
            break
        tau += 2.0 * rho
    return tau


def effective_sample_size(xs: List[float], max_lag: int = 200) -> float:
    tau = integrated_autocorr_time(xs, max_lag=max_lag)
    return len(xs) / tau


def summarize_chain(xs: List[float], acceptance_rate: float) -> dict:
    m = mean(xs)
    var = variance(xs)
    rho1 = autocorr(xs, 1)
    rho5 = autocorr(xs, 5)
    ess = effective_sample_size(xs, max_lag=200)
    return {
        "mean": m,
        "variance": var,
        "acceptance_rate": acceptance_rate,
        "lag1_autocorr": rho1,
        "lag5_autocorr": rho5,
        "ess": ess,
    }


def main() -> None:
    configs = [
        {"proposal_std": 0.1, "label": "small_step"},
        {"proposal_std": 1.0, "label": "balanced_step"},
        {"proposal_std": 8.0, "label": "large_step"},
    ]

    for cfg in configs:
        chain, acc = mh_sample(
            n_samples=5000,
            proposal_std=cfg["proposal_std"],
            burn_in=2000,
            thinning=1,
            x0=8.0,
            seed=42,
        )
        stats = summarize_chain(chain, acc)

        print(f"[{cfg['label']}] proposal_std={cfg['proposal_std']}")
        print(f"  mean            = {stats['mean']:.4f}")
        print(f"  variance        = {stats['variance']:.4f}")
        print(f"  acceptance_rate = {stats['acceptance_rate']:.4f}")
        print(f"  lag1_autocorr   = {stats['lag1_autocorr']:.4f}")
        print(f"  lag5_autocorr   = {stats['lag5_autocorr']:.4f}")
        print(f"  ess             = {stats['ess']:.1f}")
        print()

        # 标准高斯均值和方差的宽松检查
        assert abs(stats["mean"]) < 0.25, stats
        assert 0.7 < stats["variance"] < 1.3, stats
        assert 0.0 < stats["acceptance_rate"] < 1.0, stats
        assert -1.0 <= stats["lag1_autocorr"] <= 1.0, stats


if __name__ == "__main__":
    main()
```

这段代码能直接运行，且满足几个初学者最常踩坑的要求：

| 部件 | 作用 | 为什么这样写 |
|---|---|---|
| `log_target` | 定义未归一化对数密度 | MH 只需要密度比，不需要归一化常数 |
| `math.log(rng.random()) < min(0, log_alpha)` | 在对数域做接受判断 | 更稳定，避免概率域下溢 |
| `burn_in` | 丢掉前期未混合好的样本 | 减少初始值偏差 |
| `autocorr` / `ESS` | 估计样本相关性和有效样本数 | 样本数多不代表信息量大 |
| 三组步长配置 | 对比不同 proposal 尺度 | 帮助理解接受率与混合速度的权衡 |

如果用固定随机种子运行，通常会看到类似规律：

| 步长 | 接受率 | 自相关 | 典型现象 |
|---|---|---|---|
| 很小，如 0.1 | 很高 | 很强 | 几乎每步都接受，但移动太慢 |
| 适中，如 1.0 | 中等 | 较低 | 移动距离和接受率较平衡 |
| 很大，如 8.0 | 很低 | 很强 | 多数提议被拒绝，链频繁原地停留 |

这里有两个新手常见误解需要单独指出。

### 1. 接受率高不一定好

如果 proposal 步长极小，链每次都只移动一点点，接受率可以非常高，但样本之间极其相似。你得到的是“很多个几乎一样的样本”，不是高质量探索。

### 2. thinning 不是效率补丁

thinning 的作用主要是减少存储量或降低后处理成本。它不会提高原始链的信息效率。一个常见事实是：

- 保留全部样本再计算 ESS，往往比“每隔 10 步只留 1 个”更有信息。
- 真正应该优先优化的是 proposal、参数化和算法本身。

### 3. 一个工程化理解例子

假设你在做贝叶斯逻辑回归，参数是 $\beta\in\mathbb{R}^d$，后验分布为
$$
p(\beta \mid X,y)\propto p(y\mid X,\beta)\,p(\beta)
$$
这个后验通常没有简单闭式，也不能直接独立采样。若使用随机游走 MH：

- proposal 太小，链在参数空间里挪动缓慢；
- proposal 太大，候选点常落在低后验区域，接受率很低；
- 数据尺度差异很大时，某些维度难走，链会沿狭长方向慢慢爬。

这就是为什么实际工程中，MCMC 的问题通常不是“能不能写出代码”，而是“转移核与目标分布的几何结构是否匹配”。

---

## 工程权衡与常见坑

MCMC 理论上讲的是“只要链定义正确，就能渐近收敛”。工程上真正难的地方，是如何确保“定义正确”和“效率可接受”同时成立。

### 1. 遍历性失效

最基础的错误是链根本走不全状态空间。

| 错误类型 | 例子 | 后果 |
|---|---|---|
| proposal 无法移动 | 步长写成 0 | 链永远停在初值 |
| proposal 覆盖不全 | 某些离散状态永远提不到 | 不可约失效 |
| 周期性更新设计错误 | 两个状态间固定来回跳 | 非周期失效 |

这类问题的严重性最高。因为一旦遍历性失效，跑再久都不会得到正确目标分布。

### 2. 接受率公式写错

MH 最典型的实现错误有三类：

| 错误 | 说明 | 结果 |
|---|---|---|
| 忘记 proposal 比值 | 非对称 proposal 却只写 $\pi(x')/\pi(x)$ | 目标分布错误 |
| 在概率域直接相乘 | 极小概率数值下溢 | 接受率失真 |
| 留在原地概率没处理对 | 转移核不规范 | 采样行为异常 |

尤其是在独立 proposal、Langevin proposal 或其他非对称 proposal 下，遗漏
$$
\frac{q(x|x')}{q(x'|x)}
$$
会直接破坏细致平衡。

### 3. 误把 burn-in 和 thinning 当成万能修复

如果链的平稳分布本身就错了，那么：

- 丢掉前 1000 步没有用；
- thinning 每隔 20 步取一个也没有用；
- 跑 10 倍更长仍然没有用。

这些操作只能处理“已经正确但效率一般”的链，不能修复“定义错误”的链。

### 4. 高维与强相关问题

在高维参数、强相关后验、层级模型中，朴素算法常见的失败模式是混合极慢。

例如二维高斯：
$$
\begin{pmatrix}
X_1\\X_2
\end{pmatrix}
\sim \mathcal{N}
\left(
\begin{pmatrix}
0\\0
\end{pmatrix},
\begin{pmatrix}
1 & \rho\\
\rho & 1
\end{pmatrix}
\right)
$$
当 $\rho\to 1$ 时，分布质量集中在一条细长斜带上。

这时：

- 单维 Gibbs 虽然每步接受率都是 1，但一次只沿坐标轴更新，可能移动很慢。
- 随机游走 MH 若 proposal 不匹配协方差结构，也会频繁在低密度区域提议失败。

### 5. 并行化不等于无代价

并行化常见于大规模图模型或参数很多的系统。典型选择如下：

| 方案 | 优点 | 风险 | 适用场景 |
|---|---|---|---|
| 同步 Gibbs | 理论分析清楚，状态一致 | 并行效率一般 | 质量优先、规模中等 |
| 异步 Gibbs | 吞吐量更高 | 读取旧状态，可能恶化混合 | 稀疏依赖图、延迟可控 |
| 单链长跑 | 实现简单 | 易误判收敛 | 快速原型 |
| 多链并行 | 可做链间诊断 | 成本更高 | 正式实验和报告 |

异步更新最重要的风险不是“代码难写”，而是“理论假设被弱化”。旧状态延迟、写入冲突和不同步更新，都会改变链的实际转移行为。吞吐量变高，并不自动意味着有效样本数增长更快。

### 6. 工程检查清单

实际使用 MCMC，至少要检查下面几项：

| 检查项 | 最低要求 |
|---|---|
| 转移核正确性 | 是否满足平稳分布条件，是否不可约、非周期 |
| 数值稳定性 | 是否用 log-density，是否避免溢出/下溢 |
| 诊断 | 是否查看接受率、轨迹、自相关、ESS、多链差异 |
| 参数尺度 | 各维是否标准化，proposal 是否匹配尺度 |
| 复现性 | 是否固定随机种子，是否保存配置和诊断结果 |

---

## 替代方案与适用边界

当随机游走 MH 或单维 Gibbs 的混合明显过慢时，继续“多跑几万步”通常不是最优策略。更合理的做法是换算法，或者先改模型参数化。

### 1. HMC：用梯度改善探索效率

HMC，Hamiltonian Monte Carlo，适合高维连续变量。它把采样问题改写成哈密顿系统上的动力学问题。定义位置变量 $q$、动量变量 $p$，以及哈密顿量：
$$
H(q,p)=U(q)+K(p)
$$
其中
$$
U(q)=-\log \pi(q), \qquad
K(p)=\frac12 p^\top M^{-1}p
$$
这里 $M$ 是质量矩阵，决定各方向上的尺度。

HMC 的直观目标是：不是靠“随机小步试探”，而是沿着高概率区域的几何结构走较长距离，再用 MH 做纠偏。常见的 leapfrog 更新为：
$$
p_{t+\frac12}=p_t-\frac{\epsilon}{2}\nabla U(q_t)
$$
$$
q_{t+1}=q_t+\epsilon M^{-1}p_{t+\frac12}
$$
$$
p_{t+1}=p_{t+\frac12}-\frac{\epsilon}{2}\nabla U(q_{t+1})
$$

它的优势与代价可以直接对比：

| 方法 | 优势 | 代价 |
|---|---|---|
| 随机游走 MH | 实现最简单 | 高维时扩散慢 |
| HMC | 自相关更低，探索更远 | 需要梯度，调参更复杂 |

### 2. Gibbs：条件分布简单时仍然很强

如果模型能写出条件分布，并且条件分布可直接采样，那么 Gibbs 往往比硬上随机游走 MH 更自然。例如：

- 共轭贝叶斯模型；
- 潜变量图模型；
- 某些高斯模型和层级模型。

但 Gibbs 不是“接受率 1 就一定高效”。强相关时，它可能每步都接受，却依然移动缓慢。

### 3. 重参数化往往比调步长更有效

很多 MCMC 困难，不是采样器本身太差，而是目标分布几何结构太差。典型手段包括：

| 手段 | 作用 |
|---|---|
| 标准化变量 | 缩小不同维度尺度差异 |
| 非中心化参数化 | 改善层级模型中的强相关 |
| 块更新 | 一起更新强耦合变量 |
| 预条件化 | 让 proposal 或动力学匹配协方差结构 |

在层级模型里，非中心化参数化经常比盲目拉长链更有效，因为它直接改变了后验几何形状。

### 4. 什么时候不该用 MCMC

MCMC 的适用边界也需要说清：

| 场景 | 是否适合 MCMC | 原因 |
|---|---|---|
| 分布可直接采样 | 通常不需要 | 直接采样没有相关性，也无诊断成本 |
| 低维、闭式积分可算 | 通常不需要 | 数值积分或解析解更直接 |
| 高维连续复杂后验 | 很适合，但常需 HMC/NUTS | 朴素 MH 往往太慢 |
| 大量离散潜变量 | Gibbs 或专门方法更合适 | HMC 不适用于离散变量 |
| 多峰且模态分离严重 | 朴素 MCMC 风险较高 | 容易困在单峰，需要 tempering/SMC 等方法 |

所以，MCMC 的强项不是“任何分布都万能”，而是：

- 目标分布复杂；
- 归一化常数难算；
- 可以接受相关样本；
- 只要求渐近正确；
- 愿意为诊断与调参付出额外成本。

---

## 参考资料

| 来源 | 核心贡献 | 建议阅读方式 | 适用章节 |
|---|---|---|---|
| CMU Metropolis-Hastings 讲义 | 细致平衡、平稳分布与 MH 接受率的推导较清晰 | 先看接受率公式怎么从细致平衡推出，再看离散与连续状态的写法 | 核心结论、核心机制与推导 |
| Helsinki Computational Statistics 讲义 | MCMC 的定义、马尔可夫链直觉、样本相关性解释较适合入门 | 先读“为什么需要 MCMC”，再看链相关样本与估计误差的关系 | 问题定义与边界、代码实现 |
| 异步 Gibbs 相关 PMC 论文 | 讨论异步更新、延迟与混合时间风险 | 重点看并行更新带来的理论条件变化，不必一开始就深读全部证明 | 工程权衡与常见坑 |
| MCMC 教材与课程笔记 | Gibbs 是特殊 MH、HMC 的几何直觉、高维采样经验 | 把它当成扩展阅读，重点补“为什么高维下随机游走会慢” | 核心机制与推导、替代方案 |

1. CMU: https://www.math.cmu.edu/~gautam/c/2024-387/notes/09-metropolis-hastings2.html
2. Helsinki: https://www.cs.helsinki.fi/u/ahonkela/teaching/compstats1/book/markov-chain-monte-carlo-basics.html
3. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC5360990/
4. Schneppat overview: https://schneppat.com/metropolis-hastings-algorithm_mha.html
