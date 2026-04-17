## 核心结论

大数定律讨论的是一件非常基础但非常重要的事：当样本数量越来越大时，样本均值会不会逼近真实期望。

弱大数定律说的是“高概率会接近”，形式是：对任意 $\varepsilon>0$，
$$
\mathbb{P}(|\bar X_n-\mu|>\varepsilon)\to 0,\quad n\to\infty
$$
这里的“依概率收敛”可以直接理解成：样本越多，偏得很离谱的概率越小。

强大数定律说的是“几乎一定会接近”，形式是：
$$
\mathbb{P}\left(\lim_{n\to\infty}\bar X_n=\mu\right)=1
$$
这里的“几乎必然收敛”可以直接理解成：除了概率为 0 的极少数路径，长期看样本均值都会收敛到真实期望。它比弱大数定律更强。

一个直接结论是：只要方差有限，样本均值的标准误差满足
$$
\mathrm{SE}(\bar X_n)=\frac{\sigma}{\sqrt n}
$$
这意味着误差不是按 $1/n$，而是按 $1/\sqrt n$ 缩小。样本数扩大 4 倍，误差才减半。

玩具例子：掷公平硬币，记正面为 1、反面为 0，则真实期望 $\mu=0.5$。掷得越多，正面比例越接近 0.5。  
真实工程例子：Monte Carlo 积分、MCMC 采样、mini-batch SGD 都依赖“平均值最终逼近真实期望”这个事实工作。

| 结论 | 数学形式 | 直白解释 | 强度 |
|---|---|---|---|
| 弱大数定律 | $\mathbb{P}(|\bar X_n-\mu|>\varepsilon)\to 0$ | 偏差大的概率趋近 0 | 较弱 |
| 强大数定律 | $\mathbb{P}(\lim \bar X_n=\mu)=1$ | 几乎每条样本路径都收敛 | 更强 |

---

## 问题定义与边界

先定义对象。设 $X_1,\dots,X_n$ 是随机变量，样本均值为
$$
\bar X_n=\frac{1}{n}\sum_{i=1}^n X_i
$$
目标参数是总体期望
$$
\mu=\mathbb{E}[X]
$$

问题是：在什么条件下，$\bar X_n$ 会逼近 $\mu$？

最常见的边界条件有三类。

| 条件 | 含义 | 作用 |
|---|---|---|
| 独立同分布 IID | 每次样本彼此独立，且来自同一分布 | 最标准的大数定律场景 |
| 有限期望 | $\mathbb{E}[|X|]<\infty$ | 强大数定律常见基本条件 |
| 有限方差 | $\mathrm{Var}(X)<\infty$ | 方便用切比雪夫不等式证明弱大数定律 |

这里的 IID 可以理解成“每次抽样都像重新从同一个盒子里独立取一个球”。  
如果样本不是 IID，例如来自 MCMC 链，那么需要更强的“遍历性”条件。遍历性可以理解成：虽然样本彼此相关，但链跑得足够久，长期平均仍然能代表目标分布。

因此，大数定律不是“任何平均值都会收敛”的口号，它依赖采样机制本身是否正确。如果样本来源错了，收敛只会把你稳定地带到错误答案。

---

## 核心机制与推导

弱大数定律最容易从方差出发推导。

假设 $X_i$ IID，且 $\mathbb{E}[X_i]=\mu,\ \mathrm{Var}(X_i)=\sigma^2<\infty$。由于均值的方差满足
$$
\mathrm{Var}(\bar X_n)=\mathrm{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right)=\frac{\sigma^2}{n}
$$

再用切比雪夫不等式。切比雪夫不等式的白话是：如果方差不大，偏离均值很多的概率就不会太大。
$$
\mathbb{P}(|Y-\mathbb{E}Y|\ge \varepsilon)\le \frac{\mathrm{Var}(Y)}{\varepsilon^2}
$$

令 $Y=\bar X_n$，得到
$$
\mathbb{P}(|\bar X_n-\mu|\ge \varepsilon)\le \frac{\sigma^2}{n\varepsilon^2}
$$

右边随着 $n\to\infty$ 收敛到 0，所以
$$
\mathbb{P}(|\bar X_n-\mu|\ge \varepsilon)\to 0
$$
这就是弱大数定律。

这个推导非常关键，因为它同时解释了 Monte Carlo 的误差公式。样本均值的标准差是
$$
\sqrt{\mathrm{Var}(\bar X_n)}=\frac{\sigma}{\sqrt n}
$$
所以标准误差只按平方根缩小。很多初学者误以为“样本数翻倍，误差就减半”，这是错的；样本数要扩大到原来的 4 倍，误差才减半。

强大数定律更强，它要求的不只是“偏差大的概率越来越小”，而是“沿着几乎每一条样本路径，均值最后真的收敛到 $\mu$”。严格证明通常需要 Borel-Cantelli 引理、Kolmogorov 方法等工具。对初学者更重要的理解是：

1. 弱大数定律管的是“每个固定 $n$ 时，出错概率多大”。
2. 强大数定律管的是“整个无限序列最终会不会收敛”。

所以二者不是同一句话的两种写法，而是收敛层级不同。

Monte Carlo 积分是大数定律最典型的工程化版本。若要估计
$$
I=\mathbb{E}[f(X)]
$$
可以采样 $X_1,\dots,X_n$，然后用
$$
\hat I_n=\frac1n\sum_{i=1}^n f(X_i)
$$
估计。只要 $f(X)$ 有有限期望，$\hat I_n$ 就会趋近 $I$。标准误差仍然是
$$
\mathrm{SE}(\hat I_n)=\frac{\sigma_f}{\sqrt n}
$$
这里看不到维度 $d$ 直接出现在分母里，所以 Monte Carlo 的收敛速率在形式上与维度无关。这就是常说的“它没有传统网格积分那样直接遭遇维数诅咒”。

---

## 代码实现

下面给两个最小实现。第一个展示 Monte Carlo 均值收敛，第二个展示 mini-batch 梯度方差随 batch size 按 $1/B$ 缩小。

```python
import random
import math

def monte_carlo_uniform_mean(n, seed=0):
    random.seed(seed)
    xs = [random.random() for _ in range(n)]  # U[0,1]
    mean = sum(xs) / n
    true_mu = 0.5
    sigma = math.sqrt(1.0 / 12.0)
    se = sigma / math.sqrt(n)
    return mean, true_mu, se

mean_256, true_mu, se_256 = monte_carlo_uniform_mean(256, seed=42)
print("n=256 mean=", mean_256, "true=", true_mu, "SE≈", se_256)

# U[0,1] 的理论标准误差约为 0.289/sqrt(256) ≈ 0.018
assert abs(se_256 - math.sqrt(1/12)/16) < 1e-12
assert abs(true_mu - 0.5) < 1e-12

# 样本数增加时，理论标准误差应下降
_, _, se_1024 = monte_carlo_uniform_mean(1024, seed=42)
assert se_1024 < se_256
assert abs(se_256 / se_1024 - 2.0) < 1e-12
```

玩具例子可以直接读这段代码：从 $[0,1]$ 均匀分布采样，真实期望是 0.5，$n=256$ 时标准误差约为 0.018，也就是大约 2% 量级。

再看一个真实工程例子。设总体损失是
$$
L(\theta)=\frac1N\sum_{i=1}^N \ell_i(\theta)
$$
mini-batch 梯度
$$
g_B=\frac1{|B|}\sum_{i\in B}\nabla \ell_i(\theta)
$$
是全量梯度的无偏估计。无偏估计可以理解成：单次有噪声，但长期平均不偏。

```python
import random
import statistics

def sample_batch_mean(values, batch_size):
    batch = random.sample(values, batch_size)
    return sum(batch) / batch_size

random.seed(123)

# 假设每个样本的单样本梯度已在当前 theta 上算好
per_sample_grads = [random.gauss(2.0, 3.0) for _ in range(10000)]
full_grad = sum(per_sample_grads) / len(per_sample_grads)

def estimate_batch_variance(batch_size, trials=2000):
    ests = [sample_batch_mean(per_sample_grads, batch_size) for _ in range(trials)]
    mean_est = sum(ests) / len(ests)
    var_est = statistics.pvariance(ests)
    return mean_est, var_est

mean_32, var_32 = estimate_batch_variance(32)
mean_128, var_128 = estimate_batch_variance(128)

print("full_grad =", full_grad)
print("batch=32  mean, var =", mean_32, var_32)
print("batch=128 mean, var =", mean_128, var_128)

assert abs(mean_32 - full_grad) < 0.15
assert abs(mean_128 - full_grad) < 0.10
assert var_128 < var_32
```

这段代码验证两件事：

1. batch 梯度均值接近全量梯度，说明它是无偏估计。
2. batch size 从 32 提到 128，方差明显下降，接近缩小为原来的 $1/4$。

---

## 工程权衡与常见坑

大数定律保证的是“长期平均正确”，不保证“短期一定稳定”。工程上最常见的坑集中在采样相关性和优化超参数。

| 场景 | 典型错误 | 后果 | 正确做法 |
|---|---|---|---|
| Monte Carlo | 把相关样本当独立样本 | 标准误差被低估 | 估计有效样本数 ESS，做块平均 |
| MCMC | burn-in 不足 | 均值偏向初始状态 | 丢弃前期样本，检查混合 |
| 大 batch 训练 | 只增大 batch，不调学习率 | 梯度噪声变小但步子也偏小，训练变慢 | 常用线性放大学习率 |
| 有重尾分布 | 误以为均值一定稳定 | 收敛很慢甚至方差不存在 | 先检查分布尾部性质 |

在 SGD 中，若 batch size 变为原来的 $k$ 倍，梯度方差大约缩小为原来的 $1/k$。如果学习率不变，每步更新会变得更“保守”。这就是为什么很多分布式训练使用 Linear Scaling Rule：batch 放大多少倍，学习率也近似放大多少倍，以维持接近的训练动态。

但它不是无条件真理。模型结构、优化器、归一化层、warmup 策略都会影响可用范围。正确姿势不是背规则，而是记录训练曲线、梯度范数、loss 抖动幅度，检查“更新统计量”是否仍在可接受区间。

---

## 替代方案与适用边界

当独立采样拿不到时，可以用 MCMC。MCMC 不是直接给你独立样本，而是给你一条相关的马尔可夫链。马尔可夫链可以理解成“下一步只依赖当前状态的随机过程”。只要链满足遍历性，时间平均仍然会收敛到目标分布下的期望，这本质上仍然是大数定律的扩展版本。

| 方法 | 适用场景 | 优点 | 限制 |
|---|---|---|---|
| IID Monte Carlo | 能直接独立采样 | 分析简单，SE 明确 | 高质量采样器可能难构造 |
| MCMC | 只能知道未归一化密度 | 可处理复杂后验 | 样本相关，需要混合诊断 |
| Bootstrap | 有固定数据集，关心统计量不确定性 | 实现简单 | 依赖重采样假设 |
| Quasi-Monte Carlo | 被积函数较光滑 | 常比随机采样更快 | 理论与实现更复杂 |

真实工程里，MCMC 的关键不是“链够长就行”，而是“有效样本数有多少”。如果自相关很强，表面上有 10 万个样本，可能只有几千个有效样本。此时直接套用 $\sigma/\sqrt n$ 会严重乐观，应该用块平均、谱方差估计或 ESS 修正。

因此，适用边界可以概括为：

1. IID 且有限方差：标准结论最干净。
2. 相关样本但遍历：仍可用，但要改误差估计。
3. 重尾到连期望都不存在：大数定律本身就可能失效，先检查模型假设。

---

## 参考资料

| 来源 | 涉及内容 | 推荐阅读部分 |
|---|---|---|
| Wikipedia: Law of large numbers | WLLN、SLLN 定义与区别 | 先看弱/强版本的数学表达 |
| Bookdown Monte Carlo 章节 | Monte Carlo 平均值收敛直观 | 看样本均值估计积分 |
| DCF Modeling Monte Carlo 方法 | 标准误差 $1/\sqrt n$ 解释 | 看误差与样本数关系 |
| CrossValidated: SGD unbiased estimator | mini-batch 梯度无偏性 | 看为什么 batch 梯度是期望正确的 |
| AI Wiki: Batch Size and Learning Rate | batch 与学习率联动 | 看 Linear Scaling Rule 的工程语境 |

可进一步阅读的重点：

1. 先读弱大数定律，再读强大数定律，不要反过来。
2. 在 Monte Carlo 章节重点看“为什么误差是 $\sigma/\sqrt n$”。
3. 在 SGD 资料里重点看“无偏估计”和“方差随 batch 缩放”的关系。
4. 在 MCMC 资料里重点看“自相关、混合时间、有效样本数”。
