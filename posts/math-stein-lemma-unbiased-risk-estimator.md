## 核心结论

Stein 引理是高斯分布下的分部积分恒等式。它把“高斯噪声项和函数值之间的相关性”转换成“函数导数的期望”。

SURE，Stein 无偏风险估计，是 Stein 引理的直接应用。它把不可见的均方风险

$$
R(\delta)=\mathbb E\|\delta(Y)-\theta\|_2^2
$$

改写成一个只依赖观测值 $Y$、噪声方差 $\sigma^2$ 和估计器导数的量：

$$
\operatorname{SURE}(\delta)
=
\|Y-\delta(Y)\|_2^2
+
2\sigma^2\operatorname{div}\delta(Y)
-
d\sigma^2
$$

并且满足

$$
\mathbb E[\operatorname{SURE}(\delta)]
=
\mathbb E\|\delta(Y)-\theta\|_2^2.
$$

这里的估计器 $\delta(Y)$ 指“根据带噪数据 $Y$ 输出一个估计值的函数”；散度 $\operatorname{div}\delta(Y)$ 指估计器输出对输入各维度的总敏感度。白话说，如果你能写出估计量对数据的导数，就能在没有真值标签时估计风险。

| 对象 | 是否需要真实 $\theta$ | 含义 |
|---|---:|---|
| 真风险 $R(\delta)$ | 需要 | 估计器平均离真值多远 |
| 单次损失 $\|\delta(Y)-\theta\|^2$ | 需要 | 当前样本上的实际误差 |
| SURE | 不需要 | 真风险的无偏估计 |
| 验证集误差 | 需要干净标签 | 工程上直接评估泛化效果 |

真实工程例子是图像去噪。干净图像 $\theta$ 不可见，只能看到带噪图像 $Y$。如果去噪器是软阈值、岭回归、谱阈值这类可写出导数或可近似散度的收缩算子，就可以对阈值 $\lambda$ 做网格搜索，选择让 $\operatorname{SURE}(\delta_\lambda)$ 最小的参数，而不需要干净标签图。

---

## 问题定义与边界

研究对象是高斯均值模型：

$$
Y\in\mathbb R^d,\quad Y\sim N(\theta,\sigma^2 I_d).
$$

这里 $Y$ 是观测到的带噪向量，$\theta$ 是未知真实均值，$\sigma^2$ 是每个维度的噪声方差，$I_d$ 是 $d$ 维单位矩阵。目标是构造估计量 $\delta(Y)$，让它尽量接近 $\theta$。

风险定义为

$$
R(\delta)=\mathbb E\|\delta(Y)-\theta\|_2^2.
$$

均方风险是“平均平方误差”。它不是某一次观测的误差，而是在同一个真实 $\theta$ 下反复采样 $Y$ 后的平均表现。

新手版本的问题是：手里只有带噪数据 $Y$，没有干净答案 $\theta$，但仍想判断“这个去噪器好不好”。SURE 提供了一个只看 $Y$ 的风险打分器。这个打分器不是每次都等于真实损失，但它的期望等于真实风险。

SURE 不能脱离条件使用。它的经典形式依赖以下边界。

| 条件 | 状态 | 说明 |
|---|---|---|
| $Y\sim N(\theta,\sigma^2 I_d)$ | 可直接使用 | 经典 SURE 的标准设定 |
| $\sigma^2$ 已知 | 可直接使用 | 方差进入校正项，不能随便省略 |
| $\delta$ 可微 | 可直接使用 | 散度可以由普通导数计算 |
| $\delta$ 几乎处处可微 | 需要改造 | 软阈值等可用广义导数处理 |
| 非高斯噪声 | 不能直接用 | 需要广义 Stein 恒等式或其他准则 |
| 相关噪声协方差已知 | 需要改造 | 要把 $\sigma^2 I_d$ 换成协方差矩阵形式 |
| 方差未知 | 需要改造 | 先估计噪声水平，再分析误差传递 |

这里“几乎处处可微”指除了少数点以外都可微。例如软阈值函数在阈值点不可微，但这些点在连续分布下概率通常为 0，因此仍可在适当条件下使用。

---

## 核心机制与推导

先看一维分量形式。对足够光滑且满足可积条件的函数 $f$，Stein 引理给出：

$$
\mathbb E[(Y_i-\theta_i)f(Y)]
=
\sigma^2\mathbb E\left[\frac{\partial f(Y)}{\partial y_i}\right].
$$

协方差等价形式是：

$$
\operatorname{Cov}(Y_i,f(Y))
=
\sigma^2\mathbb E[\partial_i f(Y)].
$$

协方差是“两个随机量一起变化的程度”。这个公式说明，在高斯模型里，$Y_i$ 和函数 $f(Y)$ 的相关性可以用 $f$ 对 $y_i$ 的导数期望来表示。机制来自高斯密度的导数：

$$
\frac{\partial}{\partial y_i}p(y)
=
-\frac{y_i-\theta_i}{\sigma^2}p(y).
$$

把 $(y_i-\theta_i)p(y)$ 替换成 $-\sigma^2\partial_i p(y)$，再做分部积分，就得到 Stein 引理。

推 SURE 时，把风险展开：

$$
\begin{aligned}
\|\delta(Y)-\theta\|^2
&=
\|\delta(Y)-Y+Y-\theta\|^2 \\
&=
\|\delta(Y)-Y\|^2
+
2(\delta(Y)-Y)^\top(Y-\theta)
+
\|Y-\theta\|^2.
\end{aligned}
$$

对两边取期望。最后一项满足

$$
\mathbb E\|Y-\theta\|^2=d\sigma^2.
$$

中间交叉项用 Stein 引理处理。令 $g(Y)=\delta(Y)-Y$，则

$$
\mathbb E[g(Y)^\top(Y-\theta)]
=
\sigma^2\mathbb E[\operatorname{div}g(Y)].
$$

又因为

$$
\operatorname{div}g(Y)=\operatorname{div}\delta(Y)-d,
$$

所以

$$
R(\delta)
=
\mathbb E\left[
\|Y-\delta(Y)\|^2
+
2\sigma^2\operatorname{div}\delta(Y)
-
d\sigma^2
\right].
$$

这就是 SURE。

推导流程可以写成：

| 步骤 | 内容 |
|---|---|
| 1 | 假设 $Y$ 服从高斯分布 |
| 2 | 利用高斯密度导数做分部积分 |
| 3 | 得到 Stein 引理，把噪声相关项变成导数期望 |
| 4 | 展开均方风险 |
| 5 | 用 Stein 引理替换交叉项 |
| 6 | 得到只依赖 $Y$ 的 SURE |

玩具例子：取 $d=3,\sigma^2=1,Y=(2,1,0)$，估计器为线性收缩

$$
\delta(Y)=0.6Y.
$$

散度为

$$
\operatorname{div}\delta=3\times 0.6=1.8.
$$

残差平方为

$$
\|Y-\delta(Y)\|^2=\|(0.8,0.4,0)\|^2=0.8.
$$

所以

$$
\operatorname{SURE}=0.8+2\times1\times1.8-3=1.4.
$$

这个 $1.4$ 不是当前样本上的真实损失，因为真实 $\theta$ 没有出现在计算里。它的含义是：在重复采样意义下，它对该估计器的均方风险无偏。

James-Stein 估计展示了收缩估计的重要现象：

$$
\delta_{JS}(Y)
=
\left(1-\frac{(d-2)\sigma^2}{\|Y\|_2^2}\right)Y.
$$

样本均值估计器是 $\delta(Y)=Y$。James-Stein 估计器把 $Y$ 向 0 收缩。经典结论是，当 $d\ge 3$ 时，在多元正态均值估计的平方损失下，James-Stein 估计可以一致优于直接使用 $Y$。这不是因为 0 一定是真值，而是因为在高维里，适度降低方差带来的收益可以超过引入偏差的代价。

---

## 代码实现

实现 SURE 时要分清目标：不是求真实风险，而是给估计器计算一个可观测的风险估计，然后用它做参数选择。

下面是线性收缩 $\delta(Y)=aY$ 的最小实现。此时

$$
\operatorname{div}\delta(Y)=da.
$$

```python
import numpy as np

def sure(y, delta_fn, div_delta_fn, sigma2):
    y = np.asarray(y, dtype=float)
    delta_y = delta_fn(y)
    d = y.size
    return np.sum((y - delta_y) ** 2) + 2 * sigma2 * div_delta_fn(y) - d * sigma2

def linear_delta(a):
    return lambda y: a * y

def linear_divergence(a):
    return lambda y: y.size * a

y = np.array([2.0, 1.0, 0.0])
sigma2 = 1.0

value = sure(
    y,
    delta_fn=linear_delta(0.6),
    div_delta_fn=linear_divergence(0.6),
    sigma2=sigma2,
)

assert abs(value - 1.4) < 1e-12

candidates = [0.0, 0.3, 0.6, 0.9, 1.0]
scores = [
    (a, sure(y, linear_delta(a), linear_divergence(a), sigma2))
    for a in candidates
]
best_a, best_score = min(scores, key=lambda x: x[1])

assert best_a == 0.6
assert abs(best_score - 1.4) < 1e-12

print(scores)
print(best_a, best_score)
```

对应的参数搜索结果是：

| 参数 $a$ | SURE | 是否选中 |
|---:|---:|---|
| 0.0 | 2.0 | 否 |
| 0.3 | 1.85 | 否 |
| 0.6 | 1.4 | 是 |
| 0.9 | 1.85 | 否 |
| 1.0 | 3.0 | 否 |

真实工程里，估计器通常不只是 $aY$。常见形式包括：

| 估计器 | 形式 | 散度处理 |
|---|---|---|
| 线性收缩 | $\delta(Y)=aY$ | 闭式 $da$ |
| 岭回归 | $\delta(Y)=AY$ | 闭式 $\operatorname{tr}(A)$ |
| 软阈值 | $\operatorname{sign}(Y_i)(|Y_i|-\lambda)_+$ | 用几乎处处导数 |
| 谱阈值 | 对奇异值收缩 | 需要矩阵微分或专门公式 |
| 黑盒去噪器 | 神经网络或复杂算法 | 常用 Monte Carlo SURE |

当散度难以解析计算时，可以用 Monte Carlo SURE。核心思想是用随机扰动估计散度：

```python
import numpy as np

def mc_divergence(delta_fn, y, eps=1e-4, num_samples=64, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    estimates = []

    for _ in range(num_samples):
        z = rng.normal(size=y.shape)
        diff = delta_fn(y + eps * z) - delta_fn(y)
        estimates.append(np.dot(z, diff) / eps)

    return float(np.mean(estimates))

def soft_threshold(lam):
    def fn(y):
        return np.sign(y) * np.maximum(np.abs(y) - lam, 0.0)
    return fn

y = np.array([2.0, 1.0, 0.0])
delta = soft_threshold(0.5)
div_est = mc_divergence(delta, y, eps=1e-5, num_samples=200, seed=1)

score = sure(y, delta, lambda yy: div_est, sigma2=1.0)

assert np.isfinite(score)
assert div_est >= 0.0
```

Monte Carlo SURE 的优点是能处理复杂估计器，缺点是引入随机误差，并且扰动尺度 `eps` 需要调试。`eps` 太大时偏差明显，太小时会受数值精度影响。

---

## 工程权衡与常见坑

SURE 的有效性高度依赖噪声模型。最常见的错误是把非高斯噪声当成高斯噪声，把未知 $\sigma^2$ 当成准确常数，然后用 SURE 选参数。结果是训练时 SURE 最小，但验证集上的去噪质量反而变差。

典型真实工程例子：相机图像噪声常常包含泊松噪声、读出噪声、压缩伪影和传感器非线性。若直接假设每个像素都是独立同方差高斯噪声，SURE 会低估某些区域的风险，高估另一些区域的风险。高亮区域和暗部区域的噪声强度不同，统一 $\sigma^2$ 会导致阈值选择系统性偏移。

| 问题 | 后果 | 规避方法 |
|---|---|---|
| 非高斯噪声直接套 SURE | 风险估计有偏 | 做方差稳定化或换用适配分布的准则 |
| $\sigma^2$ 估错 | 最优参数整体偏移 | 先做噪声校准，报告敏感性 |
| 非光滑算子硬套普通导数 | 散度计算错误 | 用广义导数、平滑近似或 MC-SURE |
| 把单次 SURE 当真实损失 | 对单个样本过度解释 | 只把它当风险估计和调参指标 |
| 维度 $d$ 很小 | James-Stein 优势不稳定 | 检查 $d\ge3$ 和具体风险条件 |
| 用同一个样本反复选复杂参数 | 可能过拟合 SURE | 限制搜索空间，必要时保留验证集 |

适用条件检查表：

| 检查项 | 通过标准 |
|---|---|
| 高斯性 | 噪声近似正态，或已做近似变换 |
| 方差已知性 | $\sigma^2$ 已测量、校准或可靠估计 |
| 可微性 | 估计器可微、几乎处处可微或可估计散度 |
| 维度要求 | 收缩估计优势通常在中高维更明显 |
| 数据泄漏 | 调参过程没有使用不可用的真值标签 |

调参前可以按这个顺序检查：先确认噪声模型，再估计 $\sigma^2$，然后确认估计器的散度能否计算，最后才跑网格搜索。若前两步不可靠，SURE 的数值越精细，越可能只是精细地优化了错误目标。

---

## 替代方案与适用边界

SURE 适合无标签、噪声近似高斯、估计器可微或可近似求散度的场景。它不是通用模型选择工具。当高斯假设不成立、导数难以计算、噪声结构复杂，或者任务最终指标不是均方误差时，需要考虑其他方法。

| 方法 | 需要标签 | 主要假设 | 适用场景 | 局限 |
|---|---:|---|---|---|
| SURE | 否 | 高斯噪声、方差已知、可求散度 | 去噪、收缩估计、正则参数选择 | 假设错时会有偏 |
| 交叉验证 | 通常需要 | 样本可拆分，验证误差代表目标 | 监督学习、预测任务 | 小样本时方差大 |
| Bootstrap | 否或部分需要 | 重采样能近似数据生成过程 | 不确定性评估、稳健性分析 | 依赖重采样设计 |
| 贝叶斯后验风险 | 需要先验 | 先验和似然建模合理 | 小样本、层级模型 | 结果受先验影响 |
| 专用稳健准则 | 视任务而定 | 针对特定噪声或损失设计 | 泊松噪声、重尾噪声、离群点 | 推导和实现更复杂 |

图像去噪中，SURE 适合软阈值、小波收缩、岭类方法、谱收缩方法。如果去噪器包含强离散操作，例如硬规则分块、非连续排序、复杂后处理，散度可能很难稳定估计。如果噪声是泊松噪声、椒盐噪声或强相关噪声，直接使用经典 SURE 就不合适。

文字版决策树如下：

| 条件 | 建议 |
|---|---|
| 噪声近似高斯，$\sigma^2$ 可靠，估计器可求散度 | 优先用 SURE |
| 噪声近似高斯，但估计器是黑盒 | 尝试 Monte Carlo SURE |
| 有干净验证集，目标指标明确 | 优先用验证集或交叉验证 |
| 噪声明显非高斯 | 寻找对应分布的 Stein 公式或稳健准则 |
| 模型中先验知识很强 | 考虑贝叶斯后验风险 |
| 样本量小且评估不稳定 | 用 bootstrap 做不确定性分析 |

SURE 的核心价值是把“没有真值时无法调参”的问题，转化为“在明确统计假设下估计风险”的问题。它的边界也来自这里：假设清楚时很有用，假设含糊时不能替代验证。

---

## 参考资料

1. [Stein, 1981, Estimation of the Mean of a Multivariate Normal Distribution](https://www.stat.yale.edu/~hz68/619/Stein-1981.pdf)：建立 Stein 引理和无偏风险估计的理论背景，是理解 SURE 的核心来源。
2. [James & Stein, 1961, Estimation with Quadratic Loss](https://digicoll.lib.berkeley.edu/record/112898)：说明多元正态均值估计中收缩估计可以优于样本均值，是 James-Stein 现象的原始理论来源。
3. [Oliveira, Lei, Tibshirani, 2024, Unbiased Risk Estimation in the Normal Means Problem via Coupled Bootstrap Techniques](https://www.stat.berkeley.edu/~ryantibs/papers/cb.pdf)：讨论正态均值问题中的无偏风险估计扩展，适合继续理解现代风险估计方法。
4. [Nobel, Candès, Boyd, 2023, Tractable Evaluation of Stein’s Unbiased Risk Estimator with Convex Regularizers](https://stanford.edu/~boyd/papers/sure_tractable_eval.html)：讨论带凸正则项时如何可计算地评估 SURE，连接理论公式和工程优化。
5. [Stanford EE364B Notes, Stein’s Unbiased Risk Estimate](https://web.stanford.edu/class/ee364b/lectures/sure_slides.pdf)：用优化和信号处理视角介绍 SURE，适合作为工程实现补充材料。

建议阅读顺序：

| 顺序 | 资料 | 目的 |
|---:|---|---|
| 1 | Stein 1981 | 先建立 Stein 引理和 SURE 概念 |
| 2 | James & Stein 1961 | 理解为什么收缩估计可能更优 |
| 3 | Stanford SURE notes | 用计算视角复习公式 |
| 4 | Nobel, Candès, Boyd 2023 | 看凸正则场景下的可计算实现 |
| 5 | Oliveira, Lei, Tibshirani 2024 | 了解无偏风险估计的现代扩展 |
