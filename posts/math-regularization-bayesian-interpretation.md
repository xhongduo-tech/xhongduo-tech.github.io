## 核心结论

正则化的贝叶斯解释是：给模型参数 $\beta$ 加一个先验分布，再用最大后验估计（MAP）求最可能的参数值。最大后验估计，白话说，就是“既看数据支持什么参数，也看我们事先相信什么参数更合理”。

一句话版结论：如果只看数据，模型会尽量把训练误差压低；如果再告诉模型“参数一般不要太大”或“很多参数应该接近 0”，这就是把经验写成先验，正则化就是这种写法的数学形式。

核心公式是：

$$
\beta_{\text{MAP}}
= \arg\max_\beta p(\beta \mid y)
= \arg\max_\beta p(y \mid \beta)p(\beta)
$$

其中 $p(y \mid \beta)$ 是似然，表示“给定参数后，数据出现的概率”；$p(\beta)$ 是先验，表示“在看数据之前，我们认为哪些参数更合理”。

| 方法 | 贝叶斯先验 | MAP 目标中的正则项 | 直观效果 |
|---|---|---|---|
| 无正则 MLE | 无明确先验，或等价于均匀先验 | 无 | 只拟合数据，容易过拟合 |
| Ridge / L2 | 高斯先验 | $\|\beta\|_2^2$ | 参数整体变小，平滑收缩 |
| Lasso / L1 | Laplace 先验 | $\|\beta\|_1$ | 部分参数变成 0，产生稀疏解 |
| Elastic Net | L1 与 L2 混合先验 | $\lambda_1\|\beta\|_1+\lambda_2\|\beta\|_2^2$ | 既稀疏，又稳定 |

所以，正则化不是“额外惩罚项”这么简单。它可以统一解释为：把对参数的偏好写成先验，最后求的不是纯最大似然估计（MLE），而是最大后验估计（MAP）。

---

## 问题定义与边界

本文讨论的是参数化模型。参数化模型，白话说，就是模型由一组有限参数控制，比如线性回归里的系数 $\beta$。

设：

$$
y \in \mathbb{R}^n,\quad X \in \mathbb{R}^{n \times p},\quad \beta \in \mathbb{R}^p
$$

其中 $n$ 是样本数，$p$ 是特征数。线性回归假设：

$$
y = X\beta + \epsilon
$$

若噪声服从高斯分布：

$$
y \mid \beta \sim \mathcal{N}(X\beta, \sigma^2 I)
$$

这表示：给定参数 $\beta$ 后，真实观测值 $y$ 会围绕预测值 $X\beta$ 上下波动，波动大小由噪声方差 $\sigma^2$ 控制。

玩具例子：只有一个特征时，模型是 $y=\beta x+\epsilon$。如果观测点很少，直接用数据拟合出的 $\beta$ 可能很大；如果先假设“$\beta$ 通常不要太夸张”，得到的参数会更稳。

真实工程例子：房价预测里，特征可能包括面积、楼层、地铁距离、学区、装修程度、历史成交均价。如果样本不多但特征很多，普通最小二乘会对噪声很敏感。加入正则化，相当于告诉模型：除非数据证据很强，否则不要让某个特征系数变得过大。

| 估计方式 | 优化目标 | 输出 | 是否表达不确定性 |
|---|---|---|---|
| MLE | 最大化 $p(y\mid\beta)$ | 一个参数点 | 否 |
| MAP | 最大化 $p(y\mid\beta)p(\beta)$ | 一个参数点 | 否 |
| Full Bayesian | 计算完整 $p(\beta\mid y)$ | 一个后验分布 | 是 |

本文边界如下：

| 边界 | 说明 |
|---|---|
| 噪声假设 | 主要以高斯噪声线性回归为例 |
| 模型类型 | 可推广到逻辑回归、广义线性模型 |
| 推断层次 | 只讨论 MAP 点估计，不讨论完整后验采样 |
| 特征处理 | 默认特征应先标准化，否则正则强度不可比 |
| 超参数 | $\lambda,\tau^2,b$ 通常通过验证集或交叉验证选择 |

MAP 只给“最可能的参数”，不告诉你参数有多不确定。如果任务需要置信区间、后验分布或预测不确定性，就不能只停在 MAP。

---

## 核心机制与推导

从 Bayes 公式开始：

$$
p(\beta \mid y)=\frac{p(y\mid\beta)p(\beta)}{p(y)}
$$

因为 $p(y)$ 与 $\beta$ 无关，所以最大化后验时可以忽略：

$$
\beta_{\text{MAP}}
= \arg\max_\beta p(y\mid\beta)p(\beta)
$$

对它取负对数，就得到：

$$
\beta_{\text{MAP}}
= \arg\min_\beta [-\log p(y\mid\beta)-\log p(\beta)]
$$

这就是“数据项 + 正则项”。数据项来自似然，正则项来自先验。

先看 L2。若参数先验是高斯分布：

$$
\beta \sim \mathcal{N}(0,\tau^2 I)
$$

高斯先验，白话说，就是相信参数大多围绕 0 波动，越大的参数越不常见，但不要求参数必须等于 0。

高斯噪声带来的负对数似然为：

$$
-\log p(y\mid\beta)=\frac{1}{2\sigma^2}\|y-X\beta\|_2^2 + C
$$

高斯先验带来的负对数先验为：

$$
-\log p(\beta)=\frac{1}{2\tau^2}\|\beta\|_2^2 + C'
$$

合起来：

$$
\beta_{\text{MAP}}
= \arg\min_\beta
\left[
\frac{1}{2\sigma^2}\|y-X\beta\|_2^2
+
\frac{1}{2\tau^2}\|\beta\|_2^2
\right]
$$

乘掉公共常数后，就是 Ridge：

$$
\arg\min_\beta \|y-X\beta\|_2^2+\lambda\|\beta\|_2^2
$$

其中常见写法下 $\lambda=\sigma^2/\tau^2$。$\tau^2$ 越小，先验越相信参数靠近 0，正则越强。

再看 L1。若每个参数独立服从 Laplace 分布：

$$
\beta_j \sim \text{Laplace}(0,b)
$$

Laplace 先验，白话说，就是比高斯分布更尖地集中在 0 附近，同时尾部更厚，因此更鼓励很多参数直接变成 0。

Laplace 先验的负对数形式是：

$$
-\log p(\beta)=\frac{1}{b}\sum_j |\beta_j| + C
$$

因此：

$$
\beta_{\text{MAP}}
= \arg\min_\beta
\left[
\frac{1}{2\sigma^2}\|y-X\beta\|_2^2
+
\frac{1}{b}\sum_j |\beta_j|
\right]
$$

这就是 Lasso 的 L1 正则目标。

Elastic Net 可以写成混合先验：

$$
p(\beta) \propto
\exp(-\lambda_1\|\beta\|_1-\lambda_2\|\beta\|_2^2)
$$

对应 MAP 目标：

$$
\arg\min_\beta
\|y-X\beta\|_2^2
+
\lambda_1\|\beta\|_1
+
\lambda_2\|\beta\|_2^2
$$

| 正则 | 先验形状 | 对参数的假设 | 效果 |
|---|---|---|---|
| L2 | 高斯，平滑 | 参数应小，但不必为 0 | 平滑收缩 |
| L1 | Laplace，在 0 处尖 | 很多参数应无效 | 稀疏 |
| Elastic Net | L1 + L2 | 参数要稀疏，也要稳定 | 稳定 + 稀疏 |

一个单参数玩具例子：设 $x=1,y=0.6,\sigma^2=1$。

高斯先验 $\beta\sim \mathcal{N}(0,1)$ 时，目标是：

$$
\frac{1}{2}(0.6-\beta)^2+\frac{1}{2}\beta^2
$$

求导得 $\beta_{\text{MAP}}=0.3$。L2 把 $0.6$ 收缩到 $0.3$，但没有变成 0。

若使用 L1：

$$
\frac{1}{2}(0.6-\beta)^2+|\beta|
$$

soft-threshold 后得到 $\beta_{\text{MAP}}=0$。soft-threshold，白话说，就是先把参数往 0 拉，拉过头时直接设为 0。

不同论文和库会把常数放在不同位置：

| 写法差异 | 常见位置 | 影响 |
|---|---|---|
| $\frac{1}{2\sigma^2}$ | 似然项前 | 改变 $\lambda$ 的数值解释 |
| $\frac{1}{n}$ | 经验风险平均 | 样本数变化时影响正则尺度 |
| $\frac{1}{2}$ | 平方损失前 | 方便求导 |
| `alpha` | sklearn 参数 | 不一定等于论文里的 $\lambda$ |
| `l1_ratio` | sklearn ElasticNet | 控制 L1 与 L2 比例 |

---

## 代码实现

实现时先统一目标函数。比如采用：

$$
\frac{1}{2n}\|y-X\beta\|_2^2+\alpha\|\beta\|_2^2
$$

或：

$$
\frac{1}{2n}\|y-X\beta\|_2^2+\alpha\rho\|\beta\|_1+\frac{\alpha(1-\rho)}{2}\|\beta\|_2^2
$$

其中 $\rho$ 对应 `l1_ratio`。`l1_ratio=1` 是 Lasso，`l1_ratio=0` 接近 Ridge。

伪代码流程：

```python
X = standardize(X)
ridge = fit_ridge(X, y, alpha=1.0)
lasso = fit_lasso(X, y, alpha=0.1)
enet = fit_elastic_net(X, y, alpha=0.1, l1_ratio=0.5)
print(coef_)
```

下面是一个可运行的最小例子，包含从零实现 Ridge 闭式解，并调用 sklearn 观察 Lasso 和 Elastic Net 的稀疏效果。

```python
import numpy as np
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)

n, p = 80, 6
X = rng.normal(size=(n, p))
true_beta = np.array([3.0, 0.0, 0.0, 1.5, 0.0, -2.0])
y = X @ true_beta + rng.normal(scale=0.5, size=n)

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

def fit_ridge_closed_form(X, y, alpha):
    n, p = X.shape
    return np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ y)

ridge_beta = fit_ridge_closed_form(Xs, y, alpha=5.0)

lasso = Lasso(alpha=0.1, fit_intercept=True, max_iter=10000)
lasso.fit(Xs, y)

enet = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True, max_iter=10000)
enet.fit(Xs, y)

ridge_pred = Xs @ ridge_beta
lasso_pred = lasso.predict(Xs)
enet_pred = enet.predict(Xs)

print("ridge:", np.round(ridge_beta, 3))
print("lasso:", np.round(lasso.coef_, 3))
print("enet :", np.round(enet.coef_, 3))
print("mse:", {
    "ridge": round(mean_squared_error(y, ridge_pred), 4),
    "lasso": round(mean_squared_error(y, lasso_pred), 4),
    "enet": round(mean_squared_error(y, enet_pred), 4),
})

assert ridge_beta.shape == (p,)
assert np.sum(np.isclose(lasso.coef_, 0.0)) >= 1
assert mean_squared_error(y, lasso_pred) < 1.0
```

| 方法 | 典型系数表现 | 训练误差 | 是否容易出现 0 系数 |
|---|---|---|---|
| Ridge | 所有系数整体变小 | 通常较低 | 不容易 |
| Lasso | 无关特征可被压成 0 | 可能略高 | 容易 |
| Elastic Net | 相关特征更稳定，部分为 0 | 通常折中 | 容易 |

| 数学目标参数 | sklearn 参数 | 说明 |
|---|---|---|
| L2 强度 $\lambda$ | `Ridge(alpha=...)` | sklearn 目标函数常数与推导写法不完全相同 |
| L1 强度 $\lambda$ | `Lasso(alpha=...)` | `alpha` 越大，越稀疏 |
| 总正则强度 | `ElasticNet(alpha=...)` | 同时放大 L1 与 L2 |
| L1 占比 | `ElasticNet(l1_ratio=...)` | 越接近 1 越像 Lasso |

代码中的关键步骤是标准化。标准化，白话说，就是把不同特征调整到相近尺度，避免某个特征只是因为单位不同而被正则项区别对待。

---

## 工程权衡与常见坑

第一个常见坑是忘记标准化。L1 和 L2 都直接作用在系数 $\beta$ 上。如果一个特征单位是“万元”，另一个特征单位是“元”，同样的业务含义会对应不同大小的系数。正则项看到的是系数大小，不知道单位含义。

| 场景 | 未标准化时 | 标准化后 |
|---|---|---|
| 特征 A 范围 $0\sim1$ | 系数可能较大，更容易被惩罚 | 惩罚尺度更公平 |
| 特征 B 范围 $0\sim10000$ | 系数可能很小，看似更“便宜” | 与其他特征可比 |
| Lasso 变量选择 | 可能选中尺度占优的特征 | 更接近真实信号 |
| Ridge 收缩 | 收缩强度不均衡 | 收缩解释更稳定 |

真实工程例子：广告点击率预估中，特征可能包括用户年龄、历史点击次数、广告价格、文本 embedding、类别 one-hot。如果不做合适缩放，Lasso 选出来的“重要特征”可能只是尺度更占便宜，而不是业务上真的更重要。

| 坑点 | 原因 | 处理方式 |
|---|---|---|
| `MAP ≠ posterior uncertainty` | MAP 只取后验峰值 | 需要不确定性时做完整贝叶斯推断 |
| L1 在 0 处不可导 | $|\beta|$ 在 0 没有普通导数 | 用坐标下降或近端算法 |
| 系数常数对不上 | 不同库吸收了 $1/n,1/2,\sigma^2$ | 先读目标函数定义 |
| Elastic Net 不一定更好 | 组效应依赖特征相关结构 | 用交叉验证比较 |
| 超参数没有先验推断 | $\lambda,\tau^2,b$ 常被手动选 | 可用层次贝叶斯建模 |

还有一个容易误解的点：Lasso 的稀疏性不是因为它“知道哪些特征没用”，而是因为 L1 几何形状在坐标轴上有尖点。优化等高线碰到这些尖点时，某些坐标就会变成 0。

Elastic Net 的价值主要出现在相关特征很多时。纯 Lasso 在一组高度相关特征中可能随机选一个，结果不稳定；L2 部分会让相关特征更倾向于一起保留，从而提高稳定性。但这不是定理式保证，仍然取决于数据相关性、噪声和正则强度。

---

## 替代方案与适用边界

正则化的贝叶斯解释适合“参数估计 + 先验偏好”的场景。它回答的是：如果我相信参数应该小、稀疏或稳定，那么优化目标应该怎么写。

| 方法 | 适用场景 | 不适合场景 |
|---|---|---|
| Ridge | 共线性强，需要稳定估计 | 强变量筛选需求 |
| Lasso | 需要稀疏解，需要变量筛选 | 高度相关特征中选择不稳定 |
| Elastic Net | 既要筛选，又希望相关特征更稳定 | 特征少且无明显共线 |
| Bayesian Hierarchical Model | 需要不确定性和超参数推断 | 只需要快速点估计 |

真实工程选择场景：广告点击率预估里，特征数量很大，很多特征强相关。第一步常用 Ridge 稳住估计；如果还要挑出少数关键变量，可以尝试 Elastic Net；如果业务要求解释“哪些变量被选中”，再检查 Lasso 或 Elastic Net 的稳定性，而不是只看一次训练结果。

不同模型中的类比如下：

| 模型 | 正则化贝叶斯解释 |
|---|---|
| 线性回归 | 高斯似然 + 参数先验 |
| 逻辑回归 | Bernoulli 似然 + 参数先验 |
| 广义线性模型 | 指数族似然 + 参数先验 |
| 深度模型 | 权重衰减可类比高斯先验，但完整解释更复杂 |

何时不用 MAP：

| 需求 | 为什么 MAP 不够 | 更合适方法 |
|---|---|---|
| 需要参数置信区间 | MAP 只有一个点 | 后验采样或近似推断 |
| 需要预测不确定性 | 点估计不表达分布宽度 | 贝叶斯预测分布 |
| 数据极少且先验很重要 | 单点解可能误导 | 层次贝叶斯 |
| 超参数本身需推断 | 交叉验证只选点值 | 给超参数加先验 |

结论是：Ridge、Lasso、Elastic Net 可以先作为工程上稳定、便宜、可解释的 MAP 工具。若问题核心是“不确定性”，就应该转向完整贝叶斯方法，而不是把正则化解释过度延伸。

---

## 参考资料

| 本文结论 | 对应参考 |
|---|---|
| Ridge 是 L2 正则，缓解共线性 | Hoerl & Kennard, 1970 |
| Lasso 使用 L1 正则产生稀疏解 | Tibshirani, 1996 |
| Elastic Net 结合 L1 与 L2 | Zou & Hastie, 2005 |
| Laplace 先验与稀疏化 | Williams, 1995 |
| MLE / MAP 基础 | Stanford CS109 MLE/MAP 课程材料 |

建议阅读顺序：先看 Stanford CS109 理解 MLE 与 MAP；再看 Ridge、Lasso、Elastic Net 三篇经典论文；最后回到贝叶斯解释，理解正则项如何来自先验。

- Ridge：Hoerl, A. E., & Kennard, R. W. 1970. *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. https://doi.org/10.1080/00401706.1970.10488634
- Lasso：Tibshirani, R. 1996. *Regression Shrinkage and Selection Via the Lasso*. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
- Elastic Net：Zou, H., & Hastie, T. 2005. *Regularization and Variable Selection Via the Elastic Net*. https://doi.org/10.1111/j.1467-9868.2005.00503.x
- Bayesian / Laplace prior：Williams, P. M. 1995. *Bayesian Regularization and Pruning Using a Laplace Prior*. https://doi.org/10.1162/neco.1995.7.1.117
- Bayesian Interpretation of Regularization：https://link.springer.com/chapter/10.1007/978-3-030-95860-2_4
- Stanford CS109：*Maximum Likelihood Estimation and Maximum A Posteriori*. https://web.stanford.edu/class/archive/cs/cs109/cs109.1248/lectures/20-MaximumLikelihoodEstimation/
