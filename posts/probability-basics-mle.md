## 随机变量与概率分布

随机变量 X 是样本空间 Ω 到实数 ℝ 的可测映射：X: Ω → ℝ。离散型取可数值，连续型取 ℝ 的子区间。概率分布描述随机变量取值的规律：离散型用概率质量函数 P(X = x)，连续型用概率密度函数 f_X(x)，满足 ∫ f_X(x)dx = 1。

**常见分布**

| 分布 | 支撑集 | 参数 | PMF/PDF |
|------|--------|------|---------|
| 伯努利 | {0, 1} | p ∈ [0,1] | P(X=x) = p^x (1-p)^{1-x} |
| 多项 | {0,...,k} | p₁,...,p_k, n | P(x) = n!/(x₁!...x_k!) ∏ p_i^{x_i} |
| 高斯 | ℝ | μ, σ² | f(x) = (2πσ²)^{-1/2} exp(-(x-μ)²/(2σ²²)) |

伯努利是二分类的数学抽象：X=1 表示事件发生，概率 p；X=0 表示不发生，概率 1-p。高斯分布由中心极限定理保证，大量独立同分布随机变量之和趋向正态分布，是自然界最常见的分布。

---

## 期望与方差

期望是随机变量的"中心位置"，定义为随机变量按概率加权平均。离散型：E[X] = ∑_x x·P(X=x)；连续型：E[X] = ∫ x·f_X(x)dx。这刻画了分布的位置参数。

方差度量取值偏离期望的程度：Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²。对于 X ~ N(μ, σ²)，有 E[X] = μ，Var(X) = σ²。方差是二阶矩，反映分布的离散程度，标准差 σ = √Var(X) 与 X 同量纲。

期望的线性性质：E[aX + bY] = aE[X] + bE[Y]。当 X、Y 独立时，Var(X + Y) = Var(X) + Var(Y)。这两个性质是推导许多统计量的基础。

---

## 贝叶斯定理

贝叶斯定理描述条件概率之间的转换：

$$P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)} \propto P(X|\theta)P(\theta)$$

其中：θ 为未知参数（如高斯分布的 μ、σ²），X 为观测数据。后验分布 P(θ|X) 结合了先验知识 P(θ) 和数据提供的似然 P(X|θ)。分母 P(X) = ∫ P(X|θ)P(θ)dθ 是归一化常数，称为证据。

贝叶斯视角下，参数本身是随机变量，具有分布。频率学派则认为参数是固定但未知的常数。在机器学习中，正则化可视为对参数先验的引入：L2 正则对应高斯先验，L1 对应拉普拉斯先验。

---

## 最大似然估计

MLE 是参数估计的经典方法：给定观测数据 X = {x₁, ..., x_n}，选择 θ̂ 使似然函数最大：

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} L(\theta; X) = \arg\max_{\theta} \prod_{i=1}^n P(x_i|\theta)$$

直接最大化乘积计算困难，取对数得对数似然：

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log P(x_i|\theta)$$

对数单调递增，最大化对数似然等价于最大化似然。对数将乘积转为求和，避免数值下溢，简化求导。

**高斯分布的 MLE**

设 X₁, ..., X_n ~ iid N(μ, σ²)，似然函数：

$$L(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$

对数似然：

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2$$

对 μ 求偏导并令为零：

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0 \implies \hat{\mu} = \frac{1}{n}\sum_{i=1}^n x_i$$

对 σ² 求偏导：

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \hat{\mu})^2 = 0$$

$$\implies \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2$$

这意味着：高斯分布的 MLE 恰好是样本均值和样本方差（分母 n 而非 n-1，是有偏估计）。

---

## 交叉熵损失与对数似然

考虑 K 分类问题：输入 x，输出标签 y ∈ {1, ..., K}（one-hot 编码）。模型输出概率向量 p̂ = [p̂₁, ..., p̂_K]，其中 p̂_k = P(y=k|x; θ) 由 softmax 归一化：

$$\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

假设训练样本 {(x⁽ⁱ⁾, y⁽ⁱ⁾)} 独立同分布，似然函数：

$$L(\theta) = \prod_{i=1}^n P(y^{(i)}|x^{(i)}; \theta) = \prod_{i=1}^n \prod_{k=1}^K \hat{p}_k^{(i)\mathbb{1}\{y^{(i)}=k\}}$$

对数似然：

$$\ell(\theta) = \sum_{i=1}^n \sum_{k=1}^K \mathbb{1}\{y^{(i)}=k\} \log \hat{p}_k^{(i)}$$

最大化对数似然等价于最小化负对数似然：

$$-\ell(\theta) = -\sum_{i=1}^n \sum_{k=1}^K \mathbb{1}\{y^{(i)}=k\} \log \hat{p}_k^{(i)}$$

这正是交叉熵损失的定义：

$$H(y, \hat{p}) = -\sum_{k=1}^K y_k \log \hat{p}_k$$

其中 y 是 one-hot 标签向量，$\hat{p}$ 是模型预测概率。对于 n 个样本，总损失：

$$J(\theta) = \frac{1}{n}\sum_{i=1}^n H(y^{(i)}, \hat{p}^{(i)})$$

**二分类特例**：K=2 时，交叉熵退化为：

$$H(y, \hat{p}) = -[y \log \hat{p} + (1-y) \log(1-\hat{p})]$$

这是二分类任务常用的二元交叉熵损失。

---

## Python 实现

```python
import numpy as np

def mle_gaussian(samples):
    """高斯分布的最大似然估计"""
    n = len(samples)
    mu_hat = np.mean(samples)
    sigma2_hat = np.var(samples, ddof=0)  # MLE 用分母 n
    return mu_hat, sigma2_hat

# 示例
np.random.seed(42)
samples = np.random.normal(loc=5.0, scale=2.0, size=1000)
mu_est, sigma2_est = mle_gaussian(samples)
print(f"μ̂ = {mu_est:.3f}, σ̂² = {sigma2_est:.3f}")
```

```python
def cross_entropy_loss(y_true, y_pred, eps=1e-15):
    """
    多分类交叉熵损失
    y_true: (n_samples, n_classes) one-hot 标签
    y_pred: (n_samples, n_classes) 概率预测（已 softmax）
    """
    # 避免对数计算时的数值问题
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# 示例
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
print(f"Cross Entropy: {cross_entropy_loss(y_true, y_pred):.4f}")
```

---

## 工程细节与局限性

**MLE 的局限性**：
1. 过拟合：当参数数量接近样本数时，MLE 可能拟合噪声
2. 无解情况：某些分布的 MLE 不存在（如均匀分布 U[0,θ]，最大样本值小于真实 θ）
3. 数值稳定性：直接计算乘积易下溢，必须用对数空间运算

**交叉熵工程要点**：
1. 数值裁剪：log(0) 无定义，需将概率裁剪到 [ε, 1-ε]，ε 取 1e-15 左右
2. LogSumExp 技巧：计算 softmax 的 log 时，log(∑e^z) 用 max(z) + log(∑e^{z-max}) 避免溢出
3. 标签平滑：将 one-hot 标签改为 y = (1-α)e_k + α/K，缓解模型过度自信，α ∈ [0,1] 通常取 0.1

**与 MAP 的关系**：
最大后验估计（MAP）引入先验 P(θ)：

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \log P(X|\theta) + \log P(\theta)$$

当先验 P(θ) 为均匀分布时，MAP 退化为 MLE。高斯先验等价于 L2 正则，Laplace 先验等价于 L1 正则。

---

## 参考资料</think>1. Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. Chapter 2-4.
2. Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*. MIT Press, 2016. Chapter 3.
3. Murphy, K. P. *Machine Learning: A Probabilistic Perspective*. MIT Press, 2012. Chapter 4.
4. https://arxiv.org/abs/1412.6572 (Label Smoothing Regularization)
