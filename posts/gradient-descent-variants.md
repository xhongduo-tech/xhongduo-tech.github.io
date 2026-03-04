## 梯度下降三形态：SGD、Mini-batch、Full-batch 的收敛分析

梯度下降的核心思想：沿梯度反方向迭代更新参数，使目标函数值单调下降。给定目标函数 $f(\theta): \mathbb{R}^d \to \mathbb{R}$，标准梯度下降更新规则为：

$$\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$$

其中 $\eta > 0$ 为学习率。当数据规模 $n$ 较大时，$\nabla f(\theta) = \frac{1}{n}\sum_{i=1}^n \nabla \ell_i(\theta)$ 的计算成本成为瓶颈，衍生出三种变体。

---

## Full-batch Gradient Descent

Full-batch GD 使用整个训练集计算梯度：

$$\theta_{t+1} = \theta_t - \eta \frac{1}{n} \sum_{i=1}^n \nabla \ell_i(\theta_t)$$

该梯度是 $\nabla f(\theta_t)$ 的无偏估计，方差为零，更新方向确定。

### 凸函数收敛性质

设 $f$ 为 $L$-光滑且 $\mu$-强凸函数，条件数 $\kappa = L/\mu$。固定学习率 $\eta < 2/L$ 时：

$$f(\theta_t) - f^* \leq \left(1 - \eta \mu\right)^t [f(\theta_0) - f^*]$$

最优学习率 $\eta^* = \frac{2}{L+\mu}$，收敛速率 $O\left(\left(\frac{\kappa-1}{\kappa+1}\right)^t\right)$，线性收敛。

### 非凸函数中的局限性

深层网络的损失函数存在大量鞍点而非局部极小。鞍点的 Hessian 矩阵既有正特征值也有负特征值：

$$\nabla f(\theta^*) = 0, \quad \lambda_{\text{min}}(H) < 0, \quad \lambda_{\text{max}}(H) > 0$$

Full-batch GD 在鞍点附近的更新幅度 $\|\theta_{t+1} - \theta_t\| = O(\eta\|\nabla f\|)$，当梯度接近零时更新停滞，难以利用负曲率方向逃离。

### 内存与计算瓶颈

单次迭代需存储全部 $n$ 个样本，复杂度 $O(nd)$。ImageNet-1K ($n \approx 1.28\times 10^6$, $d \approx 10^8$) 单张 VGG-16 特征图约需 500GB 内存，超出单机容量。

---

## Stochastic Gradient Descent

SGD 每步随机采样一个样本或一个 mini-batch 计算梯度：

$$\theta_{t+1} = \theta_t - \eta_t \nabla \ell_{i_t}(\theta_t), \quad i_t \sim \text{Uniform}\{1,\dots,n\}$$

$\nabla \ell_{i_t}(\theta_t)$ 是真实梯度的无偏估计：$\mathbb{E}[\nabla \ell_{i_t}(\theta_t)] = \nabla f(\theta_t)$，但存在方差。

### 收敛分析：凸情形

设 $f$ 为 $L$-光滑、$\mu$-强凸，$\mathbb{E}[\|\nabla \ell_i(\theta)\|^2] \leq G^2$。使用递减学习率 $\eta_t = \frac{c}{\mu t + L}$：

$$\mathbb{E}[f(\theta_t) - f^*] \leq \frac{G^2 + 2\mu(f(\theta_0)-f^*)}{2\mu t + 2c} = O\left(\frac{1}{t}\right)$$

相较于 Full-batch 的线性收敛，SGD 收敛速率降至次线性，但单步复杂度从 $O(nd)$ 降至 $O(d)$。

### 随机性的逃离机制

在鞍点附近，单个样本的梯度估计 $\nabla \ell_i(\theta)$ 往往偏离真实梯度 $\nabla f(\theta) \approx 0$，噪声使 SGD 能够沿负曲率方向逃离。

考虑二维鞍点 $f(x,y) = x^2 - y^2$，真实梯度 $\nabla f = [2x, -2y]^T$。添加高斯噪声 $\epsilon \sim \mathcal{N}(0,\sigma^2 I)$ 后：

$$\theta_{t+1} - \theta_t = -\eta [2x, -2y]^T - \eta \epsilon$$

沿 $y$ 方向（负曲率），负梯度 $+\eta 2y$ 与噪声叠加，即使 $y$ 接近零，噪声也可推动参数远离鞍点。

### 收敛震荡与方差问题

SGD 的收敛轨迹呈现"之"字形震荡。设真实梯度方向为 $u$，噪声方差为 $\sigma^2$，则每次更新与最优方向夹角 $\alpha$ 满足：

$$\cos\alpha = \frac{u^T(u+\epsilon)}{\|u\| \cdot \|u+\epsilon\|} \approx \frac{\|u\|}{\sqrt{\|u\|^2+\sigma^2}}$$

当 $\sigma \gg \|u\|$ 时，$\cos\alpha \to 0$，梯度方向完全随机化，收敛停滞。

---

## Mini-batch Gradient Descent

Mini-batch 是 SGD 的推广：每步采样 $B$ 个样本计算平均梯度：

$$\theta_{t+1} = \theta_t - \eta_t \frac{1}{B} \sum_{j=1}^B \nabla \ell_{i_{t,j}}(\theta_t)$$

梯度方差随 batch size 增大而减小。假设样本梯度独立同分布，方差关系：

$$\text{Var}\left(\frac{1}{B}\sum_{j=1}^B g_j\right) = \frac{1}{B^2}\sum_{j=1}^B \text{Var}(g_j) = \frac{\sigma^2}{B}$$

### 收敛速率权衡

给定固定计算预算（总样本访问量 $S$），Full-batch 执行 $S/n$ 次迭代，Mini-batch 执行 $S/B$ 次。

凸函数下，收敛所需迭代次数与梯度方差呈正相关，与步长平方呈负相关。Mini-batch 以 $O(\sqrt{B})$ 的迭代次数换得 $O(1/\sqrt{B})$ 的梯度噪声。

### Batch Size 对优化轨迹的影响

凸函数中，batch size 主要影响收敛稳定性而非最终解的质量。非凸函数中，batch size 通过控制梯度噪声影响泛化性能。

| Batch Size | 梯度噪声 | 收敛稳定性 | 泛化性能 | 计算效率 |
|------------|---------|-----------|---------|---------|
| 1 (SGD)    | 高      | 震荡      | 较好    | 低并行  |
| 32-256     | 中      | 平滑      | 最优    | 高并行  |
| 4096+      | 低      | 稳定      | 稍差    | 饱和    |

---

## 线性缩放规则

线性缩放规则：当 batch size 扩大 $k$ 倍时，学习率可同步放大 $k$ 倍以维持相似的收敛行为。

### 直观解释

Mini-batch 梯度方差为 $\sigma^2/B$，增大 batch size 等价于降低梯度噪声。噪声是限制学习率的主要因素——高噪声下大学习率会导致发散。因此，噪声减半时可安全地将学习率加倍。

### 严格证明思路

设 $\hat{g}_B = \frac{1}{B}\sum_{i=1}^B g_i$ 为 batch size $B$ 下的梯度估计，$\hat{g}_{kB}$ 为 batch size $kB$ 下的估计。两者期望相同，方差关系：

$$\text{Var}(\hat{g}_{kB}) = \frac{\text{Var}(\hat{g}_B)}{k}$$

考虑参数更新的一阶泰勒展开：

$$f(\theta - \eta \hat{g}_B) \approx f(\theta) - \eta \hat{g}_B^T \nabla f(\theta) + \frac{\eta^2}{2} \hat{g}_B^T H(\theta) \hat{g}_B$$

期望下降量：

$$\mathbb{E}[f(\theta) - f(\theta - \eta \hat{g}_B)] = \eta \|\nabla f(\theta)\|^2 - \frac{\eta^2}{2} \text{Tr}\left(H(\theta) \text{Var}(\hat{g}_B)\right) + \frac{\eta^2}{2} \nabla f(\theta)^T H(\theta) \nabla f(\theta)$$

当 $\text{Var}(\hat{g}_B)$ 缩小至 $1/k$ 时，取 $\eta' = k\eta$ 使得期望下降量保持同阶。此证明依赖于梯度估计的高斯性假设和 Hessian 的平滑性。

### 适用范围与限制

线性缩放规则在以下条件下有效：
1. 模型宽度足够大（过参数化），Hessian 条件数稳定
2. Batch size 在合理范围内（通常 $B \leq 8192$）
3. 数据分布无剧烈变化（无异常样本簇）

当 batch size 超过临界值（通常 $10^4 \sim 10^5$），线性缩放规则失效：
- 梯度方差已降至非凸地形噪声水平，继续增大 batch 不再显著降低噪声
- 学习率受限于模型稳定性（如 BN 层的数值稳定性），而非梯度噪声

---

## 工程实践要点

### Learning Rate Warmup

大规模训练中采用大 batch size 时，直接使用线性缩放的学习率会导致初期训练不稳定。Warmup 策略：初始学习率设为目标值的 $1/10$，线性增长至目标值（前 5% 训练步数）。

原因：模型初始化时梯度分布不稳定，大学习率易引发数值爆炸。Warmup 期间参数移动幅度较小，逐渐适应大学习率的更新规模。

### Batch Size 与 GPU 内存

单张 GPU 的 batch size 受显存限制，主要消耗：
- 激活值：$O(B \times d_{\text{hidden}} \times L)$
- 梯度：$O(B \times d_{\text{hidden}} \times L)$
- 优化器状态（Adam）：$O(2 \times d_{\text{model}})$

梯度累积（gradient accumulation）可绕过显存限制：累积 $k$ 个小 batch 的梯度后执行一次参数更新，等效于 batch size 扩大 $k$ 倍，但通信开销不变。

### 动态 Batch Size 调度

训练后期逐渐增大 batch size 可改善收敛稳定性。策略：从初始 $B_0$ 开始，每 $E$ 个 epoch 按 $B_{t+1} = \min(B_{\text{max}}, \lfloor B_t \times 1.1 \rfloor)$ 增大。

依据：训练初期损失曲面陡峭，需要高梯度噪声探索；训练后期进入平坦区域，降低噪声有助于精确定位最优解。

---

## 局限性与开放问题

1. **理论差距**：非凸设置下的收敛速率仍依赖强假设（如 PL 条件），实际深度网络的收敛行为难以严格刻画。

2. **泛化悖论**：SGD 的高梯度噪声反而带来更好泛化，机制尚无统一定论——主流解释包括"平坦最小值理论"和"随机正则化效应"。

3. **极大规模 batch**：当 $B$ 接近数据集规模 $n$ 时，mini-batch 退化为 full-batch，线性缩放规则完全失效。当前研究聚焦于自适应学习率与 batch size 的联合调整。

4. **自适应优化器**：Adam、LAMB 等自适应方法对 batch size 的敏感度低于 SGD，但引入新的超参数（$\beta_1, \beta_2$），调参复杂度上升。

---

## 代码实现

以下代码对比三种梯度下降形态在合成数据上的收敛轨迹：

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_data(n=1000, d=10, noise=0.1):
    """生成线性回归数据"""
    X = np.random.randn(n, d)
    theta_true = np.random.randn(d)
    y = X @ theta_true + noise * np.random.randn(n)
    return X, y, theta_true

def mse_loss(X, y, theta):
    """均方误差损失"""
    return np.mean((X @ theta - y)**2) / 2

def mse_gradient(X, y, theta):
    """梯度"""
    return X.T @ (X @ theta - y) / len(y)

def full_batch_gd(X, y, lr=0.01, epochs=100):
    """Full-batch 梯度下降"""
    n, d = X.shape
    theta = np.zeros(d)
    losses = []
    for _ in range(epochs):
        grad = mse_gradient(X, y, theta)
        theta -= lr * grad
        losses.append(mse_loss(X, y, theta))
    return theta, losses

def sgd(X, y, lr=0.01, epochs=100):
    """随机梯度下降"""
    n, d = X.shape
    theta = np.zeros(d)
    losses = []
    indices = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(indices)
        for i in indices:
            grad = mse_gradient(X[i:i+1], y[i:i+1], theta)
            theta -= lr * grad
        losses.append(mse_loss(X, y, theta))
    return theta, losses

def mini_batch_gd(X, y, lr=0.01, batch_size=32, epochs=100):
    """Mini-batch 梯度下降"""
    n, d = X.shape
    theta = np.zeros(d)
    losses = []
    n_batches = (n + batch_size - 1) // batch_size
    for _ in range(epochs):
        indices = np.random.permutation(n)
        for b in range(n_batches):
            batch_idx = indices[b*batch_size:(b+1)*batch_size]
            grad = mse_gradient(X[batch_idx], y[batch_idx], theta)
            theta -= lr * grad
        losses.append(mse_loss(X, y, theta))
    return theta, losses

# 实验
np.random.seed(42)
X, y, theta_true = generate_data(n=2000, d=20)

theta_full, loss_full = full_batch_gd(X, y, lr=0.01, epochs=100)
theta_sgd, loss_sgd = sgd(X, y, lr=0.01, epochs=100)
theta_mb32, loss_mb32 = mini_batch_gd(X, y, lr=0.01, batch_size=32, epochs=100)
theta_mb256, loss_mb256 = mini_batch_gd(X, y, lr=0.01, batch_size=256, epochs=100)

# 可视化
plt.figure(figsize=(10, 6))
plt.semilogy(loss_full, label='Full-batch', linewidth=2)
plt.semilogy(loss_sgd, label='SGD (B=1)', alpha=0.7)
plt.semilogy(loss_mb32, label='Mini-batch (B=32)', alpha=0.7)
plt.semilogy(loss_mb256, label='Mini-batch (B=256)', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.title('Gradient Descent Variants: Convergence Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

线性缩放规则验证：

```python
def linear_scaling_experiment(X, y, base_lr=0.01, base_batch=32, scale_factors=[1, 2, 4, 8]):
    """验证线性缩放规则"""
    results = {}
    for k in scale_factors:
        batch_size = base_batch * k
        lr = base_lr * k  # 线性缩放
        _, losses = mini_batch_gd(X, y, lr=lr, batch_size=batch_size, epochs=50)
        results[f'B={batch_size}, lr={lr:.3f}'] = losses
    return results

results = linear_scaling_experiment(X, y)

plt.figure(figsize=(10, 6))
for label, losses in results.items():
    plt.plot(losses, label=label)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Linear Scaling Rule: lr ∝ batch_size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 参考资料

1. Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. *Proceedings of COMPSTAT*, 177–186.

2. Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017). On large-batch training for deep learning: Generalization gap and sharp minima. *ICLR*.

3. Smith, S. L., Bansal, N., Butts, J., Gao, A., & Le, Q. V. (2018). Don't decay the learning rate, increase the batch size. *ICLR*.

4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 8: Optimization for Training Deep Models.
