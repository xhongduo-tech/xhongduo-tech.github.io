## 内部协变量偏移

深度网络训练时，每层输入分布随前层参数更新而漂移。这种现象称为内部协变量偏移（Internal Covariate Shift）。

数学上，第 $l$ 层输入 $\mathbf{h}^{l} = f_l(\mathbf{h}^{l-1}; \theta^{l})$。当 $\theta^{l-1}$ 更新为 $\theta^{l-1} + \Delta \theta^{l-1}$，$\mathbf{h}^{l}$ 的分布 $P(\mathbf{h}^{l})$ 发生变化，第 $l+1$ 层需要不断调整 $\theta^{l+1}$ 来适应新分布。

这导致两个问题：优化空间扭曲（梯度方向与最优下降方向偏离），学习率需保守设置（否则某层输入突变引发震荡）。

---

## BatchNorm 精确定义

BatchNorm 对 mini-batch 内每个特征维度独立做归一化，然后引入可学习参数恢复表达能力。

给定 mini-batch $\{\mathbf{x}_i\}_{i=1}^m$，$\mathbf{x}_i \in \mathbb{R}^d$，BatchNorm 对每个特征维度 $j$：

$$\mu_j = \frac{1}{m}\sum_{i=1}^m x_{i,j}, \quad \sigma_j^2 = \frac{1}{m}\sum_{i=1}^m (x_{i,j} - \mu_j)^2$$

归一化后：

$$\hat{x}_{i,j} = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$$

其中 $\epsilon$ 防止除零（典型值 $10^{-5}$）。可学习参数 $\gamma \in \mathbb{R}^d, \beta \in \mathbb{R}^d$：

$$y_{i,j} = \gamma_j \hat{x}_{i,j} + \beta_j$$

这意味着 $\gamma$ 控制缩放，$\beta$ 控制平移。若 $\gamma_j = \sqrt{\sigma_j^2 + \epsilon}$ 且 $\beta_j = \mu_j$，则 $y_{i,j} = x_{i,j}$，BN 等价于恒等变换——网络有"退出归一化"的能力。

维度：$\mathbf{x}_i \in \mathbb{R}^{d}$，$\gamma, \beta \in \mathbb{R}^{d}$，$\mu, \sigma \in \mathbb{R}^{d}$。

---

## 前向传播实现

```python
import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5, momentum=0.9,
                       running_mean=None, running_var=None, mode='train'):
    """
    x: (N, D) 输入数据
    gamma, beta: (D,) 可学习参数
    mode: 'train' 或 'test'
    running_mean, running_var: 推理时的运行统计量
    返回: (N, D) 输出, cache 用于反向传播
    """
    N, D = x.shape
    
    if mode == 'train':
        # 计算当前 batch 的均值和方差
        mu = np.mean(x, axis=0)  # (D,)
        var = np.var(x, axis=0)  # (D,)
        
        # 更新运行统计量（指数移动平均）
        if running_mean is None:
            running_mean = mu
            running_var = var
        else:
            running_mean = momentum * running_mean + (1 - momentum) * mu
            running_var = momentum * running_var + (1 - momentum) * var
            
        # 归一化
        x_centered = x - mu
        x_normalized = x_centered / np.sqrt(var + eps)
        out = gamma * x_normalized + beta
        
        # 缓存中间结果用于反向传播
        cache = (x, x_normalized, mu, var, gamma, beta, eps)
        
    else:  # mode == 'test'
        # 推理时使用训练期间累积的运行统计量
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalized + beta
        cache = None
    
    return out, cache, running_mean, running_var
```

关键设计：训练时用 batch 统计量计算梯度，同时用指数移动平均（EMA）维护全局统计量；推理时直接使用 EMA 值，不依赖 batch 数据。

---

## 反向传播推导

需要求梯度 $\frac{\partial L}{\partial \mathbf{x}_i}$。从 $L$ 对 $y_{i,j}$ 开始：

$$\frac{\partial L}{\partial y_{i,j}} = \frac{\partial L}{\partial y_{i,j}}$$

对 $\gamma_j, \beta_j$ 直接累加：

$$\frac{\partial L}{\partial \gamma_j} = \sum_{i=1}^m \frac{\partial L}{\partial y_{i,j}} \cdot \hat{x}_{i,j}$$
$$\frac{\partial L}{\partial \beta_j} = \sum_{i=1}^m \frac{\partial L}{\partial y_{i,j}}$$

对 $\hat{x}_{i,j}$：

$$\frac{\partial L}{\partial \hat{x}_{i,j}} = \frac{\partial L}{\partial y_{i,j}} \cdot \gamma_j$$

现在需要 $\frac{\partial L}{\partial x_{i,j}}$。注意到 $\hat{x}_{i,j}$ 依赖于所有 $x_{k,j}$（通过 $\mu_j, \sigma_j^2$）：

$$\frac{\partial \hat{x}_{i,j}}{\partial x_{i,j}} = \frac{\partial}{\partial x_{i,j}} \left( \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} \right)$$

分拆为三项（简记 $\sigma_j = \sqrt{\sigma_j^2 + \epsilon}$）：

$$\frac{\partial \hat{x}_{i,j}}{\partial x_{i,j}} = \frac{1}{\sigma_j} - \frac{1}{\sigma_j^3}(x_{i,j} - \mu_j) \cdot \frac{\partial \mu_j}{\partial x_{i,j}} - \frac{1}{\sigma_j^3}(x_{i,j} - \mu_j) \cdot \frac{1}{2} \cdot 2(x_{i,j} - \mu_j) \cdot \frac{\partial \sigma_j^2}{\partial x_{i,j}}$$

其中：

$$\frac{\partial \mu_j}{\partial x_{i,j}} = \frac{1}{m}, \quad \frac{\partial \sigma_j^2}{\partial x_{i,j}} = \frac{2}{m}(x_{i,j} - \mu_j)$$

合并后：

$$\frac{\partial \hat{x}_{i,j}}{\partial x_{i,j}} = \frac{1}{\sigma_j} - \frac{1}{m\sigma_j^3}(x_{i,j} - \mu_j) - \frac{2}{m\sigma_j^3}(x_{i,j} - \mu_j)^2$$

进一步整理为更简洁的形式：

$$\frac{\partial L}{\partial x_{i,j}} = \frac{\gamma_j}{m\sigma_j} \left[ m \cdot \frac{\partial L}{\partial y_{i,j}} - \sum_{k=1}^m \frac{\partial L}{\partial y_{k,j}} - \hat{x}_{i,j} \sum_{k=1}^m \frac{\partial L}{\partial y_{k,j}} \hat{x}_{k,j} \right]$$

这意味着梯度不仅取决于当前位置，还依赖于整个 batch 的梯度分布——BN 引入了样本间的耦合。

---

## 反向传播实现

```python
def batch_norm_backward(dout, cache):
    """
    dout: (N, D) 上层梯度
    cache: 前向传播缓存的中间结果
    返回: dx, dgamma, dbeta
    """
    x, x_normalized, mu, var, gamma, beta, eps = cache
    N, D = x.shape
    
    # 对 x_normalized 的梯度
    d_x_normalized = dout * gamma  # (N, D)
    
    # 对方差 sigma^2 的梯度
    # var = mean((x - mu)^2)
    # dvar = d_x_normalized * (x - mu) * (-0.5) * (var + eps)^(-3/2)
    d_var = np.sum(d_x_normalized * (x - mu) * (-0.5) * np.power(var + eps, -1.5), axis=0)
    
    # 对均值 mu 的梯度
    # mu 影响两项：通过 x - mu，通过 var
    d_mu = np.sum(d_x_normalized * (-1) / np.sqrt(var + eps), axis=0) + \
           d_var * np.mean(-2 * (x - mu), axis=0)
    
    # 对 x 的梯度
    # x 通过三项影响输出：直接项、mu 项、var 项
    d_x_centered = d_x_normalized / np.sqrt(var + eps)  # 直接项
    d_x_mu = d_mu / N  # mu 对 x 的贡献
    d_x_var = d_var * 2 * (x - mu) / N  # var 对 x 的贡献
    dx = d_x_centered + d_x_mu + d_x_var
    
    # 对 gamma, beta 的梯度
    dgamma = np.sum(dout * x_normalized, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    return dx, dgamma, dbeta
```

计算复杂度：前向 $O(m \cdot d)$，反向 $O(m \cdot d)$。梯度稳定性的关键在于 $\epsilon$ 防止方差为零时的数值爆炸。

---

## Batch Size = 1 时的退化

设 $m = 1$，则 $\mu = x_1$，$\sigma^2 = 0$。代入归一化公式：

$$\hat{x}_1 = \frac{x_1 - x_1}{\sqrt{0 + \epsilon}} = 0$$

这意味着 BN 输出恒为 $\beta$，输入信息完全丢失。梯度方面：

$$\frac{\partial L}{\partial x_1} = 0$$

梯度消失，无法学习。因此 BN 要求 batch size $\geq 2$，实际工程中常用 32、64、128。

| Batch Size | $\mu_j$ | $\sigma_j^2$ | 行为 |
|------------|---------|--------------|------|
| 1 | $x_{1,j}$ | 0 | 输出恒为 $\beta_j$，无梯度 |
| 2 | $(x_{1,j}+x_{2,j})/2$ | $((x_{1,j}-x_{2,j})/2)^2$ | 可训练但高方差 |
| $\geq 32$ | 稳定估计 | 稳定估计 | 训练稳定 |

---

## 推理时的统计量

推理时通常无法获得完整的 mini-batch，或 batch size 变化。解决方案是用训练期间累积的运行统计量：

$$\hat{\mu}_{\text{test}} = \text{EMA}(\mu_{\text{train}}), \quad \hat{\sigma}_{\text{test}}^2 = \text{EMA}(\sigma_{\text{train}}^2)$$

EMA 更新：$\theta_t = \text{momentum} \times \theta_{t-1} + (1 - \text{momentum}) \times \theta_{\text{batch}}$，典型 momentum = 0.9 或 0.99。

工程陷阱：模型加载后若忘记冻结 running mean/var，推理阶段的 batch 会污染统计量。此外，momentum 选取需权衡：高 momentum 稳定但更新慢，低 momentum 响应快但噪声大。

---

## 局限性与替代方案

BN 的核心限制是依赖 batch 统计量，导致以下场景失效：

1. **小 batch 训练**：检测、分割等任务输入尺寸各异，实际 batch size 常被迫设为 2-4，此时估计不稳定
2. **RNN**：序列长度 $T$ 可变，BN 需对时间步 padding，破坏时序依赖
3. **Transformer**：自注意力机制本身已引入位置归一化效果，BN 与并行推理冲突
4. **分布式训练**：多机多卡时若不做全局同步，各卡统计量不一致

替代方案对比：

| 方法 | 统计范围 | 适用场景 | 训练/推理一致性 |
|------|----------|----------|-----------------|
| BatchNorm | mini-batch | CV 分类/检测 | 依赖 EMA |
| LayerNorm | 样本维度内 | Transformer / RNN | 一致 |
| InstanceNorm | 样本×通道 | 风格迁移 | 一致 |
| GroupNorm | 通道分组 | 小 batch CV | 一致 |

---

## 工程实践建议

1. **位置选择**：BN 通常放在卷积/全连接之后、激活函数之前。现代实践（如 ResNet v2）倾向于 BN → ReLU 的顺序，这样梯度直接作用于激活值，缓解梯度消失。
2. **初始化**：$\gamma$ 初始化为 1，$\beta$ 初始化为 0，使初始阶段 BN 接近恒等变换。
3. **数值稳定性**：$\epsilon$ 在 float32 时取 $10^{-5}$，混合精度训练（float16）时建议 $10^{-3}$。
4. **性能开销**：前向额外约 15-20% 计算量（主要来自减法、除法），但允许更高学习率（典型 2-10 倍），总体训练时间下降。
5. **与 Dropout 共存**：BN 后接 Dropout 时， Dropout 率需适当降低（0.1 以下），否则噪声叠加影响收敛。

---

## 参考资料

1. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
2. Wu, Y., & He, K. (2018). Group Normalization. ECCV. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
3. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
4. Santos, L. (2023). Understanding Batch Normalization. Towards Data Science.
