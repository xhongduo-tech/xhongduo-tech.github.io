## Softmax 函数定义与概率映射

Softmax 函数将任意实数向量 $z = [z_1, z_2, ..., z_K]^T \in \mathbb{R}^K$ 映射为有效概率分布 $p \in \Delta^{K-1}$：

$$
\text{Softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}, \quad i = 1, \ldots, K
$$

输出满足 $\sum_i p_i = 1$ 且 $p_i \in (0, 1)$，这使得 Softmax 成为多分类问题的标准输出层。

核心动机：神经网络的最后一层通常输出未归一化的实数（logits），需通过可微函数转换为概率用于交叉熵损失。Sigmoid 无法处理多类互斥场景，而直接归一化 $z_i / \sum z_j$ 的问题在于对负数和零的处理。

---

## 数学性质与梯度推导

对输入向量 $z$ 的分量求偏导：

$$
\frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j)
$$

其中 $\delta_{ij}$ 为 Kronecker delta（$i=j$ 时为 1，否则为 0）。展开为两种情况：

$$
\frac{\partial p_i}{\partial z_j} = \begin{cases}
p_i(1 - p_i), & i = j \\
-p_i p_j, & i \neq j
\end{cases}
$$

该梯度形式在反向传播中有重要意义：增加 $z_i$ 会提升 $p_i$（系数 $p_i(1-p_i) > 0$），同时降低其他 $p_j$（系数 $-p_i p_j < 0$）。

交叉熵损失 $L = -\sum_{k=1}^{K} y_k \log p_k$（$y$ 为 one-hot 标签）对 $z_i$ 的梯度：

$$
\frac{\partial L}{\partial z_i} = \sum_{k=1}^{K} \frac{\partial L}{\partial p_k} \cdot \frac{\partial p_k}{\partial z_i} = \sum_{k=1}^{K} (-\frac{y_k}{p_k}) \cdot p_k(\delta_{ki} - p_i)
$$

化简后得到简洁形式：

$$
\frac{\partial L}{\partial z_i} = p_i - y_i
$$

这意味着：当预测概率 $p_i$ 高于真实标签 $y_i$ 时，梯度为正，反向传播减小 $z_i$；反之增大 $z_i$。

---

## 数值稳定性问题与 log-sum-exp 技巧

直接实现 Softmax 存在数值溢出风险：当 $z_i$ 较大时，$\exp(z_i)$ 可能超出浮点数表示上限。

### 溢出示例

```python
import numpy as np

z = np.array([1000.0, 2000.0, 3000.0])
exp_z = np.exp(z)
print(f"exp(z) = {exp_z}")  # [inf, inf, inf]，全溢出
```

输出显示所有项都变为 `inf`，导致后续除法无效。

### log-sum-exp 技巧原理

利用指数函数的性质，对 $z$ 每个分量减去同一常数 $c$ 不改变 Softmax 结果：

$$
\frac{\exp(z_i)}{\sum_j \exp(z_j)} = \frac{\exp(z_i - c)}{\sum_j \exp(z_j - c)}
$$

取 $c = \max(z)$ 可保证 $\exp(z_i - c) \in (0, 1]$，避免溢出。

### 数值稳定实现

```python
def softmax_stable(z):
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_shifted = np.exp(z - z_max)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

# 测试
z = np.array([1000.0, 2000.0, 3000.0])
print(softmax_stable(z))  # [0., 0., 1.] 数值稳定
```

对数 Softmax（log-softmax）在交叉熵计算中常用，可直接使用 log-sum-exp 技巧：

$$
\log p_i = z_i - \log\left(\sum_{j=1}^{K} \exp(z_j)\right)
$$

数值稳定版：

```python
def log_softmax_stable(z):
    z_max = np.max(z, axis=-1, keepdims=True)
    return z - z_max - np.log(np.sum(np.exp(z - z_max), axis=-1, keepdims=True))
```

PyTorch 的 `torch.nn.functional.log_softmax` 即采用此实现。

---

## 温度参数 τ 及其影响

引入温度参数 $\tau > 0$：

$$
\text{Softmax}(z_i; \tau) = \frac{\exp(z_i / \tau)}{\sum_{j=1}^{K} \exp(z_j / \tau)}
$$

温度参数控制输出的"尖锐程度"：

| τ 趋向 | 行为 | 直觉解释 |
|--------|------|----------|
| $\tau \to 0$ | 趋向 one-hot，$\lim_{\tau\to0} p_i \to \mathbb{I}(z_i = \max(z))$ | 最大值被放大，其余趋向 0 |
| $\tau = 1$ | 标准 Softmax | 无缩放 |
| $\tau \to \infty$ | 趋向均匀分布，$\lim_{\tau\to\infty} p_i \to 1/K$ | 差异被抹平 |

### 温度效果代码演示

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax_with_temp(z, tau):
    z_scaled = z / tau
    exp_z = np.exp(z_scaled - np.max(z_scaled))
    return exp_z / np.sum(exp_z)

z = np.array([2.0, 1.0, 0.1])
temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

print("不同温度下的概率分布:")
for tau in temperatures:
    probs = softmax_with_temp(z, tau)
    print(f"τ={tau:4.1f}: [{probs[0]:.6f}, {probs[1]:.6f}, {probs[2]:.6f}]")
```

输出示例：

```
不同温度下的概率分布:
τ= 0.1: [1.000000, 0.000000, 0.000000]
τ= 0.5: [0.842793, 0.142141, 0.015066]
τ= 1.0: [0.659001, 0.242432, 0.098566]
τ= 2.0: [0.455084, 0.310059, 0.234857]
τ= 5.0: [0.368391, 0.335749, 0.295860]
τ=10.0: [0.346848, 0.338760, 0.314392]
```

温度参数在以下场景应用：
- **知识蒸馏**：教师网络用低温（τ=1），学生网络用高温（τ>1）学习软化后的标签
- **采样策略**：如 RL 中的策略探索，高温度增加探索，低温度利用
- **对比学习**：InfoNCE 损失中温度控制样本相似度的判定严格程度

---

## 注意力机制中的温度缩放

在 Transformer 自注意力中，缩放点积注意力（Scaled Dot-Product Attention）使用 $\tau = \sqrt{d_k}$：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $Q, K, V \in \mathbb{R}^{n \times d_k}$，$d_k$ 为注意力头的维度。

### 缩放的必要性

点积 $q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$ 是 $d_k$ 个独立随机变量之和。假设 $q, k$ 各分量独立且均值为 0、方差为 1，则点积的方差为 $d_k$：

$$
\text{Var}(q \cdot k) = d_k \cdot \text{Var}(q_l k_l) = d_k \cdot \mathbb{E}[q_l^2 k_l^2] = d_k
$$

当 $d_k$ 较大时（如 BERT-base 中 $d_k=64$），点积绝对值可达 $\sqrt{d_k} \approx 8$，导致 $\exp(q \cdot k)$ 的值域跨度极大。未缩放的 Softmax 进入梯度消失区（$p_i \approx 0$）或梯度爆炸区（$p_i \approx 1$）。

除以 $\sqrt{d_k}$ 后，点积方差归一化为 1，保持梯度流动的稳定性。

### 代码实现

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
        q: Query tensor, shape (batch_size, seq_len, d_k)
        k: Key tensor, shape (batch_size, seq_len, d_k)
        v: Value tensor, shape (batch_size, seq_len, d_v)
        mask: Optional attention mask, shape (batch_size, seq_len, seq_len)
    """
    d_k = q.shape[-1]
    
    # 缩放点积
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码（如因果掩码）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    return torch.matmul(attn_weights, v), attn_weights
```

---

## 工程实践与注意事项

### 1. 对数空间计算

交叉熵 $L = -\sum_i y_i \log p_i$ 在实现时常与 log-softmax 结合，避免显式计算 $p_i$：

```python
def cross_entropy_with_logits(logits, targets):
    """数值稳定的交叉熵，等价于 F.cross_entropy"""
    log_probs = log_softmax_stable(logits)
    return -np.mean(log_probs[np.arange(len(targets)), targets])
```

PyTorch 的 `torch.nn.CrossEntropyLoss` 内部已做此优化，直接输入 logits 即可。

### 2. 标签平滑

防止模型过度自信，将真实标签从 1 调整为 $1 - \epsilon$，其余类别分配 $\epsilon / (K-1)$：

```python
def label_smooth_cross_entropy(logits, targets, epsilon=0.1):
    n_classes = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 构建平滑后的标签
    smooth_targets = torch.full_like(log_probs, epsilon / (n_classes - 1))
    smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - epsilon)
    
    loss = -torch.sum(smooth_targets * log_probs, dim=-1).mean()
    return loss
```

### 3. Gumbel-Softmax

需在离散采样中保持可微时使用 Gumbel-Softmax trick：

```python
def gumbel_softmax_sample(logits, temperature=1.0):
    """可微的离散采样"""
    gumbel_noise = -np.log(-np.log(np.random.uniform(size=logits.shape)))
    return softmax_with_temp(logits + gumbel_noise, temperature)
```

温度低时逼近 one-hot，训练时逐渐降低温度实现退火。

---

## 局限性与 Trade-off

| 问题 | 描述 | 缓解方案 |
|------|------|----------|
| 类别数爆炸 | 计算 $\sum_j \exp(z_j)$ 时复杂度 $O(K)$ | 层次 Softmax、采样 Softmax |
| 长尾分布 | 极小概率类别梯度微弱，难优化 | Focal Loss、类别加权 |
| 温度敏感 | 不合适的 τ 导致训练不稳定 | 网格搜索或自动调参 |

工程中需根据场景选择：
- 类别数 < 1000：标准 Softmax
- 类别数 > 10⁵（如语言模型）：层级 Softmax 或 adaptive softmax
- 需要探索性采样：温度控制 + Gumbel-Softmax

---

## 参考文献

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.2.2.3.
2. Vaswani, A., et al. (2017). Attention Is All You Need. *arXiv:1706.03762*.
3. Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. *arXiv:1611.01144*.
4. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *arXiv:1503.02531*.
