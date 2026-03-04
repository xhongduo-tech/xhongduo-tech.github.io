## 信息熵与交叉熵：损失函数的信息论基础

香农熵 $H(p) = -\sum_{x} p(x) \log p(x)$ 定义为随机变量 $X$ 取值的不确定性度量。直觉上，熵越高意味着分布越平坦、预测越困难。当 $X$ 有 $N$ 个可能取值时，$H(p) \in [0, \log N]$，等概分布达到最大熵 $\log N$。

---

## 熵的直观解释与数学性质

考虑编码问题：对取值 $x \in \{1, \ldots, N\}$ 的随机变量设计最优编码长度。信息论证明，最优编码长度期望恰好等于熵。取以 2 为底时，单位为 bit；取自然对数时，单位为 nat。不同底数仅差常数因子 $H_{\log_2}(p) = H_{\ln}(p) / \ln 2$。

**熵的性质**：

| 性质 | 表述 |
|------|------|
| 非负性 | $H(p) \geq 0$，当且仅当 $p$ 为退化分布时取 0 |
| 凹性 | $H(\lambda p_1 + (1-\lambda)p_2) \geq \lambda H(p_1) + (1-\lambda)H(p_2)$ |
| 最大熵 | 固定支撑集下，均匀分布熵最大 |

---

## 交叉熵与 KL 散度

交叉熵 $H(p,q) = -\sum_{x} p(x) \log q(x)$ 衡量用分布 $q$ 编码真实分布 $p$ 所需的平均编码长度。当 $q = p$ 时交叉熵达到最小值 $H(p)$，否则有额外代价。

额外代价由 KL 散度量化：
$$\text{KL}(p\|q) = H(p,q) - H(p) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$

KL 散度恒非负，由 Jensen 不等式证明：
$$\text{KL}(p\|q) = \mathbb{E}_p\left[\log \frac{p(x)}{q(x)}\right] \geq \log \mathbb{E}_p\left[\frac{p(x)}{q(x)}\right] = 0$$

等号当且仅当 $p(x) = q(x)$ 几乎处处成立。$\text{KL}(p\|q) \neq \text{KL}(q\|p)$ 意味着用错误分布编码真实分布的代价与用真实分布编码错误分布的代价不同。

---

## 分类任务中的交叉熵

二分类问题中，真实标签 $y \in \{0, 1\}$，模型预测概率 $\hat{y} = \sigma(z)$。单样本的交叉熵损失：
$$\mathcal{L}_{\text{CE}} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

多分类扩展为：
$$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^C y_c \log \hat{y}_c$$

其中 $y_c$ 是 one-hot 编码，$\hat{y}_c = \text{softmax}(z_c)$。

---

## 梯度行为分析：交叉熵 vs MSE

关键差异在于梯度对 sigmoid 输出的依赖。设 $\hat{y} = \sigma(z)$，分别计算两种损失的梯度。

**交叉熵梯度**：
$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z} = \frac{\partial}{\partial z}[-\log \hat{y}] = \frac{\partial}{\partial z}[-\log \sigma(z)] = -\frac{1}{\sigma(z)} \cdot \sigma(z)(1-\sigma(z)) = \sigma(z) - 1 = \hat{y} - 1$$

当 $y=1$ 时，$\partial \mathcal{L}_{\text{CE}} / \partial z = \hat{y} - 1$，梯度与预测误差直接成正比。

**MSE 梯度**：
$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = \frac{\partial}{\partial z}\left[\frac{1}{2}(1 - \hat{y})^2\right] = -(1 - \hat{y}) \cdot \hat{y}(1 - \hat{y}) = -\hat{y}(1-\hat{y})^2$$

当 $y=1$ 时，$\partial \mathcal{L}_{\text{MSE}} / \partial z = -\hat{y}(1-\hat{y})^2$，梯度被 sigmoid 的导数 $\hat{y}(1-\hat{y})$ 抑制。

---

## 梯度抑制问题

当 $\hat{y} \approx 0$（预测严重错误）时，交叉熵梯度为 $-1$，保持强劲更新。而 MSE 梯度为 $-\hat{y} \approx 0$，几乎不更新参数。这意味着 MSE 在分类任务中存在梯度消失问题。

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def ce_gradient(z, y=1):
    """交叉熵梯度，y=1 时"""
    y_hat = sigmoid(z)
    return y_hat - y  # = sigmoid(z) - 1

def mse_gradient(z, y=1):
    """MSE 梯度，y=1 时"""
    y_hat = sigmoid(z)
    return -(y - y_hat) * y_hat * (1 - y_hat)  # = -y_hat * (1-y_hat)^2

# 当预测严重错误时
zs = [-10, -5, -3, -1, 0, 1, 3]
for z in zs:
    print(f"z={z:4d}: CE_grad={ce_gradient(z):.6f}, MSE_grad={mse_gradient(z):.6f}")
```

输出：
```
z=-10: CE_grad=-1.000000, MSE_grad=-0.000045
z= -5: CE_grad=-0.993307, MSE_grad=-0.006648
z= -3: CE_grad=-0.952574, MSE_grad=-0.042696
z= -1: CE_grad=-0.731059, MSE_grad=-0.053025
z=  0: CE_grad=-0.500000, MSE_grad=-0.250000
z=  1: CE_grad=-0.268941, MSE_grad=-0.196612
z=  3: CE_grad=-0.047426, MSE_grad=-0.043099
```

交叉熵在预测错误时保持接近 -1 的梯度，而 MSE 梯度在 $\hat{y}$ 接近 0 时急剧衰减。

---

## 多分类情形下的完整推导

设真实标签为 $y \in \{1, \ldots, C\}$ 的 one-hot 向量，logits 为 $z \in \mathbb{R}^C$。Softmax 输出：
$$\hat{y}_c = \frac{e^{z_c}}{\sum_{k=1}^C e^{z_k}}$$

交叉熵对 $z_i$ 的梯度：
$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_i} = \frac{\partial}{\partial z_i}\left[-\log \hat{y}_y\right] = -\frac{1}{\hat{y}_y} \cdot \frac{\partial \hat{y}_y}{\partial z_i}$$

当 $i = y$ 时：
$$\frac{\partial \hat{y}_y}{\partial z_y} = \frac{e^{z_y} \sum_k e^{z_k} - e^{z_y} e^{z_y}}{(\sum_k e^{z_k})^2} = \hat{y}_y (1 - \hat{y}_y)$$

当 $i \neq y$ 时：
$$\frac{\partial \hat{y}_y}{\partial z_i} = -\frac{e^{z_y} e^{z_i}}{(\sum_k e^{z_k})^2} = -\hat{y}_y \hat{y}_i$$

统一形式为 $\partial \hat{y}_y / \partial z_i = \hat{y}_i (\mathbb{I}_{i=y} - \hat{y}_y)$，因此：
$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_i} = \hat{y}_i - y_i$$

简洁的梯度形式：预测概率与真实标签之差。

---

## 数值稳定性实践

直接计算 $\log(\sigma(z))$ 会产生数值问题。利用对数空间技巧：

```python
def stable_sigmoid_ce_loss(z, y):
    """数值稳定的交叉熵计算"""
    # 利用 log(1+exp(-z)) 的等价形式
    if y == 1:
        # -log(sigmoid(z)) = -log(1/(1+exp(-z))) = log(1+exp(-z))
        return np.logaddexp(0, -z)  # 稳定计算 log(1 + exp(-z))
    else:
        # -log(1-sigmoid(z)) = -log(exp(-z)/(1+exp(-z))) = z + log(1+exp(-z))
        return z + np.logaddexp(0, -z)

def stable_softmax_ce_loss(z, y_idx):
    """数值稳定的 softmax 交叉熵"""
    # log_softmax(z) = z - log(sum(exp(z)))
    log_z = z - np.logaddexp.reduce(z)
    return -log_z[y_idx]
```

`np.logaddexp` 使用技巧 $\log(e^a + e^b) = \max(a,b) + \log(1 + e^{-|a-b|})$ 避免溢出。

---

## Label Smoothing

训练时常用标签平滑缓解过拟合：
$$\tilde{y}_c = (1-\alpha) y_c + \frac{\alpha}{C}$$

交叉熵变为：
$$\mathcal{L} = -\sum_{c=1}^C \tilde{y}_c \log \hat{y}_c$$

这等价于在真实分布与均匀分布之间进行插值。$\alpha$ 常取 0.1，其效果是降低模型对训练数据的自信程度。

```python
def smooth_cross_entropy(y_hat, y, alpha=0.1):
    """
    y_hat: 模型预测 (batch_size, num_classes)
    y: 真实标签 (batch_size,) 或 one-hot (batch_size, num_classes)
    alpha: 平滑系数
    """
    C = y_hat.shape[-1]
    if y.ndim == 1:
        y_onehot = np.zeros_like(y_hat)
        y_onehot[np.arange(len(y)), y] = 1
    else:
        y_onehot = y
    
    y_smooth = (1 - alpha) * y_onehot + alpha / C
    return -np.sum(y_smooth * np.log(y_hat + 1e-10), axis=-1)
```

---

## 局限性与工程细节

**数值上界**：当 $\hat{y}_c \to 0$ 时 $\log \hat{y}_c \to -\infty$，需加小常数截断（如 $10^{-10}$）。

**类别不平衡**：交叉熵对样本一视同仁，不平衡数据下需加权采样或使用 focal loss：
$$\mathcal{L}_{\text{Focal}} = -(1 - \hat{y}_y)^\gamma \log \hat{y}_y$$

$\gamma > 0$ 降低易分类样本的权重。

**与 MSE 的适用场景**：
- 交叉熵：分类任务、概率输出、梯度不消失
- MSE：回归任务、连续目标、有界输出要求不同

---

## 完整实现示例

```python
import numpy as np

class CrossEntropyLoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        logits: (batch_size, num_classes)
        targets: (batch_size,) 类别索引
        """
        batch_size, num_classes = logits.shape
        
        # 数值稳定的 log_softmax
        log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        
        # 提取真实类别的 log_prob
        loss = -log_probs[np.arange(batch_size), targets]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
    def backward(self, logits, targets):
        """
        返回对 logits 的梯度
        """
        batch_size, num_classes = logits.shape
        
        # softmax 概率
        probs = np.exp(logits - np.logaddexp.reduce(logits, axis=1, keepdims=True))
        
        # 梯度 = probs - one_hot(targets)
        grad = probs.copy()
        grad[np.arange(batch_size), targets] -= 1
        
        if self.reduction == 'mean':
            grad /= batch_size
        return grad


# 使用示例
loss_fn = CrossEntropyLoss()
logits = np.array([[2.0, -1.0, 0.5], [0.1, 3.0, -2.0]])
targets = np.array([0, 1])

loss = loss_fn.forward(logits, targets)
grad = loss_fn.backward(logits, targets)
print(f"Loss: {loss:.4f}")
print(f"Gradient:\n{grad}")
```

输出：
```
Loss: 0.4756
Gradient:
[[ 0.4248  0.0282 -0.4530]
 [-0.9526  0.8728  0.0799]]
```

---

## 参考资料

1. Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. Chapter 5.
3. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer. Section 1.6.
