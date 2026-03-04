## Adam 优化器：一阶矩与二阶矩的联合自适应

Adam（Adaptive Moment Estimation）将梯度的一阶矩（动量）与二阶矩（梯度方差）联合估计，实现对每个参数的自适应学习率。数学上，第 t 步更新结合了累积梯度的方向信息 $m_t$ 与幅度缩放 $v_t$：$\theta_{t+1} = \theta_t - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$。

### 动机与问题

随机梯度下降（SGD）对所有参数使用统一学习率，难处理稀疏梯度的参数。高维空间中，不同特征的梯度尺度差异极大——某些维度梯度持续较大，某些维度长期为零或极小。统一学习率会导致：大梯度维度步长过大（震荡），小梯度维度收敛缓慢。

Momentum 通过指数移动平均（EMA）累积历史梯度方向，加速收敛并抑制震荡；RMSProp 用梯度平方的 EMA 归一化学习率，缓解不同梯度尺度问题。Adam 直接融合两者：一阶矩估计梯度均值（方向），二阶矩估计梯度方差（幅度）。

---

## 精确定义

设目标函数为 $f(\theta)$，$\theta \in \mathbb{R}^d$ 为参数向量，$g_t = \nabla_\theta f(\theta_{t-1})$ 为第 t 步梯度。

Adam 维护两个状态变量：

- 一阶矩估计（动量）：$m_t = \mathbb{E}[g_t]$，用指数移动平均近似
- 二阶矩估计（非中心方差）：$v_t = \mathbb{E}[g_t^2]$，同样用指数移动平均近似

更新规则（偏差修正前）：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\theta_t &= \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

其中：
- $\beta_1, \beta_2 \in [0, 1)$ 为衰减率，控制历史记忆长度
- $\alpha$ 为学习率（步长）
- $\epsilon$ 为数值稳定常数
- $g_t^2$ 表示逐元素平方

---

## 数学推导

### 指数移动平均的偏差

$m_0 = 0, v_0 = 0$ 的初始化导致前几步估计严重偏向零。展开 $m_t$ 的递归式：

$$
\begin{aligned}
m_1 &= \beta_1 m_0 + (1 - \beta_1) g_1 = (1 - \beta_1) g_1 \\
m_2 &= \beta_1 m_1 + (1 - \beta_1) g_2 = \beta_1(1 - \beta_1) g_1 + (1 - \beta_1) g_2 \\
m_3 &= \beta_1^2(1 - \beta_1) g_1 + \beta_1(1 - \beta_1) g_2 + (1 - \beta_1) g_3
\end{aligned}
$$

通项：$m_t = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i$

若 $g_i \approx g$（梯度稳定），则：

$$
\mathbb{E}[m_t] \approx g \cdot (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} = g \cdot (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = g \cdot (1 - \beta_1^t)
$$

$1 - \beta_1^t < 1$，即 $m_t$ 的期望是真实均值的 $(1 - \beta_1^t)$ 倍，存在系统性低估。

### 偏差修正

定义修正因子：$1 - \beta_1^t$。当 $t$ 较小时，$\beta_1^t \approx 1$，偏差显著；当 $t \to \infty$ 时，$\beta_1^t \to 0$，修正因子趋近 1。

修正后的一阶矩和二阶矩：

$$
\begin{aligned}
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t}
\end{aligned}
$$

最终更新规则：

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

这意味着除以 $\sqrt{\hat{v}_t}$ 实现了参数级的自适应学习率：梯度方差大的维度（方向不确定）步长缩小，梯度方差小的维度（方向稳定）步长放大。

---

## 代码实现

### 核心逻辑

```python
import numpy as np

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params  # 待优化参数列表（numpy 数组）
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # 状态初始化
        self.m = [np.zeros_like(p) for p in self.params]  # 一阶矩
        self.v = [np.zeros_like(p) for p in self.params]  # 二阶矩
        self.t = 0  # 时间步
    
    def step(self, grads):
        """参数更新，grads 是与 params 对应的梯度列表"""
        self.t += 1
        
        # 偏差修正因子
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # 更新一阶矩（指数移动平均）
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # 更新二阶矩（梯度平方的指数移动平均）
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            
            # 偏差修正
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2
            
            # 参数更新
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

### 超参数选择依据

| 超参数 | 默认值 | 作用 | 选择依据 |
|--------|--------|------|----------|
| $\beta_1$ | 0.9 | 一阶矩衰减率 | $1/(1-\beta_1)=10$，近似最近 10 步平均梯度 |
| $\beta_2$ | 0.999 | 二阶矩衰减率 | $1/(1-\beta_2)=1000$，较长的历史窗口估计方差 |
| $\epsilon$ | $10^{-8}$ | 数值稳定 | 防止除零，不影响更新幅度的数量级 |

$\beta_1=0.9$ 意味着有效窗口约 10 步（$0.9^{10} \approx 0.35$），捕捉短期梯度趋势；$\beta_2=0.999$ 意味着有效窗口约 1000 步（$0.999^{1000} \approx 0.37$），对长期方差进行平滑估计。两者差异使 Adam 区分短期方向（$m_t$）和长期稳定性（$v_t$）。

---

## AdamW：权重衰减解耦

原始 Adam 论文中的权重衰减实现方式存在问题：将 $L_2$ 正则化直接加入损失函数 $f(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|_2^2$，导致梯度 $g_t$ 中包含 $-\lambda \theta_{t-1}$ 项。Adam 的自适应缩放会进一步放大或衰减这一项，与权重衰减的初衷（独立缩放惩罚）背离。

AdamW 将权重衰减从梯度更新中解耦：

$$
\theta_t = (1 - \alpha \lambda) \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

权重衰减项 $(1 - \alpha \lambda)$ 直接作用于参数，不经过 Adam 的自适应缩放。这在 Transformer 等大模型中效果显著，因为自适应缩放会过度衰减小权重的衰减项。

```python
class AdamW(Adam):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, beta1, beta2, eps)
        self.weight_decay = weight_decay
    
    def step(self, grads):
        self.t += 1
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # 权重衰减（解耦，不经过自适应缩放）
            if self.weight_decay != 0:
                param *= (1 - self.lr * self.weight_decay)
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) ** 2
            
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

## 对比分析：Adam 与 SGD+Momentum

### 更新轨迹差异

SGD+Momentum 更新：$\theta_{t+1} = \theta_t - \alpha \hat{m}_t$

Adam 更新：$\theta_{t+1} = \theta_t - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

核心区别：Adam 对每个维度独立缩放步长。假设参数有两个维度：

- 维度 1：$g^{(1)}$ 始终为 $1$，$v^{(1)}$ 收敛到 $1$
- 维度 2：$g^{(2)}$ 在 $\{-10, 10\}$ 间震荡，$v^{(2)}$ 收敛到 $100$

SGD+Momentum 对两者使用相同步长 $\alpha$。Adam 对维度 1 的有效步长为 $\alpha / \sqrt{1} = \alpha$，对维度 2 的有效步长为 $\alpha / \sqrt{100} = 0.1\alpha$——大幅衰减梯度震荡维度。

这解释了 Adam 在稀疏梯度任务（如 NLP 推荐系统）中的优势：高频词特征（梯度大）被适度抑制，低频词特征（梯度小）被相对放大。

### 收敛特性

| 特性 | SGD+Momentum | Adam |
|------|--------------|------|
| 通用泛化能力 | 更好（更简单的优化路径） | 略差（可能陷入尖锐极小值） |
| 初始学习率敏感度 | 高 | 低（自适应缩放） |
| 稀疏梯度适应 | 差 | 优秀 |
| 超参数调优需求 | 高 | 低 |
| 内存占用 | $O(d)$ | $O(2d)$（需存储 $m_t, v_t$） |

---

## 局限性与工程陷阱

### Adam 不总是最优

部分场景下 SGD+Momentum 泛化优于 Adam，尤其在大规模计算机视觉任务（ImageNet 训练 ResNet）。原因：Adam 的自适应缩放使优化路径更"聪明"，可能过度拟合训练数据，泛化到未见数据时性能下降。Sharpness-Aware Minimization（SAM）等技巧可缓解这一问题。

### 学习率预热

Adam 在训练初期（$t$ 很小时），$\sqrt{\hat{v}_t}$ 的估计不稳定，$v_t$ 从零开始，导致初期步长过大。常见做法是学习率预热：前 N 步线性增加学习率至目标值。

```python
def get_lr(warmup_steps, target_lr, step):
    if step < warmup_steps:
        return target_lr * step / warmup_steps
    return target_lr
```

### 数值稳定性问题

$\epsilon=10^{-8}$ 在 float16 精度下可能过小，导致 `sqrt(v_hat) + eps ≈ eps`。FP16 训练时应增大 $\epsilon$ 至 $10^{-4}$ 或使用 master weights（FP32 复制参数用于更新）。

### 内存开销

$m_t$ 和 $v_t$ 各占与参数相同的空间。对于 7B 参数模型（FP16），参数占用 14GB，Adam 状态额外占用 28GB——总计 42GB。大模型训练常采用 ZeRO 优化器状态分片或 AdamW 8bit 变体减少显存占用。

---

## 适用边界

| 场景 | 推荐优化器 | 原因 |
|------|-----------|------|
| NLP 预训练（GPT/BERT） | AdamW | 稀疏词嵌入、大规模超参数搜索成本高 |
| 计算机视觉（ResNet/ViT） | SGD+Momentum + cosine annealing | 泛化性能更优，实验验证充分 |
| 推荐/点击率预估 | Adam | 稀疏特征占比高，自适应缩放至关重要 |
| 小数据集精调 | SGD+Momentum | 泛化优势更明显 |
| 8bit 量化训练 | AdamW-8bit | 减少内存开销 |

---

## 参考资料

1. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980
2. Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. arXiv:1711.05101
3. Wilson, A. C., et al. (2017). The Marginal Value of Adaptive Gradient Methods in Machine Learning. NeurIPS 2017
