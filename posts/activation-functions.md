## 激活函数定义

激活函数 f: ℝ → ℝ 引入非线性，使神经网络能拟合任意函数。没有激活函数，多层网络等价于单层线性变换。数学上，网络表达能力取决于激活函数的非线性性质。

---

## Sigmoid: σ(x) = 1/(1+e^{-x})

**性质**：输出范围 (0, 1)，单调递增，关于 (0, 0.5) 中心对称。

导数：

$$
\frac{d\sigma}{dx} = \sigma(x)(1-\sigma(x))
$$

最大值在 x=0 处取得，σ'(0) = 0.25。当 |x| > 5 时，|σ'(x)| < 0.0067，进入饱和区。

**梯度消失分析**：t 层网络中梯度传递因子为 σ'(x)^t。假设输入分布标准差为 1，约 68% 的激活值落在 [-1,1]，对应的导数范围 [0.20, 0.25]。保守估计平均导数 0.22：

| 层数 | 梯度衰减因子 | 剩余梯度比例 |
|------|-------------|-------------|
| 5    | 0.22^5      | 0.515%      |
| 10   | 0.22^10     | 0.00265%    |
| 20   | 0.22^20     | 7.0×10⁻⁹    |

10 层后梯度缩小约 3.8 万倍（实际更严重，因部分单元进入饱和区）。

**工程问题**：
- 输出非零中心：各层输入分布随前一层参数漂移，训练不稳定
- 计算需指数运算，GPU 友好度差

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # 防止溢出

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 梯度消失演示
layers = 10
grad_factor = 0.22  # 假设平均梯度
print(f"{layers}层后梯度衰减: {grad_factor**layers:.2e}")
```

**适用场景**：二分类输出层，隐藏层已被淘汰。

---

## ReLU: f(x) = max(0, x)

**性质**：正区线性（f(x)=x，f'(x)=1），负区恒零（f(x)=0，f'(x)=0）。非零中心，非有界，计算高效。

**梯度直通特性**：正区梯度无衰减传递，缓解了深层网络训练难题。理论保证：若网络权重初始化合理，至少 50% 的单元在训练初期处于激活状态（假设输入对称分布）。

**稀疏激活**：负区输出为零，产生稀疏表示。实际训练中，VGG-16 约 70%-80% 的 ReLU 单元处于非激活状态，减少计算量。

**Dead ReLU 问题**：当单元进入负区且其输入权重与偏置使所有训练样本都激活负区，该单元永久死亡。数学条件：

$$
\forall x \in \mathcal{D}, \quad w^\top x + b < 0
$$

其中 𝓓 为训练数据分布。一旦发生，反向传播无法更新该单元权重。

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Dead ReLU 模拟
def simulate_dead_relu():
    n_samples = 1000
    X = np.random.randn(n_samples, 10)  # 标准正态输入
    w = np.random.randn(10) * 0.1      # 小权重
    b = -5.0                            # 负偏置
    
    # 所有样本都落在负区
    activations = relu(X @ w + b)
    dead_ratio = np.mean(activations == 0)
    return dead_ratio

print(f"Dead ReLU 比例: {simulate_dead_relu():.2%}")
```

**工程缓解方案**：
- Leaky ReLU: f(x) = max(αx, x)，α ∈ (0,1)，常用 α=0.01
- PReLU: α 作为可学习参数
- 合理初始化（He 初始化）：权重从 𝒩(0, 2/n_in) 采样，n_in 为输入维度

---

## GELU: f(x) = x·Φ(x)

Φ(x) 为标准正态累积分布函数：

$$
\Phi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x} e^{-t^2/2}dt
$$

**近似表达式**（数值误差 < 0.001）：

$$
\text{GELU}(x) \approx x \cdot \frac{1}{2}\left[1 + \text{tanh}\left(\sqrt{\frac{2}{\pi}}\left(x + \frac{0.044715x^3}{1}\right)\right)\right]
$$

导数：

$$
\text{GELU}'(x) = \Phi(x) + x\phi(x) = \Phi(x) + \frac{x}{\sqrt{2\pi}}e^{-x^2/2}
$$

其中 φ(x) 为标准正态 PDF。

**与 ReLU 对比**：GELU 是 ReLU 的平滑近似。当 x → ∞，GELU(x) ≈ x；当 x → -∞，GELU(x) ≈ 0。关键区别：GELU 是概率门控——以概率 Φ(x) 让输入通过，而非确定性截断。

**数学直观**：假设激活前的隐藏层输出 h 服从正态分布，则 h 的期望为 𝔼[h·1_{h>t}]，其中 1_{h>t} 为示性函数。GELU 是这个期望的近似。

| 激活函数 | 非光滑性 | 梯度消失 | 稀疏性 | 计算复杂度 |
|---------|---------|---------|-------|-----------|
| Sigmoid  | 无       | 严重    | 无    | 高（exp） |
| ReLU     | x=0     | 轻微    | 强    | 低        |
| GELU     | 无       | 轻微    | 弱    | 中（tanh） |

```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    tanh_arg = np.sqrt(2/np.pi) * (x + 0.044715 * x**3)
    sech_sq = 1 - np.tanh(tanh_arg)**2
    return 0.5 * (1 + np.tanh(tanh_arg)) + 0.5 * x * sech_sq * np.sqrt(2/np.pi) * (1 + 0.134145 * x**2)

# 曲线对比
import matplotlib.pyplot as plt
x = np.linspace(-4, 4, 1000)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, gelu(x), label='GELU')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, sigmoid_derivative(x), label="Sigmoid'")
plt.plot(x, relu_derivative(x), label="ReLU'")
plt.plot(x, gelu_derivative(x), label="GELU'")
plt.legend()
```

**默认选择**：BERT、GPT-2/GPT-3 均使用 GELU 作为隐藏层激活。NLP 任务表现优于 ReLU，平滑性有助于优化器收敛。

---

## SwiGLU: LLaMA 系列的 FFN 激活

SwiGLU 是激活函数与架构的混合设计，应用于前馈神经网络（FFN）：

$$
\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV)
$$

其中 Swish(x) = x·σ(βx)，β 为可调参数（常用 β=1），σ 为 sigmoid。⊙ 表示逐元素乘法。

**与标准 FFN 对比**：标准 FFN 为：

$$
\text{FFN}(x) = \text{GELU}(xW_1)W_2
$$

SwiGLU 引入线性投影 xV 和逐元素相乘，增加模型容量。

**LLaMA 实现**：FFN 隐藏维度 d_ff = 8/3 · d_model（非标准的 4·d_model）：

```python
def swiglu(x, W_gate, W_up, W_down):
    """
    x: [batch, seq_len, d_model]
    W_gate, W_up, W_down: [d_model, d_ff]
    """
    gate = torch.sigmoid(x @ W_gate)  # 门控
    up = x @ W_up                      # 上投影
    return (gate * up) @ W_down        # 下投影

# PyTorch nn.Module 形式
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or (8 * d_model // 3)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

**Swish**：σ(x) = x·sigmoid(x)，当 β=1 时称为 SiLU。性质：

- 非单调：在 x < -1.27 处略低于 ReLU，在 x > -1.27 处高于 ReLU
- 平滑导数：Swish'(x) = sigmoid(x) + x·sigmoid(x)·(1-sigmoid(x))

**性能对比**（LLaMA 论文）：

| FFN 变体       | Perplexity (PTB) | 参数量 |
|---------------|------------------|-------|
| ReLU-FFN      | 15.84            | 100%  |
| GELU-FFN      | 15.52            | 100%  |
| SwiGLU-FFN    | 15.01            | 150%  |

SwiGLU 以 1.5× 参数量换取显著性能提升。

---

## 激活函数选择决策树

```
输出层（分类）
├── 二分类 → Sigmoid
└── 多分类 → Softmax（非本主题）

隐藏层（现代 Transformer）
├── 优先 GELU（BERT/GPT 系列）
└── 可选 SwiGLU（LLaMA 系列，需增加参数量）

隐藏层（CNN / 轻量级模型）
├── ReLU（默认，速度最优）
├── LeakyReLU/PReLU（Dead ReLU 严重时）
└── GELU（追求精度时）

历史遗留/教学
└── Sigmoid/Tanh（仅作对比）
```

**性能 trade-off 总结**：

| 维度       | Sigmoid | ReLU   | GELU   | SwiGLU |
|-----------|---------|--------|--------|--------|
| 梯度消失   | 严重    | 轻微   | 轻微   | 轻微   |
| 计算效率   | 低      | 最高   | 中     | 低     |
| 表现力     | 低      | 中     | 高     | 最高   |
| 参数开销   | 无      | 无     | 无     | +50%   |
| 工业采用   | 输出层  | CNN    | BERT   | LLaMA  |

---

## 参考文献

1. Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv:1606.08415.
2. Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for Activation Functions. arXiv:1710.05941.
3. Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.
4. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971.
