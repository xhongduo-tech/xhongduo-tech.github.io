## 感知机到多层感知机：深度学习的起点

感知机是神经网络的最早形态，由 Frank Rosenblatt 于 1958 年提出。给定输入向量 $x \in \mathbb{R}^d$，感知机的输出为 $y = \text{sign}(w \cdot x + b)$，其中 $w \in \mathbb{R}^d$ 为权重向量，$b \in \mathbb{R}$ 为偏置。几何上，$\{x \mid w \cdot x + b = 0\}$ 定义了一个超平面，将输入空间分割为两个半空间：决策边界为 $w \cdot x + b > 0$ 时输出 $+1$，否则输出 $-1$。

感知机的学习目标是最小化分类错误，或等价地找到一个能正确分类所有训练样本的超平面。Rosenblatt 证明了当训练数据线性可分时，感知机学习算法会在有限步内收敛——Perceptron Convergence Theorem。该定理指出，若存在单位长度的 $w^*$ 使得对所有样本 $i$ 都有 $y_i (w^* \cdot x_i) \ge \gamma$（$\gamma > 0$ 为间隔），则感知机算法的更新次数上界为 $(R/\gamma)^2$，其中 $R = \max_i \|x_i\|$。

---

线性可分是感知机工作的充分条件，却非充分必要。XOR 问题是展示单层感知机局限性的经典反例：输入 $(x_1, x_2) \in \{0, 1\}^2$，输出 $y = x_1 \oplus x_2$。四种样本为 $(0,0) \to 0$、$(0,1) \to 1$、$(1,0) \to 1$、$(1,1) \to 0$。不存在线性分类器能同时满足 $w_1 x_1 + w_2 x_2 + b$ 的符号要求——$(0,0)$ 与 $(1,1)$ 需要相同符号，而 $(0,1)$ 与 $(1,0)$ 需要相反符号，这在平面上不可能同时成立。

证明：设存在 $w_1, w_2, b$ 满足 XOR 要求。则：
- $(0,0) \to 0$: $b \le 0$
- $(1,1) \to 0$: $w_1 + w_2 + b \le 0$
- $(0,1) \to 1$: $w_2 + b > 0$
- $(1,0) \to 1$: $w_1 + b > 0$

从 $w_1 + b > 0$ 和 $w_2 + b > 0$ 相加得 $w_1 + w_2 + 2b > 0$，与 $w_1 + w_2 + b \le 0$ 矛盾。此矛盾证明单层感知机无法解决 XOR 问题。

---

多层感知机通过引入隐藏层和非线性激活函数突破了线性边界。最基础的两层 MLP 结构为：输入层 $x \in \mathbb{R}^d$，隐藏层 $h = \sigma(W_1 x + b_1)$，输出层 $y = W_2 h + b_2$（回归）或 $y = \text{softmax}(W_2 h + b_2)$（分类）。这里 $\sigma(\cdot)$ 是逐元素非线性激活，如 ReLU $\sigma(z) = \max(0, z)$ 或 Sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$。

隐藏层的意义在于将原始输入空间映射到一个可分的特征空间。以 XOR 为例，设隐藏层有 2 个神经元，使用 ReLU 激活，令 $W_1 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$，$b_1 = \begin{bmatrix} -0.5 \\ 0.5 \end{bmatrix}$。则四种输入的隐藏层输出为：
- $(0,0) \to h = [0, 0.5]$
- $(0,1) \to h = [0.5, 0]$
- $(1,0) \to h = [0.5, 1]$
- $(1,1) \to h = [1.5, 0]$

此时 $h$ 空间中两类样本线性可分，单层输出层即可完成分类。

---

1989 年的 Universal Approximation Theorem 为 MLP 的表达能力提供了理论保障：含一个隐藏层的前馈网络，若隐藏层神经元数量足够且激活函数满足特定条件（如 Sigmoid），则可一致逼近任意定义在紧集上的连续函数，逼近误差可任意小。

定理表述（Cybenko, 1989）：设 $\sigma: \mathbb{R} \to \mathbb{R}$ 是连续 sigmoidal 激活函数，$I = [0,1]^d$，$F(I)$ 为 $I$ 上所有连续函数构成的空间。对任意 $f \in F(I)$ 和任意 $\epsilon > 0$，存在整数 $n$、权重向量 $w_i \in \mathbb{R}^d$、偏置 $b_i \in \mathbb{R}$、输出权重 $v_i \in \mathbb{R}$，使得
$$
\left| f(x) - \sum_{i=1}^n v_i \sigma(w_i \cdot x + b_i) \right| < \epsilon, \quad \forall x \in I
$$

该定理只保证存在性，未给出 $n$ 的上界——实际中所需神经元数量可能呈指数级增长，这是 MLP 的主要工程瓶颈之一。

---

前向传播的矩阵形式使批量计算高效。设输入矩阵为 $X \in \mathbb{R}^{B \times d}$（$B$ 为 batch size），则：

$$
Z^{(1)} = X W_1^{(d \times h)} + \mathbf{1} b_1^{\top}, \quad Z^{(1)} \in \mathbb{R}^{B \times h}
$$

$$
H = \sigma(Z^{(1)}), \quad H \in \mathbb{R}^{B \times h}
$$

$$
Z^{(2)} = H W_2^{(h \times o)} + \mathbf{1} b_2^{\top}, \quad Z^{(2)} \in \mathbb{R}^{B \times o}
$$

$$
\hat{Y} = \text{softmax}(Z^{(2)}), \quad \hat{Y} \in \mathbb{R}^{B \times o}
$$

其中 $\mathbf{1} \in \mathbb{R}^B$ 为全 1 向量，$o$ 为输出类别数。损失函数采用交叉熵：
$$
L = -\frac{1}{B} \sum_{i=1}^B \sum_{j=1}^o Y_{ij} \log \hat{Y}_{ij}
$$

反向传播通过链式法则计算梯度：
$$
\frac{\partial L}{\partial Z^{(2)}} = \hat{Y} - Y
$$

$$
\frac{\partial L}{\partial W_2} = H^{\top} \frac{\partial L}{\partial Z^{(2)}}, \quad \frac{\partial L}{\partial b_2} = \mathbf{1}^{\top} \frac{\partial L}{\partial Z^{(2)}}
$$

$$
\frac{\partial L}{\partial H} = \frac{\partial L}{\partial Z^{(2)}} W_2^{\top}
$$

$$
\frac{\partial L}{\partial Z^{(1)}} = \frac{\partial L}{\partial H} \odot \sigma'(Z^{(1)})
$$

$$
\frac{\partial L}{\partial W_1} = X^{\top} \frac{\partial L}{\partial Z^{(1)}}, \quad \frac{\partial L}{\partial b_1} = \mathbf{1}^{\top} \frac{\partial L}{\partial Z^{(1)}}
$$

$\odot$ 表示逐元素乘法，$\sigma'$ 为激活函数导数。

---

用 NumPy 手工实现解决 XOR 问题的两层 MLP。

```python
import numpy as np

# XOR 数据: [x1, x2] -> y
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 网络结构: 2 -> 4 -> 1
d_in, d_hidden, d_out = 2, 4, 1
np.random.seed(42)

# Xavier 初始化
W1 = np.random.randn(d_in, d_hidden) * np.sqrt(2.0 / (d_in + d_hidden))
b1 = np.zeros((1, d_hidden))
W2 = np.random.randn(d_hidden, d_out) * np.sqrt(2.0 / (d_hidden + d_out))
b2 = np.zeros((1, d_out))

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    # 前向传播
    z1 = X @ W1 + b1  # (4, 4)
    a1 = sigmoid(z1)  # (4, 4)
    z2 = a1 @ W2 + b2  # (4, 1)
    a2 = sigmoid(z2)  # (4, 1)
    
    # 损失（MSE 用于回归输出）
    loss = np.mean((a2 - y) ** 2)
    
    # 反向传播
    dz2 = (a2 - y) * sigmoid_derivative(z2)  # (4, 1)
    dW2 = a1.T @ dz2  # (4, 1)
    db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, 1)
    
    dz1 = (dz2 @ W2.T) * sigmoid_derivative(z1)  # (4, 4)
    dW1 = X.T @ dz1  # (2, 4)
    db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, 4)
    
    # 参数更新
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# 测试输出
print("\nFinal predictions:")
predictions = sigmoid((sigmoid(X @ W1 + b1) @ W2 + b2))
for i in range(4):
    print(f"XOR({int(X[i,0])}, {int(X[i,1])}) = {predictions[i,0]:.4f}")
```

输出：
```
Epoch 0, Loss: 0.260899
Epoch 2000, Loss: 0.246276
Epoch 4000, Loss: 0.047640
Epoch 6000, Loss: 0.014414
Epoch 8000, Loss: 0.007588

Final predictions:
XOR(0, 0) = 0.0417
XOR(0, 1) = 0.9541
XOR(1, 0) = 0.9552
XOR(1, 1) = 0.0488
```

权重矩阵维度：$W_1 \in \mathbb{R}^{2 \times 4}$，$b_1 \in \mathbb{R}^{1 \times 4}$，$W_2 \in \mathbb{R}^{4 \times 1}$，$b_2 \in \mathbb{R}^{1 \times 1}$。

---

**工程细节与 trade-off**

| 方面 | 说明 | 影响 |
|------|------|------|
| 初始化 | Xavier/Glorot 适用于 tanh，He 适用于 ReLU | 不当初始化导致梯度消失/爆炸 |
| 激活选择 | ReLU 避免梯度消失但有死亡神经元问题 | Sigmoid 饱和时梯度接近零 |
| 层数深度 | 超过 3 层可能需要残差连接 | 深层网络退化现象 |
| 过拟合 | 小数据集上易发生 | 需 Dropout、L2 正则化 |
| 优化器 | SGD 收敛慢，Adam 适应性好 | Adam 可能泛化略差于 SGD |

**局限性**：MLP 无法捕捉空间局部性和平移等变性——图像相邻像素的相关性、序列的时间顺序都需要专门结构（CNN、RNN/Transformer）处理。此外，MLP 的参数量随输入维度线性增长，高维输入下不可行。

**参考资料**：
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain.
- Minsky, M., & Papert, S. (1969). Perceptrons: An introduction to computational geometry.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function.
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
