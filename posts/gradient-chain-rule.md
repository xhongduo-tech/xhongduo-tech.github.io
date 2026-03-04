## 梯度与链式法则：反向传播的数学基础

梯度是多元函数变化最快的方向向量，其模长表示最大变化率。对于函数 $f: \mathbb{R}^n \to \mathbb{R}$，梯度 $\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n}\right]^T$ 在点 $\mathbf{x}$ 处指向 $f$ 增长最快的方向，反向则指向最快下降方向——这是梯度下降算法的几何基础。

---

## 从导数到梯度

单变量导数 $f'(x)$ 描述函数在某点的瞬时变化率。多元函数中，偏导数 $\frac{\partial f}{\partial x_i}$ 固定其他变量，仅沿坐标轴 $x_i$ 方向的变化率。方向导数 $\nabla_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u}$ 给出沿单位方向 $\mathbf{u}$ 的变化率，其中 $\mathbf{u}$ 为单位向量。

$$\nabla_{\mathbf{u}} f(\mathbf{x}) = \sum_{i=1}^n \frac{\partial f}{\partial x_i} u_i = \nabla f(\mathbf{x})^T \mathbf{u}$$

由 Cauchy-Schwarz 不等式 $|\nabla f \cdot \mathbf{u}| \leq \|\nabla f\| \cdot \|\mathbf{u}\| = \|\nabla f\|$，当 $\mathbf{u}$ 与 $\nabla f$ 同向时取等号。这意味着：梯度方向上方向导数最大，梯度模长即最大变化率。梯度下降 $x_{t+1} = x_t - \eta \nabla f(x_t)$ 正是沿负梯度方向移动，以最快速度减小目标函数。

---

## 链式法则

单变量链式法则 $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$ 描述复合函数的导数。多元情况下，设 $z = f(\mathbf{y})$ 且 $\mathbf{y} = g(\mathbf{x})$，其中 $\mathbf{y} \in \mathbb{R}^m, \mathbf{x} \in \mathbb{R}^n$，则：

$$\frac{\partial z}{\partial x_j} = \sum_{i=1}^m \frac{\partial z}{\partial y_i} \frac{\partial y_i}{\partial x_j}$$

矩阵形式写作雅可比矩阵的乘积：

$$\frac{\partial z}{\partial \mathbf{x}} = \frac{\partial z}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{1 \times n}$$

其中 $\frac{\partial z}{\partial \mathbf{y}} \in \mathbb{R}^{1 \times m}$ 是行向量，$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}$ 是雅可比矩阵。这意味着：梯度通过链式法则逐层向后传递，每经过一个函数变换，梯度右乘该变换的雅可比矩阵。

---

## 计算图与反向传播

计算图中每个节点表示一个张量，每条边表示一次函数变换。前向传播计算各节点数值，反向传播则从输出节点开始，沿边逆向传播梯度。每条边上的梯度传递对应一次链式法则应用：输出梯度乘以该边变换的雅可比矩阵，得到上游梯度。

| 前向操作 | 反向梯度公式 | 雅可比维度 |
|---------|-------------|-----------|
| $y = Wx + b$ | $\frac{\partial L}{\partial x} = W^T \frac{\partial L}{\partial y}$ | $d_{in} \times d_{out}$ |
| $y = \sigma(x)$ | $\frac{\partial L}{\partial x} = \sigma(x)(1-\sigma(x)) \odot \frac{\partial L}{\partial y}$ | 对角矩阵 |
| $y = \max(0, x)$ | $\frac{\partial L}{\partial x} = \mathbb{I}(x > 0) \odot \frac{\partial L}{\partial y}$ | 对角矩阵 |

$\odot$ 表示逐元素乘积。反向传播的核心优势在于：避免重复计算中间导数，所有中间梯度只需计算一次，复杂度与前向传播同阶 $O(V+E)$，其中 $V$ 为节点数，$E$ 为边数。

---

## 三层网络手推实例

考虑三层全连接网络，输入层 $h^{(0)} \in \mathbb{R}^2$，隐层 $h^{(1)}, h^{(2)} \in \mathbb{R}^3$，输出层 $h^{(3)} \in \mathbb{R}$。激活函数为 ReLU，损失函数为 MSE $L = \frac{1}{2}(h^{(3)} - y)^2$。

### 网络参数与输入

设初始参数（简化示例）：

```
W^(1) = [[1, -1],   b^(1) = [0, 0, 0]^T
        [2,  0],
        [-1, 1]]
W^(2) = [[1, 2, 0],  b^(2) = [0, 0, 0]^T
        [0, -1, 1],
        [1, 0, -1]]
W^(3) = [[1, -1, 1]] b^(3) = [0]
```

输入 $x = [1, 2]^T$，标签 $y = 1$。

### 前向传播

**第一层**：$a^{(1)} = W^{(1)}x + b^{(1)}$

$$a^{(1)} = \begin{bmatrix} 1 & -1 \\ 2 & 0 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} -1 \\ 2 \\ 1 \end{bmatrix}$$

$$h^{(1)} = \text{ReLU}(a^{(1)}) = [\max(0,-1), \max(0,2), \max(0,1)]^T = [0, 2, 1]^T$$

**第二层**：$a^{(2)} = W^{(2)}h^{(1)} + b^{(2)}$

$$a^{(2)} = \begin{bmatrix} 1 & 2 & 0 \\ 0 & -1 & 1 \\ 1 & 0 & -1 \end{bmatrix} \begin{bmatrix} 0 \\ 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 4 \\ -1 \\ -1 \end{bmatrix}$$

$$h^{(2)} = \text{ReLU}(a^{(2)}) = [4, 0, 0]^T$$

**第三层**：$a^{(3)} = W^{(3)}h^{(2)} + b^{(3)}$

$$a^{(3)} = [1, -1, 1] \begin{bmatrix} 4 \\ 0 \\ 0 \end{bmatrix} = 4$$

$$\hat{y} = h^{(3)} = a^{(3)} = 4$$

**损失**：$L = \frac{1}{2}(4 - 1)^2 = 4.5$

### 反向传播

**输出层梯度**：$\frac{\partial L}{\partial h^{(3)}} = \frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 4 - 1 = 3$

**第三层参数梯度**：

$$\frac{\partial L}{\partial W^{(3)}} = \frac{\partial L}{\partial h^{(3)}} \cdot (h^{(2)})^T = 3 \cdot [4, 0, 0] = [12, 0, 0]$$

$$\frac{\partial L}{\partial b^{(3)}} = \frac{\partial L}{\partial h^{(3)}} = 3$$

**向第二层传播**：

$$\frac{\partial L}{\partial h^{(2)}} = (W^{(3)})^T \frac{\partial L}{\partial h^{(3)}} = [1, -1, 1]^T \cdot 3 = [3, -3, 3]^T$$

经过 ReLU 导数（$h_2^{(2)} > 0$ 时为 1，否则为 0）：

$$\frac{\partial L}{\partial a^{(2)}} = [3, 0, 0]^T$$

**第二层参数梯度**：

$$\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial a^{(2)}} \cdot (h^{(1)})^T = [3, 0, 0]^T \cdot [0, 2, 1] = \begin{bmatrix} 0 & 6 & 3 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

$$\frac{\partial L}{\partial b^{(2)}} = \frac{\partial L}{\partial a^{(2)}} = [3, 0, 0]^T$$

**向第一层传播**：

$$\frac{\partial L}{\partial h^{(1)}} = (W^{(2)})^T \frac{\partial L}{\partial a^{(2)}} = \begin{bmatrix} 1 & 0 & 1 \\ 2 & -1 & 0 \\ 0 & 1 & -1 \end{bmatrix} \begin{bmatrix} 3 \\ 0 \\ 0 \end{bmatrix} = [3, 6, 0]^T$$

经过 ReLU 导数（$h_1^{(1)}=0$ 导数为 0）：

$$\frac{\partial L}{\partial a^{(1)}} = [0, 6, 0]^T$$

**第一层参数梯度**：

$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial a^{(1)}} \cdot x^T = [0, 6, 0]^T \cdot [1, 2] = \begin{bmatrix} 0 & 0 \\ 6 & 12 \\ 0 & 0 \end{bmatrix}$$

$$\frac{\partial L}{\partial b^{(1)}} = \frac{\partial L}{\partial a^{(1)}} = [0, 6, 0]^T$$

---

## Python 实现

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# 初始化参数
W1 = np.array([[1, -1], [2, 0], [-1, 1]])
b1 = np.zeros(3)
W2 = np.array([[1, 2, 0], [0, -1, 1], [1, 0, -1]])
b2 = np.zeros(3)
W3 = np.array([[1, -1, 1]])
b3 = np.array([0.0])

x = np.array([1.0, 2.0])
y = 1.0

# 前向传播
a1 = W1 @ x + b1
h1 = relu(a1)          # [0, 2, 1]
a2 = W2 @ h1 + b2
h2 = relu(a2)          # [4, 0, 0]
a3 = W3 @ h2 + b3
h3 = a3                 # 4

loss = 0.5 * (h3 - y)**2  # 4.5

# 反向传播
dL_dh3 = h3 - y         # 3
dL_dW3 = dL_dh3 * h2.reshape(1, -1)  # [[12, 0, 0]]
dL_db3 = dL_dh3         # 3

dL_dh2 = W3.T @ np.array([[dL_dh3]]).reshape(-1)  # [3, -3, 3]
dL_da2 = relu_derivative(a2) * dL_dh2  # [3, 0, 0]

dL_dW2 = np.outer(dL_da2, h1)
dL_db2 = dL_da2

dL_dh1 = W2.T @ dL_da2   # [3, 6, 0]
dL_da1 = relu_derivative(a1) * dL_dh1  # [0, 6, 0]

dL_dW1 = np.outer(dL_da1, x)
dL_db1 = dL_da1

print("Loss:", loss)
print("dL/dW1:\n", dL_dW1)
```

输出：

```
Loss: 4.5
dL/dW1:
 [[ 0.  0.]
 [ 6. 12.]
 [ 0.  0.]]
```

---

## 工程细节与 Trade-off

**梯度消失与爆炸**：深层网络中，梯度乘积 $\prod_{l=1}^L \frac{\partial h^{(l)}}{\partial h^{(l-1)}}$ 可能指数衰减或增长。ReLU 缓解梯度消失，但不能解决梯度爆炸。实际中配合 Batch Normalization 和梯度裁剪使用。

**内存消耗**：反向传播需要缓存前向计算的所有中间激活值，显存消耗与层数和批量大小成正比。梯度检查点（Gradient Checkpointing）牺牲计算时间换取空间：仅缓存部分中间值，需要时重新计算。

**数值稳定性**：Sigmoid/Tanh 饱和区导数接近 0 导致梯度消失。除零问题通过添加 $\epsilon$ 避免softmax logit差分过大。混合精度训练下，FP32 梯度累加器防止 FP16 下溢。

**自动微分框架**：PyTorch 的动态计算图允许控制流分支，TensorFlow 的静态图可提前优化。两者本质相同：构建计算图，通过反向模式自动微分高效计算雅可比-向量积。

---

## 局限性与适用边界

链式法则要求函数处处可微。ReLU 在 0 点不可导，工程上定义导数为 0 或 1（subgradient）。不可导点测度为零，不影响梯度下降收敛。

对于不可导激活（如 Leaky ReLU）或非光滑损失（如 Hinge Loss），次梯度仍可用于优化。深度学习中常用非凸损失，链式法则给出局部梯度信息，全局最优依赖初始化和优化算法。

反向传播的时间复杂度为 $O(V)$，但无法并行化不同层的梯度计算——这是串行的本质限制。反向传播变体（如并行反向传播）尝试在分布式环境下加速，但增加通信开销。

---

## 参考资料

1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. Chapter 6: Deep Feedforward Networks.

3. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: a survey. Journal of Machine Learning Research, 18(153), 1-43.
