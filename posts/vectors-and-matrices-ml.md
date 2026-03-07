## 核心结论

神经网络的每一层计算可以精确描述为：

$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}, \quad \mathbf{a} = \sigma(\mathbf{z})$$

其中 $\mathbf{W} \in \mathbb{R}^{m \times n}$ 是线性变换，$\sigma$ 是逐元素非线性激活。三条不可违背的规则：

1. **维度规则**：$\mathbf{A} \in \mathbb{R}^{m \times n}$，$\mathbf{B} \in \mathbb{R}^{n \times p}$，则 $\mathbf{AB} \in \mathbb{R}^{m \times p}$，内维必须匹配
2. **梯度规则**：前向用 $\mathbf{W}$，反向用 $\mathbf{W}^T$；梯度 $\partial L / \partial \mathbf{W} = \delta \mathbf{x}^T$ 是外积
3. **并行规则**：GPU 加速的本质是 GEMM（通用矩阵乘法），批量输入 $\mathbf{X} \in \mathbb{R}^{n \times B}$ 才能充分利用张量核心

---

## 问题定义与边界

### 符号约定

全文统一以下记号：

| 符号 | 含义 | 维度 |
|------|------|------|
| $\mathbf{x}$ | 输入列向量 | $\mathbb{R}^{n \times 1}$ |
| $\mathbf{W}$ | 权重矩阵 | $\mathbb{R}^{m \times n}$ |
| $\mathbf{b}$ | 偏置列向量 | $\mathbb{R}^{m \times 1}$ |
| $\mathbf{z}$ | 预激活（线性输出） | $\mathbb{R}^{m \times 1}$ |
| $\mathbf{a}$ | 激活后输出 | $\mathbb{R}^{m \times 1}$ |
| $L$ | 标量损失 | $\mathbb{R}$ |
| $\delta$ | 损失对 $\mathbf{z}$ 的梯度 $\partial L/\partial \mathbf{z}$ | $\mathbb{R}^{m \times 1}$ |

**存储约定**：NumPy/PyTorch 默认行优先（C-contiguous），矩阵 $\mathbf{W}[i,j]$ 存储在地址 $\text{base} + i \cdot n + j$。cuBLAS 使用列优先，通过传入转置标志绕过显式转置开销。

### Running Example 定义

贯穿全文的 2 层全连接网络：

$$\mathbf{x} = \begin{bmatrix}1 \\ 0.5\end{bmatrix}, \quad \mathbf{W}_1 = \begin{bmatrix}0.2 & 0.3 \\ 0.1 & -0.5\end{bmatrix}, \quad \mathbf{b}_1 = \begin{bmatrix}0.1 \\ -0.2\end{bmatrix}$$

$$\mathbf{W}_2 = \begin{bmatrix}0.4 & -0.3\end{bmatrix}, \quad b_2 = 0.05, \quad y = 1$$

维度链：$\mathbb{R}^{2} \xrightarrow{\mathbf{W}_1 \in \mathbb{R}^{2\times2}} \mathbb{R}^{2} \xrightarrow{\text{ReLU}} \mathbb{R}^{2} \xrightarrow{\mathbf{W}_2 \in \mathbb{R}^{1\times2}} \mathbb{R}^{1} \xrightarrow{\text{sigmoid}} \mathbb{R}^{1}$

### 本文边界

- 仅讨论全连接层；卷积层可视为带约束的稀疏矩阵乘法，不在此展开
- 矩阵乘法复杂度 $O(mnp)$ 是精确表示；Strassen 算法 $O(n^{2.807})$ 在深度学习实践中不常用
- 逆矩阵在神经网络中几乎不出现（权重矩阵通常非方阵），仅在分析时提及

---

## 核心机制���推导

### 3.1 向量内积与余弦相似度

向量 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ 的内积：

$$\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^{n} u_i v_i = \mathbf{u}^T \mathbf{v}$$

L2 范数与余弦相似度：

$$\|\mathbf{v}\|_2 = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}, \qquad \cos\theta = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \in [-1, 1]$$

**矩阵乘法的内积视角**：$\mathbf{C} = \mathbf{AB}$ 的第 $(i,j)$ 元素是 $\mathbf{A}$ 第 $i$ 行与 $\mathbf{B}$ 第 $j$ 列的内积：

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

因此 $\mathbf{z} = \mathbf{W}\mathbf{x}$ 的第 $i$ 个元素是权重矩阵第 $i$ 行（即第 $i$ 个神经元的权重向量）与输入 $\mathbf{x}$ 的内积，衡量输入在该权重方向上的投影。

### 3.2 矩阵乘法的前向传播

以 Running Example 计算第一层：

$$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 = \begin{bmatrix}0.2 \cdot 1 + 0.3 \cdot 0.5 \\ 0.1 \cdot 1 + (-0.5) \cdot 0.5\end{bmatrix} + \begin{bmatrix}0.1 \\ -0.2\end{bmatrix} = \begin{bmatrix}0.35 \\ -0.05\end{bmatrix}$$

ReLU 激活 $\sigma(z) = \max(0, z)$：

$$\mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1) = \begin{bmatrix}0.35 \\ 0\end{bmatrix}$$

第二层（线性 + sigmoid）：

$$z_2 = \mathbf{W}_2 \mathbf{a}_1 + b_2 = 0.4 \cdot 0.35 + (-0.3) \cdot 0 + 0.05 = 0.19$$

$$\hat{y} = \sigma(z_2) = \frac{1}{1+e^{-0.19}} \approx 0.5474$$

**维度验证**：$(1 \times 2)(2 \times 1) = (1 \times 1)$，输出是标量。

### 3.3 转置矩阵与反向传播

使用二元交叉熵损失：$L = -[y \log\hat{y} + (1-y)\log(1-\hat{y})]$

链式法则逐层求梯度。关键恒等式的推导：

设 $\mathbf{z} = \mathbf{W}\mathbf{x}$，$L$ 是关于 $\mathbf{z}$ 的函数，记 $\delta = \partial L / \partial \mathbf{z} \in \mathbb{R}^m$。

对 $\mathbf{x}$ 的梯度：利用全微分 $dL = \delta^T d\mathbf{z} = \delta^T \mathbf{W} d\mathbf{x}$，故

$$\frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^T \delta \in \mathbb{R}^n$$

对 $\mathbf{W}$ 的梯度：$dL = \delta^T d\mathbf{W} \mathbf{x}$，利用迹技巧 $\text{tr}(A^T B) = \text{vec}(A)^T \text{vec}(B)$，得

$$\frac{\partial L}{\partial \mathbf{W}} = \delta \mathbf{x}^T \in \mathbb{R}^{m \times n}$$

**几何直觉**：前向用 $\mathbf{W}$ 将 $\mathbf{x}$ 从 $n$ 维映射到 $m$ 维；反向用 $\mathbf{W}^T$ 将误差信号从 $m$ 维映射回 $n$ 维，两者互为伴随算子。

以 Running Example 计算 $\partial L / \partial \mathbf{W}_1$（省略中���激活梯度展开）：

$$\frac{\partial L}{\partial z_2} = \hat{y} - y \approx 0.5474 - 1 = -0.4526$$

$$\frac{\partial L}{\partial \mathbf{a}_1} = \mathbf{W}_2^T \cdot (-0.4526) = \begin{bmatrix}0.4 \\ -0.3\end{bmatrix} \cdot (-0.4526) = \begin{bmatrix}-0.1810 \\ 0.1358\end{bmatrix}$$

ReLU 反向（$z_1^{(1)} = 0.35 > 0$，$z_1^{(2)} = -0.05 \leq 0$）：

$$\delta_1 = \frac{\partial L}{\partial \mathbf{z}_1} = \begin{bmatrix}-0.1810 \\ 0\end{bmatrix}$$

$$\frac{\partial L}{\partial \mathbf{W}_1} = \delta_1 \mathbf{x}^T = \begin{bmatrix}-0.1810 \\ 0\end{bmatrix} \begin{bmatrix}1 & 0.5\end{bmatrix} = \begin{bmatrix}-0.1810 & -0.0905 \\ 0 & 0\end{bmatrix}$$

维度验证：$(2 \times 1)(1 \times 2) = (2 \times 2)$，与 $\mathbf{W}_1$ 形状一致。

### 3.4 GPU 加速：GEMM 与张量核心

GPU 加速矩阵乘法的层次：

1. **SIMD**（Single Instruction Multiple Data）：一条指令并行处理 16/32/64 个浮点数
2. **张量核心**（Tensor Core）：A100 上一个时钟周期完成 $16 \times 16 \times 16$ 的 FP16 矩阵乘加，峰值 312 TFLOPS（FP16）vs CPU ~2 TFLOPS（FP32）
3. **共享内存分块**（Tiling）：将 $\mathbf{W}$ 和 $\mathbf{X}$ 分成 $16 \times 16$ 的瓦片载入 SRAM，减少全局显存访问，访问带宽是性能瓶颈

批处理的必要性：$B=1$ 时退化为矩阵-向量乘法（GEMV），张量核心利用率 $< 5\%$；$B \geq 32$ 时 GEMM 利用率 $> 70\%$。

---

## 代码实现

### 4.1 基础向量运算

```python
import numpy as np

# Running example 参数
x = np.array([1.0, 0.5])
W1 = np.array([[0.2, 0.3], [0.1, -0.5]])
b1 = np.array([0.1, -0.2])
W2 = np.array([[0.4, -0.3]])
b2 = np.array([0.05])
y = 1.0

# 内积、范数、余弦相似度
u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])
dot = np.dot(u, v)                          # 32.0
norm_u = np.linalg.norm(u)                  # sqrt(14) ≈ 3.742
cosine = dot / (norm_u * np.linalg.norm(v)) # ≈ 0.9746

assert abs(dot - 32.0) < 1e-10
assert abs(norm_u - np.sqrt(14)) < 1e-10
assert abs(cosine - 0.9746318) < 1e-6
print(f"dot={dot}, norm={norm_u:.4f}, cosine={cosine:.6f}")
```

### 4.2 前向传播

```python
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(x, W1, b1, W2, b2):
    z1 = W1 @ x + b1          # (2,2)@(2,) + (2,) → (2,)
    a1 = relu(z1)
    z2 = W2 @ a1 + b2         # (1,2)@(2,) + (1,) → (1,)
    y_hat = sigmoid(z2)
    return z1, a1, z2, y_hat

z1, a1, z2, y_hat = forward(x, W1, b1, W2, b2)

assert z1.shape == (2,), f"z1 shape error: {z1.shape}"
assert abs(z1[0] - 0.35) < 1e-10 and abs(z1[1] - (-0.05)) < 1e-10, f"z1={z1}"
assert abs(a1[0] - 0.35) < 1e-10 and a1[1] == 0.0
assert abs(z2[0] - 0.19) < 1e-10, f"z2={z2}"
print(f"z1={z1}, a1={a1}, z2={z2[0]:.4f}, y_hat={y_hat[0]:.4f}")
```

### 4.3 反向传播

```python
def backward(x, z1, a1, z2, y_hat, W1, W2, y):
    # 损失对 z2 的梯度（二元交叉熵 + sigmoid）
    dz2 = y_hat - y                          # (1,)

    # 第二层梯度
    dW2 = np.outer(dz2, a1)                 # (1,)×(2,) → (1,2)
    db2 = dz2

    # 回传到 a1
    da1 = W2.T @ dz2                         # (2,1)@(1,) → (2,)

    # ReLU 反向
    dz1 = da1 * (z1 > 0).astype(float)      # 逐元素，dead neuron 处梯度为 0

    # 第一层梯度
    dW1 = np.outer(dz1, x)                  # (2,)×(2,) → (2,2)
    db1 = dz1

    return dW1, db1, dW2, db2

dW1, db1, dW2, db2 = backward(x, z1, a1, z2, y_hat, W1, W2, y)

assert dW1.shape == W1.shape, f"dW1 shape: {dW1.shape}"
assert abs(dW1[1, 0]) < 1e-10, "死亡神经元梯度应为0"
print(f"dW1=\n{dW1}")
print(f"dW2={dW2}")
```

### 4.4 批处理前向传播

```python
# 验证批处理维度规则：(m,n)@(n,B) → (m,B)
B = 8
X_batch = np.random.randn(2, B)             # 输入矩阵，每列是一个样本

Z1_batch = W1 @ X_batch + b1[:, None]      # (2,2)@(2,8) + (2,1) → (2,8)
A1_batch = relu(Z1_batch)
Z2_batch = W2 @ A1_batch + b2[:, None]     # (1,2)@(2,8) + (1,1) → (1,8)
Y_hat_batch = sigmoid(Z2_batch)

assert Z1_batch.shape == (2, B)
assert Z2_batch.shape == (1, B)
print(f"批量输出形状: {Y_hat_batch.shape}")  # (1, 8)

# 验证批处理结果与逐样本一致
_, _, _, y_hat_single = forward(X_batch[:, 0], W1, b1, W2, b2)
assert abs(Y_hat_batch[0, 0] - y_hat_single[0]) < 1e-10
```

---

## 工程权衡与常见坑

### 梯度爆炸：深层网络的反向传播病态

$L$ 层网络反向传播时，梯度是多个权重矩阵转置的连乘：

$$\frac{\partial L}{\partial \mathbf{x}^{(1)}} = \mathbf{W}_L^T \mathbf{W}_{L-1}^T \cdots \mathbf{W}_1^T \delta_L$$

若每层谱范数 $\sigma_1(\mathbf{W}_\ell) > 1$，梯度范数指数增长：$\|\partial L / \partial \mathbf{x}^{(1)}\|_2 \leq \prod_\ell \sigma_1(\mathbf{W}_\ell) \cdot \|\delta_L\|_2$

```python
# 演示梯度爆炸
np.random.seed(42)
depth = 20
x_demo = np.random.randn(64)
grad = np.ones(64)

# 不初始化的情况（均值0，方差1）
for _ in range(depth):
    W = np.random.randn(64, 64)   # 谱范数约为 sqrt(64) ≈ 8
    grad = W.T @ grad

print(f"未初始化梯度范数: {np.linalg.norm(grad):.2e}")   # 爆炸

# He 初始化（适用于 ReLU，方差 2/n）
grad = np.ones(64)
for _ in range(depth):
    W = np.random.randn(64, 64) * np.sqrt(2.0 / 64)   # 谱范数约为 sqrt(2) ≈ 1.41，但期望范数收敛
    grad = W.T @ grad

print(f"He初始化梯度范数: {np.linalg.norm(grad):.2e}")   # 稳定
```

He 初始化将每层权重方差设为 $2/n_{in}$，使得前向传播时方差保持为 1（假设 ReLU 激活）。

### 工程权衡对比表

| 问题 | 触发条件 | 根因 | 解决方案 | 代价 |
|------|----------|------|----------|------|
| **梯度爆炸** | 深层、大权重 | $\prod \sigma_1(\mathbf{W}_\ell) \gg 1$ | He/Xavier 初始化；梯度剪裁 $\|\mathbf{g}\| > \tau$ | 额外超参 $\tau$ |
| **梯度消失** | 深层、Sigmoid | $\prod \sigma_1(\mathbf{W}_\ell) \ll 1$ | ReLU 激活；残差连接；BatchNorm | 增加激活计算量 |
| **显存溢出** | 批大小过大 | 中间激活存储 $O(BmL)$ | 梯度��积；混合精度（FP16） | 训练速度降低 |
| **转置开销** | 显式 `.T` + GEMM | 破坏显存访问局部性 | cuBLAS `transA/transB` 标志 | —（库内处理） |
| **矩阵维度错误** | 向量作行向量乘 | `(n,) vs (n,1)` 语义不同 | 使用 `@` 算符；显式 `reshape` | — |

### 数值稳定性：Softmax 溢出

Softmax 计算 $\text{softmax}(\mathbf{z})_i = e^{z_i} / \sum_j e^{z_j}$ 在 $z_i > 88$（FP32 上溢）时失败。标准技巧：减去 $\max(\mathbf{z})$，数学等价但数值稳定：

```python
def stable_softmax(z):
    z_shifted = z - np.max(z)    # 最大值变为 0，其余 ≤ 0，exp 不溢出
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum()

z_test = np.array([1000.0, 1001.0, 1002.0])
# 直接计算会 nan，stable_softmax 正确
result = stable_softmax(z_test)
assert abs(result.sum() - 1.0) < 1e-10
print(result)  # [0.0900, 0.2447, 0.6652]
```

---

## 替代方案与适用边界

### 三种矩阵表示的对比

| 方案 | 存储复杂度 | 乘法复杂度 | 适用场景 | 精度损失 |
|------|-----------|-----------|---------|---------|
| **稠密 FP32** | $O(mn)$ | $O(mnp)$ | 稀疏度 $< 50\%$，标准训练 | 0（基线） |
| **稠密 FP16/BF16** | $O(mn/2)$ | $O(mnp)$（张量核心 2× 快）| 混合精度训练，A100/H100 | $< 0.1\%$ |
| **INT8 量化（QAT）** | $O(mn/4)$ | $O(mnp)$（整数 SIMD，4-8×）| 推理加速，RTX/A10G | $< 0.5\%$ top-1 |
| **稀疏（CSR）** | $O(\text{nnz})$ | $O(\text{nnz} \cdot p)$ | 剪枝后稀疏度 $> 90\%$ | 取决于剪枝策略 |
| **低秩分解** $\mathbf{W} \approx \mathbf{UV}^T$ | $O((m+n)r)$ | $O((m+n)rp)$，$r \ll n$ | LoRA 微调，某些 Attention 层 | 取决于秩 $r$ |

### 低秩分解的实现

LoRA（Low-Rank Adaptation）将更新量分解为两个小矩阵：$\Delta \mathbf{W} = \mathbf{U}\mathbf{V}^T$，$\mathbf{U} \in \mathbb{R}^{m \times r}$，$\mathbf{V} \in \mathbb{R}^{n \times r}$，$r \ll \min(m, n)$。

```python
def low_rank_forward(x, W_frozen, U, V):
    """
    W_frozen: (m,n) 冻结权重
    U: (m,r), V: (n,r) 低秩更新
    """
    # 完整权重：W_frozen + U @ V.T
    # 但不显式构造完整矩阵（避免 O(mn) 存储）
    z_frozen = W_frozen @ x          # (m,)
    z_update = U @ (V.T @ x)        # (m,r)@(r,) via (n,r).T@(n,)→(r,)
    return z_frozen + z_update

m, n, r = 512, 512, 8
W_frozen = np.random.randn(m, n) * 0.01
U = np.random.randn(m, r) * 0.01
V = np.random.randn(n, r) * 0.01
x_demo = np.random.randn(n)

z_lora = low_rank_forward(x_demo, W_frozen, U, V)
z_full = (W_frozen + U @ V.T) @ x_demo   # 参考实现

assert np.allclose(z_lora, z_full, atol=1e-10), "低秩分解结果不匹配"
# 参数量对比：完整 W 为 512*512=262144；低秩更新仅 2*512*8=8192（节省 97%）
print(f"完整参数量: {m*n}, 低秩参数量: {(m+n)*r}, 压缩比: {m*n/((m+n)*r):.1f}×")
```

### 稀疏的边界条件

稀疏矩阵乘法（cuSPARSE/SpMM）在稀疏度 $> 90\%$ 时才优于稠密 GEMM，原因是稀疏格式本身有索引开销（CSR 格���额外存储行指针和列索引），且无法充分利用张量核心的规则化计算模式。结构化剪枝（如 2:4 稀疏，每 4 个元素中保留 2 个）是 NVIDIA 支持的硬件加速格式，可在 A100 上实现 2× 加速而不损失张量核心效率。

---

## 参考资料

1. **Golub, G. H. & Van Loan, C. F.** (2013). *Matrix Computations*, 4th ed. Johns Hopkins University Press. — 矩阵乘法、条件数、LU 分解的权威参考，第 1-3 章

2. **NVIDIA. cuBLAS Library User Guide** (2024). — GEMM 接口、转置标志、张量核心使用，[docs.nvidia.com/cuda/cublas](https://docs.nvidia.com/cuda/cublas/)

3. **He, K., Zhang, X., Ren, S. & Sun, J.** (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. *ICCV 2015*. — He 初始化的理论推导，基于谱范数分析

4. **Hu, E. J. et al.** (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. arXiv:2106.09685 — 低秩分解在微调中的应用，包���秩选择的经验分析

5. **Micikevicius, P. et al.** (2018). Mixed Precision Training. *ICLR 2018*. arXiv:1710.03740 — FP16 混合精度训练的数值稳定性分析，损失缩放技术
