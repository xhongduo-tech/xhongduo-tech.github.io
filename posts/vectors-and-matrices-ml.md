看起来写入权限还未授予。我先把完整文章内容输出在这里，你可以直接复制，或者授权后我帮你写入文件。

---

以下是 `posts/vector-matrix-neural-network-math.md` 的完整内容：

```markdown
## 核心定位

神经网络的前向传播，本质是一系列矩阵乘法与逐元素非线性变换的交替组合。一个全连接层的计算可以精确表述为：

$$\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$

其中 $\mathbf{x} \in \mathbb{R}^n$ 是输入向量，$\mathbf{W} \in \mathbb{R}^{m \times n}$ 是权重矩阵，$\mathbf{b} \in \mathbb{R}^m$ 是偏置向量，$\sigma$ 是非线性激活函数。$n$ 维输入经线性变换映射到 $m$ 维输出——维度变换由矩阵形状决定。

理解这一计算链条，需要从向量运算、矩阵乘法、特殊矩阵操作三个层次逐步建立。

---

## 向量：神经网络的基本数据单元

神经网络中的所有数据——输入特征、隐藏状态、梯度——都以向量形式存在。一个 $n$ 维列向量：

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

### 内积（Dot Product）

两个同维向量 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ 的内积定义为：

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

内积的几何意义：$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$，度量两个向量的方向一致性。在神经网络中，全连接层每个输出神经元的计算就是权重向量与输入向量的内积加偏置。

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# 内积：三种等价写法
dot1 = np.dot(a, b)        # 32.0
dot2 = a @ b               # 32.0
dot3 = np.sum(a * b)       # 逐元素相乘再求和 = 1*4 + 2*5 + 3*6 = 32.0

print(f"dot product: {dot1}")  # 32.0
```

### L2 范数（Euclidean Norm）

向量 $\mathbf{x} \in \mathbb{R}^n$ 的 L2 范数：

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{\mathbf{x} \cdot \mathbf{x}}$$

L2 范数度量向量的"长度"。在深度学习中，L2 正则化（weight decay）通过惩罚 $\|\mathbf{W}\|_2^2$ 防止权重过大；梯度裁剪通过限制 $\|\nabla\|_2$ 防止梯度爆炸。

```python
x = np.array([3.0, 4.0])

norm = np.linalg.norm(x)           # sqrt(9 + 16) = 5.0
norm_manual = np.sqrt(np.dot(x, x))  # 等价手动计算

print(f"L2 norm: {norm}")  # 5.0
```

### 余弦相似度（Cosine Similarity）

$$\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}$$

取值范围 $[-1, 1]$。$1$ 表示方向完全一致，$0$ 表示正交，$-1$ 表示方向相反。余弦相似度只关注方向，忽略幅值——这正是 embedding 检索和对比学习中使用它的原因：两个语义相近的句子，其 embedding 向量方向接近，但幅值可能差异很大。

```python
a = np.array([1.0, 0.0, 1.0])
b = np.array([0.0, 1.0, 1.0])

cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# 分子: 0 + 0 + 1 = 1
# 分母: sqrt(2) * sqrt(2) = 2
# cos_sim = 0.5

print(f"cosine similarity: {cos_sim:.4f}")  # 0.5000
```

三种向量运算的对比：

| 运算 | 输入 | 输出 | 神经网络用途 |
|------|------|------|-------------|
| 内积 | 两个 $n$ 维向量 | 标量 | 全连接层的单个神经元计算 |
| L2 范数 | 一个 $n$ 维向量 | 非负标量 | 正则化、梯度裁剪、归一化 |
| 余弦相似度 | 两个 $n$ 维向量 | $[-1,1]$ 标量 | embedding 检索、对比学习损失 |

---

## 矩阵乘法：维度变换的核心机制

### 维度规则

矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 与 $\mathbf{B} \in \mathbb{R}^{n \times p}$ 的乘积 $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times p}$。乘法合法的充要条件：$\mathbf{A}$ 的列数等于 $\mathbf{B}$ 的行数。

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}, \quad i \in [1,m], \; j \in [1,p]$$

这意味着 $\mathbf{C}$ 的第 $(i,j)$ 元素是 $\mathbf{A}$ 的第 $i$ 行与 $\mathbf{B}$ 的第 $j$ 列的内积。整个矩阵乘法可以理解为 $m \times p$ 次内积运算。

### 具体数值示例

以一个全连接层为例：输入维度 3，输出维度 2。

$$\mathbf{W} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3}, \quad \mathbf{x} = \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix} \in \mathbb{R}^{3}$$

$$\mathbf{W}\mathbf{x} = \begin{bmatrix} 1 \cdot 1 + 2 \cdot 0 + 3 \cdot (-1) \\ 4 \cdot 1 + 5 \cdot 0 + 6 \cdot (-1) \end{bmatrix} = \begin{bmatrix} -2 \\ -2 \end{bmatrix} \in \mathbb{R}^{2}$$

3 维输入被映射到 2 维输出。维度变换完全由权重矩阵的形状 $(2, 3)$ 决定。

```python
W = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape: (2, 3)
x = np.array([1, 0, -1])   # shape: (3,)

y = W @ x  # shape: (2,)  → 3维 → 2维
print(f"W @ x = {y}")  # [-2 -2]
```

### 批量矩阵乘法

实际训练中不会逐条处理样本。设 batch size 为 $B$，输入矩阵 $\mathbf{X} \in \mathbb{R}^{B \times n}$（每行一个样本），权重 $\mathbf{W} \in \mathbb{R}^{n \times m}$（注意此处转置存储以适配行向量惯例）：

$$\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b} \in \mathbb{R}^{B \times m}$$

$B$ 个样本的线性变换在一次矩阵乘法中完成。

```python
B = 4   # batch size
n = 3   # 输入维度
m = 2   # 输出维度

X = np.random.randn(B, n)   # (4, 3)
W = np.random.randn(n, m)   # (3, 2)
b = np.random.randn(m)      # (2,)  广播到每个样本

Y = X @ W + b               # (4, 3) @ (3, 2) + (2,) → (4, 2)
print(f"input shape:  {X.shape}")   # (4, 3)
print(f"output shape: {Y.shape}")   # (4, 2)
```

维度变换链的追踪是 debug 神经网络的基本功。一个三层 MLP 的维度流：

$$\mathbb{R}^{B \times 784} \xrightarrow{W_1: 784 \times 256} \mathbb{R}^{B \times 256} \xrightarrow{W_2: 256 \times 128} \mathbb{R}^{B \times 128} \xrightarrow{W_3: 128 \times 10} \mathbb{R}^{B \times 10}$$

```python
# 三层 MLP 维度追踪
dims = [784, 256, 128, 10]
x = np.random.randn(32, dims[0])  # batch=32, MNIST 28x28 展平

for i in range(len(dims) - 1):
    W = np.random.randn(dims[i], dims[i+1]) * 0.01  # 小随机初始化
    b = np.zeros(dims[i+1])
    x = x @ W + b
    x = np.maximum(x, 0)  # ReLU
    print(f"Layer {i+1}: {dims[i]} → {dims[i+1]}, output shape: {x.shape}")
# Layer 1: 784 → 256, output shape: (32, 256)
# Layer 2: 256 → 128, output shape: (32, 128)
# Layer 3: 128 → 10, output shape: (32, 10)
```

---

## 转置与逆矩阵

### 转置（Transpose）

矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 的转置 $\mathbf{A}^\top \in \mathbb{R}^{n \times m}$，定义为 $(A^\top)_{ij} = A_{ji}$。行列互换。

关键性质：

$$(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top$$

这个性质在反向传播推导中反复出现。前向传播 $\mathbf{y} = \mathbf{W}\mathbf{x}$，反向传播中输入梯度为 $\frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^\top \frac{\partial L}{\partial \mathbf{y}}$，权重梯度为 $\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{y}} \mathbf{x}^\top$。转置操作在梯度计算中承担维度对齐的角色。

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape: (2, 3)

AT = A.T                    # shape: (3, 2)
print(f"A shape:  {A.shape}")   # (2, 3)
print(f"AT shape: {AT.shape}")  # (3, 2)
print(f"A[0,1] == AT[1,0]: {A[0,1] == AT[1,0]}")  # True

# 验证 (AB)^T = B^T A^T
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])  # shape: (3, 2)

lhs = (A @ B).T           # 先乘后转
rhs = B.T @ A.T           # 先各自转再反序乘
print(f"(AB)^T == B^T A^T: {np.allclose(lhs, rhs)}")  # True
```

### 逆矩阵（Inverse）

方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 的逆 $\mathbf{A}^{-1}$ 满足 $\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}_n$。逆矩阵存在的充要条件：$\det(\mathbf{A}) \neq 0$（即 $\mathbf{A}$ 满秩）。

计算意义：$\mathbf{A}\mathbf{x} = \mathbf{b}$ 的解为 $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$。逆矩阵"撤销"了 $\mathbf{A}$ 的线性变换。

```python
A = np.array([[2.0, 1.0],
              [5.0, 3.0]])

A_inv = np.linalg.inv(A)
print(f"A:\n{A}")
print(f"A_inv:\n{A_inv}")
# [[ 3. -1.]
#  [-5.  2.]]

# 验证 A @ A_inv = I
print(f"A @ A_inv:\n{A @ A_inv}")
# [[1. 0.]
#  [0. 1.]]

# 解线性方程 Ax = b
b = np.array([4.0, 11.0])
x = A_inv @ b
print(f"solution x = {x}")  # [1. 2.]

# 验证 A @ x = b
print(f"A @ x = {A @ x}")   # [ 4. 11.]
```

实际工程中几乎不显式计算逆矩阵。原因：

| 方法 | 时间复杂度 | 数值稳定性 | 适用场景 |
|------|-----------|-----------|---------|
| 显式求逆 $\mathbf{A}^{-1}\mathbf{b}$ | $O(n^3)$ | 差，误差累积 | 极少使用 |
| LU 分解求解 | $O(n^3)$ | 好 | 一般线性方程组 |
| Cholesky 分解 | $O(n^3/3)$ | 好 | 对称正定矩阵 |
| 迭代法（CG 等） | $O(n^2 k)$, $k$=迭代次数 | 依赖条件数 | 大规模稀疏系统 |

`np.linalg.solve(A, b)` 内部使用 LU 分解，比 `np.linalg.inv(A) @ b` 更快且数值更稳定。

```python
# 推荐方式：直接求解而非显式求逆
x_solve = np.linalg.solve(A, b)
x_inv = np.linalg.inv(A) @ b

print(f"solve: {x_solve}")  # [1. 2.]
print(f"inv:   {x_inv}")    # [1. 2.]
# 结果一致，但 solve 数值更稳定，尤其在矩阵条件数大时
```

---

## 神经网络中的矩阵运算全景

将上述基础运算组合，构成神经网络的完整计算图。以单层前向+反向传播为例，展示矩阵运算如何串联：

$$\text{前向:} \quad \mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}, \quad \mathbf{A} = \text{ReLU}(\mathbf{Z})$$

$$\text{反向:} \quad \frac{\partial L}{\partial \mathbf{Z}} = \frac{\partial L}{\partial \mathbf{A}} \odot \mathbf{1}_{Z>0}$$

$$\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Z}}, \quad \frac{\partial L}{\partial \mathbf{b}} = \sum_{\text{batch}} \frac{\partial L}{\partial \mathbf{Z}}, \quad \frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Z}} \mathbf{W}^\top$$

其中 $\odot$ 是逐元素乘法（Hadamard product），$\mathbf{1}_{Z>0}$ 是 ReLU 的导数掩码。

```python
np.random.seed(42)

# 维度设定
B, n_in, n_out = 4, 3, 2

# 前向传播
X = np.random.randn(B, n_in)             # (4, 3) 输入
W = np.random.randn(n_in, n_out) * 0.1   # (3, 2) 权重
b = np.zeros(n_out)                        # (2,)   偏置

Z = X @ W + b          # (4, 3) @ (3, 2) = (4, 2) 线性变换
A = np.maximum(Z, 0)   # (4, 2) ReLU 激活

# 假设上游梯度（从损失函数回传）
dA = np.random.randn(B, n_out)  # (4, 2)

# 反向传播
dZ = dA * (Z > 0).astype(float)  # (4, 2) ReLU 导数掩码
dW = X.T @ dZ                     # (3, 4) @ (4, 2) = (3, 2) 权重梯度
db = np.sum(dZ, axis=0)           # (2,) 偏置梯度（batch 求和）
dX = dZ @ W.T                     # (4, 2) @ (2, 3) = (4, 3) 输入梯度

print(f"dW shape: {dW.shape}")  # (3, 2) — 与 W 同形
print(f"db shape: {db.shape}")  # (2,)   — 与 b 同形
print(f"dX shape: {dX.shape}")  # (4, 3) — 与 X 同形
```

反向传播中每个梯度的形状都与对应参数完全一致——这不是巧合，而是矩阵求导的直接结果。转置在这里的作用是对齐内积的求和维度。

---

## 为什么 GPU 擅长矩阵运算

矩阵乘法 $\mathbf{C} = \mathbf{A}\mathbf{B}$，其中 $\mathbf{A} \in \mathbb{R}^{m \times n}$，$\mathbf{B} \in \mathbb{R}^{n \times p}$，需要 $m \times n \times p$ 次乘法和 $m \times (n-1) \times p$ 次加法。关键观察：$\mathbf{C}$ 中的 $m \times p$ 个元素可以独立计算——每个 $C_{ij}$ 只依赖 $\mathbf{A}$ 的第 $i$ 行和 $\mathbf{B}$ 的第 $j$ 列，与其他元素无数据依赖。

### CPU vs GPU 的架构差异

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数 | 8-64 个大核 | 数千个小核（CUDA cores） |
| 单核能力 | 强，复杂控制流 | 弱，简单算术 |
| 缓存 | 大（L1/L2/L3 共数十 MB） | 小（每个 SM 数百 KB） |
| 内存带宽 | ~50-100 GB/s | ~1-3 TB/s（HBM） |
| 适合任务 | 分支密集、串行逻辑 | 大规模数据并行 |

GPU 的优势来自两个层面：

**SIMT（Single Instruction, Multiple Threads）并行**：GPU 以 warp（32 个线程）为单位执行同一条指令。矩阵乘法中每个线程计算 $\mathbf{C}$ 的一部分元素，数千线程同时执行相同的乘加操作。这与 CPU 的 SIMD（Single Instruction, Multiple Data）类似但规模大几个数量级——CPU 的 AVX-512 一次处理 16 个 float32，而一块 A100 GPU 有 6912 个 CUDA 核心。

**内存带宽**：矩阵乘法的算术强度（arithmetic intensity）为 $O(n)$ FLOP/byte（对方阵而言），属于计算密集型。GPU 的高带宽内存（HBM）提供远超 CPU 的数据吞吐，配合高 FLOP/s 的计算单元，使矩阵乘法能充分利用硬件。

### Tensor Core 与混合精度

现代 GPU（Volta 架构起）引入 Tensor Core，专门加速矩阵乘加运算（GEMM）。一个 Tensor Core 每周期执行一次 $4 \times 4 \times 4$ 的矩阵乘加：

$$\mathbf{D}_{4\times4} = \mathbf{A}_{4\times4} \cdot \mathbf{B}_{4\times4} + \mathbf{C}_{4\times4}$$

A100 的 Tensor Core 在 FP16 下达到 312 TFLOPS，是 FP32 CUDA Core 的 16 倍。这驱动了混合精度训练的普及：前向传播用 FP16 加速矩阵乘法，梯度累积和参数更新保持 FP32 精度。

```python
# NumPy 验证：矩阵乘法中每个元素可独立计算
m, n, p = 3, 4, 2
A = np.random.randn(m, n)
B = np.random.randn(n, p)

# 方法 1: 直接矩阵乘法
C1 = A @ B

# 方法 2: 逐元素独立计算（模拟 GPU 并行）
C2 = np.zeros((m, p))
for i in range(m):       # 每个 (i,j) 独立，可并行
    for j in range(p):
        C2[i, j] = np.dot(A[i, :], B[:, j])  # 行向量与列向量内积

print(f"Results match: {np.allclose(C1, C2)}")  # True
```

### 计算量的直觉

一个 GPT-3 规模的模型（175B 参数），单次前向传播约需 $350 \times 10^{12}$ 次浮点运算（350 TFLOPS），几乎全部是矩阵乘法。A100 GPU 在 FP16 Tensor Core 下理论峰值 312 TFLOPS——单卡单次前向传播就需约 1 秒。训练中每步包含前向 + 反向（约 $3 \times$ 前向的计算量），在数万亿 token 上迭代，这解释了为什么大模型训练需要数千块 GPU 并行数月。

---

## 常见坑点与工程细节

**维度不匹配**：`matmul` 报错 `shapes (a,b) and (c,d) not aligned: b != c`。debug 方法：在每次矩阵运算前打印 shape，逐层追踪维度流。PyTorch 中用 `tensor.shape` 而非 `tensor.size()` 保持一致性。

**广播（Broadcasting）陷阱**：`(4, 3) + (3,)` 合法（偏置加法），但 `(4, 3) + (4,)` 会被广播为 `(4, 3) + (4, 1)` 而非报错——结果在数学上无意义但不会抛异常。始终显式检查广播行为。

```python
# 广播陷阱示例
X = np.ones((4, 3))
b_correct = np.ones(3)     # (3,) → 广播到 (4, 3) 的每一行，正确
b_wrong = np.ones(4)        # (4,) → 广播为 (4, 1) → (4, 3)，语义错误

print((X + b_correct).shape)  # (4, 3) — 正确：每行加同一偏置
print((X + b_wrong[:, None]).shape)  # (4, 3) — 注意这会按列广播
# 永远显式指定广播维度，不要依赖自动推断
```

**数值精度**：FP32 有约 7 位有效数字，FP16 只有约 3.3 位。大矩阵乘法中，累加大量小数会导致精度损失。解决方案：混合精度训练中使用 FP32 做梯度累积；推理中使用 BF16（指数位宽与 FP32 一致，不容易溢出）而非 FP16。

| 格式 | 有效位数 | 指数范围 | 适用场景 |
|------|---------|---------|---------|
| FP32 | ~7.2 位 | $\pm 3.4 \times 10^{38}$ | 参数更新、梯度累积 |
| FP16 | ~3.3 位 | $\pm 6.5 \times 10^{4}$ | 前向/反向矩阵乘法（配合 loss scaling） |
| BF16 | ~3.4 位 | $\pm 3.4 \times 10^{38}$ | 训练/推理通用（不需要 loss scaling） |
| INT8 | 8 bit 整数 | $[-128, 127]$ | 推理量化 |

**转置开销**：`np.array` 默认行优先（C order）。对 `(m, n)` 矩阵，访问同一行的元素在内存中连续。`.T` 返回 view 而非拷贝，但后续计算可能因非连续访问导致 cache miss。对性能敏感的场景，用 `np.ascontiguousarray(A.T)` 显式复制为连续内存。

---

## 参考资料

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 2: Linear Algebra. MIT Press. https://www.deeplearningbook.org/contents/linear_algebra.html
2. NVIDIA. *CUDA C++ Programming Guide: Warp-Level Matrix Multiply-Accumulate*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
3. Micikevicius, P. et al. (2018). *Mixed Precision Training*. ICLR 2018. https://arxiv.org/abs/1710.03740
4. Harris, C.R. et al. (2020). *Array programming with NumPy*. Nature, 585, 357-362. https://numpy.org/doc/stable/reference/routines.linalg.html
5. Jia, Z. et al. (2018). *Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking*. https://arxiv.org/abs/1804.06826
```

---

对应的 `posts.json` 条目：

```json
{
  "title": "向量与矩阵：神经网络计算的数学基础",
  "slug": "vector-matrix-neural-network-math",
  "date": "2026-03-04",
  "author": "both",
  "tags": ["底层原理", "线性代数", "GPU"],
  "summary": "从内积、范数、余弦相似度到矩阵乘法维度规则，用 NumPy 代码验证每个运算，串联前向/反向传播的完整矩阵计算链，解析 GPU 并行加速矩阵运算的硬件机制。"
}
```

---
