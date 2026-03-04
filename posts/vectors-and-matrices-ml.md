## 向量与矩阵：神经网络计算的数学基础

神经网络本质上是复合函数 $f(x) = σ(W_l σ(W_{l-1} ... σ(W_1 x + b_1) ... + b_{l-1}) + b_l)$，每层都是线性变换（矩阵乘法）加上非线性激活函数。理解向量与矩阵的代数运算，是掌握神经网络计算的第一步。

---

## 向量：数据的坐标表示

向量是 $n$ 维空间中的有向线段，记作 $\mathbf{v} \in \mathbb{R}^n$。在神经网络中，向量用于表示：
- 输入特征：$\mathbf{x} = [x_1, x_2, ..., x_d]^T \in \mathbb{R}^d$
- 模型参数：$\mathbf{w} = [w_1, w_2, ..., w_d]^T \in \mathbb{R}^d$
- 隐藏状态：$\mathbf{h} = [h_1, h_2, ..., h_m]^T \in \mathbb{R}^m$

### 内积

两个向量 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ 的内积定义为：
$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \mathbf{u}^T \mathbf{v}
$$

内积衡量两个向量的对齐程度：当 $\mathbf{u}$ 与 $\mathbf{v}$ 同向时取最大值，正交时为零，反向时取负值。这意味着内积捕获了向量间的方向相关性，这正是线性模型 $\hat{y} = \mathbf{w}^T \mathbf{x}$ 的核心——寻找与标签最相关的特征方向。

### L2 范数

向量的 L2 范数（欧氏长度）定义为：
$$
\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2} = \sqrt{\mathbf{v}^T \mathbf{v}}
$$

L2 范数在神经网络中的应用：正则化项 $\lambda \|\mathbf{w}\|_2^2$ 防止过拟合，梯度裁剪 $\|\nabla_\theta \mathcal{L}\|_2 \leq c$ 避免梯度爆炸，词向量归一化 $\hat{\mathbf{e}} = \mathbf{e} / \|\mathbf{e}\|_2$ 消除词频影响。

### 余弦相似度

余弦相似度忽略向量长度差异，仅比较方向：
$$
\text{cos}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \in [-1, 1]
$$

语义搜索、推荐系统、注意力机制广泛使用余弦相似度，因为词频、文档长度等量级因素不应影响语义匹配。

---

## 矩阵：线性变换的表示

矩阵 $A \in \mathbb{R}^{m \times n}$ 表示从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 的线性变换 $\mathbf{y} = A\mathbf{x}$。矩阵的第 $i$ 行、第 $j$ 列元素记作 $A_{i,j}$。

### 矩阵乘法维度规则

矩阵乘法 $C = AB$ 要求 $A$ 的列数等于 $B$ 的行数：$A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}$，则 $C \in \mathbb{R}^{m \times p}$。

元素级定义为：
$$
C_{i,j} = \sum_{k=1}^n A_{i,k} B_{k,j}
$$

**数值示例**：设 $A \in \mathbb{R}^{2 \times 3}, B \in \mathbb{R}^{3 \times 2}$。

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad
B = \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix}
$$

$$
C = AB = \begin{bmatrix}
1 \times 7 + 2 \times 9 + 3 \times 11 & 1 \times 8 + 2 \times 10 + 3 \times 12 \\
4 \times 7 + 5 \times 9 + 6 \times 11 & 4 \times 8 + 5 \times 10 + 6 \times 12
\end{bmatrix} = \begin{bmatrix} 58 & 64 \\ 139 & 154 \end{bmatrix}
$$

这意味着矩阵乘法本质上是 $m \times p$ 次内积运算，每次内积涉及 $n$ 个元素的乘加。复杂度为 $O(mnp)$。

---

## 转置：空间映射的逆转

矩阵 $A \in \mathbb{R}^{m \times n}$ 的转置 $A^T \in \mathbb{R}^{n \times m}$ 满足 $(A^T)_{i,j} = A_{j,i}$。

转置的性质：
- $(A^T)^T = A$
- $(AB)^T = B^T A^T$（注意顺序反转）
- $(\mathbf{u} \cdot \mathbf{v}) = \mathbf{u}^T \mathbf{v} = \mathbf{v}^T \mathbf{u}$

在神经网络中，转置用于：
- 梯度反向传播：$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{x}^T$
- 注意力机制：$Attention(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- 数据布局转换：batch-first 到 sequence-first

**数值示例**：
$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad
A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
$$

---

## 逆矩阵：变换的可逆性

方阵 $A \in \mathbb{R}^{n \times n}$ 的逆矩阵 $A^{-1}$ 满足 $AA^{-1} = A^{-1}A = I_n$，其中 $I_n$ 是单位矩阵。逆矩阵存在的充要条件是 $\det(A) \neq 0$（矩阵非奇异）。

逆矩阵的计算意义：如果 $\mathbf{y} = A\mathbf{x}$，则 $\mathbf{x} = A^{-1}\mathbf{y}$。逆矩阵实现了线性变换的逆转。

对于 $2 \times 2$ 矩阵，逆矩阵有显式公式：
$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \quad
A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

**数值示例**：
$$
A = \begin{bmatrix} 2 & 3 \\ 1 & 2 \end{bmatrix}, \quad \det(A) = 2 \times 2 - 3 \times 1 = 1
$$

$$
A^{-1} = \frac{1}{1} \begin{bmatrix} 2 & -3 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 2 & -3 \\ -1 & 2 \end{bmatrix}
$$

验证：$AA^{-1} = \begin{bmatrix} 2 \times 2 + 3 \times (-1) & 2 \times (-3) + 3 \times 2 \\ 1 \times 2 + 2 \times (-1) & 1 \times (-3) + 2 \times 2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$

在神经网络中，逆矩阵出现较少，因为：
- 神经网络的变换通常是高维的（$n > 10^3$），求逆计算昂贵（$O(n^3)$）
- 权重矩阵经常接近奇异，数值不稳定
- 训练过程依赖梯度下降而非显式求逆（如最小二乘法 $W = (X^TX)^{-1}X^Ty$）

例外：白化操作 $Z = Λ^{-\frac{1}{2}}U^T X$（特征分解）、高斯过程、贝叶斯线性回归。

---

## NumPy 代码验证

```python
import numpy as np

# 向量操作
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# 内积：两种等价方式
dot_product1 = np.dot(u, v)          # 1*4 + 2*5 + 3*6 = 32
dot_product2 = u @ v                 # Python 3.5+ 运算符
assert dot_product1 == dot_product2 == 32

# L2 范数
l2_norm = np.linalg.norm(v)          # sqrt(16+25+36) ≈ 8.77496
assert np.isclose(l2_norm, 8.774964387392123)

# 余弦相似度
cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
# 32 / (sqrt(14) * sqrt(77)) ≈ 0.9746
assert np.isclose(cosine, 0.9746318461970762)

# 矩阵乘法
A = np.array([[1, 2, 3], [4, 5, 6]])      # 2×3
B = np.array([[7, 8], [9, 10], [11, 12]]) # 3×2
C = A @ B                                  # 2×2
assert np.array_equal(C, np.array([[58, 64], [139, 154]]))

# 批量矩阵乘法（batch 维度）
batch_A = np.random.randn(32, 10, 5)    # (batch=32, m=10, n=5)
batch_B = np.random.randn(32, 5, 8)     # (batch=32, n=5, p=8)
batch_C = batch_A @ batch_B             # (32, 10, 8)
assert batch_C.shape == (32, 10, 8)

# 转置
A_T = A.T
assert A_T.shape == (3, 2)
assert np.array_equal(A_T, np.array([[1, 4], [2, 5], [3, 6]]))

# 逆矩阵（仅限方阵）
M = np.array([[2, 3], [1, 2]])
M_inv = np.linalg.inv(M)
identity = M @ M_inv
assert np.allclose(identity, np.eye(2))

# 广播机制：神经网络前向传播的核心
# 模拟全连接层：y = Wx + b
batch_size, input_dim, hidden_dim = 16, 784, 256
W = np.random.randn(input_dim, hidden_dim)  # (784, 256)
b = np.random.randn(hidden_dim)              # (256,)
X = np.random.randn(batch_size, input_dim)  # (16, 784)

# X @ W: (16, 784) × (784, 256) = (16, 256)
# + b: (16, 256) + (256,) → b 广播为 (1, 256)，再广播到 (16, 256)
y = X @ W + b
assert y.shape == (16, 256)
```

---

## GPU 与矩阵运算的亲和性

GPU 专为大规模并行计算设计，其架构特征与矩阵乘法高度契合。

### SIMD（单指令多数据）并行

GPU 的每个 Streaming Multiprocessor（SM）包含数千个 CUDA 核心，能同时执行相同的算术指令于不同数据。矩阵乘法中的 $m \times p$ 次内积可完全并行化——每个输出元素 $C_{i,j}$ 由独立的线程计算，内部 $n$ 次乘加通过 warp-level 原语（如 `dp4a`、`mma.sync`）加速。

以 NVIDIA A100（10880 CUDA 核心为例）：
- 单精度浮点吞吐量：19.5 TFLOPS
- Tensor Core（FP16）：312 TFLOPS（混合精度矩阵乘法）
- 相比 CPU（Intel Xeon Platinum ~1.5 TFLOPS），加速比可达 20-200×

### 内存层次与数据复用

矩阵乘法的计算强度（flops/byte）为 $O(n^3 / n^2) = O(n)$。对于 $n=4096$ 的矩阵乘法，每个字节需执行约 4096 次运算，远超 CPU 缓存容量。GPU 通过以下机制缓解带宽瓶颈：

| 存储层级 | 容量 | 带宽 | 延迟 |
|---------|------|------|------|
| HBM2e | 40-80 GB | 1.5-2.0 TB/s | ~400 ns |
| L2 Cache | 40 MB | 3-4 TB/s | ~30 ns |
| Shared Memory | 128 KB (SM) | 30+ TB/s | ~5 ns |
| Register | 64 KB (warp) | 100+ TB/s | ~1 cycle |

cuBLAS 的分块算法（blocking）将大矩阵分解为适合 shared memory 的子块（如 64×64），大幅减少全局内存访问。这意味着矩阵乘法是 compute-bound 而非 memory-bound，GPU 的算力优势得以充分发挥。

### 工程细节：优化矩阵乘法的 trade-off

| 优化技术 | 优势 | 代价 | 适用场景 |
|---------|------|------|---------|
| 混合精度（FP16/BF16） | 吞吐量提升 4-8×，显存占用减半 | 数值精度下降，需损失缩放 | 推理、大模型训练 |
| 立方累积（Tensor Core） | 专用硬件单元，10× 加速 | 仅支持特定矩阵尺寸（8/16/32） | 深度学习框架底层 |
| Flash Attention | 优化注意力矩阵乘法的 I/O | 实现复杂度高，依赖内存布局 | Transformer 模型 |
| 量化（INT8） | 吞吐量提升 4×，显存占用减 3/4 | 需校准数据集，精度损失约 1-3% | 端侧部署 |

局限性：矩阵乘法加速受限于 batch size。当 $m$ 或 $p$ 过小（如 $m=1, p=1$ 的单样本推理），并行度不足，GPU 利用率可能低于 10%。此时 CPU 推理或动态批处理（dynamic batching）更高效。

---

## 小结

神经网络的基础运算归结为矩阵乘法与逐元素非线性变换。向量的内积、范数、余弦相似度构成了特征提取与匹配的核心算子；矩阵的乘法、转置实现了高维空间的线性映射；而 GPU 的 SIMD 架构与层次化内存设计，使大规模矩阵运算的并行效率达到 CPU 的数十倍。

工程实践中，理解每个操作的维度与复杂度是优化的前提。当计算瓶颈不在算力而在内存带宽时，分块计算、混合精度、算子融合等技术能显著提升性能；当 batch size 过小无法饱和 GPU 时，动态批处理或 CPU offload 是必要权衡。

**参考资料**

1. Strang, G. (2016). Introduction to Linear Algebra (5th ed.). Wellesley-Cambridge Press.
2. NVIDIA. (2022). CUDA C++ Programming Guide.
3. Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv:2205.14135.
