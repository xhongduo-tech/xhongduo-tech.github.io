## 核心结论

Tensor Parallel，简称 TP，本质上是把**同一层**的参数和计算拆到多张 GPU 上，同时要求前向输出和反向梯度都与单卡实现严格一致。对新手来说，可以先把它理解成一句话：

**不是“一层交给一张卡算完”，而是“同一层由多张卡各算一部分，再用通信把数值语义补完整”。**

最容易混淆的一点是：**Megatron 论文里的 `f/g` 共轭通信原语，描述的是一个“列并行 + 行并行”成对出现的计算块，而不是孤立的 `ColumnParallelLinear` 单层全部通信细节。** 论文给出的结论是：

$$
f_{\text{fwd}}(X)=X,\qquad f_{\text{bwd}}(dY)=\operatorname{AllReduce}(dY)
$$

$$
g_{\text{fwd}}(Y)=\operatorname{AllReduce}(Y),\qquad g_{\text{bwd}}(dZ)=dZ
$$

把这两行式子翻成工程语言，就是下表：

| 位置 | 前向 | 反向 | 数学含义 | 典型位置 |
|---|---|---|---|---|
| `f` | Identity | AllReduce | 前向先不合并，反向把多条分支对同一输入的梯度求和 | ColumnParallelLinear 的输入侧 |
| `g` | AllReduce | Identity | 前向把多个局部部分和相加成完整输出，反向梯度直接复制到各分支 | RowParallelLinear 的输出侧 |

但如果只看**单独一个列并行线性层**，结论要更细：

1. 权重按输出维切分后，每张卡先得到自己的局部输出分片。
2. 如果下一层本来就接受这种分片布局，前向可以**完全不通信**。
3. 如果下一层要求每张卡都拿到完整激活，前向做的是 **AllGather**。
4. 这时它在反向的共轭操作通常是 **Split** 或 **ReduceScatter**，而不是论文里那种 `g=AllReduce`。

所以准确说法不是“Column Parallel 前向一定 AllReduce”，而是：

- **孤立的 `ColumnParallelLinear`**：常见前向通信是“无通信”或 `AllGather`。
- **Megatron 的 MLP / Attention 成对 TP 块**：第一层列并行先本地算，第二层行并行在前向末尾做一次 `g=AllReduce`；反向则在对应位置做一次 `f=AllReduce`。

可以先记住最稳妥的一条判断规则：

| 你手里的局部结果是什么 | 下一步需要什么 | 常见通信 |
|---|---|---|
| 完整结果的不同片段 | 拼成完整张量 | AllGather / Split |
| 同一结果的不同部分和 | 求出总和 | AllReduce / ReduceScatter |

---

## 问题定义与边界

问题本质是：**当一层权重被切到多张卡后，前向怎样保证输出和单卡一致，反向怎样保证梯度和单卡一致。**

本文只讨论 Transformer 里最常见的 TP 场景，即线性层：

$$
Y = XA
$$

其中：

- $X \in \mathbb{R}^{n\times h}$：输入激活，这里把 batch 维和 sequence 维先合并成一个大维度 $n$
- $A \in \mathbb{R}^{h\times m}$：权重矩阵
- $Y \in \mathbb{R}^{n\times m}$：输出激活

这里的“边界”很重要。因为 TP 里常见的通信错误，不是 shape 对不上，而是**数学语义对不上**。张量形状能跑通，不代表数值就等价于单卡。

### 列并行的边界

“列并行”是把矩阵按输出维切开。设：

$$
A=[A_0, A_1, \dots, A_{p-1}],\qquad A_i\in\mathbb{R}^{h\times m_i}
$$

于是：

$$
Y = XA = [XA_0, XA_1, \dots, XA_{p-1}]
$$

第 $i$ 张卡只保存 $A_i$，因此本地能算出：

$$
Y_i = XA_i
$$

注意这里的 $Y_i$ 不是“错误的近似”，而是**完整输出的一个合法列分片**。因此边界立刻出现：

- 如果后续算子只需要自己的 $Y_i$，就不必通信。
- 如果后续算子要完整 $Y=[Y_0,\dots,Y_{p-1}]$，就必须做拼接通信。
- 如果后续算子要的不是拼接，而是所有分支的和，那就不是 Gather，而是 Reduce。

### 行并行的边界

“行并行”是把矩阵按输入维切开。设：

$$
A=
\begin{bmatrix}
A_0 \\
A_1 \\
\vdots \\
A_{p-1}
\end{bmatrix},\qquad A_i\in\mathbb{R}^{h_i\times m}
$$

同时输入也按最后一维切成：

$$
X=[X_0, X_1, \dots, X_{p-1}],\qquad X_i\in\mathbb{R}^{n\times h_i}
$$

则有：

$$
Y = XA = \sum_{i=0}^{p-1} X_iA_i
$$

第 $i$ 张卡本地只能算出：

$$
Y_i^{\text{partial}} = X_iA_i
$$

这里最关键的区别是：这些 $Y_i^{\text{partial}}$ 不是不同片段，而是**同一个输出张量的不同部分和**。所以必须求和，不能拼接。

### 两类通信语义

这就是 TP 中最常见的两类通信语义：

| 场景 | 通信前每卡持有 | 通信原语 | 通信后每卡持有 | 典型位置 |
|---|---|---|---|---|
| 分片拼接 | 完整输出的一个片段 | AllGather | 完整拼接结果 | 单独的 ColumnParallelLinear 输出暴露给外部 |
| 部分和求总和 | 同一输出的一个部分和 | AllReduce | 完整求和结果 | RowParallelLinear 输出 |

把这个区别说清楚，比死记任何单层结论都更重要。

### 一个两卡玩具例子

设 `tp_size=2`，两张卡各存一半参数。

如果是列并行：

- GPU0 算出 $Y_0=XA_0$
- GPU1 算出 $Y_1=XA_1$

如果后续层接受分片输入，那么直接把 $Y_0$ 喂给下一层的第 0 片、把 $Y_1$ 喂给下一层的第 1 片即可，不需要通信。

如果外部接口要求每张卡都拿到完整激活，就要做：

$$
Y = \operatorname{AllGather}(Y_0, Y_1)
$$

如果是行并行：

- GPU0 算出局部部分和 $Z_0=X_0B_0$
- GPU1 算出局部部分和 $Z_1=X_1B_1$

完整输出是：

$$
Z = Z_0 + Z_1
$$

因此要做：

$$
Z = \operatorname{AllReduce}(Z_i)
$$

### 为什么漏通信会直接错

如果漏掉该有的通信，后果通常不是“性能稍差”，而是**数学定义已经变了**：

| 漏掉的通信 | 实际出错点 | 结果 |
|---|---|---|
| 列并行后应 `AllGather` 却没做 | 下一层以为拿到完整激活，实际上只拿到局部列块 | 前向值错误 |
| 行并行后应 `AllReduce` 却没做 | 输出少了其他分支的部分和 | 前向值错误 |
| 共享输入分支反向应求和却没做 | 每张卡只看到自己的局部输入梯度 | 反向值错误 |
| 布局变了但没按原分片规则切回 | 梯度 shape 可能能对上，但语义不对应 | 训练不稳定或收敛到错误解 |

因此，TP 的核心不是“把张量分了”，而是“**分片布局、通信语义、后续算子预期**三者必须一致”。

---

## 核心机制与推导

先看 Megatron 论文里最经典的两层 MLP 块。设第一层列并行，第二层行并行：

$$
H = \phi(XA),\qquad Z = HB
$$

其中：

$$
A=[A_0, A_1, \dots, A_{p-1}]
$$

$$
B=
\begin{bmatrix}
B_0 \\
B_1 \\
\vdots \\
B_{p-1}
\end{bmatrix}
$$

并满足：

- $A_i\in\mathbb{R}^{h\times 4h/p}$
- $B_i\in\mathbb{R}^{4h/p\times h}$

这里 $\phi$ 是逐元素非线性，例如 GeLU。所谓“逐元素”，意思是每个位置独立做函数映射，不需要把不同卡上的值混在一起。

### 1. 第一层列并行为什么前向不通信

由列拼接的分配律：

$$
XA = [XA_0, XA_1, \dots, XA_{p-1}]
$$

于是每张卡本地得到：

$$
H_i = \phi(XA_i)
$$

之所以这一步不需要通信，是因为：

1. 每张卡都有完整输入 $X$
2. 自己的权重列块 $A_i$ 足以生成自己的输出列块
3. 激活函数 $\phi$ 是逐元素的，不要求跨卡聚合

所以在第一层之后，系统中保存的是一个**按特征维切分的激活布局**：

$$
H = [H_0, H_1, \dots, H_{p-1}]
$$

这一步最容易让新手误解成“输出不完整”。准确说法是：**对整个系统来说输出是完整的，只是它被分散保存在多张卡上。**

### 2. 第二层行并行为什么前向必须求和

第二层权重按输入维切分，因此每张卡消费对应的 $H_i$，本地计算：

$$
Z_i = H_iB_i
$$

注意每个 $Z_i$ 的形状都是：

$$
Z_i\in\mathbb{R}^{n\times h}
$$

它们不是不同列块，而是对同一个最终输出 $Z$ 的局部贡献。因此：

$$
Z = \sum_{i=0}^{p-1} Z_i
$$

工程上通常用 `AllReduce(sum)` 完成这一步：

$$
Z = \operatorname{AllReduce}(Z_i)
$$

这就是论文里的 `g`：

$$
g_{\text{fwd}}(Y)=\operatorname{AllReduce}(Y)
$$

所以 `g` 前向做的不是拼接，而是**把局部部分和还原成完整输出**。

### 3. 为什么 `g` 的反向是 Identity

设损失为 $\mathcal{L}(Z)$。因为：

$$
Z = \sum_{i=0}^{p-1} Z_i
$$

所以对任意 $i$ 都有：

$$
\frac{\partial \mathcal{L}}{\partial Z_i}
=
\frac{\partial \mathcal{L}}{\partial Z}
\cdot
\frac{\partial Z}{\partial Z_i}
=
\frac{\partial \mathcal{L}}{\partial Z}
$$

记上游梯度为 $dZ$，则：

$$
dZ_i = dZ
$$

这就是 `g` 的反向是 Identity 的原因。它不是说“完全没有通信成本”，而是说**从数学伴随算子看，不需要再做一个额外的求和操作**；每个分支都直接收到同一个完整上游梯度即可。

### 4. 第一层反向为什么需要 `f=AllReduce`

从第二层反向开始，每张卡本地得到：

$$
dH_i = dZ B_i^\top
$$

再继续回传到输入 $X$：

$$
dX_i = dH_i A_i^\top
$$

但因为第一层列并行时，所有分支共享同一个输入 $X$，所以完整输入梯度应该是所有分支贡献之和：

$$
dX = \sum_{i=0}^{p-1} dX_i
$$

因此需要：

$$
dX = \operatorname{AllReduce}(dX_i)
$$

这就是论文里的 `f` 反向：

$$
f_{\text{bwd}}(dY)=\operatorname{AllReduce}(dY)
$$

而它的前向是 Identity：

$$
f_{\text{fwd}}(X)=X
$$

原因也很直接：前向到这个位置时，系统还不需要把输入做额外聚合；真正需要求和的是反向，因为多条并行路径都对同一个输入有贡献。

### 5. `f/g` 本质上是一对伴随通信

把前向和反向连起来看，Megatron 的 `f/g` 可以理解成：

| 原语 | 前向职责 | 反向职责 | 解决的问题 |
|---|---|---|---|
| `f` | 保持共享输入原样进入并行分支 | 把并行分支对共享输入的梯度求和 | 恢复正确的输入梯度 |
| `g` | 把多个局部部分和还原为完整输出 | 把上游梯度直接分发回各分支 | 恢复正确的模块输出 |

所以 `f/g` 不是“哪一层固定用哪个通信”，而是**成对出现的数学语义**。

### 6. 为什么 AllGather/Split 不是论文里的 `g`

需要特别补清一个边界：**AllGather / Split 也是一组前后向共轭通信，但它不是 Megatron 论文图里那组 `f/g = Identity / AllReduce`。**

例如，单独的 `ColumnParallelLinear` 若要求每张卡都拿到完整输出：

$$
Y = \operatorname{AllGather}(Y_0, Y_1, \dots, Y_{p-1})
$$

这里前向做的是拼接，而不是求和。

如果上游给出完整梯度 $dY$，反向通常要把它按原来的列切分规则切回去：

$$
(dY_0, dY_1, \dots, dY_{p-1}) = \operatorname{Split}(dY)
$$

如果这个切回过程顺带还承担梯度归约，工程上也可能实现成 `ReduceScatter`。它和论文里的 `g=AllReduce` 解决的是**不同类型的张量关系**：

| 情况 | 各卡局部结果的关系 | 正确通信 |
|---|---|---|
| 列并行输出 | 不同片段 | AllGather |
| 行并行输出 | 同一张量的部分和 | AllReduce |

把这两种情况混在一起，就是很多 TP 入门文章最容易说错的地方。

### 7. Attention 里的同类结构

Attention 中也有类似模式。例如 QKV 投影常做列并行，输出投影常做行并行。其原因和 MLP 一样：

- 前面的列并行把更宽的中间特征分散到各卡
- 中间若是逐头独立或逐元素计算，可以保留这种分片布局
- 最后的输出投影再通过行并行把多个局部贡献求和还原成完整隐藏状态

所以论文里常说“每个 Transformer 层前向两次 AllReduce，反向两次 AllReduce”，本质上说的是**Attention 子块和 MLP 子块各有一组 `f/g`**，而不是说每个 `ColumnParallelLinear` 自己一定有一次前向 AllReduce。

---

## 代码实现

下面给出一个**最小可运行**的 `python` 例子。它做两件事：

1. 验证**孤立的 `ColumnParallelLinear`** 在“保持分片输出”和“收集完整输出”两种模式下都与单卡数值一致。
2. 验证 **Megatron 式“第一层列并行 + 第二层行并行”** 的前向和反向与单卡严格等价。

代码只依赖 `numpy`，不依赖 `torch.distributed`。通信原语用普通数组操作模拟，因此可以直接本地运行看清楚数学关系。

```python
import numpy as np

rng = np.random.default_rng(0)


def split_last_dim(x, parts):
    """把最后一维均匀切成 parts 段。"""
    assert x.shape[-1] % parts == 0
    chunk = x.shape[-1] // parts
    return [x[..., i * chunk:(i + 1) * chunk] for i in range(parts)]


def all_gather_concat(shards, axis=-1):
    """模拟 AllGather 后按特征维拼接。"""
    return np.concatenate(shards, axis=axis)


def all_reduce_sum(partials):
    """模拟 AllReduce(sum)。"""
    out = np.zeros_like(partials[0])
    for x in partials:
        out = out + x
    return out


def assert_close(name, a, b, atol=1e-10):
    if not np.allclose(a, b, atol=atol):
        raise AssertionError(f"{name} mismatch\n{a}\n!=\n{b}")


# ------------------------------------------------------------
# 例 1: 孤立的 ColumnParallelLinear
# ------------------------------------------------------------
# X: [batch, hidden]
X = rng.normal(size=(2, 4))

# W 按输出维切分，W = [W0, W1]
W0 = rng.normal(size=(4, 3))
W1 = rng.normal(size=(4, 3))
W = np.concatenate([W0, W1], axis=1)

# 单卡结果
Y_dense = X @ W  # [2, 6]

# 多卡局部结果
Y0 = X @ W0      # [2, 3]
Y1 = X @ W1      # [2, 3]

# 方案 A: 输出保持分片布局，不通信
Y_shards = [Y0, Y1]
assert_close("column_parallel_shard_0", Y_dense[:, :3], Y_shards[0])
assert_close("column_parallel_shard_1", Y_dense[:, 3:], Y_shards[1])

# 方案 B: 输出需要完整张量，做 AllGather
Y_gather = all_gather_concat([Y0, Y1], axis=1)
assert_close("column_parallel_allgather_forward", Y_dense, Y_gather)

# 若上游给出完整梯度 dY，则反向需要按原规则切回去
dY = rng.normal(size=Y_dense.shape)
dY0, dY1 = split_last_dim(dY, parts=2)
assert_close("column_parallel_split_backward_0", dY[:, :3], dY0)
assert_close("column_parallel_split_backward_1", dY[:, 3:], dY1)

# 每张卡本地都可计算自己的权重梯度
dW0 = X.T @ dY0
dW1 = X.T @ dY1
dW_dense = X.T @ dY
assert_close("column_parallel_wgrad_0", dW_dense[:, :3], dW0)
assert_close("column_parallel_wgrad_1", dW_dense[:, 3:], dW1)

# 输入梯度需要把各分支贡献求和，这对应 f_backward
dX0 = dY0 @ W0.T
dX1 = dY1 @ W1.T
dX_tp = all_reduce_sum([dX0, dX1])
dX_dense = dY @ W.T
assert_close("column_parallel_dgrad_allreduce", dX_dense, dX_tp)

print("Example 1 passed: isolated ColumnParallelLinear is correct.")


# ------------------------------------------------------------
# 例 2: Megatron 式 MLP 块
# 第一层列并行，第二层行并行
# ------------------------------------------------------------
A0 = rng.normal(size=(4, 3))
A1 = rng.normal(size=(4, 3))
A = np.concatenate([A0, A1], axis=1)   # [4, 6]

B0 = rng.normal(size=(3, 5))
B1 = rng.normal(size=(3, 5))
B = np.concatenate([B0, B1], axis=0)   # [6, 5]

# 单卡前向
H_dense = X @ A
Z_dense = H_dense @ B

# TP 前向
# 第一层 Column Parallel: 各卡得到自己的激活分片
H0 = X @ A0
H1 = X @ A1

# 第二层 Row Parallel: 各卡计算局部部分和
Z0 = H0 @ B0
Z1 = H1 @ B1

# g_forward = AllReduce(sum)
Z_tp = all_reduce_sum([Z0, Z1])
assert_close("mlp_forward", Z_dense, Z_tp)

# 单卡反向
dZ = rng.normal(size=Z_dense.shape)
dH_dense = dZ @ B.T
dX_dense = dH_dense @ A.T

# TP 反向
# g_backward = Identity
dH0 = dZ @ B0.T
dH1 = dZ @ B1.T

# 各卡本地计算参数梯度
dA0 = X.T @ dH0
dA1 = X.T @ dH1
dB0 = H0.T @ dZ
dB1 = H1.T @ dZ

dA_dense = X.T @ dH_dense
dB_dense = H_dense.T @ dZ

assert_close("mlp_dA0", dA_dense[:, :3], dA0)
assert_close("mlp_dA1", dA_dense[:, 3:], dA1)
assert_close("mlp_dB0", dB_dense[:3, :], dB0)
assert_close("mlp_dB1", dB_dense[3:, :], dB1)

# 回传到共享输入 X 时，需要 f_backward = AllReduce(sum)
dX0 = dH0 @ A0.T
dX1 = dH1 @ A1.T
dX_tp = all_reduce_sum([dX0, dX1])

assert_close("mlp_backward", dX_dense, dX_tp)

print("Example 2 passed: Megatron TP block is numerically equivalent.")
```

这段代码可以直接运行，输出应为：

```text
Example 1 passed: isolated ColumnParallelLinear is correct.
Example 2 passed: Megatron TP block is numerically equivalent.
```

代码里对应的通信位置如下：

| 例子 | 阶段 | 本地计算 | 通信原语 | 作用 |
|---|---|---|---|---|
| `ColumnParallelLinear` | 前向 | `Y_i = X @ W_i` | 无 或 `AllGather` | 输出可保留分片，也可拼回完整张量 |
| `ColumnParallelLinear` | 反向 | `dX_i = dY_i @ W_i.T` | `AllReduce` | 恢复共享输入的完整梯度 |
| `Column + Row` 块 | 第一层前向 | `H_i = X @ A_i` | 无 | 列并行后仍可独立算 |
| `Column + Row` 块 | 第二层前向 | `Z_i = H_i @ B_i` | `g_forward = AllReduce` | 把部分和还原成完整输出 |
| `Column + Row` 块 | 第二层反向 | `dH_i = dZ @ B_i.T` | 无 | `g_backward = Identity` |
| `Column + Row` 块 | 第一层反向 | `dX_i = dH_i @ A_i.T` | `f_backward = AllReduce` | 恢复共享输入的完整梯度 |

### 和真实工程实现的对应关系

在 Transformer 的 FFN 中，工程实现通常是：

1. `fc1` 用 `ColumnParallelLinear`，把 hidden 从 $h$ 扩到 $4h$
2. GeLU 在每张卡本地做
3. `fc2` 用 `RowParallelLinear`，把 $4h$ 收回 $h$
4. `fc2` 前向末尾做一次 `AllReduce`，这对应 `g`
5. 反向回到 `fc1` 输入梯度时再做一次 `AllReduce`，这对应 `f`

Megatron Core 文档里还能看到更进一步的工程化参数，例如：

- `gather_output`
- `input_is_parallel`
- `disable_grad_reduce`
- `allreduce_dgrad`

这些参数控制的都不是新的数学，而是**同一数学语义在不同模块边界如何落地**。

---

## 工程权衡与常见坑

工程里真正难的不是“知道要通信”，而是“知道该在哪一层、哪个方向、用哪种通信”。

### 常见坑

| 坑 | 具体误解 | 后果 | 规避方式 |
|---|---|---|---|
| 把论文里的 `g=AllReduce` 套到所有 `ColumnParallelLinear` | 以为列并行输出天然需要求和 | 多做一次无意义通信，甚至直接做错 | 先判断局部结果是“片段”还是“部分和” |
| 该 `AllGather` 时没做 | 以为 shape 对得上就说明语义对 | 下一层拿到残缺激活 | 明确模块边界要求的是分片布局还是完整布局 |
| 该 `f_backward=AllReduce` 时漏掉 | 以为每卡本地算完输入梯度就结束了 | 梯度缺失，训练不收敛 | 只要多个并行分支共享同一输入，反向就要把贡献求和 |
| 把 `AllGather` 和 `AllReduce` 混为一谈 | 忽略“拼接”和“求和”是不同操作 | 前向值直接错误 | 先画出张量关系，再选通信原语 |
| 只看单层，不看相邻层布局 | 以为每层都该输出完整激活 | 产生额外通信 | 把两层或整个子块作为分析单位 |
| 开了 sequence parallel 还按普通 TP 理解通信 | 忽略激活在序列维上的切分 | shape 和通信路径都变复杂 | 单独检查 `AllGather/ReduceScatter` 路径 |

### 一个很常见的新手问题

很多人会问：“反向最后统一 AllReduce 一次不就行了吗？”

问题在于，反向不是一个“整图结束再一次性提交”的过程，而是**逐层产生梯度、逐层继续回传**。某一层的输入梯度一旦算出来，这个梯度对应的通信就已经具备发起条件。

如果你把通信全部拖到反向末尾，通常会出现两件事：

1. 前面 GPU 在算，网络链路闲着
2. 最后所有卡一起等通信，形成明显尾部气泡

因此工程上更好的做法是：**`dgrad` 一旦可用，就尽早发起异步通信，同时继续计算 `wgrad`。**

可以用下表理解：

| 方案 | 时序特点 | 结果 |
|---|---|---|
| 差的方案 | 先把整层反向都算完，再统一发起通信 | 通信和计算几乎串行，GPU 等网络 |
| 更好的方案 | `dgrad` 一出来就异步 AllReduce / ReduceScatter，同时继续算 `wgrad` | 计算和通信重叠，吞吐更高 |

这就是 Megatron Core 中类似 `linear_with_grad_accumulation_and_async_allreduce` 这类实现要解决的问题。它优化的是**等待关系**，不是数学定义。

### 通信量直觉

以环形 AllReduce 为例，若一次通信张量大小是 $N$ 字节、TP 组大小为 $p$，则总通信量近似为：

$$
2N\cdot \frac{p-1}{p}
$$

如果张量形状是 $[b,s,h]$，元素字节数为 $q$，则：

$$
N = bshq
$$

因此一次 AllReduce 的通信量近似是：

$$
2bshq\cdot \frac{p-1}{p}
$$

这个式子至少给出一个实用直觉：**TP 的主通信成本更接近“模块边界激活大小”而不是“参数大小”**。所以隐藏维、batch、序列长度一大，通信很容易成为瓶颈。

---

## 替代方案与适用边界

不要把 `Column Parallel` 当成 TP 的全部。Transformer 里常见的是**列并行和行并行配对使用**，因为它们在前向和反向的通信位置上天然互补。

| 方案 | 前向通信 | 反向通信 | 适用条件 | 典型接口 |
|---|---|---|---|---|
| `ColumnParallelLinear` 单层，输出保持分片 | 无 | 共享输入梯度通常要 AllReduce | 下一层接受分片布局 | MLP 第一层、QKV 投影 |
| `ColumnParallelLinear` 单层，输出要完整 | AllGather | Split 或 ReduceScatter，再加上输入梯度求和路径 | 外部接口要求完整激活 | 独立暴露的线性层 |
| `RowParallelLinear` 单层 | AllReduce 求和输出 | 输入梯度按分片本地可算，其他通信视上游布局而定 | 输入已经按特征维分片 | MLP 第二层、输出投影 |
| Megatron 式 `Column + Row` 成对块 | 块末尾一次 `g=AllReduce` | 块入口对应一次 `f=AllReduce` | 相邻线性层切分方向可互补 | FFN、Attention 子块 |

可以把决策压缩成两句：

- **如果多张卡产出的是“完整结果的不同片段”，用 Gather / Scatter。**
- **如果多张卡产出的是“同一结果的不同部分和”，用 Reduce。**

### 真实工程里的两个边界

#### 1. TP 不是越大越好

如果网络带宽已经成为瓶颈，继续增大 TP size 往往收益递减，甚至变差。原因很简单：

- TP 变大后，每张卡算得更少
- 但跨卡通信更频繁，等待占比更高

因此大模型训练通常需要与其他并行方式一起设计：

| 并行方式 | 主要解决什么问题 | 和 TP 的关系 |
|---|---|---|
| Data Parallel | 扩大全局 batch、复制参数训练 | 与 TP 正交，常一起使用 |
| Pipeline Parallel | 拆层，降低单卡参数驻留 | 减少单阶段显存和算力压力 |
| Sequence Parallel | 降低非 TP 激活显存 | 经常和 TP 绑定使用 |
| Expert Parallel | 在 MoE 中拆专家参数和路由 | 与 TP 的通信热点不同 |

#### 2. Sequence Parallel 会改变激活通信路径

如果启用了 Sequence Parallel，很多激活布局会从“每卡持有完整序列”变成“每卡持有部分序列”。这时常见路径会变成：

- 前向：`AllGather`
- 反向：`ReduceScatter`

这和基础 TP 里的 `f/g` 不是一回事，不能直接把论文里那组公式硬套过来。

### 最稳妥的判断方法

遇到具体实现时，最稳妥的方式不是背结论，而是连续问两个问题：

1. 这一步每张卡拿到的是“分片”还是“部分和”？
2. 下一步需要的是“完整拼接结果”还是“完整求和结果”？

如果这两个问题答清楚了，通信原语基本就不会选错。

---

## 参考资料

1. NVIDIA ADLR, *MegatronLM: Training Billion+ Parameter Language Models Using GPU Model Parallelism*  
   Megatron 项目主页，包含原始论文入口和 Figure 2 的经典通信示意图。适合先看整体结构，再对照 `f/g` 在 MLP 与 Attention 中的位置。  
   https://research.nvidia.com/labs/adlr/MegatronLM/

2. Mohammad Shoeybi, Mostofa Patwary, Raul Puri, et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*  
   TP 的经典论文。重点看第 3 节对 MLP 与 self-attention 的并行化方式，以及 Figure 2 中 `f/g` 的共轭定义。  
   https://arxiv.org/abs/1909.08053

3. NVIDIA Megatron Core Documentation, `core.tensor_parallel.layers.ColumnParallelLinear` / `RowParallelLinear`  
   工程实现最直接的参考。重点关注 `gather_output`、`input_is_parallel`、`skip_bias_add`、`disable_grad_reduce` 等参数如何改变模块边界上的通信行为。  
   https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html

4. NVIDIA Megatron Core Documentation, Tensor Parallel API Guide  
   适合查看 `linear_with_grad_accumulation_and_async_allreduce`、`allreduce_dgrad` 等工程优化接口。这里能看到“数学不变，但通信与计算可以异步重叠”的具体落点。  
   https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html

5. DeepSpeed Documentation, Megatron-LM Integration  
   适合补工程视角，尤其是 TP 与 PP、DP、ZeRO 组合时的边界。  
   https://www.deepspeed.ai/tutorials/megatron/

6. Microsoft Research, *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*  
   适合在理解基础 TP 后继续看更大规模集群中的性能权衡、并行组合与吞吐瓶颈。  
   https://www.microsoft.com/en-us/research/publication/efficient-large-scale-language-model-training-on-gpu-clusters/
