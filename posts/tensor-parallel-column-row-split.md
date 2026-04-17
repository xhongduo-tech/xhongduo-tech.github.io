## 核心结论

Tensor Parallel，直白说就是“把同一层里的矩阵乘法拆给多张卡一起算”。在 Megatron-LM 的 MLP 里，最常见也最重要的组合是：

1. 第一层全连接 `FC1` 用列并行 `ColumnParallelLinear`
2. 第二层全连接 `FC2` 用行并行 `RowParallelLinear`
3. 整个 MLP 前向只在 `FC2` 末尾做一次 `AllReduce`

这套组合的价值不在“能并行”本身，而在“通信次数最少”。`FC1` 按输出维切分后，每张卡只算自己负责的输出通道，本地就能完成前向；`FC2` 按输入维切分后，每张卡先算出一部分结果，最后把这些部分结果相加，得到完整输出。于是，一个两层 MLP 块只保留一次全局求和通信。

对零基础读者，可以先记一个最重要的图景：

| 层 | 切分方式 | 每张卡拿到什么 | 前向是否立刻通信 | 为什么 |
|---|---|---|---|---|
| `FC1` | 列并行 | 一部分输出通道 | 否 | 不同输出通道彼此独立 |
| 激活函数 | 本地执行 | 本地局部激活 | 否 | 激活逐元素计算 |
| `FC2` | 行并行 | 一部分输入通道 | 末尾需要一次 `AllReduce` | 各卡输出是“部分和”，要相加才完整 |

如果张量并行度为 $t$，小批量为 $b$，序列长度为 $s$，隐藏维度为 $h$，那么这一类 MLP 的核心通信量可写成：

$$
O\left(\frac{b \cdot s \cdot h}{t}\right)
$$

这里的意思不是“完全没有代价”，而是“每张卡只持有 $1/t$ 的隐藏分片，单次通信的数据量按分片后规模计算”。结论很直接：当带宽是瓶颈时，`Column -> Row` 的串联通常比“每层都多次同步”的实现更有效。

---

## 问题定义与边界

问题定义很明确：大模型里的全连接层太大，单卡放不下，或者单卡算得太慢，需要把一个线性层沿某个维度切到多张 GPU 上。

设输入张量形状是：

$$
X \in \mathbb{R}^{b \times s \times h}
$$

其中：

- $b$ 是 batch size，白话说就是“一次喂进去多少样本”
- $s$ 是 sequence length，白话说就是“每个样本有多少个 token”
- $h$ 是 hidden size，白话说就是“每个 token 的特征维度”
- $t$ 是 tensor parallel degree，白话说就是“同一层分给多少张卡一起算”

在线性层 $Y = XW + b$ 中，权重矩阵怎么切，决定了通信发生在什么位置。

### 列并行

列并行指按输出维切分权重。白话说：每张卡负责“生成一部分输出特征”。

若 $W \in \mathbb{R}^{h \times 4h}$，则可以按列切成：

$$
W = [W_1, W_2, ..., W_t]
$$

每张卡计算：

$$
Y_i = XW_i
$$

这里每个 $Y_i$ 只是完整输出的一部分，但它本身就是合法的局部输出，因此前向不用先通信。

### 行并行

行并行指按输入维切分权重。白话说：每张卡只吃输入特征的一部分，然后算出“部分贡献”。

若第二层权重 $V \in \mathbb{R}^{4h \times h}$，可按行切分为：

$$
V =
\begin{bmatrix}
V_1 \\
V_2 \\
\vdots \\
V_t
\end{bmatrix}
$$

输入激活也对应切成 $A_1, A_2, ..., A_t$，每张卡计算：

$$
Z_i = A_i V_i
$$

此时完整输出是：

$$
Z = \sum_{i=1}^{t} Z_i
$$

所以必须做一次全局求和，也就是 `AllReduce(sum)`。

### 边界在哪里

Tensor Parallel 不是“切完就万事大吉”。它的边界主要有三条：

| 边界 | 含义 | 直接影响 |
|---|---|---|
| 每层仍要同步 | 只是减少到最少，不是完全消除 | 层数越多，累计通信越多 |
| 带宽限制 | 跨节点 RDMA 往往慢于节点内 NVLink | TP 度过大时收益下降 |
| 算子耦合 | 只有按特定顺序组合，才能把通信压到块尾 | 随意切分会引入额外同步 |

一个新手可理解的版本是：把长度为 $h$ 的隐藏向量切成 $t$ 片，每片给一张卡。这确实减轻了单卡压力，但只要某一步需要“看到完整结果”，就必须让所有卡一起凑出全局张量。Megatron 的关键不是避免这件事，而是把这件事压缩到一层块的最末尾。

---

## 核心机制与推导

先看标准 MLP：

$$
\text{MLP}(X) = \phi(XW_1)W_2
$$

其中 $\phi$ 是激活函数，常见是 GeLU。白话说，激活函数就是“先做一次非线性变换，再送进下一层”。

在 Transformer 里，通常有：

- $W_1 \in \mathbb{R}^{h \times 4h}$
- $W_2 \in \mathbb{R}^{4h \times h}$

### 为什么 `FC1` 适合列并行

对 $W_1$ 做列切分后，每张卡得到：

$$
W_1^{(i)} \in \mathbb{R}^{h \times \frac{4h}{t}}
$$

于是本地输出为：

$$
A_i = \phi(XW_1^{(i)})
$$

因为每张卡负责的是不同输出通道，所以这些 $A_i$ 不需要先合并，就可以直接交给下一层。这里的关键点是：激活函数逐元素作用，不会打破这种局部性。

### 为什么 `FC2` 适合行并行

把 $W_2$ 按输入维切分：

$$
W_2^{(i)} \in \mathbb{R}^{\frac{4h}{t} \times h}
$$

每张卡拿到上一层自己的局部激活 $A_i$，本地计算：

$$
Z_i = A_i W_2^{(i)}
$$

每个 $Z_i$ 的形状都是 $b \times s \times h$，但它不是完整输出，而是完整输出中的一部分和，因此：

$$
Z = \sum_{i=1}^{t} Z_i
$$

这就是为什么前向只需在 `RowParallelLinear` 末尾做一次 `AllReduce(sum)`。

### 玩具例子：$b=2, s=1, h=4, t=2$

这是最适合入门的例子。

输入张量：

$$
X \in \mathbb{R}^{2 \times 1 \times 4}
$$

第一层扩到 $4h=16$ 太大，不利于演示，这里做等比例缩小：假设中间维是 4，最终输出维也是 4，只看“切分机制”而不看真实比例。

#### 第一步：Column Parallel

`FC1` 把输出维 4 切给两张卡，每张卡负责 2 个输出通道：

- GPU0 产生 $A_0 \in \mathbb{R}^{2 \times 1 \times 2}$
- GPU1 产生 $A_1 \in \mathbb{R}^{2 \times 1 \times 2}$

这里没有通信。

#### 第二步：Row Parallel

`FC2` 的输入维是 4，被切成两半，所以：

- GPU0 吃 $A_0$，算出 $Z_0 \in \mathbb{R}^{2 \times 1 \times 4}$
- GPU1 吃 $A_1$，算出 $Z_1 \in \mathbb{R}^{2 \times 1 \times 4}$

最后：

$$
Z = Z_0 + Z_1
$$

若只看“每张卡本地持有的输入激活分片”，元素数是：

$$
2 \times 1 \times 2 = 4
$$

两张卡合起来对应完整隐藏分片元素数：

$$
2 \times 1 \times 4 = 8
$$

这就是很多文章里提到的 `8 b s h` 这类直观例子背后的含义：通信不是围绕完整未切分大矩阵，而是围绕每卡持有的局部激活分片展开。

### 通信量怎么理解

设 `RowParallelLinear` 输入分片形状为：

$$
A_i \in \mathbb{R}^{b \times s \times \frac{h}{t}}
$$

则每张卡本地持有的激活规模是：

$$
b \cdot s \cdot \frac{h}{t}
$$

因此，按每卡视角，通信规模写成：

$$
O\left(\frac{b \cdot s \cdot h}{t}\right)
$$

如果从“整个并行组总共搬了多少数据”去看，还要乘上 collective 的实现系数；工程里常见 ring all-reduce，真实字节数会受到卡数和协议影响。但文章里通常关心的是“为什么 TP 切分后每卡通信压力下降”，此时用上面的表达式最清楚。

下面用一个表把推导压缩一下：

| 项 | 形状 | 每卡元素数 |
|---|---|---|
| 输入激活分片 $A_i$ | $b \times s \times h/t$ | $bsh/t$ |
| 本地部分输出 $Z_i$ | $b \times s \times h$ | $bsh$ |
| 触发同步的位置 | `FC2` 末尾 | 1 次 |
| 常用记法中的每卡通信规模 | 按输入分片估算 | $O(bsh/t)$ |

这里容易混淆的一点是：`Z_i` 的输出维度看起来已经是完整的 $h$，为什么还说通信受益于切分？原因是矩阵乘法的输入只吃了 $1/t$ 的通道，真正需要跨卡聚合的是这些局部贡献；工程上，Megatron 通过合理布局把同步压到一次 collective，而不是在中间反复 gather 和 reduce。

---

## 代码实现

下面给一个最小可运行的 Python 版本。它不是分布式框架代码，而是用普通 Python 列表模拟两张卡的 `Column -> Row` 流程，重点看数据怎么切、什么时候求和。

```python
from typing import List

def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    assert len(a[0]) == len(b)
    out = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            s = 0.0
            for k in range(len(b)):
                s += a[i][k] * b[k][j]
            row.append(s)
        out.append(row)
    return out

def add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    assert len(a) == len(b) and len(a[0]) == len(b[0])
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def relu(x: List[List[float]]) -> List[List[float]]:
    return [[max(0.0, v) for v in row] for row in x]

def slice_columns(w: List[List[float]], parts: int) -> List[List[List[float]]]:
    cols = len(w[0])
    assert cols % parts == 0
    step = cols // parts
    return [[[row[c] for c in range(i * step, (i + 1) * step)] for row in w] for i in range(parts)]

def slice_rows(w: List[List[float]], parts: int) -> List[List[List[float]]]:
    rows = len(w)
    assert rows % parts == 0
    step = rows // parts
    return [w[i * step:(i + 1) * step] for i in range(parts)]

def column_parallel_linear(x, w1_parts):
    return [matmul(x, w_part) for w_part in w1_parts]

def row_parallel_linear(local_inputs, w2_parts):
    partials = [matmul(local_inputs[i], w2_parts[i]) for i in range(len(local_inputs))]
    out = partials[0]
    for i in range(1, len(partials)):
        out = add(out, partials[i])  # 模拟 allreduce(sum)
    return out

# b=2, h=4, hidden_expand=4, t=2
x = [
    [1.0, 2.0, 3.0, 4.0],
    [0.5, 1.0, 1.5, 2.0],
]

w1 = [
    [1, 0, 2, 0],
    [0, 1, 0, 2],
    [1, 1, 1, 1],
    [2, 0, 0, 1],
]

w2 = [
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
]

w1_parts = slice_columns(w1, 2)   # Column Parallel: 按输出维切
w2_parts = slice_rows(w2, 2)      # Row Parallel: 按输入维切

local_a = column_parallel_linear(x, w1_parts)
local_a = [relu(part) for part in local_a]
y_tp = row_parallel_linear(local_a, w2_parts)

# 对照：不切分的普通 MLP
y_ref = matmul(relu(matmul(x, w1)), w2)

assert y_tp == y_ref
assert len(local_a) == 2
assert len(local_a[0][0]) == 2  # 每张卡只持有一半中间激活
print("tensor parallel result:", y_tp)
```

这段代码对应的逻辑就是：

```python
local_out = ColumnParallelLinear(input)   # 每张卡拿到局部输出通道
local_partial = RowParallelLinear(local_out)  # 每张卡只消费自己的输入分片
output = allreduce_sum(local_partial)     # 末尾一次求和得到完整输出
```

真实工程里不会手写 `allreduce_sum`，而是用 NCCL 或框架封装的 collective。Megatron-LM 的重点不是“有两个并行线性层”，而是“这两个线性层的切分方向正好互补”，所以能把中间张量一直保持为局部分片状态，直到必须汇总时才同步。

### 真实工程例子

以 4 张 A100 训练 7B 级 GPT 为例，MLP 和 Attention 都会触发 TP 通信，但 `Column -> Row` 这种设计把 MLP 内的通信压到块尾。工程上常见现象是：

- 单节点内 NVLink 带宽高，1 次 `AllReduce` 的代价可接受
- 如果改成“每经过一个并行线性层就同步一次”，延迟会明显上升
- 当 TP 继续扩大到跨节点时，RDMA 带宽跟不上，通信开始盖过计算收益

所以，很多实际系统会优先让 TP 组尽量留在单节点内，例如 4 卡机就设 `tp=4`，8 卡跨两节点时未必愿意直接上 `tp=8`。

---

## 工程权衡与常见坑

### 1. “只有一次 AllReduce”不等于“通信可以忽略”

正确表述是：一个 MLP 块内部只保留一次关键同步，而不是完全没有同步。Transformer 层很多，层层叠加后，通信仍然是总时延的重要组成部分。

### 2. TP 度不是越大越好

`tp=2`、`tp=4` 通常还能从显存和计算并行中获益；但如果 `tp=8` 需要跨节点，收益可能被通信抵消。因为这时不再只是“多几张卡”，而是“多经过一层更慢的网络”。

| 场景 | TP 度 | 典型互联 | 通信时间趋势 | 计算时间趋势 | 常见结论 |
|---|---|---|---|---|---|
| 单节点 4 卡 | 2 或 4 | NVLink | 较低 | 明显下降 | 通常划算 |
| 双节点 8 卡 | 8 | RDMA/InfiniBand | 明显上升 | 继续下降但收益变小 | 可能不划算 |
| 小模型 | 2 以上 | 任意 | 占比偏高 | 本来就不重 | 往往不该上 TP |

### 3. AllReduce 不重叠会形成“红绿灯效应”

所谓重叠，白话说就是“边算边传”。如果计算流和通信流完全串行，每层末尾都停下来等 `AllReduce`，就像每过一个路口都遇到红灯。层数一多，尾部延迟会堆积。

### 4. 切分方向一旦错配，就会引入额外通信

最经典的好组合是 `ColumnParallelLinear -> activation -> RowParallelLinear`。如果你把两层都做成列并行，或者都做成行并行，中间张量很可能需要额外 gather 或 reduce，通信次数上升，Megatron-LM 的优势就没了。

### 5. 偏置、Dropout、残差连接也要考虑布局

新手容易只盯着矩阵乘法，忽略后处理。实际上：

- Bias 是否分片持有
- Dropout 在通信前还是后
- Residual add 是否要求完整张量

这些都决定中间张量能否继续保持“局部状态”。工程中，很多性能退化不是线性层本身的问题，而是线性层后面接的算子打破了并行布局。

---

## 替代方案与适用边界

Tensor Parallel 不是唯一方案。重点不是“谁绝对更好”，而是“哪种通信模式更适合当前硬件和模型”。

| 方案 | 核心做法 | 通信特点 | 适用边界 |
|---|---|---|---|
| `Column -> Row` MLP | 第一层列并行，第二层行并行 | 块内尽量只保留一次 `AllReduce` | 大模型、单节点高带宽、Megatron 风格实现 |
| 纯 Row 或纯 Column | 两层都用同一类切分 | 中间常需额外同步 | 教学验证、小规模实验 |
| 每层都显式 Gather/Reduce | 简化实现 | 通信频次高 | TP 度小、网络很强、代码简单优先 |
| ZeRO | 参数、梯度、优化器状态分片 | 更关注训练状态切分，不直接替代层内并行 | 显存压力大，但单层计算还能放下 |
| ZeRO + TP | 状态分片和张量并行混合 | 通信路径更复杂，但资源利用更高 | 超大模型训练 |
| Pipeline Parallel | 按层切到不同设备 | 跨 stage 传激活，不是层内切分 | 模型层数多、单层太大或节点数多 |

### 什么时候优先选 `Column -> Row`

适合以下场景：

- 单层矩阵很大，单卡显存吃紧
- 节点内带宽高，希望把 TP 通信限制在节点内
- 使用 Megatron-LM 一类已经把并行布局打磨好的框架

### 什么时候不一定选它

以下场景要谨慎：

- 模型不大，单卡能轻松放下
- 集群是弱互联，跨节点带宽一般
- 团队更在意实现简单，而不是极致吞吐
- 训练系统已深度依赖 ZeRO/FSDP，TP 只是补充而非核心

一个新手可理解的比较是：

- “每层都 AllReduce”像每走一步都汇报一次进度，逻辑简单，但很慢
- “Column -> Row 只在块尾 AllReduce”像每个人先把自己那部分工作做完，最后统一对账一次，通信更省

---

## 参考资料

- DeepWiki: Megatron-LM Tensor Parallelism 3.1.1  
  用途：给出 `ColumnParallelLinear` 与 `RowParallelLinear` 的整体设计和通信位置。

- Benathi Blog: Tensor Parallelism  
  用途：解释为什么列并行和行并行可以互补，适合建立直觉。

- Hiascend 文档：Tensor Parallel 通信建模  
  用途：给出通信量表达式，帮助理解 $O(bsh/t)$ 这类估算。

- Dev Community: Tensor Parallelism by Hand  
  用途：提供小尺寸数值例子，适合手算和验证。

- Megatron-LM 相关工程文章与社区复盘  
  用途：补充真实训练中的带宽、延迟、TP 度选择等工程问题。
