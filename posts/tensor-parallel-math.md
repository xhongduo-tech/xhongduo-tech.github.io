## 核心结论

Tensor Parallel，简称 TP，本质上是在保持矩阵乘法结果不变的前提下，把线性层的权重矩阵拆到多张 GPU 上并行计算。它不是“随便把模型切开”，而是针对大规模线性层的严格数学分解。

最核心的数学事实有两个：

1. 按列切分权重时，设
   $$
   A = [A_1, A_2, \dots, A_n], \quad A_i \in \mathbb{R}^{h \times m_i}
   $$
   则
   $$
   Y = XA = [XA_1, XA_2, \dots, XA_n]
   $$
   每张卡只需计算自己的局部输出
   $$
   Y_i = XA_i
   $$
   前向阶段不需要求和通信。因为列切分后的结果天然也是按列拼接。

2. 按行切分权重时，设
   $$
   A = \begin{bmatrix}
   A_1 \\
   A_2 \\
   \vdots \\
   A_n
   \end{bmatrix}, \quad A_i \in \mathbb{R}^{h_i \times m}
   $$
   同时把输入按最后一维切成
   $$
   X = [X_1, X_2, \dots, X_n], \quad X_i \in \mathbb{R}^{(bs)\times h_i}
   $$
   则
   $$
   Y = XA = \sum_{i=1}^n X_i A_i
   $$
   每张卡只能得到一个局部部分和
   $$
   Y_i^{\text{partial}} = X_i A_i
   $$
   最后必须做一次 AllReduce，或者等价的求和聚合，才能得到完整输出。

在 Transformer 的 MLP 中，业界常见做法是第一层列并行、第二层行并行。原因很直接：第一层扩维后的各个分片，正好可以直接喂给第二层的行切分权重，中间不必先把完整激活拼回去。

因此，一个标准 MLP 模块的前向主通信通常只发生在第二层末尾；反向再发生一次与之对应的主通信。所以常见估算里，一个 MLP 模块前向加反向有两次主要集合通信。

若把一次 AllReduce 的张量大小记为 $b \times s \times h$，并按 FP32 的 4 字节估算，则一次环形 AllReduce 的通信量近似为：

$$
4bsh \times 2 \times \frac{n-1}{n}
= 8bsh \times \frac{n-1}{n}
$$

其中：

- $b$ 是 batch size
- $s$ 是序列长度
- $h$ 是隐藏维度
- $n$ 是 TP 组内 GPU 数

这个式子对应的是**一次** AllReduce。若按一个 Column-then-Row MLP 的前向加反向整体估算，主通信量通常约为它的两倍。关键结论不变：TP 的主通信量主要由模块边界上的激活张量大小决定，而不是由中间扩展维度 $4h$ 决定。

---

## 问题定义与边界

TP 解决的问题，不是“模型怎么拆都行”，而是“如何拆分线性层，既让多卡并行计算，又不把通信成本放大到不可接受”。

这里讨论的边界限定为 Transformer 或 MLP 中的线性层，不讨论卷积并行、专家并行，也不讨论纯数据并行。研究对象统一写成：

$$
Y = XA
$$

其中：

- $X \in \mathbb{R}^{(bs)\times h}$，把 batch 维和 sequence 维展平后看成一个大批次
- $A \in \mathbb{R}^{h \times m}$
- $Y \in \mathbb{R}^{(bs)\times m}$

TP 的目标是：

- 把权重矩阵 $A$ 分到多张卡上，降低单卡参数显存占用
- 让每张卡只做一部分矩阵乘法，降低单卡算力压力
- 保证最终输出与单卡计算严格一致，而不是近似一致

先把常见术语放在同一张表里：

| 术语 | 精确定义 | 在本文里的含义 |
|---|---|---|
| 列并行 | 按输出维切分权重矩阵 | $A=[A_1,\dots,A_n]$ |
| 行并行 | 按输入维切分权重矩阵 | $A=[A_1;\dots;A_n]$ |
| 局部输出 | 单张卡直接算出的结果 | 可能是完整列块，也可能只是部分和 |
| 聚合 | 跨卡组合局部结果 | 常见为 AllReduce 或 AllGather |
| TP 组 | 一起做张量并行的一组 GPU | 大小为 $n$ |

再看列并行和行并行的边界差别：

| 切分方式 | 权重切分 | 每卡拿到的输入 | 每卡输出 | 是否需要聚合 |
|---|---|---|---|---|
| 列并行 | 按输出维切 $A_i$ | 完整 $X$ | $Y_i = XA_i$ | 前向通常不需要 |
| 行并行 | 按输入维切 $A_i$ | 对应分片 $X_i$ | $Y_i^{\text{partial}} = X_iA_i$ | 需要求和 |
| 组合 MLP | 第一层列切，第二层行切 | 模块输入完整 | 模块输出完整 | 模块内部封装通信 |

“模块内部封装通信”的意思是：从模块外部看，MLP 仍然接收完整输入、输出完整张量；切分、局部计算、AllReduce 都在模块内部完成。这样堆叠多个 block 时，外层代码不需要理解每一层的切分细节。

两张卡的最小例子如下：

- 第一层：把 $A \in \mathbb{R}^{4\times 6}$ 按列切成两个 $4\times 3$
- 第二层：把 $B \in \mathbb{R}^{6\times 4}$ 按行切成两个 $3\times 4$

这正好对应 MLP 的“先扩维，再投回”的结构。第一层的输出天然分成两段，第二层刚好按这两段分别消费，所以中间不需要先做一次“收集完整激活，再重新切开”的额外通信。

因此，TP 真正有价值的边界是：

- 线性层占模型算力和参数的大头
- 隐藏维和中间维足够大
- GPU 之间有高速互联
- 模块结构允许相邻线性层的切分方向互补

---

## 核心机制与推导

先从最小数学分解开始。

### 1. 列并行为什么前向不需要求和

设
$$
A = [A_1, A_2], \quad A_1 \in \mathbb{R}^{h\times m_1},\ A_2 \in \mathbb{R}^{h\times m_2}
$$

那么
$$
XA = X[A_1, A_2] = [XA_1, XA_2]
$$

这不是近似，也不是工程技巧，而是矩阵乘法对列拼接的严格分配律。

如果把形状写清楚，会更容易看懂：

- $X \in \mathbb{R}^{(bs)\times h}$
- $A_1 \in \mathbb{R}^{h\times m_1}$
- $A_2 \in \mathbb{R}^{h\times m_2}$

则：

- $XA_1 \in \mathbb{R}^{(bs)\times m_1}$
- $XA_2 \in \mathbb{R}^{(bs)\times m_2}$

把它们按列拼起来，正好得到
$$
Y \in \mathbb{R}^{(bs)\times (m_1+m_2)}
$$

所以两张卡都拿完整输入 $X$：

- 卡 0 计算 $Y_1 = XA_1$
- 卡 1 计算 $Y_2 = XA_2$

最后在逻辑上把输出看成
$$
Y = [Y_1, Y_2]
$$
即可。

这里的“逻辑上拼接”非常重要。它表示后续层如果本来就按这一分片布局消费数据，那么系统不必真的把所有数据物理搬回同一张卡。很多工程优化都建立在这一点上。

反向传播时，列并行对应的直觉也一致：

- 权重梯度 $\nabla A_i = X^\top \nabla Y_i$ 可以在各卡本地计算
- 但输入梯度
  $$
  \nabla X = \sum_{i=1}^n \nabla Y_i A_i^\top
  $$
  需要把各卡贡献求和

也就是说，列并行通常把主要通信推到反向的输入梯度聚合阶段，而不是前向输出阶段。

### 2. 行并行为什么必须聚合

设
$$
A = \begin{bmatrix}
A_1 \\
A_2
\end{bmatrix}, \quad
X = [X_1, X_2]
$$

则
$$
XA = [X_1, X_2]
\begin{bmatrix}
A_1 \\
A_2
\end{bmatrix}
= X_1A_1 + X_2A_2
$$

这一步比列并行更容易让新手困惑，因为这里不是拼接，而是求和。原因是行切分后，矩阵乘法沿着被切开的内积维度发生累加。

把形状写出来更直观：

- $X_1 \in \mathbb{R}^{(bs)\times h_1}$
- $X_2 \in \mathbb{R}^{(bs)\times h_2}$
- $A_1 \in \mathbb{R}^{h_1\times m}$
- $A_2 \in \mathbb{R}^{h_2\times m}$

于是：

- $X_1A_1 \in \mathbb{R}^{(bs)\times m}$
- $X_2A_2 \in \mathbb{R}^{(bs)\times m}$

它们的形状相同，含义是对同一个输出张量的两部分贡献，所以必须逐元素相加，而不是并排拼接。

因此两张卡分别只能算：

- 卡 0：$Z_1 = X_1A_1$
- 卡 1：$Z_2 = X_2A_2$

完整输出是：
$$
Z = Z_1 + Z_2
$$

所以前向阶段必须做一次求和聚合。工程上最常见的实现是 AllReduce。它的结果是所有卡都得到同一个总和张量。

反向传播时，行并行的结构恰好和前向互补：

- 输入梯度 $\nabla X_i = \nabla Y A_i^\top$ 可以按分片本地得到
- 权重梯度 $\nabla A_i = X_i^\top \nabla Y$ 也能本地计算
- 若上一层需要完整输入布局，则可能在别的位置发生重新分发

因此，行并行的“前向要聚合、反向更自然”，列并行则常常相反。这也是二者常成对出现的原因。

### 3. 为什么 MLP 常用 Column-then-Row

一个标准 MLP 可以写成：

$$
H = \phi(XW_1), \quad Y = HW_2
$$

其中：

- $W_1 \in \mathbb{R}^{h\times 4h}$，负责扩维
- $W_2 \in \mathbb{R}^{4h\times h}$，负责投回隐藏维
- $\phi$ 是逐元素激活函数，如 GELU、ReLU、SiLU

把第一层按列切分：

$$
W_1 = [W_{1,1}, W_{1,2}, \dots, W_{1,n}]
$$

则
$$
XW_1 = [XW_{1,1}, XW_{1,2}, \dots, XW_{1,n}]
$$

若激活函数 $\phi$ 是逐元素的，则它与分片布局兼容：

$$
H = \phi(XW_1)
= [\phi(XW_{1,1}), \phi(XW_{1,2}), \dots, \phi(XW_{1,n})]
$$

于是可以记为
$$
H = [H_1, H_2, \dots, H_n], \quad H_i = \phi(XW_{1,i})
$$

再把第二层按行切分：

$$
W_2 =
\begin{bmatrix}
W_{2,1}\\
W_{2,2}\\
\vdots\\
W_{2,n}
\end{bmatrix}
$$

由于 $H$ 本来就按列分成了 $[H_1,\dots,H_n]$，因此每张卡正好消费自己的那一段：

$$
Y_i^{\text{partial}} = H_i W_{2,i}
$$

完整输出为：
$$
Y = \sum_{i=1}^n Y_i^{\text{partial}}
$$

这里的关键不是“两个线性层都能切”，而是“第一层的输出分片布局，正好是第二层需要的输入分片布局”。因此中间不需要额外聚合。通信被推迟到了第二层末尾。

这就是 Megatron 风格 TP 的核心设计原则：让相邻层的分片方向互补，把通信点压缩到模块边界，而不是每经过一层就做一次完整收集。

一个更工程化的判断表如下：

| 层位置 | 常见切分 | 局部输出含义 | 是否立刻通信 |
|---|---|---|---|
| MLP 第一层 | 列并行 | 中间激活的一段列块 | 不需要 |
| 激活函数 | 逐元素本地算子 | 不改变分片布局 | 不需要 |
| MLP 第二层 | 行并行 | 最终输出的部分和 | 需要求和 |
| 模块外部 | 完整隐藏态 | 残差、LayerNorm 可继续消费 | 已完成聚合 |

### 4. 玩具例子

下面给一个完整的两卡数值结构，但先不写具体数值，先把形状关系看清楚。

设输入
$$
X=
\begin{bmatrix}
1&2&3&4\\
5&6&7&8
\end{bmatrix}
\in \mathbb{R}^{2\times4}
$$

第一层权重
$$
A\in\mathbb{R}^{4\times6}
$$
按列切成
$$
A_1,A_2\in\mathbb{R}^{4\times3}
$$

于是两张卡分别得到：
$$
Y_1 = XA_1,\quad Y_2 = XA_2
$$

完整中间激活是
$$
Y=[Y_1,Y_2]\in\mathbb{R}^{2\times6}
$$

第二层权重
$$
B\in\mathbb{R}^{6\times4}
$$
按行切成
$$
B_1,B_2\in\mathbb{R}^{3\times4}
$$

于是两张卡各算：
$$
Z_1 = Y_1B_1,\quad Z_2 = Y_2B_2
$$

最终：
$$
Z = Z_1 + Z_2
$$

如果把一行输出单独展开，会更直观看到“为什么是求和”。设某个样本对应的中间激活是
$$
y = [y^{(1)}, y^{(2)}]
$$
其中 $y^{(1)}\in\mathbb{R}^{1\times 3}, y^{(2)}\in\mathbb{R}^{1\times 3}$，则
$$
yB = [y^{(1)}, y^{(2)}]
\begin{bmatrix}
B_1\\
B_2
\end{bmatrix}
= y^{(1)}B_1 + y^{(2)}B_2
$$

所以第二层不是“各卡各出一部分列”，而是“各卡各算同一个输出的部分贡献”。

这个例子展示了 TP 的标准套路：

1. 切分权重矩阵，而不是随意切模型结构。
2. 让每张卡只做本地矩阵乘法。
3. 只有在数学上确实需要求和时，才做集合通信。
4. 最终输出与单卡结果严格一致。

### 5. 通信量为什么是 $8bsh\times\frac{n-1}{n}$

先把前提说清楚。设一次集合通信处理的激活张量形状为：

$$
[b, s, h]
$$

若按 FP32 估算，每个元素占 4 字节，则张量大小是：

$$
S = 4bsh
$$

对环形 AllReduce，常见通信体积估算为：

$$
2S \times \frac{n-1}{n}
$$

原因是它可以分解为两阶段：

1. Reduce-Scatter
2. AllGather

每一阶段的通信体积都近似为
$$
S \times \frac{n-1}{n}
$$
因此总计为
$$
2S \times \frac{n-1}{n}
$$

代入 $S = 4bsh$，得到一次 AllReduce 的通信量：

$$
4bsh \times 2 \times \frac{n-1}{n}
= 8bsh\times\frac{n-1}{n}
$$

如果使用 BF16 或 FP16，每个元素通常是 2 字节，则对应变成：

$$
4bsh\times\frac{n-1}{n}
$$

所以这个公式本身并不是“固定常数”，它依赖于两个前提：

- 通信张量的形状是 $[b,s,h]$
- 元素按 4 字节估算

但更重要的结论不受这些细节影响：通信量与**模块边界上的隐藏态大小**成正比，而不是与中间扩展维 $4h$ 成正比。

以 MLP 为例：

- 第一层输出宽度是 $4h$
- 第二层输入宽度也是 $4h$
- 但如果采用 Column-then-Row，主通信发生在第二层输出侧
- 第二层输出的形状回到了 $[b,s,h]$

这就是为什么 TP 看起来在处理“超宽中间层”，但主通信量却不一定随中间层宽度线性爆炸。

---

## 代码实现

下面先给一个**纯 Python 标准库**的玩具实现。它不依赖 `numpy`、`torch` 或分布式框架，可以直接运行，用来验证“单卡结果”和“列并行 + 行并行 + 求和”结果完全一致。

```python
from math import isclose

# 输入 X: shape = [2, 4]
X = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
]

# 第一层 A: [4, 6]，按列切成两个 [4, 3]
A = [
    [1.0, 0.0, 2.0, 1.0, 3.0, 0.0],
    [0.0, 1.0, 1.0, 2.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0, 2.0, 2.0],
    [2.0, 0.0, 1.0, 0.0, 1.0, 3.0],
]

# 第二层 B: [6, 4]，按行切成两个 [3, 4]
B = [
    [1.0, 0.0, 2.0, 1.0],
    [0.0, 1.0, 1.0, 0.0],
    [2.0, 1.0, 0.0, 1.0],
    [1.0, 2.0, 0.0, 1.0],
    [0.0, 1.0, 2.0, 2.0],
    [1.0, 0.0, 1.0, 3.0],
]


def matmul(left, right):
    rows = len(left)
    inner = len(left[0])
    cols = len(right[0])
    assert inner == len(right)

    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            value = 0.0
            for k in range(inner):
                value += left[i][k] * right[k][j]
            row.append(value)
        out.append(row)
    return out


def relu(matrix):
    return [[max(v, 0.0) for v in row] for row in matrix]


def split_columns(matrix, split_at):
    left = [row[:split_at] for row in matrix]
    right = [row[split_at:] for row in matrix]
    return left, right


def split_rows(matrix, split_at):
    top = matrix[:split_at]
    bottom = matrix[split_at:]
    return top, bottom


def add(left, right):
    return [
        [a + b for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def allclose(left, right, tol=1e-9):
    for left_row, right_row in zip(left, right):
        for a, b in zip(left_row, right_row):
            if not isclose(a, b, rel_tol=tol, abs_tol=tol):
                return False
    return True


A1, A2 = split_columns(A, 3)
B1, B2 = split_rows(B, 3)

# 单卡基线
Y_full = relu(matmul(X, A))
Z_full = matmul(Y_full, B)

# Tensor Parallel
# 第一层列并行
Y1 = relu(matmul(X, A1))
Y2 = relu(matmul(X, A2))

# 第二层行并行
Z1 = matmul(Y1, B1)
Z2 = matmul(Y2, B2)

# AllReduce 之后的完整输出
Z_tp = add(Z1, Z2)

assert allclose(Z_full, Z_tp)

print("Z_full =", Z_full)
print("Z_tp   =", Z_tp)
```

这段代码的价值在于：它把 TP 的数学本质完整暴露出来了，但没有引入任何框架细节。你可以先确认这个脚本看懂了，再去看真实分布式实现。

若需要一个更接近工程实际的最小示例，可以用 `torch.distributed` 手写两卡版本。下面脚本假设有 2 张 GPU，并通过 `torchrun` 启动：

```python
# 保存为 tp_demo.py
# 启动方式：
# torchrun --standalone --nproc_per_node=2 tp_demo.py

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size == 2, "This demo expects exactly 2 GPUs."

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 为了对比单卡基线，这里每个 rank 都构造完整矩阵。
    # 教学脚本这样写更直观；真实 TP 系统通常只持有本地 shard。
    X = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        device=device,
    )

    A = torch.tensor(
        [
            [1.0, 0.0, 2.0, 1.0, 3.0, 0.0],
            [0.0, 1.0, 1.0, 2.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 2.0, 2.0],
            [2.0, 0.0, 1.0, 0.0, 1.0, 3.0],
        ],
        device=device,
    )

    B = torch.tensor(
        [
            [1.0, 0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 2.0, 2.0],
            [1.0, 0.0, 1.0, 3.0],
        ],
        device=device,
    )

    A_shards = torch.chunk(A, world_size, dim=1)  # 列切
    B_shards = torch.chunk(B, world_size, dim=0)  # 行切

    A_local = A_shards[rank].contiguous()
    B_local = B_shards[rank].contiguous()

    # 单卡基线
    Y_full = F.relu(X @ A)
    Z_full = Y_full @ B

    # TP 前向
    Y_local = F.relu(X @ A_local)   # 列并行
    Z_local = Y_local @ B_local     # 行并行的局部部分和

    # AllReduce 聚合得到完整输出
    dist.all_reduce(Z_local, op=dist.ReduceOp.SUM)

    assert torch.allclose(Z_local, Z_full)

    if rank == 0:
        print("Z_full =")
        print(Z_full.cpu())
        print("Z_tp =")
        print(Z_local.cpu())

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

这个脚本体现了三个关键点：

- 第一层列并行时，每个 rank 只保留自己的输出列块。
- 第二层行并行时，每个 rank 只产生完整输出的一个部分和。
- 只有在 `dist.all_reduce` 之后，才得到与单卡完全一致的最终结果。

真实工程里，常见封装不是手写 `all_reduce`，而是通过 Megatron-LM、PyTorch DTensor 或框架封装的 TP API 来描述“这一层按列切”“下一层按行切”。但底层数学并没有变化。

---

## 工程权衡与常见坑

TP 不是“卡越多越好”的通用答案，它高度依赖互联带宽、矩阵规模和模块结构。

先看工程判断表：

| 维度 | TP 的特点 | 风险 |
|---|---|---|
| 显存 | 权重按卡分摊，单卡参数显存下降 | 激活显存未必同比下降 |
| 计算 | 大矩阵可并行，单层吞吐通常改善 | 小矩阵时启动和通信开销占比高 |
| 通信 | 每个 TP 模块都有集合通信点 | 带宽不足时性能迅速恶化 |
| 实现 | 适合模块化封装 | 调试、日志、数值对齐更复杂 |
| 拓扑 | 单机多卡收益更稳定 | 跨节点 TP 风险显著增大 |

常见坑主要有五类。

第一类是把 TP 放到低带宽互联上。  
TP 的问题不在“有没有通信”，而在“通信离算子非常近”。如果每一层都需要等 AllReduce 结束，低带宽网络会直接把矩阵乘法的加速收益吃掉。单机 NVLink/NVSwitch 通常更适合 TP，跨节点时则必须非常谨慎。

第二类是切分方向和层结构不匹配。  
如果上一层列切，下一层却不能直接消费这个列块布局，就会出现“先 Gather 完整激活，再重新切分”的额外搬运。这样虽然数学上仍然正确，但通信次数增加，TP 的优势会明显下降。

第三类是误把中间维度当成通信主因。  
以 MLP 为例，中间层是 $4h$，很多人会直觉地认为通信也会按 $4h$ 爆炸。经典 TP 设计恰恰是在规避这一点：让中间的 $4h$ 尽量停留在本地分片布局里，把主通信压到输出边界的 $h$ 维上。

第四类是忽略算子兼容性。  
逐元素激活、dropout、bias 加法通常都能保留分片布局；但如果中间插入必须看到完整特征维的算子，例如某些非局部归一化、跨特征重排或额外投影，就可能打断“列切输出直接喂给行切输入”的链路。

第五类是只看显存，不看端到端吞吐。  
TP 的确能把超大层拆开，让模型“能跑起来”；但能跑不代表快。如果 batch 太小、序列太短、内核尺寸太碎，通信和 kernel launch 的占比会升高，实际吞吐可能不如更少的卡数，甚至不如单卡。

新手还容易忽略一个事实：TP 的收益是分层的，而不是全局平均的。  
如果模型里只有少数超大线性层值得切分，而其他层很小，那么“值得 TP 的部分”和“只是在等待通信的部分”可能并不平衡。工程上通常要结合 profile 看每个 block 的算时和通信时，而不是只看理论 FLOPs。

真实的 Megatron 风格 Transformer block，一般把主通信点限制在：

- Attention 输出投影之后
- FFN 第二层之后

这类设计的价值在于：block 内部虽然复杂，但 block 对外仍然是“输入完整、输出完整”的普通模块接口。这种封装能力是 TP 可维护性的前提。

---

## 替代方案与适用边界

TP 不是唯一并行方案。更准确地说，它解决的是“单层线性层太宽，单卡放不下或算不动”的问题。

对比几种常见并行方式：

| 方案 | 核心思路 | 通信频率 | 更适合什么场景 |
|---|---|---|---|
| Tensor Parallel | 同一层权重拆到多卡 | 每层较频繁 | 单机高速互联、超大线性层 |
| Pipeline Parallel | 不同层放到不同卡 | 阶段边界通信 | 模型层数深、跨节点更常见 |
| Data Parallel / FSDP | 不同样本或参数副本分布到多卡 | 主要在梯度同步或参数重建 | 通用训练扩展 |
| 2D/3D Parallel | 组合 TP、PP、DP/FSDP | 多种通信同时存在 | 超大模型训练 |

TP 的核心优势是：  
它直接解决“单层矩阵太大”这个问题。只要算子结构合适，拆分后仍能保持严格等价。

PP 的核心优势是：  
它把模型层分配到不同设备，减少“每一层都通信”的问题，更适合跨节点。但代价是流水线调度复杂，容易出现 bubble。

FSDP 或 DP 的核心优势是：  
实现路径更通用，对模型结构要求低，很多时候是大规模训练的基础配置。但它解决的是样本并行或参数副本问题，不直接降低单层 GEMM 的宽度。

因此可以把适用边界总结为：

- 模型的瓶颈在超大线性层，优先考虑 TP。
- 网络带宽一般，尤其是跨节点带宽一般时，优先限制 TP 只在节点内使用。
- 层数很多、跨机训练明显时，PP 往往比大范围 TP 更稳。
- 大规模训练通常不是单独选 TP，而是把 TP 限制在单节点内，再与 PP、FSDP 或 DP 组合。
- 对小模型、短序列、小 batch，TP 往往得不偿失。

一个实用判断标准是：  
如果一次模块计算的本地 GEMM 时间，还不足以覆盖一次集合通信的等待时间，那么继续加大 TP 规模通常不会带来正收益。

---

## 参考资料

- PyTorch 官方文档，`torch.distributed.tensor.parallel`：包含 `ColwiseParallel`、`RowwiseParallel`、`parallelize_module` 等 API  
  https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html

- Shoeybi et al., “Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism”  
  https://arxiv.org/abs/1909.08053

- Narayanan et al., “Reducing Activation Recomputation in Large Transformer Models”  
  https://arxiv.org/abs/2205.05198

- NVIDIA NCCL User Guide，集合通信与 AllReduce 说明  
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

- Megatron-LM 项目仓库，可直接查看 Transformer block 中 TP 的典型实现  
  https://github.com/NVIDIA/Megatron-LM
