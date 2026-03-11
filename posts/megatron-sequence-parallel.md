## 核心结论

Megatron-LM 的序列并行，英文是 Sequence Parallelism，简称 SP。它的含义不是“把整个 Transformer 都按序列维度切开”，而是更具体的一件事：

在 **Tensor Parallel，张量并行** 的进程组内部，原本会在每张卡上完整复制的那部分非张量并行激活，不再按完整序列保存，而是沿序列维度切分到不同卡上。

直白说：

- Attention 和 MLP 这类大算子，继续按 hidden 维做 Tensor Parallel。
- LayerNorm、Dropout、残差相关缓存这类逐 token 独立的操作，改成每张卡只处理自己负责的那一段 token。

它解决的问题很具体。Tensor Parallel 只拆掉了大矩阵乘法，但很多“参数不大、计算不重”的层，仍然让每张卡保留完整的 $[B,S,H]$ 激活。这里：

- $B$ 表示 batch size
- $S$ 表示序列长度
- $H$ 表示 hidden size

当序列变长时，这些没有被 TP 拆掉的激活，往往先成为显存瓶颈。

它的核心收益也很直接。如果 Tensor Parallel 大小是 $t$，那么被序列并行覆盖的那部分激活，会从每卡

$$
O(SBH)
$$

下降到每卡

$$
O\left(\frac{SBH}{t}\right)
$$

这不是经验近似，而是设计目标本身。

先看一个最小直观例子。假设总序列长度始终是 1024：

| TP 大小 $t$ | 原始每卡序列长度 | 序列并行后每卡序列长度 | 非 TP 激活占比 |
|---|---:|---:|---:|
| 1 | 1024 | 1024 | 100% |
| 2 | 1024 | 512 | 50% |
| 4 | 1024 | 256 | 25% |
| 8 | 1024 | 128 | 12.5% |

如果 4 卡 TP 训练一层 Transformer，原本每张卡都要为 1024 个 token 保存 LayerNorm 和 Dropout 的激活；启用序列并行后，每张卡只保留其中 256 个 token 的对应激活。Attention 和 MLP 仍然执行 TP，但非 TP 激活不再全量复制。

这就是为什么 Sequence Parallel 常被称为“TP 的补全方案”：

- TP 负责拆算力热点和参数热点。
- SP 负责拆激活热点，尤其是那些原本在每张卡上重复保存的非 TP 激活。

---

## 问题定义与边界

先把边界说清楚。序列并行优化的不是“整个模型的所有中间状态”，而是那些满足下面条件的模块：

1. 计算是逐 token 独立的。
2. 不依赖别的 token 的上下文信息。
3. 不要求当前卡长期持有完整序列表示。
4. 在 TP 结构下，原本容易被完整复制。

在 Megatron-LM 里，典型对象包括：

- LayerNorm / RMSNorm
- Dropout
- 一些逐位置的残差输入输出缓存
- 其他不需要跨 token 聚合的逐元素算子

它为什么会浪费显存，可以从“谁被切了，谁没被切”来理解。

| 模块 | 主要并行维度 | 是否天然支持 TP | 为什么会占显存 |
|---|---|---|---|
| Attention / MLP 的线性层 | hidden 维 | 是 | 参数和大部分计算已按 hidden 切开 |
| LayerNorm / RMSNorm | token 内部做归一化 | 否，默认全量复制 | 每张卡都保留完整 $[B,S,H]$ |
| Dropout | 逐元素 | 否，默认全量复制 | 掩码和输出都可能按完整序列保存 |
| 残差接口缓存 | 与层输入输出同形状 | 否，默认全量复制 | 为反向传播保留完整激活 |

这里最容易误解的一点是：LayerNorm 虽然也要在 hidden 维上做均值和方差，但它是 **对每个 token 独立** 地在最后一维上归一化，不需要和别的 token 通信。所以它完全可以只在本地序列段上执行。

看一个玩具例子。假设某层输入是：

$$
B=2,\quad S=2048,\quad H=4096,\quad t=4
$$

那么完整输入激活元素数是：

$$
2 \times 2048 \times 4096 = 16{,}777{,}216
$$

如果不做序列并行，即便 Attention 和 MLP 已经按 TP=4 切开，LayerNorm、Dropout、残差相关缓存仍然可能在每张卡上各保留完整一份这 1677 万个元素。若使用 BF16，每个元素 2 字节，仅这一份激活就是：

$$
16{,}777{,}216 \times 2 \approx 32\text{ MB}
$$

如果这一层有 3 份类似缓存，单层就接近 96 MB；当层数从几十层增长到上百层，显存就会很快变成第一瓶颈。

所以它的适用边界也很明确：

| 判断问题 | 结论 |
|---|---|
| `tensor_model_parallel_size = 1` 能否单独使用 SP？ | 通常没有意义，SP 本来就是围绕 TP 组设计的 |
| SP 优化的是参数显存吗？ | 不是，主要优化激活显存 |
| SP 能替代 Pipeline Parallel 吗？ | 不能，目标不同 |
| SP 能替代 Context Parallel 吗？ | 不能，后者解决的是长序列注意力本身的切分 |
| SP 有没有代价？ | 有，需要额外的集体通信 |

可以把它和另外几种并行方式放在一起对比：

| 方案 | 主要切分对象 | 主要收益 | 主要代价 |
|---|---|---|---|
| Tensor Parallel | 参数和大矩阵计算 | 降低单卡参数与单次 GEMM 压力 | 线性层相关通信 |
| Sequence Parallel | 非 TP 激活 | 降低每卡激活常驻显存 | 每层增加布局切换通信 |
| Context Parallel | 长序列本身与注意力状态 | 支持更长上下文 | 注意力通信更复杂 |
| Activation Checkpointing | 激活保存策略 | 用算力换显存 | 反向重算带来时间开销 |

新手可以记一个最实用的判断句：

如果你已经开了 TP，但显存还是因为 LayerNorm、Dropout、残差缓存这类激活而爆掉，那么该看的通常不是“再拆参数”，而是“这些非 TP 激活是不是该做 Sequence Parallel”。

---

## 核心机制与推导

Megatron-LM 里可以把一层 Transformer 粗略拆成两段：

1. 非 TP 区域：LayerNorm、Dropout、残差相关逐 token 操作。
2. TP 区域：Attention、MLP 这类按 hidden 维切分的大算子。

序列并行的关键不是“从头到尾保持同一种布局”，而是：

- 在非 TP 区域，使用按序列切分的局部表示。
- 进入 TP 区域前，临时恢复完整序列表示。
- 离开 TP 区域后，再把输出重新切回局部序列表示。

这就是很多资料中提到的 $g$ 和 $\bar g$ 变换。

### 1. 局部表示

假设 TP 大小为 $t$，则每张卡只持有自己的序列段：

$$
X_{\text{local}} \in \mathbb{R}^{B \times S/t \times H}
$$

例如：

- 全局张量形状是 $[2, 1024, 4096]$
- TP 大小是 4
- 那么每卡局部形状就是 $[2, 256, 4096]$

此时 LayerNorm、Dropout、残差加法，都可以在这个局部张量上直接完成。

### 2. 进入 TP 区域前的 All-Gather

Attention 和 MLP 所在的 TP 区域，通常需要在接口上看到完整序列。因此进入这些模块之前，要对序列维做一次 All-Gather：

$$
g(X_{\text{local}}) = X_{\text{full}} \in \mathbb{R}^{B \times S \times H}
$$

这一步的含义不是“做数值归约”，而是“把分散在不同卡上的序列段重新拼成完整表示”。

如果 $t=4$，每张卡持有 256 个 token，那么 All-Gather 后，每张卡都会临时拿到完整的 1024 个 token 表示，供 TP Attention 或 TP MLP 使用。

### 3. 离开 TP 区域后的 Reduce-Scatter

经过 Attention 或 MLP 之后，再通过 Reduce-Scatter 把输出重新压回局部序列布局：

$$
\bar g(Y_{\text{full}}) = Y_{\text{local}} \in \mathbb{R}^{B \times S/t \times H}
$$

这一步的目的也不是单纯“同步”，而是恢复节省显存的表示形式。之后 Dropout、残差等非 TP 模块继续只处理本地序列段。

### 4. 一个完整的数据流

把前面的过程串起来，可以写成：

$$
X_{\text{local}}
\overset{\text{LN}}{\longrightarrow}
\tilde X_{\text{local}}
\overset{g}{\longrightarrow}
\tilde X_{\text{full}}
\overset{\text{TP Block}}{\longrightarrow}
Y_{\text{full}}
\overset{\bar g}{\longrightarrow}
Y_{\text{local}}
\overset{\text{Dropout/Residual}}{\longrightarrow}
Z_{\text{local}}
$$

前向阶段的核心就是两次布局切换：

- 局部序列 $\rightarrow$ 完整序列
- 完整序列 $\rightarrow$ 局部序列

反向传播则对应相反方向的通信，保证梯度和前向布局一致。

### 5. 显存推导

如果某类非 TP 激活原本每卡都保存完整 $[B,S,H]$，那么每卡的激活量近似是：

$$
M_{\text{base}} = c \cdot SBH
$$

其中 $c$ 表示常数项，取决于你实际保存了多少份中间状态，比如输入、输出、dropout mask、残差缓存等。

启用 SP 后，这部分激活会按序列切到 $t$ 张卡上，因此每卡变成：

$$
M_{\text{sp}} = c \cdot \frac{SBH}{t}
$$

所以：

$$
\frac{M_{\text{sp}}}{M_{\text{base}}} = \frac{1}{t}
$$

这就是“按 TP 大小等比例下降”的来源。

### 6. 用具体数字算一遍

设：

$$
s=1024,\quad b=2,\quad h=4096,\quad t=4
$$

完整激活元素数：

$$
sbh = 1024 \times 2 \times 4096 = 8{,}388{,}608
$$

序列并行后，每卡只保留：

$$
\frac{sbh}{t} = \frac{8{,}388{,}608}{4} = 2{,}097{,}152
$$

即：

- 原始每卡：8388608 个元素
- SP 后每卡：2097152 个元素
- 比例：25%

如果元素类型是 BF16，每个元素 2 字节，那么单份激活从大约 16 MB 降到 4 MB。若一层要保存多份类似激活，节省会被进一步放大。

### 7. 为什么论文里的式子也会出现 $1/t$

Megatron 相关论文里，激活项常写成类似：

$$
sbh \left(34 + \frac{5a_s}{h}\right)
$$

引入序列并行后，受影响的那部分会变成：

$$
\frac{sbh}{t}\left(34 + \frac{5a_s}{h}\right)
$$

这里：

- $s$ 是 sequence length
- $b$ 是 batch size
- $h$ 是 hidden size
- $a_s$ 是 attention heads 数量
- $t$ 是 tensor parallel size

对初学者而言，不需要记住系数 34 和 $\frac{5a_s}{h}$ 的来源。真正需要记住的是：

论文中复杂公式的本质，仍然是在说明一件简单的事：被 SP 覆盖的那部分激活，会按 TP 大小均摊。

### 8. 为什么它能和重算配合

Activation recomputation（激活重算）解决的是“少存、多算”问题，SP 解决的是“不要在每张卡重复存”问题。两者优化的对象不同，所以可以叠加：

- 重算减少需要长期保存的激活数量。
- SP 减少每份激活在每张卡上的复制量。

因此在超大模型训练里，常见组合不是“TP 或 SP 二选一”，而是：

- TP 处理大算子切分
- SP 处理非 TP 激活复制
- selective recomputation 进一步压缩显存与计算开销

---

## 代码实现

下面先给一个真正可运行的 Python 示例。它不依赖 PyTorch 分布式环境，但能把“序列切分、聚合、再切回去”的核心逻辑跑通，并验证形状和显存比例。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ShardInfo:
    rank: int
    shape: tuple[int, int, int]
    elements: int


def num_elements(shape: tuple[int, int, int]) -> int:
    b, s, h = shape
    return b * s * h


def split_sequence(batch: int, seq_len: int, hidden: int, tp_size: int) -> List[ShardInfo]:
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if seq_len % tp_size != 0:
        raise ValueError("seq_len must be divisible by tp_size")

    local_seq = seq_len // tp_size
    shards = []
    for rank in range(tp_size):
        shape = (batch, local_seq, hidden)
        shards.append(ShardInfo(rank=rank, shape=shape, elements=num_elements(shape)))
    return shards


def all_gather_sequence(shards: List[ShardInfo], full_seq_len: int) -> List[ShardInfo]:
    if not shards:
        raise ValueError("shards must not be empty")

    batch, _, hidden = shards[0].shape
    gathered_shape = (batch, full_seq_len, hidden)

    gathered = []
    for shard in shards:
        gathered.append(
            ShardInfo(
                rank=shard.rank,
                shape=gathered_shape,
                elements=num_elements(gathered_shape),
            )
        )
    return gathered


def reduce_scatter_sequence(gathered: List[ShardInfo], tp_size: int) -> List[ShardInfo]:
    if not gathered:
        raise ValueError("gathered must not be empty")

    batch, full_seq_len, hidden = gathered[0].shape
    if full_seq_len % tp_size != 0:
        raise ValueError("full_seq_len must be divisible by tp_size")

    local_seq = full_seq_len // tp_size
    local_shape = (batch, local_seq, hidden)

    scattered = []
    for shard in gathered:
        scattered.append(
            ShardInfo(
                rank=shard.rank,
                shape=local_shape,
                elements=num_elements(local_shape),
            )
        )
    return scattered


def main() -> None:
    batch = 2
    seq_len = 1024
    hidden = 4096
    tp_size = 4
    bytes_per_elem = 2  # bf16/fp16

    full_shape = (batch, seq_len, hidden)
    full_elems = num_elements(full_shape)
    full_mb = full_elems * bytes_per_elem / 1024 / 1024

    local_shards = split_sequence(batch, seq_len, hidden, tp_size)
    assert all(shard.shape == (2, 256, 4096) for shard in local_shards)

    gathered = all_gather_sequence(local_shards, seq_len)
    assert all(shard.shape == full_shape for shard in gathered)

    scattered = reduce_scatter_sequence(gathered, tp_size)
    assert [shard.shape for shard in scattered] == [shard.shape for shard in local_shards]

    local_elems = local_shards[0].elements
    local_mb = local_elems * bytes_per_elem / 1024 / 1024
    ratio = local_elems / full_elems

    print("full shape:", full_shape)
    print("per-rank local shape:", local_shards[0].shape)
    print("full elems:", full_elems)
    print("local elems:", local_elems)
    print("memory ratio:", ratio)
    print("full activation MB:", round(full_mb, 2))
    print("local activation MB:", round(local_mb, 2))


if __name__ == "__main__":
    main()
```

这段代码运行后会得到几个关键事实：

- 全局激活形状是 `(2, 1024, 4096)`
- 4 卡 SP 后，每卡局部激活形状是 `(2, 256, 4096)`
- 每卡元素数正好是原来的 `1/4`
- 局部表示可以通过“聚合再切回”的方式与完整表示来回转换

如果想更直观地看“每张卡负责哪一段 token”，再看一个更小的例子：

```python
def token_ranges(seq_len: int, tp_size: int):
    if seq_len % tp_size != 0:
        raise ValueError("seq_len must be divisible by tp_size")

    local_seq = seq_len // tp_size
    ranges = []
    for rank in range(tp_size):
        start = rank * local_seq
        end = start + local_seq
        ranges.append((rank, start, end))
    return ranges


if __name__ == "__main__":
    for rank, start, end in token_ranges(seq_len=16, tp_size=4):
        print(f"rank {rank}: tokens [{start}, {end})")
```

输出逻辑是：

- rank 0 负责 token `[0, 4)`
- rank 1 负责 token `[4, 8)`
- rank 2 负责 token `[8, 12)`
- rank 3 负责 token `[12, 16)`

这就是“按序列段切分”的最简单语义。

如果把它映射回 Megatron-LM，可以写成更接近实际工程的伪代码：

```python
# x_local: [B, S/t, H]

# 非 TP 区域：只在本地序列段上执行
x_local = layernorm(x_local)

# 进入 TP Attention / TP MLP 前，恢复完整序列表示
x_full = all_gather_seq(x_local)   # [B, S, H]

# TP 区域：按 hidden 维做张量并行
y_full = tensor_parallel_block(x_full)

# 离开 TP 区域后，再切回局部序列段
y_local = reduce_scatter_seq(y_full)  # [B, S/t, H]

# 后续非 TP 区域继续只处理局部序列段
y_local = dropout(y_local)
z_local = residual_add(y_local)
```

这里有三个实现重点。

### 1. 通信的作用是“切换布局”，不是“补数学正确性”

- `all_gather_seq` 的作用：把每张卡各自持有的 token 段拼回完整序列表示。
- `reduce_scatter_seq` 的作用：把完整序列输出重新分发成局部表示。

很多新手会把这两步误解成“普通同步操作”。更准确的说法是：它们是在不同计算阶段之间切换张量布局。

### 2. 通信必须发生在 TP 组内部

Sequence Parallel 是 TP 的补充机制，所以通信范围必须是对应的 **tensor parallel group**。如果错误地在更大的进程组上做 gather/scatter，形状可能看起来还能对上，但数值语义已经错了。

### 3. 张量形状要一直对齐

工程里最容易出错的不是“有没有开 `--sequence-parallel`”，而是某一层误把：

- `[B,S,H]`
- `[B,S/t,H]`

混用在一起。

最稳妥的检查方式不是只看运行是否报错，而是逐层确认：

- 进入非 TP 模块时是否是局部序列表示
- 进入 TP 模块前是否已经 gather
- 离开 TP 模块后是否已经 scatter
- 残差分支两侧的布局是否一致

可以用下面的检查表：

| 检查点 | 正确状态 |
|---|---|
| LayerNorm 输入 | `[B,S/t,H]` |
| Attention / MLP 输入接口 | 通常恢复为 `[B,S,H]` |
| TP 块输出离开后 | 再次切回 `[B,S/t,H]` |
| 残差相加两端 | 形状和布局必须一致 |
| Dropout 输入输出 | 一般都在局部序列布局下 |

---

## 工程权衡与常见坑

Sequence Parallel 的收益来自显存下降，但代价是通信增加。所以它不是“必开优化”，而是一个标准的算力、显存、网络带宽三方权衡。

先看最核心的账本变化：

| 指标 | 不开 SP | 开 SP |
|---|---|---|
| 非 TP 激活显存 | 高，且在每张卡重复 | 约降到原来的 $1/t$ |
| 每层额外通信 | 少 | 增加 All-Gather / Reduce-Scatter |
| 对网络拓扑敏感性 | 中等 | 更高 |
| 对长序列训练帮助 | 有限 | 明显 |
| 对参数显存的改善 | 无 | 无 |

这意味着一个非常重要的事实：

显存下降，不等于训练一定更快。

如果你的硬件是单机 NVLink，高带宽下新增通信的代价可能可接受；如果是跨机、低带宽网络，那么每层两次额外集体通信可能直接把 step time 拉长。

常见坑如下：

| 问题 | 本质原因 | 现象 | 规避方式 |
|---|---|---|---|
| 在错误的进程组里做通信 | TP 组定义错误 | 形状可能对，但数值不对 | 所有 gather/scatter 显式绑定 TP group |
| 漏掉 `g/\bar g` 转换 | 布局切换不完整 | 后续层只拿到局部片段 | 逐层检查输入输出布局 |
| 序列长度不能整除 TP | 切分不均匀 | 直接报错，或补 pad 后逻辑复杂 | 配置阶段保证 $S \bmod t = 0$ |
| 与 activation checkpoint 边界不一致 | 前后向重算路径不同 | 梯度异常或通信次序错误 | 把 checkpoint 边界与通信边界一起测试 |
| 残差分支布局不一致 | 一支是 full，一支是 local | 运行时报 shape 错或静默数值错 | 残差相加前统一布局 |
| 网络带宽不足 | 通信成为新瓶颈 | 显存降了，但吞吐下降 | 优先同机高速互联，必要时减少 TP 粒度 |

### 新手最容易忽略的两个问题

#### 问题一：LayerNorm“便宜”，为什么反而值得优化？

因为显存问题不只看 FLOPs，还看“要保存多少张量用于反向传播”。LayerNorm 的计算量确实远小于大矩阵乘法，但只要它的输入输出和相关缓存仍按完整 $[B,S,H]$ 保存在每张卡上，长序列下就会持续吃显存。

也就是说：

- 计算热点不一定等于显存热点。
- 参数热点也不一定等于激活热点。

SP 正是在补这一层空缺。

#### 问题二：为什么 TP Attention / TP MLP 之前还要恢复完整序列？

因为序列并行不是要改写 TP 算子的数学定义，而是要在 **不破坏 TP 主体设计** 的前提下，减少非 TP 区域的激活复制。

所以 Megatron-LM 采用的是一种更工程化的方案：

- 非 TP 区域用 local sequence layout 节省显存
- TP 主体区在接口处恢复 full sequence layout
- 算完再切回 local layout

这个思路的优点是兼容现有 TP 结构，缺点是需要明确的布局切换通信。

### 一个更接近真实工程的例子

假设你在 8 张卡上训练大模型，配置：

- TP = 8
- 序列长度 = 8192
- hidden size = 8192
- 数据类型 = BF16

此时很多团队遇到的第一类瓶颈并不是参数放不下，而是：

- LayerNorm 输入输出
- Dropout 中间态
- 残差缓存
- 反向传播需要保留的若干激活

在每张卡上都按完整 8192 token 保存，导致显存先爆。

开 SP 后，理论上这些非 TP 激活每卡只保留：

$$
\frac{8192}{8} = 1024
$$

个 token 对应的局部段。显存问题可能立刻缓解。但这时新瓶颈可能变成：

- 每层的 gather/scatter 通信变多
- 跨机链路占用提高
- step time 被网络拉长

所以实际调优顺序通常应该是：

1. 先确认原瓶颈是不是非 TP 激活。
2. 开 SP，看显存峰值是否按预期下降。
3. 再看 step time 与链路利用率。
4. 如果通信成了新瓶颈，再评估 TP 粒度、重算策略和集群拓扑。

不是“看到省显存就结束”，而是“看瓶颈是否真正转移到了更可接受的位置”。

---

## 替代方案与适用边界

Sequence Parallel 最适合的场景是：

你已经启用了 Tensor Parallel，但模型仍然被激活显存卡住，而且这些激活里有大量来自非 TP 区域的完整序列副本。

它不是唯一方案，更常见的是和其他技术组合使用。

| 方案 | 主要解决什么 | 优点 | 代价 | 适用边界 |
|---|---|---|---|---|
| 仅 TP | 参数与大矩阵计算切分 | 实现成熟，能解决大算子压力 | 非 TP 激活仍复制 | 模型能跑，但长序列显存偏紧 |
| TP + Sequence Parallel | 非 TP 激活切分 | 每卡激活约降到 $1/t$ | 增加集体通信 | TP 下显存瓶颈主要来自激活复制 |
| TP + Activation Checkpointing | 减少激活保存 | 不依赖额外网络带宽 | 反向更慢 | 网络一般、算力相对富余 |
| TP + SP + Selective Recompute | 同时压缩激活和重算范围 | 大模型训练整体更均衡 | 实现与调参更复杂 | 超大模型训练常见组合 |
| Context Parallel | 注意力与长上下文本身切分 | 支持更长序列 | 注意力通信复杂度更高 | 超长上下文成为主瓶颈时 |

最容易混淆的是 Sequence Parallel 和 Context Parallel。两者虽然都“和序列有关”，但解决的问题完全不同。

| 对比项 | Sequence Parallel | Context Parallel |
|---|---|---|
| 主要目标 | 减少非 TP 激活复制 | 切分长序列注意力本身 |
| 关注瓶颈 | LayerNorm、Dropout、残差等激活显存 | KV、attention 中间态、长上下文计算 |
| 是否依赖 TP 语境 | 通常是 | 不完全相同 |
| 适合的问题 | TP 已开，但激活显存仍高 | 序列长到注意力本身撑不住 |

可以用一句话区分：

- Sequence Parallel 关注“哪些激活不该在每张卡上重复保存”。
- Context Parallel 关注“超长序列的注意力本身如何分布到多卡”。

一个实用判断标准是：

1. 先分析显存构成，而不是先选并行名词。
2. 如果显存大头来自非 TP 激活复制，优先看 Sequence Parallel。
3. 如果显存和计算大头来自超长注意力本身，优先看 Context Parallel。
4. 如果两者都存在，就可能需要叠加。
5. 如果开 SP 后速度下降明显，再用 selective recomputation 或调整 TP 粒度做平衡。

再给一个决策表：

| 观察到的现象 | 更可能优先考虑的方案 |
|---|---|
| TP 已经开了，但 LayerNorm / 残差激活占显存大头 | Sequence Parallel |
| 序列继续拉长后，attention 的 KV 和中间态成为主瓶颈 | Context Parallel |
| 显存不足但网络较弱、算力较强 | Activation Checkpointing |
| 显存和吞吐都要兼顾，且系统实现能力足够 | TP + SP + Selective Recompute |

最终要记住的不是某个术语，而是选择逻辑：

并行策略应该跟着瓶颈走，而不是跟着概念走。

---

## 参考资料

- Megatron Core Developer Guide: Tensor Parallel and Sequence Parallel  
  https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html

- Megatron-LM: Context and Sequence Parallelism, DeepWiki  
  https://deepwiki.com/NVIDIA/Megatron-LM/4.5-context-and-sequence-parallelism

- Hugging Face Accelerate: Megatron-LM usage guide  
  https://huggingface.co/docs/accelerate/main/en/usage_guides/megatron_lm

- Reducing Activation Recomputation in Large Transformer Models, MLSys 2023  
  https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf

- Training Compute-Optimal Large Language Models, Appendix on Megatron-style parallel training details  
  https://arxiv.org/abs/2203.15556
