## 核心结论

Sequence Parallelism，简称 SP，可以理解为“把一条长序列的 token 分给多张 GPU 保管和计算”。它的核心作用不是减少参数，也不是减少优化器状态，而是减少**激活**占用。激活就是前向传播时为了反向传播暂存的中间结果，长序列训练里它常常比参数更早把显存撑满。

设输入张量为 $X \in \mathbb{R}^{S \times B \times H}$，其中 $S$ 是序列长度，$B$ 是 batch size，$H$ 是 hidden size。SP 会把序列维切成 $t$ 份，每张卡只保存

$$
X_i \in \mathbb{R}^{(S/t) \times B \times H}
$$

这意味着，当 `t=4` 时，每张 GPU 常驻保存的 token 激活大约缩小为原来的四分之一。这里说的是“常驻布局”，不是说所有算子都永远只看局部 token。遇到必须看到完整序列的算子，仍然要通过通信把分片临时拼回去。

一个最短结论可以直接记住：

| 方案 | 切分对象 | 主要节省什么 | 典型通信 |
|---|---|---|---|
| Data Parallelism | 数据批次 | 不直接省单卡显存 | 梯度同步 |
| Tensor Parallelism | 权重或 hidden 维 | 参数、部分激活 | all-reduce |
| Sequence Parallelism | 序列维 | 激活 | all-gather / reduce-scatter |

玩具例子很直接。假设 `S=8, t=2`，原来每张卡都要保留 8 个 token 的激活。开启 SP 后，GPU0 只保留 token `0~3`，GPU1 只保留 token `4~7`。局部能算的就在本地算，必须看全序列时再临时通信。这就是它支持超长上下文训练的根本原因。

---

## 问题定义与边界

SP 解决的问题是：**长序列训练时，激活内存过大**。这里的“长序列”通常指 4k、8k、32k 甚至更长上下文。模型不一定是算不动，很多时候是中间结果放不下。

对初级工程师，一个常见误解是：既然分布式训练已经把模型摊到多卡上了，为什么还会爆显存。原因在于，参数只是一部分，训练时还要保存大量中间张量。Transformer 在长序列下，许多层的激活大小会随着 $S$ 线性甚至更敏感地增长，最终先卡死的是激活，而不是权重文件。

SP 的边界也必须说清楚：

| 维度 | SP 是否直接优化 | 说明 |
|---|---|---|
| 激活内存 | 是 | 主要收益来源 |
| 参数内存 | 否 | 参数仍按原并行方式存放 |
| 优化器状态 | 否 | Adam 的一阶二阶状态不变 |
| 通信总量 | 不一定显著下降 | 主要改变通信形态和时机 |
| 长序列可训练性 | 是 | 尤其适合激活成为瓶颈时 |

因此，SP 不是“万能省显存开关”。如果你的瓶颈是参数过大，比如单个模型权重已经塞不进单卡，那么更优先的是 Tensor Parallelism、FSDP 或 ZeRO 这类方案。如果瓶颈是优化器状态，也要找针对状态切分的策略。SP 只针对“每层中间结果太大”这一类问题。

还有两个边界条件常被忽略。

第一，$t$ 是序列切分数，不是 batch 切分数。也就是说，SP 切的是一条样本内部的 token 轴，不是把不同样本分给不同卡。

第二，若 $S \bmod t \neq 0$，通常需要 padding。padding 就是先补齐到能整除的长度，再切片；否则 shape 会不对齐，通信和 kernel 实现都会变复杂。

---

## 核心机制与推导

SP 的机制可以概括成一句话：**局部算子本地算，全局算子通信后再算，然后把输出重新切回局部布局**。

这里“局部算子”是指只依赖单个 token 或本地张量统计的算子，例如 LayerNorm。LayerNorm 可以理解为“对每个 token 的特征维做标准化”，它不需要看到别的 token，因此天然适合在本地分片上执行。Dropout 也是类似情况，它只是对本地激活做随机失活。

相反，“全局算子”是指它的输入或输出语义要求完整序列布局，或者它和 Tensor Parallelism 的输出归并耦合在一起。这时就不能继续只拿着半截序列硬算。

形式化地写，输入切分为：

$$
X \in \mathbb{R}^{S \times B \times H}
\quad \rightarrow \quad
X_i \in \mathbb{R}^{(S/t) \times B \times H}, \; i=0,\dots,t-1
$$

本地算子直接作用在分片上：

$$
Y_i = \text{LN}(X_i), \quad Z_i = \text{Dropout}(Y_i)
$$

如果后续某一步需要完整序列，就执行 all-gather。all-gather 可以理解为“每张卡把自己的局部片段发给所有卡，最终每张卡都拼出完整序列”：

$$
Z = AG(Z_0, Z_1, \dots, Z_{t-1})
$$

如果后续某一步的输出还要回到分片布局，则执行 reduce-scatter。reduce-scatter 可以理解为“先把各卡待合并结果按位置求和，再把结果切回各卡各自负责的一段”：

$$
O_i = RS(O^{(0)}, O^{(1)}, \dots, O^{(t-1)})
$$

很多资料会强调一个重要关系：all-reduce 本质上可以拆成 reduce-scatter 加 all-gather。这意味着 SP 不是发明一种全新通信，而是在原本并行训练已有的通信图上，重新安排“何时完整、何时分片”的激活布局。

看一个玩具例子。设 `S=8, B=1, H=2, t=2`。

- GPU0 持有 `X_0`，shape 是 `[4,1,2]`，对应 token `0~3`
- GPU1 持有 `X_1`，shape 是 `[4,1,2]`，对应 token `4~7`

处理流程可以写成：

1. GPU0、GPU1 各自在本地对 `[4,1,2]` 做 LayerNorm 和 Dropout。
2. 如果接下来某个模块要求完整序列，则两张卡做 all-gather，各自拿到 `[8,1,2]`。
3. 执行该全局模块。
4. 若后续继续按 SP 布局走，则把输出通过 reduce-scatter 切回 `[4,1,2]`。

这个流程的关键不是数学复杂，而是**张量布局切换**。训练系统要持续知道：当前张量到底是“完整序列布局”还是“序列分片布局”。很多 bug 都不是算子公式错，而是布局状态错。

真实工程例子更典型。训练长上下文 GPT 时，常见配置是 `TP=4/8 + sequence_parallel=True`。这样做的目标不是减少参数，而是让 8k、16k、32k 上下文下的激活更容易塞进显存，并减少必须做 activation checkpointing 的范围。activation checkpointing 可以理解为“前向时少存，反向时重算”，它能省显存，但会增加计算时间。SP 的价值就在于：先把激活常驻内存压低，再把不得不重算的部分缩到更小。

---

## 代码实现

实现 SP 时，重点不是写出复杂算法，而是把三件事做对：

| 代码点 | 要求 |
|---|---|
| shape 管理 | 明确哪些张量是 `[S/t,B,H]`，哪些是 `[S,B,H]` |
| 通信时机 | 只在确实需要完整序列时通信 |
| 结果回切 | 全局算子之后及时切回分片布局 |

下面先给一个可运行的 Python 玩具实现。它不依赖真实多卡通信，而是用列表模拟 `all-gather` 和 `reduce-scatter` 的语义，方便看清数据流。

```python
import math

def split_sequence(x, parts):
    assert len(x) % parts == 0
    chunk = len(x) // parts
    return [x[i * chunk:(i + 1) * chunk] for i in range(parts)]

def all_gather(chunks):
    full = []
    for c in chunks:
        full.extend(c)
    return [full[:] for _ in chunks]

def reduce_scatter(full_outputs, parts):
    # 这里模拟“先求和再按序列切分”
    # full_outputs: 每个 rank 都有一份完整输出，形状一致
    assert len(full_outputs) > 0
    length = len(full_outputs[0])
    for out in full_outputs:
        assert len(out) == length

    reduced = [0.0] * length
    for out in full_outputs:
        for i, v in enumerate(out):
            reduced[i] += v

    return split_sequence(reduced, parts)

def local_layer_norm(chunk):
    # 简化版 LN：对每个 token 的特征向量做归一化
    result = []
    for token in chunk:
        mean = sum(token) / len(token)
        var = sum((v - mean) ** 2 for v in token) / len(token)
        std = math.sqrt(var + 1e-5)
        result.append([(v - mean) / std for v in token])
    return result

def global_op(full_seq):
    # 一个假设需要完整序列的操作：把每个 token 加上全序列均值
    hidden = len(full_seq[0])
    seq_mean = [sum(token[h] for token in full_seq) / len(full_seq) for h in range(hidden)]
    return [[token[h] + seq_mean[h] for h in range(hidden)] for token in full_seq]

# toy input: S=8, H=2, t=2
x = [
    [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
    [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0],
]

parts = 2
chunks = split_sequence(x, parts)
assert len(chunks) == 2
assert len(chunks[0]) == 4 and len(chunks[1]) == 4

# 1) local ops
local_chunks = [local_layer_norm(c) for c in chunks]

# 2) all-gather to full sequence
gathered = all_gather(local_chunks)
assert len(gathered) == 2
assert len(gathered[0]) == 8

# 3) global op on each rank
full_outputs = [global_op(full) for full in gathered]

# 4) reduce-scatter back
scattered = reduce_scatter(full_outputs, parts)
assert len(scattered) == 2
assert len(scattered[0]) == 4 and len(scattered[1]) == 4

print("rank0 local tokens:", len(scattered[0]))
print("rank1 local tokens:", len(scattered[1]))
```

这个例子里，`local_layer_norm` 在分片上本地执行；`global_op` 则故意设计成依赖全序列均值，强制展示为什么必须先 all-gather。最后再 reduce-scatter 回去，保持每张卡只持有自己的 token 片段。

如果换成真实工程中的 PyTorch 分布式代码，思路并不变，只是通信原语来自 `torch.distributed`：

```python
# x: [S/t, B, H] on each rank

x = layer_norm(x)          # local op
x = dropout(x)             # local op

xs = [torch.empty_like(x) for _ in range(world_size)]
dist.all_gather(xs, x, group=group)
x_full = torch.cat(xs, dim=0)    # [S, B, H]

y_full = global_op(x_full)       # op requiring full sequence

input_for_rs = y_full.contiguous()
y_local = torch.empty_like(x)
dist.reduce_scatter_tensor(y_local, input_for_rs, group=group)
```

真实工程里要额外关注两点。

第一，某些层是否真的“需要完整序列”，取决于实现细节，而不是看名字拍脑袋判断。第二，和 Tensor Parallelism 组合时，输出布局常常会在“TP 分片”和“SP 分片”之间切换，所以 shape 注释必须写清楚，否则调试成本很高。

---

## 工程权衡与常见坑

SP 的收益主要来自显存下降，但它不是免费午餐。只要做了更多布局切换和通信，就会带来实现复杂度与性能权衡。

先看收益和代价：

| 维度 | SP 的典型效果 |
|---|---|
| 显存占用 | 明显下降，尤其是长序列 |
| 训练吞吐 | 可能提升，也可能因通信受限 |
| 实现复杂度 | 中等到高 |
| 调试难度 | 高于纯 DP |
| 长序列收益 | 很高 |
| 短序列收益 | 可能不明显 |

最常见的坑有五类。

第一，忘记在全局依赖层前做 all-gather。后果最危险，因为模型可能不会报错，只是 silently wrong。也就是代码能跑，但语义错了。比如本应依赖完整序列统计的操作，只看到了半截 token，结果训练变差，却不容易定位。

第二，误以为 SP 会减少参数显存。不会。参数量、优化器状态、梯度副本这些问题，SP 都不是直接解法。如果判断错瓶颈，就会选错优化方向。

第三，认为通信成本可以忽略。SP 常见通信是 `all-gather` 和 `reduce-scatter`。从大 O 量级看，它未必比原本的 TP 通信更夸张，但在真实集群上，kernel latency、链路带宽、拓扑结构都会影响效果。特别是小 batch、强同步场景，通信延迟容易放大。

第四，忽略 padding。若 `S` 不能被切分数整除，很多实现会先补齐到最近的倍数。补齐后不仅要处理通信，还要在 loss 或 mask 中消除 padding 对训练的影响。

第五，只开 TP 不开 SP，却期待激活显存明显下降。在 Megatron-LM 风格实现里，SP 通常就是 TP 的配套优化。TP 解决的是参数和部分计算分摊，SP 进一步把激活布局切开，两者联用才更完整。

真实工程例子里，训练 8k 或 32k 上下文 LLM 时，经常会遇到这样一个现象：参数能装下，batch 也不算太大，但一开长序列显存就炸。此时单纯压 batch size 常常不够，因为问题根源不是样本数，而是单样本内部序列太长。SP 的价值就在这里非常直接。

---

## 替代方案与适用边界

SP 适合的前提是：**长序列、激活是主要瓶颈、并且训练系统已经在用或计划用 Tensor Parallelism**。如果这三个条件不成立，收益可能就不值得额外复杂度。

可以和其他方案对比着看：

| 方案 | 适合什么问题 | 主要节省什么 | 主要代价 |
|---|---|---|---|
| Sequence Parallelism | 长序列激活过大 | 激活 | 通信、实现复杂度 |
| Activation Checkpointing | 前向缓存过多 | 激活 | 反向重算 |
| Tensor Parallelism | 参数或单层计算过大 | 参数、部分激活 | 通信 |
| Data Parallelism | 扩大总体吞吐 | 不直接省单卡显存 | 梯度同步 |
| Pipeline Parallelism | 层数过深、单卡放不下整网 | 层驻留压力 | pipeline bubble |

判断是否该上 SP，可以按下面的逻辑做。

如果显存报告显示参数和优化器状态占了大头，那优先考虑参数切分或状态切分方案，而不是 SP。

如果显存主要爆在中间激活，尤其在序列长度翻倍时显存明显上升，那 SP 非常值得评估。

如果序列本来很短，比如 512 token 左右，而集群通信又一般，那么 SP 可能得不偿失。因为这时节省的激活不大，通信与实现开销反而更突出。

如果已经用了 activation checkpointing，也不代表 SP 没价值。两者经常是互补关系：SP 先把常驻激活压低，checkpointing 再进一步减少必须保存的部分。工程上常见的最优点不是二选一，而是组合使用。

一句实用判断可以记住：**当“激活”比“参数”更先成为长序列训练的上限时，优先考虑 SP；当“参数/状态”先成为上限时，优先看别的并行方案。**

---

## 参考资料

| 想确认什么 | 优先看哪里 |
|---|---|
| SP 的原理、收益和与重算的关系 | Korthikanti 等人的 MLSys 2023 论文 |
| `all_gather`、`reduce_scatter_tensor` 的 API 语义 | PyTorch 官方文档 |
| 在 Megatron-LM 训练栈中的使用方式 | Hugging Face Accelerate 的 Megatron-LM 指南 |
| 真实工程实现细节 | NVIDIA/Megatron-LM 源码 |

1. Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, MLSys 2023  
   https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html

2. PyTorch Distributed Documentation: `all_gather`, `reduce_scatter_tensor`  
   https://docs.pytorch.org/docs/stable/distributed.html

3. Hugging Face Accelerate, Megatron-LM usage guide, including Sequence Parallelism  
   https://huggingface.co/docs/accelerate/main/en/usage_guides/megatron_lm

4. NVIDIA Megatron-LM repository  
   https://github.com/NVIDIA/Megatron-LM
