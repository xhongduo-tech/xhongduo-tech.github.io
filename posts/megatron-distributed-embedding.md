## 核心结论

Megatron-LM 的 `VocabParallelEmbedding` 本质是把词表嵌入矩阵按词表维度切开。词表维度就是“第几个 token 对应哪一行”的那一维。若词表大小为 $V$、隐藏维度为 $H$，标准 embedding 参数量为

$$
P = V \times H
$$

当 tensor parallel 大小为 $N_{\text{TP}}$ 时，每张卡只保存约

$$
P_{\text{local}} = \frac{V \times H}{N_{\text{TP}}}
$$

个参数。

这不是“近似替代”，而是“分布式实现后的等价计算”。前向时，每张 GPU 只查自己负责的词表分片；查不到的 token 直接输出 0；随后通过 `AllReduce(sum)` 把各卡局部结果相加，得到与单卡 embedding 完全同形状、同数值的输出。反向时，梯度天然只会落到本卡持有的那些行，因此只更新本地分片，不需要同步整张表。

对新手可以这样理解：256k 个词像一本 256000 页的字典，TP=8 时 8 张卡每张只拿 32000 页。某个 token 来了，如果不在我的页码范围，我就返回全 0；所有卡把结果相加后，就等价于“整本字典都在一张卡上查了一次”。

以 $V=256000$、$H=12288$ 为例，总参数量约为

$$
256000 \times 12288 = 3{,}145{,}728{,}000
$$

约 31.46 亿参数。TP=8 后，每卡仅需保存约 3.93 亿参数。这就是 Megatron-LM 能把超大 embedding 放进有限显存的重要原因。

---

## 问题定义与边界

问题先定义清楚。Embedding 就是“把离散的 token 编号映射成连续向量”的层。离散编号可以理解为词表中的索引，连续向量可以理解为神经网络后续层能处理的数值表示。

它的参数矩阵形状是 $E \in \mathbb{R}^{V \times H}$。其中：

| 变量 | 含义 | 白话解释 |
|---|---|---|
| $V$ | Vocabulary size | 词表里一共有多少个 token |
| $H$ | Hidden size | 每个 token 要映射成多长的向量 |
| $N_{\text{TP}}$ | Tensor parallel size | 同一层被几张卡一起分担 |
| $P$ | Parameter count | 这一层总共多少参数 |

embedding 的麻烦在于，它随 $V$ 和 $H$ 线性增长，不像卷积层那样参数相对可控。只要词表和隐藏维度一起变大，它就会快速变成显存热点。

玩具例子先看小的。假设词表只有 8 个 token，隐藏维度是 4，那么 embedding 就是一个 $8 \times 4$ 的矩阵。TP=2 时，第 0 张卡保留前 4 行，第 1 张卡保留后 4 行。输入 token=6，只会在第 1 张卡命中，第 0 张卡返回全 0，最后相加得到正确结果。

真实工程里，数字会很大。Megatron 系列大模型常见配置是词表几十万、隐藏维度上万。比如 $V=256000$、$H=12288$，仅 embedding 就达到 31 亿级参数。若用 BF16，每个参数 2 字节，仅这一个层就需要大约 6 GB 显存，还没算梯度、优化器状态、激活值。若不切分，单卡很快吃不消。

这里的边界也要说清楚：

1. 讨论对象是 Megatron-LM / Megatron-Core 里的 tensor parallel embedding。
2. 目标是让前向输出与普通 embedding 一致，形状仍然是 `(batch, seq, hidden)`。
3. 这里说的是“按词表维度切分”，不是按隐藏维度切分，也不是数据并行。
4. 它通常与 tensor-parallel cross entropy 配套，因为输出 logits 也常按词表维切分。

问题边界图可以用文字表示：

```text
不切分:
Embedding 参数 = V × H
显存压力全部压在单卡

TP 切分后:
每卡参数 = (V / TP) × H
显存下降近似线性
代价: 每次前向/反向增加通信
```

---

## 核心机制与推导

Megatron-LM 的关键做法，是先给每张卡分一个连续的词表范围。连续范围的意思是“第几号 token 到第几号 token 归我管”。常见接口名是 `vocab_range_from_global_vocab_size`，它根据全局词表大小和 rank 计算每张卡的区间：

$$
[v_{\text{start}}^{(r)}, v_{\text{end}}^{(r)})
$$

其中 $r$ 是当前 tensor parallel rank。

### 1. 前向：局部查表 + 全局求和

设全局 embedding 矩阵为 $E \in \mathbb{R}^{V \times H}$，它被拆成 $N_{\text{TP}}$ 份：

$$
E = 
\begin{bmatrix}
E^{(0)} \\
E^{(1)} \\
\cdots \\
E^{(N_{\text{TP}}-1)}
\end{bmatrix}
\quad,\quad
E^{(r)} \in \mathbb{R}^{\frac{V}{N_{\text{TP}}} \times H}
$$

对于输入 token id $x$，只有一个 rank 满足：

$$
v_{\text{start}}^{(r)} \le x < v_{\text{end}}^{(r)}
$$

该 rank 会把全局 id 映射成局部 id：

$$
x_{\text{local}} = x - v_{\text{start}}^{(r)}
$$

然后查表得到向量；其他 rank 直接返回零向量。于是每张卡得到一个局部输出 $y^{(r)}$，且同一 token 只有一张卡非零：

$$
y = \sum_{r=0}^{N_{\text{TP}}-1} y^{(r)}
$$

这就是为什么 `AllReduce(sum)` 能恢复完整 embedding。它不是在“聚合多个不同答案”，而是在“把唯一正确答案从零背景里捞出来”。

### 2. 反向：只更新本地命中的行

embedding 的梯度更新有个重要特征：它对参数矩阵是稀疏行更新。稀疏的意思是“只改少数几行，不会整张表一起变”。如果一个 batch 里只出现了若干 token，那么只这些 token 对应的行有梯度。

设上游传来的梯度为 $\frac{\partial L}{\partial y}$。由于每个 token 的前向只在一个 rank 上命中，因此对应的参数梯度也只会落在那个 rank 的本地行上。换句话说：

- 本卡负责的 token 行，会累积梯度并更新；
- 不属于本卡范围的 token，本卡根本没有那一行参数，也不会更新。

这就是“前向全局等价，反向局部更新”的核心。

### 3. 数值例子

看题目里的配置：

| 项目 | 数值 |
|---|---|
| 词表大小 $V$ | 256000 |
| 隐藏维度 $H$ | 12288 |
| TP 大小 $N_{\text{TP}}$ | 8 |
| 总参数 $P$ | 3,145,728,000 |
| 每卡参数 $P/8$ | 393,216,000 |

这里每卡大约保存 3.93 亿参数。若使用 BF16，单参数 2 字节，则每卡 embedding 权重约 786 MB。若不切分，单卡则约 6.29 GB，仅这一层就已经很重。

### 4. AllReduce 流程文本图

```text
输入 token ids
   ↓
每张卡判断: token 是否在我的 [v_start, v_end)
   ↓
在范围内: 做 local embedding lookup
不在范围内: 输出 0 向量
   ↓
所有卡得到局部输出 (B, S, H)
   ↓
AllReduce(sum)
   ↓
每张卡都拿到完整输出 (B, S, H)
```

这个设计与后续 tensor-parallel 层兼容，因为每张卡在前向结束后都拥有同样形状的 embedding 输出。

真实工程例子可以直接落到大模型训练。SC21 中 Megatron 的 145B 模型采用 TP=8、hidden size=12288 的配置。此时 embedding 不切分会明显加重单卡显存压力，而词表并行后，embedding 被压到每卡一份分片，才有空间留给注意力、MLP、优化器状态和激活检查点。

---

## 代码实现

下面先用一个可运行的 Python 玩具实现模拟 `VocabParallelEmbedding` 的核心行为。它不是 PyTorch 分布式代码，但机制一致。

```python
import numpy as np

def vocab_range(global_vocab_size, tp_size, rank):
    per = global_vocab_size // tp_size
    start = rank * per
    end = start + per
    return start, end

def local_lookup(weight_shard, input_ids, start, end):
    batch, seq = input_ids.shape
    hidden = weight_shard.shape[1]
    out = np.zeros((batch, seq, hidden), dtype=weight_shard.dtype)

    for i in range(batch):
        for j in range(seq):
            token = input_ids[i, j]
            if start <= token < end:
                local_id = token - start
                out[i, j] = weight_shard[local_id]
    return out

# 玩具配置
V = 8
H = 3
TP = 2

# 全量 embedding
full_weight = np.arange(V * H).reshape(V, H)

# 切分到两张卡
shard0 = full_weight[0:4]
shard1 = full_weight[4:8]

input_ids = np.array([[0, 3, 4, 7]])

s0, e0 = vocab_range(V, TP, 0)
s1, e1 = vocab_range(V, TP, 1)

out0 = local_lookup(shard0, input_ids, s0, e0)
out1 = local_lookup(shard1, input_ids, s1, e1)

# 模拟 all_reduce(sum)
final_out = out0 + out1

# 与单卡结果对比
expected = full_weight[input_ids]

assert np.array_equal(final_out, expected)
assert np.array_equal(final_out[0, 0], full_weight[0])
assert np.array_equal(final_out[0, 2], full_weight[4])
print("ok")
```

这个例子里，token 0 和 3 由 rank 0 命中，token 4 和 7 由 rank 1 命中。两个局部结果相加后，和单卡查表完全一致。

如果换成 Megatron-LM 风格的伪代码，逻辑通常是这样：

```python
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, tp_group):
        super().__init__()
        self.tp_size = get_world_size(tp_group)
        self.rank = get_rank(tp_group)
        self.vocab_start, self.vocab_end = vocab_range_from_global_vocab_size(
            num_embeddings, self.rank, self.tp_size
        )
        self.num_embeddings_per_partition = self.vocab_end - self.vocab_start
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

    def forward(self, input_ids):
        # 1. 找出哪些 token 属于本卡负责范围
        input_mask = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)

        # 2. 全局 id 转局部 id；不属于本卡的先放成 0，后面再清零
        masked_input = input_ids.clone() - self.vocab_start
        masked_input[input_mask] = 0

        # 3. 本地查表
        local_output = F.embedding(masked_input, self.weight)

        # 4. 不属于本卡范围的位置置 0
        local_output[input_mask] = 0.0

        # 5. 所有卡求和，恢复完整输出
        torch.distributed.all_reduce(local_output, op=torch.distributed.ReduceOp.SUM)

        return local_output
```

这里最容易误解的一点是：为什么不用 `all_gather` 把所有 embedding 行收集回来？原因是没必要。

- `all_gather` 的语义是“把每卡不同分片拼起来”。
- 这里每个 token 的正确向量只存在于一张卡上，其他卡全是 0。
- 所以按元素求和就能恢复结果，`all_reduce(sum)` 更直接，也更符合 Megatron 的并行接口设计。

输入输出关系可以概括成下表：

| 阶段 | 张量形状 | 含义 |
|---|---|---|
| 输入 `input_ids` | `(B, S)` | token 编号 |
| 本地分片 `weight` | `(V/TP, H)` | 当前 rank 持有的 embedding 行 |
| 局部输出 `local_output` | `(B, S, H)` | 只有命中本 rank 的位置非零 |
| `AllReduce(sum)` 后输出 | `(B, S, H)` | 与普通 embedding 一致 |

新手版本的口头描述可以直接记成一句话：`if token not in my range: return zeros`。

---

## 工程权衡与常见坑

这个方案很有效，但不是没有代价。核心代价是通信。显存省下来了，前向和部分反向会多一次跨卡同步。是否划算，取决于互联速度、TP 大小、microbatch 大小和后续层的并行结构。

常见坑如下：

| 问题 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| `vocab_size` 不能整除 TP | loss 异常、logits 对不上 | 每卡词表分片不齐，后续并行 softmax 对齐失败 | 把 vocab pad 到 TP 的倍数 |
| 忘记处理越界 token | 查表错位 | 全局 id 没减 `vocab_start` | 显式做 global-to-local 映射 |
| 未清零非本地 token 输出 | 输出污染 | 非命中 token 被错误查表 | 对 mask 位置强制置 0 |
| padding token 处理不一致 | 梯度异常或无效更新 | pad 行参与训练 | 统一 pad id 和 loss mask |
| TP 设太大 | 吞吐下降 | 通信占比上升 | 结合互联带宽调 TP，而非只看显存 |

题目里提到的典型工程坑，是词表大小与 TP 不整除。比如 `vocab_size=500003`、`TP=8`。此时每卡无法得到完全对齐的分片，尤其和并行 cross entropy 一起用时，logits 范围容易错位，训练可能直接发散。常见修复方式是把词表补到 `500008`，也就是 pad 到 8 的倍数。

这个坑为什么严重？因为 embedding 并行通常不是孤立存在的，它和输出层 logits 的词表切分往往共享同一套分片逻辑。输入 embedding 切一套、输出 logits 切另一套，模型就会在前后两端出现语义不一致。

另一个权衡是 AllReduce 成本。embedding 输出张量形状是 `(B, S, H)`，当 batch、序列长度、隐藏维度都大时，这次求和通信会很可观。因此它适合高带宽、低时延互联，比如 NVLink、InfiniBand 环境；若跨机网络较弱，TP 开太大会把节省的显存换成明显的吞吐损失。

---

## 替代方案与适用边界

`VocabParallelEmbedding` 不是唯一方案，只是在 Megatron 这种同构 GPU 集群里非常自然。

下面给出几种常见思路对比：

| 方案 | 适用环境 | 通信模式 | 优点 | 缺点 |
|---|---|---|---|---|
| 词表维 Tensor Parallel | 同构 GPU、低延迟互联 | `AllReduce(sum)` | 与 Megatron 其他 TP 层一致，显存线性下降 | 通信依赖强，要求词表切分对齐 |
| 参数服务器式 Embedding Sharding | 多机分布式、异构环境 | 请求-响应式拉取 | 更灵活，可跨不同设备扩展 | 查询延迟高，热点 token 易拥塞 |
| Pipeline 切层 | 模型按层切分 | 激活传递 | 层间职责清晰 | 不能直接解决单层 embedding 过大问题 |
| CPU / NVMe Offload | 显存极紧张场景 | 按需搬运 | 能突破显存上限 | 延迟高，训练吞吐差 |

给新手的直观解释：

- Megatron 的办法像是“每个人只保管字典的一部分页码，但每次查完大家立刻把答案合并”。
- 参数服务器的办法像是“字典放在不同服务器上，谁需要哪个词，就向对应服务器发请求拿结果”。

两者的差别在于，Megatron 假设参与并行的设备关系很紧密，能高效做集体通信；参数服务器则更适合机器分散、职责独立的环境，但单次查询延迟通常更高。

因此，Megatron-LM 的这套 embedding 并行更适用于以下边界：

1. 训练集群是同构 GPU，且 TP 组内网络很好。
2. 模型整体已经采用 tensor parallel，embedding 只是其中一环。
3. 需要与并行输出层、并行 cross entropy 保持一致切分。
4. 更看重训练吞吐和实现一致性，而不是极端灵活的跨设备部署。

如果场景是多租户服务、异构硬件，或者 embedding 表远大于计算主体，参数服务器、混合并行、甚至 embedding offload 可能更合理。

---

## 参考资料

1. NVIDIA Megatron-Core API Documentation，`VocabParallelEmbedding` 与 tensor-parallel layers 说明。适合确认接口、前向行为与并行层设计。
2. NVIDIA Megatron-LM / Megatron-Core 相关实现与 issue 讨论，尤其词表切分、并行 cross entropy、`vocab_size` 补齐问题。适合排查工程错误。
3. SC21 论文《Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM》。适合理解 145B 训练中的 TP 配置和系统级收益。
4. 技术博客与代码解析文章，对 `AllReduce(sum)` 恢复 embedding 输出、局部 mask 与 local id 映射有直观示例。适合建立实现直觉。
5. Megatron-LM 相关 illustrated notes。适合理解 embedding、logits、tensor parallel 之间的整体数据流关系。
