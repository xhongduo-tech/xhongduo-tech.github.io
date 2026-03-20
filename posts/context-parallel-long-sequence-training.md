## 核心结论

Context Parallel，简称 CP，就是把一条很长的输入序列按“序列长度”切成多段，分别放到多张 GPU 上。白话解释：不是每张卡都看完整篇文章，而是每张卡只保管自己负责的那一段。

它解决的问题很直接：长序列训练时，Attention 的显存压力主要来自 $Q,K,V$ 以及中间注意力分数，序列一长，单卡先爆的通常就是这里。CP 的做法是把总长度 $S$ 按 CP 组大小 $N$ 均分，每张卡只持有 $S/N$ 个 token 的局部片段。比如 $S=128K,\ N=8$，那么每张卡只需要处理 $16K$ 个 token。

CP 的关键不在“切分”本身，而在切分之后如何仍然得到“全局 Attention”的正确结果。这里用到 Ring Attention。Ring Attention 的意思是：每张卡保留本地的 Query，Key 和 Value 片段沿着 GPU 环形传递；每轮拿到一个新的 KV 块，就和本地 Q 计算一次局部注意力，再把结果按数值稳定的方式累积起来。等所有 KV 块都轮完一圈，得到的输出就等价于对完整序列直接做一次 Attention。

因此，CP 的本质可以概括为三句话：

| 项目 | 单卡长序列 Attention | Context Parallel |
|---|---|---|
| 序列存储 | 单卡持有全部 $S$ | 每卡只持有 $S/N$ |
| 额外通信 | 无 | Attention 阶段需要 KV 通信 |
| 正确性 | 直接全局计算 | 通过 Ring Attention 等价恢复全局结果 |
| 适用场景 | 序列较短、通信敏感 | 序列很长、显存先成为瓶颈 |

对初级工程师来说，可以先记住一个结论：CP 不是减少计算量，而是把原本单卡放不下的长上下文，拆成多卡协同完成，代价是 Attention 阶段多了跨卡通信。

---

## 问题定义与边界

问题定义很明确：当模型训练或推理的上下文长度变成 $32K$、$128K$、甚至 $1M$ 时，单卡已经无法在可接受的显存内完成标准自注意力计算，需要一种“保持结果不变，但把序列分到多卡上”的并行方式。

标准自注意力的核心公式是：

$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

其中 $Q,K,V$ 分别是 Query、Key、Value。白话解释：Query 表示“我想查什么”，Key 表示“每个位置提供什么索引”，Value 表示“真正要取出来的信息”。

如果序列长度是 $S$，隐藏维度是 $d$，那么仅从形状上看，注意力分数矩阵就接近 $S \times S$。这意味着随着 $S$ 增长，显存和带宽压力都会迅速上升。

CP 的边界也要讲清楚，不然很容易把它和别的并行策略混在一起：

| 边界问题 | CP 的处理方式 |
|---|---|
| 切分维度是什么 | 沿序列维度切分 |
| 哪些层需要跨卡通信 | 主要是 Attention |
| LayerNorm/MLP 是否跨卡 | 通常不需要，按本地 token 独立计算 |
| 是否改变模型参数量 | 不改变 |
| 是否减少总 FLOPs | 基本不减少，只重分布计算与存储 |
| CP 可以无限增大吗 | 不行，受通信和拓扑限制 |

这里的“按 token 独立计算”很重要。LayerNorm、逐位置 MLP、残差加法，这些操作只依赖当前位置的向量，不依赖整条序列，所以序列切开后，每张卡可以各算各的，不需要通信。真正需要全局视野的是 Attention，因为一个 token 可能要看见序列中所有历史 token。

玩具例子可以这样理解。假设一条序列长度是 8，CP 大小是 2，那么 GPU0 持有前 4 个 token，GPU1 持有后 4 个 token。对于 MLP，GPU0 只算前 4 个，GPU1 只算后 4 个即可；但对于 Attention，前 4 个 token 的 Query 可能还需要看到后 4 个 token 的 Key/Value，反过来也一样，所以 Attention 不能只做本地片段，必须把远端 KV 也纳入计算。

真实工程里，问题通常不是“我想把 8 个 token 分两半”，而是“我需要 128K 上下文做预训练，但 A100 单卡放不下完整 KV 和中间激活”。这时 CP 的价值才真正出现。

---

## 核心机制与推导

先统一记号。设总序列长度为 $S$，CP 组大小为 $N$，每张 GPU 持有长度为 $S/N$ 的局部片段。第 $i$ 张卡上的局部张量记为 $Q_i,K_i,V_i$。

### 1. 序列切分

原始序列：

$$
X = [X_0, X_1, \dots, X_{N-1}]
$$

其中每个 $X_i$ 是一个长度约为 $S/N$ 的连续片段。经过线性投影后，每张卡得到：

$$
Q_i = X_i W_Q,\quad K_i = X_i W_K,\quad V_i = X_i W_V
$$

白话解释：每张卡先只从自己手里的 token 生成局部的 $Q,K,V$。

### 2. 目标不是局部 Attention，而是全局 Attention

如果只做本地 Attention，那么第 $i$ 张卡只能算：

$$
\tilde O_i = \text{softmax}(Q_i K_i^T)V_i
$$

这不对，因为它只看见本地片段。正确目标应该是：

$$
O_i = \text{softmax}\left(Q_i [K_i^T, K_{i+1}^T, \dots, K_{i+N-1}^T]\right)[V_i, V_{i+1}, \dots, V_{i+N-1}]
$$

这里下标按环形取模。白话解释：第 $i$ 张卡上的 Query，最终要和全体卡的 Key/Value 交互，才能得到真正等价于全局序列的输出。

### 3. Ring Attention 的环形传递

Ring Attention 的核心流程是：

1. 每张卡先拿本地 $Q_i$ 和本地 $K_i,V_i$ 计算一轮局部贡献。
2. 然后把本地 $K_i,V_i$ 发给下一个 GPU，同时从上一个 GPU 接收一个新的 KV 块。
3. 收到新的 KV 块后，再用同一个 $Q_i$ 继续计算下一轮贡献。
4. 重复 $N$ 轮，直到所有 KV 块都被本卡的 $Q_i$ 看过一遍。

可以用一个极简图示表示：

```text
GPU0 -> GPU1 -> GPU2 -> ... -> GPU(N-1) -> GPU0
  ^                                      |
  |--------------------------------------|
```

传递的是 KV 块，不是 Q。原因很简单：本卡输出只关心“本地 Query 对全局 KV 的注意力”，所以固定本地 $Q_i$，让所有远端 KV 依次流过来，通信成本更可控。

### 4. 为什么分块后仍然等价

直接把所有 $Q_iK_j^T$ 拼起来再做 softmax 很简单，但 Ring Attention 是分轮计算。难点在于 softmax 不是线性的，不能简单把每轮结果直接相加。

解决方法是使用“在线 softmax 累积”。白话解释：每轮不仅保存当前输出，还要保存当前行最大值和归一化分母，这样多轮合并后，结果和一次性对全量 logits 做 softmax 是一致的。

对某个 Query 行，假设第 $t$ 轮拿到 logits 块 $L^{(t)}$，对应 value 块 $V^{(t)}$。维护三个量：

- $m^{(t)}$：截至当前轮的最大值
- $\ell^{(t)}$：归一化分母
- $a^{(t)}$：加权后的分子累积

更新公式可写成：

$$
m^{(t)} = \max(m^{(t-1)}, \max(L^{(t)}))
$$

$$
\ell^{(t)} = e^{m^{(t-1)}-m^{(t)}}\ell^{(t-1)} + \sum_j e^{L^{(t)}_j - m^{(t)}}
$$

$$
a^{(t)} = e^{m^{(t-1)}-m^{(t)}}a^{(t-1)} + \sum_j e^{L^{(t)}_j - m^{(t)}}V^{(t)}_j
$$

最终输出为：

$$
O = \frac{a^{(N)}}{\ell^{(N)}}
$$

这和 FlashAttention 的块式 softmax 思路是一致的，只不过 FlashAttention 的块来自同一张卡上的分块，而 Ring Attention 的块来自不同 GPU 顺序传递过来的 KV 片段。

### 5. 两个例子

玩具例子：总长度 $S=8$，CP=2。GPU0 持有 token 0 到 3，GPU1 持有 token 4 到 7。GPU0 的本地 Query 想得到全局输出，就要先看本地 $K_0,V_0$，再看来自 GPU1 的 $K_1,V_1$。GPU1 同理。两轮后就完成。

真实工程例子：LLaMA 类模型做 $128K$ 上下文训练，CP=8。每张卡只保留 $16K$ token 的局部激活和局部 KV，Attention 中通过 8 轮环形 KV 交换得到全局结果。这样做不会让单张卡存下完整 128K 的 KV，因此显存压力从“单卡必须承受全部长度”变成“单卡只承受局部长度，外加一份轮转缓冲”。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现。它不依赖多 GPU，而是在单机上模拟“分块 KV 逐轮累积”，验证结果和一次性全量 Attention 相同。

```python
import math
import torch

torch.manual_seed(0)

def full_attention(q, k, v):
    scores = q @ k.T / math.sqrt(q.size(-1))
    probs = torch.softmax(scores, dim=-1)
    return probs @ v

def ring_attention_sim(q_local, k_chunks, v_chunks):
    # 在线 softmax 累积：m 是行最大值，l 是分母，a 是分子
    n_q = q_local.size(0)
    d_v = v_chunks[0].size(-1)

    m = torch.full((n_q, 1), float("-inf"))
    l = torch.zeros((n_q, 1))
    a = torch.zeros((n_q, d_v))

    scale = math.sqrt(q_local.size(-1))

    for k_chunk, v_chunk in zip(k_chunks, v_chunks):
        logits = q_local @ k_chunk.T / scale
        block_max = logits.max(dim=-1, keepdim=True).values
        new_m = torch.maximum(m, block_max)

        exp_prev = torch.exp(m - new_m) * l
        exp_block = torch.exp(logits - new_m)

        new_l = exp_prev + exp_block.sum(dim=-1, keepdim=True)
        new_a = torch.exp(m - new_m) * a + exp_block @ v_chunk

        m, l, a = new_m, new_l, new_a

    return a / l

# 玩具例子：总长度 8，切成 2 个 chunk，模拟 CP=2
S = 8
D = 4
X = torch.randn(S, D)

Wq = torch.randn(D, D)
Wk = torch.randn(D, D)
Wv = torch.randn(D, D)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

# GPU0 持有前 4 个 token 的 query
q_local = Q[:4]

# 环形收到的两个 KV 块：先本地，再远端
k_chunks = [K[:4], K[4:]]
v_chunks = [V[:4], V[4:]]

out_ring = ring_attention_sim(q_local, k_chunks, v_chunks)
out_full = full_attention(q_local, K, V)

assert torch.allclose(out_ring, out_full, atol=1e-5)
print("ring attention simulation matches full attention")
```

这个例子验证了一件事：只要在线 softmax 的累积写对，按块处理 KV 的结果就和一次性全量计算相同。

工程里真正的实现不会像上面这样手写 Python 循环，而是放在高性能 Attention kernel 或框架包装层中。控制流程通常是下面这样：

```python
def context_parallel_attention_forward(q_local, k_local, v_local, cp_group):
    kv_buffer = (k_local, v_local)
    state = init_online_softmax_state(q_local)

    for _ in range(cp_group.size()):
        state = accumulate_attention(q_local, kv_buffer, state)
        send_to_next_rank(kv_buffer, cp_group)
        kv_buffer = recv_from_prev_rank(cp_group)

    return finalize_online_softmax(state)
```

这个流程里有三个工程重点：

| 阶段 | 主要动作 | 是否通信 |
|---|---|---|
| 本地投影 | 生成本地 $Q_i,K_i,V_i$ | 否 |
| 局部独立层 | LayerNorm、MLP、残差 | 否 |
| Attention 累积 | 本地 Q 对当前 KV 块做一次贡献计算 | 否 |
| Ring 轮换 | 把 KV 发给下一卡，接收上一卡 KV | 是 |

如果放到 Megatron 或 NeMo 这类框架里，配置上通常就是设置 `context_parallel_size`。框架会负责建立 CP 通信组，并在 Attention 路径里插入 KV 轮换和累积逻辑。用户需要理解的不是每一行底层 kernel，而是并行语义：Q 固定在本地，KV 在组内轮转，其他逐 token 层保持本地执行。

真实工程例子可以这样理解：你在 8 张 GPU 上训练 128K 上下文，配置 `context_parallel_size=8` 后，非 Attention 层仍像“每卡处理自己那 16K token”，只有进入 Attention 时，框架才启动 8 轮 KV 交换，把全局上下文拼回来。

---

## 工程权衡与常见坑

CP 最核心的收益是显存下降，但代价是通信上升。这个权衡不能模糊处理。

### 1. CP 越大，不一定越好

当 CP 从 2 增加到 4、8、16 时，每张卡持有的序列更短，局部显存压力更小；但对应地，Attention 里需要更多轮 KV 交换。若 GPU 之间互联不好，或者通信和计算无法重叠，吞吐反而可能下降。

### 2. 通信开销只在 Attention，不代表可以忽略

Attention 本来就是 Transformer 的核心热点层。把最重的层变成“计算 + 点对点通信”混合路径后，链路带宽、延迟、拓扑都会影响结果。NVLink 环境通常更适合 CP；如果是弱互联环境，收益会明显打折。

### 3. CP 组和其他并行维度要对齐

真实训练常常同时使用 DP、TP、PP、FSDP。这里最容易出错的是并行组划分不一致。白话解释：如果本该共享同一份权重的卡，被错误地分到了不同同步组，它们虽然都在跑“同一个模型”，但参数更新路径已经不一致了，最后会导致梯度异常或根本不收敛。

常见风险可以直接列成表：

| 坑 | 影响 | 规避方式 |
|---|---|---|
| CP 组划分错误 | 权重副本不一致，训练不收敛 | 先确定 DP/TP/PP 网格，再嵌入 CP |
| CP 盲目开太大 | 通信吞吐成为瓶颈 | 先测 2/4/8 的拐点，不要直接拉满 |
| 忽略通信重叠 | GPU 等待时间长 | 用异步 P2P、流水线化调度 |
| 只关注前向显存 | 反向激活仍可能爆 | 联合检查激活重算与梯度缓冲 |
| 把 CP 当成 Sequence Parallel | 策略选错，收益不明显 | 先确认瓶颈在长序列 Attention |

### 4. 数值稳定性不能省

Ring Attention 不是“每块 softmax 后直接加起来”。如果这么做，结果就是错的。必须用在线 softmax 累积最大值和分母，否则跨块合并会失真。长序列场景下，这个问题更严重，因为 logits 的动态范围更大。

### 5. 因果掩码也要跟分块逻辑一致

训练自回归模型时，未来 token 不能被看到。做分块 Ring Attention 时，不能因为 KV 是远端传来的，就忘记正确应用 causal mask。否则模型会看到不该看到的信息，训练虽然能跑，但语义已经错了。

---

## 替代方案与适用边界

CP 不是唯一方案，也不是默认最优方案。选型要看瓶颈究竟在哪里。

### 1. Sequence Parallelism

Sequence Parallelism 也是沿序列相关维度做拆分，但它的目标通常是配合 Tensor Parallel 降低激活开销，而不是专门解决超长上下文下的全局 KV 存储问题。白话解释：它更像是“已有张量并行后的内存优化”，不是“把 128K 序列硬拆到多卡上”的主方案。

### 2. Tensor Parallelism

Tensor Parallelism，简称 TP，就是把一个大矩阵按列或按行切到多卡。它擅长解决模型参数和矩阵乘规模过大问题，但对“序列太长导致 Attention KV 爆显存”并不总是最直接。若瓶颈是隐藏维度或参数矩阵，TP 有效；若瓶颈主要来自长序列，CP 更对症。

### 3. 稀疏注意力或压缩记忆

Sparse Attention、Memory Compression、局部窗口注意力这类方法，思路是“不再精确看全局所有 token”。它们的好处是理论和实践上都能减少复杂度，但代价是改变了原始 Attention 语义，训练和调参都更复杂。CP 则属于“保持全局精确 Attention，只改变并行执行方式”。

可以用一个决策矩阵快速判断：

| 场景 | 是否优先考虑 CP | 原因 |
|---|---|---|
| 序列只有 4K 或 8K | 否 | 单卡通常还能承受，通信不划算 |
| 序列 32K 到 128K，显存先爆在 Attention | 是 | 直接沿序列切分最有效 |
| 参数量过大但序列不长 | 否 | 更该优先 TP/FSDP |
| 想保留精确全局 Attention | 是 | CP 不改数学定义 |
| 可以接受近似 Attention | 视情况 | 稀疏或压缩方法可能更省算力 |
| 多卡之间互联很弱 | 谨慎 | CP 收益可能被通信吞掉 |

一个实用判断标准是：如果你的问题是“模型太大”，先看 TP/FSDP；如果你的问题是“上下文太长”，优先看 CP；如果两者都严重，就做组合并行。

对初级工程师来说，可以用一句话记忆适用边界：CP 主要解决“长序列放不下”，不是解决“一切分布式训练问题”。

---

## 参考资料

### 论文
- Context Parallelism for Scalable Million-Token Inference：提出 Ring Attention，用于把超长上下文的精确 Attention 分布到多设备上。
- FlashAttention 系列论文：理解在线 softmax 与块级累积的数值稳定性来源。

### 官方文档
- NVIDIA Megatron Core Developer Guide, Context Parallelism：说明 `context_parallel_size`、CP 组行为以及 KV 交互逻辑。
- NVIDIA NeMo / Megatron-LM 相关文档：用于查看 CP 与 TP、PP、DP 的组合方式。

### 源码与技术说明
- DeepWiki 的 Megatron-LM Context and Sequence Parallelism 页面：适合快速建立整体结构认知。
- PyTorch Discuss 上关于 TorchTitan 与长上下文训练的讨论：适合理解社区实现思路与工程细节。
- 工程博客如 tinkerings.dev、Medium 上的 LLaMA 长上下文训练文章：适合理解真实部署中的通信和拓扑权衡。
