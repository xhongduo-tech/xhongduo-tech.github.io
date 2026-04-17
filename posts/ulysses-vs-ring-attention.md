## 核心结论

Ulysses 和 Ring Attention 解决的是同一个问题：当序列很长时，单张 GPU 放不下完整注意力，需要把计算和通信拆到多张卡上。两者的区别不在“都能并行”，而在“沿哪个维度切分、通信怎么走、瓶颈落在哪里”。

术语先解释：**head** 是多头注意力里并行的多个“小注意力通道”；**All-to-All** 是一种集体通信，意思是每张卡都同时给其他所有卡发一部分数据；**ring** 是环形通信，意思是每张卡只和前后相邻两张卡传数据。

结论可以压缩成一句话：

- **Ulysses** 沿 head 维度分工，先用两次 All-to-All 把 QKV 从“按序列分片”改成“按 head 分片”，每张 GPU 暂时拿到全序列，但只算自己负责的部分 head。
- **Ring Attention** 沿序列维推进，本地 Q 不动，KV 在 GPU 环上轮流传递，每张 GPU 逐步看完全局上下文。

如果记一个最重要的工程判断，就是下面这张表。

| 维度 | Ulysses | Ring Attention |
|---|---|---|
| 主要切分方向 | head | sequence |
| 通信模式 | 两次 All-to-All | KV 环形传递 |
| 典型通信量 | $O(\frac{8}{N}bsd)$ | $O(4bsd)$ |
| 是否依赖 head 数 | 强依赖 | 基本不依赖 |
| 是否容易扩到超长序列 | 有上限 | 更强 |
| 更适合 | NVLink/NVSwitch、高带宽、head 充足 | head 不够整除、卡数不规则、上下文继续变长 |

玩具例子可以这样理解。把每张 GPU 想成一位老师。Ulysses 是先把全班作业重新分发，让每位老师都看到全班，但只批改自己负责的题型；Ring Attention 则不改学生归属，每位老师保留自己这部分学生，只把别班的答卷一圈一圈传过来看。

---

## 问题定义与边界

问题定义很明确：在完整 self-attention 不做近似的前提下，让长度 $s$ 很大的上下文还能训练或推理。这里的“完整”很重要，意思是每个 token 仍然能看到全序列，而不是只看局部窗口。

设：

- $b$ 表示 batch size
- $s$ 表示序列长度
- $h$ 表示 attention head 数
- $d$ 表示 hidden size，通常有 $d=h\times d_{head}$
- $N$ 表示并行用到的 GPU 数，或对应的序列并行度

边界条件主要有三类。

| 问题 | Ulysses 限制 | Ring Attention 限制 |
|---|---|---|
| head 数是否够分 | 必须尽量让 head 能被并行度整除；纯 Ulysses 还受 $N \le h$ 约束 | 基本不受 head 数限制 |
| 网络拓扑是否够快 | 对 All-to-All 非常敏感，慢网络会放大代价 | 只和邻居传输，通常更稳 |
| 序列能否继续增长 | 可扩展，但会被 head 数和并行布局卡住 | 理论上更容易继续扩到更长序列 |

新手最容易误解的一点是：**Ulysses 不是“通信与序列长度完全无关”**。更准确的说法是，它的通信不再像传统 all-gather 那样随着“设备数和序列一起放大”而恶化得那么快，并且在 head 足够、布局合适时单位设备的通信更容易被摊薄。Ring Attention 的通信则更直接，序列越长，需要传递的 KV 总量越大。

举一个简单边界例子。8 卡集群、12 个 head：

- 对 Ulysses，12 不能很好地均分给 8 张卡，负载会难看，甚至根本不满足实现要求。
- 对 Ring Attention，只要序列能切分，7 卡、8 卡、任意 head 数都还能工作，只是每轮要等待上一站的 KV。

---

## 核心机制与推导

### 1. Ulysses 怎么做

Ulysses 的起点是：非注意力层可以按序列分片独立算，但注意力层要求“看见全序列”。所以它在进入 attention 前做一次重排。

流程可以概括成三步：

1. 每张卡原本只持有本地序列块的 Q/K/V。
2. 通过 **pre-attention All-to-All**，把数据从“按序列切”改成“按 head 切”。
3. 每张卡现在拿到全序列，但只有部分 head，于是本地完成完整 attention。
4. 通过 **post-attention All-to-All** 再把输出改回原来的序列分片布局。

可以把它看成“先换装，再计算，再换回去”。

通信量常写成：

$$
C_{\text{Ulysses}} = O\!\left(\frac{8}{N}bsd\right)
$$

这里的直觉是：前向和反向都要做两次重排，Q/K/V 与输出梯度都会参与通信，所以常数项来自多次张量交换。关键不是常数 8 本身，而是 **它能被并行度 $N$ 摊薄**。但代价是 $N$ 不能无限加，因为你最终还是要把 head 分给具体 GPU。

### 2. Ring Attention 怎么做

Ring Attention 的思路不同。它不把数据大规模重排成 head-parallel，而是保持本地 Q 块不动，只让 KV 块在 GPU 之间环形流动。

流程如下：

1. 每张卡保留自己的 $Q_i, K_i, V_i$。
2. 先算一次本地 $Q_i$ 对本地 $K_i,V_i$ 的注意力。
3. 然后把本地 KV 发给下一张卡，同时从上一张卡收到新的 KV。
4. 重复这个过程，直到本地 $Q_i$ 看过所有卡的 KV。

于是每张 GPU 最终都完成“自己这段 query 对全局 key/value”的精确注意力。

它的通信量常写成：

$$
C_{\text{Ring}} = O(4bsd)
$$

这里没有 $\frac{1}{N}$ 这一项，原因是它不是一次大规模全互连重分发，而是固定模式的分步环传。优点是对 head 数不敏感，缺点是每一步都依赖前一步送来的 KV，存在串行等待。

### 3. 两者的本质差异

- Ulysses 是“先全局洗牌，再本地完整计算”。
- Ring 是“本地不动，远端 KV 轮流送来”。

玩具例子：有 4 台机器、16 个 token、8 个 head。

- Ulysses：每台机器原来拿 4 个 token。通信后，每台机器都能看到 16 个 token，但只负责 2 个 head。
- Ring：每台机器始终只负责自己那 4 个 token 的 query，但会分 4 轮接收其他机器的 KV，最后完成全局注意力。

真实工程例子：在 SWIFT 或类似统一序列并行框架里，常见做法不是“二选一”，而是局部网络范围内先用 Ulysses 吃掉高带宽收益，再在更大范围上用 Ring 扩展上下文。这相当于“机内全互连做快交换，机间做可持续扩展”。

---

## 代码实现

下面先给一个可运行的通信量判断脚本。它不是框架代码，而是一个能帮助理解调度决策的最小模型。

```python
from dataclasses import dataclass

@dataclass
class Config:
    batch: int
    seq_len: int
    hidden: int
    heads: int
    sp_degree: int

def ulysses_comm(cfg: Config) -> int:
    assert cfg.batch > 0 and cfg.seq_len > 0 and cfg.hidden > 0
    assert cfg.sp_degree > 0
    return (8 * cfg.batch * cfg.seq_len * cfg.hidden) // cfg.sp_degree

def ring_comm(cfg: Config) -> int:
    assert cfg.batch > 0 and cfg.seq_len > 0 and cfg.hidden > 0
    return 4 * cfg.batch * cfg.seq_len * cfg.hidden

def can_use_pure_ulysses(cfg: Config) -> bool:
    return cfg.heads >= cfg.sp_degree and cfg.heads % cfg.sp_degree == 0

def choose_strategy(cfg: Config) -> str:
    if can_use_pure_ulysses(cfg) and ulysses_comm(cfg) <= ring_comm(cfg):
        return "ulysses"
    if cfg.heads < cfg.sp_degree or cfg.heads % cfg.sp_degree != 0:
        return "ring_or_hybrid"
    return "hybrid"

toy = Config(batch=2, seq_len=16384, hidden=4096, heads=32, sp_degree=8)
assert ulysses_comm(toy) == 134217728
assert ring_comm(toy) == 536870912
assert choose_strategy(toy) == "ulysses"

edge = Config(batch=2, seq_len=16384, hidden=4096, heads=12, sp_degree=8)
assert can_use_pure_ulysses(edge) is False
assert choose_strategy(edge) == "ring_or_hybrid"
```

这段代码表达了两个事实：

- 当 head 足够且能整除时，Ulysses 通常有更低的通信量。
- 当 head 不够分时，纯 Ulysses 在调度层面就会失效，只能切到 Ring 或混合策略。

Ulysses 的新手版伪码如下：

```python
# 输入是按序列切分的本地 token 块
qkv_local = project(local_tokens)

# 第一次 All-to-All: 从 sequence-parallel 改成 head-parallel
qkv_full_seq_partial_head = all_to_all(qkv_local)

# 本地执行完整序列、部分 head 的 attention
partial_out = attention(qkv_full_seq_partial_head)

# 第二次 All-to-All: 还原回 sequence-parallel
out_local = all_to_all(partial_out, reverse=True)
```

Ring Attention 的伪码如下：

```python
local_q, local_k, local_v = project(local_tokens)
kv_buffer = (local_k, local_v)
out = attention(local_q, local_k, local_v)

for step in range(world_size - 1):
    ring_send(kv_buffer, next_rank)
    kv_buffer = ring_recv(prev_rank)
    out += attention(local_q, kv_buffer[0], kv_buffer[1])
```

实现上的关键差别是：

- Ulysses 的核心是布局变换是否高效。
- Ring 的核心是环形 buffer、流水重叠、以及每轮 softmax 统计量如何稳定合并。

---

## 工程权衡与常见坑

下面这张表比“哪个好”更重要，因为真正决定成败的是约束条件。

| 坑 | 触发条件 | 规避方式 |
|---|---|---|
| Ulysses 的 All-to-All 很慢 | 跨机以太网、无 NVLink/NVSwitch | 只在机内做 Ulysses，机间改用 Ring |
| head 无法整除并行度 | 例如 12 heads 配 8 卡 | 降低 Ulysses 度数，或切换 Ring/混合 |
| Ring 串行等待明显 | chunk 太小，单步计算遮不住通信 | 增大 chunk，做 compute-comm overlap |
| 负载不均 | 卡数奇数、异构拓扑、GQA/MQA 头分布特殊 | 做分层并行，避免全局单一策略 |
| 数值实现复杂 | 分块 softmax 合并不严谨 | 使用成熟内核或先做严格单测 |

新手常见误区有两个。

第一，把 Ulysses 理解成“天然更先进”。这不对。它只是**在高带宽、head 可整除**时非常划算。网络一慢，All-to-All 会迅速成为主瓶颈。

第二，把 Ring 理解成“只是省事但更慢”。这也不对。Ring 的强项不是绝对低通信量，而是**扩展边界更宽**。当你要从 128K 继续拉到 512K、1M，上下文继续增长时，它往往比纯 Ulysses 更能活下来。

真实工程例子：假设你在一组 8 卡机器上训练长上下文 SFT。模型是 12 个 query heads，机内有 NVSwitch，但跨机只有普通 RoCE。此时最佳方案往往不是“8 卡纯 Ulysses”，而是机内 4 卡做 Ulysses，跨机再加 Ring。这样既利用了机内高速 All-to-All，又避免了 head 分不匀和跨机全互连过慢的问题。

---

## 替代方案与适用边界

Ulysses 和 Ring 不是互相替代，更像两个可以拼装的积木。

| 场景 | 更适合的选择 |
|---|---|
| head 多、卡数规则、机内高速互连 | Ulysses |
| head 少、卡数不规则、需要继续拉长上下文 | Ring Attention |
| 机内快、机间慢，且上下文极长 | 混合：局部 Ulysses + 全局 Ring |

可以把决策流程写成一句话：

1. 先看 head 能不能整除并行度。
2. 再看 All-to-All 跑在什么网络上。
3. 最后看上下文是不是还会继续翻倍增长。

如果三者都对 Ulysses 友好，就优先 Ulysses；如果其中任意一项不友好，Ring 或混合方案更稳。

还有几种常见替代方向，但它们解决的问题不同：

- **FlashAttention**：主要优化单卡或局部 kernel 的显存与算子效率，不直接解决跨卡长序列并行。
- **上下文并行/块状并行的其他实现**：本质上也在处理序列切分，只是通信原语和布局细节不同。
- **近似注意力**：能进一步降成本，但改变了“精确全局注意力”这个前提，适用边界不同。

所以更准确的说法是：Ring Attention 是 Ulysses 的补充，而不是简单替换品。纯 Ulysses 解决“高效”，Ring 更擅长解决“还能继续扩”。混合方案解决的是“在真实集群里怎么落地”。

---

## 参考资料

1. [DeepSpeed-Ulysses Sequence Parallelism - EmergentMind](https://www.emergentmind.com/topics/deepspeed-ulysses-sequence-parallelism)  
用途：用于核对 Ulysses 的两次 All-to-All、head-parallel 重排和工程限制。

2. [DeepSpeed Ulysses: Optimizing Long Sequence Transformers](https://www.emergentmind.com/papers/2309.14509)  
用途：用于确认 Ulysses 的原始系统目标、可扩展性讨论和与传统方案的差异。

3. [Ulysses + Ring Attention 原理与 SWIFT 工程实践 - Hugging Face Blog](https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention)  
用途：用于整理面向工程实现的直观解释、混合策略和 SWIFT 实战背景。

4. [Ring Attention: Blockwise Transformers - EmergentMind](https://www.emergentmind.com/topics/ring-attention-with-blockwise-transformers)  
用途：用于核对 Ring Attention 的环形 KV 传递、块级计算和扩展边界。
