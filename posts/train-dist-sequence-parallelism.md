## 核心结论

序列并行是把 Transformer 的**序列维**切到多张 GPU 上计算。序列维就是输入 token 排成的一条长度轴，例如一段 4096 token 的上下文，序列维长度就是 4096。它的直接目标不是减少参数量，而是降低注意力计算里的**激活显存峰值**。激活显存就是前向传播时为了反向传播必须暂存的中间张量，它往往比参数本身更早把显存打满。

对长上下文训练，真正先爆的是 attention 相关激活。设 batch 为 $B$，序列长度为 $S$，隐藏维为 $H$，多头数为 $N_h$。如果不做序列并行，单卡上 attention 相关中间量随序列长度快速增长，核心压力来自 $S$ 很大时的二次项。序列并行把输入从单卡上的 $(B,S,H)$ 拆成多卡上的 $[(B,S/n,H)\times n]$，每张卡只处理自己那一段 token，于是单卡上需要持有的 Q/K/V 和部分 attention 中间结果显著变小。

它为什么成立，关键不在“局部算局部”，而在“**拆分前后有通信重排**”。工程上通常在 attention 前后各插一次 all-to-all，把原本按 head 或 hidden 布局的数据，临时改造成按 sequence shard 布局，attention 算完再恢复。DeepSpeed 里常见的 primitive 是 `_SeqAllToAll`。它不是复制模型参数，而是在激活流里做“拆分 → 通信 → 局部 attention → 通信 → 还原”。

新手版可以这样理解：原来是一张 GPU 拿着整篇长文做注意力；序列并行后，4 张 GPU 各拿 1/4 的 token。它们先交换好各自该处理的数据，再分别算自己那块，最后把结果拼回原顺序。模型看到的仍然是完整上下文，但单卡显存只承担其中一部分。

下面这个对比能先抓住重点：

| 方案 | 单卡持有的序列长度 | 单卡激活压力 | 通信需求 | 更适合 |
|---|---:|---:|---:|---|
| 原始 attention | $S$ | 高，长序列下快速上升 | 无或很少 | 短序列、单卡或小规模并行 |
| 序列并行，$n$ 卡 | $S/n$ | 明显下降 | attention 前后各一次 all-to-all | 长上下文训练 |
| 直观收益 | 从整段变成分段 | 单卡峰值约按 shard 比例下降 | 用带宽换显存 | 4K、32K、64K 以上更明显 |

---

## 问题定义与边界

问题本身很直接：长序列训练把 attention 激活推到不可接受的规模。这里的“长”不是抽象概念，而是 4K、16K、64K 这类上下文长度。很多模型在参数还没大到极限时，就已经因为激活显存装不下而无法训练更长上下文或更大 batch。

如果只从复杂度记忆，标准 attention 常写成 $O(S^2)$。这表示序列长度翻倍时，注意力矩阵规模会接近四倍。虽然工程实现里 FlashAttention 等方法能减少显式物化大矩阵的成本，但长序列下，Q/K/V、局部缓存、反向所需中间量仍然会持续给显存施压，所以“序列维切分”仍然有意义。

序列并行的边界也很明确。

第一，它只切**序列维**，不直接切参数矩阵。如果你的问题主要是参数太大，比如 embedding、MLP 或线性层权重放不下，序列并行不是第一选择。

第二，它依赖并行组划分正确。常见约束可以先记成：

$$
S \bmod (\text{seq\_parallel\_size}) = 0
$$

在很多工程实现里，还要进一步满足并行网格兼容，例如：

$$
\text{world\_size} \bmod (\text{seq\_parallel\_size} \times \text{data\_parallel\_size}) = 0
$$

这句话的白话解释是：总 GPU 数必须能被你的并行方案整齐分组，否则通信组根本建不出来。

第三，多头数也常要对齐。多头就是把 attention 切成多个子空间并行计算的机制。如果实现假设每个 shard 均匀拿一部分 head，那么常见约束是：

$$
\text{num\_heads} \bmod \text{seq\_parallel\_size} = 0
$$

若不能整除，就需要额外逻辑，例如 `uneven_heads_all2all()` 处理不均匀分配，否则 token 分片和 head 分片可能错位。

一个玩具例子最容易说明边界。假设你有长度 $S=4096$ 的输入，用 4 张卡做序列并行。那每张卡处理 $1024$ 个 token，看起来很自然。但如果你改成 3 张卡，且实现只支持均匀切分，那么 $4096/3$ 不是整数，通信重排就会变得复杂甚至直接失败。再比如 `num_heads=32`，`seq_parallel_size=6`，也不能均匀分。

所以，序列并行不是“多拿几张卡就行”，而是“序列长度、head 数、并行组尺寸都要同时兼容”。

---

## 核心机制与推导

先看输入。假设 attention 前某层输入张量形状是：

$$
X \in \mathbb{R}^{B \times S \times H}
$$

其中 $B$ 是 batch，$S$ 是序列长度，$H$ 是隐藏维。序列并行规模为 $n$ 时，目标是让每张卡只持有：

$$
X_i \in \mathbb{R}^{B \times (S/n) \times H}
$$

这一步不是简单 `chunk` 完就结束，因为实际分布式训练里，张量布局往往同时受到 tensor parallel、head 切分和 kernel 输入要求影响。于是工程上常用 `_SeqAllToAll` 先做一次布局转换。all-to-all 的意思是“每张卡都向组内其他卡发一部分数据，同时从其他卡收一部分数据”，本质是一次全互换。

一个简化过程可以写成：

1. 输入按当前并行布局进入 `_SeqAllToAll`
2. 通信后，每张卡拿到自己负责的序列分片
3. 在本卡上做 local attention
4. 再做一次 `_SeqAllToAll`，把输出还原回后续层需要的布局

伪代码如下：

```python
x = seq_all_to_all(x, splits=n)      # scatter: 重排到序列分片布局
y = local_attention(x)               # 每张卡只算自己的 token shard
out = seq_all_to_all(y, splits=n)    # gather: 恢复输出布局
```

这个机制为什么能省显存，可以用一个最小推导看。

假设原来单卡需要处理长度为 $S$ 的 Q/K/V，每张量大致规模与 $S \times H$ 成正比。做了序列并行后，每张卡只处理 $S/n$ 个 token，于是 Q/K/V 局部规模变成：

$$
(S/n) \times H
$$

如果只看单卡上与序列长度线性相关的激活，近似下降到原来的 $1/n$。若某些中间 attention 缓存也随局部序列长度缩小，则峰值显存也会一起下降。以 $S=4096,n=4$ 为例，单卡上 token 相关输入规模从 `4096 × H` 变成 `1024 × H`，这是最直观的 4 倍分摊。

需要注意，序列并行不是让每张卡只看局部上下文、丢掉全局信息。真正的实现会通过通信和布局设计，保证 attention 所需的上下文信息在逻辑上仍是完整的。新手可以把它想成“把大表切成小表分开算，但在切之前和切之后都做过严格的行列重排，所以结果等价于在一张大表上按规则完成计算”。

反向传播同样成立。因为 `_SeqAllToAll` 前向做了重排，反向就必须做对应的逆重排。很多实现中 forward/backward 都走 `dist.all_to_all_single()`，这样梯度会沿着与前向一致的路径返回，不会出现某张卡丢梯度或梯度错位的问题。

可以把流程压缩成一个简图：

```text
Input (B, S, H)
  -> SeqAllToAll / scatter
Local shard on each GPU: (B, S/n, H)
  -> Local Attention
Local output on each GPU
  -> SeqAllToAll / gather
Output (restored layout)
```

玩具例子再具体一点。设一句话被切成 8 个 token，2 张卡做序列并行。

- GPU0 先拿 token 0-3
- GPU1 先拿 token 4-7
- 经过一次布局重排后，两张卡各自拿到自己该算的 attention 输入格式
- 每张卡只对 4 个 token 的局部 shard 调 kernel
- 结果再通过 all-to-all 拼回统一输出顺序

这里“每张卡只处理 4 个 token”说的是单卡持有的主序列分片，不是模型只认识 4 个 token。训练语义仍针对完整 8 token 序列。

真实工程例子则更有代表性。训练 64K 上下文 LLM 时，单张 A100 40G 往往先被 attention 激活压爆。工程上常会把 `tensor parallel + sequence parallel + pipeline parallel` 混合使用，再配合 FlashAttention 的分块计算。FlashAttention 的白话解释是“不显式存完整大注意力矩阵，而是边分块边算边归约”。这样，序列并行负责把 token 维切薄，FlashAttention 负责把块内 attention 算得更省，两者叠加，才可能把 64K 训练推到可运行区间。

---

## 代码实现

先给一个可运行的 Python 玩具实现。它不依赖 GPU，也不做真正分布式通信，只模拟“按序列切分后单卡持有的数据量变化”，用于验证基本约束。

```python
from dataclasses import dataclass

@dataclass
class SeqParallelConfig:
    seq_len: int
    hidden_size: int
    num_heads: int
    seq_parallel_size: int
    data_parallel_size: int
    world_size: int

def validate(cfg: SeqParallelConfig) -> bool:
    assert cfg.seq_len % cfg.seq_parallel_size == 0, "seq_len 必须能被 seq_parallel_size 整除"
    assert cfg.world_size % (cfg.seq_parallel_size * cfg.data_parallel_size) == 0, \
        "world_size 必须兼容并行组划分"
    assert cfg.num_heads % cfg.seq_parallel_size == 0, \
        "num_heads 不能均分时需要额外 uneven-head 逻辑"
    return True

def shard_shape(batch_size: int, cfg: SeqParallelConfig):
    local_seq = cfg.seq_len // cfg.seq_parallel_size
    return (batch_size, local_seq, cfg.hidden_size)

def activation_ratio(cfg: SeqParallelConfig) -> float:
    # 这里只估算与 token 数线性相关的局部激活缩放比例
    return 1.0 / cfg.seq_parallel_size

cfg = SeqParallelConfig(
    seq_len=4096,
    hidden_size=4096,
    num_heads=32,
    seq_parallel_size=4,
    data_parallel_size=2,
    world_size=8,
)

assert validate(cfg)
assert shard_shape(2, cfg) == (2, 1024, 4096)
assert abs(activation_ratio(cfg) - 0.25) < 1e-9
```

上面这个例子只验证约束，不代表真实 kernel 行为。真实实现的重点在 `_SeqAllToAll`、通信流和 attention kernel 的衔接。

一个更接近工程结构的伪代码如下：

```python
def distributed_attention(x, seq_group, n):
    # x: [B, S, H] under current parallel layout
    x_local = seq_all_to_all(x, group=seq_group, splits=n, async_op=True)
    y_local = flash_attention(x_local)
    y = seq_all_to_all(y_local, group=seq_group, splits=n, reverse=True, async_op=True)
    return y
```

这里的几个点要看清：

| 组件 | 作用 | 白话解释 |
|---|---|---|
| `_SeqAllToAll` | 重排张量布局 | 先把 token 片段发到该去的 GPU，再把结果拿回来 |
| `DistributedAttention` | 封装前后通信与局部 attention | 把“通信 + 本地算子”当一个整体层使用 |
| `FlashAttention` | 高效 attention kernel | 分块计算，减少显式大矩阵存储 |
| `async_op=True` | 异步通信 | 让通信尽量和计算重叠，不要串行等待 |
| `sp_stream` | 专门的 sequence parallel stream | 给序列并行通信单独安排执行流，减少阻塞 |

前向通常是 `scatter -> local attention -> gather`，反向则按对应路径做逆向通信。很多系统会把序列并行放进更大的并行网格里，例如：

- data parallel 复制模型副本，分 batch
- tensor parallel 切线性层或 head
- sequence parallel 切 token 维
- pipeline parallel 切层

真实工程例子可以设成这样：你要训练一个 7B 模型，上下文长度从 8K 提到 64K。只靠 data parallel 不会解决单卡激活峰值；只靠 tensor parallel 也主要在切参数或 head。于是你会把每层 attention 包进 sequence parallel 容器，在注意力前后做 `_SeqAllToAll`，并强制通信与 FlashAttention overlap。这样显存曲线通常先明显下降，然后吞吐是否可接受，取决于互联带宽和 overlap 做得够不够好。

---

## 工程权衡与常见坑

序列并行最大的代价是通信。它不是免费午餐，而是“用带宽换显存”。如果你的序列并不长，通信成本可能大于显存收益。

例如 batch 里每个样本只有 256 token，4 张卡做序列并行后，每卡只算 64 token。这个时候局部 attention 已经很小，但前后两次 all-to-all 还照样发生。结果常见现象不是更快，而是更慢。因为你节省的显存并不关键，额外通信却是真实付出的。

常见坑可以直接列成表：

| 常见坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| world size mismatch | 初始化并行组失败 | `world_size` 不能整齐划分 | 先验证 `world_size % (sp * dp) == 0` |
| head mismatch | 运行时报 shape 或通信错误 | `num_heads` 不能均分到 shard | 保证 `num_heads % sp == 0`，或使用 uneven-head 实现 |
| short sequence throughput drop | 吞吐下降 | token 太短，通信占比过高 | 短序列退回 tensor parallel 或增大 token 数 |
| communication non-overlap | 显存省了但速度很差 | 通信和计算串行等待 | 使用 `async_op=True`、独立 `sp_stream`、减少同步点 |
| shard 不均匀 | 个别卡更慢 | 序列切分不整齐或 padding 处理差 | 统一分片长度，必要时显式 pad |
| 只看显存不看端到端吞吐 | 方案“能跑但不值得” | 忽略带宽和互联拓扑 | 同时测 step time、MFU、通信占比 |

还要注意一个误区：序列并行不一定自动带来更高吞吐。它的第一收益是“让原本跑不起来的长上下文跑起来”，第二收益才可能是“在可接受吞吐下训练更多 token”。如果通信拓扑不好，例如跨节点 all-to-all 很慢，那么理论上的显存收益可能被端到端时延抵消。

另一个常见失败模式是 mesh 设计混乱。mesh 就是并行组的二维或多维布局。你可能同时用了 data parallel、tensor parallel、pipeline parallel，再加 sequence parallel。如果不先把 world size 和各维度乘积算清楚，问题不会出在模型数学上，而是出在通信组根本无法正确构造。

---

## 替代方案与适用边界

序列并行不是唯一方案。至少要和 tensor parallel、context parallel 区分清楚。

tensor parallel 的核心是切参数矩阵或 head。它更像“同一层权重分摊到多卡”。如果你的瓶颈是大线性层、head-heavy 结构，或者序列并不长，那么 tensor parallel 往往比 sequence parallel 更直接，通信模式也常更稳定。

context parallel 在 Megatron 语境下更偏向对上下文处理路径做更广义的并行设计，适合更复杂的长上下文切分策略。它和 sequence parallel 有关联，但不是简单同义词。对初学者而言，可以先把 sequence parallel 理解为“围绕 attention 的序列维分片方案”，而 context parallel 是“更大范围处理长上下文的并行框架”。

下面这张表有助于快速区分：

| 方案 | 主要切分对象 | 参数是否复制 | 主要通信 | 更适合的场景 |
|---|---|---|---|---|
| 序列并行 | 序列维/token 维 | 参数通常不因 SP 本身增加复制 | all-to-all | 长上下文、attention 激活先爆 |
| tensor parallel | 权重矩阵/head | 参数按张量维切分 | all-reduce / all-gather 等 | 模型层很宽、head 多、短到中等序列 |
| context parallel | 更广义的上下文处理路径 | 视实现而定 | 取决于实现 | 极长上下文、复杂混合并行 |

适用边界可以压缩成一句判断：

- 当 sequence length 远大于单卡可承受范围，且 attention 激活是主要瓶颈时，优先考虑序列并行。
- 当序列还不长，或者主要压力来自参数与线性层，优先考虑 tensor parallel。
- 当训练已经进入超长上下文、跨多类并行共同设计的阶段，再考虑 context parallel 或混合方案。

新手版判断法也很实用：如果你还只是 2K token，而且单卡或 tensor parallel 已经能稳跑，就不要为了“听起来先进”去引入 all-to-all；如果你开始冲 32K、64K，上下文一拉长显存就炸，序列并行才真正进入主舞台。

---

## 参考资料

- Hugging Face, *Sequence Parallelism for Long Context Training*：适合先建立整体概念，理解长上下文训练里为什么要切序列维。  
  https://huggingface.co/docs/trl/main/distributing_training
- DeepSpeed Sequence Parallelism 概览：包含 `_SeqAllToAll`、`DistributedAttention`、FPDT 等实现细节，适合看工程落地。  
  https://deepwiki.com/deepspeedai/DeepSpeed/4.4-sequence-parallelism
- NVIDIA Megatron, *Context and Sequence Parallelism*：适合对比 sequence parallel 与 context parallel 的边界。  
  https://deepwiki.com/NVIDIA/Megatron-LM/4.5-context-and-sequence-parallelism

阅读顺序：先看 Hugging Face 的概览，再看 DeepSpeed 的实现细节，最后补 Megatron 的边界对比。
