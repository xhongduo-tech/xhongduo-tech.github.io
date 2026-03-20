## 核心结论

MQA 与 GQA 解决的是同一个问题：推理时 KV Cache 太大。KV Cache 可以理解为“模型为了后续逐 token 生成而保留下来的 Key/Value 历史记录”。在标准 MHA（Multi-Head Attention，多头注意力）里，$h$ 个 Query 头通常对应 $h$ 组 KV 头，因此缓存大小会随头数线性增长。

GQA（Grouped-Query Attention，分组查询注意力）把 $h$ 个 Query 头分成 $g$ 组，每组共享一套 Key/Value；MQA（Multi-Query Attention，多查询注意力）则更激进，直接让所有 Query 头共享同一套 Key/Value。于是：

- MHA：$n_{\text{kv\_heads}} = h$
- GQA：$n_{\text{kv\_heads}} = g,\ 1 < g < h$
- MQA：$n_{\text{kv\_heads}} = 1$

KV Cache 的近似公式是：

$$
KV_{\text{cache}} \approx 2 \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{precision} \times \text{seq\_len}
$$

其中：

- $L$ 是层数，也就是 Transformer block 的数量
- $n_{\text{kv\_heads}}$ 是 KV 头数，也就是实际要缓存多少组 K/V
- $d_{\text{head}}$ 是每个头的维度，也就是每个注意力头的向量长度
- `precision` 是每个元素占多少字节，例如 FP16 是 2 字节
- `seq_len` 是上下文长度，也就是已经缓存了多少 token

因为公式里 $n_{\text{kv\_heads}}$ 是线性项，所以把 MHA 的 $h$ 改成 GQA 的 $g$，缓存就按 $g/h$ 线性缩小；改成 MQA 的 1，则缩小到原来的 $1/h$。

一个直观玩具例子：想象 64 个售货员要回答顾客问题。MHA 相当于 64 个售货员各自维护 64 份库存表；MQA 相当于 64 个售货员共用 1 份库存表；GQA 相当于把 64 个售货员分成 8 组，每组共用 1 份库存表。显存的本质成本来自“库存表”的存储，而不是售货员本身。于是 GQA 在逻辑上保留了很多独立查询能力，但把库存表数量从 64 份压到 8 份。

真实工程上，LLaMA-2 70B 使用的是 GQA：64 个注意力头，8 个 KV 头。它的含义不是“只剩 8 个头在工作”，而是“64 个 Query 头仍然独立计算，但只保留 8 套 Key/Value 供分组共享”。这正是它能把推理显存和吞吐做平衡的原因。

---

## 问题定义与边界

问题定义很简单：在大模型推理里，权重是静态成本，KV Cache 是随请求数和上下文长度增长的动态成本。模型越大、上下文越长、并发越高，KV Cache 越容易成为瓶颈。

对零基础读者，先抓住一条线：生成第 $t$ 个 token 时，模型不想把前 $t-1$ 个 token 再全算一遍，所以会把每层历史 token 的 K 和 V 存起来，下一个 token 直接复用。这就是 KV Cache。它节省算力，但会占显存。

以 70B 量级模型、4K context、FP16 为例，若采用 MHA 且有 64 个 KV 头，则单请求缓存可能达到约 10GB；若改成 GQA 且 $g=8$，则约为 1.25GB。这个差异直接决定一张卡上能放多少并发请求。

下面给一个规模边界表。这里使用的是 LLaMA-2 70B 常见配置近似值：$L=80$，$d_{\text{head}}=128$，FP16，`seq_len=4096`。

| KV 头数 | 注意力形式 | 单请求 4K Cache | 若 KV 预算为 10GB，可容纳并发请求数 | 显存压力判断 |
|---|---:|---:|---:|---|
| 64 | MHA | 10.0GB | 1 | 几乎没有弹性 |
| 8 | GQA-8 | 1.25GB | 8 | 常见生产折中 |
| 4 | GQA-4 | 0.625GB | 16 | 更省显存，但质量更敏感 |
| 1 | MQA | 0.156GB | 64 | 极致压缩，但质量风险更高 |

这个问题的边界也要说清楚：

- 第一，GQA 不是“白拿性能”。它用更少的 KV 表达能力换更小的缓存，因此存在质量回撤的可能。
- 第二，GQA 主要优化的是推理阶段，尤其是长上下文和高并发场景。训练阶段是否划算，要看实现和目标。
- 第三，它不解决所有显存问题。权重、激活值、框架额外开销仍然存在。
- 第四，GQA 并不是运行时插件那么简单。若模型原生就是 MHA，想无损切到 GQA，通常需要重新训练或至少做少量 uptraining。uptraining 可以理解为“在原模型基础上再补一小段训练，让模型适应新结构”。

所以问题边界可以概括成一句话：GQA 讨论的是“如何在尽量不明显伤害质量的前提下，按 $g/h$ 比例线性缩小 KV Cache，从而提升长上下文与高并发推理的可部署性”。

---

## 核心机制与推导

先从标准 MHA 开始。设输入张量是：

$$
x \in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

其中：

- $B$ 是 batch size，也就是一次处理几个样本
- $T$ 是序列长度，也就是当前 token 数
- $d_{\text{model}}$ 是隐藏维度，也就是每个 token 的主表示长度

在 MHA 中，通常有：

$$
Q = xW_Q,\quad K = xW_K,\quad V = xW_V
$$

若头数为 $h$，每头维度为 $d_{\text{head}}$，则：

$$
d_{\text{model}} = h \times d_{\text{head}}
$$

于是 reshape 后的形状是：

$$
Q, K, V \rightarrow (B, T, h, d_{\text{head}})
$$

每个 Query 头都对应自己的一组 K/V，因此缓存也是按 $h$ 份存。

GQA 的关键变化只发生在 K/V 上。它保留 Query 的独立头数为 $h$，但把 K/V 的头数缩成 $g$：

$$
Q \rightarrow (B, T, h, d_{\text{head}})
$$

$$
K, V \rightarrow (B, T, g, d_{\text{head}})
$$

同时要求：

$$
h \bmod g = 0
$$

也就是 $h$ 必须能被 $g$ 整除。这样每个 KV 头负责一组 Query 头，组大小为：

$$
r = h / g
$$

可以把 Query reshape 为：

$$
Q \rightarrow (B, T, g, r, d_{\text{head}})
$$

于是第 $i$ 组中的 $r$ 个 Query 头，共享第 $i$ 组的 K/V。注意，这里“共享”的是 K/V，不是 Query。本质上是：

- Query 仍然是每个头独立的
- Key/Value 在组内共享
- 因此同组内不同头，仍可能学出不同注意力模式

这点很重要，因为很多初学者会误以为“共享 KV 就等于头不独立了”。实际上不同头的 Query 向量不同，打分结果仍会不同。共享的是“被查询的库存表”，不是“查询动作”本身。

注意力打分可写成组内形式：

$$
\text{score}_{b,t,g_i,r_j,s}
=
\frac{Q_{b,t,g_i,r_j,:}\cdot K_{b,s,g_i,:}}{\sqrt{d_{\text{head}}}}
$$

其中：

- $g_i$ 是第 $i$ 个组
- $r_j$ 是组内第 $j$ 个 Query 头
- $s$ 是被关注的历史位置

这里能看出，组内多个 Query 头都对同一个组的 K 做点积，但因为各自 Query 不同，所以注意力分布仍可能不同。

再看缓存推导。每层每个 token 需要存 K 和 V 两份，因此单层、单 token 的缓存字节数约为：

$$
2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{precision}
$$

乘上层数 $L$ 和序列长度 `seq_len`，得到总缓存：

$$
KV_{\text{cache}} \approx 2 \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{precision} \times \text{seq\_len}
$$

于是：

- MHA：$n_{\text{kv\_heads}}=h$
- GQA：$n_{\text{kv\_heads}}=g$
- MQA：$n_{\text{kv\_heads}}=1$

压缩比就是：

$$
\frac{KV_{\text{GQA}}}{KV_{\text{MHA}}} = \frac{g}{h}
$$

LLaMA-2 70B 的真实工程例子可以直接代入。它有 80 层、64 个 Query 头、8 个 KV 头、每头 128 维，FP16 即 2 字节。则单 token KV 为：

$$
2 \times 80 \times 8 \times 128 \times 2 = 327{,}680\ \text{bytes} \approx 320\text{KB}
$$

若上下文是 4096 token，则单请求总缓存约为：

$$
327{,}680 \times 4096 \approx 1.25\text{GB}
$$

如果同样结构不用 GQA，而是 64 个 KV 头，则再乘 8 倍，约为 10GB。按“每层”来看，也可理解为从约 256MB 降到 32MB。原因完全来自 $64 \rightarrow 8$ 的线性变化，即 $g/h=1/8$。

---

## 代码实现

下面给一个可运行的玩具实现，用纯 Python 展示“头分组”和“KV Cache 大小计算”这两件事。它不是高性能框架实现，但足够把形状关系讲清楚。

```python
from math import prod

def kv_cache_bytes(layers, num_kv_heads, head_dim, seq_len, precision_bytes=2):
    return 2 * layers * num_kv_heads * head_dim * precision_bytes * seq_len

def group_mapping(num_query_heads, num_kv_heads):
    assert num_query_heads % num_kv_heads == 0
    group_size = num_query_heads // num_kv_heads
    mapping = {}
    for q_head in range(num_query_heads):
        kv_group = q_head // group_size
        mapping[q_head] = kv_group
    return mapping

# 玩具例子：8 个 Query 头，2 个 KV 头
mapping = group_mapping(8, 2)
assert mapping[0] == 0
assert mapping[3] == 0
assert mapping[4] == 1
assert mapping[7] == 1

# LLaMA-2 70B 近似配置
mha = kv_cache_bytes(layers=80, num_kv_heads=64, head_dim=128, seq_len=4096, precision_bytes=2)
gqa = kv_cache_bytes(layers=80, num_kv_heads=8, head_dim=128, seq_len=4096, precision_bytes=2)

# GQA-8 相对 MHA 恰好缩小 8 倍
assert mha // gqa == 8

# 单请求 GQA 约 1.25GB
gqa_gb = gqa / (1024 ** 3)
assert 1.2 < gqa_gb < 1.3

print("group mapping:", mapping)
print("GQA cache (GB):", round(gqa_gb, 3))
```

如果换成接近框架实现的伪代码，核心逻辑是下面这样：

```python
# x: (batch, seq, d_model)
q = linear_q(x)  # (batch, seq, h * d)
k = linear_k(x)  # (batch, seq, g * d)
v = linear_v(x)  # (batch, seq, g * d)

q = q.reshape(batch, seq, h, d)
k = k.reshape(batch, seq, g, d)
v = v.reshape(batch, seq, g, d)

# 每组 query 头数量
group_size = h // g

# (batch, seq, g, group_size, d)
q_grouped = q.reshape(batch, seq, g, group_size, d)

# 推理时 KV Cache 只按 g 个头存，不按 h 个头存
kv_cache_k.append(k)  # shape: (batch, total_seq, g, d)
kv_cache_v.append(v)  # shape: (batch, total_seq, g, d)

# 组内 attention：每个 query 组只和对应 KV 组交互
scores = einsum("b t g r d, b s g d -> b g r t s", q_grouped, kv_cache_k)
weights = softmax(scores / sqrt(d), dim=-1)
out = einsum("b g r t s, b s g d -> b t g r d", weights, kv_cache_v)

# 再 reshape 回 (batch, seq, h, d) -> (batch, seq, d_model)
out = out.reshape(batch, seq, h, d)
```

这里最容易忽略的一点是：KV Cache 的持久化结构必须跟着 `g` 走，而不是跟着 `h` 走。也就是说，如果模型是 GQA-8，你的缓存张量维度就应该是 `(batch, seq, 8, d)`，而不是 `(batch, seq, 64, d)`。否则结构上看似用了 GQA，实际缓存并没有省下来。

真实工程例子里，这个修改通常体现在两处：

- K/V 投影层输出维度从 `h * d_head` 改为 `g * d_head`
- Cache allocator 申请内存时，以 `num_key_value_heads=g` 为准

这就是部署里“模型结构改了，但显存没明显下降”的常见排查方向。

---

## 工程权衡与常见坑

GQA 的核心 trade-off 很明确：$g$ 越小，显存越省；但 $g$ 越小，组内 Query 头共享的 K/V 越多，表达能力约束也越强，质量回撤风险会增加。工业上常见选择是 $g \in [4, 16]$，其中 $g=8$ 很常见，因为它通常接近 MHA 质量，同时又能拿到明显的缓存收益。

一个简表：

| g 值 | 相对 MHA 的 KV 压缩率 | 质量影响趋势 | reshape/内核复杂度 |
|---:|---:|---|---|
| 16 | 4x | 很小 | 低 |
| 8 | 8x | 通常是常见甜点区 | 低 |
| 4 | 16x | 开始更敏感 | 中 |
| 1 | 64x | 接近 MQA，质量风险最高 | 低 |

常见坑主要有五类。

第一，`g` 不能随便取。必须满足 `g | h`，也就是 `g` 整除 `h`。例如 `h=64` 时选 `g=8`、`g=4`、`g=16` 都可以；选 `g=7` 就会出问题。理论上你可以靠 padding 或额外 tensor reorder 硬凑，但实际会引入不必要的数据搬运，吞吐经常变差。

第二，只改模型投影，不改缓存结构。这样前向里 K/V 头数是 `g`，但缓存容器还是按 `h` 申请，最后显存收益被抵消。

第三，忽略内核支持。FlashAttention、PagedAttention、不同推理框架对 GQA 的支持成熟度不同。若底层 kernel 没有针对 GQA 做好路径，可能出现“理论省显存，实测速度没上去”甚至倒退。

第四，把 MHA checkpoint 机械改成 GQA 后直接上线。论文和工程实践都说明，从 MHA 到 GQA/MQA 往往需要少量 uptraining 适配。5% 预训练算力级别的继续训练，常被用来恢复结构切换带来的质量损失。

第五，只盯着单请求显存，不看系统总吞吐。真实服务里，GQA 的价值常常不在单次推理快多少，而在“同样显存预算下能容纳更多并发请求”。例如 70B 模型在 H100×8 上，4K context 下若从约 10GB/请求降到约 1.25GB/请求，调度器的可操作空间会立刻变大。

一个简短 checklist：

- `g` 必须整除 `h`
- 核对 `num_key_value_heads` 是否同步进入 cache allocator
- 确认推理框架底层 kernel 支持 GQA
- 若从 MHA 改造而来，预留 uptraining 或蒸馏验证
- 线上监控困惑度、长上下文检索、推理吞吐三个指标，不要只看单一 benchmark

---

## 替代方案与适用边界

GQA 不是唯一方案，但它是工程上很平衡的一种。

先看四类常见方案对比：

| 方案 | KV 底层头数 | 质量损失趋势 | 工程复杂度 | 适用边界 |
|---|---:|---|---|---|
| MHA | $h$ | 最小 | 低 | 显存充足、追求原始质量 |
| GQA | $g$ | 通常小于 MQA | 中 | 长上下文、高并发、主流折中 |
| MQA | 1 | 通常高于 GQA | 低到中 | 极端省显存场景 |
| KV 量化 | 不改头数 | 取决于量化位宽 | 中到高 | 已有模型不改结构时 |

MQA 的极端例子很好理解：64 个 Query 头共享 1 套 K/V，缓存直接缩小 64 倍。这对部署非常诱人，但代价是所有头都在查同一本“库存表”，表达多样性压缩得最厉害。相比之下，GQA 若取 `g=4` 或 `g=8`，仍保留多套组级 K/V，因此通常更容易把质量控制在较小回撤内。

GQA 的适用边界主要是两类：

- 你可以训练或继续训练模型，那么 GQA 往往是首选结构性优化手段。
- 你不能改训练，只能做部署层优化，那么更现实的是 KV 量化、分页缓存、前缀缓存等运行时技术。

实际系统里，它们并不冲突。常见组合是：

- 结构层：GQA 把 KV 头数从 $h$ 降到 $g$
- 存储层：再把 FP16 KV 改成 FP8 或更低位宽
- 调度层：配合 paged KV cache、continuous batching 提升整体吞吐

所以不要把 GQA 和 KV 量化看成二选一。前者是“少存几套 K/V”，后者是“每套 K/V 存得更便宜”。两者叠加时，显存收益通常是乘法关系。

一个真实工程判断标准是：

- 如果你要尽量保留模型原质量，优先 MHA 或 GQA-8
- 如果显存极其紧张但仍要跑大模型，优先 GQA-4、MQA，或 GQA 加 KV 量化
- 如果模型已经定型、不能重训，优先运行时量化和缓存优化，不要假设可以无代价把 MHA 改成 GQA

---

## 参考资料

- Joshua Ainslie 等，arXiv/EMNLP 2023《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》— 给出 GQA 定义，并说明从 MHA checkpoint 向 MQA/GQA 转换可用约 5% 原预训练算力做 uptraining，用于本文的结构定义、uptraining 边界与质量权衡。https://arxiv.org/abs/2305.13245
- Meta Llama 团队，Llama 2 Model Card — 明确 Llama 2 70B 使用 GQA，用于本文“70B 真实工程例子”的模型结构事实。https://github.com/meta-llama/llama/blob/main/MODEL_CARD.md
- Hugging Face 上的 Llama-2-70B 配置镜像 — 展示 `num_attention_heads=64`、`num_key_value_heads=8`，用于本文的维度与缓存算例。https://huggingface.co/NousResearch/Llama-2-70b-hf/blob/main/config.json
- Brian Su《LLM Serving from Scratch》— 系统化给出 KV Cache 公式与 LLaMA 70B 量级的缓存算例，用于本文的公式、1.25GB 单请求估算和并发边界讨论。https://briansu.co/articles/optimization/llm-serving
- Michael Brenndoerfer《Grouped Query Attention: Memory-Efficient LLM Inference》— 以工程视角总结 GQA 的缓存缩放、GQA-8 作为常见折中配置，以及 GQA 相比 MQA 的质量-速度平衡，用于本文的工程解释与 trade-off 归纳。https://mbrenndoerfer.com/writing/grouped-query-attention-gqa-efficient-llm-inference
- Michael Brenndoerfer《Multi-Query Attention: Memory-Efficient LLM Inference》— 总结 MQA 的极致 KV 压缩与更明显的质量风险，用于本文在替代方案部分对 MQA 的对照说明。https://mbrenndoerfer.com/writing/multi-query-attention-memory-efficient-inference
