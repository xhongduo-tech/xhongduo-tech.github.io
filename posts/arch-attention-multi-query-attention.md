## 核心结论

MQA，中文常译为“多查询注意力”，可以理解为“很多个查询头，共享一套键和值”。它保留多头注意力里多个 Query head 的表达能力，但把 Key/Value head 压缩成 1 个，因此主要优化的是**推理阶段的 KV Cache**。

先记最重要的公式：

$$
\text{KVCache}_{\text{MHA}} = 2 \times L \times n \times h \times d
$$

$$
\text{KVCache}_{\text{MQA}} = 2 \times L \times n \times 1 \times d
$$

这里：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $L$ | 上下文长度 | 已经生成或读入了多少个 token |
| $n$ | batch size | 一次并行处理多少条请求 |
| $h$ | Query 头数 | 同时从多少个“观察角度”看输入 |
| $d$ | head 维度 | 每个头内部的向量长度 |

压缩因子就是 $h$。如果模型有 32 个 Query 头，那么 MQA 相比标准多头注意力（MHA）通常能把 KV Cache 压到原来的 $1/32$。

这件事为什么重要？因为大语言模型做自回归生成时，每生成一个新 token，都要反复读取过去所有 token 的 key/value。这个缓存不只占显存，还要反复参与显存带宽传输。长上下文、长回复、大 batch 推理时，瓶颈常常不是算力，而是**内存容量和带宽**。

玩具例子：假设一个 32 头模型已经看过 2048 个 token。标准 MHA 要保存 32 份历史 K 和 32 份历史 V；MQA 只保存 1 份 K 和 1 份 V。对新手来说，可以把它想成：原来系统给 32 个“提问者”各建一座独立仓库，现在变成 32 个提问者共用 1 座仓库。仓库少了，内存和搬运成本就一起下降。

工程上，PaLM、Falcon 等模型采用这类设计，本质上是在做一个明确交换：**用少量质量下降，换更高吞吐和更低推理成本**。在不少场景里，这个交换是划算的。

---

## 问题定义与边界

要理解 MQA，先要明确它解决的不是“训练太慢”，而是**Decoder-only 模型在推理时 KV Cache 过大**的问题。

KV Cache，就是“历史注意力状态缓存”。白话说，模型为了避免每次都把前文重新算一遍，会把以前 token 的 key 和 value 存下来，后续生成时直接复用。这样算力省了，但内存占用会随序列长度线性增长。

标准 MHA 的问题在于：每个注意力头都有自己的一套 K/V。于是每一层都要缓存：

- 每个 token 的 key
- 每个 token 的 value
- 每个 head 各自一份

如果模型层数多、头数多、上下文长，这个缓存会非常大。并且生成阶段每次只吐一个 token，属于典型的“反复读历史缓存”的过程，所以显存带宽压力也很重。

可以用一个简化表格先看差异：

| 方案 | Query 头数 | KV 头数 | 单层 KV Cache 规模 |
|---|---:|---:|---|
| MHA | 32 | 32 | $2 \times L \times n \times 32 \times d$ |
| GQA | 32 | 4 或 8 | $2 \times L \times n \times 4d$ 或 $8d$ |
| MQA | 32 | 1 | $2 \times L \times n \times d$ |

这里的边界也要说清楚：

1. MQA主要优化**推理阶段**，尤其是长上下文生成。
2. 它不改变“Query 有多个头”这个事实，因此不是把多头注意力彻底删掉。
3. 它不是没有代价。共享 K/V 以后，不同头在“记忆表示”上的自由度下降，通常会带来小幅质量损失。
4. 它更适合“吞吐、延迟、显存”敏感的部署场景，不一定适合所有高精度任务。

真实工程例子：如果你在一张显存有限的 GPU 上部署一个 7B 级别模型，希望支持更长的上下文和更大的 batch，那么标准 MHA 常常先卡死在 KV Cache 上，而不是矩阵乘法算不动。MQA 的价值就在这里，它让“原本放不下或吞吐太低”的配置变得可运行。

---

## 核心机制与推导

标准多头注意力里，每个 head 都有独立的 Query、Key、Value 投影。MQA 改动的核心只有一句话：

**Q 仍然按多头拆分，K/V 不再按头拆分，只保留一套共享表示。**

“投影”这个词的白话解释是：把输入向量通过一个线性层映射成注意力计算需要的空间。

设输入是 $X \in \mathbb{R}^{B \times T \times D_{\text{model}}}$。在 MHA 中：

- $Q_i = XW_{Q_i}$
- $K_i = XW_{K_i}$
- $V_i = XW_{V_i}$

每个 head 都有自己的 $K_i,V_i$。

在 MQA 中则变成：

- 每个 head 仍有自己的 $Q_i$
- 所有 head 共享同一个 $K,V$

写成公式是：

$$
Q_i = XW_{Q_i}, \quad K = XW_K, \quad V = XW_V
$$

第 $i$ 个 Query 头的注意力输出为：

$$
\text{Attention}_i = \text{softmax}\left(\frac{Q_i K^\top}{\sqrt{d}}\right)V
$$

这里有一个容易误解的点：虽然公式里 $K,V$ 没有 head 下标，但并不意味着“只有一个头在工作”。真实情况是：

- 每个 Query head 仍在独立计算自己的注意力分数
- 只是这些分数都去匹配同一份 K
- 最后都从同一份 V 里取信息

可以把它理解为：32 个不同的查询者，仍然从 32 个角度提问，但他们查的是同一个档案库。

从张量形状看更直观。假设：

- Query 头数为 $h$
- 每头维度为 $d$
- 那么 $Q$ 的形状通常是 `[B, h, T, d]`
- 而 $K,V$ 在 MQA 中是 `[B, 1, T, d]`

注意这里的 `1` 很关键，表示只有 1 个 KV head。后续做矩阵乘法时，框架会在 head 维度上自动广播，等价于把这 1 份 K/V 提供给所有 Query 头使用。

为什么这样还能工作？因为多头注意力真正承担“多视角建模”的主要部分在 Query 的差异化上。不同 Query head 仍会产生不同的打分分布，所以它们并没有退化成完全一样的注意力头。只是这些头共享了记忆索引结构，因此表达能力比完整 MHA 更弱，但又没有弱到完全不可用。

再看一个具体数字。若 $L=2048, n=1, h=32, d=1024$，则：

$$
\text{KVCache}_{\text{MHA}} = 2 \times 2048 \times 1 \times 32 \times 1024
$$

$$
\text{KVCache}_{\text{MQA}} = 2 \times 2048 \times 1 \times 1 \times 1024
$$

两者相差正好 32 倍。这就是 MQA 在部署里非常有吸引力的根本原因。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，用最小例子展示 MQA 的核心：`q` 保持多头，`k/v` 只有一个头，并在计算时广播。

```python
import math

def shape_of_mha_kv_cache(L, n, h, d):
    return 2 * L * n * h * d

def shape_of_mqa_kv_cache(L, n, d):
    return 2 * L * n * d

def mqa_attention(q, k, v):
    """
    q: [heads, seq_q, dim]
    k: [1, seq_k, dim]
    v: [1, seq_k, dim]
    返回: [heads, seq_q, dim]
    """
    heads = len(q)
    seq_q = len(q[0])
    seq_k = len(k[0])
    dim = len(q[0][0])

    out = []
    for h in range(heads):
        head_out = []
        for t in range(seq_q):
            scores = []
            for j in range(seq_k):
                dot = sum(q[h][t][m] * k[0][j][m] for m in range(dim))
                scores.append(dot / math.sqrt(dim))

            max_score = max(scores)
            exps = [math.exp(s - max_score) for s in scores]
            denom = sum(exps)
            probs = [e / denom for e in exps]

            vec = []
            for m in range(dim):
                value = sum(probs[j] * v[0][j][m] for j in range(seq_k))
                vec.append(value)
            head_out.append(vec)
        out.append(head_out)
    return out

L, n, h, d = 2048, 1, 32, 1024
assert shape_of_mha_kv_cache(L, n, h, d) == 32 * shape_of_mqa_kv_cache(L, n, d)

q = [
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.5, 0.5], [1.0, 1.0]],
]
k = [[[1.0, 0.0], [0.0, 1.0]]]
v = [[[10.0, 1.0], [1.0, 10.0]]]

out = mqa_attention(q, k, v)
assert len(out) == 2
assert len(out[0]) == 2
assert len(out[0][0]) == 2
```

这个例子不是高性能实现，但足够说明两件事：

1. `q` 有多个头，所以不同头仍然会得出不同注意力结果。
2. `k/v` 只有一份，但可以被所有头复用。

如果换成实际框架，比如 PyTorch，典型逻辑会长这样：

```python
import torch
import math

def mqa_step(x, wq, wk, wv, past_k=None, past_v=None):
    # x: [batch, seq, hidden]
    # wq: produce [batch, seq, heads * dim]
    # wk/wv: produce [batch, seq, dim]
    batch, seq, _ = x.shape
    heads = 4
    dim = 8

    q = wq(x).view(batch, seq, heads, dim).transpose(1, 2)   # [B, H, T, D]
    k = wk(x).unsqueeze(1)                                    # [B, 1, T, D]
    v = wv(x).unsqueeze(1)                                    # [B, 1, T, D]

    if past_k is not None:
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_probs, v)                         # [B, H, T, D]

    return out, (k, v)
```

这里的关键点有三个：

| 实现点 | 作用 | 常见错误 |
|---|---|---|
| `q` 保持 `[B, H, T, D]` | 保留多 Query 头 | 把 Q 也压成单头，结果退化过度 |
| `k/v` 为 `[B, 1, T, D]` | 共享 KV | 错把 KV 仍拆成 H 份，失去 MQA 意义 |
| `torch.cat(..., dim=2)` | 沿时间维拼接 cache | 拼错维度，导致历史 token 顺序损坏 |

---

## 工程权衡与常见坑

MQA 的收益非常直接，但它不是免费午餐。

先看一个部署视角的比较：

| 方案 | KV 头数 | 内存占用 | 吞吐潜力 | 质量风险 |
|---|---:|---|---|---|
| MHA | 32 | 高 | 低到中 | 最低 |
| GQA | 4-8 | 中 | 中到高 | 中等 |
| MQA | 1 | 最低 | 最高 | 最高 |

这里“质量风险”的白话解释是：模型可能更容易在复杂任务上失分，尤其是需要细粒度头部分工的场景。

常见工程收益包括：

1. 更长上下文。原本 8k 放不下，压缩 KV 后可能能跑到更长。
2. 更大 batch。同样显存下可以同时服务更多请求。
3. 更高吞吐。因为读写缓存更少，解码阶段更容易跑满。
4. 更低成本。特别是在显存昂贵的推理集群里，收益很直接。

真实工程例子：部署一个对话模型时，如果标准 MHA 在 16 路并发时显存接近打满，切到 MQA 后，KV Cache 缩小，常常可以把 batch size 提上去，或者把上下文长度从 4k 拉到 8k 以上。对在线服务来说，这意味着单卡吞吐提高，单位请求成本下降。

但常见坑也很明确：

1. **不要把“显存省了”误解为“总是更快”**。如果系统瓶颈在别处，比如采样逻辑、网络 IO、张量并行通信，那 MQA 收益会被稀释。
2. **质量下降不是常数**。资料里常见说法是 1% 到 3% 左右，但不同任务、不同模型、不同训练方法差异很大。摘要生成、闲聊类任务通常更能接受；复杂推理、精确检索、代码生成任务可能更敏感。
3. **训练和推理要一致设计**。不能拿一个纯 MHA 训练好的模型，简单在推理时把 KV 强行并成一个头，就假定效果不变。结构改了，通常需要对应训练或蒸馏。
4. **缓存布局要稳定**。很多 bug 不是理论问题，而是张量 shape 错、拼接轴错、mask 广播错。
5. **注意多卡带宽**。MQA 不只省显存容量，也省通信压力；但如果你的并行策略依赖额外同步，收益需要重新测。

因此，MQA 更像一个面向部署的结构优化，而不是单纯的“更先进注意力”。

---

## 替代方案与适用边界

MQA 最大的问题是压得太狠。很多时候，工程上更常见的折中是 GQA，中文常译为“分组查询注意力”。

GQA 的意思是：不是所有 Query 头都共享 1 套 K/V，而是分组共享。比如 32 个 Query 头分成 4 组，每组共享 1 套 K/V，那么 KV 头数就是 4。这样缓存仍然比 MHA 小很多，但表达能力通常比 MQA 更强。

可以用表格快速判断：

| 方案 | KV head 数 | 适合场景 | 不适合场景 |
|---|---:|---|---|
| MHA | 等于 Query 头数 | 追求质量上限、训练资源充足 | 显存紧张的长上下文部署 |
| GQA | 4 或 8 等分组值 | 大多数通用部署，质量和成本折中 | 极端显存受限场景 |
| MQA | 1 | 长上下文、高并发、成本敏感服务 | 对细粒度质量极度敏感任务 |

还有一种常见思路是“分层混用”：

- 底层或大多数层使用 MQA/GQA，节省主要缓存
- 少数关键层保留完整 MHA，保住表达能力

白话理解是：不是每一层都必须同样省。某些层更负责基础模式提取，某些层更负责复杂语义整合。给关键层保留更高自由度，有时能在不显著增加缓存的前提下稳住质量。

例如，在一个多轮对话模型中，你可以让前几层或后几层保留更多 KV 头，其余层采用 MQA。这种设计没有统一标准，要靠基准测试决定，但思路很实用：**把预算花在最关键的层上**。

因此，适用边界可以总结成三句：

1. 如果部署瓶颈主要是 KV Cache，MQA 值得优先考虑。
2. 如果任务对精度很敏感，优先看 GQA，而不是直接上 MQA。
3. 如果你有结构搜索或微调能力，分层混用通常比“全层一刀切”更稳。

---

## 参考资料

- Michael Brenndoerfer, Multi-Query Attention: Memory-Efficient LLM Inference  
- RoadmapsLLM, Multi-Query Attention (MQA)  
- EmergentMind, Multi-Query Attention  
- Vaibhav Ahluwalia, Caching Strategies for LLM Systems Part 3: Multi-Query Attention and Memory Efficient Decoding
