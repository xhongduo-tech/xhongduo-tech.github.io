## 核心结论

KV Cache 压缩的本质，是把“每生成一个 token 就要继续增长的注意力记忆”压到 GPU 能承受的范围内。KV Cache 可以直白理解为：模型为了后续继续看历史内容，提前把每个 token 的 Key 和 Value 向量存起来，避免重复计算。

核心公式先给出：

$$
\text{KV\_bytes} \approx 2 \times H \times d \times L \times b \times s_{\text{bytes}}
$$

其中：

- $H$ 是层数
- $d$ 是每个 KV head 的维度
- $L$ 是当前上下文长度
- $b$ 是每个 token 实际保存的 KV head 数
- $s_{\text{bytes}}$ 是每个数值占多少字节，比如 FP16/BF16 通常是 2 字节
- 前面的系数 2 表示同时存 K 和 V

这条公式说明了一个直接事实：KV Cache 会随 $L$ 线性增长。上下文越长，缓存越大；并发越高，总显存越容易爆掉。

主流压缩方法可以分成三类：

| 类别 | 直接控制的变量 | 代表方法 | 典型效果 |
|------|----------------|----------|----------|
| 共享 KV 头 | 降低 $b$ | MQA、GQA | 内存直接按头数比例下降 |
| 缩短保留长度 | 限制有效 $L$ | Sliding Window | 内存从线性增长变成近似常数上限 |
| 只保留重要 token | 稀疏化有效 $L$ | H2O、SnapKV | 用较小缓存保留大部分关键信息 |

在工程上，GQA 是最稳妥的基础配置，Sliding Window 是最直接的上限控制，H2O 和 SnapKV 则是在“还不够省”的情况下继续挤内存。它们常常不是互斥关系，而是组合关系。

---

## 问题定义与边界

问题不是“KV Cache 会不会占内存”，而是“它会在线上系统里快到什么程度把显存吃光”。

先看一个简化表。下面用给定配置展示上下文增长时的 KV Cache 规模，重点是观察线性趋势：

| Context Length | KV Cache (per request) |
|----------------|------------------------|
| 2,048          | ~0.6 GB                |
| 32,768         | ~10 GB                 |
| 128,000        | ~40 GB                 |

这张表的含义很直接：如果单请求 128K token 已经要 40GB，那么 4 个并发请求就是约 160GB，仅 KV Cache 就可能超过单卡部署的显存预算。

这里的边界要说清楚：

1. KV Cache 只影响推理，不影响训练主流程。它是生成阶段为了复用历史注意力结果而保留的中间状态。
2. 压缩目标不是一味追求最小内存，而是在“显存、吞吐、质量”三者之间找平衡。
3. 某些策略需要模型结构先支持，例如 GQA/MQA 常常是在模型设计阶段就定下来的；而 Sliding Window、H2O、SnapKV 更接近推理侧策略。
4. 长上下文场景不只看单请求长度，还必须看并发。线上系统真正爆显存，往往不是因为一个 128K 请求，而是 8 个中长请求同时进来。

玩具例子先给一个最低门槛版本。

假设一个模型有 32 层，每层原本使用 32 个 attention head，每个 head 的 KV 维度是 128，上下文长度是 128K，精度用 2 字节。则：

- 普通多头注意力 MHA：每个 token 存 32 份 KV
- 如果改成 GQA，8 个查询头共享 1 组 KV，那么只需要 8 份 KV
- 内存会从约 32GB 降到约 8GB

这就是为什么很多人第一次接触 GQA 时会觉得“只是改了 head 分组，怎么省这么多”。原因很简单，因为它直接改的是公式里的 $b$。

真实工程例子更说明问题。以 70B 级模型、128K 上下文为例，单请求就可能吃掉数十 GB KV Cache。如果服务还要留出模型权重、激活、中间 buffer 和调度空间，那么即使使用大显存 GPU，也会因为并发而迅速逼近上限。这时不做压缩，系统很难稳定上线。

---

## 核心机制与推导

### 1. 为什么 KV Cache 线性增长

自回归生成时，第 $t$ 个 token 需要关注前面 $1 \ldots t-1$ 的历史 token。为了不在每一步都重新计算旧 token 的 K/V，系统会把它们缓存下来。

所以当上下文从 $L$ 变成 $L+1$ 时，缓存不是重算，而是追加一条记录。于是总内存近似满足：

$$
\text{KV\_bytes}(L) = O(L)
$$

如果再考虑 batch size 或并发请求数 $N$，总缓存近似变成：

$$
\text{Total\_KV} \approx N \times \text{KV\_bytes}(L)
$$

这就是线上服务里“上下文长度和并发数双重放大”的根源。

### 2. GQA / MQA：通过共享降低 $b$

GQA 是 Grouped Query Attention，白话解释是：多个查询头共用一组 Key/Value 头。MQA 是它的更极端版本，所有查询头共享 1 组 KV。

假设查询头数为 $n_q$，KV 头数为 $n_{kv}$，则每个 token 需要保存的 KV 份数，从 $n_q$ 下降到 $n_{kv}$。因此缓存占用大致按下面的比例变化：

$$
\frac{\text{KV after}}{\text{KV before}} \approx \frac{n_{kv}}{n_q}
$$

例如：

- MHA：$n_q = 32, n_{kv}=32$
- GQA：$n_q = 32, n_{kv}=8$
- MQA：$n_q = 32, n_{kv}=1$

则内存比例分别约为：

- MHA：$32/32 = 1$
- GQA：$8/32 = 1/4$
- MQA：$1/32$

所以 GQA 能稳定拿到 4 倍左右的容量收益，而 MQA 虽然更省，但精度退化通常更明显。

### 3. Sliding Window：直接限制有效历史长度

Sliding Window，白话解释是：模型不是永远保留全部历史，而是只保留最近一段窗口，比如最近 4096 或 8192 个 token。

这相当于把公式里的 $L$ 从“总长度”改成“窗口长度 $w$”：

$$
\text{KV\_bytes} \approx 2 \times H \times d \times w \times b \times s_{\text{bytes}}
$$

如果 $w$ 固定，内存就不会随着总生成长度无限增长。工程上通常用环形缓冲区实现，也就是新 token 进来时覆盖最老位置。

它的代价也很明确：窗口之外的信息会消失。如果任务依赖很远的上下文，质量会下降。

### 4. H2O / SnapKV：不是全删，而是选重要的删

H2O 可以理解为 Heavy Hitter Oracle，白话解释是：把历史里反复被关注、对后续最有用的一批 token 留下来，同时再给最近 token 留预算。

SnapKV 的核心思路是：在 prefill 阶段先看一段观察窗口，估计哪些历史位置在后续最可能重要，只把这些关键位置的 KV 保留到解码阶段。

两者都不是简单“留最近”或“留最早”，而是在做预算分配：

$$
L_{\text{effective}} = L_{\text{recent}} + L_{\text{important}}
$$

其中：

- $L_{\text{recent}}$ 保证最近上下文不丢
- $L_{\text{important}}$ 保证远处但关键的信息不丢

这类方法比纯 Sliding Window 更适合长链推理、长文问答、多跳检索，因为它们尝试保住“远程但关键”的记忆。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，演示三件事：

1. 如何计算不同 KV 头配置下的缓存大小
2. 如何用固定窗口做 Sliding Window eviction
3. 如何做一个极简版的“按分数保留 top-k token”

```python
from collections import deque
from math import isclose

def kv_cache_bytes(num_layers, kv_heads, head_dim, seq_len, bytes_per_elem=2):
    # 2 表示 K 和 V 各一份
    return 2 * num_layers * kv_heads * head_dim * seq_len * bytes_per_elem

def to_gb(num_bytes):
    return num_bytes / (1024 ** 3)

# 玩具例子：32 层，128 维，128K token
layers = 32
head_dim = 128
seq_len = 128 * 1024

mha_bytes = kv_cache_bytes(layers, kv_heads=32, head_dim=head_dim, seq_len=seq_len)
gqa_bytes = kv_cache_bytes(layers, kv_heads=8, head_dim=head_dim, seq_len=seq_len)
mqa_bytes = kv_cache_bytes(layers, kv_heads=1, head_dim=head_dim, seq_len=seq_len)

assert isclose(gqa_bytes / mha_bytes, 0.25)
assert isclose(mqa_bytes / mha_bytes, 1 / 32)

# 简化的 sliding window
window = deque(maxlen=4)
for token_id in [10, 11, 12, 13, 14]:
    window.append(token_id)

assert list(window) == [11, 12, 13, 14]

# 简化的 snapkv top-k 选择
scores = {
    0: 0.1,
    1: 0.9,
    2: 0.3,
    3: 0.8,
    4: 0.2,
}
k = 2
selected = sorted(scores, key=scores.get, reverse=True)[:k]

assert selected == [1, 3]

print("MHA GB:", round(to_gb(mha_bytes), 2))
print("GQA GB:", round(to_gb(gqa_bytes), 2))
print("MQA GB:", round(to_gb(mqa_bytes), 2))
print("Sliding window:", list(window))
print("Selected tokens:", selected)
```

上面这段代码没有实现真实注意力，只实现了压缩策略的骨架。它对应的直觉是：

- `kv_cache_bytes` 演示公式如何工作
- `deque(maxlen=4)` 演示窗口满了就淘汰最旧 token
- `selected == [1, 3]` 演示按重要性分数保留关键位置

下面再给一个更接近推理系统的数据流伪代码。

```python
def prefill_snapkv(attn_scores, obs_start, obs_end, keep_k):
    # attn_scores: [num_heads, seq_len]
    keep_indices = {}
    for head in range(len(attn_scores)):
        window_scores = attn_scores[head][obs_start:obs_end]
        topk_local = top_k_indices(window_scores, keep_k)
        keep_indices[head] = [obs_start + i for i in topk_local]
    return keep_indices

def decode_step(query, kv_cache, keep_indices, ring_buffer):
    outputs = []
    for layer in kv_cache.layers:
        for head in layer.heads:
            kv = kv_cache[layer][head]

            # 重要位置 + 最近窗口位置
            sparse_kv = gather(kv, keep_indices[head] + ring_buffer.live_positions())
            out = attend(query[layer][head], sparse_kv)
            outputs.append(out)

    new_kv = project_to_kv(query)
    ring_buffer.append(new_kv)  # 满了就覆盖最旧 token
    return merge(outputs)
```

真实工程里，这段逻辑会更复杂，至少还要处理：

- 每层、每头独立预算
- prefill 与 decode 的不同访存模式
- 稀疏索引带来的 gather/scatter 开销
- CUDA kernel 是否支持非连续 KV block
- attention mask 与位置编码是否还能保持一致

真实工程例子可以这样理解。假设你在做一个长文档问答服务：

- 基础模型已使用 GQA，把 KV 头从 32 压到 8
- 再设置 Sliding Window，只保留最近 8K token
- 但考虑到用户可能问“第一页提到的指标是多少”，于是额外用 SnapKV/H2O 留下少量远程关键 token

这样系统就不是“要么全保留，要么全删除”，而是变成“三层防线”：

1. GQA 先把每个 token 的单位存储成本降下来
2. Window 把总历史长度做硬上限
3. H2O/SnapKV 把少量真正重要的远程信息救回来

这比单独使用某一个策略更符合线上部署逻辑。

---

## 工程权衡与常见坑

先看对比表：

| 策略 | 常见坑 | 规避 |
|------|--------|------|
| Sliding Window | 丢失远程依赖 | 增大窗口，或增加 global path / retrieval path |
| SnapKV | 观察窗口没覆盖真正关键 token | 观察窗口不要过短，并为最近 token 保留固定预算 |
| H2O | 不同 head 的重要性分布不同，统一预算会误删 | 按头分预算，设最小保留量 |
| GQA/MQA | 过度共享导致表达能力下降 | 优先用 GQA，不轻易直接切到 MQA |
| KV 量化 | 访存省了，但数值误差变大 | 先在低敏感层试，分层评估精度退化 |

几个最常见的误区需要单独指出。

第一，很多人以为 Sliding Window 一开，内存问题就彻底解决了。不是。它只是把内存上限控制住，但如果任务依赖远程事实，正确率可能立刻下降。比如窗口设成 512，而问题问的是 10000 token 之前的一条定义，这种场景很容易答错。

第二，很多人以为重要性筛选一定比窗口更优。也不是。重要性方法的收益依赖“分数是否真的能代表未来会不会再用到”。如果打分机制偏了，模型会删掉看起来冷门、但实际上决定答案的 token。

第三，压缩比和吞吐不是一回事。某些稀疏方法虽然理论内存更省，但如果需要复杂的 gather/scatter，或者 KV block 不连续，实际 kernel 效率未必更好。线上系统最终看的是端到端吞吐，而不是纸面压缩率。

第四，压缩策略会和模型架构强耦合。不同模型的层数、KV 头数、RoPE 设置、cache layout 都不同。一个在模型 A 上有效的 budget，不一定能直接平移到模型 B。

实际调参时，建议按这个顺序做：

1. 先确定单请求最大上下文和目标并发
2. 计算未压缩 KV 占用
3. 优先启用模型原生支持的 GQA
4. 再用 Sliding Window 给出硬上限
5. 最后在确实需要时叠加 H2O 或 SnapKV

这个顺序的原因是：前两步是算账，第三步最稳，第四步最直接，第五步最灵活但也最复杂。

---

## 替代方案与适用边界

除了 GQA、Sliding Window、H2O、SnapKV，另一条路线是进一步压缩数值本身，也就是降低 $s_{\text{bytes}}$。例如把 FP16/BF16 的 KV 改成更低比特量化，甚至到 2-bit 级别。MiniKV 一类方法属于这个方向。

还有一类方法不直接依赖“过去观察到的注意力”，而是估计未来 query 可能会关注哪些位置，再提前预测哪些 KV 可以删。这类方法可以概括为 Expected Attention 路线。

下面给一个简化对比表：

| 方法 | 压缩原理 | 适用边界 | 主要质量权衡 |
|------|----------|----------|--------------|
| Expected Attention | 预测未来 query 分布删 KV | 希望在 prefill 和 decode 都做压缩 | 预测不准会误删重要位置 |
| SnapKV | 观察窗口重要性 + top-k 保留 | 长上下文问答、摘要、对话 | 可能漏掉低频但关键 token |
| MiniKV | 低比特量化 + 分层预算 | 极端显存紧张场景 | 数值误差积累可能伤精度 |
| H2O | heavy hitter + 最近 token 平衡 | 长链推理、跨段引用 | 预算分配需要细调 |
| Sliding Window | 固定窗口截断历史 | 对局部上下文依赖更强的任务 | 远程依赖容易丢失 |

适用边界可以这样理解：

- 如果你的模型本身已经是 GQA，那么它就是默认起点，不必再把 MHA 当主线方案。
- 如果你的任务是聊天、代码补全、局部改写，Sliding Window 常常先够用。
- 如果你的任务是长文问答、跨章节检索、多跳推理，就需要 H2O 或 SnapKV 这类“保关键远程信息”的策略。
- 如果你只有一张 80GB 卡，还要扛 64K 甚至更长上下文，且允许轻微质量下降，量化型 KV 压缩会变得有吸引力。
- 如果系统要求不改模型、主要改推理后端，那么推理期的重要性筛选通常比重新训练更现实。

一个实用判断标准是：先问你的任务更怕什么。

- 更怕 OOM：优先 GQA + Window
- 更怕远程依赖丢失：优先 H2O / SnapKV
- 更怕极限显存不够：再考虑 KV 量化
- 更怕后端实现复杂度：少叠策略，先上最稳的两层

---

## 参考资料

- SnapKV，NeurIPS 2024，核心思路是通过观察窗口估计重要 token，并在解码阶段只保留关键 KV。
- H2O: Heavy-Hitter Oracle for Efficient Generative Inference，核心思路是保留高频重要 token 与最近 token 的平衡集合。
- Oakland L08 handout，包含 MHA、GQA、MQA 的 KV 内存公式与推导。
- Spheron GPU Memory Guide，给出了长上下文模型在不同 context length 下的显存占用量级示例。
- MiniKV / Expected Attention 相关资料，讨论了低比特量化和未来注意力分布预测两条压缩路线。
