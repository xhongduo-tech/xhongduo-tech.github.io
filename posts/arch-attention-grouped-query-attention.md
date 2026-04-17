## 核心结论

Grouped-Query Attention，简称 GQA，可以理解为“多组 Query 头共享较少数量的 Key/Value 头”的注意力结构。白话说，不是每个注意力头都单独维护一套 KV，而是若干个 Query 头共用同一套 Key 和 Value。

它解决的核心问题不是训练速度，而是**自回归解码时的 KV cache 开销**。KV cache 就是推理阶段把历史 token 的 Key/Value 存下来，后续生成新 token 时直接复用，避免重复计算。对大模型来说，推理常常不是算力先到极限，而是显存带宽和缓存读写先到极限。

GQA 的关键结论有三条：

| 结构 | Query 头数 | 独立 KV 头数 | KV cache 相对大小 | 质量趋势 | 推理效率趋势 |
|---|---:|---:|---:|---|---|
| MHA | $h$ | $h$ | $1$ | 最高 | 最慢 |
| GQA | $h$ | $G$ | $G/h$ | 接近 MHA | 明显更快 |
| MQA | $h$ | $1$ | $1/h$ | 最容易掉点 | 最快 |

第一，GQA 是 MHA 和 MQA 之间的中间态。MHA 是 Multi-Head Attention，白话说是“每个头各管各的”；MQA 是 Multi-Query Attention，白话说是“所有 Query 共用同一套 KV”；GQA 则是在两者之间做分组共享。

第二，GQA 的缓存压缩比例非常直接：如果有 $h$ 个 Query 头、$G$ 个 KV 组，那么 KV cache 大小会缩小到原来的 $G/h$，也就是节省 $h/G$ 倍的 KV 重复存储。

第三，工程上它已经不是实验概念。LLaMA 2/3、Mistral 等模型都采用了 GQA，一种常见配置是 `32 个 Query 头 -> 8 个 KV 组`。这意味着每 4 个 Query 头共享 1 套 KV，KV cache 只保留原来的 $1/4$。

一个玩具例子：如果班里有 32 个学生提问，每个人问题不同，相当于 32 个 Query 头；但老师只准备 8 份参考资料，相当于 8 组 KV。每 4 个学生共享 1 份资料。学生提问角度仍然不同，但资料存储量从 32 份降到 8 份。

一个真实工程例子：LLaMA-2 70B 和 LLaMA-3 系列在长上下文推理时，KV cache 会占用大量显存。采用 GQA 后，同样长度的上下文能容纳更大的 batch，或者在相同显存下支持更长序列，这对在线推理服务很关键。

---

## 问题定义与边界

先定义问题。GQA 讨论的是**推理阶段的注意力缓存优化**，不是“任何时候都能免费提速”的通用技巧。它主要影响的是 decoder-only Transformer，也就是 GPT、LLaMA 这一类自回归模型。

为什么推理阶段会卡在 KV cache 上？因为生成第 $t$ 个 token 时，当前 Query 只需要算一次，但它要和前面 $1 \sim t-1$ 的所有 Key 做匹配，还要把对应的 Value 加权求和。所以历史 token 的 KV 必须保留。

在标准 MHA 中，若：

- batch size 是 $B$
- 序列长度是 $L$
- 注意力头数是 $h$
- 每个头的维度是 $d_h$

那么每层 KV cache 的规模近似是：

$$
\text{KV}_{\text{MHA}} = 2 \cdot B \cdot L \cdot h \cdot d_h
$$

其中前面的 2 表示 K 和 V 各存一份。

在 GQA 中，独立 KV 头数从 $h$ 变成 $G$，于是：

$$
\text{KV}_{\text{GQA}} = 2 \cdot B \cdot L \cdot G \cdot d_h
$$

两者相除可得：

$$
\frac{\text{KV}_{\text{GQA}}}{\text{KV}_{\text{MHA}}}=\frac{G}{h}
$$

这就是为什么 GQA 的收益非常清楚，它不是“可能节省”，而是公式上直接节省。

下面用一个小表看压缩率：

| $h$ | $G$ | 每组 Query 数 $h/G$ | KV 保留比例 $G/h$ | KV 节省比例 |
|---:|---:|---:|---:|---:|
| 32 | 32 | 1 | 1 | 0% |
| 32 | 8 | 4 | 1/4 | 75% |
| 32 | 4 | 8 | 1/8 | 87.5% |
| 32 | 1 | 32 | 1/32 | 96.875% |

边界也要说清楚：

1. GQA 不会减少 Query 头数。它减少的是**独立 KV 头数**。
2. GQA 不会自动提升训练质量。它的核心收益在推理内存和带宽。
3. GQA 不是越小的 $G$ 越好。$G$ 太小会逼更多 Query 共用同一套 KV，表达能力会下降。
4. GQA 最适合在**长上下文、大 batch、服务端推理**场景发挥优势。短序列、小模型、低并发下，收益可能不明显。

如果把注意力头想成“提问的人”，那 Query 是问题，Key/Value 是被检索的资料。MHA 是每个人带一份自己的资料，最灵活但最贵；MQA 是全班共用一份资料，最省但容易不够细；GQA 是按小组共用资料，属于成本和表达力之间的折中。

---

## 核心机制与推导

GQA 的结构本质很简单：Query 仍然保留 $h$ 个头，但 Key 和 Value 只保留 $G$ 组。

设第 $j$ 个 Query 头属于第 $g(j)$ 组，那么它的注意力仍然是标准形式：

$$
\text{Attn}(Q_j, K_{g(j)}, V_{g(j)})=
\text{softmax}\left(\frac{Q_j K_{g(j)}^\top}{\sqrt{d_k}}\right)V_{g(j)}
$$

这条公式很重要，因为它说明了两件事：

1. **Query 仍然独立发问。**每个 $Q_j$ 不同，所以不同头仍能关注不同位置。
2. **KV 被组内共享。**同组的多个 Query 头使用同一套 $K_g, V_g$。

这就是 GQA 的核心分离：**Query 的多样性保留，KV 的冗余被压缩。**

### 组共享为什么还能工作

在 MHA 里，不同头往往学习到不同的注意模式，比如有的头偏向局部语法，有的头偏向远距离依赖，有的头偏向实体对齐。GQA 并不是直接取消这些 Query 头，而是让这些头在“检索资料库”时共享资料表示。

这意味着模型仍能保留一部分多头分工能力，因为真正决定“看哪里”的是 $QK^\top$，而不是只有 K/V 本身。只要共享后的 KV 没有损失过多结构信息，多个 Query 头仍能形成差异化关注。

### 从 MHA 到 GQA 的退化关系

GQA 可以看成一个连续家族：

- 当 $G=h$ 时，每个 Query 头都有独立 KV，退化为 MHA。
- 当 $G=1$ 时，所有 Query 头共享一套 KV，退化为 MQA。
- 当 $1 < G < h$ 时，是中间折中。

这个关系很重要，因为它告诉我们 GQA 不是全新注意力公式，而是**改变 KV 头共享粒度**。

### 玩具例子：32 头分 8 组

设：

- Query 头数 $h=32$
- KV 组数 $G=8$
- 每组大小 $s=h/G=4$

那么映射关系可以写成：

- 头 `0,1,2,3` 使用 `KV group 0`
- 头 `4,5,6,7` 使用 `KV group 1`
- ...
- 头 `28,29,30,31` 使用 `KV group 7`

此时每层 KV cache 从：

$$
2 \cdot B \cdot L \cdot 32 \cdot d_h
$$

变成：

$$
2 \cdot B \cdot L \cdot 8 \cdot d_h
$$

压缩比是：

$$
32/8=4
$$

也就是 KV cache 只保留原来的四分之一，节省 75%。

### 真实工程例子：长上下文推理服务

设一个在线推理服务要跑 `batch=16`、`context=8192` 的大语言模型。此时每层都要保存大量历史 token 的 KV。假设模型本来是 32 个头的 MHA，那么 KV cache 会随着层数线性累加，显存占用和读取带宽都很重。

如果换成 `G=8` 的 GQA：

- Query 头仍是 32，保持较强表达力
- KV 头降到 8，缓存规模立刻降为 1/4
- 同样显存下可以放更大 batch，或者支持更长上下文
- 在连续解码阶段，每步需要搬运的缓存更少，因此吞吐更高

这也是为什么很多现代大模型在训练时直接选 GQA，而不是等部署时再想办法压缩。

### 简化图示

可以用下面的结构理解：

| Query 头 | 所属组 | 共享的 KV |
|---|---:|---:|
| $Q_0,Q_1,Q_2,Q_3$ | 0 | $(K_0,V_0)$ |
| $Q_4,Q_5,Q_6,Q_7$ | 1 | $(K_1,V_1)$ |
| ... | ... | ... |
| $Q_{28},Q_{29},Q_{30},Q_{31}$ | 7 | $(K_7,V_7)$ |

组内“多个 Query 独立算分数，共用同一组 K/V”，就是 GQA 全部机制的核心。

---

## 代码实现

实现 GQA 时，最重要的是区分两件事：

1. **结构定义**：怎么把 `h` 个 Query 头映射到 `G` 个 KV 组。
2. **权重初始化**：如果从已有 MHA checkpoint 改造，怎么把原来每头独立的 K/V 投影合并成组共享投影。

常见做法是 `group_size = h // G`，然后用：

$$
g(j)=\left\lfloor \frac{j}{\text{group\_size}} \right\rfloor
$$

把第 `j` 个 Query 头映射到第 `g(j)` 个组。

如果从 MHA checkpoint 转到 GQA，一个实用初始化方法是对组内原始 K/V 权重做 mean-pool。mean-pool 就是“把一组向量逐元素取平均”，白话说是把多份头权重合成一份组权重。

下面给一个可运行的 Python 玩具实现，演示三件事：

- 如何建立 `head_to_group`
- 如何对 K/V 权重做组内平均
- 如何验证 KV cache 压缩比例

```python
import math

def build_head_to_group(num_q_heads: int, num_kv_groups: int):
    assert num_q_heads > 0
    assert num_kv_groups > 0
    assert num_q_heads % num_kv_groups == 0

    group_size = num_q_heads // num_kv_groups
    mapping = [head // group_size for head in range(num_q_heads)]

    assert len(mapping) == num_q_heads
    assert min(mapping) == 0
    assert max(mapping) == num_kv_groups - 1
    return mapping

def mean_pool_heads(head_vectors, num_groups):
    """
    head_vectors: list[list[float]]
    每个元素代表一个 head 的投影参数，长度必须一致。
    """
    num_heads = len(head_vectors)
    assert num_heads % num_groups == 0
    group_size = num_heads // num_groups
    dim = len(head_vectors[0])

    for vec in head_vectors:
        assert len(vec) == dim

    pooled = []
    for g in range(num_groups):
        start = g * group_size
        group = head_vectors[start:start + group_size]
        avg = [sum(values) / group_size for values in zip(*group)]
        pooled.append(avg)

    assert len(pooled) == num_groups
    assert all(len(vec) == dim for vec in pooled)
    return pooled

def kv_cache_units(batch, seq_len, num_kv_heads, head_dim):
    # 用“元素个数”表示缓存规模，省略 dtype 字节数
    return 2 * batch * seq_len * num_kv_heads * head_dim

# 玩具例子：32 个 Query 头分成 8 组
h = 32
G = 8
mapping = build_head_to_group(h, G)
assert mapping[:8] == [0, 0, 0, 0, 1, 1, 1, 1]
assert mapping[-4:] == [7, 7, 7, 7]

# 构造 32 个假的 K 头权重，每个头维度为 3
k_heads = [[float(i), float(i + 1), float(i + 2)] for i in range(h)]
pooled_k = mean_pool_heads(k_heads, G)

assert len(pooled_k) == 8
assert pooled_k[0] == [1.5, 2.5, 3.5]  # 头 0,1,2,3 的均值
assert pooled_k[1] == [5.5, 6.5, 7.5]  # 头 4,5,6,7 的均值

# 验证 KV cache 缩放
batch = 2
seq_len = 1024
head_dim = 128

mha_cache = kv_cache_units(batch, seq_len, h, head_dim)
gqa_cache = kv_cache_units(batch, seq_len, G, head_dim)

assert gqa_cache * 4 == mha_cache
assert gqa_cache / mha_cache == 0.25

print("head_to_group ok")
print("mean pooling ok")
print("KV cache reduced to 1/4")
```

如果把这个逻辑放进真实 Transformer 层，伪代码大致是：

```python
group_size = h // G
head_to_group = [head // group_size for head in range(h)]

for token in sequence:
    q = project_q(token)         # shape: [h, d]
    k = project_k(token)         # shape: [G, d]
    v = project_v(token)         # shape: [G, d]

    kv_cache_k.append(k)
    kv_cache_v.append(v)

    outputs = []
    for head in range(h):
        g = head_to_group[head]
        scores = softmax(q[head] @ kv_cache_k[:, g, :].T / math.sqrt(d))
        out = scores @ kv_cache_v[:, g, :]
        outputs.append(out)
```

这里最关键的不是语法，而是映射逻辑：

- `group_size = h / G`
- `head_to_group[head]` 决定当前 Query 头应访问哪一组 KV
- KV cache 只按 `G` 存，不按 `h` 存

如果是从已有 MHA 模型改造，常见工程路线是：

1. 保留原始 Query 投影头数不变。
2. 把每组对应的 K/V 投影做 mean-pool，初始化为共享权重。
3. 用少量继续预训练或蒸馏，让模型适应新的共享结构。

论文里一个重要经验是：从 MHA checkpoint 出发，经 mean-pool 初始化后再做少量 uptraining，通常比从头训练 GQA 更现实。

---

## 工程权衡与常见坑

GQA 的价值明确，但实现时有几个坑非常常见。

| 问题 | 现象 | 原因 | 建议 |
|---|---|---|---|
| 不做 uptraining | 精度下降 | 共享 KV 改变了原模型分工 | 用 MHA checkpoint mean-pool 初始化后继续训练 |
| `h % G != 0` | 映射不均匀 | 某些组 Query 更多，组间负载不平衡 | 优先保证整除，避免不等大小分组 |
| `G` 太小 | 质量接近 MQA，掉点明显 | 过多 Query 共用同一套 KV | 从 4、8 这类中等值开始试 |
| 只看显存不看带宽 | 实测加速不明显 | 某些场景瓶颈不在 KV | 结合 batch、上下文长度、硬件一起评估 |
| 框架实现不匹配 | shape 错误或 cache 索引错位 | Query 头数和 KV 头数维度定义不同 | 明确区分 `num_heads` 与 `num_kv_heads` |

### 坑 1：以为“直接缩 KV 头数”就行

这是最常见误区。假设你有一个已经训练好的 MHA 模型，直接把 K/V 投影从 32 头改成 8 头，然后随机初始化，这通常会明显掉质量。原因很简单：原模型学到的是“每头独立 KV 表示”，而新结构变成“组共享 KV 表示”，函数族已经变了。

更稳妥的做法是：

- 先把一组内原始 K 权重做均值合并
- 再把一组内原始 V 权重做均值合并
- 以此作为 GQA 初始权重
- 然后做少量 uptraining

这相当于从原模型的近邻点开始优化，而不是重新从随机点出发。

### 坑 2：忽略整除关系

如果某层有 33 个 Query 头，你硬设 `G=8`，那就会出现分组不均匀。比如前 7 组每组 4 个头，最后 1 组有 5 个头。这样做不是绝对不行，但会带来两个问题：

1. 实现复杂度更高，索引和 layout 不统一。
2. 某些组承担更多 Query，mean-pool 初始化也不再对称。

对工程系统来说，统一 shape 很重要。所以大多数实现会优先选择让 `h % G == 0`。

### 坑 3：把 GQA 的收益理解成“纯显存节省”

实际上它的关键收益常常是**显存带宽和缓存搬运量**。推理每生成一个 token，都要读历史 KV。读得越多，越容易卡在内存系统而不是计算单元。因此 GQA 在长序列和连续解码时的收益往往比在短 prompt 时更明显。

### 坑 4：G 选太小

虽然 `G=1` 的 MQA 最省缓存，但它也最容易带来质量损失。GQA 之所以被广泛采用，正是因为它在“足够省”和“不要掉太多”之间找到了中间点。很多模型选 `G=8`，本质上是经验上比较稳的折中，而不是数学上唯一最优。

### 真实工程例子：服务端模型部署

假设你维护一个问答服务，模型需要处理长文档问答。原本 MHA 版本在 `batch=8, context=16k` 时显存接近上限，吞吐也不理想。切换到 GQA 后：

- 每层 KV cache 降低
- 相同 GPU 可承载更大的并发
- 在峰值流量时不必频繁降 batch
- 质量相比 MQA 更稳，尤其是复杂多轮问答

但前提是你不能只改推理代码，必须保证模型权重本身已经按照 GQA 结构训练或适配过。

---

## 替代方案与适用边界

GQA 不是唯一方案。它只是当前工程上性价比较高的一种。

先看三种基础结构的定位：

| 类型 | 独立 KV 头数 | 缓存开销 | 质量 | 典型定位 |
|---|---:|---|---|---|
| MHA | $h$ | 最高 | 最稳 | 训练优先、显存宽裕 |
| GQA | $G$ | 中等 | 接近 MHA | 主流大模型推理 |
| MQA | 1 | 最低 | 风险最大 | 极致压缩、极致吞吐 |

如果目标是“尽量不掉质量”，MHA 仍然最安全，因为每个头都有完整自由度。如果目标是“显存和速度压到极限”，MQA 最激进，但质量风险更高。GQA 的价值就是给出一个中间带。

还有一些替代或扩展思路：

| 方法 | 核心想法 | 优点 | 局限 |
|---|---|---|---|
| QCQA | 数据驱动分组 | 分组更聪明 | 实现和训练更复杂 |
| WGQA | 加权聚合 KV | 比简单平均更灵活 | 额外参数和训练成本 |
| AsymGQA | 非均匀分组 | 能针对层或头做差异化设计 | 系统实现复杂 |
| KV Cache Quantization | 量化缓存数值精度 | 不改注意力结构也能省显存 | 可能影响数值稳定性 |
| Sliding Window / Chunk Attention | 只保留局部上下文 | 长序列很省 | 不适合必须全局依赖的任务 |

适用边界也要明确：

1. 如果你是从零训练中小模型，且推理不是瓶颈，MHA 可能更简单。
2. 如果你做的是大模型在线服务，尤其是长上下文推理，GQA 往往是默认优先项。
3. 如果你极端追求吞吐，并且任务对质量掉点不太敏感，可以考虑 MQA。
4. 如果你已经在做 KV 量化、PagedAttention、FlashAttention，那么 GQA 可以和这些技术叠加，它不是互斥关系。

一个简单选型原则：

- 先问自己瓶颈是不是 KV cache。
- 如果是，再看是否需要接近 MHA 的质量。
- 如果两者都成立，优先试 GQA，而不是直接跳到 MQA。

---

## 参考资料

- Ainslie, Joshua, et al. “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.” EMNLP 2023. https://aclanthology.org/2023.emnlp-main.298/
- IBM Think. “What is grouped query attention (GQA)?” https://www.ibm.com/think/topics/grouped-query-attention
- Emergent Mind. “Grouped-Query Attention (GQA).” https://www.emergentmind.com/topics/group-query-attention-gqa
- Paras Dahal. “Grouped Query Attention (GQA).” https://parasdahal.com/notes//Grouped%2BQuery%2BAttention%2B%28GQA%29
- Omri Mallis. “Techniques for KV Cache Optimization.” https://www.omrimallis.com/posts/techniques-for-kv-cache-optimization/
