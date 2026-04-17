## 核心结论

GQA，Grouped Query Attention，中文可叫“分组查询注意力”，它的核心做法是：**保留每个注意力头各自独立的 Query，改为让多个头共享同一组 Key/Value**。白话说，判断问题的人还是很多个，但查资料的档案柜按组共用。

这件事在长上下文模型里很关键，因为 decoder 推理时最贵的不是重新算一遍历史，而是把历史 token 的 K/V 缓存在显存里反复读取。MHA，Multi-Head Attention，中文叫“多头注意力”，每个头都要存自己的 K/V；上下文一长，KV cache 会直接按头数膨胀。GQA把缓存单位从“头”改成“组”，于是把 KV cache 从 $O(T\cdot H\cdot d_h)$ 压到 $O(T\cdot G\cdot d_h)$。

如果头数是 $H=32$，组数是 $G=8$，那么缓存就从 32 份缩成 8 份，约等于原来的 $1/4$。这不是只省一点显存，而是直接影响长上下文推理是否能跑、能否提高吞吐、以及 GPU 带宽是否成为瓶颈。

一个适合理解的玩具例子是：32 个侦探各自形成判断，但他们共享 8 个资料档案柜。侦探还是 32 个，所以“怎么看问题”仍然有多样性；档案柜只有 8 个，所以“存多少历史资料”明显变少。这就是 GQA 相对 MHA 和 MQA 的中间位置：**比 MHA 省资源，比 MQA 保留更多表达能力**。

| 机制 | 头数 $H$ | 组数 $G$ | KV cache 规模 |
|---|---:|---:|---:|
| MHA | 32 | 32 | $O(T\cdot 32\cdot d_h)$ |
| GQA | 32 | 8 | $O(T\cdot 8\cdot d_h)$ |
| MQA | 32 | 1 | $O(T\cdot 1\cdot d_h)$ |

工程上，组数常取 8 或 16。原因不是神秘经验，而是这个区间往往能在**质量接近 MHA**的前提下，换来 **2 倍到 4 倍级别的 KV 带宽和缓存节省**。因此，GQA 已经成为长上下文 decoder 模型里的主流折中方案。

---

## 问题定义与边界

先定义问题。这里讨论的是 **decoder-only Transformer 在自回归推理阶段** 的注意力缓存问题，不是训练阶段的全序列并行，也不是 encoder-only 模型。

当模型逐 token 生成时，当前 token 的 Query 需要和历史所有 token 的 Key 做匹配，再用权重对历史 Value 求加权和。为了避免每生成一个 token 都把历史重新投影一次，系统会把每一层每个历史 token 的 K/V 存进 KV cache。于是显存和带宽成本随着上下文长度 $T$ 线性增长。

对 MHA 来说，若有 $H$ 个头、每头维度 $d_h$，则单层缓存量近似为：

$$
\text{KV cache} \propto T \cdot H \cdot d_h
$$

更准确地说，K 和 V 各一份，所以常写成：

$$
\text{bytes} = 2 \cdot T \cdot H \cdot d_h \cdot \text{dtype\_bytes}
$$

这正是长上下文的压力来源。上下文从 2K 增长到 32K，不只是“序列长了 16 倍”，而是**每一层都要背着 16 倍历史缓存走**。

MQA，Multi-Query Attention，中文叫“多查询单 KV 注意力”，把所有头共享同一份 K/V，即 $G=1$。它极端节省资源，但也最容易损失注意力多样性。白话说，大家问的问题不同，但只能查同一个档案柜，很多头最后会互相挤压表达空间。

GQA 就是在两者中间插入一个可调参数 $G$：

$$
G=H \Rightarrow \text{MHA}
$$

$$
1<G<H \Rightarrow \text{GQA}
$$

$$
G=1 \Rightarrow \text{MQA}
$$

于是 KV cache 规模变成：

$$
\text{KV cache} \propto T \cdot G \cdot d_h
$$

这说明 GQA 的边界非常清楚：

| 维度 | MHA | GQA | MQA |
|---|---|---|---|
| 多样性 | 最高 | 中等到高 | 最低 |
| KV 开销 | 最高 | 可调 | 最低 |
| 长上下文友好性 | 一般 | 好 | 最强 |
| 质量风险 | 最低 | 可控 | 较高 |

这里要强调一个边界：GQA主要解决的是**推理期缓存和带宽问题**，不是万能加速器。若模型瓶颈在前馈层、采样逻辑、通信开销，或者上下文本来就很短，GQA 的收益会变小。

---

## 核心机制与推导

GQA 的核心不是“减少 Query 头”，而是“只减少 K/V 头”。这点必须分清。

设模型总头数为 $H$，分成 $G$ 组，每组有：

$$
r = \frac{H}{G}
$$

个 Query 头共享一组 K/V。对于第 $i$ 个 Query 头，它属于组：

$$
g(i) = \left\lfloor \frac{i}{r} \right\rfloor
$$

注意力计算可以写成：

$$
\text{Attn}_i(Q_i, K_{g(i)}, V_{g(i)}) = \text{softmax}\left(\frac{Q_i K_{g(i)}^\top}{\sqrt{d_h}}\right)V_{g(i)}
$$

这里最关键的地方是：**$Q_i$ 仍然每头独立**。所以即使多个头共享同一组 K/V，它们依然可能学到不同的查询模式。白话说，几个人共用一个资料库，不代表他们会得出相同结论，因为他们问的问题不同。

这也是为什么 GQA 不等价于“粗暴砍头”。如果直接减少总头数，模型会同时失去查询多样性和表示容量；而 GQA 保留了大部分查询侧的多样性，只压缩了缓存侧的冗余。

从复杂度看，MHA 单层 KV cache 为：

$$
\underbrace{T}_{上下文长度}
\cdot
\underbrace{H}_{头数}
\cdot
\underbrace{d_h}_{每头维度}
$$

GQA 改为：

$$
T \cdot G \cdot d_h
$$

因此压缩率是：

$$
\frac{\text{GQA cache}}{\text{MHA cache}}=\frac{G}{H}
$$

如果 $H=32, G=8$，则压缩率为：

$$
\frac{8}{32}=\frac{1}{4}
$$

也就是只保留四分之一的 K/V 缓存。

玩具例子可以直接算。设：

- 上下文长度 $T=2048$
- 头数 $H=32$
- 组数 $G=8$
- 每头维度 $d_h=128$
- 数据类型 FP16，即每个数 2 字节

那么单层 MHA 的 KV cache 为：

$$
2 \cdot 2048 \cdot 32 \cdot 128 \cdot 2 \approx 33.5\text{ MB}
$$

GQA 则是：

$$
2 \cdot 2048 \cdot 8 \cdot 128 \cdot 2 \approx 8.4\text{ MB}
$$

差距不是抽象公式，而是每层少了约 25 MB。模型层数一多，总量就会非常明显。

经验上，组数和质量常见关系大致如下：

| 组数 $G$ | 相对 MHA 压缩率 | 质量风险 | 典型判断 |
|---:|---:|---|---|
| 32 | 1x | 极低 | 等价于 MHA |
| 16 | 1/2 | 低 | 常见安全区 |
| 8 | 1/4 | 低到中 | 长上下文常用折中 |
| 4 | 1/8 | 中到高 | 需仔细验证 |
| 1 | 1/32 | 高 | 即 MQA，极限压缩 |

真实工程例子是长上下文 decoder 模型。以 Gemini 一类系统为代表，目标不是只看论文指标，而是让模型在数万 token 会话中仍有可接受的延迟和吞吐。此时 GPU 往往不是纯算力不够，而是**显存容量与缓存读取带宽先顶到墙**。GQA 的价值就在这里：每层只缓存少量组 K/V，显著缓解带宽压力，同时保持比 MQA 更稳定的 perplexity 和任务质量。

---

## 代码实现

实现 GQA 时，最重要的改动有两处：**缓存结构**和**从 MHA 权重迁移到 GQA 权重**。

第一处是缓存结构。MHA 通常存 `[B, H, T, d_h]` 的 K/V；GQA 改成 `[B, G, T, d_h]`。计算时，Query 仍然是 `[B, H, 1, d_h]`，只是它要按分组映射到对应的 K/V。

下面是一个可运行的 Python 玩具实现，用来展示 KV cache 大小和分组映射关系：

```python
import math

def kv_cache_bytes(seq_len, num_kv_heads, head_dim, dtype_bytes=2):
    # K 和 V 各一份
    return 2 * seq_len * num_kv_heads * head_dim * dtype_bytes

def grouped_head_map(num_q_heads, num_kv_groups):
    assert num_q_heads % num_kv_groups == 0
    ratio = num_q_heads // num_kv_groups
    return [h // ratio for h in range(num_q_heads)]

H = 32
G = 8
T = 2048
D = 128

mha_bytes = kv_cache_bytes(T, H, D)
gqa_bytes = kv_cache_bytes(T, G, D)

assert mha_bytes == 4 * gqa_bytes
assert grouped_head_map(8, 2) == [0, 0, 0, 0, 1, 1, 1, 1]

print("MHA MB:", round(mha_bytes / (1024 * 1024), 2))
print("GQA MB:", round(gqa_bytes / (1024 * 1024), 2))
```

对应的伪代码可以写成：

```python
# q: [B, H, 1, Dh]
# k_cache, v_cache: [B, G, T, Dh]
# 每 r 个 query head 共享一组 KV
r = H // G

outputs = []
for h in range(H):
    g = h // r
    score = softmax((q[:, h] @ k_cache[:, g].transpose(-1, -2)) / sqrt(Dh))
    out = score @ v_cache[:, g]
    outputs.append(out)
```

第二处是从 MHA checkpoint 迁移到 GQA。论文和工程实践中常用的办法是 **mean-pooling 合并 K/V 权重**。白话说，原来一组里有多个独立 K/V 头，现在把它们先取平均，得到组级别 K/V 初始化，再做少量额外训练恢复性能。

伪代码如下：

```python
# 原始 MHA 权重: [H, d_model, Dh]
# 目标 GQA 权重: [G, d_model, Dh]
assert H % G == 0
r = H // G

for g in range(G):
    start = g * r
    end = (g + 1) * r
    Wk_gqa[g] = Wk_mha[start:end].mean(axis=0)
    Wv_gqa[g] = Wv_mha[start:end].mean(axis=0)
```

一个简单流程可以概括为：

1. 从已有 MHA checkpoint 读取参数。
2. 按组对 K/V projection 权重做 mean-pooling。
3. 保留 Query projection 不变，替换成 GQA 结构。
4. 用少量额外训练继续优化，让新分组重新适应任务分布。

这个过程常被称为 uptraining。它不是“从头重训”，而是“在已有好模型上做结构迁移后的补偿训练”。论文结论表明，这种补偿的额外算力只需原训练预算的一小部分，约 5% 量级，就能让 GQA 逼近 MHA 质量。

---

## 工程权衡与常见坑

GQA 的核心权衡很直接：**组数越小，缓存越省；但共享越强，注意力多样性越容易受损**。

常见坑有三个。

第一，组数压得过低。比如从 32 头直接压到 2 组，虽然缓存极省，但很多本应分散建模的模式会挤到同一组 K/V 上，表现为 perplexity 上升、检索精度下降、长链依赖更脆弱。白话说，这像把原来 8 条路并成 2 条，局部一定会堵。

第二，只改结构，不做迁移恢复。如果直接随机初始化新的组级 K/V 权重，模型通常会明显退化。正确做法是：**降组 -> K/V mean-pooling -> uptraining**。这条链路缺一不可。

第三，忽略并行策略。真实推理系统里，GQA 不只是数学结构变化，还会影响张量并行和缓存布局。如果 KV 组数和设备切分方式不协调，可能理论上省了缓存，实际却在通信上补回来，收益被吃掉。

下面这张表更适合工程判断：

| 组数设置 | 质量风险 | 记忆/速度收益 | 典型问题 |
|---|---|---|---|
| 接近头数，如 16/32 | 低 | 中等 | 节省不够激进 |
| 中等组数，如 8 | 低到中 | 高 | 多数场景的折中点 |
| 很小组数，如 2/4 | 中到高 | 很高 | 长依赖和细粒度模式受损 |
| 单组，即 MQA | 高 | 极高 | 容易出现能力塌缩 |

还有一个常被忽略的点：GQA 节省的是 **推理期 KV cache**，所以它对长上下文、多轮对话、流式生成最有价值；如果业务输入普遍只有几百 token，那么收益远没有 32K、128K 上下文时明显。

---

## 替代方案与适用边界

如果把 MHA、GQA、MQA 看成一条连续谱，就比较容易选型。

MHA 适合短上下文、追求最强表示能力的场景。它保留每个头独立的 Q/K/V，多样性最好，但缓存和带宽成本最高。对于上下文不长、资源充足的任务，MHA 仍然是最稳的基线。

MQA 适合极限压缩场景。所有头共享一份 K/V，资源最省，但质量风险最大。它更像“省电模式”，适用于对延迟和成本异常敏感、且能接受一定精度损失的场景。

GQA 适合大多数长上下文 decoder 模型。它把组数 $G$ 作为一个旋钮，允许工程上按资源预算调节。白话说，MHA 是专业模式，MQA 是省电模式，GQA 是自动模式：不追求理论最强，也不追求极端压缩，而是争取最划算的综合点。

| 方案 | 适合上下文长度 | 质量 | 资源消耗 | 适用边界 |
|---|---|---|---|---|
| MHA | 短到中 | 最高 | 最高 | 精度优先，资源宽松 |
| GQA | 中到长 | 高 | 中等 | 长上下文主流折中 |
| MQA | 很长 | 中到低 | 最低 | 极端带宽/显存受限 |

因此，GQA 不是“比 MHA 更先进”的统一替代，而是**在长上下文时代更符合系统约束**。只要瓶颈来自 KV cache，GQA 往往就是最实用的结构改造；但若任务极度依赖细粒度头间差异，或者上下文很短，MHA 仍可能更合适。

---

## 参考资料

- IBM, *What is grouped query attention (GQA)?*：解释 GQA 作为 MHA 与 MQA 之间的统一框架，以及其在 Gemini、Llama 等长上下文模型中的工程价值。https://www.ibm.com/think/topics/grouped-query-attention
- Mandliya, *Model Architecture Optimizations*：给出 KV cache 的数量级公式和 32 头、8 组、2048 token 的具体数值例子。https://mandliya.github.io/posts/model_architecture_optimizations/
- Emergent Mind, *Grouped Query Attention*：总结 GQA 的定义、复杂度公式、组数与质量之间的经验关系。https://www.emergentmind.com/topics/grouped-query-attention
- Emergent Mind, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*：概述 EMNLP 2023 论文中的 uptraining、mean-pooling 和恢复精度思路。https://www.emergentmind.com/articles/2305.13245
- ACL Anthology, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*：原始论文入口，重点看从 MHA checkpoint 迁移到 GQA 的实验设计与结果。https://aclanthology.org/2023.emnlp-main.298/
- Inflexion AI 知识文章：提供面向初学者的直观解释，适合理解“Q 仍独立、KV 按组共享”的基本图景。https://www.inflexionai.site/knowledge-hub/gqa
