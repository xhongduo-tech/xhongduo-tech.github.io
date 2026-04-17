## 核心结论

多查询注意力（MQA，Multi-Query Attention）和分组查询注意力（GQA，Grouped-Query Attention）解决的不是“注意力能不能算出来”，而是“模型在自回归推理时，KV 缓存会不会把显存容量和内存带宽拖成瓶颈”。

先把对象说清楚。所谓 KV 缓存，是指在生成阶段把历史 token 的 Key（键）和 Value（值）保存下来。这样生成下一个 token 时，不需要重新计算整段历史的 K/V，只要计算当前 token 的 Query（查询），再去读取历史缓存做匹配即可。这个做法节省了重复计算，但会引入持续增长的缓存占用。

标准多头注意力（MHA，Multi-Head Attention）里，每个 Query 头都有自己独立的一组 K/V 头，因此缓存最大；MQA 让所有 Query 头共享同一组 K/V，因此缓存最小；GQA 则位于两者之间，把多个 Query 头分组后共享 K/V，在质量和效率之间做折中。

它们的核心关系可以压缩成一句话：

$$
\text{KV Cache} \propto n_{\text{kv\_heads}}
$$

这句话的含义很具体：推理阶段的缓存占用、缓存写入量、以及后续每一步从显存中读取历史 K/V 的带宽压力，主要由 `n_kv_heads` 决定，而不是由总 Query 头数 `n_heads` 决定。

| 机制 | Query 头数 | KV 头数 | KV 缓存占用 | 推理速度倾向 | 精度倾向 |
| --- | --- | --- | --- | --- | --- |
| MHA | H | H | 最高 | 基线 | 通常最高 |
| GQA | H | G，且 `1 < G < H` | 中等 | 通常更快 | 通常接近 MHA |
| MQA | H | 1 | 最低 | 通常最快 | 更容易掉点 |

工程上，GQA 往往是最稳的“甜点区”。原因不是概念上更先进，而是它只压缩 K/V 头，不减少 Query 头。也就是说，模型仍然保留较细粒度的“提问方式”，但把“历史记忆库”的副本数压缩了。在长上下文推理里，这通常能显著降低显存占用和带宽压力，同时比 MQA 更容易保住质量。

公开资料已经给出不少例子。Ainslie 等人在 GQA 论文里明确把它定义为介于 MHA 与 MQA 之间的结构；Meta 的 Llama 2 70B 模型卡明确写到使用 GQA；Mistral 7B 论文也明确说明采用 GQA；OpenAI 在 `gpt-oss` 的公开架构说明中写明使用 grouped multi-query attention，group size 为 8。相反，“某闭源模型一定用了 GQA”这类说法，如果没有官方文档，就只能算行业推测，不能写成结论。

如果只记一句话，可以记这个版本：

> MQA/GQA 的优化目标是减少推理时要缓存和搬运的 K/V 数量，而不是减少 Query 头本身。

---

## 问题定义与边界

这篇文章讨论的问题边界很窄，也必须先限定清楚：

1. 讨论对象是**自回归生成阶段的推理优化**
2. 主要瓶颈是**KV 缓存带来的显存占用和内存带宽压力**
3. 不讨论训练阶段总 FLOPs 是否降低
4. 不讨论稀疏注意力、线性注意力、状态空间模型等“改写注意力形式”的方案

为什么要单独强调“推理阶段”？因为训练和推理不是同一个瓶颈模型。

训练时，模型一次处理整段序列，主要还要考虑前向、反向、激活保存、优化器状态、并行通信等成本。推理时尤其是解码阶段，模型是按 token 逐步生成的，每一步都要读取历史 K/V。此时瓶颈经常不是“算不动”，而是“历史缓存读写太重”。

生成第 $t$ 个 token 时，当前 Query 记为 $Q_t$，历史 Key/Value 记为 $K_{1:t}, V_{1:t}$。注意力计算的核心操作是：

1. 用 $Q_t$ 与历史所有 $K_{1:t}$ 算相似度
2. 经过 softmax 得到权重
3. 用这些权重对历史所有 $V_{1:t}$ 做加权求和

如果不缓存历史 K/V，那么每生成一个新 token，都要把之前整段历史重新投影成 K/V，重复计算非常多。KV 缓存正是为了解决这件事。但一旦缓存打开，问题就变成：历史越长，要保存和读取的 K/V 就越多。

单层 KV 缓存大小的近似公式是：

$$
\text{bytes} \approx 2 \times B \times T \times n_{\text{kv\_heads}} \times d_h \times \text{bytes\_per\_elem}
$$

其中：

- $B$：batch size
- $T$：当前上下文长度
- $n_{\text{kv\_heads}}$：K/V 头数
- $d_h$：每个头的维度
- $\text{bytes\_per\_elem}$：每个元素占用字节数，例如 fp16/bf16 通常为 2
- 前面的 `2`：表示同时缓存 K 和 V

如果模型总共有 $L$ 层，则总 KV 缓存近似为：

$$
\text{total bytes} \approx 2 \times L \times B \times T \times n_{\text{kv\_heads}} \times d_h \times \text{bytes\_per\_elem}
$$

这个公式值得拆开理解。对于固定模型而言，$L$、$d_h$ 常常已经定死，部署时更容易变化的是：

- 序列长度 $T$
- 并发批量 $B$
- 精度类型
- K/V 头数 $n_{\text{kv\_heads}}$

其中最适合靠结构设计提前控制的，就是 `n_kv_heads`。

下面用一个最简单的玩具例子算数量级。假设：

- `B = 1`
- `T = 2048`
- `d_h = 128`
- `bytes_per_elem = 2`（fp16 或 bf16）
- 单层缓存

则有：

| 配置 | `n_kv_heads` | 单层 KV 缓存 |
| --- | --- | --- |
| MHA，32 头 | 32 | 33.6 MB |
| GQA，8 个 KV 头 | 8 | 8.4 MB |
| MQA，1 个 KV 头 | 1 | 1.0 MB |

为什么是这个数量级？直接代入公式：

$$
\text{bytes}_{\text{MHA}} = 2 \times 1 \times 2048 \times 32 \times 128 \times 2 = 33{,}554{,}432
$$

折算成 MiB 约为：

$$
\frac{33{,}554{,}432}{1024^2} = 32 \text{ MiB}
$$

很多文章会把它约写为 `33.6 MB`，这是十进制 MB 和二进制 MiB 混用导致的近似。工程上最好分清：

- `MB`：按 $10^6$ 计
- `MiB`：按 $2^{20}$ 计

同理：

$$
\text{bytes}_{\text{GQA}} = 2 \times 1 \times 2048 \times 8 \times 128 \times 2 = 8{,}388{,}608 \approx 8 \text{ MiB}
$$

$$
\text{bytes}_{\text{MQA}} = 2 \times 1 \times 2048 \times 1 \times 128 \times 2 = 1{,}048{,}576 \approx 1 \text{ MiB}
$$

这还只是单层。如果模型有 32 层，则近似变成：

| 配置 | 单层缓存 | 32 层总缓存 |
| --- | --- | --- |
| MHA | 32 MiB | 1024 MiB，约 1 GiB |
| GQA | 8 MiB | 256 MiB |
| MQA | 1 MiB | 32 MiB |

如果是更长上下文，例如 `T=8192`，则缓存再乘以 4。也就是说，长上下文下 KV 缓存往往会从“一个小优化点”变成“部署能否成立的前置条件”。

这里再给一个边界判断表，方便新手区分问题类型：

| 问题 | MQA/GQA 是否直接解决 |
| --- | --- |
| 推理时 KV 缓存太大 | 是 |
| 解码阶段带宽压力太高 | 是 |
| 训练总 FLOPs 太高 | 否，不是主要目标 |
| 超长序列注意力本身仍要看很多历史位置 | 否，需要别的机制配合 |
| 模型表达能力不足 | 不能直接解决，甚至可能有损失 |

真实工程里，大模型层数高、头数多、上下文长时，KV 缓存会成为主要内存消费者之一。此时 GQA 的价值不是“锦上添花”，而是决定模型能否在给定硬件预算下稳定跑起来。

---

## 核心机制与推导

先把三个矩阵用最直白的方式重新定义一遍：

- Query（Q，查询）：当前 token 想从历史里找什么
- Key（K，键）：历史 token 用什么方式被索引
- Value（V，值）：历史 token 真正提供的内容

这三个名字很抽象，但最重要的是它们在计算里的角色不同。Q 和 K 决定“该关注谁”，V 决定“取回什么内容”。

### 1. 标准 MHA

在标准多头注意力（MHA）中，第 $i$ 个头各自有独立的投影：

$$
Q_i = XW_i^Q,\quad K_i = XW_i^K,\quad V_i = XW_i^V
$$

该头的注意力输出为：

$$
\text{Attn}_i = \text{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_h}}\right)V_i
$$

如果一共有 $H$ 个头，那么就有 $H$ 组独立的 K/V。这样做的好处是每个头都能学习不同的匹配方式和不同的内容表示，表达能力最强；坏处是推理时每个头都要有自己的缓存副本，成本最大。

### 2. MQA

MQA 的变化很直接：每个头仍然保留自己的 Query，但所有头共享同一组 K/V。

数学上可以写成：

$$
Q_i = XW_i^Q,\quad K = XW^K,\quad V = XW^V
$$

于是：

$$
\text{Attn}_i = \text{softmax}\left(\frac{Q_iK^\top}{\sqrt{d_h}}\right)V
$$

这里的变化不是“只有一个头在工作”，而是：

- Query 头依然是多个
- 只是所有 Query 头访问同一个历史记忆库

因此，MQA 不是把多头注意力彻底砍成单头，而是把“提问方式”和“历史记忆库”分离后，只压缩了后者。

### 3. GQA

GQA 处在 MHA 与 MQA 之间。设：

- Query 头总数为 $H$
- KV 头数为 $G$
- 通常要求 $H$ 能被 $G$ 整除

定义映射函数 $g(i)$，表示第 $i$ 个 Query 头对应哪个 KV 组，则：

$$
\text{Attn}_i = \text{softmax}\left(\frac{Q_iK_{g(i)}^\top}{\sqrt{d_h}}\right)V_{g(i)}
$$

如果 `H = 32, G = 8`，则每个 KV 组服务 4 个 Query 头。最常见的规则是均匀分组：

$$
g(i) = \left\lfloor \frac{i}{H/G} \right\rfloor
$$

也就是：

$$
g(i) = \left\lfloor \frac{i}{4} \right\rfloor
\quad \text{for } H=32, G=8
$$

映射结果如下：

| Query 头编号 | 对应 KV 组 |
| --- | --- |
| 0, 1, 2, 3 | 0 |
| 4, 5, 6, 7 | 1 |
| 8, 9, 10, 11 | 2 |
| ... | ... |
| 28, 29, 30, 31 | 7 |

于是缓存缩小比例就是：

$$
\frac{\text{GQA cache}}{\text{MHA cache}} = \frac{G}{H}
$$

例如 `H=32, G=8`：

$$
\frac{\text{GQA cache}}{\text{MHA cache}} = \frac{8}{32} = \frac{1}{4}
$$

`G=1` 时就是 MQA：

$$
\frac{\text{MQA cache}}{\text{MHA cache}} = \frac{1}{32}
$$

### 4. 为什么 GQA 往往比 MQA 更稳

核心原因是它只压缩 K/V 的多样性，不完全压扁 Query 的多样性。

可以把 MHA、GQA、MQA 看成下面这条连续谱：

$$
\text{MHA} \quad \longrightarrow \quad \text{GQA} \quad \longrightarrow \quad \text{MQA}
$$

对应的是：

$$
n_{\text{kv\_heads}}: H \rightarrow G \rightarrow 1
$$

越往右，缓存越省；但 K/V 的表示容量也越被压缩。

对于新手，一个直观理解是：

- Query 头负责“问问题”
- K/V 头负责“保存历史记忆的索引和内容”

GQA 保留了大量不同的提问方式，只让若干提问者共用同一本资料册；MQA 则让所有提问者都共用同一本资料册。后者更省，但更容易因为“所有人都看同一份摘要”而丢失细节。

### 5. 为什么推理速度会提升

在解码阶段，模型一边生成，一边不断追加新的 K/V 到缓存中，同时每一步都要读取历史缓存。若 `n_kv_heads` 下降，则至少有三类成本下降：

1. 新 token 写入缓存的数据量更少
2. 读取历史 K/V 的数据量更少
3. 内核在组织 K/V 相关张量时的带宽压力更低

因此，GQA/MQA 的收益往往在以下场景更明显：

- 长上下文
- 较大 batch
- 层数多的模型
- 生成阶段远比 prefilling 阶段更敏感的部署环境

这里顺便区分两个阶段：

| 阶段 | 主要含义 | GQA/MQA 收益通常是否明显 |
| --- | --- | --- |
| Prefilling | 把已有长提示一次性灌入模型 | 可能有收益，但不一定最明显 |
| Decoding | 一次生成一个 token | 通常更明显 |

因为 decoding 是持续重复读取历史 KV 的过程，所以它最容易被缓存带宽拖慢。

---

## 代码实现

实现 GQA 时，最容易混淆的一点是：

> 最终注意力输出仍然有 `n_heads` 个 Query 头，但真正生成、存储和追加到缓存里的 K/V 只有 `n_kv_heads` 个头。

下面先给一个可以直接运行的最小 Python 例子，验证缓存公式、分组映射和缓存比例。

```python
from math import isclose

def kv_cache_bytes(batch, seq_len, head_dim, n_kv_heads, bytes_per_elem=2, num_layers=1):
    return 2 * batch * seq_len * head_dim * n_kv_heads * bytes_per_elem * num_layers

def bytes_to_mib(num_bytes):
    return num_bytes / (1024 ** 2)

def group_id(query_head_id, n_heads, n_kv_heads):
    assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
    heads_per_group = n_heads // n_kv_heads
    return query_head_id // heads_per_group

def demo():
    n_heads = 32
    n_kv_heads = 8
    batch = 1
    seq_len = 2048
    head_dim = 128

    assert group_id(0, n_heads, n_kv_heads) == 0
    assert group_id(3, n_heads, n_kv_heads) == 0
    assert group_id(4, n_heads, n_kv_heads) == 1
    assert group_id(31, n_heads, n_kv_heads) == 7

    mha = kv_cache_bytes(batch, seq_len, head_dim, n_kv_heads=32)
    gqa = kv_cache_bytes(batch, seq_len, head_dim, n_kv_heads=8)
    mqa = kv_cache_bytes(batch, seq_len, head_dim, n_kv_heads=1)

    assert mha == 33554432
    assert gqa == 8388608
    assert mqa == 1048576

    assert isclose(gqa / mha, 0.25)
    assert isclose(mqa / mha, 1 / 32)

    print(f"MHA: {mha} bytes, {bytes_to_mib(mha):.2f} MiB")
    print(f"GQA: {gqa} bytes, {bytes_to_mib(gqa):.2f} MiB")
    print(f"MQA: {mqa} bytes, {bytes_to_mib(mqa):.2f} MiB")

if __name__ == "__main__":
    demo()
```

预期输出类似：

```text
MHA: 33554432 bytes, 32.00 MiB
GQA: 8388608 bytes, 8.00 MiB
MQA: 1048576 bytes, 1.00 MiB
```

这个例子只验证了“缓存怎么算”，还没有进入张量实现。下面再给一个可运行的 PyTorch 示例，用最直接的方式演示 GQA 的张量形状变化。

```python
import torch

def gqa_attention(x, wq, wk, wv, n_heads, n_kv_heads, head_dim):
    """
    x:  [batch, seq_len, model_dim]
    wq: [model_dim, n_heads * head_dim]
    wk: [model_dim, n_kv_heads * head_dim]
    wv: [model_dim, n_kv_heads * head_dim]
    """
    bsz, seq_len, model_dim = x.shape
    assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
    assert wq.shape == (model_dim, n_heads * head_dim)
    assert wk.shape == (model_dim, n_kv_heads * head_dim)
    assert wv.shape == (model_dim, n_kv_heads * head_dim)

    repeat = n_heads // n_kv_heads

    q = x @ wq
    k = x @ wk
    v = x @ wv

    q = q.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)      # [B, H, T, Dh]
    k = k.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)   # [B, G, T, Dh]
    v = v.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)   # [B, G, T, Dh]

    # 这里为了演示分组共享，显式把 KV 沿头维复制到 H。
    # 真正高效的推理实现通常不会物理复制，而是在 kernel 里做逻辑广播。
    k_expanded = k.repeat_interleave(repeat, dim=1)                  # [B, H, T, Dh]
    v_expanded = v.repeat_interleave(repeat, dim=1)                  # [B, H, T, Dh]

    scale = head_dim ** -0.5
    scores = torch.matmul(q, k_expanded.transpose(-1, -2)) * scale   # [B, H, T, T]
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_expanded)                            # [B, H, T, Dh]

    return out

def demo():
    torch.manual_seed(0)

    batch = 2
    seq_len = 5
    model_dim = 32
    n_heads = 4
    n_kv_heads = 2
    head_dim = 8

    x = torch.randn(batch, seq_len, model_dim)
    wq = torch.randn(model_dim, n_heads * head_dim)
    wk = torch.randn(model_dim, n_kv_heads * head_dim)
    wv = torch.randn(model_dim, n_kv_heads * head_dim)

    out = gqa_attention(x, wq, wk, wv, n_heads, n_kv_heads, head_dim)
    assert out.shape == (batch, n_heads, seq_len, head_dim)
    print("output shape:", out.shape)

if __name__ == "__main__":
    demo()
```

这段代码可以运行，但要明确两件事。

### 1. 它是“教学实现”，不是高性能实现

`repeat_interleave` 在这里是为了让形状更直观。它把 `[B, G, T, Dh]` 扩成 `[B, H, T, Dh]`，便于后续和 Query 头一一对齐。但在真正的推理框架里，如果你把 K/V 真复制出来，可能会把本来节省的显存又补回去，所以高性能实现通常会在 CUDA kernel 或 fused attention kernel 内部处理这种组映射，而不是物理复制张量。

### 2. 缓存里应该存原始的 `n_kv_heads` 版本

真正部署时，KV cache 应该按下面的形状存：

$$
K_{\text{cache}}, V_{\text{cache}} \in \mathbb{R}^{B \times G \times T \times d_h}
$$

而不是：

$$
\mathbb{R}^{B \times H \times T \times d_h}
$$

后者只是为了计算方便临时展开，前者才是真正省内存的关键。

### 3. 一个更接近工程现实的伪代码

如果把解码阶段拆开，逻辑更像下面这样：

```python
# 仅示意，不是完整框架代码
# 每来一个新 token，只新增当前 token 的 q, k, v

q_t = project_q(x_t)   # [B, H, Dh]
k_t = project_k(x_t)   # [B, G, Dh]
v_t = project_v(x_t)   # [B, G, Dh]

append_to_kv_cache(k_t, v_t)  # 缓存按 G 个头存储

# 在 attention kernel 内部：
# 对于第 i 个 query head，读取对应的 KV group = g(i)
# 而不是事先把 cache 物理复制到 H 个头
out_t = grouped_attention(q_t, kv_cache, head_to_group_map)
```

### 4. 新手最容易踩的形状坑

下面这个表能快速排雷：

| 张量 | MHA 常见形状 | GQA 常见形状 |
| --- | --- | --- |
| Q 投影输出 | `[B, T, H, Dh]` | `[B, T, H, Dh]` |
| K 投影输出 | `[B, T, H, Dh]` | `[B, T, G, Dh]` |
| V 投影输出 | `[B, T, H, Dh]` | `[B, T, G, Dh]` |
| KV Cache | `[B, H, T, Dh]` | `[B, G, T, Dh]` |
| 最终输出头数 | `H` | 仍然是 `H` |

如果只记一条实现规则，可以记这个：

> GQA 改的是 K/V 的生成与缓存方式，不是把输出头数从 `H` 变成 `G`。

---

## 工程权衡与常见坑

### 坑 1：把 GQA 理解成“改个配置就行”

这通常不成立。一个已经训练好的 MHA 模型，如果直接把 `num_key_value_heads` 从 `H` 改成更小值，会立刻遇到几个问题：

- K/V 投影矩阵形状变了
- 头和组的映射关系变了
- 历史权重语义不再一一对应
- 配合 RoPE 等位置编码时，原本各头的结构也发生了变化

结果通常不是“小幅波动”，而是前向输出明显漂移，质量直接下降。

Ainslie 等人的论文给出的实用途径是：从 MHA checkpoint 出发，把模型改成 GQA/MQA 结构后，再进行少量继续训练（uptraining），用较小比例的额外计算把模型重新适配到共享 K/V 的结构上。论文里强调，这样可以在较低额外成本下把质量拉回到接近 MHA 的水平。

### 坑 2：忽略整除关系

最常见的工程设置都要求：

$$
H \bmod G = 0
$$

也就是 `n_heads % n_kv_heads == 0`。

原因不是数学上绝对做不到非整除，而是非整除会让很多事情都复杂化：

- 分组大小不均匀
- kernel 实现更麻烦
- 张量切分不规整
- 张量并行和流水并行更难处理
- 性能调优空间变差

因此工程上通常优先选择规整组合，例如：

| `n_heads` | 可选 `n_kv_heads` |
| --- | --- |
| 32 | 32, 16, 8, 4, 2, 1 |
| 64 | 64, 32, 16, 8, 4, 2, 1 |
| 128 | 128, 64, 32, 16, 8, 4, 2, 1 |

实际常见选择往往不是越小越好，而是选一个既能显著压缩缓存、又不明显伤质量的点。

### 坑 3：误以为 GQA 一定提升总吞吐

这也不成立。GQA 主要优化的是 K/V 相关的内存与带宽成本。如果系统瓶颈根本不在这里，而是在：

- 大矩阵乘算力
- kernel launch 开销
- 通信开销
- CPU 侧调度
- 采样和后处理

那么 GQA 带来的整体加速可能就没有预期大。

比较稳妥的判断方式是把性能拆成两个阶段测：

| 指标 | 建议单独测量 |
| --- | --- |
| Prefill latency | 长提示一次性输入时的延迟 |
| Decode latency | 每生成一个 token 的平均延迟 |
| Tokens/s | 不同上下文长度下的吞吐 |
| Peak memory | 峰值显存占用 |
| KV cache size | 理论与实际缓存占用是否一致 |

如果 decode latency 明显下降，而 prefill 变化不大，这通常就说明收益主要来自 KV 缓存读写压力下降，而不是整体算子都变快了。

### 坑 4：把“省缓存”和“省训练成本”混为一谈

GQA 并不等于训练成本按同样比例下降。原因很简单：

- 训练需要完整前向和反向
- 激活保存仍然是大头之一
- 数据并行、张量并行、流水并行的通信还在
- 优化器状态和梯度也不受 KV cache 公式直接控制

因此这类结构最直接的收益场景是**推理部署**，尤其是长上下文解码，而不是“训练预算直接减少相同比例”。

### 坑 5：错误地物理复制 K/V

这是实现层面很常见的坑。很多初学者先写出一个能跑的 GQA，再为了方便把 K/V 用 `repeat` 或 `repeat_interleave` 扩成与 Query 一样的头数，最后把这个扩展结果也存进缓存。这样会发生什么？

- 逻辑上还是 GQA
- 物理上又退回了接近 MHA 的缓存大小

所以要区分两件事：

- 计算时为了对齐形状，临时逻辑广播 K/V
- 缓存时按原始 `n_kv_heads` 存

前者可以存在，后者必须坚持。

### 坑 6：只看平均质量，不看长上下文退化

GQA/MQA 的质量影响不一定在短基准上立即明显，但可能在以下场景放大：

- 长文档问答
- 多轮对话记忆
- 代码补全中的远距离依赖
- 检索增强生成里的长证据串联

因此评估不能只看一个总分。更好的做法是分场景看：

| 场景 | 为什么要重点看 |
| --- | --- |
| 短上下文常规任务 | 检查是否有基础能力回退 |
| 长上下文任务 | 检查共享 K/V 是否损伤远距离依赖 |
| 推理速度 | 验证是否真的换来延迟收益 |
| 显存占用 | 验证理论压缩是否落地到实现 |

### 一个更贴近工程的迁移路径

| 阶段 | 建议动作 | 原因 |
| --- | --- | --- |
| 结构设计 | 先确定 `n_heads` 与 `n_kv_heads`，优先整除 | 简化 kernel、分组与并行切分 |
| 权重初始化 | 按组聚合原 K/V 头做初始化 | 比随机初始化更稳定 |
| 继续训练 | 用少量预训练语料做 uptraining | 恢复共享 K/V 带来的表达损失 |
| 推理接入 | KV cache 元数据改为按 `n_kv_heads` 管理 | 真正获得缓存收益 |
| 内核优化 | 避免物理复制 K/V，改用组映射广播 | 防止把显存节省抵消掉 |
| 性能评测 | 分开测 prefill 与 decode | 两个阶段的瓶颈不同 |
| 质量评测 | 分短上下文与长上下文任务 | 避免只看平均分误判 |

如果要把这条路径再压缩成一句工程建议，可以写成：

> GQA 不是“把参数名改一下”，而是“模型结构、权重迁移、缓存布局和推理 kernel 都要一起改”。

---

## 替代方案与适用边界

从表达能力和缓存成本的角度看，MHA、GQA、MQA 不是三种互不相干的类别，而是一条连续谱。

当 `G = H` 时，GQA 退化为 MHA：

$$
G = H \Rightarrow \text{GQA} = \text{MHA}
$$

当 `G = 1` 时，GQA 退化为 MQA：

$$
G = 1 \Rightarrow \text{GQA} = \text{MQA}
$$

所以实际问题不是“哪个更先进”，而是“在给定显存预算、延迟目标和质量约束下，选哪个点最合适”。

### 三者对比

| 方案 | 优点 | 缺点 | 更适合 |
| --- | --- | --- | --- |
| MHA | 表达最充分，结构最标准 | KV 缓存最大，长上下文推理压力最高 | 训练优先、质量优先、资源较充足场景 |
| GQA | 质量与效率平衡，工程上最常用 | 需要选组数并处理权重迁移/内核适配 | 长上下文推理、在线服务、主流生产部署 |
| MQA | 缓存最小，实现概念直接 | 更容易损伤质量，尤其在复杂长依赖任务中 | 极限压缩、边缘设备、超严格内存预算 |

### 一个对新手足够实用的选择规则

| 你的主要约束 | 更优先考虑 |
| --- | --- |
| 显存很紧，但不能明显掉质量 | GQA |
| 显存极紧，愿意为部署牺牲更多质量 | MQA |
| 推理不是核心成本，训练/表达能力更重要 | MHA |
| 模型已经存在且不想大改架构 | 看是否支持 GQA uptraining，而不是硬改配置 |

也可以把它压成三句判断：

- 显存紧，但质量还重要，先看 GQA
- 显存极端紧张，才认真考虑 MQA
- 推理压力不大时，MHA 仍然是最直接的基线

### GQA 解决什么，不解决什么

这部分很重要，因为很多文章会把 GQA 的边界写模糊。

GQA 直接解决的是：

- KV cache 占用过大
- 解码阶段读取历史 K/V 的带宽压力
- 长上下文推理时的部署成本

GQA 不直接解决的是：

- 注意力对超长序列仍需访问大量历史位置的问题
- 训练总计算量与总通信量的问题
- 模型本身在任务上的上限能力问题

因此当上下文进一步增大到更极端尺度时，GQA 往往还要和其他策略配合使用，例如：

| 方案 | 主要解决的问题 |
| --- | --- |
| 滑动窗口注意力 | 缩小每步需要关注的历史范围 |
| 块稀疏注意力 | 让注意力矩阵只计算部分块 |
| 页式 KV 缓存管理 | 提高缓存管理和复用效率 |
| 量化 KV cache | 进一步压缩缓存字节数 |
| 连续批处理/动态批处理 | 提升服务端吞吐效率 |

换句话说，GQA 解决的是“历史记忆副本太多”，但不解决“要看的历史位置本身仍然很多”。

### 什么时候 GQA 不是第一优先级

下面这些情况里，GQA 可能不是先动的旋钮：

| 场景 | 说明 |
| --- | --- |
| 主要瓶颈在 CPU 采样、调度或网络 I/O | 先优化系统栈更直接 |
| 上下文很短，且 batch 很小 | KV cache 还没大到成为主瓶颈 |
| 模型质量要求极高，任何结构改动都要慎重 | 可能先保持 MHA |
| 已有高性能内核和缓存系统深度绑定 MHA | GQA 的接入成本可能较高 |

最后给一句最稳妥的边界判断：

> GQA 是推理阶段极其实用的 KV 缓存压缩方案，但它不是所有注意力瓶颈的总开关。

---

## 参考资料

- Joshua Ainslie 等，GQA 原始论文，定义了从 MHA checkpoint 迁移到 GQA/MQA 的 uptraining 路径，并给出“质量接近 MHA、速度接近 MQA”的核心结论：[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://aclanthology.org/2023.emnlp-main.298/)
- Meta 的 Llama 2 70B 模型卡，公开说明 70B 使用 GQA，是工程落地中最常被引用的公开实例之一：[Llama 2 70B Chat Model Card](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf/blob/main/README.md)
- Mistral 7B 论文，明确写到模型使用 GQA 与 sliding window attention 组合来降低推理成本：[Mistral 7B](https://arxiv.org/abs/2310.06825)
- Hugging Face 的 Llama 文档，可作为配置层面的辅助入口，帮助理解 `num_key_value_heads` 这类实现参数在模型库中的对应关系：[Transformers Llama Docs](https://huggingface.co/docs/transformers/model_doc/llama)
- Sebastian Raschka 的技术文章，对 KV 缓存公式、分组比例和 MHA/GQA/MQA 的关系有清晰的数量级推导，适合做直观补充：[Grouped-Query Attention](https://sebastianraschka.com/llms-from-scratch/ch04/04_gqa/)
- IBM 的入门解释文章，适合第一次接触 MQA/GQA 的读者建立概念地图，但应与论文和模型卡交叉阅读：[What is grouped query attention?](https://www.ibm.com/think/topics/grouped-query-attention)
- OpenAI 对 `gpt-oss` 的公开架构说明，明确写到 grouped multi-query attention，且 group size 为 8，可作为近年公开模型中的新增例子：[Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/)
