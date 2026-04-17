## 核心结论

Mistral-7B 的 `Sliding Window Attention`，直白说就是“每一层只看最近一段历史，而不是看全部历史”。它把单层注意力窗口固定为 $W=4096$，所以单层计算和单层 KV Cache 都不再随着总上下文长度线性膨胀。

真正重要的不是“只能看 4096”，而是“32 层叠起来以后，信息可以逐层往前传”。论文给出的理论感受野是：

$$
\text{theoretical receptive field} = L \times W = 32 \times 4096 = 131072
$$

这里的“感受野”可以理解为“最后一层某个位置，理论上最多能间接接触到多远之前的信息”。这不是说模型在 131K token 上一定和全注意力一样强，而是说它在结构上具备跨多个窗口传递信息的路径。

另一个关键点是 `Rolling Buffer KV Cache`。白话说，它不是把所有历史 token 的 K、V 一直追加下去，而是只保留一个固定长度为 $W$ 的环形缓冲区。第 $i$ 个位置的 KV 写到 `cache[i mod W]`，超过窗口就覆盖旧值。因此推理期 KV 显存复杂度从“随序列变长”变成“固定为 $O(W)$”。

对工程部署来说，这个设计的意义非常直接：Mistral 可以在长序列生成时保持固定大小的 KV Cache；再配合 FlashAttention 2 的滑窗实现与 chunked prefill，预填充阶段也可以处理超过 4096 的提示词，而不是被窗口长度直接卡死。

---

## 问题定义与边界

传统因果自注意力里，位置 $i$ 会看见它之前的全部 token。白话说，就是“当前词能翻完整本前文”。这样做的效果好理解，但推理阶段有两个成本：

| 方案 | 单层可直接访问历史 | KV Cache 增长方式 | 理论跨层感受野 | 适合场景 |
|---|---:|---:|---:|---|
| 全注意力 | 全部历史 | 随序列长度增长 | 全部历史 | 短上下文、高精度依赖 |
| 固定滑窗 | 最近 $W$ 个 token | 固定为 $O(W)$ | 单层仅 $W$ | 长上下文、控显存 |
| 滑窗 + 多层堆叠 + Rolling Buffer | 最近 $W$ 个 token | 固定为 $O(W)$ | 约 $L \times W$ | 长上下文推理、在线生成 |

Mistral 要解决的问题，不是“单层怎么更聪明”，而是“推理时怎么支持长上下文，而不让 KV Cache 无限长大”。

这里要先划清边界：

1. 滑窗注意力限制的是“单层直接可见范围”，不是“模型完全忘掉更早内容”。
2. $L \times W$ 是理论传播上界，不等于实际任务效果上界。
3. Rolling Buffer 解决的是“缓存显存”问题，不直接降低模型参数显存。
4. 预填充超过窗口长度可行，前提是实现支持分块 prefill 和正确的滑窗 mask。

一个新手容易混淆的点是：“既然每层只看 4096，那模型怎么处理 10K、20K、甚至更长输入？”答案是两部分一起工作：

- 结构层面：旧信息可以通过中间层隐藏状态逐层向后传。
- 工程层面：prompt 可以按块预填充，缓存只保留最近窗口。

**玩具例子**

设窗口 $W=5$，层数 $L=3$。第一层第 12 个 token 只能直接看 8 到 12；第二层第 12 个 token 可以接触第一层里已经汇总过的更早信息；到第三层时，理论上最远可追溯到 $12-15$ 左右的位置。也就是：

$$
L \times W = 3 \times 5 = 15
$$

这就是“单层局部，叠层全局得更远”的基本思想。

---

## 核心机制与推导

先看滑窗本身。对第 $k$ 层、位置 $i$ 的隐藏状态 $h_i^{(k)}$，Mistral 只让它访问上一层中区间 $[i-W, i]$ 的表示。白话说，就是“当前层只向上一层借最近一段历史”。

如果把每一层都看成一次“向前搬运信息”的过程，那么每过一层，最远可传播 $W$ 个 token。于是经过 $L$ 层，理论上最远传播距离就是：

$$
R(L) = L \times W
$$

对 Mistral-7B：

- $L=32$
- $W=4096$

所以：

$$
R(32) = 32 \times 4096 = 131072
$$

这就是论文里“约 131K 理论注意力跨度”的来源。

但真正让它在推理中省显存的，是 Rolling Buffer。定义一个长度为 $W$ 的缓存数组 `cache[0...W-1]`。第 $i$ 个 token 的 K、V 写入：

$$
\text{slot}(i) = i \bmod W
$$

含义很简单：

- 当 $i < W$ 时，缓存逐步填满。
- 当 $i \ge W$ 时，新 token 会覆盖最早一轮写入的槽位。
- 因为当前层本来也只需要最近 $W$ 个 token，所以被覆盖掉的更旧 KV 不再需要。

可以把它想成钟表式循环写入。若 $W=4$，写入槽位序列就是：

| token 位置 $i$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 写入槽位 $i \bmod 4$ | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 |

这就是“环形缓冲区”。白话说，缓存只有 4 个格子，第 5 个 token 来了就复用第 1 个格子。

KV Cache 的内存可以直接估算。对解码模型，一层每个 token 要存 K 和 V 两份，因此：

$$
\text{KV bytes} = 2 \times L \times W \times n_{kv} \times d_{head} \times \text{dtype\_bytes}
$$

Mistral-7B 中：

- $L=32$
- $W=4096$
- $n_{kv}=8$，即 KV 头数为 8
- $d_{head}=128$
- fp16 下 `dtype_bytes = 2`

代入可得：

$$
2 \times 32 \times 4096 \times 8 \times 128 \times 2 = 536{,}870{,}912 \text{ bytes}
$$

约等于 `512 MiB`。这就是为什么很多工程文章会说“单 batch 下 Mistral 的滚动 KV 大约半个 GB”。

**真实工程例子**

假设你在一张 24GB 显卡上部署 `mistralai/Mistral-7B-v0.1` 做在线问答：

- 用户 prompt 可能有 12K token；
- 生成时还要继续解码；
- 如果你按传统 append KV 的方式缓存 12K 全历史，KV 会继续随长度增长；
- 如果启用滑窗 + rolling buffer，解码阶段始终只维护 4096 长度的缓存。

这样做的结果不是“模型参数更小”，而是“长对话时显存不会因为 KV 失控而突然爆掉”。这对服务端吞吐尤其关键。

---

## 代码实现

下面先给一个最小可运行的 Python 玩具实现，用来验证两个事实：

1. 槽位写入遵守 `i mod W`
2. 理论感受野是 `L * W`

```python
def rolling_slots(seq_len: int, window: int):
    return [i % window for i in range(seq_len)]

def theoretical_receptive_field(layers: int, window: int) -> int:
    return layers * window

# 玩具例子：W=5, L=3
W = 5
L = 3
slots = rolling_slots(seq_len=12, window=W)

assert slots == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
assert theoretical_receptive_field(L, W) == 15

# Mistral-7B 的理论跨度
assert theoretical_receptive_field(32, 4096) == 131072

def kv_cache_bytes(layers, window, n_kv_heads, head_dim, dtype_bytes):
    return 2 * layers * window * n_kv_heads * head_dim * dtype_bytes

mistral_kv = kv_cache_bytes(
    layers=32,
    window=4096,
    n_kv_heads=8,
    head_dim=128,
    dtype_bytes=2,  # fp16
)

assert mistral_kv == 536_870_912
assert mistral_kv / 1024 / 1024 == 512

print("all checks passed")
```

上面这段代码没有实现注意力，只实现了“缓存映射和容量估算”这两个最核心的工程逻辑。

如果写成更接近推理框架的伪代码，核心循环通常长这样：

```python
W = 4096
k_cache = [None] * W
v_cache = [None] * W

for i, x_i in enumerate(tokens):
    q_i, k_i, v_i = project(x_i)

    slot = i % W
    k_cache[slot] = k_i
    v_cache[slot] = v_i

    visible_k, visible_v = gather_last_window(
        k_cache, v_cache, current_pos=i, window=W
    )

    y_i = sliding_window_attention(
        q_i, visible_k, visible_v, window=W
    )
```

如果接入 FlashAttention 2，思路是两段：

1. `prefill` 阶段把长 prompt 按块处理；
2. `decode` 阶段在固定大小 KV buffer 上原地更新。

示意代码如下：

```python
# 伪代码，不是可直接运行的完整生产实现
model = load_mistral(attn_implementation="flash_attention_2")
window = 4096

# 1. 分块预填充 prompt
for chunk in chunked(prompt_tokens, size=window):
    model.prefill(chunk, sliding_window=window)

# 2. 解码阶段：只维护固定窗口
for step in range(max_new_tokens):
    next_token = model.decode_one(
        sliding_window=window,
        use_rolling_kv=True,
    )
    generated.append(next_token)
```

在 Hugging Face 的 Mistral 文档中，`flash-attn >= 2.3.0` 被明确要求用于滑窗注意力支持。工程上常见的配置是：

- `attn_implementation="flash_attention_2"`
- 半精度加载，如 `torch.float16` 或 `bfloat16`
- 左填充 batch，避免 batched generation 下缓存对齐问题
- 使用模型的绝对位置索引参与位置编码，而不是把缓存槽位误当作真实位置

最后一点很关键：`cache[i mod W]` 只是存储位置，不是 token 的真实绝对位置。RoPE 这类位置编码仍然必须基于真实 token index。

---

## 工程权衡与常见坑

滑窗注意力不是“免费午餐”。它是在“直接全历史可见性”和“可部署性”之间做交换。

先看最常见的容量估算。以 Mistral-7B、fp16、rolling buffer 开启为例：

| Batch/Beam 数 | Rolling Buffer KV | 若上下文扩到 32K 且不滚动 | 结论 |
|---|---:|---:|---|
| 1 | 0.5 GB | 4 GB | 单路还能勉强扛 |
| 4 | 2 GB | 16 GB | 很快挤压参数和激活空间 |
| 8 | 4 GB | 32 GB | 大多数单卡直接不可部署 |

这里 32K 不滚动约为 4096 的 8 倍，所以从 `512 MiB` 变成约 `4 GiB`。如果再乘上 batch 或 beam，线性放大非常快。

常见坑主要有五类：

1. **把理论感受野当成等价全注意力**
   $L \times W$ 只说明“有传播路径”，不说明远距离信息不会衰减。跨很多层传递后，信号可能弱化，任务效果未必等同全局注意力。

2. **只配滑窗 mask，不做 rolling cache**
   这会导致“算子是局部的，但缓存还在无限长大”。结果是计算省了一部分，显存却没真正省下来。

3. **把缓存槽位当作真实位置**
   `i mod W` 只是写入地址。位置编码仍要用绝对位置 $i$，否则长上下文下顺序信息会错乱。

4. **长 prompt 不做 chunked prefill**
   如果一次性把超长输入塞进去，内存峰值可能在预填充阶段先爆。正确做法是按窗口或更小块切片预填充。

5. **忽略 batch/beam 的线性放大**
   一条 beam 就是一份独立缓存。`beam=8` 基本就是 8 份 KV 成本，尤其在长上下文生成里非常敏感。

实际规避方法通常是：

- 控制窗口大小 `W`，不要盲目放大
- 长 prompt 采用 chunked prefill
- 严格限制 `beam size` 和服务端 batch
- 使用 GQA，减少 KV 头数
- 优先用支持滑窗的 FlashAttention 实现
- 做容量预算时把 KV、参数、临时激活分开算

---

## 替代方案与适用边界

如果任务天然就是 2K 到 4K 的短上下文，而且你非常依赖“当前 token 能直接访问所有历史”，那么传统全注意力仍然是更简单的选择。白话说，就是“把整本前文都摊在桌面上”，不用依赖跨层接力传信息。

如果任务是长文生成、长日志分析、长会话问答，滑窗 + rolling buffer 更有现实意义，因为它首先解决的是“能不能稳定部署”。

下面给出一个更完整的比较：

| 机制 | 直接可见历史 | KV 显存增长 | 远程信息路径 | 实施复杂度 | 典型适用场景 |
|---|---|---|---|---|---|
| 全注意力 | 全部 | 随长度增长 | 直接一步到达 | 低 | 短上下文、高保真依赖 |
| 滑窗 + Rolling Buffer | 最近 $W$ | 固定 $O(W)$ | 依赖层间传播 | 中 | 长上下文推理、在线服务 |
| Longformer/块稀疏注意力 | 局部 + 稀疏全局点 | 通常低于全注意力 | 依赖稀疏图设计 | 中到高 | 长文档编码 |
| Recurrence / Memory 层 | 当前块 + 外部记忆 | 可控 | 依赖显式记忆更新 | 高 | 超长序列、状态持续任务 |
| Paged KV / 分页缓存 | 历史可保留更长 | 更灵活但非固定最小 | 直接历史访问 | 高 | 高吞吐推理系统 |

要特别强调一个边界：Mistral 的方案适合“固定窗口、稳定服务、较低显存”的工程目标；如果你的任务需要频繁精确引用非常久远的具体 token，全注意力、检索增强、外部记忆，甚至显式 RAG，往往更稳。

所以不要把滑窗理解成“比全注意力更强”，更准确的说法是：它用局部可见性换取了长上下文部署能力，再利用深层堆叠把一部分远程依赖补回来。

---

## 参考资料

- Mistral AI, *Mistral 7B*, arXiv:2310.06825, 2023。用途：理论与架构定义，包含 SWA、Rolling Buffer、Chunked Prefill。链接：https://arxiv.org/abs/2310.06825
- Mistral AI, *Mistral 7B* PDF, 2023。用途：原始公式与图示说明，含 `i mod W` 缓存写入规则。链接：https://arxiv.org/pdf/2310.06825.pdf
- Hugging Face Transformers, *Mistral model documentation*, 持续更新。用途：实践层配置说明，包含 `flash-attn >= 2.3.0`、滑窗与 memory efficient cache 管理。链接：https://huggingface.co/docs/transformers/en/model_doc/mistral
- Dao-AILab, *FlashAttention* 官方仓库，持续更新。用途：工程实现参考，包含 sliding window local attention 与 KV cache inference 接口。链接：https://github.com/Dao-AILab/flash-attention
- Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023。用途：底层高效注意力实现背景。链接：https://tridao.me/publications/flash2/flash2.pdf
