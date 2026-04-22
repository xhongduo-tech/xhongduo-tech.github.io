## 核心结论

Paged Attention 的本质是把 KV cache 当作“分页管理”的对象，而不是为每个请求预留一整段连续显存。

一句总定义：`Paged Attention = 固定大小分页 + block table 映射 + 按需分配 + 前缀共享`。

KV cache 是 Transformer 解码时保存历史 token 的 Key 和 Value 的缓存。它的作用是避免每生成一个新 token 都重新计算全部历史 token 的 K/V。Paged Attention 不改变注意力公式，它改变的是 KV cache 在显存里的组织和寻址方式。

传统做法通常要求一个请求的 KV cache 放在连续内存中。请求长度不确定时，系统往往要提前预留较大的空间。短请求用不满，长请求又可能导致重新调度或分配失败。Paged Attention 借鉴操作系统虚拟内存，把一条序列切成固定大小的块，用 `block table` 记录“逻辑块到物理块”的映射。逻辑块是序列视角下的第几个块，物理块是真正在显存里的块编号。

新手版本可以这样理解：如果把 KV cache 想成书架，传统方法要求每本书都占一整排连续位置；Paged Attention 允许把书分散放在多个固定大小的小格子里，再用目录表去查位置。这样空位更少，也更容易插入新书。

| 维度 | 传统连续分配 | Paged Attention |
|---|---|---|
| KV 存储方式 | 每个请求一段连续显存 | 多个固定大小物理块 |
| 分配粒度 | 通常按请求或最大长度预留 | 按 block 按需分配 |
| 内存碎片 | 容易出现内部碎片 | 主要浪费最后一个块 |
| 前缀共享 | 实现复杂或代价较高 | 可共享相同前缀块 |
| 注意力公式 | 不变 | 不变 |
| 主要收益 | 实现简单 | 提升显存利用率和并发 batch 承载能力 |

核心结论是：Paged Attention 解决的不是“注意力怎么算”，而是“历史 K/V 怎么存、怎么找、怎么共享”。它的价值集中在推理服务，尤其是高并发、长上下文、beam search 和 parallel sampling 场景。

---

## 问题定义与边界

自回归大模型一次生成一个 token。生成第 $t$ 个 token 时，需要访问前面所有 token 的 K/V。若不缓存，每一步都要重新计算历史 token，成本很高。因此推理系统会保存 KV cache。

问题在于：请求长度通常不可预测。在线聊天里，有的用户只问一句短问题，有的用户带着长文档提问；有的请求生成 20 个 token 就结束，有的请求要生成 200 个 token。如果系统按最大长度给每个请求预留连续 KV cache，短请求会浪费大量显存。显存被浪费后，服务端能放进同一个 batch 的请求数减少，吞吐下降。

新手版本：两个请求，一个只生成 20 个 token，另一个要生成 200 个 token。如果都按最长请求预留整段空间，短请求会浪费很多显存；分页后，短请求只拿自己实际需要的块。

| 问题 | 传统做法的代价 | Paged Attention 的对应解法 |
|---|---|---|
| 请求长度差异大 | 短请求占用未使用空间 | 按需分配 block |
| 需要连续显存 | 容易受碎片影响 | 物理块不要求连续 |
| 解码阶段动态增长 | 预留过多或频繁搬迁 | 新 token 触发新块分配 |
| 多候选共享 prompt | 每个候选重复保存前缀 KV | 共享前缀块 |
| batch 调度压力高 | 显存利用率低，batch 变小 | 提高可承载并发数 |

边界必须说清楚：Paged Attention 主要解决推理服务中的 KV cache 管理问题。它不改变 Transformer 的注意力数学定义，也不是训练阶段完整显存管理方案。

适用场景：

| 类型 | 是否适合 | 原因 |
|---|---:|---|
| 在线推理服务 | 适合 | 请求长度动态变化 |
| 高并发 batch 推理 | 适合 | 显存利用率直接影响吞吐 |
| beam search | 适合 | 多个候选共享同一前缀 |
| parallel sampling | 适合 | 多路采样共享 prompt |
| 长上下文多请求并发 | 适合 | KV cache 占用显著 |
| 训练反向传播 | 不直接解决 | 训练还要保存激活和梯度 |
| 模型权重显存压力 | 不直接解决 | 权重存储不是 KV cache |
| 激活占用 | 不直接解决 | 激活管理属于训练内存优化问题 |

所以，Paged Attention 的准确定位是：推理阶段的 KV cache 分页管理方法。

---

## 核心机制与推导

设序列长度为 $L$，block size 为 $B$。block size 是每个 KV 块能容纳的 token 数。序列会被切成若干逻辑块：

$$
N = \lceil L / B \rceil
$$

对第 $t$ 个 token，它所在的逻辑块号和块内偏移是：

$$
b = \lfloor t / B \rfloor
$$

$$
o = t \bmod B
$$

然后通过 block table 找到物理块：

$$
p = block\_tables[seq][b]
$$

最终访问的是物理块 $p$ 中偏移 $o$ 的 K/V。

注意力计算本身仍然是：

$$
Attention(q, K, V) = \sum_j softmax(q \cdot k_j / \sqrt{d}) v_j
$$

变化只发生在 $k_j, v_j$ 的取址方式上。传统实现可以把 `k_cache[seq, token]` 看成连续数组访问；Paged Attention 中，系统必须先算出 token 对应的逻辑块，再通过 `block_tables` 找到物理块，最后从物理块内读取。

玩具例子：设 $B = 16$，某请求 $L = 37$，则：

$$
N = \lceil 37 / 16 \rceil = 3
$$

三个块总容量是 48 个 token 位置，浪费 11 个位置。浪费只发生在最后一块，不需要为整个最大长度预留空间。

假设 `seq0` 的 block table 是 `[7, 3, 9]`，表示逻辑块 0 放在物理块 7，逻辑块 1 放在物理块 3，逻辑块 2 放在物理块 9。

| token 位置 `t` | 逻辑块 `b=floor(t/B)` | 块内偏移 `o=t mod B` | 物理块 `p` | 实际 KV 地址 |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 7 | `k_cache[7][0]` |
| 15 | 0 | 15 | 7 | `k_cache[7][15]` |
| 16 | 1 | 0 | 3 | `k_cache[3][0]` |
| 31 | 1 | 15 | 3 | `k_cache[3][15]` |
| 32 | 2 | 0 | 9 | `k_cache[9][0]` |
| 36 | 2 | 4 | 9 | `k_cache[9][4]` |

前缀共享是 Paged Attention 的另一个关键收益。前缀是多个序列开头相同的一段 token。比如一个 prompt 产生 4 个 beam，前 32 个 token 完全相同。若 $B = 16$，前 32 个 token 正好对应两个 block。这两个 block 可以被 4 个候选共同引用。

新手版本：一个句子前 32 个 token 被 4 个候选共同使用，就像 4 个人共看同一本书的前两章；从第 33 个 token 开始分叉后，每个人再写自己的后续内容。

如果不共享，4 个候选各自保存 3 个块，共 12 个物理块。共享后，前 2 个块只存一份，4 个候选各自再分配 1 个分叉块，总共是：

$$
2 + 4 = 6
$$

物理块数减少一半。

copy-on-write 是写时复制，意思是多个序列共享同一个物理块时，如果其中一个序列要修改这个块，必须先复制一份，避免影响其他序列。触发条件是：

```text
refcnt[p] > 1 且需要写入新 token
```

其中 `refcnt[p]` 是物理块 `p` 的引用计数，表示有多少条序列正在引用它。

---

## 代码实现

代码层面通常由三部分组成：块分配器、`block table` 维护、Paged Attention kernel 的块式读取。

块分配器负责管理空闲物理块。`block table` 负责记录每条序列的逻辑块到物理块映射。Paged Attention kernel 是实际执行注意力计算的底层代码，它不能假设 KV cache 连续，必须通过 `block_tables` 间接定位物理块。

新手版本：原来的代码可能直接用 `k_cache[seq, token_id]` 访问；分页后要改成先找到 `block_id`，再用 `offset` 访问 `k_cache[phys_block, offset]`，否则读到的不是正确 token。

最小访问逻辑如下：

```python
block_id = token_id // B
offset = token_id % B
phys_block = block_tables[seq_id][block_id]
k = k_cache[phys_block][offset]
v = v_cache[phys_block][offset]
```

下面是一个可运行的 Python 玩具实现，用普通列表模拟 Paged Attention 的寻址过程：

```python
from math import ceil

B = 4

# seq0 的逻辑块 0、1、2 分别映射到物理块 2、0、3
block_tables = {
    "seq0": [2, 0, 3]
}

# 每个物理块有 B 个位置。这里用字符串模拟 K/V 内容。
k_cache = {
    0: ["k16", "k17", "k18", "k19"],
    2: ["k0", "k1", "k2", "k3"],
    3: ["k32", "k33", "k34", "k35"],
}

v_cache = {
    0: ["v16", "v17", "v18", "v19"],
    2: ["v0", "v1", "v2", "v3"],
    3: ["v32", "v33", "v34", "v35"],
}

def read_kv(seq_id, token_id):
    block_id = token_id // B
    offset = token_id % B
    phys_block = block_tables[seq_id][block_id]
    return k_cache[phys_block][offset], v_cache[phys_block][offset]

assert ceil(10 / B) == 3
assert read_kv("seq0", 0) == ("k0", "v0")
assert read_kv("seq0", 5) == ("k17", "v17")
assert read_kv("seq0", 9) == ("k33", "v33")
```

copy-on-write 的简化伪代码如下：

```python
def ensure_writable(seq_id, logical_block):
    old_block = block_tables[seq_id][logical_block]

    if refcnt[old_block] > 1:
        new_block = alloc_new_block()
        copy_block(old=old_block, new=new_block)
        refcnt[old_block] -= 1
        refcnt[new_block] = 1
        block_tables[seq_id][logical_block] = new_block

    return block_tables[seq_id][logical_block]
```

这里有一个容易写错的点：复制时必须保存旧块编号和新块编号。如果把变量都叫 `phys_block`，可能在分配新块后覆盖旧值，导致复制源和目标混乱。真实工程里的 kernel 和调度代码更复杂，但核心约束不变：任何读取历史 K/V 的地方，都要通过 `block_tables` 做一次映射。

真实工程例子：在线对话服务做 `parallel sampling`，同一个 prompt 同时采样 4 个回答。prompt 部分的 KV cache 只保存一份，4 个采样分支共享这些物理块。每个分支生成不同 token 后，再为新增 token 分配自己的块。这样可以在相同显存下放入更多请求，提高 batch size 和吞吐。

---

## 工程权衡与常见坑

Paged Attention 的第一个权衡是 block size。

块太大时，最后一块浪费增加，前缀共享粒度变粗。块太小时，`block table` 变大，元数据访问增加，kernel 需要处理更多间接寻址，开销会上升。

新手版本：如果把块大小设成 1，虽然几乎没有块内浪费，但目录表会很大，查表次数也暴增；如果把块大小设成 128，短序列会在最后一块浪费很多位置，前缀共享也不够细。

| block size | 优点 | 缺点 |
|---|---|---|
| 较小 | 浪费少，共享粒度细 | block table 更大，寻址开销更高 |
| 较大 | 元数据少，访问模式更规整 | 最后一块浪费多，共享粒度粗 |
| 中等 | 在浪费和开销间折中 | 需要结合模型、硬件和请求分布调参 |

连续内存和分页内存也不是单向优劣关系。

| 方案 | 优点 | 缺点 |
|---|---|---|
| 连续 KV cache | 实现简单，地址计算直接 | 长度变化大时浪费显存 |
| Paged Attention | 显存利用率高，支持共享和按需分配 | kernel 和调度实现更复杂 |
| 混合策略 | 可针对简单请求走快速路径 | 系统复杂度更高 |

常见坑主要有四类。

第一，仍然假设 KV 连续。自定义 CUDA kernel、推理 kernel 或调试代码如果直接按连续 token 下标访问，会读错数据。分页后，逻辑相邻的 token 不一定在物理上连续。

第二，忽略引用计数。共享前缀块时，如果没有维护 `refcnt`，某个分支写入新 token 可能破坏其他分支正在使用的 KV。

第三，copy-on-write 触发条件不完整。只有当 `refcnt[p] > 1` 且需要写入对应物理块时才需要复制。如果每次写都复制，性能会下降；如果该复制时不复制，结果会错误。

第四，把 Paged Attention 当成训练显存优化的完整答案。它主要处理 KV cache。训练中的激活、梯度、优化器状态、反向传播调度仍然需要其他方案。

检查清单：

| 检查项 | 应确认的问题 |
|---|---|
| `block_tables` 访问 | 所有 K/V 读取是否都经过逻辑块到物理块映射 |
| `refcnt` | 共享块是否维护引用计数 |
| copy-on-write | 写共享块前是否复制 |
| 最后一块浪费 | block size 是否适合请求长度分布 |
| 推理与训练区分 | 是否误把推理 KV 管理当作训练完整内存管理 |
| kernel 假设 | 底层实现是否仍假设 KV 物理连续 |

---

## 替代方案与适用边界

Paged Attention 不是唯一方案，也不是所有部署都必须引入。它适合的问题是高并发、动态长度、长上下文、多候选解码。它的主要收益是服务端吞吐和显存利用率。

如果只是低并发、短文本、离线批处理任务，固定预留连续 KV cache 可能已经足够。此时引入分页管理、引用计数和定制 kernel，复杂度可能超过收益。

新手版本：如果只是一个低并发、短文本的离线批处理任务，固定预留连续 KV cache 的复杂度可能不值得；但如果是在线聊天服务，Paged Attention 的收益通常明显。

| 方案 | 核心思路 | 适合场景 | 不适合场景 |
|---|---|---|---|
| 传统连续 KV cache | 每个请求使用连续缓存 | 序列长度稳定、低并发、实现优先 | 长度差异大、高并发 |
| Paged Attention | KV cache 分页并用 block table 映射 | 高并发、长上下文、beam search、服务端部署 | 极简部署、短序列低并发 |
| 静态最大长度预分配 | 按最大上下文长度一次性分配 | 请求形状固定、调度简单 | 显存紧张、长度变化大 |
| 请求分桶 | 按长度相近的请求组 batch | 离线批处理、长度分布可控 | 在线动态请求 |
| 前缀缓存 | 对常见 prompt 的 KV 复用 | 系统提示词固定、RAG 模板稳定 | prompt 高度随机 |

适用边界总结：

| 场景 | 是否优先考虑 Paged Attention |
|---|---:|
| 高并发在线推理 | 是 |
| 长上下文服务 | 是 |
| beam search | 是 |
| parallel sampling | 是 |
| 服务端多租户部署 | 是 |
| 低并发短序列 | 不优先 |
| 一次性离线任务 | 不优先 |
| 对 kernel 简化要求极高 | 不优先 |
| 训练反向传播内存优化 | 不能单独依赖 |

准确理解 Paged Attention，要把它放在推理系统工程里看：它不是新的注意力算法，而是让注意力在服务端更高效运行的 KV cache 内存管理方法。

---

## 参考资料

1. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
2. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/vllm)
3. [Paged Attention - vLLM](https://docs.vllm.ai/en/v0.18.0/design/paged_attention/)
4. [vllm-project/vllm](https://github.com/vllm-project/vllm)
5. [vllm.attention.ops.paged_attn](https://docs.vllm.ai/en/stable/api/vllm/attention/ops/paged_attn.html)
