## 核心结论

PagedAttention 的本质不是“更快的注意力公式”，而是“更合理的 KV Cache 内存管理方式”。KV Cache 就是模型把历史 token 的 key/value 中间结果暂存起来，避免每生成一个新 token 都把整段上下文重新算一遍。传统推理服务通常按“最大上下文长度”给每个请求预留一整段连续显存，这会把大量短请求变成空洞，常见浪费可达到 60% 到 80%。PagedAttention 把 KV Cache 切成固定大小的 block，再用 block table 把“逻辑顺序”映射到“物理位置”，于是显存不再要求连续，只按真实生成进度增长。

这带来三个直接后果。第一，外部碎片基本消失，因为空闲 block 可以被任意请求复用。第二，内部碎片被压缩到“最后一个 block 没填满”这一种情况；当 block 大小 $B=16$ 时，平均浪费通常可控制在 4% 以下。第三，显存利用率提升会直接转成更大的 batch size，而更大的 batch size 通常就是更高吞吐，所以 vLLM 论文里能看到 2 到 4 倍吞吐提升。

| 方案 | 分配方式 | 典型碎片来源 | 显存利用率 | 扩展性 |
|---|---|---|---|---|
| 传统 KV Cache 预分配 | 每请求一段连续大块 | 预留过多、释放后形成洞 | 低 | 长上下文和混合长度请求下很差 |
| PagedAttention | 固定大小 block 按需分配 | 仅最后一块可能未满 | 高 | 适合连续 batching 和长上下文 |

玩具例子可以直接看成两种“修路”方式。传统方式是先把整条可能走到的路一次性铺满；PagedAttention 是走到哪铺到哪，而且每段路砖都统一规格，可以拆下来给别的车队继续用。这个比喻只用于直觉，严格对应到工程里就是“连续大块预留”与“固定 block 按需分配”的区别。

---

## 问题定义与边界

问题定义很具体：在 LLM 推理中，KV Cache 会随着序列增长而动态变大，但服务端往往不知道每个请求最终会生成多长，所以保守做法是按最大长度提前分配。这样设计简单，但代价是显存浪费和可并发请求数下降。

设某请求最大允许长度为 $L_{\max}$，实际只生成了 $L_{\text{actual}}$。如果按最大长度连续预分配，则该请求的内部碎片率近似为：

$$
\text{fragmentation} \approx \frac{L_{\max} - L_{\text{actual}}}{L_{\max}}
$$

例如，请求 A 实际只需要 128 token，但系统先给它预留了 256 token 的 KV 空间，那么：

$$
\frac{256-128}{256} = 50\%
$$

这还只是单请求的内部碎片。更难处理的是外部碎片：多个请求陆续结束后，会在显存里留下很多不连续的小洞。即使总空闲显存足够，也可能因为没有“足够大的一整段连续空间”而无法接纳新请求。

这个问题的边界也要说清楚：

| 边界 | 说明 |
|---|---|
| 解决对象 | KV Cache 的分配、回收、复用 |
| 不直接解决 | 模型权重太大、算力不足、网络延迟 |
| 主要收益场景 | 请求长度分布差异大、长短请求混跑、连续 batching |
| 收益有限场景 | 小模型、很短上下文、显存本来就富余 |

真实工程例子是 24GB 显存的单卡部署。假设模型权重已经占掉大半显存，剩余空间要同时容纳 activation 和 KV Cache。此时如果所有请求都按 4K 或 8K 上下文预留，短请求会把本可用于并发的显存提前锁死，服务端表现就是 batch 上不去、队列变长、吞吐下降，最后看起来像“GPU 没打满但请求还是慢”。

---

## 核心机制与推导

PagedAttention 借的是操作系统分页思想，但不需要把它神秘化。逻辑上，每个请求的 KV Cache 仍然是按 token 顺序增长；物理上，它被拆成固定大小为 $B$ 的 block，散落在显存不同位置。block table 记录“第几个逻辑块，落在第几个物理块”。

如果某请求当前已有 $T$ 个 token，则它需要的逻辑块数为：

$$
\text{logical\_blocks} = \left\lceil \frac{T}{B} \right\rceil
$$

当 $B=16$、请求长度为 128 token 时，需要 $8$ 个 block；长度为 256 token 时，需要 $16$ 个 block。于是：

- Request A: 128 token，需要 8 个 block
- Request B: 256 token，需要 16 个 block

如果传统方案把两者都按 256 token 预留，则总占用是 512 token 容量，其中 A 浪费 128。  
如果采用 PagedAttention，则只分配 $8+16=24$ 个 block，对应 384 token 容量，浪费只可能出现在最后一个 block 未填满时。

内部碎片上界也很好算。每个请求最多只会有一个尾块未满，因此单请求最多浪费 $B-1$ 个 token，对应上界：

$$
\text{tail\_waste\_ratio} \le \frac{B-1}{T}
$$

若从“单个 block 的最坏填充率”看，也常写成近似：

$$
\text{waste\_per\_request} \approx \frac{B-1}{B}
$$

但这只是“最后一个块内部”的最坏局部视角，不是整体平均碎片率。实际整体平均碎片率远低于这个值，因为前面的块都是满的。也正因为如此，vLLM 和相关技术解读常把默认 block size 设为 16，经验上能把总体浪费压到 4% 以下。

| block 大小 B | 128 token 请求所需块数 | 2048 token 请求所需块数 | metadata 条目数趋势 | 尾块浪费上界 |
|---|---:|---:|---|---|
| 8 | 16 | 256 | 高 | 更低 |
| 16 | 8 | 128 | 中 | 低 |
| 32 | 4 | 64 | 低 | 更高 |
| 64 | 2 | 32 | 更低 | 更高 |

这里的 metadata，白话说就是“描述数据的数据”，例如块表项、填充 token 数、引用计数。B 越小，块越多，metadata 越大；B 越大，metadata 越省，但尾块浪费会增加。这就是 PagedAttention 最核心的参数权衡。

---

## 代码实现

实现层面要做两件事：一是内存管理器负责 block 的申请、释放和复用；二是 attention kernel 负责根据 block table 去正确读取分散存储的 K/V。attention kernel 就是 GPU 上真正执行注意力计算的底层程序。

先看一个可运行的玩具实现，它不计算真实 attention，只模拟分页分配与释放逻辑：

```python
from dataclasses import dataclass, field
from math import ceil

BLOCK_SIZE = 16

@dataclass
class Block:
    block_id: int
    used_tokens: int = 0
    owner: str | None = None

@dataclass
class RequestState:
    req_id: str
    token_count: int = 0
    logical_to_physical: list[int] = field(default_factory=list)

class PagedKVCache:
    def __init__(self, num_blocks: int):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_list = list(range(num_blocks))
        self.requests: dict[str, RequestState] = {}

    def _alloc_block(self, req_id: str) -> int:
        assert self.free_list, "OOM: no free blocks"
        bid = self.free_list.pop(0)
        self.blocks[bid].used_tokens = 0
        self.blocks[bid].owner = req_id
        return bid

    def append_tokens(self, req_id: str, n_tokens: int) -> None:
        state = self.requests.setdefault(req_id, RequestState(req_id))
        for _ in range(n_tokens):
            need_new_block = (
                not state.logical_to_physical or
                self.blocks[state.logical_to_physical[-1]].used_tokens == BLOCK_SIZE
            )
            if need_new_block:
                state.logical_to_physical.append(self._alloc_block(req_id))
            last_bid = state.logical_to_physical[-1]
            self.blocks[last_bid].used_tokens += 1
            state.token_count += 1

    def release(self, req_id: str) -> None:
        state = self.requests.pop(req_id)
        for bid in state.logical_to_physical:
            self.blocks[bid].used_tokens = 0
            self.blocks[bid].owner = None
            self.free_list.append(bid)

    def allocated_capacity(self, req_id: str) -> int:
        return len(self.requests[req_id].logical_to_physical) * BLOCK_SIZE

cache = PagedKVCache(num_blocks=64)
cache.append_tokens("A", 128)
cache.append_tokens("B", 256)

assert cache.requests["A"].token_count == 128
assert cache.requests["B"].token_count == 256
assert cache.allocated_capacity("A") == 128
assert cache.allocated_capacity("B") == 256
assert len(cache.requests["A"].logical_to_physical) == ceil(128 / BLOCK_SIZE)
assert len(cache.requests["B"].logical_to_physical) == ceil(256 / BLOCK_SIZE)

cache.release("A")
before = len(cache.free_list)
cache.append_tokens("C", 32)
after = len(cache.free_list)

assert cache.requests["C"].token_count == 32
assert len(cache.requests["C"].logical_to_physical) == 2
assert after == before - 2
```

上面这段代码对应的核心逻辑可以概括成：

```python
if block_table[req_id][logical_idx] is EMPTY:
    physical_block = allocate_block()
    block_table[req_id][logical_idx] = physical_block

write_kv_to_block(physical_block, offset_in_block)
update_token_count(req_id)
```

真正落到 GPU kernel 时，流程不是“先把 KV 拼回连续内存”，而是计算某个 query token 需要访问哪些逻辑块，再通过 block table 找到对应物理块，直接 gather 这些散落的 K/V 数据并完成 attention。这样做避免了额外拷贝，但要求 kernel 天生理解分页布局。

| 字段 | 含义 | 作用 |
|---|---|---|
| request_id | 请求标识 | 定位该请求的块表 |
| logical_block_idx | 逻辑块编号 | 表示序列中的第几个块 |
| physical_block_id | 物理块编号 | 指向显存中的真实位置 |
| filled_tokens | 当前块已写入 token 数 | 确定尾块写入偏移 |
| ref_count | 引用计数 | 支持前缀共享和 copy-on-write |

真实工程里，前缀缓存也依赖这套结构。多个请求如果共享同一段 prompt，可以让它们指向同一批物理 block；只有当生成分叉时，才通过 copy-on-write 分配新块。这样不仅解决碎片，还减少重复 KV 存储。

---

## 工程权衡与常见坑

PagedAttention 不是“白送收益”。最大成本在于它改了 KV 的物理布局，因此 attention kernel、调度器和内存管理器必须协同设计。

第一类坑是 kernel 复杂度。普通连续内存 attention kernel 只需要线性扫描 K/V；分页后必须做地址映射、块级遍历、访存合并和边界处理。Microsoft 的 vAttention 工作就明确批评了这一点：PagedAttention 虽然解决了动态分配问题，但把分页逻辑压进 attention kernel，会带来软件复杂度、可移植性问题和重复实现成本。

第二类坑是 block 大小选择。它不是越小越好。

| B 大小 | 碎片率 | metadata 开销 | kernel 访存负担 | 适合场景 |
|---|---|---|---|---|
| 小，如 8/16 | 低 | 高 | 查表更多 | 显存紧张、请求长度分散 |
| 中，如 16/32 | 较低 | 可接受 | 平衡 | 通用部署 |
| 大，如 64/128 | 更高 | 低 | 可能更友好，也可能因局部性变差而退化 | 请求长度更稳定 |

第三类坑是框架和硬件绑定。某些 paged kernel 对 CUDA 版本、GPU 架构、head size、block size 有明确限制。换句话说，PagedAttention 的收益建立在“你有一套能吃分页布局的高质量 kernel”之上。如果团队频繁切换 FlashAttention、FlashInfer、Triton 实现或跨 NVIDIA/AMD 平台迁移，这部分维护成本不能忽略。

第四类坑是你以为自己在优化“算子速度”，实际问题却在“容量管理”。PagedAttention 的第一收益通常不是单步 attention 更快，而是显存利用率更高，使连续 batching 更稳定、可并发请求更多、排队更短。指标上更常见的改善是整体吞吐，而不是单 token kernel latency 一定下降。

---

## 替代方案与适用边界

PagedAttention 很强，但不是唯一方案。替代思路的关键差异在于：到底是让应用层显式管理分页，还是把分页交给 CUDA 驱动和统一虚拟内存机制。

| 方案 | 核心思想 | 依赖 | 易实现度 | 适合边界 |
|---|---|---|---|---|
| PagedAttention | 应用层把 KV 切成 block，kernel 按块读取 | 需要分页感知 kernel | 中到高 | 高吞吐服务、需要极致显存利用 |
| vAttention / demand paging | 保持虚拟地址连续，物理页按需映射 | CUDA 虚拟内存与驱动支持 | 中 | 不想重写 attention kernel |
| 统一内存 / UVM | GPU/CPU 间自动迁移页 | 硬件与驱动支持 | 高 | 开发效率优先，性能可预测性较弱 |
| KV Offloading | 热数据在 GPU，冷数据放 CPU/磁盘 | 额外传输链路 | 中 | 超长上下文、显存绝对不足 |

真实工程例子可以看 IBM Foundation Model Stack 的工作：它把 FlexAttention 和 PagedAttention 结合，在 NVIDIA L4 24GB 上处理 128 到 2048 token 的全局 KV cache，延迟随长度大致线性增长，而不是因为缓存策略失衡出现明显恶化。这类方案适合“显存不大但要稳住长上下文”的部署环境。

而如果你的团队不愿意维护定制 paged kernel，或者必须兼容多种现成 attention backend，那么 vAttention 这类方案更现实。它保留“KV 在虚拟地址空间上连续”的编程模型，把按需物理分配交给 CUDA demand paging。代价是你把一部分控制权交给底层内存系统，调优空间和可解释性通常不如显式分页。

最终判断标准很简单：

- 如果瓶颈是 KV Cache 碎片导致 batch 起不来，PagedAttention 通常是第一选择。
- 如果瓶颈是跨平台维护成本，显式分页未必划算。
- 如果上下文并不长，或者显存很宽裕，PagedAttention 的收益可能不值得其实现复杂度。

---

## 参考资料

| 资源名称 | URL | 内容简介 |
|---|---|---|
| Efficient Memory Management for Large Language Model Serving with PagedAttention | https://arxiv.org/abs/2309.06180 | PagedAttention 原始论文，给出问题定义、设计与吞吐结果 |
| vLLM 论文解读与工程拆解 | https://akrisanov.com/vllm/ | 对 KV Cache 碎片、block table、连续 batching 的直观解释 |
| vLLM Paged Attention 设计文档 | https://docs.vllm.ai/en/v0.11.2/design/paged_attention/ | 介绍 paged attention kernel 的内存布局与执行方式 |
| Paged Attention and vLLM | https://training.continuumlabs.ai/inference/why-is-inference-important/paged-attention-and-vllm | 面向工程实践的概述，解释逻辑块、物理块和块表 |
| vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention | https://arxiv.org/abs/2405.04437 | 替代方案，讨论如何不改 paged kernel 也实现动态 KV 管理 |
| Microsoft Research: vAttention 页面 | https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/ | 总结 vAttention 的动机、优势与性能对比 |
| Paged Attention Meets FlexAttention: Unlocking Long-Context Efficiency in Deployed Inference | https://arxiv.org/abs/2506.07311 | IBM FMS 的工程实践，讨论 L4 24GB 上长上下文推理表现 |
