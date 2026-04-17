## 核心结论

PagedAttention 的核心作用，不是改变注意力公式本身，而是改变 **KV Cache 的内存组织方式**。KV Cache 就是模型在推理时为每个 token 保存的 key/value 历史状态，后续生成会反复读取它。传统做法通常给每个请求预留一整段连续空间，长度按 `max_len` 估算；问题是请求真实长度不确定，短请求会留下大量空洞，形成显著的显存碎片和浪费。

PagedAttention 借鉴操作系统分页，把 KV Cache 切成固定大小的 block，例如 16 个 token 一块。请求只在真正生成到某个范围时才申请对应 block，再通过 block table 把“逻辑上的第几个块”映射到“物理上放在哪块显存”。这让显存从“按最大长度一次性预占”变成“按实际长度逐块分配”。

它带来三个直接结果：

1. 显存浪费显著下降。传统连续预留常见会有 20% 到 40% 的浪费，在长度分布很散、并发很高时更高。
2. 相同前缀可以共享 block。prefix sharing 的意思是多个请求如果前缀一样，就复用已经算好的 KV block，不必重复存一份。
3. 吞吐量通常更高。vLLM 以 PagedAttention 为基础，在许多在线推理场景下，相比直接使用 HuggingFace Transformers 服务，整体吞吐量常能高出 2 到 4 倍。

一个新手版的玩具理解是：传统方法像给每个用户先画一条 2048 格的停车位，不管他只停 20 格还是 2000 格，都先整条占住；PagedAttention 则把停车位切成一段段小格，车开到哪一段才占哪一段。

| 方案 | 分配方式 | 内存浪费 | 前缀复用 | 典型适用场景 |
| --- | --- | --- | --- | --- |
| 传统连续预留 | 按 `max_len` 一次性占连续空间 | 高 | 弱 | 低并发、短上下文 |
| PagedAttention | 按 block 按需分配 | 低 | 强 | 高并发、长上下文、共享 prompt |

---

## 问题定义与边界

问题先要说清：这里讨论的是 **LLM 推理阶段的 KV Cache 管理**，不是训练时的激活值管理，也不是权重量化。目标是同一张 GPU 上尽可能容纳更多并发请求，并让每个请求在生成过程中持续可读到自己的历史 KV。

为什么传统方法会浪费？因为服务端在收到请求时，通常无法确定它最终会生成多长。为了避免中途扩容和搬移，工程上很自然会为每个请求按上限预留一块连续空间。假设上限是 `max_len`，实际只用了 `L` 个 token，那么浪费率可近似写成：

$$
\text{waste\_rate} \approx 1 - \frac{L}{\text{max\_len}}
$$

如果 `max_len = 2048`，但某个请求只生成了 200 个 token，那么：

$$
1 - \frac{200}{2048} \approx 90.2\%
$$

这不是说所有实现都一定浪费 90%，而是说明“按上限预留”的结构性问题：真实长度越短，尾部空闲越大。

玩具例子可以直接看一个客服 bot。系统统一有一个短 prompt，例如“你是企业客服助手”，几十个用户同时发问。传统方式可能给每个会话都先占满 2048 token 的 KV 空间，哪怕很多用户只问一句话、只生成几十个 token。结果是 GPU 很快被“尚未使用但已经预占”的空间塞满。

真实工程例子更典型：企业内部 RAG 服务里，大量请求共享同一个系统提示词、同一个格式化模板，只有检索段落和用户问题不同。此时如果每个请求都把公共前缀重新写入一份 KV Cache，不仅显存重复占用，还会重复做 prefill 计算。PagedAttention 把这一类“共享但不连续”的历史缓存管理问题变成块级复用问题。

它的边界也要明确：

| 边界 | 是否属于 PagedAttention 直接解决的问题 | 说明 |
| --- | --- | --- |
| KV Cache 碎片 | 是 | 核心目标 |
| 前缀共享 | 是 | 通过 block 复用实现 |
| 权重显存过大 | 否 | 这属于量化、张量并行等问题 |
| 单次 attention 算法复杂度 | 否 | 公式本身仍是 attention |
| 网络通信瓶颈 | 否 | 需要服务框架和调度优化 |

所以，PagedAttention 解决的是“**历史状态怎么放**”，不是“**模型参数怎么压**”。

---

## 核心机制与推导

PagedAttention 的关键抽象有两个：

1. **逻辑 block**：请求按 token 顺序切出的第 `i` 块。
2. **物理 block**：实际放在 GPU 显存中的一块缓存。

block 是固定大小的，例如每块 `B=16` 个 token。若当前请求长度为 `L`，则所需 block 数是：

$$
N = \left\lceil \frac{L}{B} \right\rceil
$$

再用一个 block table 记录映射关系：

$$
PT[i] = \text{physical\_block\_id}
$$

这里的 `PT` 可以理解为页表。页表就是“逻辑位置到物理位置的映射表”，白话解释是：用户以为数据按顺序排着放，实际上底层可以分散在很多地方，只要表能查到就行。

### 玩具例子

设 `B = 16`，请求当前已有 `L = 20` 个 token。

那么：

$$
N = \left\lceil \frac{20}{16} \right\rceil = 2
$$

也就是说，只需要 2 个 block：
- block 0 放前 16 个 token
- block 1 放后 4 个 token

传统连续预留若按 2048 token 算，相当于预留了 `2048 / 16 = 128` 个 block；现在只真正占了 2 个。即使第二块只用了 4 个 token，尾部浪费也只局限在这一个 block 内，而不会蔓延到整条序列的剩余长度。

### 逻辑到物理的映射

假设某请求的 block table 如下：

| 逻辑 block 索引 | 物理 block ID |
| --- | --- |
| 0 | 18 |
| 1 | 42 |
| 2 | 7 |

这表示请求逻辑上的第 0、1、2 段 token，并不是连续放在显存中的 18、19、20 号块，而是分别放在 18、42、7。对 attention kernel 来说，只要按逻辑顺序查表取出对应的 K/V 即可。

### 为什么能减少碎片

传统做法要求一个请求的 KV 空间尽量连续，否则后续读取难管理。连续要求一强，就很容易出现“大块空位不足，小块空位很多”的问题，这就是外部碎片。PagedAttention 把大块需求拆成等长小块，分配器只需要找到若干个空闲 block，不需要找到一大片连续区域，因此显著降低碎片。

### prefix sharing 的机制

如果两个请求前缀完全相同，例如都以同一个 system prompt 开头，那么前缀对应的逻辑 block 可以直接指向同一组物理 block。多个请求共享这些只读块，直到某个请求在后续生成中写入分叉内容，才触发 copy-on-write。

copy-on-write 的意思是“写时复制”，白话解释是：先共享一份，谁要改、谁再单独复制。这样可以避免在分叉之前就提前复制所有前缀缓存。

### attention 读取方式

attention 端不再默认“当前序列的 K/V 在一片连续地址上”，而是做两步：

1. 通过 token 所在的逻辑 block 计算 block index 和 block 内偏移。
2. 通过 `PT[block_index]` 找到物理 block，再读取对应位置的 K/V。

因此，PagedAttention 本质上是在 kernel 侧引入了一层块级间接寻址。代价是多了一次查表；收益是显存利用率和复用能力明显提高。

---

## 代码实现

下面先给一个最小可运行的 Python 玩具实现。它不包含 GPU kernel，只模拟 block 分配、页表映射和 prefix sharing 的核心行为。

```python
from math import ceil

class BlockAllocator:
    def __init__(self):
        self.next_id = 0
        self.refcnt = {}

    def alloc(self):
        block_id = self.next_id
        self.next_id += 1
        self.refcnt[block_id] = 1
        return block_id

    def incref(self, block_id):
        self.refcnt[block_id] += 1

    def decref(self, block_id):
        self.refcnt[block_id] -= 1
        if self.refcnt[block_id] == 0:
            del self.refcnt[block_id]

class PagedKVSequence:
    def __init__(self, block_size, allocator):
        self.block_size = block_size
        self.allocator = allocator
        self.block_table = []   # logical_block_idx -> physical_block_id
        self.length = 0

    def append_tokens(self, n_tokens):
        target_blocks = ceil((self.length + n_tokens) / self.block_size)
        while len(self.block_table) < target_blocks:
            self.block_table.append(self.allocator.alloc())
        self.length += n_tokens

    def share_prefix_from(self, other, n_tokens):
        n_blocks = ceil(n_tokens / self.block_size)
        self.block_table = other.block_table[:n_blocks].copy()
        for block_id in self.block_table:
            self.allocator.incref(block_id)
        self.length = n_tokens

    def cow_last_block_if_shared(self):
        if not self.block_table:
            return
        last = self.block_table[-1]
        if self.allocator.refcnt[last] > 1:
            self.allocator.decref(last)
            self.block_table[-1] = self.allocator.alloc()

allocator = BlockAllocator()
seq_a = PagedKVSequence(block_size=16, allocator=allocator)
seq_a.append_tokens(20)

assert seq_a.length == 20
assert len(seq_a.block_table) == 2

seq_b = PagedKVSequence(block_size=16, allocator=allocator)
seq_b.share_prefix_from(seq_a, n_tokens=16)

assert seq_b.length == 16
assert seq_b.block_table[0] == seq_a.block_table[0]
assert allocator.refcnt[seq_a.block_table[0]] == 2

seq_b.cow_last_block_if_shared()
assert seq_b.block_table[0] != seq_a.block_table[0]
assert allocator.refcnt[seq_a.block_table[0]] == 1
```

这段代码体现了三件事：

1. 长度 20、block size 16 时，只分到 2 个 block。
2. 新请求可以共享旧请求的前缀 block。
3. 当共享块要被改写时，才执行 copy-on-write。

下面给一个更贴近推理流程的伪代码：

```python
def prefill(request, tokens):
    for token in tokens:
        logical_idx = request.length // BLOCK_SIZE
        if logical_idx == len(request.block_table):
            request.block_table.append(alloc_block())
        write_kv(request, token)
        request.length += 1

def decode_one_token(request, new_token):
    logical_idx = request.length // BLOCK_SIZE
    if logical_idx == len(request.block_table):
        request.block_table.append(alloc_block())

    if is_shared_block(request.block_table[logical_idx]):
        request.block_table[logical_idx] = copy_on_write(request.block_table[logical_idx])

    write_kv(request, new_token)
    request.length += 1
    return attention_lookup(request.block_table, request.length)
```

如果把它映射到真实工程，顺序通常是：

| 阶段 | 主要动作 | 是否可能复用已有 block |
| --- | --- | --- |
| prefill | 处理输入 prompt，生成初始 KV | 是 |
| block table fill | 建立逻辑块到物理块映射 | 是 |
| decode | 每步生成新 token，必要时新分配 block | 是 |
| branch/diverge | 分支续写时触发 copy-on-write | 是 |

### 真实工程例子

假设一个在线文档助手，所有请求都有相同系统提示词：

- “你是一个严格遵守公司知识库的问答助手”
- 后接固定的输出格式模板
- 再接用户问题

那么这段公共前缀在一天里可能被成千上万次重复使用。传统实现里，每个请求都需要重新占一份前缀 KV。PagedAttention 则可以把前缀拆成若干 block，后来的请求直接引用这些 block table 项。这样 prefill 成本下降，显存中的重复数据也减少。

---

## 工程权衡与常见坑

PagedAttention 不是“切成块就一定更好”。真正落地时，最核心的参数是 block size。

如果 block 太小，例如 4 或 8 token 一块，尾部浪费会更少，但元数据会变多。元数据就是为管理数据而额外存的表项、引用计数、索引结构。block 越多，block table 越长，查表越频繁，调度和 kernel 组织更复杂。

这个开销可以近似理解为：

$$
\text{metadata\_cost} \approx \text{block\_count} \times \text{meta\_per\_block}
$$

反过来，如果 block 太大，例如 64 或 128 token 一块，block 数变少了，但每个请求最后一块的尾部空闲会变大，等于又向“粗粒度预分配”退回去。

一个常见折中是 16 到 32 token 一块。

| Block Size | 元数据开销 | 尾部浪费 | 查表次数 | 常见评价 |
| --- | --- | --- | --- | --- |
| 16 | 较高 | 最低 | 较多 | 常见平衡点 |
| 32 | 中等 | 低 | 中等 | 另一常见平衡点 |
| 64 | 低 | 偏高 | 较少 | 更简单，但浪费上升 |

### 常见坑 1：只看单请求，不看并发

单个请求测试时，PagedAttention 的查表和管理开销可能让人误以为“没快多少”。但它真正的收益主要来自 **并发提升** 和 **缓存复用**，而不是单条请求的理论最短延迟。评估时要看整体吞吐、同卡并发数、平均显存占用。

### 常见坑 2：忽略 copy-on-write

如果实现共享前缀，却在分支生成时直接原地写共享 block，就会污染其他请求的历史 KV，结果是输出错误且难排查。只要有共享，就必须有明确的引用计数和 copy-on-write 逻辑。

### 常见坑 3：把 block table 查找写成高开销热点

PagedAttention 不是让 CPU 每个 token 手动查很多次表，而是要把这层映射尽量融合进 GPU 访问路径，或者至少让批处理后的查找成本被吞吐量摊薄。否则，显存省下来了，算子调度却成了瓶颈。

### 常见坑 4：前缀共享条件判断过宽

只有“token 序列完全一致”的前缀才能安全共享。看起来语义相同但 token 化结果不同，或者模板中某些变量位置不同，都不能直接复用。工程上应基于 token 序列或稳定哈希判断，而不是字符串表面相似度。

一个直观类比是切披萨：切得太碎，盒子和标签的成本高；切得太粗，最后几块总剩很多吃不完。block size 就是在这两种损失之间找折中点。

---

## 替代方案与适用边界

PagedAttention 很强，但不是所有场景都必须上。

### 方案一：静态 KV 预分配

这是最直接的方案。实现简单，访问连续，对 kernel 友好。若场景是低并发、短序列、长度分布稳定，例如固定模板分类器或小型内部工具，那么静态分配完全可能更省工程成本。

### 方案二：Prefix Cache，但不做分页

有些系统会单独做前缀缓存，例如把系统 prompt 的 KV 预先算好，下次直接复用。这能减少重复 prefill，但如果底层还是连续大块管理，那么遇到大量不同长度请求时，显存碎片问题仍然存在。也就是说，prefix cache 解决的是“算没算过”，PagedAttention 解决的是“内存怎么摆”。

### 方案三：更激进的内存层级管理

还可以把部分 KV 放到 CPU、NVMe 或远端缓存，再按需换入 GPU。这类方案能扩大可服务上下文长度，但会引入更明显的数据搬运延迟，适用于超长上下文、离线批处理或延迟要求不极端的场景。

| 方案 | 并发适应性 | 长上下文适应性 | 前缀复用 | 实现复杂度 |
| --- | --- | --- | --- | --- |
| 静态 KV 预分配 | 低到中 | 低 | 弱 | 低 |
| Prefix Cache | 中 | 中 | 中到强 | 中 |
| PagedAttention | 高 | 高 | 强 | 中到高 |
| 分层 KV 缓存 | 中 | 很高 | 中 | 高 |

### 适用边界总结

如果你的服务特征是下面这些，PagedAttention 通常值得上：

- 并发高
- 请求长度差异大
- 长上下文常见
- 大量请求共享系统 prompt 或模板前缀

如果你的服务特征是下面这些，静态方案可能更合适：

- 并发低
- 输出长度稳定
- 上下文较短
- 团队不想承担复杂缓存管理

所以它不是“唯一正确方案”，而是“高并发 LLM Serving 下非常有效的默认方案”。

---

## 参考资料

| 资源名称 | 说明 | 主要贡献 | 链接 |
| --- | --- | --- | --- |
| Kwon 等，《Efficient Memory Management for Large Language Model Serving with PagedAttention》 | PagedAttention 论文 | 给出分页式 KV Cache 管理的核心设计与实验结果 | https://arxiv.org/abs/2309.06180 |
| vLLM 官方项目 | PagedAttention 的工程实现来源 | 展示如何在 LLM serving 中落地块级 KV 管理 | https://github.com/vllm-project/vllm |
| vLLM 官方文档 | 使用与架构说明 | 补充 block 管理、prefix caching、部署方式 | https://docs.vllm.ai/ |
| Hugging Face Papers 页面 | 论文索引与摘要入口 | 便于快速查看论文背景与引用信息 | https://huggingface.co/papers/2309.06180 |
| A. Krisanov 对 vLLM/PagedAttention 的解析 | 面向工程读者的讲解 | 用更直观的方式解释 block/page table 思路 | https://akrisanov.com/vllm/ |
| Introl 的 vLLM 生产部署文章 | 生产案例总结 | 说明共享前缀、吞吐提升和部署经验 | https://introl.com/blog/vllm-production-deployment-inference-serving-architecture |
