## 核心结论

PagedAttention 的核心不是“把 attention 算得更快”，而是“把 KV Cache 管得更省”。KV Cache 是模型为后续 token 复用历史上下文而保存的 key/value 张量，可以理解成“解码阶段不断增长的上下文缓存”。传统做法通常希望这段缓存放在一整块连续显存里，结果是请求长度一旦不可预测，就会出现两类浪费：一类是提前预留太多，一类是释放后留下难以复用的碎片。

PagedAttention 的做法是把 KV Cache 按固定大小的 block 分页。block 可以理解成“固定容量的小页”，例如每页容纳 16 或 32 个 token 的 KV。每个请求不再持有一整段连续物理内存，而是维护一张块表，把“逻辑上连续的第 0 页、第 1 页、第 2 页”映射到“物理上离散的第 17、42、5 号页”。这和操作系统页表管理虚拟内存是同一类思想。

它带来的直接收益有三个：

| 维度 | 传统 KV Cache | PagedAttention |
| --- | --- | --- |
| 内存预留 | 常按最大长度预留连续大块 | 按需申请小块 |
| 碎片管理 | 容易出现外部碎片 | 主要只剩最后一页尾部浪费 |
| 前缀共享 | 共享困难，常重复存储 | 可按块共享，适合 Beam Search 和并行采样 |

给零基础读者的直观版本是：每生成新 token，不是“再扩容一整条缓存”，而是“页满了就从内存池领一页新纸”。请求结束后，把没人在用的页归还池子。逻辑上仍然是一条连续序列，物理上却可以分散存放。

---

## 问题定义与边界

问题本身不是 attention 数学公式有错，而是工程实现里 KV Cache 的生命周期太动态。一个请求可能只生成 20 个 token，也可能生成 2000 个 token；多个请求还会并发到达、并发结束。连续分配最怕这种“长度未知、频繁增长、频繁回收”的负载。

如果仍然用连续内存，常见策略是为每个请求预留一段够大的缓存。这样实现简单，但会带来两个问题：

1. 请求没用满时，预留空间直接浪费。
2. 请求释放顺序不一致时，可用空间会被切成很多小洞，形成外部碎片。外部碎片就是“总空闲容量够，但没有足够大的连续块可分配”。

新手版本的玩具例子可以这样理解：三个人来租房，甲只住一周，乙住半年，丙住三天。传统做法像是给每个人都预留一整套长期公寓，结果很多房间空着；PagedAttention 更像按页租用标准间，需要时再加一间，不需要就退。

PagedAttention 解决的问题边界也要讲清楚：

| 它解决的 | 它不直接解决的 |
| --- | --- |
| KV Cache 动态增长带来的碎片 | 单次矩阵乘法的理论复杂度 |
| 多请求并发下的显存利用率 | 模型参数本身过大带来的显存压力 |
| Beam/采样共享前缀时的重复缓存 | attention kernel 之外的全部系统瓶颈 |

如果 block 容量为 $B$，序列长度为 $L$，那么需要的块数是：

$$
\text{blocks}=\left\lceil \frac{L}{B} \right\rceil
$$

对应的块内利用率可以写成：

$$
\text{内存利用率}=\frac{L}{\lceil L/B \rceil \times B}
$$

这说明浪费通常集中在最后一个块。只要 $B$ 不要大得离谱，尾页浪费就会被压到很低。

---

## 核心机制与推导

PagedAttention 的关键对象有三个：block pool、block table、reference count。

block pool 是“块池”，可以理解成“预先准备好的一堆固定规格空页”。  
block table 是“块表”，可以理解成“逻辑页号到物理页号的映射表”。  
reference count 是“引用计数”，可以理解成“这一页当前被多少序列共同使用”。

### 1. 逻辑连续，物理离散

设 block 大小为 $B=16$。如果一个序列有 40 个 token，那么它需要：

$$
\left\lceil \frac{40}{16} \right\rceil = 3
$$

也就是 3 个块。逻辑上这 3 个块是 `[0,1,2]`，但物理上可能对应 `[17,42,5]`。attention kernel 在计算时，不再假设 KV 在一片连续地址里，而是先查块表，再去对应物理块里取数据。

这就是“逻辑连续序列映射到物理离散页”。

### 2. 为什么浪费主要只在尾页

40 个 token，块大小 16，那么总容量是 $3 \times 16 = 48$，只空出最后 8 个 token 的位置。浪费比例是：

$$
\frac{48-40}{48}=\frac{8}{48}\approx 16.7\%
$$

这是单条 40-token 序列的例子。真实服务里，序列通常更长，而且块大小常取 16 或 32，浪费只发生在最后一块，因此整体平均浪费会显著低于传统连续预留。vLLM 公开材料中给出的经验结论是，尾页浪费通常能压到很低，接近最优利用率。

### 3. 前缀共享与 Copy-on-Write

真实服务里，一个 prompt 可能派生出多个候选输出，比如 Beam Search 或并行采样。此时前缀 token 完全相同，它们的 KV 也完全相同。最直接的方法是每条分支都复制一份 KV，但这会浪费大量显存。

PagedAttention 用共享块解决这个问题：多个序列的块表都指向同一组前缀物理块，同时把这些块的引用计数加一。只有当某个分支要在“共享块上继续写入”时，才触发 Copy-on-Write。Copy-on-Write 的白话解释是“先共享，直到必须改动时才复制”。

下面是一个简化的块表示例：

| logical id | physical id | ref count | 说明 |
| --- | --- | --- | --- |
| 0 | 17 | 3 | 三条分支共享的 prompt 第 1 页 |
| 1 | 42 | 3 | 三条分支共享的 prompt 第 2 页 |
| 2 | 5 | 1 | 当前分支自己的尾页 |

### 4. 真实工程例子

vLLM 和采用其思路的推理系统，在高并发、多采样场景中收益明显。原因不是单个请求延迟必然暴降，而是显存利用率提高后，可同时容纳更多请求与更大 batch。对于在线推理服务，这通常直接转化为吞吐率提升，因为系统常常先卡在显存而不是算力。

---

## 代码实现

下面的 Python 代码不是 GPU 版 attention kernel，而是一个可运行的“内存池 + 块表 + 引用计数 + Copy-on-Write”玩具实现。它用很小的对象把 PagedAttention 的核心管理逻辑表达出来。

```python
from dataclasses import dataclass, field
from math import ceil


BLOCK_SIZE = 4


@dataclass
class PhysicalBlock:
    block_id: int
    tokens: list = field(default_factory=list)
    ref_count: int = 0


class BlockPool:
    def __init__(self, capacity: int):
        self.blocks = {i: PhysicalBlock(i) for i in range(capacity)}
        self.free_ids = list(range(capacity))

    def allocate_block(self) -> int:
        assert self.free_ids, "out of blocks"
        block_id = self.free_ids.pop()
        block = self.blocks[block_id]
        block.tokens = []
        block.ref_count = 1
        return block_id

    def incref(self, block_id: int) -> None:
        self.blocks[block_id].ref_count += 1

    def decref(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count -= 1
        assert block.ref_count >= 0
        if block.ref_count == 0:
            block.tokens = []
            self.free_ids.append(block_id)


class SequenceState:
    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.block_table = []  # logical_block_id -> physical_block_id
        self.length = 0

    def _ensure_tail_block_writable(self):
        if not self.block_table or self.length % BLOCK_SIZE == 0:
            self.block_table.append(self.pool.allocate_block())
            return

        tail_pid = self.block_table[-1]
        tail_block = self.pool.blocks[tail_pid]
        if tail_block.ref_count > 1:
            # Copy-on-Write: 共享尾页被写入时才复制
            new_pid = self.pool.allocate_block()
            self.pool.blocks[new_pid].tokens = list(tail_block.tokens)
            self.pool.decref(tail_pid)
            self.block_table[-1] = new_pid

    def append_token(self, token: int):
        self._ensure_tail_block_writable()
        tail_pid = self.block_table[-1]
        self.pool.blocks[tail_pid].tokens.append(token)
        self.length += 1

    def fork(self):
        child = SequenceState(self.pool)
        child.block_table = list(self.block_table)
        child.length = self.length
        for pid in child.block_table:
            self.pool.incref(pid)
        return child

    def release(self):
        for pid in self.block_table:
            self.pool.decref(pid)
        self.block_table = []
        self.length = 0

    def materialize(self):
        tokens = []
        for pid in self.block_table:
            tokens.extend(self.pool.blocks[pid].tokens)
        return tokens[:self.length]


# 玩具例子：prefix 共享 + Copy-on-Write
pool = BlockPool(capacity=8)
base = SequenceState(pool)

for t in [10, 11, 12, 13, 14]:
    base.append_token(t)

child = base.fork()
assert base.materialize() == [10, 11, 12, 13, 14]
assert child.materialize() == [10, 11, 12, 13, 14]

child.append_token(99)
assert base.materialize() == [10, 11, 12, 13, 14]
assert child.materialize() == [10, 11, 12, 13, 14, 99]

used_blocks = ceil(child.length / BLOCK_SIZE)
utilization = child.length / (used_blocks * BLOCK_SIZE)
assert abs(utilization - 0.75) < 1e-9  # 6 token / 8 capacity

base.release()
child.release()
assert len(pool.free_ids) == 8
```

这段代码对应的模块职责可以概括为：

| 模块 | 责任 | 关键接口 |
| --- | --- | --- |
| `block_pool` | 管理固定大小物理块的分配与回收 | `allocate_block` / `incref` / `decref` |
| `block_table` | 维护逻辑块到物理块的映射 | `logical_block_id -> physical_block_id` |
| `kernel` | 根据块表遍历并读取 KV | 查表、聚合、计算 attention |

如果把它映射回真实系统，attention kernel 的核心伪逻辑大致是：

1. 读取当前序列的 block table。
2. 按逻辑块顺序找到对应 physical block。
3. 从每个 physical block 里取出 K/V。
4. 对有效 token 范围执行注意力计算。
5. 对最后一块只处理真实长度，不处理未填充尾部。

这里必须强调：PagedAttention 要求 attention kernel 支持分页读取。也就是说，核心难点不是“写一个池子”，而是“让 kernel 能高效按块表访存”。

---

## 工程权衡与常见坑

PagedAttention 节省显存，但实现复杂度明显上升。传统连续 KV Cache 的地址计算非常直接，而分页后你需要同时维护块池、块表、引用计数、并发安全和分页版 kernel。

常见问题如下：

| 问题 | 现象 | 原因 | 避坑策略 |
| --- | --- | --- | --- |
| block leak | 显存越跑越少 | 引用计数没减到 0 | 统一生命周期管理，回收路径做断言 |
| lock contention | 高并发下分配变慢 | 池管理锁冲突 | 分层池、无锁队列或线程局部缓存 |
| logical/physical 错配 | 结果错误或越界 | 块表更新时索引错位 | 块表不可变快照 + 严格单测 |
| 尾页有效长度错误 | attention 读到脏数据 | 没区分容量和真实 token 数 | 记录 sequence length，kernel 按有效长度裁剪 |
| 错误共享 | 分支互相污染 | 共享块被直接写入 | 共享页写入前强制 Copy-on-Write |

新手可以把 `release` 忘记减引用理解成“租期结束忘记交钥匙”。房子表面上没人住，系统却认为还被占着，最终内存池会耗尽。

另一个工程权衡是 block 大小。block 太小，碎片更少，但块表更长，查表与调度开销更高；block 太大，块表更短，但尾页浪费变多。实际工程里，16 或 32 token 是常见折中，因为它同时兼顾了碎片率和 kernel 访存效率。

---

## 替代方案与适用边界

PagedAttention 不是任何场景都必须上。它最适合的条件是：并发高、序列长、长度不可预测、还存在前缀共享需求。

| 场景 | 更适合的方案 | 优势 | 限制 |
| --- | --- | --- | --- |
| 高并发长序列在线服务 | PagedAttention | 显存利用率高，支持共享前缀 | 需要分页 kernel 和内存管理器 |
| 依赖现有 kernel 生态，想少改代码 | vAttention | 保留连续虚拟地址，减少 kernel 改造 | 依赖底层虚拟内存与系统支持 |
| 请求短且长度稳定 | 传统预分配 KV Cache | 实现简单，调试成本低 | 长序列和并发波动时浪费明显 |

vAttention 可以视为另一条路线：目标同样是动态管理 KV Cache，但它尽量保留连续虚拟地址，把“按需映射物理页”的工作交给更底层的内存机制，因此不必像 PagedAttention 那样显式重写分页式 attention kernel。代价是它依赖系统层能力和实现环境。

所以判断是否需要 PagedAttention，可以先问三个问题：

1. 我是否经常被 KV Cache 显存占用卡住？
2. 我的请求长度是否高度不稳定？
3. 我是否大量使用 Beam Search、并行采样或共享 prompt 的场景？

如果这三个问题大多回答“是”，PagedAttention 很可能有价值。反过来，如果你的服务每次只生成 32 个 token、并发也低，那么连续预分配更简单，工程总成本反而更低。

---

## 参考资料

| 来源 | 类型 | 覆盖主题 |
| --- | --- | --- |
| [vLLM Blog: vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/vllm) | Blog | PagedAttention 动机、碎片问题、前缀共享、吞吐收益 |
| [vLLM Paper: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | 论文 | PagedAttention 设计、vLLM 系统、实验结果 |
| [vLLM Docs: Paged Attention](https://docs.vllm.ai/en/stable/design/paged_attention.html) | 文档 | 分页式 attention kernel 的实现思路 |
| [Hugging Face TGI Docs: PagedAttention](https://huggingface.co/docs/text-generation-inference/en/conceptual/paged_attention) | 文档 | 面向工程实践的概念解释 |
| [Microsoft Research: vAttention](https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/) | 论文/项目页 | PagedAttention 的替代路线与工程复杂度比较 |
| [vAttention arXiv](https://arxiv.org/abs/2405.04437) | 论文 | 连续虚拟地址 + 按需物理映射的方案 |
