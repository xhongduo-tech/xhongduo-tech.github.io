## 核心结论

GPU 显存碎片化的关键，不是“显存还剩多少”，而是“剩下的显存能不能以合适的形状被再次拿出来用”。这里的“形状”可以先理解成连续大块、可复用缓存块、或满足当前 stream 依赖的块。总量够但形状不对，分配一样会失败。

最常见的误解是把显存问题只看成 OOM。实际工程里，碎片化更早出现的症状通常是并发能力下降、尾延迟抖动放大、服务运行越久越不稳定。原因很直接：分配器需要越来越频繁地找块、切块、回收块，甚至向驱动申请新块，而这几步都可能把快路径变成慢路径。

玩具例子先看一个。把 16 MiB 空闲显存切成 `4 + 4 + 4 + 4 MiB` 四段，此时总空闲量 `T = 16 MiB`，最大连续空闲块 `L_max = 4 MiB`。如果来了一个 `8 MiB` 请求，它仍然会失败，因为一次分配通常要求拿到一整块满足条件的空间，而不是从四处拼接小块。

真实工程例子更典型。在线大模型推理服务里，KV cache 是“保存历史 token 键值对的显存区域”，会随着上下文增长而扩容。某台卡上 `nvidia-smi` 显示还剩 16 GiB，但 allocator 里最大的连续空闲块只有 1 GiB，于是一次 4 GiB 的 KV cache 扩容仍然直接 OOM。这时问题不是“总量不够”，而是“分配器已经把可复用空间切碎了”。

外部碎片化常用一个直观指标表示：

$$
F_{ext} = 1 - \frac{L_{max}}{T}
$$

如果 `T` 很大，但 `L_max` 很小，说明空闲块被切得很碎，外部碎片率就高。

内部碎片化是另一件事。它指“程序真正只想要 6.1 MiB，但分配器按桶或按粒度给了 8 MiB”。这里的“按桶”可以白话理解成“只发固定尺寸盒子，不发散装”。常见定义是：

$$
F_{int} = 1 - \frac{U}{A}
$$

其中 `U` 是真实有效载荷，`A` 是 allocator 实际分配量。`A > U` 的差值，就是内部浪费。

下面这张表说明“总空闲够”和“能否分配大块”不是一回事：

| 总空闲显存 `T` | 最大连续块 `L_max` | 请求大小 | 是否能分配 | 现象 |
|---|---:|---:|---|---|
| 16 MiB | 16 MiB | 8 MiB | 能 | 无碎片影响 |
| 16 MiB | 8 MiB | 8 MiB | 能 | 勉强满足 |
| 16 MiB | 4 MiB | 8 MiB | 不能 | 总量够，但连续块不够 |
| 20 GiB | 1 GiB | 4 GiB | 不能 | 在线服务常见 OOM 形态 |

引子可以先写成这样：

```text
allocate(size):
    block = find_free_block_with_size_at_least(size)
    if block exists:
        maybe_split(block, size)
        return block
    else:
        request_new_memory_from_driver()
```

决定成败的不是 `free_total >= size`，而是 `find_free_block_with_size_at_least(size)` 是否成立。

---

## 问题定义与边界

先把概念切开，不然后面很容易混淆。

外部碎片化，指空闲空间总量还在，但被切成很多小块，导致大请求拿不到连续空间。内部碎片化，指分配器因为对齐、桶化、页粒度等原因，多给了用不到的空间。缓存占用则是第三类现象：块已经不再被张量使用，但分配器把它留在缓存池中，暂时不还给驱动，准备后续复用。

统一记号如下：

- `T`：空闲显存总量
- `L_max`：最大连续空闲块
- `R`：allocator 保留的总显存
- `U`：业务真正使用的显存
- `W = R - U`：被保留但当前未真正使用的显存

这三个问题对用户看到的表象不同：

| 类型 | 本质 | 常见指标 | 典型症状 |
|---|---|---|---|
| 外部碎片 | 空闲块被切碎 | `L_max / T` 低 | 总量够但大块申请失败 |
| 内部碎片 | 向上取整或按桶浪费 | `U / A` 低 | 明明任务不大，实际占用偏高 |
| 缓存占用 | 已释放块未回驱动 | `R - U` 高 | `reserved` 高、`allocated` 低 |

这里必须明确边界。本文讨论的是 GPU 显存管理，重点是运行时 allocator 如何组织显存块，不是 CPU 堆碎片，也不是单次 `cudaMalloc` 的 API 调用开销分析。单次 `cudaMalloc` 慢，更多是同步和驱动路径问题；碎片化则是长期运行后空间形状恶化的问题。

新手常看 `nvidia-smi`，但它只能看到驱动层视角中的已占用总量，不能直接告诉你：

- allocator 里哪些块是活跃分配
- 哪些块只是缓存保留
- 最大连续空闲块到底多大
- 某个 stream 上的空闲块是否能立刻复用

真实工程里，PyTorch 这类框架会同时暴露 `allocated` 和 `reserved`。`allocated` 可以白话理解成“现在真被张量拿着的显存”，`reserved` 则是“框架已经圈下来、自己管着的总显存”。于是你经常看到 `reserved` 很高、`allocated` 不高。这并不一定是泄漏，更可能是缓存池在工作。

玩具例子可以这样想。你手里有 1000 元，但都是十张 100 元纸币；商家偏要你一次交一张 1000 元整钞，这当然不现实。这个类比不完全等价于物理地址连续性，但足够说明“总量够”和“可立即满足一次分配”是两件事。

真实工程里更常见的是：张量已经释放了，但对应 block 还待在 caching allocator 的池子里，于是 `reserved` 高、`allocated` 低。你如果只看 `nvidia-smi`，会误以为“显存已经被占满”；你如果只看业务逻辑，会误以为“张量都释放了，为什么还不能分配”。这就是边界没分清。

---

## 核心机制与推导

显存分配器本质上做四件事：找块、切块、回收、合并。

“找块”是按大小类或空闲链表找到一个足够大的块；“大小类”可以白话理解成“把相近尺寸的请求放进同一组，减少搜索成本”。“切块”是把大块拆成请求需要的大小，剩余部分重新挂回空闲池。“回收”是在对象释放后把块放回可复用结构。“合并”则是在相邻空闲块都回收后，把它们拼成更大块，降低外部碎片。

一个最小流程如下：

```text
allocate(size):
    rounded = round_up(size)
    block = find_fit(rounded)
    if not block:
        block = grow_pool_from_driver(rounded)
    if block.size - rounded >= split_threshold:
        used, rest = split(block, rounded)
        insert_free(rest)
        return used
    mark_used(block)
    return block

free(block):
    mark_free(block)
    buddy_or_neighbors = find_adjacent_free_blocks(block)
    merged = merge_if_possible(block, buddy_or_neighbors)
    insert_free(merged)
```

为什么“总量够但请求失败”？推导很直接。设空闲总量为 `T`，最大连续空闲块为 `L_max`。如果一次请求大小是 `S`，那么分配成功的必要条件通常是：

$$
L_{max} \ge S
$$

注意是必要条件，不是 `T >= S`。当 `T >= S` 但 `L_max < S` 时，说明空闲总量足够，但块形状不满足。代回外部碎片率：

$$
F_{ext} = 1 - \frac{L_{max}}{T}
$$

当 `T = 16`、`L_max = 4` 时，`F_ext = 75\%`。这时 8 MiB 请求失败不是偶发，而是空间结构已经不支持这类请求。

为什么内部浪费不能忽略？设请求真实需要 `U = 6.1 MiB`，allocator 按 2 MiB 粒度上取整，实际发放 `A = 8 MiB`。那么：

$$
F_{int} = 1 - \frac{6.1}{8} \approx 23.75\%
$$

单次看浪费不大，但在高并发服务里，如果有成百上千个这类对象同时存在，内部碎片会把峰值占用整体抬高，从而提前触发 OOM 或迫使 batch 变小。

下面这张表把“请求大小”和“空闲块分布”的关系列出来：

| 请求大小 | 当前空闲块分布 | `T` | `L_max` | 是否可分配 | `F_ext` |
|---|---|---:|---:|---|---:|
| 4 MiB | `8 + 4 + 4` | 16 | 8 | 能 | 50% |
| 8 MiB | `8 + 4 + 4` | 16 | 8 | 能 | 50% |
| 12 MiB | `8 + 4 + 4` | 16 | 8 | 不能 | 50% |
| 8 MiB | `4 + 4 + 4 + 4` | 16 | 4 | 不能 | 75% |

玩具例子可以把一张 A4 纸想成一个大块。你把它剪成很多便签纸后，记短笔记更灵活，但再想要一张完整海报时就做不到了。分配器里的 `split` 就像“剪开”，`merge` 就像“重新拼回原尺寸”。问题在于，现实分配器通常只能合并物理或逻辑上相邻、状态又匹配的块，不可能随便把两个 4 MiB 散块拼成一个可用 8 MiB 大块。

真实工程例子更接近在线推理。某个 8 MiB 请求到来时，系统里有很多 4 MiB 空闲块，总空闲 16 MiB，看起来绰绰有余。但 allocator 不支持跨块拼接，也没有合适的 remap 机制，于是 8 MiB 请求依然失败。这里失败并不是 bug，而是分配模型本来就要求“单块满足”。

控制碎片化，核心就看三件事：

| 设计点 | 作用 | 代价 |
|---|---|---|
| 切分粒度 | 提高小请求命中率 | 过细会放大外部碎片 |
| 合并策略 | 恢复大块，降低 `F_ext` | 合并检查有元数据和同步成本 |
| 缓存回收阈值 | 提升复用率，减少系统调用 | 阈值太高会让 `reserved` 持续膨胀 |

所以 allocator 设计不是“越省越好”或“越快越好”，而是在吞吐、延迟、碎片率之间找工作负载能承受的平衡点。

---

## 代码实现

下面先给一个教学版最小 allocator。它不操作真实 GPU，只模拟“块表、分配、释放、合并、快照”，目的是让碎片化变成可观察对象。你可以直接运行它。

```python
from dataclasses import dataclass

@dataclass
class Block:
    start: int
    size: int
    free: bool = True

class SimpleAllocator:
    def __init__(self, total_size: int):
        self.blocks = [Block(0, total_size, True)]

    def allocate(self, size: int):
        for i, block in enumerate(self.blocks):
            if block.free and block.size >= size:
                ptr = block.start
                remain = block.size - size
                self.blocks[i] = Block(block.start, size, False)
                if remain > 0:
                    self.blocks.insert(i + 1, Block(block.start + size, remain, True))
                return ptr
        raise MemoryError(f"cannot allocate {size}")

    def free(self, ptr: int):
        for i, block in enumerate(self.blocks):
            if block.start == ptr and not block.free:
                self.blocks[i].free = True
                self._merge()
                return
        raise ValueError(f"invalid ptr {ptr}")

    def _merge(self):
        merged = []
        for block in self.blocks:
            if merged and merged[-1].free and block.free and merged[-1].start + merged[-1].size == block.start:
                merged[-1].size += block.size
            else:
                merged.append(Block(block.start, block.size, block.free))
        self.blocks = merged

    def snapshot(self):
        free_blocks = [b.size for b in self.blocks if b.free]
        total_free = sum(free_blocks)
        largest_free = max(free_blocks) if free_blocks else 0
        ext_frag = 0.0 if total_free == 0 else 1 - largest_free / total_free
        return {
            "blocks": [(b.start, b.size, b.free) for b in self.blocks],
            "total_free": total_free,
            "largest_free": largest_free,
            "external_fragmentation": round(ext_frag, 4),
        }

# 玩具例子：16 单位空间
alloc = SimpleAllocator(16)
p1 = alloc.allocate(4)
p2 = alloc.allocate(4)
p3 = alloc.allocate(4)
p4 = alloc.allocate(4)

alloc.free(p1)
alloc.free(p3)
snap = alloc.snapshot()

assert snap["total_free"] == 8
assert snap["largest_free"] == 4
assert snap["external_fragmentation"] == 0.5

try:
    alloc.allocate(8)
    assert False, "should fail because largest free block is only 4"
except MemoryError:
    pass

alloc.free(p2)
snap2 = alloc.snapshot()
assert snap2["largest_free"] == 12

p5 = alloc.allocate(8)
assert p5 == 0
```

这段代码做了三件关键事：

- `allocate(size)`：顺序找第一个够大的空闲块，必要时 `split`
- `free(ptr)`：标记为空闲
- `_merge()`：把相邻空闲块合并

如果你把 `_merge()` 去掉，再跑同样的测试，会发现碎片化更快恶化。这说明“回收不等于恢复可分配能力”，只有合并成功，大块请求才真的有机会回来。

下面用表格列一下教学实现里的函数职责：

| 函数 | 作用 | 对碎片化的影响 |
|---|---|---|
| `allocate` | 找到空闲块并切分 | 切分过多会升高外部碎片 |
| `free` | 释放使用中的块 | 只释放不合并，效果有限 |
| `_merge` | 合并相邻空闲块 | 直接降低外部碎片 |
| `snapshot` | 观察 `T`、`L_max`、`F_ext` | 帮助定位问题，而不是解决问题 |

真实工程里当然不会只靠顺序扫描。常见做法是用 size class、空闲链表、红黑树、页粒度池化等结构，把查找复杂度和并发开销控制住。教学模型只负责解释机制，不负责逼近工业性能。

落到真实框架时，可以把观察分成三层：

- 模拟实现：理解 `split / merge / cache`
- 真实框架接口：例如 PyTorch 的 `memory_allocated`、`memory_reserved`、`memory_snapshot`
- 生产建议：根据请求分布调 size class、池阈值、释放策略

真实工程例子里，排查在线推理抖动时，通常先看某段时间内 `reserved / allocated` 是否持续走高，再结合 snapshot 看大块是否越来越少。如果 `reserved` 高企且 `L_max / T` 下降，就说明不是单纯“模型变大”，而是缓存池内部结构在恶化。

---

## 工程权衡与常见坑

分配器没有免费午餐。提高复用率，往往要保留更多块；保留更多块，又会让 `reserved` 变大。频繁把缓存还给驱动，能降低表面占用，但会引入更多系统调用、同步和重新建池成本。工程上真正关心的不是“某一刻能不能分配成功”，而是“服务连续跑几天后，峰值、尾延迟和 OOM 概率是否仍然可预测”。

一个常见误区是误读 `nvidia-smi`。它看到的是大盘子，不是菜盘格局。某块显存已经被框架保留但当前未使用，`nvidia-smi` 一样会算作占用；而 allocator 内部某类大块已经耗尽，`nvidia-smi` 并不会直接告诉你。

另一个误区是过度依赖 `empty_cache()`。它的作用可以白话理解成“把缓存里暂时不用的盒子退回仓库”，但它不会压缩仍在使用中的对象，更不会让一个本来就超出容量上限的模型突然装得下。所以它适合释放缓存压力，不适合作为常规热路径手段。

下面是常见坑对照表：

| 常见坑 | 错误理解 | 实际问题 | 更合理的处理 |
|---|---|---|---|
| 误读 `nvidia-smi` | 看到剩余显存就以为还能分配 | 看不到 `L_max` 和缓存池形状 | 同时看 `reserved/allocated/snapshot` |
| 过度依赖 `empty_cache()` | 以为能解决所有 OOM | 只能释放未使用缓存 | 只在阶段切换或主动回收时使用 |
| 热路径频繁 `cudaMalloc/free` | 认为最直接就最好 | 系统调用和同步成本高 | 优先池化分配器 |
| 多 stream 处理不当 | 释放后立刻全局可复用 | 依赖未完成时不能安全复用 | 正确记录 stream 依赖 |
| split 过细 | 小块命中率更高就是好事 | 大块越来越难恢复 | 设定 split 阈值和 size class |

真实工程里，多 stream 是一个经常被低估的问题。stream 可以白话理解成“GPU 上的命令队列”。如果一个块在 stream A 上刚释放，但 stream B 想立刻复用它，分配器必须确认 A 上所有相关工作都真的完成了，否则会发生数据竞争。于是很多 allocator 会延迟该块进入“全局可复用”状态。结果就是：看起来空闲很多，实际上当前 stream 不一定能用。

示意代码可以写成：

```python
class StreamAwareBlock:
    def __init__(self, size):
        self.size = size
        self.pending_streams = set()

    def record_stream(self, stream_id):
        self.pending_streams.add(stream_id)

    def can_reuse(self):
        return len(self.pending_streams) == 0

block = StreamAwareBlock(1024)
block.record_stream("stream_1")
assert not block.can_reuse()
block.pending_streams.remove("stream_1")
assert block.can_reuse()
```

“为什么服务跑几天后才爆炸，而不是一开始就爆炸？”因为碎片化通常依赖历史路径。服务刚启动时，池子很新，块分布还干净；随着不同长度请求、不同 batch、不同并发模式不断交错，缓存池逐渐形成很多“只适合旧请求、不适合新请求”的残留块。于是平均指标可能没明显变坏，但尾部请求越来越容易碰到坏形状，最后在一个峰值时刻突然 OOM。这也是为什么长稳压测比短压测更能暴露 allocator 问题。

---

## 替代方案与适用边界

没有一种分配器适合所有工作负载。训练、离线推理、在线推理、超高并发服务，对显存分配模式的要求完全不同。核心不是“谁先进”，而是“谁更匹配当前分配波动”。

下面给出常见方案对比：

| 方案 | 优点 | 缺点 | 适用场景 | 不适用场景 |
|---|---|---|---|---|
| 传统 `cudaMalloc/free` | 接口直接，语义清晰 | 开销高，易引入同步 | 小实验、低频分配 | 高频动态分配服务 |
| caching allocator | 复用快，减少驱动调用 | `reserved` 易膨胀，可能积累碎片 | 常规训练与推理 | 长时间高波动在线服务 |
| `cudaMallocAsync` + pool | 更适合池化和异步流顺序管理 | 需要理解 pool 行为与阈值 | 在线推理、多 stream | 对运行时版本受限的环境 |
| 内存池 + size class | 可针对业务调优 | 实现复杂，调参成本高 | 固定请求分布服务 | 需求变化快的小项目 |
| VMM / remap / defrag | 可在更高层面缓解碎片 | 复杂度高，依赖底层能力 | 极端长稳高并发场景 | 普通业务场景 |

玩具例子可以把它理解成不同收纳工具。旅行箱适合整体搬运，收纳袋适合细分物品，纸箱适合临时存放。不存在统一最优，只有负载匹配与否。

真实工程里可以粗分为三类：

- 小规模实验：默认 caching allocator 往往够用，重点是少做无意义的动态 shape 波动。
- 长时间在线服务：更值得关注池化分配器、`cudaMallocAsync`、释放阈值、KV cache 预留策略。
- 极端波动负载：如果请求长度、batch、并发都剧烈变化，必须显式关注 size class、release threshold、甚至 defrag/remap 能力。

边界也要说清。什么时候碎片化是主因？当你看到：

- 总空闲不少，但大块申请失败
- `reserved` 长期远高于 `allocated`
- 长稳运行后问题明显恶化
- 调小 batch 一点点就恢复，且 snapshot 显示大块稀缺

什么时候根本问题不是碎片化？当：

- 模型参数本身就装不下
- 峰值 workspace 天生过大
- batch 上限明显超卡容量
- 激活、KV cache、临时张量总峰值本就超过物理上限

这两类问题的处理方式不同。前者应该优化 allocator 和请求形状，后者应该减模型、减 batch、做量化或重算，或者换更大卡。把两者混为一谈，会导致调参方向完全错误。

配置层面的伪代码可以写成：

```text
if workload_is_short_lived and low_variance:
    use_default_caching_allocator()
elif workload_is_online and long_running:
    use_memory_pool()
    tune_release_threshold()
    stabilize_request_shapes()
elif workload_has_extreme_variance:
    add_size_classes()
    monitor_snapshot()
    evaluate_async_allocator_or_defrag()
```

结论仍然很朴素：碎片化不是“高级优化议题”，而是服务化之后的基础稳定性问题。只要你的工作负载会长时间动态申请和释放显存，allocator 设计就已经进入主路径。

---

## 参考资料

| 来源 | 类型 | 适合阅读的章节 | 备注 |
|---|---|---|---|
| NVIDIA 官方博客 | 官方说明 | 核心机制、替代方案 | 适合理解 stream-ordered allocator 背景 |
| CUDA Programming Guide | 官方文档 | 问题边界、工程实现 | 适合查 memory pool 与 release threshold |
| NVIDIA 论文 | 研究论文 | 核心机制、工程权衡 | 适合理解高吞吐 allocator 设计目标 |
| PyTorch 文档 | 官方文档 | 问题边界、工程排查 | 适合看 `reserved/allocated/empty_cache` |
| PyTorch 源码 | 源码实现 | 代码实现、工程细节 | 适合看 split、size class、缓存策略 |

1. [Using the NVIDIA CUDA Stream-Ordered Memory Allocator, Part 1](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/)
2. [CUDA C Programming Guide - Stream-Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator)
3. [Throughput-oriented GPU memory allocation](https://research.nvidia.com/publication/2019-02_throughput-oriented-gpu-memory-allocation)
4. [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)
5. [PyTorch CUDACachingAllocator.cpp](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)
