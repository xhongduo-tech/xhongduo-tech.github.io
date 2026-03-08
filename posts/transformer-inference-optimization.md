## 核心结论

Transformer 推理优化里，真正决定吞吐、成本和可承载并发的，通常不是单次矩阵乘法本身，而是 **KV Cache 管理与调度**。KV Cache 的作用很直接：前面 token 已经计算出的 Key/Value 不再重复计算，后续解码直接复用。如果这部分管理粗糙，系统常见瓶颈就不是“算不动”，而是显存占坑、碎片、长短请求互相拖累、以及调度空转。

当前主流方案可以分成四类：

| 优化 | 解决什么问题 | 常见收益 |
|---|---|---|
| PagedAttention | KV Cache 预留过多、显存碎片严重 | 让 KV 近似按需分配，常见可显著提升显存利用率与整体吞吐 |
| Continuous Batching | 静态批处理要“整批一起等” | 每个解码步都补新请求，显著提高 GPU 利用率 |
| Speculative Decoding | 一次前向通常只产出 1 个 token | 用草稿模型先猜多步，再由目标模型验收，低批量下常见 2-3x 单请求加速 |
| Dynamic Batching | 流量波动导致批太小或排队太久 | 在吞吐和延迟之间做在线折中 |

可以把它们理解成三层优化：

| 层级 | 核心手段 | 主要目标 |
|---|---|---|
| 内存层 | PagedAttention / paged KV | 降低显存浪费，提高可并发数 |
| 调度层 | Continuous Batching + Dynamic Batching | 减少空槽与等待，提高设备利用率 |
| 算法层 | Speculative Decoding | 减少目标模型真正执行的串行步数 |

生产系统通常会叠加使用，而不是只选一个。先把 KV Cache 放对，再把批次调对，最后才谈进一步压缩单步串行成本。

一个最小玩具例子就能说明差异。假设有一个 512-token 聊天请求，页大小设为 64 token。传统方案可能按“最大上下文 2048”一次性预留整块 KV；分页方案只需分配

$$
8=\left\lceil \frac{512}{64}\right\rceil
$$

页。请求结束后，这 8 页立即回收，空出来的页可以立刻给下一个 32-token 请求使用，而不必等整批结束。这就是它能提高并发数的根本原因。

---

## 问题定义与边界

先把问题边界说清楚。本文讨论的是 **自回归解码阶段** 的推理优化，也就是模型已经完成预填充（prefill，指把提示词整段跑完）之后，按 token 逐步生成输出的阶段。这个阶段有三个工程特征：

1. 每一步都依赖前一步输出，天然串行。
2. 单步 FLOPs 不一定极大，但会频繁读写历史 KV，因此常受 **显存带宽、缓存布局和调度策略** 约束。
3. 请求长度分布极不均匀，导致“同样是 1 次服务调用，资源生命周期差异很大”。

为了先建立数量级直觉，可以把单个 token 的 KV 开销近似写成：

$$
M_{\text{kv/token}} \approx 2 \times n_{\text{layers}} \times n_{\text{kv-heads}} \times d_{\text{head}} \times \text{bytes}
$$

其中：

| 符号 | 含义 |
|---|---|
| $2$ | 同时存 Key 和 Value |
| $n_{\text{layers}}$ | Transformer 层数 |
| $n_{\text{kv-heads}}$ | KV 头数，GQA/MQA 会直接影响它 |
| $d_{\text{head}}$ | 每个头的维度 |
| `bytes` | 每个元素所占字节数，取决于 FP16/BF16/FP8 等精度 |

若序列长度为 $S$，则单请求 KV 开销近似为：

$$
M_{\text{req}}(S)=S \times M_{\text{kv/token}}
$$

这条公式的意义是：**KV Cache 对序列长度线性增长**。所以长上下文和高并发不是两个独立问题，它们会共同把显存推向上限。

为了让新手更容易建立直觉，可以看一个简单数值例子。假设模型配置满足：

- $n_{\text{layers}}=32$
- $n_{\text{kv-heads}}=8$
- $d_{\text{head}}=128$
- 每个元素用 FP16，即 `bytes = 2`

则

$$
M_{\text{kv/token}} \approx 2 \times 32 \times 8 \times 128 \times 2 = 131072 \text{ bytes} \approx 128 \text{ KB}
$$

这意味着：

| 序列长度 | 单请求 KV 开销 |
|---|---|
| 512 token | 约 64 MB |
| 2048 token | 约 256 MB |
| 8192 token | 约 1 GB |

这还只是单个请求的 KV，不含模型权重、激活、临时 workspace，也没算多并发。于是旧方案的问题就显现出来了：很多实现虽然“会缓存”，但 **缓存分配方式很笨**。常见做法是按最大上下文长度预留连续显存。比如同时来了 3 个请求，长度分别是 50、300、700 token，如果系统按 2048 token 为每条请求预留 KV，那么大量显存只是“占着位置”，并没有真实数据。

长度不确定时，这种过度预留和连续分配会带来两类浪费：

| 浪费类型 | 含义 | 结果 |
|---|---|---|
| 过度预留 | 先按最大长度占坑 | 平均大量空闲空间不可复用 |
| 外部碎片 | 空闲显存被切成很多小块 | 总空闲够，但凑不出大连续块 |

PagedAttention 的边界正好在这里。它不改变模型权重，不改变注意力公式，也不减少每个 token 理论上要参与的注意力计算；它改变的是 **KV Cache 的物理存放方式**。把一段逻辑连续的上下文，映射到一组物理上不连续的页。

若页大小为 $B$ 个 token，则长度为 $S$ 的序列只需要：

$$
P=\left\lceil \frac{S}{B}\right\rceil
$$

页。此时单个请求的最大内部碎片不超过一页，即浪费 token 数满足：

$$
0 \le \text{waste} < B
$$

这条不等式很重要。它说明分页方案把浪费从“可能接近整段最大长度”压缩成“最多损失最后一页的尾部空间”。所以它解决的是 **显存管理效率**，不是“每个 token 的理论计算量”。

---

## 核心机制与推导

### 1. PagedAttention：把 KV Cache 从“整块分配”改成“页式分配”

PagedAttention 可以直译为“分页注意力”。它的核心思想很像操作系统的虚拟内存：**先把 KV Cache 切成固定大小的页，再用页表记录逻辑页到物理页的映射关系**。

最关键的数据结构通常是 `block table`。对每条序列来说，它记录的是：

| 逻辑页号 | 对应物理块号 |
|---|---|
| 0 | 17 |
| 1 | 3 |
| 2 | 42 |

也就是说，逻辑上连续的序列，物理上不必连续存放。这样做带来三个直接结果：

1. 序列只有在真正长到下一页时才申请新页。
2. 序列结束后，页可以立即回收到全局空闲池。
3. 多条序列共享前缀时，可以共享前缀页，而不是复制整段 KV。

如果页大小为 $B$，序列长度从 $S$ 增长到 $S+1$ 时，只有在满足

$$
(S+1) \bmod B = 1
$$

时才需要新分配一页。否则只是把新 token 写进当前尾页。于是单 token 增长不再对应单次大块显存申请，而是变成“多数时候只写已有页，少数时候申请新页”。

对新手来说，可以把这个机制理解成下面的对比：

| 方案 | 逻辑 |
|---|---|
| 连续分配 | “先给每个请求画一整块停车位，哪怕它最后只停一小段” |
| 分页分配 | “停车位切成很多小格，请求开到哪一格就占哪一格” |

共享前缀时，常见机制是 **copy-on-write**。意思是“先共享，只有当某个分支真的要写新内容时，才把相关页复制出来”。这在 beam search、多路采样、树状 agent 搜索里很常见。多个候选在分叉前共享同一段前缀页，直到某个候选开始扩展不同 token，才在尾部申请新页。

可以把单请求分页后的显存开销近似写成：

$$
M_{\text{paged}}(S) = \left\lceil \frac{S}{B} \right\rceil \times B \times M_{\text{kv/token}}
$$

它与理想值 $S \times M_{\text{kv/token}}$ 的差就是尾页浪费。于是有：

$$
0 \le M_{\text{paged}}(S)-M_{\text{req}}(S) < B \times M_{\text{kv/token}}
$$

这就是分页的核心收益来源：把浪费上界固定住。

### 2. Continuous Batching：每一步都重排批次

Continuous Batching 可以直译为“连续批处理”。它的核心不是“批更大”，而是 **批次在每个解码步都允许重排**。

静态批处理的问题很直接。假设一个批里有 4 条请求，长度分别为 20、20、20、200 token。前 3 条很快结束，但如果系统必须等第 4 条结束才能重新组下一批，那么后面很长一段时间 GPU 实际上只在服务 1 条请求。设备没有停机，但利用率明显下降。

连续批处理把“批”的概念从“同一时刻一起进入的一组请求”，改成“当前这个 token 步仍在活跃的槽位集合”。只要某个请求结束，它的槽位就能在下一个 step 被新请求补上。

设同时活跃槽位数为 $S_{\text{active}}$，每一步平均能有效推进的序列数为 $T_{\text{step}}$，则吞吐可粗略写成：

$$
\text{throughput} \propto S_{\text{active}} \cdot T_{\text{step}}
$$

其中真正可控的核心量，是让 $S_{\text{active}}$ 尽量稳定接近硬件允许的上限，而不是在长尾阶段掉到很低。

更细一点地看，静态批处理与连续批处理的差异可以写成：

| 机制 | 槽位释放时机 | 新请求进入时机 |
|---|---|---|
| 静态批处理 | 整批结束后 | 下一整批开始时 |
| 连续批处理 | 单请求结束后 | 下一解码步即可补入 |

于是连续批处理本质上是在减少两类浪费：

| 浪费 | 说明 |
|---|---|
| 空槽浪费 | 某些请求结束后，槽位闲置但不能立即复用 |
| 长尾浪费 | 短请求被迫陪长请求一起占着调度周期 |

这也是为什么它对“长短请求混合”的负载尤其有效。

### 3. Speculative Decoding：用便宜模型减少“昂贵步数”

Speculative Decoding 就是“投机解码”。直白解释是：**先让便宜模型猜几个 token，再由目标模型一次性验收这些候选，尽量把多步串行解码压缩成较少的目标模型前向**。

设草稿模型一次提出 $D$ 个 draft token，目标模型接受率为 $a$，则一次验收后，平均被接受的草稿 token 数约为：

$$
E[\text{accept}] \approx aD
$$

如果拒绝了一部分 token，就需要回退相应的尾部状态，回退规模近似为：

$$
\text{rewind}=D(1-a)
$$

它是否划算，主要由三件事决定：

1. 草稿模型是否足够便宜。
2. 接受率 $a$ 是否足够高。
3. 回退带来的 KV 写入、撤销和调度开销是否被收益覆盖。

这类方法最容易被误解成“主模型一次就能生成多个 token”。更准确的说法是：**主模型不再为每个 token 都独立跑一次完整解码，而是通过验证机制，把多个候选 token 的检查合并到更少的主模型调用中**。

对新手来说，可以用下面这个表理解它：

| 角色 | 作用 | 代价 |
|---|---|---|
| 草稿模型 | 快速提出候选 token 序列 | 算得便宜，但可能猜错 |
| 目标模型 | 负责最终验收 | 算得贵，但结果可信 |
| 回退逻辑 | 撤销未通过的尾部 token | 增加实现复杂度 |

如果 tokenizer 不一致，或者草稿模型与目标模型分布差得太远，$a$ 会显著下降。此时系统会频繁“先分配、再回退、再补写”尾页，收益很快被抵消。所以工程上经常把 **共享 tokenizer** 作为硬约束。

### 4. 一个完整玩具例子

假设页大小是 64 token，系统最多维持 4 个活跃槽位。现在有如下请求：

| 请求 | 总长度 |
|---|---|
| A | 50 |
| B | 300 |
| C | 700 |
| D | 32 |

按分页方式计算：

- A 需要 $1=\lceil 50/64 \rceil$ 页
- B 需要 $5=\lceil 300/64 \rceil$ 页
- C 需要 $11=\lceil 700/64 \rceil$ 页

因此最初只需分配 17 页，而不是先给 3 条请求各预留一整块 2048-token KV。

接下来发生的过程是：

1. 时刻 0：A、B、C 进入活跃集合。
2. A 很快结束，它的 1 页立即回收。
3. 下一个解码步，D 立刻补进空槽，只分配 1 页。
4. 如果 D 开启投机解码，草稿模型先猜 3 个 token，目标模型验收 2 个，则本轮净前进 2 个 token，仅回退 1 个 token 的尾部位置。

这个例子说明三件事：

| 优化 | 回答的问题 |
|---|---|
| PagedAttention | KV Cache 怎么存 |
| Continuous Batching | 槽位什么时候补 |
| Speculative Decoding | 每次尽量前进几步 |

三者并不冲突，反而正好覆盖不同层次的瓶颈。

### 5. 真实工程例子

在线聊天服务的流量通常不是均匀分布，而是明显长短混合。一个更接近真实的工程场景通常同时包含：

| 请求类型 | 输入长度 | 输出长度 | 典型特征 |
|---|---|---|---|
| 标题生成 | 30-80 token | 20-50 token | 很短、频繁、对延迟敏感 |
| 问答 / 总结 | 500-1500 token | 100-300 token | 数量多、长度中等 |
| 长上下文分析 | 8k 以上 | 不确定 | 数量少，但极占显存 |

如果用静态批处理，超长请求会拖住整批；如果用固定大块 KV 预留，短请求也要跟着占坑。于是现代推理系统常见做法是：

- 用分页 KV 减少每条请求的显存占坑。
- 用连续批处理持续填满活跃槽位。
- 在低批量、低延迟场景增加投机解码降低单请求时延。
- 再用动态批处理控制入口排队时间，避免“为了凑批而把延迟凑高”。

可以把这套组合理解成一个简单的调度管线：

$$
\text{请求到达} \rightarrow \text{短时间窗聚合} \rightarrow \text{进入连续调度} \rightarrow \text{按页管理 KV} \rightarrow \text{可选投机验收}
$$

真正的工程收益，通常不是某个单点算法突然把算子提速了多少，而是整条服务链路的等待、占坑和空转被压缩了。

---

## 代码实现

下面给出一个可直接运行的 Python 玩具实现，把“分页 KV + 连续补位 + 投机回退”三个概念串起来。它不复现真实 CUDA 内核，但能正确模拟：

- 页的按需分配与回收
- 活跃槽位的连续补位
- 投机解码的接受与回退
- 请求完成后的资源释放

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import ceil
from typing import Deque, Dict, List, Optional


PAGE_SIZE = 64


def pages_needed(tokens: int, page_size: int = PAGE_SIZE) -> int:
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return 0 if tokens == 0 else ceil(tokens / page_size)


def rewind_tokens(draft_len: int, accepted: int) -> int:
    if draft_len < 0:
        raise ValueError("draft_len must be non-negative")
    if not 0 <= accepted <= draft_len:
        raise ValueError("accepted must be in [0, draft_len]")
    return draft_len - accepted


class PagePool:
    """A minimal page allocator that reuses freed page ids."""

    def __init__(self) -> None:
        self._next_page_id = 0
        self._free_pages: List[int] = []

    def alloc(self) -> int:
        if self._free_pages:
            return self._free_pages.pop()
        page_id = self._next_page_id
        self._next_page_id += 1
        return page_id

    def free(self, page_id: int) -> None:
        self._free_pages.append(page_id)

    @property
    def free_pages(self) -> int:
        return len(self._free_pages)

    @property
    def total_pages_ever_allocated(self) -> int:
        return self._next_page_id


@dataclass
class Request:
    name: str
    prompt_tokens: int
    target_new_tokens: int
    generated_tokens: int = 0
    page_table: List[int] = field(default_factory=list)

    @property
    def total_tokens_in_cache(self) -> int:
        return self.prompt_tokens + self.generated_tokens

    @property
    def remaining_tokens(self) -> int:
        return self.target_new_tokens - self.generated_tokens

    @property
    def finished(self) -> bool:
        return self.generated_tokens >= self.target_new_tokens


def ensure_capacity(req: Request, pool: PagePool, page_size: int = PAGE_SIZE) -> None:
    needed = pages_needed(req.total_tokens_in_cache, page_size)
    while len(req.page_table) < needed:
        req.page_table.append(pool.alloc())


def free_all_pages(req: Request, pool: PagePool) -> None:
    while req.page_table:
        pool.free(req.page_table.pop())


def append_generated_tokens(req: Request, count: int, pool: PagePool, page_size: int = PAGE_SIZE) -> None:
    if count < 0:
        raise ValueError("count must be non-negative")
    req.generated_tokens += count
    ensure_capacity(req, pool, page_size)


def rollback_generated_tokens(req: Request, count: int, page_size: int = PAGE_SIZE) -> None:
    if count < 0 or count > req.generated_tokens:
        raise ValueError("invalid rollback size")
    req.generated_tokens -= count

    needed = pages_needed(req.total_tokens_in_cache, page_size)
    while len(req.page_table) > needed:
        req.page_table.pop()


def admit_waiting_requests(
    waiting: Deque[Request],
    active: List[Request],
    pool: PagePool,
    max_slots: int,
    page_size: int = PAGE_SIZE,
) -> None:
    while waiting and len(active) < max_slots:
        req = waiting.popleft()
        ensure_capacity(req, pool, page_size)  # allocate prompt pages
        active.append(req)


def decoding_step(
    active: List[Request],
    pool: PagePool,
    speculative_plan: Optional[Dict[str, tuple[int, int]]] = None,
    page_size: int = PAGE_SIZE,
) -> List[str]:
    logs: List[str] = []
    speculative_plan = speculative_plan or {}

    for req in active:
        if req.finished:
            continue

        draft_len, accepted = speculative_plan.get(req.name, (1, 1))
        draft_len = min(draft_len, req.remaining_tokens)
        accepted = min(accepted, draft_len)

        append_generated_tokens(req, draft_len, pool, page_size)
        rollback = rewind_tokens(draft_len, accepted)
        if rollback:
            rollback_generated_tokens(req, rollback, page_size)

        logs.append(
            f"{req.name}: draft={draft_len}, accepted={accepted}, "
            f"generated={req.generated_tokens}, pages={len(req.page_table)}"
        )

    return logs


def release_finished_requests(active: List[Request], pool: PagePool) -> List[Request]:
    remaining: List[Request] = []
    finished: List[Request] = []
    for req in active:
        if req.finished:
            free_all_pages(req, pool)
            finished.append(req)
        else:
            remaining.append(req)
    active[:] = remaining
    return finished


def run_simulation(
    requests: List[Request],
    max_slots: int = 2,
    page_size: int = PAGE_SIZE,
    speculative_schedule: Optional[Dict[int, Dict[str, tuple[int, int]]]] = None,
) -> List[str]:
    waiting: Deque[Request] = deque(requests)
    active: List[Request] = []
    pool = PagePool()
    timeline: List[str] = []
    step = 0
    speculative_schedule = speculative_schedule or {}

    while waiting or active:
        admit_waiting_requests(waiting, active, pool, max_slots, page_size)
        timeline.append(
            f"step={step} admitted={[req.name for req in active]} free_pool={pool.free_pages}"
        )

        logs = decoding_step(
            active,
            pool,
            speculative_plan=speculative_schedule.get(step),
            page_size=page_size,
        )
        timeline.extend(logs)

        finished = release_finished_requests(active, pool)
        if finished:
            timeline.append(
                f"step={step} finished={[req.name for req in finished]} free_pool={pool.free_pages}"
            )

        step += 1

    timeline.append(f"total_pages_ever_allocated={pool.total_pages_ever_allocated}")
    return timeline


if __name__ == "__main__":
    # Basic page math
    assert pages_needed(0) == 0
    assert pages_needed(50) == 1
    assert pages_needed(300) == 5
    assert pages_needed(512) == 8
    assert pages_needed(700) == 11

    # Speculative rollback math
    assert rewind_tokens(draft_len=3, accepted=2) == 1
    assert rewind_tokens(draft_len=4, accepted=4) == 0

    # A small mixed workload:
    # A: short request
    # B: medium request
    # C: very short request
    reqs = [
        Request(name="A", prompt_tokens=50, target_new_tokens=2),
        Request(name="B", prompt_tokens=300, target_new_tokens=5),
        Request(name="C", prompt_tokens=32, target_new_tokens=1),
    ]

    # At step 1, let B try speculative decoding: draft 3, accept 2.
    timeline = run_simulation(
        reqs,
        max_slots=2,
        speculative_schedule={
            1: {
                "B": (3, 2),
            }
        },
    )

    for line in timeline:
        print(line)
```

这段代码有几个值得注意的点：

| 模块 | 作用 |
|---|---|
| `PagePool` | 模拟全局页池，支持申请与复用 |
| `Request.page_table` | 模拟逻辑页到物理页的映射 |
| `admit_waiting_requests` | 模拟连续批处理的补位 |
| `decoding_step` | 模拟普通解码或投机解码 |
| `release_finished_requests` | 请求完成后立即释放页 |

运行后你会看到一个时间线，体现出三个关键行为：

1. 请求进入活跃槽位时先分配 prompt 所需页。
2. 某个请求结束后，页会立刻回收到池里。
3. 下一步可以立即补入新请求，而不必等待整批结束。

如果落到具体框架，vLLM 的常见配置重点是给调度器足够的活跃序列上限和 token 预算，例如：

```python
from vllm import AsyncLLMEngine, EngineArgs

engine = AsyncLLMEngine.from_engine_args(
    EngineArgs(
        model="meta-llama/Llama-3.1-70B",
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        scheduler_delay_factor=0.0,
    )
)
```

这些参数背后的含义可以简化理解为：

| 参数 | 作用 |
|---|---|
| `max_num_seqs` | 同时活跃的序列数上限 |
| `max_num_batched_tokens` | 单轮调度的 token 预算 |
| `scheduler_delay_factor` | 调度器等待补位的激进程度 |

在 TensorRT-LLM 一类系统里，投机解码通常通过专门配置打开，比如 `MTPDecodingConfig(...)` 或 draft-target 解码配置。最重要的工程前提不是“功能打开了”，而是 **draft 与 target 的 tokenizer、采样约束和回退路径必须一致**，否则接受率和稳定性都会明显变差。

---

## 工程权衡与常见坑

这类优化不是“开关一开就快”。收益很真实，代价也很真实。代价主要体现在三类地方：调度复杂度、元数据开销、以及延迟控制难度。

| 事项 | 风险 | 建议 |
|---|---|---|
| 页太小 | block table 更大，索引与管理开销更高 | 一般从 16/32/64 token 级别开始压测 |
| 页太大 | 单请求尾页浪费增加 | 长短请求混合明显时不要盲目放大页 |
| `max_num_seqs` 过大 | 槽位很多，但 token 预算不够，调度抖动变大 | 与 `max_num_batched_tokens` 联合调 |
| 批等待时间过长 | 平均吞吐上升，但 p95/p99 延迟恶化 | 动态批处理必须绑定严格延迟预算 |
| draft/target tokenizer 不一致 | 接受率低，投机收益快速退化 | 强制复用同一 tokenizer |
| 只看 batch size，不看显存余量 | 高峰期容易 OOM | 调度器要做 memory-aware admission |

一个常见误解是：“连续批处理一定降低延迟。”这并不总成立。连续批处理本身减少的是槽位空转，但如果入口层为了凑批设置了明显排队窗口，平均吞吐可能上升，尾延迟却会变坏。Dynamic Batching 的作用，正是在吞吐和等待之间给出一个在线折中。

另一个常见误解是：“Speculative Decoding 在任何场景都能提速。”它更常见的高收益场景是：

| 场景 | 原因 |
|---|---|
| 低批量 | 目标模型尚未被完全打满，存在可压缩的串行时间 |
| 强时延约束 | 更关心单请求响应速度，而不是总吞吐 |
| 草稿模型足够便宜且匹配良好 | 高接受率能摊薄验证成本 |

如果目标模型本来就处在高批量、高利用率状态，额外引入 draft 路径未必划算。

还要注意前缀共享的生命周期管理。共享前缀能明显省内存，但会带来引用计数、copy-on-write、会话过期回收等管理成本。这个成本通常比“给每个请求复制整段 KV”小很多，但不是零。系统做大以后，最常见的问题不是概念不对，而是边角状态没处理全：

- 请求取消时页有没有及时回收
- 投机回退时尾页计数有没有正确更新
- 前缀共享后某个分支写入时是否错误污染其他分支
- admission control 是否同时看了 token 预算和显存余量

这些问题任何一个出错，结果都不是“慢一点”，而是 OOM、错误输出或长尾延迟失控。

---

## 替代方案与适用边界

这几类优化并不只属于某一个框架，但不同框架的侧重点不同。

| 平台 | 侧重点 | 适用场景 |
|---|---|---|
| vLLM | PagedAttention、连续调度、OpenAI 兼容生态成熟 | 通用高并发在线服务 |
| TGI | 连续批处理、paged KV、流式输出 | Hugging Face 生态部署 |
| TensorRT-LLM | inflight batching、paged KV、量化、投机解码 | NVIDIA GPU 上追求极致性能 |
| SGLang | RadixAttention、前缀缓存、连续批处理 | 共享前缀多、Agent/RAG/结构化流程 |

适用边界也要说清楚：

1. 如果请求都很短、并发也低，PagedAttention 的收益可能不如量化、更小模型或更简单的服务化方案。
2. 如果业务极度重视单请求时延，Speculative Decoding 往往比“更激进地凑大批”更值得先做。
3. 如果请求长度差异大，Continuous Batching 的收益通常非常明显，因为它直接减少“短请求陪长请求罚站”的时间。
4. 如果前缀高度复用，比如统一系统提示词、固定模板、RAG 公共前缀很多，那么页共享和前缀缓存的价值会进一步放大。
5. 如果模型不是自回归生成模型，或者根本不依赖 KV Cache，这一整套 paged KV 机制就不是优化重点。

也可以把这些方案与其他常见优化做一个简单区分：

| 优化方向 | 主要解决什么 | 是否直接作用于 KV Cache |
|---|---|---|
| 量化 | 降低权重与算子成本 | 否，更多作用于权重和计算 |
| GQA / MQA | 减少 KV 头数，降低每 token KV 体积 | 是 |
| PagedAttention | 改善 KV 的存放和复用方式 | 是 |
| Continuous Batching | 提高活跃槽位利用率 | 间接相关 |
| Speculative Decoding | 减少主模型真实串行步数 | 间接相关 |

更实用的选择顺序通常是：

- **先解决显存占坑**：PagedAttention / paged KV
- **再解决 GPU 空转**：Continuous Batching
- **再解决单步串行**：Speculative Decoding
- **最后做入口折中**：Dynamic Batching

这个顺序通常比一开始就把所有特性都堆上更稳，因为它更符合线上瓶颈出现的先后顺序。

---

## 参考资料

| 资料 | 类型 | 作用 | 链接 |
|---|---|---|---|
| vLLM 项目主页 | 项目 | PagedAttention 的工程实现入口 | https://github.com/vllm-project/vllm |
| vLLM Paged Attention 设计文档 | 官方文档 | 解释 block/page table 与调度实现 | https://docs.vllm.ai/en/stable/design/paged_attention.html |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | 论文 | PagedAttention 的核心设计与实验 | https://arxiv.org/abs/2309.06180 |
| Orca: A Distributed Serving System for Transformer-Based Generative Models | 论文 | iteration-level scheduling 与连续批处理思路 | https://www.usenix.org/conference/osdi22/presentation/yu |
| Hugging Face TGI 文档 | 官方文档 | 连续批处理与服务部署实践 | https://huggingface.co/docs/text-generation-inference/index |
| TensorRT-LLM Speculative Decoding 文档 | 官方文档 | NVIDIA 推理栈中的投机解码配置与边界 | https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html |
| Accelerating Large Language Model Decoding with Speculative Sampling | 论文 | 经典 speculative decoding 方案 | https://arxiv.org/abs/2302.01318 |
| NVIDIA Triton Dynamic Batching 文档 | 官方文档 | 动态批处理的入口层配置与延迟权衡 | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html |
| SGLang 文档首页 | 官方文档 | RadixAttention、前缀缓存与服务特性 | https://docs.sglang.ai/ |
