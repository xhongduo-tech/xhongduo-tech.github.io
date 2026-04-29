## 核心结论

vLLM 调度器的本质，不是“把请求按顺序排队”，而是一个在线批处理系统：它在每一轮生成 token 时，同时看 `token budget`、`KV cache`、等待请求、运行请求和优先级，然后决定“谁现在进 batch、谁先解码、谁被抢占”。

这里最容易被初学者误解的一点是：LLM serving 的第一瓶颈往往不是算力，而是显存里的 `KV cache`。`KV cache` 可以白话理解为“模型为了继续续写，必须一直留在显存里的上下文记忆”。算力不够会让每步都慢；KV 不够会让很多请求根本塞不进来。

一个长 prompt 为什么会卡住后续请求？因为它不仅要做一次长 `prefill`，还会留下大量 KV block。只要这些 block 不被释放，后面的短请求即使只想生成几个 token，也可能没有足够显存进入运行队列。

| 维度 | 算力瓶颈 | 显存/KV 瓶颈 |
|---|---|---|
| 直观表现 | GPU 核心忙，单步计算慢 | GPU 还有算力，但新请求装不下 |
| 主要来源 | 模型大、算子慢、批太大 | 上下文长、并发高、KV cache 占满 |
| 对用户的影响 | 所有人都慢 | 有些请求直接排队、被抢占、尾延迟抖动 |
| 常见优化 | 算子融合、量化、并行 | PagedAttention、调度、抢占、参数限流 |

所以，vLLM 的核心价值不是“它比别人快一点”，而是它把长短请求混跑、多租户竞争和显存分页管理，统一进一套高吞吐调度逻辑中。理解这套逻辑，才真正理解现代 LLM serving 为什么能把 GPU 利用率拉高。

---

## 问题定义与边界

先限定边界。本文讨论的是 vLLM 在线推理时的调度，不讨论训练，不讨论分布式一致性协议，也不讨论模型精度。重点只有一个：当很多请求同时来，且请求长度差异很大时，系统怎样在有限显存里把更多 token 挤进 GPU。

几个基础术语先定清楚：

| 术语 | 白话解释 | 在系统里的作用 |
|---|---|---|
| 等待队列 | 还没开始跑的请求集合 | 等待被装入 batch |
| 运行队列 | 当前已经占用 GPU 资源的请求集合 | 持续参与 prefill 或 decode |
| `prefill` | “先把整段输入读一遍”的阶段 | 计算 prompt 对应的首批 KV |
| `decode` | “每次只续一个 token”的阶段 | 反复读取已有 KV，继续生成 |
| `KV cache` | 模型续写时保留的上下文记忆 | 决定一个请求会占多少显存 |
| `token budget` | 本轮 batch 最多允许处理的 token 配额 | 防止一次塞太多 token 导致延迟失控 |

新手常见问题是：“一个请求为什么会一直占着 GPU？”原因是请求不是算完 prompt 就结束。进入 `decode` 后，它每生成一个 token，都要继续依赖之前的 KV cache，所以这份显存会跟着请求一直活到生成结束或被抢占。

另一个问题是：“为什么短追问不能简单插队？”因为插队不是把字符串插进去，而是要把它的 `prefill` token 和后续 KV 一起塞进显存。如果 GPU 剩余 block 不够，哪怕这个请求只想回答一句话，它也进不来。于是调度器必须做更复杂的决策：是让它继续等，还是暂停一个已经在跑的长请求，把显存腾出来。

一个玩具例子：

- 请求 A：历史上下文 6000 token，正在生成第 20 个 token
- 请求 B：新来的追问只有 30 token，希望尽快得到回答
- GPU：算力还够，但 KV block 只剩很少

如果没有抢占，B 只能干等 A 释放显存。  
如果有抢占，调度器可以暂停 A，把 block 让给 B，让 B 更快返回。代价是 A 后面恢复时要么重算，要么换入。

这就是 vLLM 调度问题的本体：不是“谁先排队”，而是“在有限 block 和 token 配额下，怎样动态重排请求，兼顾吞吐和延迟”。

---

## 核心机制与推导

vLLM 的第一层机制是 `PagedAttention`。它可以白话理解为“把 KV cache 不再当成一整块连续显存，而是拆成固定大小的分页块来管理”。传统连续分配的问题是，一个请求越长，需要的显存越大，而且长度不稳定，容易产生碎片和预留浪费。PagedAttention 则把每个序列的 KV 切成若干 block，block 不要求在物理显存里连续。

设块大小为 $B$，一个序列当前总长度为 $L$，则：

$$
N=\lceil L/B \rceil
$$

其中 $N$ 是需要的块数。实际占用槽位是

$$
U=N \cdot B
$$

其中前面若干槽位被真实 token 使用，最后一个块可能没填满，因此浪费槽位为

$$
W=U-L
$$

浪费率是

$$
R=W/U
$$

玩具推导例子，取 $B=16,\ L=37$：

- $N=\lceil 37/16 \rceil=3$
- $U=3\times16=48$
- $W=48-37=11$
- $R=11/48\approx22.9\%$

这说明固定块大小一定会带来尾块浪费，但它换来了两个更重要的收益：

1. 分配简单。需要新 token 时，只要再拿一个 block，不必搬动整段 KV。
2. 调度灵活。请求暂停、恢复、共享前缀时，都能以 block 为单位操作。

第二层机制是 `Preemption`，也就是抢占。白话解释是：“新请求装不下时，先把某个正在跑的请求停下来，把它占着的显存释放或换出，等以后再恢复。”

它常见有两种恢复路径：

| 机制 | 白话解释 | 主要代价 | 适合场景 |
|---|---|---|---|
| PagedAttention | KV 按块分页管理 | 尾块有少量浪费 | 长短请求混跑、前缀共享、显存紧张 |
| Preemption | 显存不够时暂停请求 | 会增加恢复成本 | 多租户竞争、高并发 |
| `RECOMPUTE` | 释放 GPU 上的 KV，以后从头重算 prompt | 重复算前向 | 单序列、重算成本可接受 |
| `SWAP` | 把 KV 换到 CPU，之后再换回 GPU | CPU/GPU 传输慢 | 多序列组或不适合重算 |

为什么会有 `RECOMPUTE` 和 `SWAP` 两条路？因为“暂停”不是一个动作，而是一个代价模型。

- `RECOMPUTE`：现在最省显存，恢复时最费算力。
- `SWAP`：现在不丢状态，恢复时要付数据传输代价。

从工程上看，调度器每轮都在做一个局部最优问题：  
给定本轮 `token budget`，优先安排哪些 `decode`，是否插入 `prefill`，如果 block 不够，抢占谁最划算。

真实工程例子是多租户聊天服务。A 租户在做长文总结，prompt 8000 token；B 租户在做客服问答，prompt 只有 60 token；C 租户还开了多路采样。没有分页和抢占时，A 很容易把大量 KV 长时间占住；有了 vLLM 调度器后，系统可以优先推进一批 decode 请求，把短问答先返回，同时在必要时暂停低收益或低优先级的长请求，整体吞吐和尾延迟都更稳定。

---

## 代码实现

实现视角里，核心不在模型前向，而在调度器。模型负责“算”；调度器负责“决定谁有资格现在算”。

源码层面，理解这几个入口最有帮助：

- `_schedule_priority_preemption`
- `_preempt_by_recompute`
- `_preempt_by_swap`

它们表达的不是复杂语法，而是三个关键动作：按优先级检查是否要抢占、走重算路径、走换出路径。

下面是一个可运行的简化 Python 版本，用来模拟“有 block 限额时如何调度和抢占”。它不是 vLLM 源码，但逻辑结构与真实系统接近。

```python
from dataclasses import dataclass, field
from math import ceil

BLOCK_SIZE = 16
TOTAL_BLOCKS = 8

@dataclass
class Request:
    req_id: str
    prompt_len: int
    generated: int = 0
    priority: int = 0
    status: str = "WAITING"   # WAITING / RUNNING / SWAPPED / FINISHED
    mode: str = "RECOMPUTE"   # RECOMPUTE / SWAP

    def total_len(self) -> int:
        return self.prompt_len + self.generated

    def blocks_needed(self) -> int:
        return ceil(self.total_len() / BLOCK_SIZE)

@dataclass
class Scheduler:
    waiting: list[Request] = field(default_factory=list)
    running: list[Request] = field(default_factory=list)
    swapped: list[Request] = field(default_factory=list)

    def used_blocks(self) -> int:
        return sum(r.blocks_needed() for r in self.running)

    def free_blocks(self) -> int:
        return TOTAL_BLOCKS - self.used_blocks()

    def preempt(self) -> Request:
        victim = min(self.running, key=lambda r: (r.priority, -r.blocks_needed()))
        self.running.remove(victim)
        if victim.mode == "RECOMPUTE":
            victim.status = "WAITING"
            victim.generated = 0
            self.waiting.append(victim)
        else:
            victim.status = "SWAPPED"
            self.swapped.append(victim)
        return victim

    def try_admit(self, req: Request) -> bool:
        while self.free_blocks() < req.blocks_needed() and self.running:
            self.preempt()
        if self.free_blocks() >= req.blocks_needed():
            req.status = "RUNNING"
            self.running.append(req)
            return True
        return False

    def schedule(self):
        self.waiting.sort(key=lambda r: (-r.priority, r.prompt_len))
        admitted = []
        remaining = []
        for req in self.waiting:
            if self.try_admit(req):
                admitted.append(req.req_id)
            else:
                remaining.append(req)
        self.waiting = remaining
        return admitted

# 玩具例子：一个长请求先占住大部分 block，新短请求到来后触发抢占
long_req = Request("A", prompt_len=90, generated=10, priority=0, mode="RECOMPUTE")
short_req = Request("B", prompt_len=20, generated=0, priority=10, mode="RECOMPUTE")

s = Scheduler(running=[long_req], waiting=[short_req])
admitted = s.schedule()

assert "B" in admitted
assert any(r.req_id == "A" for r in s.waiting) or any(r.req_id == "A" for r in s.swapped)
assert sum(r.blocks_needed() for r in s.running) <= TOTAL_BLOCKS
```

这段代码表达了四个实现重点：

1. `schedule` 不是一次性排完，而是每轮都重新决策。
2. `preempt` 不是异常流程，而是资源不足时的正常控制流。
3. `RECOMPUTE` 会把请求重新放回等待态，恢复时重算。
4. `SWAP` 保留状态，但后续必须 `swap in` 才能继续运行。

伪代码可以再抽象成下面这样：

```python
def schedule_step(waiting, running, swapped, token_budget, free_blocks):
    prioritize_decode_requests(running, token_budget)

    while has_pending_prefill(waiting) and token_budget_not_exceeded():
        req = pick_next_request(waiting)

        need = estimate_new_blocks(req)
        while free_blocks < need:
            victim = choose_victim_by_priority_or_cost(running)
            if victim.can_recompute():
                preempt_by_recompute(victim)
            else:
                preempt_by_swap(victim)

            free_blocks = recalc_free_blocks()

        admit(req)
        allocate_blocks(req)

    for req in swapped:
        if free_blocks_enough_for_swap_in(req):
            swap_in(req)
            resume(req)
```

真正的工程难点不在“会不会写 if”，而在“受害者怎么选”。如果选错了，就会出现频繁重算、换入换出抖动，吞吐反而下降。

---

## 工程权衡与常见坑

抢占不是免费操作。`RECOMPUTE` 的账单记在算力上，`SWAP` 的账单记在带宽和延迟上。工程里真正调的是三件事的平衡：吞吐、尾延迟、显存利用率。

一个典型高风险场景是：长 prompt + 小显存 + 高并发。  
长 prompt 让每个请求在 `prefill` 后留下很多 KV block；小显存让系统没有缓冲区；高并发则不断引入新请求。三者叠加，调度器就容易频繁抢占。

为什么只调 `max_num_seqs` 不一定有效？因为它限制的是“同时多少个序列”，但不直接限制“这一轮一共塞多少 token”。如果 `max_num_batched_tokens` 很大，单轮 prefill 仍可能把 KV 压力拉满；反过来，`max_num_seqs` 不高，但每个请求都超长，显存一样会爆。

| 常见坑 | 表现 | 根因 | 规避手段 |
|---|---|---|---|
| `preemption` 过多 | 日志频繁出现抢占，尾延迟抖动 | KV 空间长期紧张 | 提高 `gpu_memory_utilization`，或降低并发上限 |
| `swap` 成本过高 | GPU 利用率不低，但响应仍慢 | CPU/GPU 传输成为瓶颈 | 减少需要 `SWAP` 的多序列组，优先控制请求形态 |
| 只盯 `max_num_seqs` | 参数调了但效果不稳定 | 忽略 token 级预算 | 联合观察 `max_num_batched_tokens` |
| 长 prompt 混入短问答 | 短请求排队时间异常长 | 长请求长期占据大量 KV | 做流量分层、隔离长上下文租户 |
| 恢复策略误判 | 某些请求反复被暂停恢复 | 受害者选择不合理 | 结合优先级、剩余长度、恢复成本做策略 |
| 没有监控抢占指标 | 出问题时只看到“慢” | 调度内部状态不可见 | 监控 preemption 次数、swap 行为、尾延迟分位数 |

真实工程里，常见误判是把“GPU 利用率高”当成系统健康。事实未必如此。如果 GPU 高利用率来自大量重算或频繁换入换出，那只是“忙”，不是“有效忙”。

因此，最有价值的监控不是单一 GPU 利用率，而是组合看：

- preemption 次数
- P50/P95/P99 延迟
- 每轮 batch token 数
- swap 相关开销
- 长短请求的队列等待时间

---

## 替代方案与适用边界

vLLM 不是唯一可行方案，只是它在“请求长度差异大、并发高、显存紧”的 serving 场景下非常强。

如果场景简单，比如离线批处理、输入长度接近、没有多租户竞争，那么静态 batch 或简单 FIFO 也能工作，而且实现更简单、行为更可预测。

| 方案 | 怎么做 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|---|
| 静态 batch | 预先凑一批，整批一起跑 | 简单，易实现 | 要等整批，长短请求互相拖累 | 离线推理、长度接近 |
| FIFO | 先进先出，不复杂调度 | 公平直观 | 长请求容易堵住短请求 | 低并发、单租户 |
| 连续 KV 分配 | 每请求占连续显存 | 实现直接 | 容易碎片化，扩容困难 | 请求长度稳定、显存富余 |
| PagedAttention + 调度器 | 块式 KV + 在线调度 + 抢占 | 吞吐高，适合混合负载 | 策略复杂，调参和监控要求高 | 在线服务、高并发、多租户 |

用一句最直接的话对比：

- 静态 batch：通常要等整批凑好、整批推进，批内“慢的拖快的”。
- vLLM：可以边运行边重排，在 block 不够时抢占，在 token budget 允许时插入更合适的请求。

但这不代表所有推理场景都必须上这套机制。若你的业务是固定长度 embedding，或者每天夜里跑一批离线生成，调度复杂度可能不值得。vLLM 的价值主要体现在“在线、动态、不均匀”这三个词同时成立时。

---

## 参考资料

1. [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
2. [vLLM Scheduler API 文档](https://docs.vllm.ai/en/v0.10.2/api/vllm/core/scheduler.html)
3. [vLLM Paged Attention 设计文档](https://docs.vllm.ai/en/v0.11.2/design/paged_attention/)
4. [vLLM 官方博客：Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/vllm)
5. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://huggingface.co/papers/2309.06180)
