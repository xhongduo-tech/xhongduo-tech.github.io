## 核心结论

vLLM 的主线不是“一个类把模型跑起来”，而是三层协作。

`AsyncLLMEngine` 负责在线入口。它是 `LLMEngine` 的异步包装，用 `asyncio` 后台循环持续接收请求、推进推理、向客户端流式返回结果。`LLMEngine` 负责把请求交给 `Scheduler`，并把调度结果分发给 `Worker`。`Worker` 是实际占用 GPU 的进程，每个进程通常对应一个加速器设备，内部由 `ModelRunner` 完成张量准备、CUDA Graph 捕获和模型执行。

对新手来说，可以把这三层理解成：

| 组件 | 直白定义 | 主要职责 | 典型状态/事件 |
|---|---|---|---|
| AsyncLLMEngine | 异步入口层，负责接请求和推流 | 接收 API 请求、驱动后台循环、持续吐出 token | arrive、stream output |
| LLMEngine + Scheduler | 调度决策层，决定这轮谁先跑、跑多少 | 维护 WAITING/RUNNING 队列，分配 token 预算，必要时抢占 | WAITING、RUNNING、PREEMPTED |
| Worker + ModelRunner | 执行层，真正让 GPU 算起来 | 执行 prefill/decode，更新 KV cache，返回采样结果 | compute、output |

核心闭环是：`arrive -> schedule -> run -> preempt/recompute -> output`。

这里有四个必须先记住的术语。

`prefill`：先把整段 prompt 过一遍模型，建立初始上下文状态。  
`decode`：在已有上下文上继续生成新 token。  
`KV cache`：模型把历史 token 的中间结果存到显存里，后续生成时直接复用。  
`chunked prefill`：把超长 prompt 切成多个小块分轮执行，避免一次占满整个批次预算。

vLLM V1 调度器的关键设计是：不再把世界硬分成“prefill 阶段”和“decode 阶段”，而是让每个请求都维护“已经算了多少 token”和“理论上应该算到哪里”。这样同一个调度器就能统一处理普通解码、长 prompt 分块、推测解码和恢复执行。

---

## 问题定义与边界

vLLM 要解决的问题很具体：在有限 GPU 显存和有限每轮 token 预算下，同时处理三类工作负载：

1. 新到达的请求，要尽快开始首 token 输出。
2. 正在 decode 的请求，要尽量维持低 ITL。ITL 是 inter-token latency，意思是两个输出 token 之间的间隔。
3. 超长 prompt 的请求，要尽量别把其他请求完全堵住。

如果没有调度，最容易出现两个坏结果。

第一，大 prompt 把整轮 batch 吃满，小请求虽然已经到达，却要等很久才能开始。  
第二，KV cache 爆满，系统只能停掉部分请求，否则新 token 根本塞不进显存。

这就是 vLLM 调度边界的来源：

| 约束 | 作用 | 超限后表现 |
|---|---|---|
| `max_num_batched_tokens` / `max_num_scheduled_tokens` | 每轮最多调度多少 token | 大 prompt 被切块，或等待更久 |
| `max_num_encoder_input_tokens` | 多模态/编码器输入预算上限 | 编码器相关输入被延后 |
| KV cache 容量 | 能同时保留多少请求的历史状态 | 触发 preemption 和 recompute |
| `max_num_seqs` | 同时运行的请求数量上限 | WAITING 队列堆积 |

一个玩具例子最容易看懂。

假设一轮总预算是 `64` token：

- 请求 A：正在 decode，本轮至少要继续算 `16` token
- 请求 B：新来的 prompt，还要 prefill `120` token

如果系统把 B 一次性全塞进去，A 的流式输出会被长时间卡住。  
开启 chunked prefill 后，调度器会优先放 A 的 `16`，剩下 `48` 给 B，于是 B 被拆成 `48 + 48 + 24` 三块。这样每轮 decode 都能插队执行，大 prompt 也能持续推进。

状态流可以画成下面这样：

```text
WAITING
  |
  | 被调度，分到 token 预算
  v
RUNNING
  |
  | 正常完成
  +-----------------------> OUTPUT
  |
  | KV cache 不够，释放显存
  v
PREEMPTED
  |
  | 后续重新进入等待并从头/部分重算
  v
WAITING
```

要注意一件事：`PREEMPTED` 不是“请求失败”，而是“为了腾显存，先把你让出来，稍后重算”。

---

## 核心机制与推导

vLLM V1 调度器有一个很重要的统一视角。每个请求都维护两个量：

$$
\text{num\_tokens\_with\_spec}
=
|\text{prompt}| + |\text{output}| + |\text{spec}|
$$

$$
\Delta
=
\text{num\_tokens\_with\_spec} - \text{num\_computed\_tokens}
$$

其中：

- `num_computed_tokens` 表示这个请求目前已经真正算过多少 token。
- `num_tokens_with_spec` 表示这个请求“理论上已经需要覆盖到哪里”，它把 prompt、已输出 token、推测 token 都合在一起。

调度器每一轮做的事，本质上就是让 `num_computed_tokens` 去追赶 `num_tokens_with_spec`。

这件事的价值很大，因为：

- 对 prefill，请求一开始 `output=0`，但 `prompt` 很大，所以缺口主要来自 prompt。
- 对 decode，请求 prompt 已经算完，缺口主要来自新增输出 token。
- 对 speculative decoding，草稿 token 也进入目标值，统一纳入预算。
- 对 chunked prefill，调度器只是每轮给它一部分 token 去追赶，不需要单独写一套“长 prompt 特判阶段机”。

预算可以写成：

$$
\text{token\_budget}
=
\min(\text{max\_num\_scheduled\_tokens}, \text{remaining})
$$

然后对每个请求分配：

$$
\text{scheduled\_tokens}
=
\min(\Delta, \text{token\_budget})
$$

如果是长 prompt，还会被进一步截断到单轮允许的 chunk 大小。

继续看刚才那个玩具例子：

| 轮次 | 总预算 | decode A 需求 | prefill B 剩余 | 本轮安排 | 本轮后 B 剩余 |
|---|---:|---:|---:|---|---:|
| 第 1 轮 | 64 | 16 | 120 | A:16, B:48 | 72 |
| 第 2 轮 | 64 | 16 | 72 | A:16, B:48 | 24 |
| 第 3 轮 | 64 | 16 | 24 | A:16, B:24 | 0 |

这个策略的核心不是“让所有请求公平”，而是“优先保证 decode 连续性，再把剩余预算塞给 prefill”。原因很简单：在线服务通常比吞吐更怕输出抖动。用户已经看到模型开始回答后，若两个 token 之间突然卡很久，体验会明显变差。

再看真实工程例子。

假设你用 vLLM 提供 OpenAI-compatible API，线上同时有：

- 很多短问答请求，prompt 只有几百 token，但要求流式返回
- 少量 RAG 请求，prompt 可能到 `8k` 甚至 `16k` token

如果没有 chunked prefill，长 RAG prompt 容易把 batch 顶满，短问答虽然轻量，也会排队等待。开启 chunked prefill 后，短请求 decode 会持续被优先调度，长请求的 prefill 则被拆成多轮推进。你得到的不是“长请求更快”，而是“全局交互延迟更稳”。

---

## 代码实现

下面用一个可运行的 Python 玩具调度器模拟 vLLM 的核心思想：先调度 RUNNING 请求，再调度 WAITING 请求；如果 KV 容量不足，就抢占低优先级请求并让它回到等待队列。

```python
from dataclasses import dataclass
from typing import List


@dataclass
class Request:
    req_id: str
    prompt_tokens: int
    output_tokens: int = 0
    spec_tokens: int = 0
    num_computed_tokens: int = 0
    kv_cost: int = 0
    priority: int = 0  # 数值越小优先级越高
    status: str = "WAITING"

    @property
    def num_tokens_with_spec(self) -> int:
        return self.prompt_tokens + self.output_tokens + self.spec_tokens

    @property
    def deficit(self) -> int:
        return max(0, self.num_tokens_with_spec - self.num_computed_tokens)


class ToyScheduler:
    def __init__(self, max_tokens: int, kv_capacity: int):
        self.max_tokens = max_tokens
        self.kv_capacity = kv_capacity
        self.waiting: List[Request] = []
        self.running: List[Request] = []

    def kv_used(self) -> int:
        return sum(r.kv_cost for r in self.running)

    def preempt_one(self):
        victim = max(self.running, key=lambda r: r.priority)
        self.running.remove(victim)
        victim.status = "PREEMPTED"
        victim.num_computed_tokens = 0
        self.waiting.insert(0, victim)
        return victim.req_id

    def schedule(self):
        token_budget = self.max_tokens
        scheduled = []

        # 先跑 RUNNING，模拟 decode 优先
        self.running.sort(key=lambda r: r.priority)
        for req in list(self.running):
            if token_budget <= 0:
                break
            take = min(req.deficit, token_budget)
            req.num_computed_tokens += take
            token_budget -= take
            scheduled.append((req.req_id, take, req.status))

        # 再拉 WAITING
        while self.waiting and token_budget > 0:
            req = self.waiting[0]

            # KV 不够就先抢占一个低优先级运行中请求
            if self.kv_used() + req.kv_cost > self.kv_capacity and self.running:
                self.preempt_one()
                continue

            self.waiting.pop(0)
            req.status = "RUNNING"
            self.running.append(req)

            take = min(req.deficit, token_budget)
            req.num_computed_tokens += take
            token_budget -= take
            scheduled.append((req.req_id, take, req.status))

        return scheduled


# 玩具场景：
# A 是正在 decode 的请求，本轮还差 16 token
# B 是长 prompt prefill，请求 120 token
sched = ToyScheduler(max_tokens=64, kv_capacity=10)

a = Request("A", prompt_tokens=32, output_tokens=16, num_computed_tokens=32,
            kv_cost=4, priority=0, status="RUNNING")
b = Request("B", prompt_tokens=120, kv_cost=7, priority=1)

sched.running.append(a)
sched.waiting.append(b)

round1 = sched.schedule()
assert round1[0][0] == "A"
assert round1[0][1] == 16
assert round1[1][0] == "B"
assert round1[1][1] == 48
assert b.num_computed_tokens == 48

# 再来一个更高优先级的新请求 C，KV 不够时会触发抢占
c = Request("C", prompt_tokens=20, kv_cost=6, priority=-1)
sched.waiting.insert(0, c)

round2 = sched.schedule()
# 为了让 C 进入运行，B 很可能被抢占回 waiting
statuses = {r.req_id: r.status for r in sched.running + sched.waiting}
assert statuses["C"] == "RUNNING"
assert "PREEMPTED" in statuses.values()

print("toy scheduler ok")
```

这段代码省略了真实 vLLM 里的很多细节，但保留了三个关键点。

第一，调度顺序是“先 RUNNING，后 WAITING”。这对应源码中先遍历 `running` 队列，再视剩余预算拉取 `waiting`。  
第二，token 是按预算分配的，不是按请求完整执行。  
第三，显存不够时，不是简单报错，而是 `_preempt_request()`：释放 KV/encoder cache，把请求状态置为 `PREEMPTED`，并塞回等待队列前部，后续再恢复。

把它和真实实现对齐，可以概括成下面的伪代码：

```python
def schedule():
    token_budget = max_num_scheduled_tokens

    for req in running:
        token_budget -= schedule_one(req)
        if kv_not_enough():
            preempt(low_priority_running_req)

    while waiting and token_budget > 0:
        req = waiting.peek()
        token_budget -= schedule_one(req)
```

这里最容易误解的一点是 `recompute`。它不是“从模型头开始重新训练”，而是“被抢占后，重新做必要的前向计算，恢复到继续生成所需的状态”。

---

## 工程权衡与常见坑

chunked prefill 带来的不是单向收益，而是明确权衡。

| 调整项 | 往大调的效果 | 往小调的效果 | 风险 |
|---|---|---|---|
| `max_num_batched_tokens` | TTFT 和吞吐通常更好 | ITL 往往更稳 | 太大时长 prompt 更容易打断 decode |
| `gpu_memory_utilization` | 可用 KV cache 更大，preemption 更少 | 显存更保守 | 太激进时容易逼近 OOM 边缘 |
| `tensor_parallel_size` | 模型权重分摊，单卡留给 KV 的空间更多 | 通信更少 | 太大时同步开销上升 |
| `max_num_seqs` | 并发更高 | 单批压力更小 | 过高时更容易触发抢占 |

常见坑基本都能映射到一个模式：症状不是根因，根因常常在 KV cache。

| 症状 | 先看什么 | 常见根因 | 常用调整 |
|---|---|---|---|
| 日志频繁出现 `RECOMPUTE`/preemption | KV 使用率、并发数 | KV 空间不够 | 提高 `gpu_memory_utilization`，或降低 `max_num_batched_tokens` / `max_num_seqs` |
| 首 token 很快，但后续输出一卡一卡 | decode 是否被大 prefill 干扰 | batch 太大，prefill 插入过多 | 降低 `max_num_batched_tokens` |
| 吞吐不错，但用户体感差 | ITL 指标 | decode 优先级不够或 chunk 太大 | 开启 chunked prefill，缩小 chunk 预算 |
| 多卡后反而变慢 | GPU 利用率和通信时间 | TP/PP 过大，通信抵消收益 | 回退并行规模，重新压测 |

一个真实工程调参路径可以写得很直白：

1. 线上看到 preemption 警告先别急着加机器，先看 KV 是否真的满了。
2. 如果确实经常满，先提高 `gpu_memory_utilization`。
3. 如果仍频繁 preempt，再看是否 batch 过大；适当降低 `max_num_batched_tokens` 或 `max_num_seqs`。
4. 如果模型权重本身太吃显存，再考虑提高 `tensor_parallel_size`，但必须重新测端到端延迟，因为同步开销可能抵消收益。

一个经常被忽略的点是：chunked prefill 不是“prompt 越长越划算”。如果 chunk 切得过碎，调度器循环、队列管理和多轮 kernel 发射都会增加额外开销。也就是说，过度切块会把“避免阻塞”的收益换成“调度成本上升”的损失。

---

## 替代方案与适用边界

不是所有场景都应该上 `AsyncLLMEngine + chunked prefill`。

如果你的任务是离线批处理，例如夜间批量跑摘要、分类或 embedding 风格的长上下文生成，而且没有流式输出要求，那么同步 `LLMEngine` 就已经足够。此时你更关注总吞吐，而不是用户看到的 token 间隔；调度器也可以更偏向完整 prefill，再进入 decode。

可以用下面这张矩阵做粗判断：

| 场景 | prompt 长度 | 是否流式输出 | 更合适的方案 |
|---|---|---|---|
| 在线问答 API | 短到中等 | 是 | `AsyncLLMEngine + chunked prefill` |
| 在线 RAG 服务 | 长 | 是 | `AsyncLLMEngine + chunked prefill`，重点盯 ITL 和 preemption |
| 离线批处理 | 短 | 否 | `LLMEngine` 即可 |
| 离线长上下文推理 | 长，但并发低 | 否 | `LLMEngine`，可弱化 chunked prefill 价值 |

简化成一句决策规则：

- 目标是低交互延迟：优先 `AsyncLLMEngine`，并让 decode 优先。
- 目标是纯吞吐：可以弱化异步和分块带来的复杂度。
- prompt 总长本来就明显低于 `max_num_batched_tokens`：chunked prefill 的收益会变小。
- KV cache 已经非常充足、并发也低：preemption 和 recompute 不是主要矛盾，此时调度复杂度的收益有限。

所以，“三层架构”真正解决的不是“怎么把模型跑起来”，而是“怎么在共享 GPU 上把不同生命周期的请求塞进同一个稳定系统里”。这也是 vLLM 比“单请求单前向”更像推理操作系统的原因。

---

## 参考资料

1. vLLM Architecture Overview：`AsyncLLMEngine`、`LLMEngine`、`Worker` 的职责  
https://docs.vllm.ai/en/stable/design/arch_overview/

2. vLLM Scheduler API：`num_tokens_with_spec`、`num_computed_tokens`、`_preempt_request()` 的设计与源码  
https://docs.vllm.ai/en/stable/api/vllm/v1/core/sched/scheduler/

3. vLLM Performance and Tuning：Chunked Prefill、`max_num_batched_tokens` 对 ITL/TTFT 的影响  
https://docs.vllm.ai/en/v0.4.2/models/performance.html

4. vLLM Optimization and Tuning：preemption、`gpu_memory_utilization`、`tensor_parallel_size` 的调优建议  
https://docs.vllm.ai/en/stable/configuration/optimization/
