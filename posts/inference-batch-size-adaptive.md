## 核心结论

推理批大小自适应，指服务端不把 batch size 固定死，而是在每次调度时根据当前排队请求、token 数、显存余量和延迟目标，动态决定这一轮要合并多少工作一起算。白话说，它不是“永远尽量塞满 GPU”，而是“在不把用户等太久的前提下，尽量把 GPU 喂饱”。

这件事成立的原因很直接。一次推理通常有固定开销，比如调度、内核启动、数据搬运；把多个请求合并后，这部分固定成本会被摊薄，GPU 利用率也更高，所以吞吐常常提升。但 batch 不是免费午餐。为了凑批，请求必须先排队；一旦排队时间过长，用户感知最强的 TTFT 就会变差。TTFT 是 time to first token，白话说就是“用户从发请求到看到第一个字”的时间。

因此，真实系统里真正要自适应的对象通常不是单一的“请求个数”，而是三个量的联合决策：

1. 这轮放几个请求。
2. 这轮总共放多少 token。
3. 最多允许这些请求等多久。

可以把它写成一个总括约束：

$$
\text{choose } B_t \quad \text{s.t.}\quad
n_t \le N,\;
u_t \le T,\;
\text{KV}(B_t)\le M,\;
\max_i \text{wait}_i \le D
$$

其中，$n_t$ 是本轮请求数，$u_t$ 是本轮 token 总量，$M$ 是显存或 KV cache 上限，$D$ 是排队延迟上限。

下面这个表先把最容易混淆的点压缩清楚：

| 关注点 | batch 变大时常见变化 | 为什么 |
| --- | --- | --- |
| 吞吐 TPS | 往往上升 | 固定开销被摊薄，GPU 更容易满载 |
| 平均成本 | 往往下降 | 单位请求分摊到的调度成本更低 |
| 单请求等待时间 | 往往上升 | 需要排队凑批 |
| P99 延迟 | 容易变差 | 慢请求和排队时间会放大尾部 |
| 显存压力 | 往往上升 | 更多请求和更多 token 会占更多 KV cache |

结论先说死一句：推理批大小自适应的目标不是“把 batch 调大”，而是“在吞吐、延迟、显存三者之间做运行时折中”。

---

## 问题定义与边界

如果不先定义目标和边界，“自适应 batch”很容易变成一句空话。你必须先回答三个问题：

1. 你要优化什么，是 TPS、成本，还是 TTFT/P99？
2. 你的硬约束是什么，是显存、SLO，还是队列长度？
3. 你的推理对象是什么，是普通无状态模型，还是带 KV cache 的大语言模型？

先定义几个核心术语。术语第一次出现时，不要只记英文缩写，要记住它们在系统里的角色。

| 术语 | 定义 | 白话解释 |
| --- | --- | --- |
| `batch size` | 一次调度里合并执行的请求数 | 一轮一起算多少单 |
| `token budget` | 这一轮允许装入的 token 总量上限 | 这一车最多能装多少货 |
| `KV cache` | 解码阶段保存注意力历史的缓存 | 模型记住前文要占的显存 |
| `TTFT` | Time To First Token | 用户多久看到第一个字 |
| `TPS` | Tokens Per Second 或系统吞吐指标 | 系统每秒能产出多少内容 |
| `P99 latency` | 99 分位延迟 | 最慢那 1% 请求大概有多慢 |

形式化一点，问题可以写成：

$$
\max_{B_t}\ \text{TPS}(B_t)
\quad \text{s.t.}\quad
n_t \le N,\ 
u_t \le T,\ 
\text{KV}(B_t)\le M,\ 
\max_i \text{wait}_i \le D
$$

这个公式的含义是：第 $t$ 次调度时，你想选出一批请求 $B_t$，让吞吐尽可能高，同时满足四个上限：
- 请求数不能超过 `max_batch_size`
- token 总量不能超过 `max_num_tokens`
- KV cache 不能把显存顶爆
- 任一请求等待时间不能超过延迟预算

这里最容易踩的概念坑，是把“请求数”误当成“负载大小”。对 LLM 来说，两个请求都算 1 个请求，但一个可能只有 20 个输入 token，另一个可能有 8000 个输入 token，它们对算力和显存的压力完全不是一个级别。所以 $u_t=\sum \text{tokens}_i$ 往往比单纯的请求个数更接近真实负载。

还要区分两类 batching：

1. `prefill batching`
   把输入 prompt 一次性编码进模型。白话说，就是先把“读题”这一步做完。
2. `decode batching` 或 `continuous batching`
   每一步只生成少量 token，并且每一步都可能有新请求插入。白话说，就是“边生成边重新拼车”。

这两类场景的调度逻辑不同。prefill 更像大块计算，decode 更像细粒度循环。很多新手把两者混成一个参数问题，这是不对的。

适用边界也要讲清楚：

| 场景 | 是否适合自适应 batch | 原因 |
| --- | --- | --- |
| 长短请求混合明显的 LLM 服务 | 适合 | 负载波动大，固定 batch 很浪费 |
| 请求长度接近、负载稳定的离线任务 | 未必需要 | 静态 batch 往往更简单 |
| 强 TTFT SLO 的在线聊天 | 适合但要克制 | 需要严格控排队时间 |
| 有复杂会话状态绑定的模型 | 谨慎 | 请求不一定能自由重排 |
| 极小流量服务 | 收益有限 | 很难凑批，排队成本可能更高 |

玩具例子先给一个。假设你有 4 个请求，其中 3 个只有几十个 token，1 个有 5000 个 token。如果你只按“4 个请求正好一批”来调度，看起来很整齐，但那个长请求会把本轮 prefill 时间和显存占用一起拉高，其他短请求也被迫陪跑。这说明：推理批大小自适应的边界，不在“有没有队列”，而在“请求之间是否足够同质”。

---

## 核心机制与推导

先用一个最小数值例子把机制钉住。假设一次推理的固定开销是 6 ms，每个请求增加 2 ms。

- batch=1 时，总耗时 $T(1)=6+2\times1=8$ ms
- batch=4 时，总耗时 $T(4)=6+2\times4=14$ ms

如果只看吞吐：
- batch=1，大约是 $1/0.008=125$ req/s
- batch=4，大约是 $4/0.014\approx286$ req/s

吞吐明显提高了，因为固定开销被摊薄了。

但如果为了凑够 4 个请求，你额外等了 5 ms，那么最后一个请求的端到端延迟会变成：

$$
L_{\text{e2e}} = 5 + 14 = 19\text{ ms}
$$

这就是推理调度里最重要的一句话：吞吐提升，不等于用户感知延迟一定更好。很多“压测很好看、线上被投诉”的系统，问题就出在这里。

把变量正式定义一下：

- $B_t$：第 $t$ 次调度选中的请求集合
- $n_t=|B_t|$：这一轮的请求数
- $u_t=\sum_{i\in B_t}\text{tokens}_i$：这一轮 token 总量
- $\text{wait}_i$：请求 $i$ 进入队列后已经等待的时间

为什么 `tokens_i` 比请求数更重要？因为 LLM 的成本通常和序列长度强相关。以 decode 为例，每个活跃请求都要维护自己的 KV cache；输入越长、输出越长，占用越大。把两个请求都视为“1 个单位”，会把真实资源需求压扁成错误模型。

方向性可以总结成这张表：

| batch 增大时 | 变化方向 | 解释 |
| --- | --- | --- |
| 吞吐 | 常上升 | 固定开销被摊薄 |
| GPU 利用率 | 常上升 | 更容易形成连续计算 |
| 单请求等待时间 | 常上升 | 需要时间凑批 |
| 尾延迟 | 常上升 | 排队和慢请求拖尾 |
| OOM 风险 | 常上升 | token 和 KV cache 更大 |

所以系统要做的不是“批越大越好”，而是每一轮都解一个简化决策问题：当前还能装多少？还能等多久？

一个实用的简化流程如下：

1. 请求入队。
2. 估计每个请求的输入 token、预期输出上限、KV 占用。
3. 先看队首请求是否快超出 `max_queue_delay`。
4. 在不突破 `max_batch_size`、`max_num_tokens`、显存上限的前提下尽量装。
5. 执行本轮 prefill 或 decode。
6. 更新仍未完成的请求，再进入下一轮。

如果把它写成极简“流程图文字版”，就是：

请求入队 → 估 token 与显存 → 检查等待上限 → 选择本轮 batch → 执行 prefill/decode → 更新队列

真实工程例子：一个在线代码助手，前端用户有两类请求。第一类是“解释一小段报错”，prompt 很短；第二类是“总结整个仓库结构”，prompt 很长。如果系统固定按 batch=8 收集请求，长 prompt 容易拖慢整轮 prefill，还会吃掉大量 KV cache。更合理的做法是：短请求高并发时多合批，长请求则受 `max_num_tokens` 单独约束，必要时做 chunked prefill，把超长输入分块喂给模型，而不是强迫所有请求共享同一种 batch 规则。

---

## 代码实现

代码实现的关键，不是“写一个队列”这么简单，而是把调度约束落到可执行参数上。一个最常见的配置组合如下：

| 参数 | 作用 | 误配风险 |
| --- | --- | --- |
| `max_batch_size=8` | 请求数上限 | 只控请求数，可能忽略长 prompt |
| `max_num_tokens=2048` | token 总量上限 | 设太大容易顶爆显存 |
| `max_queue_delay=5ms` | 最大凑批等待时间 | 设太大 TTFT/P99 会变差 |
| `preferred_batch_size=[2,4,8]` | 倾向选择的批大小 | 若硬件/profile 不匹配，收益不稳定 |

下面给一段可以运行的 Python 玩具实现。它不依赖 GPU，只模拟“按请求数、token 数、等待时间”联合选批的逻辑：

```python
from dataclasses import dataclass

@dataclass
class Request:
    req_id: str
    tokens: int
    wait_ms: int

def select_batch(queue, max_batch_size, max_num_tokens, max_queue_delay):
    """
    简化策略：
    1. 队首如果快超时，优先尽快发车
    2. 依次装入请求，但同时受请求数和 token budget 限制
    """
    batch = []
    used_tokens = 0

    force_dispatch = bool(queue and queue[0].wait_ms >= max_queue_delay)

    for req in queue:
        if len(batch) >= max_batch_size:
            break
        if used_tokens + req.tokens > max_num_tokens:
            # 如果已经有请求了，就先发；如果一个大请求单独就超预算，真实系统需要降级或分块
            if batch or force_dispatch:
                break
            continue
        batch.append(req)
        used_tokens += req.tokens

        # 如果是强制发车模式，允许较早结束，避免继续等更久
        if force_dispatch and len(batch) >= 1:
            break

    return batch, used_tokens

queue = [
    Request("a", tokens=200, wait_ms=6),
    Request("b", tokens=300, wait_ms=4),
    Request("c", tokens=1800, wait_ms=3),
    Request("d", tokens=100, wait_ms=2),
]

batch, used = select_batch(
    queue,
    max_batch_size=4,
    max_num_tokens=600,
    max_queue_delay=5,
)

assert [r.req_id for r in batch] == ["a"]
assert used == 200

queue2 = [
    Request("x", tokens=200, wait_ms=1),
    Request("y", tokens=250, wait_ms=1),
    Request("z", tokens=100, wait_ms=1),
]

batch2, used2 = select_batch(
    queue2,
    max_batch_size=4,
    max_num_tokens=600,
    max_queue_delay=5,
)

assert [r.req_id for r in batch2] == ["x", "y", "z"]
assert used2 == 550
```

这段代码故意保留了一个工程上很重要的细节：不能只按请求数截断。因为真实系统里，3 个短请求可能比 1 个超长请求更容易安全装进去。

再给一个调度循环骨架，帮助把思路和工程系统对上：

```python
while True:
    incoming = poll_new_requests()
    queue.extend(incoming)

    estimate_tokens_and_kv(queue)

    batch = pick_requests(
        queue=queue,
        max_batch_size=8,
        max_num_tokens=2048,
        max_queue_delay_ms=5,
        memory_headroom_ratio=0.1,
    )

    if not batch:
        continue

    run_prefill_or_decode(batch)
    update_finished_and_active_requests(queue, batch)
```

这类逻辑在不同框架里的落地点大致如下：

- Triton `dynamic batching`
  更偏传统无状态模型，把多个请求在服务端聚合。优点是配置明确，缺点是对有状态 decode 场景表达力有限。
- TensorRT-LLM `scheduler`
  更接近 LLM 运行时，会把 `max batch size`、`max num tokens` 之类的容量约束纳入调度。
- vLLM `continuous batching`
  核心思路是按步重算活跃请求集合，让 decode 阶段持续插入和退出请求，减少“整批等整批”的空转。

对新手来说，最重要的理解不是 API 名称，而是这三者都在回答同一件事：本轮哪些请求能一起跑，且不会把延迟和显存打穿。

---

## 工程权衡与常见坑

动态 batching 最常见的失败，不是“完全没提速”，而是 TTFT 和 P99 被排队时间拉爆。因为用户最先感知到的不是系统平均吞吐，而是“为什么我点了之后半天没出字”。

常见坑可以直接列成表：

| 坑点 | 后果 | 规避 |
| --- | --- | --- |
| 只看请求数，不看 token 数 | 长 prompt 混入后整批变慢 | 同时限制 `max_num_tokens` |
| `max_queue_delay` 设太大 | TTFT、P99 恶化 | 先按 SLO 反推等待上限 |
| 固定 padding 太多 | 算力浪费，吞吐虚高 | 用 ragged batching 或按长度分桶 |
| `preferred_batch_size` 乱配 | 调度抖动，收益不稳定 | 只在 profile 明显更优时启用 |
| 忽略 KV cache 余量 | OOM 或频繁回退 | 把显存 headroom 纳入调度 |
| 把无状态 batching 套到有状态模型 | 行为异常或性能不稳 | 区分 prefill 和 decode 调度 |
| 慢请求与快请求混跑 | 尾延迟变差 | 做长度分层、chunked prefill 或单独路由 |

指标也不能只盯一个。至少要一起看：

| 指标 | 为什么必须看 |
| --- | --- |
| `TTFT` | 直接对应在线交互体验 |
| `P50 / P95 / P99 latency` | 区分平均表现和尾部风险 |
| `TPS` | 衡量总体吞吐 |
| `GPU occupancy` | 判断 GPU 是否真的被喂饱 |
| `KV cache 使用率` | 判断显存是否接近风险边界 |

调参优先级建议也要明确，不然很容易反过来：

1. 先控延迟上限，特别是 `TTFT` 和 `P99`。
2. 再提吞吐，比如逐步放宽 `max_batch_size` 或 `max_num_tokens`。
3. 最后压显存风险，给 KV cache 留安全余量。

这里给一个真实工程例子。一个在线问答系统白天负载高峰时，短问答大量涌入，团队把 `max_batch_size` 从 4 提到 16，离线压测 TPS 很漂亮；但线上出现大量长上下文总结请求时，`max_num_tokens` 没跟着收紧，结果每一轮 prefill 都很重，TTFT 明显上升，部分批次还因为 KV cache 太高触发回退。最后恢复稳定的办法不是简单把 batch 再砍回去，而是三件事一起做：
- 给长请求单独路由
- 收紧 `max_queue_delay`
- 把 token budget 设成比显存理论上限更保守的值

工程上要接受一个现实：最优批大小不是常数，而是一个随流量、长度分布、显存状态变化的运行时决策。

---

## 替代方案与适用边界

动态 batching 不是唯一方案。很多时候，它甚至不是最便宜的方案。是否采用它，取决于你的请求长度分布、SLO 严格程度，以及系统是否容易受显存约束。

先做横向比较：

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 静态 batch | 实现简单，行为稳定 | 对波动负载适应差 | 离线任务、同质请求 |
| 动态 batch | 吞吐和资源利用率更灵活 | 调参与观测更复杂 | 在线混合负载 |
| 连续 batching | 适合 LLM decode，减少空转 | 实现复杂，对调度器要求高 | 大模型在线生成 |
| 微批 `micro-batching` | 降低单次峰值压力 | 吞吐未必最优 | 算子链长、显存紧张 |
| 异步排队 | 解耦前后端，削峰简单 | 不直接解决单轮算力利用率 | 流量波动明显 |
| chunked prefill | 长上下文更友好 | 调度逻辑更复杂 | 长 prompt + 在线混合请求 |

再给一个边界判断表：

| 判断问题 | 倾向 |
| --- | --- |
| 请求长度波动大吗？ | 大则更需要动态调度 |
| 有严格 TTFT / P99 SLO 吗？ | 有则必须严格限等待时间 |
| 显存是主要瓶颈吗？ | 是则要优先管 token 和 KV |
| 是有状态推理吗？ | 是则优先考虑连续 batching |
| 流量很低吗？ | 低则动态 batching 收益可能很小 |

可以把“何时不用动态 batching”说得更直白：

1. 如果请求非常同质，固定长度、固定到达节奏，静态 batch 通常更简单。
2. 如果系统流量低到几乎凑不出批，动态 batching 只会增加等待。
3. 如果业务对 TTFT 极端敏感，比如交互式补全，过于激进的凑批会伤害核心体验。
4. 如果模型或服务强依赖请求状态，能否自由重排请求本身就成问题。

最后给一个玩具对比，帮助初学者建立直觉：
- 固定长度 OCR 小图分类任务：静态 batch 往往够用，因为请求差异很小。
- 在线 LLM 聊天服务：更适合动态或连续 batching，因为输入长度和生成长度差异很大。
- 超长上下文总结：常常需要 chunked prefill，而不是单纯追求更大 batch。

所以，自适应 batching 的适用边界不是“线上就一定需要”，而是“负载异质性足够强，且系统确实能从运行时调度中获益”。

---

## 参考资料

1. [NVIDIA Triton Inference Server: Dynamic Batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)  
用于支撑动态 batching 的基本机制、`max_queue_delay` 和 `preferred_batch_size` 等配置概念，属于官方文档。

2. [NVIDIA Triton Inference Server: Ragged Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2530/user-guide/docs/user_guide/ragged_batching.html)  
用于支撑“不要强行固定 padding、长度不齐时可减少浪费”的工程建议，属于官方文档。

3. [TensorRT-LLM: Scheduler](https://nvidia.github.io/TensorRT-LLM/torch/scheduler.html)  
用于支撑 LLM 场景下 scheduler 如何管理请求装载、容量限制和运行时调度，属于官方实现文档。

4. [TensorRT-LLM: Tuning Max Batch Size and Max Num Tokens](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html)  
用于支撑本文关于 `max_batch_size`、`max_num_tokens` 联合调优的建议，属于官方性能调优文档。

5. [vLLM: Performance and Tuning](https://docs.vllm.ai/en/v0.5.2/models/performance.html)  
用于支撑 continuous batching、吞吐与延迟权衡、在线服务调优实践，属于官方文档。

6. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://huggingface.co/papers/2309.06180)  
用于支撑 KV cache、PagedAttention 与 LLM 服务显存管理的核心背景，属于论文资料。
