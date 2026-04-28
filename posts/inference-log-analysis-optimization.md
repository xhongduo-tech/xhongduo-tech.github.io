## 核心结论

推理日志分析的目标，不是把“某个请求很慢”记下来，而是把一次慢请求拆成几个可行动的阶段：`queue wait`、`prefill`、`first token`、`decode`。`queue wait` 就是请求进入服务后还没开始算时的等待时间；`prefill` 是模型先读完整个输入并建立上下文状态的阶段；`first token` 是开始产出第一个 token 的那段时间；`decode` 是后续逐个 token 生成的阶段。术语第一次看起来多，但本质上是在回答一个朴素问题：这 2 秒到底耗在哪。

核心公式是：

$$
L_{e2e} = L_q + L_{pre} + L_{ft} + L_{dec}
$$

其中，$L_{e2e}$ 是端到端总耗时，$L_q$ 是排队时间，$L_{pre}$ 是前处理或 prefill 时间，$L_{ft}$ 是首 token 时间，$L_{dec}$ 是后续解码时间。

真正有用的分析对象不是平均值，而是尾部请求，尤其是 `P95` 和 `P99`。`P99` 的白话解释是：100 个请求里最慢的那 1 个附近的表现。如果 `P99` 恶化，你必须先回答“慢在哪个阶段”，再决定优化方案。否则调优方向很容易错。

一个最小对比例子就能说明问题。同样是总耗时 2 秒：

| 请求 | queue wait | prefill | first token | decode | e2e | 更可能的问题 |
|---|---:|---:|---:|---:|---:|---|
| A | 1600 ms | 100 ms | 50 ms | 250 ms | 2000 ms | 流量突增、批处理拥塞、限流不足 |
| B | 100 ms | 100 ms | 200 ms | 1600 ms | 2000 ms | KV cache 失效、并发调度退化、生成阶段争用 |

A 和 B 的总耗时一样，但根因完全不同。A 更像系统来不及接单，B 更像模型已经开始工作，但生成阶段变慢。只看“一条 2000 ms 日志”无法区分这两类问题。

阶段耗时占比也必须看，因为占比能直接告诉你慢请求的主矛盾：

| 阶段 | 含义 | 典型占比异常时的判断 |
|---|---|---|
| `queue wait` | 请求排队等待执行 | 高占比通常先查负载、调度、batch 抖动 |
| `prefill` | 输入处理与上下文建立 | 高占比通常先查长 prompt、模板膨胀、前处理变重 |
| `first token` | 首 token 生成或 flush | 高占比通常先查首 token 调度、网络 flush、服务端 buffering |
| `decode` | 连续生成 token | 高占比通常先查 KV cache、并发争用、输出过长 |

结论可以压缩成一句话：推理日志分析的价值，不在“记录慢”，而在“拆开慢”；只有把日志与 `trace_id`、`span_id` 关联，才能区分是负载突增、batch 波动、缓存失效，还是网络抖动导致 `P99` 恶化。

---

## 问题定义与边界

本文讨论的是推理服务的慢请求定位，不是泛化的日志管理，也不是训练监控。重点对象是在线推理链路中的单请求，把请求级结构化日志和链路追踪拼起来，定位哪一段出了问题。

这里有两个边界必须先讲清楚。

第一，`TTFT` 不等于 `e2e`。`TTFT` 是 `time to first token`，白话就是“用户等到第一个 token 出现要多久”；`e2e` 是从请求进入到最终输出结束的总时间。二者关系是：

$$
TTFT = L_q + L_{pre} + L_{ft}
$$

以及：

$$
L_{e2e} = TTFT + L_{dec}
$$

第二，服务端时间不等于用户体感时间。服务端记完最后一个 token，不代表客户端已经立刻看到。代理缓冲、SSE flush、网络抖动、客户端渲染都可能让“看起来更慢”。因此要区分“服务端阶段耗时”和“端到端链路耗时”。

一条只有“耗时 2000 ms”的日志，几乎没有定位价值。因为它没有回答如下问题：

| 观察方式 | 能回答什么 | 不能回答什么 | 适用场景 |
|---|---|---|---|
| 只看总耗时 | 请求慢不慢 | 慢在哪个阶段 | 粗粒度告警 |
| 看阶段耗时 | 排队慢、prefill 慢还是 decode 慢 | 是哪一层、哪个实例、哪个 batch 触发 | 第一层定位 |
| 看端到端 trace | 哪个服务、哪个 span、哪段链路慢 | 大规模趋势的统计摘要 | 深入根因分析 |

所以，一条真正有用的推理请求日志，至少应该像这样：

```json
{"request_id":"r-42","trace_id":"t-9","queue_ms":1200,"prefill_ms":80,"ft_ms":20,"decode_ms":700,"ttft_ms":1300,"e2e_ms":2000}
```

有了这条日志，你已经可以先判断：这不是“模型纯算慢”，而更可能是“请求在前面堆住了”。如果再把它和 trace 中的网关、调度器、推理实例 span 串起来，问题范围会继续收缩。

---

## 核心机制与推导

先统一记号。单请求层面：

- $L_q$：队列等待时间，请求进入系统后尚未获得执行资格的时间。
- $L_{pre}$：prefill 时间，模型处理输入 token、建立 KV cache 等上下文状态的时间。KV cache 可以理解为“把已经算过的上下文缓存住，后续生成时不用重复算”。
- $L_{ft}$：首 token 时间，模型从完成 prefill 到第一个 token 可被发送的时间。
- $L_{dec}$：decode 时间，后续 token 连续生成的总时间。

于是有两条基础关系：

$$
TTFT = L_q + L_{pre} + L_{ft}
$$

$$
L_{e2e} = L_q + L_{pre} + L_{ft} + L_{dec}
$$

玩具例子如下。某请求被拆成：

- $L_q = 120$ ms
- $L_{pre} = 80$ ms
- $L_{ft} = 40$ ms
- $L_{dec} = 760$ ms

则：

- $TTFT = 120 + 80 + 40 = 240$ ms
- $L_{e2e} = 240 + 760 = 1000$ ms

这类拆分有两个直接价值。第一，可以从总耗时回到阶段耗时。第二，可以把阶段耗时和上下文元数据联动分析，比如 batch size、prompt token 数、输出 token 数、cache hit rate、实例 ID、region、模型版本。

但这里有一个常见误区必须单独说明：分位数不能直接相加。也就是说，严格来说：

$$
P99(L_q) + P99(L_{pre}) + P99(L_{ft}) + P99(L_{dec}) \neq P99(L_{e2e})
$$

原因不复杂。`P99` 是分布上的位置，不是普通加法下的线性量。不同阶段的尾部请求不一定发生在同一批请求上，还可能存在相关性。比如排队特别久的请求，decode 不一定也特别久；某些长输出请求 decode 很慢，但并不排队。因此，把各阶段 `P99` 直接相加，只能当近似参考，不能当严格真值。

下面这张表更适合用来指导排查：

| 阶段名 | 含义 | 常见根因 | 可观测字段 |
|---|---|---|---|
| `queue wait` | 进入系统后等待执行 | 流量突发、批处理拥塞、调度不均、限流策略过松 | `queue_ms`、`queue_len`、`batch_size` |
| `prefill` | 输入处理与上下文建立 | prompt 太长、模板膨胀、输入预处理过重 | `prefill_ms`、`prompt_tokens` |
| `first token` | 到首 token 为止的时间 | 首 token 调度延迟、网络 flush、代理缓冲 | `ft_ms`、`flush_ms` |
| `decode` | 后续输出生成 | KV cache 命中率下降、并发过高、GPU/CPU 争用 | `decode_ms`、`output_tokens`、`cache_hit` |

真实工程里，最常见的情况不是“所有请求都整体变慢”，而是“平均值变化很小，但少量请求尾部飙高”。比如平均耗时仍是 180 ms，但 `P99` 从 600 ms 升到 1000 ms。这时只看平均值会得出错误结论：系统看起来还行。但对线上用户来说，最差那部分请求已经明显退化。

---

## 代码实现

实现重点不是“打印一行日志”，而是让 `request_id` 贯穿请求生命周期，并在关键阶段持续打点。下面给一个最小可运行的 Python 版本，用来模拟“接收请求 -> 排队 -> prefill -> 首 token -> decode -> 输出结构化日志”的过程。

```python
import json
import time
import uuid


def now_ms() -> int:
    return int(time.perf_counter() * 1000)


def handle_request(prompt_tokens: int, output_tokens: int, queue_ms: int, prefill_ms: int, ft_ms: int, decode_ms: int):
    request_id = f"r-{uuid.uuid4().hex[:8]}"
    trace_id = f"t-{uuid.uuid4().hex[:8]}"

    t0 = now_ms()

    queue_enter = t0
    queue_exit = queue_enter + queue_ms

    prefill_start = queue_exit
    prefill_end = prefill_start + prefill_ms

    first_token_at = prefill_end + ft_ms
    decode_end = first_token_at + decode_ms

    ttft_ms = first_token_at - t0
    e2e_ms = decode_end - t0

    log = {
        "request_id": request_id,
        "trace_id": trace_id,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "queue_ms": queue_ms,
        "prefill_ms": prefill_ms,
        "ft_ms": ft_ms,
        "decode_ms": decode_ms,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
    }

    assert ttft_ms == queue_ms + prefill_ms + ft_ms
    assert e2e_ms == queue_ms + prefill_ms + ft_ms + decode_ms
    assert e2e_ms == ttft_ms + decode_ms

    return log


toy = handle_request(
    prompt_tokens=512,
    output_tokens=128,
    queue_ms=120,
    prefill_ms=80,
    ft_ms=40,
    decode_ms=760,
)

print(json.dumps(toy, ensure_ascii=False))
assert toy["ttft_ms"] == 240
assert toy["e2e_ms"] == 1000
```

这段代码的核心不是睡眠模拟，而是定义“阶段边界”与“结构化字段”。结构化日志的意思是字段固定、可查询、可聚合的日志，而不是随手拼接一段字符串。一个典型输出会长这样：

```json
{"request_id":"r-1","trace_id":"t-1","queue_ms":120,"prefill_ms":80,"ft_ms":40,"decode_ms":760,"ttft_ms":240,"e2e_ms":1000}
```

字段设计可以按下面的方式落地：

| 字段 | 从哪里采集 | 写入层 | 用途 |
|---|---|---|---|
| `request_id` | 网关入口生成 | 网关、中间层、推理层都透传 | 单请求检索 |
| `trace_id` | Trace 上下文生成 | 所有服务统一透传 | 日志与追踪关联 |
| `queue_ms` | 调度器或推理队列 | 调度层/推理层 | 判断是否拥塞 |
| `prefill_ms` | 模型执行打点 | 推理层 | 判断输入处理是否变重 |
| `ft_ms` | 首 token 产出时刻 | 推理层/流式发送层 | 判断 TTFT 异常来源 |
| `decode_ms` | 输出结束时刻 | 推理层 | 判断生成阶段退化 |
| `batch_size` | 动态批处理器 | 调度层 | 分析 batch 波动 |
| `cache_hit` | KV cache 统计 | 推理层 | 分析缓存失效 |

真实工程例子可以这样理解。假设线上 LLM 网关已经把 `request_id` 和 `trace_id` 贯穿起来。某天 `P99` 从 900 ms 升到 2.8 s，但平均值只涨了 80 ms。你先在日志系统中按 `trace_id` 聚合，再按阶段耗时分桶，发现有两群异常请求：

- 第一群：`queue_ms` 明显升高，`decode_ms` 正常。结论更偏向流量突发、batch 抖动、限流失效。
- 第二群：`TTFT` 正常，但 `decode_ms` 拉长。结论更偏向 KV cache 命中率下降、并发过高、实例资源争用。

这一步如果没有结构化日志，只靠 access log，很难做出来。

---

## 工程权衡与常见坑

工程上最常见的误区，是把“观测到了慢”误当成“定位了慢”。这两者不是一回事。只看平均值、不看 `P95/P99`，或者只看总耗时、不看阶段耗时，通常只够做报警，不够做优化。

另一个高频错误，是把 `TTFT` 和 `e2e` 混在一起。用户抱怨“首 token 太慢”，你却盯着总耗时优化 decode；或者用户抱怨“整段输出拖得太久”，你却只看 TTFT。方向一错，后面所有优化都可能无效。

更麻烦的是观测链路本身也会失真。如果没有统一 `trace_id`、`span_id`、批处理大小、cache 命中率、排队长度等元数据，结论经常站不住。比如你看到某实例 `decode_ms` 高，不一定真是模型慢，也可能是这个实例接到的长输出请求更多。

下面这张表是最值得长期贴在团队 wiki 里的：

| 常见坑 | 问题表现 | 规避方式 |
|---|---|---|
| 只看平均值 | 平均正常，但线上仍频繁超时 | 固定看 `P95/P99`，并按阶段拆分 |
| 没有 trace 关联 | 知道慢，但不知道慢在哪个服务或实例 | 统一透传 `trace_id`、`span_id` |
| 直接相加分位数 | 误以为各阶段 `P99` 之和就是总 `P99` | 把分位数相加仅作近似，回到请求级样本验证 |
| 忽略网络 flush | 服务端结束了，客户端还没看到 | 分离服务端耗时与端到端耗时 |
| 只看单机不看批处理和缓存 | 误判模型性能问题 | 同时记录 `batch_size`、`queue_len`、`cache_hit` |
| 只看单次异常样本 | 个例太多，无法总结模式 | 用分布、分桶和同类请求对比 |

遇到 `P99` 升高时，经验上可以按阶段先分流排查：

- 如果 `queue wait` 上升，优先查流量突发、batch 抖动、限流策略、实例数是否不足。
- 如果 `prefill` 上升，优先查 prompt 是否变长、模板是否膨胀、前处理是否引入额外开销。
- 如果 `decode` 上升，优先查 KV cache、并发调度、输出长度分布、网络 flush、实例资源争用。

这个顺序的价值在于，它先把问题从“系统慢”变成“哪一段慢”，再把“哪一段慢”映射到少数几类高概率根因。

---

## 替代方案与适用边界

不是所有场景都需要最重的全链路观测。观测系统本身也有成本，字段设计、采样、存储、查询都要付钱。因此方案要按问题粒度选择。

如果你只想知道“有没有慢请求”，简单 access log 就够了。如果你要知道“慢在哪个阶段”，必须上结构化日志。如果你要知道“是谁触发的慢、慢发生在哪个服务、哪个实例、哪个 batch”，那就需要 trace。再往上，如果要做跨天趋势分析、容量规划、回归检测，才需要指标系统和离线聚合。

一个常见的演进路径如下：

| 问题范围 | 需要的信息 | 推荐方案 |
|---|---|---|
| 只关心是否慢 | 总耗时、状态码 | access log |
| 要知道慢在哪段 | 阶段耗时、请求 ID | 结构化 JSON 日志 |
| 要知道是谁触发的慢 | `trace_id`、`span_id`、服务拓扑 | 全链路 trace |
| 要做趋势和容量规划 | 分位数、分桶、聚合统计 | 指标系统 + 离线分析 |

真实场景里，一个早期项目通常只有 Nginx access log，只能看到“请求慢”。升级到带 `request_id` 的 JSON 日志后，能回答“慢在排队还是慢在生成”。再接入 trace 后，才能进一步判断：是某个实例异常、某个 batch 特别大，还是某类长 prompt 在特定时段造成 `P99` 退化。

这就是适用边界。不要为了一个偶发 2 秒请求，一开始就上最重的平台；但也不要用“总耗时平均值”去处理本质上需要阶段拆分和链路关联的问题。轻量方案适合快速接入，重方案适合复杂定位，关键是问题和工具要匹配。

---

## 参考资料

| 类别 | 作用 |
|---|---|
| 指标与推理框架 | 定义 `queue time`、`TTFT`、`prefill`、`decode` 等阶段术语 |
| 日志与追踪规范 | 说明如何用 `TraceId`、`SpanId`、上下文传播把日志和 trace 绑定 |

1. [NVIDIA Triton Inference Server Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2600/user-guide/docs/user_guide/metrics.html)
2. [vLLM Metrics and Monitoring Design](https://docs.vllm.ai/en/stable/design/metrics.html)
3. [OpenTelemetry Logs Specification](https://opentelemetry.io/docs/specs/otel/logs/)
4. [OpenTelemetry Context Propagation](https://opentelemetry.io/docs/concepts/context-propagation/)
5. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://huggingface.co/papers/2309.06180)
6. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://systems-reading.github.io/papers/2023-04-12/)

这些资料分别支持两条主线：Triton 和 vLLM 用于统一推理阶段术语与指标口径，OpenTelemetry 用于把日志、`trace_id`、`span_id` 和跨服务上下文传播串起来。
