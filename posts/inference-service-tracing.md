## 核心结论

推理服务链路追踪，核心不是“多打一份日志”，而是用同一个 `trace_id` 把一次请求经过的所有阶段串成一条可计算的时间线。`trace_id` 可以理解为“这一次请求的统一编号”，后面所有服务都靠它把局部记录拼回同一个故事。

对推理系统来说，真正难排查的问题通常不是“请求失败了”，而是“请求成功了，但偶尔非常慢”。这种慢常常不是单点故障，而是多段耗时叠加后被放大：某个阶段排队，多一次重试，再叠加一次网络抖动，P95 或 P99 就会明显上升。单看总耗时，你只能知道“这次慢了”；看 trace，才能知道“慢在检索排队、模型重试，还是缓存未命中”。

新手可以把它理解成接力赛。日志像某个接力员的一张照片，只能看到一个局部瞬间；trace 像整场比赛的时间线，能看到每一棒何时开始、何时结束、谁把棒交给了谁、哪一棒掉速了。

一个最小数值例子很直接。假设链路是 `gateway -> scheduler -> retrieval -> model -> cache`，对应耗时分别是 6ms、4ms、18ms、90ms、2ms，那么端到端延迟近似为：

$$
L_{e2e} = \sum l_i + \sum w_j = 6 + 4 + 18 + 90 + 2 = 120\text{ms}
$$

如果模型第一次调用超时，多等了 50ms，又重试了一次并额外算了 90ms，那么新的总耗时大约变成 260ms。根因不是“系统整体慢”，而是“检索或模型链路中的额外等待与重试放大了尾延迟”。

| 对比项 | 日志 | Trace |
|---|---|---|
| 观察对象 | 单个服务的局部事件 | 一次请求的全链路过程 |
| 关联方式 | 靠关键字、人肉拼接 | 靠 `trace_id` 自动关联 |
| 能否看调用关系 | 通常较弱 | 明确的父子关系 |
| 能否拆解总耗时 | 很难 | 可以拆到每个 `span` |
| 适合定位什么 | 单点错误、业务事件 | 慢请求、重试、排队、关键路径 |

---

## 问题定义与边界

推理服务链路追踪，是指一次请求从入口到输出的整个过程中，所有关键阶段都携带同一份上下文，并记录每一段的开始时间、结束时间、耗时、错误状态、重试次数与等待时间。这里的上下文传播通常叫 `context propagation`，白话讲就是“把请求身份和调用关系一层层往下游带过去，别在中途丢了”。

几个基础术语先定清楚：

| 术语 | 含义 | 白话解释 |
|---|---|---|
| `trace_id` | 一次链路的全局标识 | 整条请求链共用的编号 |
| `span` | 链路中的一个阶段 | 某一步具体干了什么、花了多久 |
| `context propagation` | 上下文传播 | 把链路编号继续传给下游 |
| `attempt_id` | 某次重试的编号 | 第几次尝试，不和第一次混在一起 |

单段耗时通常定义为：

$$
l_i = end_i - start_i
$$

这件事解决的是“慢点定位”和“放大因素识别”。它擅长回答下面这类问题：

| 能解决 | 不能直接解决 |
|---|---|
| 哪一段最慢 | 模型准确率为什么下降 |
| 是否发生排队 | 某个参数为什么影响召回质量 |
| 是否发生重试 | 离线训练数据是否有偏差 |
| 总耗时是如何组成的 | 模型权重本身是否最优 |
| 哪条路径是关键路径 | 业务策略本身是否合理 |

边界必须明确。trace 不是业务指标，不替代 QPS、成功率、token 用量这些统计指标；trace 也不是资源监控，不替代 CPU、显存、磁盘、网卡这些基础设施数据；它更不是完整的根因分析系统，最终定位瓶颈时，往往还要结合日志、指标、profile 和压测结果。

举个边界例子。如果模型回答质量下降，trace 不能直接告诉你“为什么答案错了”。但它可以告诉你，这次请求是否因为超时触发了降级、是否跳过了 rerank、是否命中了低质量缓存，从而帮助你判断输出链路是否发生了变化。

---

## 核心机制与推导

链路延迟本质上是“计算时间 + 各类等待时间”的组合。对顺序执行的链路，可以先用最基础的工程公式描述：

$$
L_{e2e} = \sum l_i + \sum w_j
$$

其中，$l_i$ 是每一段实际执行耗时，$w_j$ 是额外等待，比如排队、网络传输、锁等待、重试间隔。这个写法不是学术定理，而是工程上非常有用的统一记号。

顺序链路最好理解。比如：

- 网关鉴权 6ms
- 调度器分发 4ms
- 检索 18ms
- 模型推理 90ms
- 缓存写回 2ms

那么总耗时就是 120ms。这个玩具例子说明，端到端延迟不是一个神秘数字，而是可分解的。

并行分支要看关键路径，而不是把所有分支直接相加。假设 retrieval 和 rerank 并行，retrieval 花 40ms，rerank 花 25ms，二者之后再进入模型 90ms，那么并行段耗时是：

$$
L = \max(path_1, path_2, \dots)
$$

也就是 `max(40, 25) = 40ms`，不是 65ms。最后端到端就是“并行段关键路径 + 后续顺序段”。

重试会放大尾延迟，因为它把“一次调用”变成了“多次下游调用”。如果失败后又重试了 $m$ 次，那么总调用次数可以记为：

$$
N = 1 + m
$$

问题不只是多算一次时间，更严重的是会额外占用连接、线程、GPU 槽位和队列资源，进一步影响别的请求。也就是说，重试不仅放大本请求，还可能制造新的排队。

看一个数值版重试例子：

- 原始链路：120ms
- 第一次模型调用超时等待：50ms
- 第二次模型重新计算：90ms

则总耗时大约变成：

$$
120 + 50 + 90 = 260\text{ms}
$$

trace 的价值就在于，它不是只给你一个 260ms，而是把它拆成若干可解释的片段：第一次模型调用起了一个 `span`，状态是 `timeout`；第二次模型调用起了另一个 `span`，属性里带 `attempt_id=2`；这两个 `span` 共同挂在同一个上游请求下面。于是你能明确判断：慢不是“模型天然很慢”，而是“超时 + 重试”造成的。

真实工程里，典型链路可能是：

`gateway -> auth -> scheduler -> vector retrieval -> rerank -> model -> cache`

某次慢请求的 trace 显示：

- `scheduler.queue_wait = 12ms`
- `retrieval.shard_wait = 140ms`
- `model.attempt_1 = timeout`
- `model.attempt_2 = 93ms`

这种情况下，根因不是“模型推理本身太慢”，而是检索分片热点导致上游已变慢，模型又因为超时策略触发重试，最后把慢请求进一步拉长。单看模型服务日志，可能只会看到一次超时和一次成功；只有 trace 才能把这些因果关系串起来。

下面用一个简化时间线表示三种常见结构：

```text
顺序链路
gateway [----]
scheduler    [--]
retrieval      [------]
model                [-----------]
cache                            [-]

并行分支
scheduler [--]
retrieval    [------]
rerank       [---]
model               [-----------]

重试链路
model attempt1 [----- timeout -----]
model attempt2                    [--------]
```

再看 `span` 字段设计，很多定位能力其实来自字段是否完整，而不是“系统是否接了 tracing SDK”。

| 字段 | 用途 | 典型示例 |
|---|---|---|
| `trace_id` | 关联整条请求 | `7b3f...` |
| `span_id` | 标识当前阶段 | `a912...` |
| `parent_span_id` | 还原调用关系 | 指向上游阶段 |
| `service.name` | 知道是哪个服务 | `retrieval-service` |
| `operation` | 知道具体动作 | `vector_search` |
| `start_time/end_time` | 计算耗时 | 时间戳 |
| `attempt_id` | 区分重试 | `1`、`2` |
| `queue_wait_ms` | 区分排队 | `140` |
| `error/timeout` | 标识异常类型 | `timeout=true` |

---

## 代码实现

实现重点不是“在哪一行打点”，而是“让上下文贯穿全链路，并把关键等待拆出来记录”。入口需要注入或创建 `traceparent`。`traceparent` 是 W3C Trace Context 规范中的标准请求头，白话讲就是“跨服务传递 trace 信息的通用格式”。

一个简单的代码结构可以是：

```text
middleware/
  - inject trace context at ingress
scheduler/
  - create child span for queue wait and dispatch
retrieval/
  - record shard wait, network, and cache hit/miss
model/
  - record inference time and retry attempts
```

下面给一个可运行的 Python 玩具实现。它不是完整 OpenTelemetry SDK，但足够说明 root span、child span、`traceparent`、重试与错误标记这几个核心点。

```python
import time
import uuid

class Span:
    def __init__(self, name, trace_id, parent_id=None, attrs=None):
        self.name = name
        self.trace_id = trace_id
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_id = parent_id
        self.attrs = attrs or {}
        self.start = time.time()
        self.end = None
        self.status = "ok"

    def set_attr(self, key, value):
        self.attrs[key] = value

    def mark_error(self, reason):
        self.status = "error"
        self.attrs["error"] = reason

    def finish(self):
        self.end = time.time()

    @property
    def duration_ms(self):
        assert self.end is not None
        return int((self.end - self.start) * 1000)

def make_traceparent(trace_id, span_id):
    return f"00-{trace_id}-{span_id}-01"

def parse_traceparent(header):
    _, trace_id, parent_span_id, _ = header.split("-")
    return trace_id, parent_span_id

def start_root_span(name):
    trace_id = uuid.uuid4().hex
    span = Span(name=name, trace_id=trace_id)
    return span

def start_child_span(name, traceparent, attrs=None):
    trace_id, parent_span_id = parse_traceparent(traceparent)
    return Span(name=name, trace_id=trace_id, parent_id=parent_span_id, attrs=attrs)

def model_call(traceparent, should_timeout_first=True):
    spans = []

    attempt1 = start_child_span("model.infer", traceparent, {"attempt_id": 1, "retry_count": 0})
    time.sleep(0.01)
    if should_timeout_first:
        attempt1.mark_error("timeout")
        attempt1.set_attr("timeout", True)
    attempt1.finish()
    spans.append(attempt1)

    if should_timeout_first:
        attempt2 = start_child_span("model.infer", traceparent, {"attempt_id": 2, "retry_count": 1})
        time.sleep(0.01)
        attempt2.finish()
        spans.append(attempt2)

    return spans

def handle_request():
    root = start_root_span("gateway.request")
    traceparent = make_traceparent(root.trace_id, root.span_id)

    scheduler = start_child_span("scheduler.dispatch", traceparent)
    scheduler.set_attr("queue_wait_ms", 12)
    time.sleep(0.005)
    scheduler.finish()

    retrieval = start_child_span("retrieval.search", traceparent)
    retrieval.set_attr("cache_hit", False)
    retrieval.set_attr("shard_wait_ms", 140)
    time.sleep(0.005)
    retrieval.finish()

    model_spans = model_call(traceparent, should_timeout_first=True)

    cache = start_child_span("cache.write", traceparent)
    time.sleep(0.002)
    cache.finish()

    root.finish()
    return root, scheduler, retrieval, model_spans, cache

root, scheduler, retrieval, model_spans, cache = handle_request()

assert len(model_spans) == 2
assert model_spans[0].attrs["attempt_id"] == 1
assert model_spans[0].attrs["timeout"] is True
assert model_spans[1].attrs["retry_count"] == 1
assert root.trace_id == scheduler.trace_id == retrieval.trace_id == cache.trace_id
assert model_spans[0].trace_id == root.trace_id
assert scheduler.duration_ms >= 0
```

这段代码体现了四个关键点：

1. 请求进入网关时创建 root span。
2. 下游阶段都从 `traceparent` 继承同一个 `trace_id`。
3. 模型重试不是覆盖第一次记录，而是新建第二个 child span。
4. `timeout`、`attempt_id`、`retry_count` 都作为属性保留，方便后续查询。

如果换成 OpenTelemetry 风格，思想也是一样的：

```python
# 伪代码
ctx = extract_traceparent_from_http_headers(request.headers) or new_context()
with tracer.start_as_current_span("gateway.request", context=ctx) as root:
    inject_traceparent_to_downstream(root.context)

    with tracer.start_as_current_span("scheduler.dispatch"):
        record_attr("queue_wait_ms", queue_wait_ms)

    with tracer.start_as_current_span("retrieval.search"):
        record_attr("network_ms", network_ms)
        record_attr("cache_hit", cache_hit)

    for attempt_id in [1, 2]:
        with tracer.start_as_current_span("model.infer") as span:
            record_attr("attempt_id", attempt_id)
            record_attr("retry_count", attempt_id - 1)
            try:
                call_model()
                break
            except TimeoutError:
                record_attr("timeout", True)
                span.set_status("error")
```

实现时，建议把排队、网络、重试等待和实际计算拆开。否则 `model.infer = 220ms` 这种粗粒度 span 没法回答“是 GPU 计算慢，还是队列太长，还是上游重试造成的”。

---

## 工程权衡与常见坑

trace 不是越多越好。采样太高会带来 CPU、内存、网络和存储开销；采样太低又会漏掉真正需要分析的慢请求。更实际的策略通常不是“全量高采样”，而是“基础低采样 + 慢请求定向采样 + 错误请求强制采样”。

采样策略可以这样理解：

| 策略 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 全量采样 | 信息最完整 | 成本最高 | 小流量或压测环境 |
| 固定低比例采样 | 成本可控 | 可能漏掉慢成功 | 常规线上 |
| 错误请求优先采样 | 易抓失败根因 | 会漏掉慢成功 | 错误排查 |
| 慢请求定向采样 | 对尾延迟最有效 | 需要阈值设计 | 推理服务线上推荐 |

一个重要规则是：慢请求定向采样通常优先于盲目的全量高采样。因为你真正关心的是 P95/P99，而不是所有正常快请求的全量细节。

常见坑最好直接列成操作表：

| 问题 | 现象 | 结果 | 规避方式 |
|---|---|---|---|
| 上下文断链 | 下游 span 找不到父节点 | 一条请求被拆成多段 | 全链路传播 `traceparent` |
| 把重试合并成一个 span | 只看到一个超长模型调用 | 无法量化重试成本 | 每次 attempt 单独记 span |
| 只采失败请求 | 大量慢成功不在样本里 | 看不到尾延迟主因 | 慢成功也要定向采样 |
| span 粒度过粗 | 只有 `model.infer=220ms` | 无法区分排队/计算/网络 | 拆分等待与执行 |
| span 粒度过细 | span 数量爆炸 | tracing 本身拖慢系统 | 只打关键边界 |
| 跨机时钟偏差 | span 顺序看起来颠倒 | 错判调用先后关系 | 做 NTP/PTP 同步并以父子关系校验 |
| 只看总耗时 | 每次都像“整体都慢” | 根因模糊 | 先看关键路径与最大 span |

新手最容易忽略的一点，是“成功但很慢”的请求往往比失败请求更值得追。因为失败通常较显眼，会触发报警；而慢成功会悄悄拉高用户体感和资源占用，长期看对系统伤害更大。

真实工程里还有一个坑：机器时钟偏差。假设检索机器和模型机器相差 20ms，图上可能会出现“模型先返回，检索后完成”的错觉。处理方式不是简单相信时间线，而是结合父子关系、单机单调时钟和同步机制去校正。

---

## 替代方案与适用边界

trace 不是唯一工具。更准确地说，它负责“把过程串起来”，而日志、指标、profile、压测分别负责提供不同层面的细节。

| 工具 | 最擅长的问题 | 不擅长的问题 | 适用边界 |
|---|---|---|---|
| Trace | 多阶段耗时拆解、关键路径、重试放大 | 单机热点函数级细节 | 跨服务、多阶段推理 |
| Logs | 单点事件、错误详情、输入输出片段 | 全链路时序关系 | 业务错误、审计 |
| Metrics | 趋势、聚合、报警 | 单次请求因果关系 | QPS、P95、错误率 |
| Profiling | CPU/内存热点、函数级瓶颈 | 跨服务调用关系 | 单机性能优化 |
| Queueing tracing | 排队时长、队列积压来源 | 业务步骤全貌 | 调度器、批处理、异步消费 |

一个简单判断规则是：如果系统存在多阶段、重试、并行分支、跨服务调用，那么 trace 的价值通常很高。反过来，如果只是一个单跳接口，例如静态页面请求或单机脚本，trace 往往不是首选，指标和日志可能已经足够。

可以把工具选择再压缩成一个工程建议表：

| 场景 | 首选工具 | 补充工具 |
|---|---|---|
| 单个数据库查询慢 | 慢查询日志 | metrics、profile |
| 单服务 CPU 飙高 | profile | metrics、logs |
| RAG 链路偶发 2 秒慢请求 | trace | metrics、logs |
| Agent 多工具编排超时 | trace | logs、queue metrics |
| 静态资源请求慢 | metrics | CDN 日志 |

所以，“trace 是否值得上”并不取决于系统是否高级，而取决于链路是否复杂。对简单单跳系统，它可能只是额外成本；对推理编排、RAG、多模型路由、Agent 工作流这类多阶段系统，它几乎是定位尾延迟的必要工具。

---

## 参考资料

标准类：

1. [W3C Trace Context](https://www.w3.org/TR/trace-context/)
2. [OpenTelemetry Traces](https://opentelemetry.io/docs/concepts/signals/traces/)
3. [OpenTelemetry Context Propagation](https://opentelemetry.io/docs/concepts/context-propagation/)

实践类：

1. [Jaeger Terminology](https://www.jaegertracing.io/docs/2.17/architecture/terminology/)
2. [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
3. [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/)

理论类：

1. [Dapper: a Large-Scale Distributed Systems Tracing Infrastructure](https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/)
2. [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)
3. [Google SRE Book](https://sre.google/sre-book/table-of-contents/)

本文中的公式是工程化归纳，目的是统一理解与落地，不是对所有系统的严格数学建模。
