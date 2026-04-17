## 核心结论

LLM 应用的可观测性，指的是把一次交互里发生的行为、性能和成本放到同一条观测链路中。白话说，不是只看“模型有没有返回结果”，而是要知道“它经过了哪些步骤、每一步花了多久、用了多少 Token、为什么失败、花了多少钱”。

如果只保留请求日志，系统只能回答“出了问题”；如果把日志、trace 和指标统一起来，系统才能回答“问题发生在第几步、影响了多少用户、是否值得优化”。对零基础到初级工程师来说，最重要的不是先上复杂平台，而是先建立一个稳定模型：每个用户请求都对应一个 trace，每个 trace 由多个 span 组成，每个 span 都带上输入摘要、状态、延迟、Token 和成本属性。

一个简单问答请求在 LangSmith 这类控制台里，通常会显示成一条链路：`prompt -> retrieval -> llm -> tool -> final_output`。你可以直接看到 retrieval 是否慢、LLM 是否输出过长、tool 是否超时，以及总成本是不是异常。这种“按单次交互下钻”的能力，是 LLM 可观测性和传统 Web 监控最大的区别。

传统监控和 LLM 可观测性的差异可以先用一张表看清：

| 维度 | 传统接口监控常见做法 | LLM 可观测性要求 |
| --- | --- | --- |
| 日志 | 记录 URL、状态码、错误栈 | 记录 prompt 摘要、响应摘要、模型名、工具结果、上下文版本 |
| 追踪 | 关注服务调用链 | 关注 prompt、检索、模型、工具、外部 API 的完整链路 |
| 指标 | QPS、平均延迟、错误率 | QPS、P99、错误率、Token、成本、缓存命中率 |
| 定位粒度 | 接口级 | 单次交互级、单个 span 级 |
| 优化目标 | 服务稳定性 | 质量、性能、费用三者同时达标 |

可以把整体流程画成一个简图：

`用户请求 -> trace_id -> [prompt span -> retrieval span -> llm span -> tool span] -> 日志/指标/告警/仪表盘`

核心结论只有一句：LLM 应用要能稳定上线，必须把“请求内容、执行链路、Token 成本、尾部延迟”放到同一个 trace 里看，否则质量、性能和费用无法归因。

---

## 问题定义与边界

LLM 可观测性关注三个维度：

1. 行为：系统做了什么。白话说，就是“这一问到底经历了哪些步骤”。
2. 性能：系统快不快、稳不稳。白话说，就是“用户会不会感觉卡”。
3. 成本：系统每次响应花了多少资源。白话说，就是“这条回答值不值这个钱”。

这里的边界非常重要。本文讨论的是“应用层”的 LLM 可观测性，不是模型训练可观测性，也不是 GPU 集群监控。我们关注的是一次线上交互，从收到用户输入开始，到输出最终结果结束，中间可能经过提示词拼装、检索、重排、模型调用、工具调用、缓存命中、后处理等步骤。

很多新手一开始只有两个东西：

- 一个 average latency 仪表盘
- 一条“本小时总 Token 数”的日志

这套数据看起来像“有监控”，但实际上不可定位。举个常见 FAQ：

“平均延迟一直是 800ms，为什么用户还在抱怨卡顿？”

原因可能是 retrieval 这个 span 经常重试，两次检索把极少数请求拖到 4 秒，但平均值被大多数快请求稀释了。又或者总 Token 没涨，但某个工具调用失败后触发 fallback，导致 LLM 重新生成了一次，成本翻倍却没法拆出来。

下面这张表可以明确边界：

| 问题范围 | 为何不可见 | 可观测性如何填补 |
| --- | --- | --- |
| 单次请求为何超时 | 只有平均延迟，没有 trace | 按 trace 查看最慢 span，定位 retrieval、tool 或 llm |
| 成本为何突然上涨 | 只有总 Token，没有分步骤数据 | 在每个 span 上记录 input/output tokens 和 cost |
| 错误为何集中爆发 | 只有错误计数，没有上下文 | 记录模型名、版本、prompt 模板、工具参数和错误类型 |
| 用户体验为何变差 | 只看平均值，不看尾部 | 增加 P95/P99 监控和最慢 trace 抽样 |
| 某次输出为何异常 | 没有请求上下文快照 | 保存 prompt 摘要、检索片段 ID、输出摘要和 trace_id |

边界再收紧一点：如果你的系统没有把工具调用、缓存命中、重试逻辑拆成 span，那么你就还没有真正具备成本归因和性能归因能力。因为你只能看到“总时长”和“总 Token”，却不知道是谁贡献了这些数字。

玩具例子可以说明这一点。假设一个问答机器人只有两步：

- step1：从知识库取 3 条文档
- step2：把文档拼进 prompt 调用模型

如果只看总延迟 900ms，你不知道问题在哪。若拆成 span 后发现 retrieval=700ms、llm=200ms，那么优化方向就不是换更快模型，而是先改检索、加缓存、减少文档数。

---

## 核心机制与推导

先定义几个最常用指标。

QPS 是每秒请求数，表示吞吐能力：

$$
QPS = \frac{\text{请求总数}}{\text{统计时间（秒）}}
$$

错误率表示失败比例：

$$
错误率 = \frac{\text{失败请求数}}{\text{总请求数}}
$$

成本最常见的近似公式是：

$$
成本 = (\text{输入 Token} \times \text{输入单价}) + (\text{输出 Token} \times \text{输出单价}) + \text{工具/外部 API 成本}
$$

P99 延迟表示最慢 1% 请求的响应时间。白话说，它反映的是“最差体验接近什么水平”，比平均值更接近用户真实感受。

LLM tracing 的核心结构也很简单：

- trace：一次完整用户请求
- span：trace 内的一个步骤

比如：

`trace(chat-123)`
`├─ span(prompt_build)`
`├─ span(retrieval)`
`├─ span(llm_call)`
`├─ span(tool_call)`
`└─ span(post_process)`

每个 span 至少要带这些属性：

| 属性 | 作用 |
| --- | --- |
| `trace_id` / `span_id` | 串起整条链路 |
| `name` | 标识步骤名 |
| `start_time` / `end_time` | 计算延迟 |
| `status` | 成功、失败、超时、重试 |
| `input_tokens` / `output_tokens` | 计算成本与长度 |
| `model` / `tool_name` | 做归因分析 |
| `metadata` | 保存 prompt 版本、文档 ID、缓存状态等 |

看一个数值例子。某路径 1 分钟收到 120 个请求，那么：

$$
QPS = 120 / 60 = 2
$$

假设 P99 延迟为 450ms，平均每条请求总共消耗 150 个 Token，单 Token 成本是 0.0003 美元，那么一分钟总成本约为：

$$
120 \times 150 \times 0.0003 = 5.4
$$

这个数值不是为了精确计费，而是为了说明：吞吐、延迟和成本是绑定关系。请求量提升时，任何一个 span 的抖动都会同时放大用户卡顿和费用风险。

再把 trace 用到定位上。假设在 LangSmith dashboard 中看到某一类问答的 P99 升高。点进最慢 trace 后，发现：

- `prompt_build`: 20ms
- `retrieval`: 320ms
- `llm_call`: 90ms
- `tool_call`: 15ms

那么结论很直接：尾部延迟主要来自 retrieval span，而不是模型本身。优化顺序应该是：

1. 检查检索是否重复请求
2. 检查是否取回过多文档
3. 检查向量库是否缺缓存或索引不合理
4. 最后才考虑更换模型

这就是“尾部指标决定用户体验”的意思。平均值告诉你系统大致健康，P99 告诉你用户什么时候会骂你。

真实工程例子是一个 RAG 系统。用户发来“怎么处理支付超时”，系统会：

- 生成查询词
- 检索知识库文档
- 调用重排器
- 组装 prompt
- 调用 LLM
- 如果命中流程类问题，再调用工单系统工具

如果 trace 完整，你就能在控制台里看到每一步的时间和 Token。若发现重排器带来 180ms 但答案质量几乎不提升，可以直接做 A/B 测试决定是否关闭。没有 trace，这种优化几乎靠猜。

---

## 代码实现

实现上不要一开始追求“平台很高级”，先把数据结构做对。最小方案是：

- 日志：记录输入摘要、输出摘要、错误信息、trace_id
- trace：把流水线拆成多个 span
- 指标：更新请求数、错误数、延迟分布、Token 和成本

下面给出一个可运行的 Python 玩具实现。它不依赖特定平台，但结构和 LangSmith、W&B Weave、自建 OpenTelemetry pipeline 的思想一致。

```python
import time
import uuid
from statistics import quantiles

class Span:
    def __init__(self, trace_id, name):
        self.trace_id = trace_id
        self.name = name
        self.start = None
        self.end = None
        self.attrs = {}
        self.status = "ok"

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def set_attr(self, key, value):
        self.attrs[key] = value

    def fail(self, reason):
        self.status = "error"
        self.attrs["error_reason"] = reason

    def __exit__(self, exc_type, exc, tb):
        self.end = time.perf_counter()
        self.attrs["latency_ms"] = round((self.end - self.start) * 1000, 2)
        if exc is not None:
            self.fail(str(exc))
        return True  # 玩具示例中吞掉异常，生产环境按需处理

class Trace:
    def __init__(self, request_name):
        self.request_name = request_name
        self.trace_id = str(uuid.uuid4())
        self.spans = []

    def span(self, name):
        s = Span(self.trace_id, name)
        self.spans.append(s)
        return s

def compute_cost(input_tokens, output_tokens, in_price=0.0002, out_price=0.0004):
    return input_tokens * in_price + output_tokens * out_price

def qps(requests, seconds):
    return requests / seconds

def error_rate(errors, total):
    return errors / total if total else 0.0

def p99(latencies_ms):
    if len(latencies_ms) < 2:
        return latencies_ms[0] if latencies_ms else 0
    # 简化写法：用 100 分位近似 P99
    return quantiles(latencies_ms, n=100)[98]

trace = Trace("faq_answer")

with trace.span("retrieval") as s:
    time.sleep(0.01)
    s.set_attr("docs", 3)
    s.set_attr("input_tokens", 12)
    s.set_attr("output_tokens", 0)

with trace.span("llm_call") as s:
    time.sleep(0.02)
    s.set_attr("model", "demo-model")
    s.set_attr("input_tokens", 120)
    s.set_attr("output_tokens", 30)
    s.set_attr("cost", compute_cost(120, 30))

total_cost = sum(span.attrs.get("cost", 0) for span in trace.spans)
latencies = [span.attrs["latency_ms"] for span in trace.spans]

assert len(trace.trace_id) > 10
assert total_cost > 0
assert qps(120, 60) == 2
assert round(error_rate(3, 100), 2) == 0.03
assert max(latencies) >= min(latencies)
```

上面代码体现了三个关键点：

1. `Trace` 代表一次完整请求。
2. `Span` 代表一次检索、一次模型调用或一次工具调用。
3. Token、延迟、状态、成本都挂在 span 属性上，便于平台展示和聚合。

如果你接 LangSmith，一般做法是在每次应用请求开始时创建 run 或 trace，并把 `prompt_build`、`retrieval`、`llm_call`、`tool_call` 作为子 run。控制台里可以直接看到链路树、每一步耗时、输入输出和标签。

如果你接 W&B Weave，思路也类似，只是它更强调 trace 和评估结合。白话说，不只是看慢不慢、贵不贵，还能把“答得对不对”一起跟踪。

应用层的伪代码可以抽象成这样：

```python
def handle_request(user_query):
    trace_id = create_trace("chat_request")

    log_event("request_received", trace_id=trace_id, query=user_query[:200])

    with span("prompt_build", trace_id=trace_id) as s:
        prompt = build_prompt(user_query)
        s.set_attr("prompt_version", "v3")

    with span("retrieval", trace_id=trace_id) as s:
        docs = search_docs(user_query)
        s.set_attr("doc_count", len(docs))
        s.set_attr("cache_hit", False)

    with span("llm_call", trace_id=trace_id) as s:
        result = call_llm(prompt, docs)
        s.set_attr("input_tokens", result.input_tokens)
        s.set_attr("output_tokens", result.output_tokens)
        s.set_attr("cost_usd", result.cost_usd)

    update_counter("requests_total", 1)
    observe_histogram("request_latency_ms", trace_latency(trace_id))
    observe_counter("cost_usd_total", total_cost(trace_id))

    log_event("request_finished", trace_id=trace_id, answer=result.text[:200])
    return result.text
```

这里日志、trace、指标分别解决不同问题：

| 手段 | 最适合回答的问题 |
| --- | --- |
| 日志 | 这次请求输入输出是什么？ |
| Trace | 这次请求卡在第几步？ |
| 指标 | 最近 10 分钟整体是否变差？ |

三者必须配合，而不是互相替代。

---

## 工程权衡与常见坑

工程上最常见的错误不是“完全没监控”，而是“监控看起来很多，但没有决策价值”。

下面这张表列出典型坑：

| 常见坑 | 危害 | 规避做法 |
| --- | --- | --- |
| 只看平均延迟 | 尾部卡顿长期不可见，用户抱怨但图表正常 | 监控 P95/P99，并保留最慢 trace 样本 |
| 只记总 Token | 成本归因不清，不知道是检索、重试还是模型输出过长 | 在每个 span 上记录 token 和 cost |
| 只有错误率，没有错误上下文 | 无法区分超时、限流、解析失败、工具异常 | 记录错误类型、模型版本、工具名、prompt 版本 |
| 没有 trace_id | 日志、指标和请求无法关联 | 统一 trace_id 贯穿入口到出口 |
| 把 prompt 全量入日志 | 可能泄露隐私或敏感数据 | 记录摘要、哈希或脱敏字段 |
| 只监控模型，不监控工具 | 工具慢或错时被误判为模型问题 | 把工具和外部 API 单独建 span |
| 不记录重试 | 重试把成本和延迟放大，但图上看不出 | 给 span 增加 retry_count 和 attempt 属性 |

平均延迟和 P99 的权衡尤其重要。平均值适合做容量规划，P99 更适合做体验告警。一个真实工程坑是：系统平均延迟只有 600ms，看起来不错，但 P99 长期 4 秒。原因是少数 retrieval 请求命中了冷数据路径并重试两次。团队如果只盯平均值，会一直误以为“偶发投诉不重要”；一旦接入 span 级 trace，最慢路径会立刻暴露。

另一个常见坑是“只统计模型 Token”。这在普通聊天产品里勉强够用，但在 RAG、Agent、工作流系统里会失真，因为真正花钱的不止模型调用。检索服务、重排器、OCR、网页抓取、数据库查询都有成本。你必须把 span-level cost 单独记录，才能知道“贵”的根因是什么。

还要补一个容易被忽略的问题：缺少评估追踪，线上质量会慢慢滑坡。白话说，系统也许没有报错、延迟也正常，但答案变差了。如果用 W&B Weave 或类似方案把 trace 和 eval 绑定，你就能把“这一类 trace 最近命中率下降”作为质量信号，而不是等用户投诉后再排查。

真实工程例子：一个客服 RAG 系统在 LangSmith 上看见成本上涨 40%。点进 trace 发现不是模型变贵，而是某个 `tool_call(search_ticket)` 因为超时被重试三次，随后 fallback 到更长 prompt，导致 LLM 输出也更长。若只有总成本曲线，你只知道“钱变多了”；有 span 级观测后，你能直接确定责任链路。

---

## 替代方案与适用边界

不是所有项目都需要同一套工具。选择方案要看复杂度、预算和团队阶段。

先看对比表：

| 方案 | 适合场景 | 优势 | 边界 |
| --- | --- | --- | --- |
| 纯日志 + Prometheus/Grafana | 个人项目、原型验证 | 轻量、便宜、易接入 | 很难看到单次 trace 级成本和链路 |
| LangSmith | LangChain/LangGraph 或多步 LLM 流程 | trace 可视化强，适合调试 prompt、retrieval、tool 链路 | 生态更偏 LLM 应用链路，不是通用全栈观测替代品 |
| W&B Weave | 需要把 tracing 和 eval 联动 | 适合把输入输出、评分、实验记录放一起看 | 对纯基础设施监控覆盖有限 |
| 自建 OpenTelemetry pipeline | 大团队、已有统一观测平台 | 可接入现有监控体系，扩展性强 | 初期建设成本高，语义规范要自己补齐 |

可以按两个场景理解。

场景一：个人项目。比如你做一个简单文档问答页，流量小、链路短。此时用日志记录请求摘要、Prometheus 统计 avg latency、错误率、总成本，已经能覆盖大部分问题。但你要清楚边界：你很难回答“这次成本是检索高还是模型高”“哪一步导致尾部延迟飙升”。

场景二：生产 RAG。比如客服、金融问答、内部知识助手，链路包含检索、重排、模型、工具、缓存和权限判断。此时建议使用 LangSmith 或 W&B 这类能展示 trace 树的平台，并在每个 span 上记录延迟、Token、cost、状态。否则系统一旦变复杂，排障成本会超过接入观测的成本。

一个实用判断规则是：

- 链路少于 2 步，先上日志和指标
- 链路达到 3 到 5 步，补 trace
- 有工具调用、重试、缓存、多模型切换时，必须做 span 级成本归因
- 有评估数据或线上质量要求时，把 trace 和 eval 绑定

替代方案不是“谁先进就选谁”，而是“谁能回答你的核心问题”。如果你的核心问题是“为什么贵”，那就优先 span 成本归因；如果核心问题是“为什么答案变差”，那就优先 trace+eval；如果核心问题是“系统会不会挂”，基础指标可能已经够用。

---

## 参考资料

1. IBM, *What is LLM Observability?*  
   贡献：给出了 LLM 可观测性的总体定义，强调行为、性能、资源消耗要同步采集，而不是只看单一监控指标。

2. LangSmith 官方文档与产品页面  
   贡献：展示了 trace / run / span 的可视化方式，适合说明 `prompt -> retrieval -> llm -> tool` 这类链路如何在 dashboard 中查看延迟、Token 和状态。

3. Weights & Biases Weave 官方文档  
   贡献：说明 tracing 与 evaluation 联动的价值，支持把输入输出、运行记录和质量评估放到同一体系中分析。

4. Braintrust 等 LLM 质量与监控资料  
   贡献：帮助统一关键指标定义，如错误率、Token 成本、尾部延迟，以及为什么只看平均值会掩盖真实用户体验。

5. OpenTelemetry 相关规范与实践资料  
   贡献：提供通用 tracing 语义和埋点思路，适合需要将 LLM 应用接入现有企业监控体系的团队。
