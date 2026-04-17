## 核心结论

多 Agent 系统的可观测性，核心不是“多打一点日志”，而是把一次运行中所有跨 Agent、跨工具、跨服务的动作，组织成一棵可追踪的因果树。这里的“因果树”可以白话理解为：谁触发了谁、谁等待了谁、哪一步最贵、哪一步最慢，都能在一个统一视图里看到。

如果没有这棵树，调试多 Agent 系统基本等于在黑盒里排查随机过程。因为这类系统同时具备三种复杂性：第一，消息很多；第二，状态分散；第三，执行路径不是固定流程，而是带概率分叉的动态协作。单看控制台日志，只能看到“发生了什么”；看不到“为什么发生”“是谁导致的”“代价传导到了哪里”。

因此，生产环境里更可靠的做法是：以 OpenTelemetry 的 `TraceID + SpanID + ParentSpanID` 为主线，把每个 Agent 消息、模型调用、工具调用、子 Agent 触发都表示成一个 span。然后在 span 上补充模型名、输入输出摘要、token、cost、latency、重试次数、质量分数等属性。这样同一条 trace 就同时承载了性能、成本、质量和错误定位信息。

一个最小玩具例子足够说明问题：Agent A 发起查询是 `span1`，A 调工具 T 是 `span2`，工具 T 又触发 Agent B 是 `span3`。三者共享同一个 `TraceID`，而 `span2.parent = span1`、`span3.parent = span2`。如果 `span2.latency = 120ms`，那么你能立刻判断：下游等待不是 Agent B 本身慢，而是被工具调用阻塞。这种定位能力，才是多 Agent 可观测性的真正价值。

| 信号层 | 关注点 | 在多 Agent 里的作用 | 典型工具 |
| --- | --- | --- | --- |
| Metrics | 延迟、吞吐、错误率、token、成本 | 看整体健康度和趋势 | Prometheus, Grafana |
| Traces | 调用链、父子关系、耗时分布 | 还原因果路径，定位瓶颈 | OpenTelemetry, Jaeger |
| Logs | 事件明细、参数摘要、异常栈 | 补足上下文，便于重播 | Python logging, ELK |
| Evaluations | 答案质量、任务成功率、人工评分 | 判断“跑完了”是否等于“跑对了” | 自定义评测, 平台评估系统 |

---

## 问题定义与边界

“可观测性”这个词，白话解释就是：系统内部虽然复杂，但你可以通过外部信号把内部状态推断出来。放到多 Agent 场景里，意思是开发者能从消息、日志、trace、指标和评测结果中，还原一次任务到底怎么执行、哪里失败、为什么失败。

问题边界先要画清楚。多 Agent 可观测性不是只看某个 LLM 请求，也不是只看某个 Agent 的对话记录，而是至少覆盖三层：

| 维度 | 传统单体服务 | 多 Agent 系统 |
| --- | --- | --- |
| 执行路径 | 大多确定性 | 常带概率性和动态分叉 |
| 状态位置 | 多集中在单服务 | 分散在 Agent 内存、工具、数据库、会话缓存 |
| 故障判断 | 成功/失败较明确 | 可能“技术成功但业务失败” |
| 关注对象 | CPU、接口、SQL | 再加消息链、token、cost、质量 |
| 调试方式 | 看接口日志和异常栈 | 需要重建跨角色因果链 |

第一层是 Agent 内部观测。比如一个 Agent 是否调用了模型、用了哪个工具、重试几次、消耗多少 token。第二层是 Agent 间观测。比如 Agent A 的一条消息是否触发了 Agent B，B 返回结果后是否又触发了审计 Agent。第三层是跨运行和跨系统观测。比如同一类任务在一周内平均成本是否上升，某个工具升级后成功率是否下降。

这里有一个新手非常容易忽略的边界：没有统一的 `runId` 或 correlation ID，再完整的日志也只是碎片。correlation ID 可以白话理解为“一次任务的统一编号”。做法并不复杂：任务入口一生成 `runId`，后面无论是 Agent、工具、模型封装层，还是 HTTP 请求、消息队列、数据库写入，都把这个 ID 带上。出问题后，先按 `runId` 过滤，再看 trace 树和结构化日志，才能像翻流水账一样重播整次执行。

所以，多 Agent 的可观测性目标不是“记录更多内容”，而是“定义统一边界后让所有信号能对齐”。如果边界没定义，常见结果是：A 框架有自己的日志，B 框架有自己的调试页，工具服务又有另一套监控，最后谁都看到了局部，但没人能回答“为什么这次任务比昨天慢 40%”。

---

## 核心机制与推导

OpenTelemetry trace 的核心对象是 trace 和 span。span 可以白话理解为“一段有开始和结束的工作单元”；trace 则是一组有关联的 span。对于树形调用链，可以写成：

$$
\text{Trace} = \{s_1, s_2, ..., s_n\}
$$

其中每个 span 都至少包含：

$$
s_i = (\text{TraceID}, \text{SpanID}, \text{ParentSpanID}, \text{Start}, \text{End}, \text{Attributes})
$$

如果 `ParentSpanID` 指向某个父节点，那么整条 trace 就形成一棵有向树。树结构的价值在于，它不仅回答“发生了哪些事”，还回答“这些事之间的依赖关系是什么”。

在多 Agent 里，一个合理的建模方式是：

1. 用户请求进入系统，创建根 span。
2. Agent A 处理任务，创建子 span。
3. Agent A 调模型，创建 model span。
4. Agent A 调工具，创建 tool span。
5. 工具触发 Agent B，再创建 Agent B span。
6. 如果 B 又调模型或工具，继续向下扩展。

下面是一个简化后的因果链表：

| Span | 名称 | Parent | latency | tokens | cost | 含义 |
| --- | --- | --- | --- | --- | --- | --- |
| span1 | Agent A Query | root | 35ms | 120 | $0.002 | A 接收并分析任务 |
| span2 | Tool T Client | span1 | 120ms | 0 | $0 | A 调外部工具 |
| span3 | Agent B Server | span2 | 40ms | 180 | $0.003 | 工具返回后触发 B |
| span4 | LLM Generate | span3 | 260ms | 900 | $0.018 | B 调大模型生成答案 |

这个表能直接说明两个事实。第一，`span3` 启动晚，不是因为 Agent B 初始化慢，而是被 `span2` 的工具等待拖住了。第二，真正最大的延迟热点其实是 `span4` 的模型生成，而不是 Agent 编排逻辑本身。

玩具例子可以更直观。假设一个旅行规划系统里，Agent A 负责拆解需求，工具 T 查询天气，Agent B 负责生成行程建议。用户问“下周去杭州三天怎么安排”。如果你只看日志，可能看到三类记录：A 收到问题、T 返回天气、B 输出建议。但你不知道顺序是否稳定，不知道哪一轮触发了哪一轮，也不知道 token 和延迟花在哪。换成 trace 后，三步全挂在同一 `TraceID` 下，因果关系明确，定位就从“猜”变成“看”。

进一步说，trace 还要和日志、指标联动。structured logging 的意思是“结构化日志”，白话解释就是日志字段固定成机器可检索的键值形式，而不是一整行自然语言。比如每条日志都带 `run_id`、`trace_id`、`agent_name`、`tool_name`、`status`。这样 trace 负责还原树，日志负责补细节，metrics 负责看趋势。三者结合，才能同时回答三个问题：

1. 单次失败发生在哪里。
2. 这类失败最近是否在变多。
3. 即使技术上成功，质量是否变差。

这也是为什么多 Agent 系统不能只依赖“输出对不对”。很多任务并非二元正确，而是质量有梯度。例如报告摘要能不能用、代码修复是否稳定、检索结果是否覆盖关键证据，往往需要额外评估信号。换句话说，好的可观测性不是只观察执行过程，还要观察结果质量。

---

## 代码实现

下面用一个最小 Python 例子演示多 Agent trace 的核心结构。它不依赖真实 OpenTelemetry SDK，也能直接运行，用来理解 `TraceID / SpanID / ParentSpanID / run_id` 如何串起因果链。

```python
from dataclasses import dataclass, field
from time import perf_counter, sleep
from uuid import uuid4


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    attributes: dict = field(default_factory=dict)
    start_ms: float = 0.0
    end_ms: float = 0.0

    @property
    def latency_ms(self) -> float:
        return round(self.end_ms - self.start_ms, 2)


class TraceRecorder:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.trace_id = uuid4().hex
        self.spans: list[Span] = []

    def start_span(self, name: str, parent_span_id: str | None = None, **attrs) -> Span:
        span = Span(
            trace_id=self.trace_id,
            span_id=uuid4().hex[:16],
            parent_span_id=parent_span_id,
            name=name,
            attributes={"run_id": self.run_id, **attrs},
            start_ms=perf_counter() * 1000,
        )
        self.spans.append(span)
        return span

    def end_span(self, span: Span):
        span.end_ms = perf_counter() * 1000


def call_tool_weather(recorder: TraceRecorder, parent_span_id: str) -> dict:
    span = recorder.start_span("tool.weather", parent_span_id, tool="weather_api")
    sleep(0.12)
    result = {"city": "Hangzhou", "forecast": "rain", "temperature": 18}
    span.attributes["output_size"] = len(result)
    recorder.end_span(span)
    return {"tool_result": result, "span_id": span.span_id}


def agent_b_plan(recorder: TraceRecorder, parent_span_id: str, tool_result: dict) -> dict:
    span = recorder.start_span(
        "agent.b.plan",
        parent_span_id,
        agent="agent_b",
        model="gpt-4.1-mini",
        prompt_tokens=180,
        completion_tokens=220,
        cost_usd=0.0042,
    )
    sleep(0.04)
    answer = f"{tool_result['city']}三天行程，注意{tool_result['forecast']}天气。"
    span.attributes["answer_chars"] = len(answer)
    recorder.end_span(span)
    return {"answer": answer, "span_id": span.span_id}


def run_workflow():
    recorder = TraceRecorder(run_id="run-demo-001")

    root = recorder.start_span("agent.a.query", None, agent="agent_a", user_query="下周去杭州三天怎么安排")
    tool_payload = call_tool_weather(recorder, root.span_id)
    result = agent_b_plan(recorder, tool_payload["span_id"], tool_payload["tool_result"])
    recorder.end_span(root)

    return recorder, result


if __name__ == "__main__":
    recorder, result = run_workflow()

    spans = {s.name: s for s in recorder.spans}
    assert spans["agent.a.query"].trace_id == spans["tool.weather"].trace_id == spans["agent.b.plan"].trace_id
    assert spans["tool.weather"].parent_span_id == spans["agent.a.query"].span_id
    assert spans["agent.b.plan"].parent_span_id == spans["tool.weather"].span_id
    assert spans["tool.weather"].latency_ms >= 100
    assert "杭州" in result["answer"]

    for s in recorder.spans:
        print({
            "name": s.name,
            "trace_id": s.trace_id,
            "span_id": s.span_id,
            "parent_span_id": s.parent_span_id,
            "latency_ms": s.latency_ms,
            "attributes": s.attributes,
        })
```

上面代码展示了三个关键点。第一，入口创建一次 `run_id` 和 `trace_id`。第二，每个子动作只新建自己的 `span_id`，同时保存 `parent_span_id`。第三，业务属性不是写死在日志文本里，而是挂到 `attributes` 上，后续更容易送到后端查询和聚合。

如果接到真实 OpenTelemetry 后端，思路基本一致，只是把自定义 `TraceRecorder` 换成 SDK 提供的 tracer provider、processor 和 exporter。下面是接近生产配置的示意代码：

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

resource = Resource.create({
    "service.name": "multi-agent-blog-demo",
    "deployment.environment": "prod",
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces")
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("multi-agent-demo")

run_id = "run-20260319-001"

with tracer.start_as_current_span("agent.a.query") as span1:
    span1.set_attribute("run_id", run_id)
    span1.set_attribute("agent.name", "agent_a")

    with tracer.start_as_current_span("tool.weather") as span2:
        span2.set_attribute("tool.name", "weather_api")
        span2.set_attribute("latency.target_ms", 120)

        with tracer.start_as_current_span("agent.b.plan") as span3:
            span3.set_attribute("agent.name", "agent_b")
            span3.set_attribute("llm.model", "gpt-4.1-mini")
            span3.set_attribute("llm.prompt_tokens", 180)
            span3.set_attribute("llm.completion_tokens", 220)
            span3.set_attribute("llm.cost_usd", 0.0042)
```

对于 AutoGen，这类配置的重点不在“如何打印日志”，而在“如何把 span 发出去”。官方 tracing 能生成链路信息，但如果没有连接 OTLP/Jaeger/OpenLIT 一类后端，这些信息通常只在本地或框架内部可见，难以和外部工具服务、网关、数据库监控对齐。新人可以先记一句最实用的话：每个 Agent 不一定都要手写 trace，但每次运行必须统一带着上下文跑完全链路。

真实工程例子更能说明收益。某团队最初分别使用 LangSmith、CrewAI 自带输出和工具服务日志，结果问题定位要来回切系统。接入框架中立的观测层后，他们把 LangChain 和 CrewAI 的 trace 统一进入同一后端，发现 CrewAI 路径整体比另一条实现慢约 25%，并最终优化到约 8% 的差距。这里真正带来收益的不是“换了框架”，而是第一次把不同框架、不同工具、不同服务放到了同一条因果链里比较。

---

## 工程权衡与常见坑

生产环境最常见的误区，是把“有日志”误认为“可观测”。日志当然重要，但它天然缺少父子关系和全链路上下文。一个工具调用失败，如果没有 trace 关系，你能看到报错，却不一定知道它来自哪次任务、由哪个 Agent 触发、是否已经重试、是否影响了最终答案。

第二个常见坑是框架默认能力的边界。AutoGen 的 logging 和 tracing 对开发调试有帮助，但默认形态并不天然等于跨服务可观测平台。很多团队一开始在本地文件、SQLite 或控制台里能看到事件，就误以为已经完成观测建设；等到系统拆成多个服务、多个 Agent 运行时，才发现链路断了。

第三个坑出现在 CrewAI 的 `verbose=True`。`verbose` 的白话意思就是“尽量多输出过程”。它适合本地理解执行流程，但在生产环境有两个问题：一是可能暴露用户输入、工具参数、模型中间输出等敏感数据；二是日志流会变得噪声极高，甚至和自定义 logging 策略冲突。调试阶段可以短期开启，生产环境更稳妥的做法是关闭 verbose，把必要字段转成结构化日志和 trace 属性，再对敏感字段做脱敏或摘要化。

下面这个表格可以帮助做方案判断：

| 方案 | 可见性 | 开销 | 集成范围 | 隐私风险 | 适用阶段 |
| --- | --- | --- | --- | --- | --- |
| AutoGen 原生日志/trace | 框架内较清晰 | 低 | 主要限 AutoGen | 中 | 本地开发、单体验证 |
| OpenLIT/OTLP 接入 | 全链路更完整 | 中 | 可对接标准后端 | 中，需治理字段 | 生产环境、跨服务 |
| Agent Observability Kit 类平台 | 跨框架视图较强 | 中到高 | LangChain/CrewAI/AutoGen 混合栈 | 中到高，取决于采集粒度 | 多框架团队、迁移期 |

还有一个容易被忽略的坑是“采集太多”。如果你把完整 prompt、完整工具输入输出、所有中间消息都原样上传，后果通常有三个：存储费用上升、查询效率下降、敏感信息外泄风险增大。实践上更建议区分层次：默认上传摘要、长度、哈希、状态码、评分、token 和成本；只有在特定采样条件或故障模式下，才提升采集粒度。

最后，很多团队能做到 tracing，却做不到 evaluation。结果是他们知道系统“慢在哪里”，却不知道“错在哪里”。对于多 Agent，性能观测和质量观测必须并行，否则只会得到“一个跑得很快但答案不可靠的系统”。

---

## 替代方案与适用边界

如果团队已经深度使用 LangChain，那么 LangSmith 往往是更自然的选择，因为它对 LangChain 生态支持完整，调试体验也成熟。但它的边界同样明确：一旦你的系统开始混用 CrewAI、AutoGen、自研工具服务，单框架平台就容易变成局部最优。

如果团队正处在迁移期，或者本来就是多框架并存，那么框架中立方案更实用。Agent Observability Kit 一类方案的价值，不在于替代 OpenTelemetry，而在于补齐“Agent 语义层”。所谓“Agent 语义层”，白话讲就是它理解 Agent、任务、工具、消息这些对象，而不是只把一切都当普通函数调用。这样做能更快落地统一视图。

是否直接上 OTLP，也要看现有基础设施。如果公司已经有标准可观测平台，例如 Jaeger、Tempo、Grafana、Datadog 或 Azure Monitor，那么优先把多 Agent trace 对接到现有 OTLP 体系通常更划算，因为可以和 API 网关、数据库、缓存、消息队列的链路直接打通。反过来，如果团队还没有统一后端，先上一个能快速展示 Agent 轨迹的平台，再逐步补齐 OTLP，也是一条现实路径。

可以用下面这个矩阵快速判断：

| 工具/方案 | 支持框架 | OTLP 准备度 | 适用场景 |
| --- | --- | --- | --- |
| LangSmith | LangChain 优先 | 一般需额外对接 | 纯 LangChain 团队 |
| AutoGen 原生 tracing | AutoGen | 可扩展但需配置 exporter | AutoGen 单框架开发阶段 |
| OpenLIT | 对接 AutoGen/AG2 等较方便 | 高 | 想快速接入 OTel 后端 |
| Agent Observability Kit | LangChain、CrewAI、AutoGen 等 | 可与现有后端配合 | 多框架并存、迁移期 |
| 纯 OpenTelemetry 自建 | 理论上框架无关 | 最高 | 已有成熟平台与工程能力 |

无论选哪条路线，有一条底线不能变：从 run 开始生成统一 correlation ID，并在 span、log、metric 中共享。否则任何平台都只能展示局部片段，无法真正重建决策链。平台决定“你在哪看”，而 correlation ID 和 trace context 决定“你能不能看全”。

---

## 参考资料

- FrankX, *The Observability Stack for Production Multi-Agent Systems*  
  https://www.frankx.ai/blog/observability-stack-multi-agent-systems-2026?utm_source=openai

- Microsoft Learn, *Tracing in Azure AI Agents / OpenTelemetry concepts*  
  https://learn.microsoft.com/en-us/azure/ai-services/agents/concepts/tracing?utm_source=openai

- OpenTelemetry, *Traces Concept*  
  https://opentelemetry.io/docs/concepts/signals/traces/

- OpenTelemetry Spec, *Trace API*  
  https://opentelemetry.io/docs/specs/otel/trace/api/

- AutoGen, *AgentChat Tracing*  
  https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tracing.html?utm_source=openai

- AutoGen, *Framework Logging*  
  https://microsoft.github.io/autogen/0.4.4/user-guide/core-user-guide/framework/logging.html?utm_source=openai

- OpenLIT, *AutoGen / AG2 Integration*  
  https://docs.openlit.io/latest/sdk/integrations/ag2?utm_source=openai

- CrewAI issue, *verbose logging and custom logger conflicts*  
  https://github.com/crewAIInc/crewAI/issues/3197?utm_source=openai

- Grizzly Peak Software, *Observability for AI Agent Systems*  
  https://www.grizzlypeaksoftware.com/library/observability-for-ai-agent-systems-tpbnn8xp?utm_source=openai

- Dev.to, *Framework-Agnostic Observability for AI Agents: Agent Observability Kit*  
  https://dev.to/seakai/framework-agnostic-observability-for-ai-agents-introducing-agent-observability-kit-da4?utm_source=openai
