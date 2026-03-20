## 核心结论

LLM 推理 API 集成的核心，不是“把 prompt 发出去”，而是把一次推理当成一个可观测、可限流、可校验的网络任务。对 OpenAI 和 Anthropic 而言，生产级做法通常是三层组合：异步客户端、流式消费、结构化约束。

异步客户端的白话解释是：同一个事件循环里同时管理很多网络请求，而不是一个请求堵住一个线程。OpenAI Python SDK 提供 `AsyncOpenAI`，Anthropic Python SDK 提供 `AsyncAnthropic`，两者底层都基于 `httpx` 风格的异步 HTTP 能力。这样做的直接收益是并发吞吐更高、连接池能复用、超时和重试策略更集中。

流式响应的白话解释是：模型一边生成，你一边收到事件，不必等整段文本结束。对 OpenAI 的 Responses API，常见文本事件序列是 `response.created` → 多个 `response.output_text.delta` → `response.completed`，中间可能插入工具调用或错误事件。流式消费适合聊天 UI、Agent 节点编排、长答案早展示，但也带来审核和状态同步难题。

结构化输出的白话解释是：不是让模型随便返回一段文本，而是要求它返回满足固定字段和类型的结果。工程上应优先使用 JSON Schema、固定枚举、受限工具集合 `allowed_tools`，并在本地再次校验。只要结构不合法，就丢弃结果，不触发下游动作。这比“先拿自由文本，再猜它是不是 JSON”稳定得多。

下面这张表是最重要的取舍：

| 方案 | 安全性 | 可测性 | 触发策略 |
|---|---|---|---|
| 结构化输出校验 | 高，字段和类型可约束 | 高，可做单元测试和回归测试 | 仅校验通过才触发工具或写库 |
| 原始文本 | 低，易混入额外指令 | 低，断言脆弱 | 常依赖字符串匹配，误触发概率高 |

玩具例子：模型要返回一个二分类结果，只允许 `{"label":"allow"|"deny","reason":"..."}`。如果返回了多余字段、缺字段、或者在 JSON 外夹带解释文字，就直接判失败，不执行任何后续逻辑。

真实工程例子：客服 Agent 要决定是否调用退款工具。上游模型只能输出 `{"action":"refund"|"reject","ticket_id":"...","amount":123}`，并且工具层只暴露 `refund_order`。只有当 schema 合法、金额范围合法、人工审批通过时，才真的发起退款请求。

---

## 问题定义与边界

这类系统要解决的问题，不是“模型能不能回答”，而是“在高并发、多模型、成本敏感、失败常见的生产环境里，回答能不能稳定落地”。

一次完整请求至少包含这些边界：

| 异常类型 | 默认行为 | 说明 |
|---|---|---|
| 408 | 重试 2 次，指数退避 | 请求超时，通常属于瞬时失败 |
| 409 | 重试 2 次，指数退避 | 冲突类状态，部分 SDK 默认可重试 |
| 429 | 重试 2 次，优先尊重 `Retry-After` | 配额或速率限制，不能立即猛打 |
| 5xx | 重试 2 次，指数退避 | 服务端暂时异常 |
| 连接错误 | 重试 2 次，指数退避 | DNS、TCP、TLS 或网络抖动 |
| 结构校验失败 | 默认不重试业务动作 | 可以重试推理，但不能直接执行下游副作用 |

成本边界也必须前置。最常用公式是：

$$
total\_cost = \frac{input\_tokens}{1{,}000{,}000}\cdot input\_price + \frac{output\_tokens}{1{,}000{,}000}\cdot output\_price
$$

截至 2026-03-21，OpenAI 定价页仍列出 `gpt-5.1` 的标准价为输入 $1.25/1M tokens、输出 $10.00/1M tokens。假设一次请求输入 120 token、输出 256 token，则：

$$
cost = \frac{120}{1{,}000{,}000}\cdot 1.25 + \frac{256}{1{,}000{,}000}\cdot 10 \approx 0.00271\ \text{USD}
$$

如果同样请求并发 5 个，总成本约 $0.01355。这个量不大，但若你把 `max_output_tokens` 拉得很高、重试不设上限、或者工具链每步都再调一次模型，成本会按工作流深度线性甚至超线性上升。

新手最容易忽略的边界是“共享资源”。例如博客 `posts` 页面上的一个提问入口，后端不该每次查询都新建一个 `httpx.AsyncClient`。正确做法是统一封装 `prompt`、`max_tokens`、`schema`、`model`、`timeout`，交给一个长期存活的异步客户端，再由 SSE 逐段解析响应。这样连接池可复用，限流也能全局生效。

---

## 核心机制与推导

OpenAI 的流式响应可以看成状态机。状态机的白话解释是：系统总处在少数几个明确状态里，并按规则迁移，而不是靠零散 if/else 拼出来。

$$
created \rightarrow delta_1 \rightarrow delta_2 \rightarrow \cdots \rightarrow completed
$$

异常路径则是：

$$
created \rightarrow delta^* \rightarrow error
$$

可写成更接近代码的流程：

```text
response.created
  -> zero or more response.output_text.delta
  -> response.completed
or
response.created
  -> zero or more delta
  -> error
```

消费逻辑的关键点只有三个：

1. `delta` 事件只负责追加缓冲区，不做副作用。
2. `completed` 才表示文本层完成，不等于业务层合法。
3. `error` 或连接中断时，必须关闭流并把该任务状态标为失败。

新手版本的伪代码如下：

```python
buffer = ""
state = "streaming"

async for event in response.sse_events():
    if event.type == "response.output_text.delta":
        buffer += event.delta
    elif event.type == "response.completed":
        state = "completed"
    elif event.type == "response.error":
        state = "failed"
        await response.aclose()
        break
```

并发控制则依赖 `asyncio.Semaphore`。它的白话解释是：同一时间最多只允许固定数量的任务进入临界区，超出的排队。这个机制比“随便 `gather` 一百个请求再看运气”可靠，因为配额、连接池、下游数据库写入能力都不是无限的。

重试策略本质上是在“恢复概率”和“尾延迟”之间做折中。OpenAI 与 Anthropic SDK 文档都说明：连接错误、408、409、429、5xx 默认会自动重试 2 次，并采用短指数退避。若你自己实现控制器，常见工程参数是起始 0.5 秒、上限 8 秒，并加随机抖动 jitter，避免雪崩重试：

```python
delay = min(0.5 * (2 ** attempt), 8.0)
await asyncio.sleep(delay + random.random() * 0.25)
```

这里的 jitter，白话解释是：给不同客户端增加一点随机时间偏移，避免所有失败请求在同一时刻再次冲击服务端。

---

## 代码实现

下面给一个可运行的最小实现。它不直接依赖真实 API，而是用玩具 SSE 事件模拟“流式拼接 JSON、校验 schema、统计成本、并发限流”的完整骨架。你可以先跑通这个模型，再把 `fake_sse_stream` 替换成 OpenAI 或 Anthropic 的真实事件源。

```python
import asyncio
import json
import random
from dataclasses import dataclass

@dataclass
class Event:
    type: str
    delta: str = ""

class CostTracker:
    def __init__(self, input_price, output_price):
        self.input_price = input_price
        self.output_price = output_price
        self.total_cost = 0.0

    def add(self, input_tokens, output_tokens):
        cost = (input_tokens / 1_000_000) * self.input_price + \
               (output_tokens / 1_000_000) * self.output_price
        self.total_cost += cost
        return cost

def validate_schema(obj):
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"action", "keyword"}:
        return False
    if obj["action"] not in {"search", "reject"}:
        return False
    if not isinstance(obj["keyword"], str):
        return False
    return True

async def fake_sse_stream(valid=True):
    yield Event("response.created")
    if valid:
        chunks = ['{"action":"search",', '"keyword":"llm api"}']
    else:
        chunks = ['{"action":"hack",', '"keyword":123,"extra":true}']
    for chunk in chunks:
        await asyncio.sleep(0.01)
        yield Event("response.output_text.delta", chunk)
    yield Event("response.completed")

async def stream_chat(task_name, sem, tracker, valid=True):
    async with sem:
        buffer = ""
        state = "streaming"
        async for event in fake_sse_stream(valid=valid):
            if event.type == "response.output_text.delta":
                buffer += event.delta
            elif event.type == "response.completed":
                state = "completed"

        if state != "completed":
            return {"task": task_name, "state": "failed", "data": None}

        try:
            data = json.loads(buffer)
        except json.JSONDecodeError:
            return {"task": task_name, "state": "invalid_json", "data": None}

        if not validate_schema(data):
            return {"task": task_name, "state": "schema_rejected", "data": None}

        tracker.add(input_tokens=120, output_tokens=len(buffer.split()))
        return {"task": task_name, "state": "ok", "data": data}

async def retry_call(coro_factory, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except RuntimeError:
            if attempt == max_retries:
                raise
            delay = min(0.5 * (2 ** attempt), 8.0)
            await asyncio.sleep(delay + random.random() * 0.25)

async def main():
    sem = asyncio.Semaphore(2)
    tracker = CostTracker(input_price=1.25, output_price=10.0)

    tasks = [
        retry_call(lambda: stream_chat("toy-ok", sem, tracker, valid=True)),
        retry_call(lambda: stream_chat("toy-bad", sem, tracker, valid=False)),
    ]
    results = await asyncio.gather(*tasks)

    assert results[0]["state"] == "ok"
    assert results[0]["data"]["action"] == "search"
    assert results[1]["state"] == "schema_rejected"
    assert tracker.total_cost > 0

asyncio.run(main())
```

接到真实 API 时，模块划分建议如下：

| 代码模块 | 职责 |
|---|---|
| `ClientWrapper` | 初始化 `AsyncOpenAI`/`AsyncAnthropic`/`httpx.AsyncClient`，统一超时、头信息、连接池 |
| `EventParser` | 消费 SSE，识别 `created/delta/completed/error`，组装文本或工具参数 |
| `RetryController` | 处理可重试错误、退避、`Retry-After` |
| `SchemaValidator` | 做 JSON Schema 或 Pydantic 校验 |
| `ToolDispatcher` | 仅在校验通过后调用 `tool.invoke()` |
| `CostTracker` | 统计 token、价格、预算阈值 |

低层 `httpx` 骨架可以写成这样，适合你自己解析 SSE：

```python
async with client.stream("POST", url, json=request_payload) as response:
    async for line in response.aiter_lines():
        event = parse_sse_line(line)
        if event.type == "response.output_text.delta":
            buffer += event.delta
        elif event.type == "response.completed":
            break
# 离开 async with 后连接自动归还
```

如果你已经在 OpenAI SDK 的高层接口上工作，通常会直接 `stream=True` 并 `async for event in stream:`；如果你在 Anthropic SDK 上工作，既可以用 `messages.stream(...)` 获取辅助封装，也可以用 `messages.create(..., stream=True)` 直接迭代原始事件。

---

## 工程权衡与常见坑

稳定性和吞吐量经常互相拉扯。流式请求时间更长，意味着连接占用更久；并发更高，意味着更容易打到 429；结构校验更严格，意味着更多结果会被丢弃，但这正是安全边界的一部分。

下面是最常见的坑：

| 坑 | 表现 | 解决方案 |
|---|---|---|
| SSE 读流外漏 | 连接池泄漏、后续请求卡住 | 用 `async with`；手动流模式必须 `aclose()` |
| 直接依赖自由文本触发工具 | 用户输入能间接操纵下游动作 | 改成结构化输出 + 白名单工具 |
| 忽略 `Retry-After` | 429 后越打越快，额度迅速耗尽 | 优先尊重服务端退避指令 |
| UI 只看最终结果 | 中途断流时前端状态与后端不一致 | 保存 partial state，并在 error 时显式标失败 |
| 每次请求都新建客户端 | TLS 握手多、连接复用差、吞吐低 | 复用单例或作用域级客户端 |
| 只统计成功请求成本 | 预算偏低估算 | 把失败、重试、工具调用也纳入成本 |

一个典型错误写法是直接把整个响应 `await response.aread()` 然后再解析。这样如果服务端长期流式输出、客户端中途异常，连接可能挂得很久。更稳妥的模板是：

```python
response = None
state = "started"
try:
    response = await client.send(request, stream=True)
    async for line in response.aiter_lines():
        event = parse_sse_line(line)
        if event.type == "error":
            state = "failed"
            break
    if state != "failed":
        state = "completed"
finally:
    if response is not None:
        await response.aclose()
```

真实工程里，还要把 `_request_id`、模型名、重试次数、输入输出 token、最终 schema 版本一起打日志。否则线上出问题时，你只能看到“用户说模型坏了”，却不知道是模型输出不合法、网络断了、还是你的工具层拒绝了执行。

---

## 替代方案与适用边界

不是所有场景都必须上 SSE、异步和工具调用。更简单的方案在低并发或离线任务里反而更稳。

| 替代方案 | 优点 | 适用边界 |
|---|---|---|
| 同步 SDK + 非流式响应 | 实现简单，调试直观 | 后台脚本、管理后台、低并发 |
| 同步/异步 + 轮询 | 兼容不支持 SSE 的平台 | 网关限制多、只关心最终结果 |
| Batch 接口 | 成本和吞吐更可控 | 大批量离线处理，非交互式 |
| 纯文本后处理成 JSON | 迁移成本低 | 风险可接受、下游无高危副作用 |
| 服务端中间层统一封装 | 前端简单、策略集中 | 多产品线共享 LLM 基础设施 |

新手版本的降级路径可以是：平台不支持 SSE，就直接调用一次完整响应，例如 `client.chat.completions.create()` 或对应供应商的非流式接口，拿到完整文本后再 `json.loads`，不合法就返回失败。这种做法牺牲首字延迟和中间态体验，但实现成本更低。

适用边界很明确：

1. 有副作用的动作，例如发邮件、退款、写数据库、调用内部工具，优先用结构化输出和白名单工具。
2. 高并发交互式应用，例如聊天、协作编辑、Agent 工作流，优先异步 + 流式 + 限流。
3. 批量离线总结、数据标注、内容改写，可以优先 Batch 或非流式。
4. 审核要求很强的场景，要谨慎使用“边生成边展示”，因为流式输出的部分片段更难实时审查。

Anthropic SDK 的重试说明、OpenAI 的 Agent 安全指南、HTTPX 对流关闭责任的说明，共同给出的启发是一致的：网络层要保守，结构层要严格，工具层要最小授权。

---

## 参考资料

1. OpenAI, Streaming API responses  
   说明 Responses API 的 SSE 事件模型，明确常见事件包括 `response.created`、`response.output_text.delta`、`response.completed`、`error`。  
   https://developers.openai.com/api/docs/guides/streaming-responses

2. OpenAI, Pricing  
   用于核对按 token 计费公式中的输入价、输出价；本文价格示例以 2026-03-21 页面可见条目为准。  
   https://developers.openai.com/api/docs/pricing

3. OpenAI, Safety in building agents  
   说明为什么要用结构化输出限制数据流、不要让不可信输入直接驱动工具调用，并建议保留工具审批。  
   https://developers.openai.com/api/docs/guides/agent-builder-safety

4. OpenAI Python SDK README  
   说明 `AsyncOpenAI`、`stream=True`、默认重试 2 次、异步客户端由 `httpx` 驱动。  
   https://github.com/openai/openai-python

5. Anthropic Python SDK README  
   说明 `AsyncAnthropic`、`messages.stream(...)`、`messages.create(..., stream=True)`、默认重试 2 次、底层可配置 `httpx`/`aiohttp`。  
   https://github.com/anthropics/anthropic-sdk-python

6. HTTPX Async Support  
   说明 `AsyncClient.stream(...)`、`Response.aclose()`、连接池复用与手动流模式下的资源释放责任。  
   https://www.python-httpx.org/async/
