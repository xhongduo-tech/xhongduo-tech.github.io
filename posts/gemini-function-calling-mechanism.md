## 核心结论

Gemini 的 Function Calling 本质上是“声明式工具协议”。声明式的意思是：开发者先用 JSON Schema 描述工具能做什么、需要什么参数，模型再根据用户请求自己决定是否调用、调用哪个、参数如何填写。它不是把自然语言硬编码成接口规则，而是把“接口定义”交给模型理解。

Gemini 这套机制有三个值得抓住的结论。

第一，Gemini 把“是否调用工具”和“如何填参数”放进了模型内部的 thinking 过程。thinking 可以理解为“模型先在内部推一遍再输出结果”。这让参数抽取通常比单纯文本生成更稳定，尤其是在函数较多、参数有 `enum`、`required`、数组和对象嵌套时，模型更容易选对函数并补齐字段。

第二，Gemini 原生支持并行调用和组合调用。并行调用的意思是“多个互不依赖的工具一次性同时调”；组合调用的意思是“前一个工具的输出作为后一个工具的输入”。这使它很适合做“先查位置，再查天气”，或“同时搜索、执行代码、调用自定义函数”这类流水线任务。

第三，Gemini 的多轮工具使用依赖 `thought_signature`。这是一段由模型返回的加密推理签名，可以把它理解为“这轮内部推理的上下文凭证”。如果你手动维护对话历史，下一轮必须把它原样带回去；在 Gemini 3 的 function calling 场景里，漏传会直接触发 4xx 校验错误。也就是说，Gemini 的 function calling 不只是“拿到一个函数名和参数就行”，而是一个带状态闭环的协议。

| 维度 | Gemini Function Calling | 对工程的直接影响 |
|---|---|---|
| 工具定义 | JSON Schema / OpenAPI 子集 | 可以把接口能力声明给模型 |
| 参数生成 | 结合 thinking 决策 | 参数抽取更像“结构化预测”而不是自由文本 |
| 并行调用 | 支持 | 适合多数据源、多工具并发 |
| 组合调用 | 支持 | 适合多步骤 agent 流程 |
| 多轮状态 | 依赖 `thought_signature` | 手动拼历史时必须严格回传 |
| SDK 体验 | 官方 SDK 可自动处理签名 | REST 手写请求时更容易踩坑 |

玩具例子很直观。声明一个 `get_weather(city, unit)`，用户说“查一下波士顿今天气温”。如果 `city` 是必填，`unit` 限定在 `["C","F"]`，Gemini 往往会直接给出形如 `{"city":"Boston","unit":"C"}` 的参数对象，而不是先吐一段解释性文本再让你自己解析。

真实工程例子则更能体现差异。一个 Live API 会话里，同时启用 Google Search、Code Execution 和自定义业务函数，用户发一句“查今天英伟达股价，算 30 天均值，再发一段中文摘要”。Gemini 可以在同一轮里发出多个工具调用，请求返回后再汇总成自然语言答案。这降低了外层 orchestrator，也就是“自己写的流程调度器”的复杂度。

---

## 问题定义与边界

Function Calling 要解决的问题，不是“让模型会写 JSON”，而是“让模型把自然语言任务可靠地映射到外部系统操作”。这里的外部系统可能是数据库、天气 API、搜索引擎、执行环境、支付接口，或者你自己写的任意函数。

更正式地说，输入有两部分：

1. 用户请求 $U$
2. 工具集合 $Tools$

其中每个工具都要声明：

- `name`：工具名
- `description`：工具用途
- `parameters`：参数结构，通常是 JSON Schema 子集
- `required`：必填字段
- 可选的 `enum`、数组、对象等约束

模型需要完成两个判断：

1. 这次要不要调用工具
2. 如果调用，应该调用哪个工具，并生成怎样的参数

边界也要讲清楚。

第一，Gemini API 是无状态的。无状态的意思是“服务端不会自动记住你上一轮的推理过程”。所以你看到的多轮工具调用效果，并不是服务端替你保存了完整 agent 状态，而是模型通过 `thought_signature` 在应用层补回推理上下文。

第二，Gemini 支持的是 OpenAPI/JSON Schema 的一个子集，不是完整任意 Schema。子集的意思是“只支持其中一部分写法”。如果把 schema 设计得过深、过复杂、包含不受支持的结构，问题通常不是“性能差一点”，而是模型选错参数、请求校验失败，或者工具调用质量明显下降。

第三，Function Calling 只解决“协议层结构化”，不自动解决“业务层正确性”。比如模型可以正确生成 `{"stock":"NVDA","window":30}`，但你后端依然需要自己校验权限、限流、重试、幂等和副作用确认。幂等的意思是“同一个请求重复执行，不会造成重复副作用”。

下面这个最小工具声明就足够说明边界：

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "unit": {"type": "string", "enum": ["C", "F"]}
    },
    "required": ["city"]
  }
}
```

这个定义能约束“字段长什么样”，但不能保证“城市名一定真实存在”，也不能保证“单位一定符合你后端地区策略”。这些仍然是应用层责任。

---

## 核心机制与推导

可以把一轮 Gemini 工具调用抽象成下面的闭环：

$$
T_t = \{U_t, Tools, H_t\}
$$

其中：

- $U_t$ 是本轮用户输入
- $Tools$ 是工具声明集合
- $H_t$ 是你回传的历史，包括先前的模型输出、工具结果，以及可能存在的 `thought\_signature`

模型经过内部 thinking 后，输出：

$$
P_t = \{FC_1, FC_2, \dots, FC_n, Text\}
$$

这里的 $FC_i$ 表示第 $i$ 个 function call。若是 thinking 模型，对应 part 里可能还会带一个签名 $\sigma_i$，即 `thought_signature`。

执行工具后，你要把结果重新塞回下一轮：

$$
T_{t+1} = T_t \cup \{FC_i, \sigma_i, FR_i\}_{i=1}^{n}
$$

其中 $FR_i$ 是 `functionResponse`。整个过程可以读成：

“用户请求 + 工具声明 + 历史”
$\rightarrow$
“模型产出函数调用”
$\rightarrow$
“应用执行工具”
$\rightarrow$
“把调用记录、签名、工具返回值再交还模型”
$\rightarrow$
“模型继续推理或输出最终答案”

这套机制有两个关键点。

第一，`thought_signature` 不是给你读的，而是给协议续命的。它是加密表示，不应该自己修改、拼接或合并。Google 文档明确要求：如果收到签名，就应按原样放回原始 part；Gemini 3 的 function calling 场景里，不回传会报 4xx。这里“放回原始 part”很重要，因为签名和它所在的内容片段是位置绑定的。

第二，并行调用和组合调用对应两种不同的依赖图。

如果多个工具互不依赖，那么是并行图：

$$
U \rightarrow \{FC_1, FC_2, FC_3\}
$$

例如“查北京天气、查上海天气、记录一次日志”，三者没有依赖关系，可以同时发出。

如果后一个工具依赖前一个工具输出，那么是串联图：

$$
U \rightarrow FC_1 \rightarrow FR_1 \rightarrow FC_2
$$

例如“获取当前位置，再查询当地天气”。`get_weather(location)` 的 `location` 就依赖 `get_current_location()` 的结果。

玩具例子：

用户说：“我现在所在城市天气怎么样？”

工具有两个：

- `get_current_location()`
- `get_weather(location)`

Gemini 不应该直接猜城市，而应先调用前者拿到位置，再调用后者拿天气。这就是组合调用。

真实工程例子：

一个内部运维助手同时暴露三个工具：

- `get_service_status(service_name)`
- `query_recent_errors(service_name, minutes)`
- `create_incident_ticket(service_name, severity, summary)`

用户说：“检查 payment 服务最近 30 分钟状态，有异常就建单。”

理想流程是：

1. 先查服务状态
2. 若状态异常，再查最近错误
3. 再决定是否创建工单

这类依赖链如果让外层业务代码硬编码，会迅速变成大量 if/else；交给 Gemini 的 compositional calling，业务层只要做好工具实现和副作用保护。

---

## 代码实现

实现上可以分成两层看。

第一层是“声明工具”。第二层是“跑协议循环”。

下面先给一个可运行的 Python 玩具实现。它不直接调用 Gemini API，而是先模拟“模型已经选定函数并产出参数”，重点展示 schema 约束和执行闭环。这样零基础读者可以先理解协议，再接官方 SDK。

```python
from typing import Dict, Any

WEATHER_DB = {
    ("Boston", "C"): {"temperature": 6, "condition": "Cloudy"},
    ("Boston", "F"): {"temperature": 43, "condition": "Cloudy"},
    ("Beijing", "C"): {"temperature": 12, "condition": "Sunny"},
}

tool_schema = {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["C", "F"]},
        },
        "required": ["city"],
    },
}

def validate_args(args: Dict[str, Any], schema: Dict[str, Any]) -> None:
    params = schema["parameters"]
    required = params.get("required", [])
    props = params.get("properties", {})
    for key in required:
        assert key in args, f"missing required field: {key}"
    if "unit" in args:
        assert args["unit"] in props["unit"]["enum"], "unit must be C or F"
    assert isinstance(args["city"], str) and args["city"], "city must be non-empty string"

def get_weather(city: str, unit: str = "C") -> Dict[str, Any]:
    key = (city, unit)
    assert key in WEATHER_DB, "city/unit not found in WEATHER_DB"
    return WEATHER_DB[key]

# 模拟模型生成的 function call
function_call = {
    "name": "get_weather",
    "arguments": {"city": "Boston", "unit": "C"},
}

assert function_call["name"] == tool_schema["name"]
validate_args(function_call["arguments"], tool_schema)
result = get_weather(**function_call["arguments"])

assert result["temperature"] == 6
assert result["condition"] == "Cloudy"
print(result)
```

上面这段代码体现了一个核心原则：不要把“模型产出的参数”直接当真。即使工具 schema 已经声明过，应用层仍然要再校验一次。

再看接近真实接口的 Python SDK 模板。这里的重点不是每个字段都背下来，而是理解四个对象：

- `function declaration`
- `tools`
- 模型响应里的 `function_call`
- 回传给下一轮的 `functionResponse` 与完整历史

```python
from google import genai
from google.genai import types

schedule_meeting = {
    "name": "schedule_meeting",
    "description": "Creates a meeting with attendees/date/time/topic.",
    "parameters": {
        "type": "object",
        "properties": {
            "attendees": {"type": "array", "items": {"type": "string"}},
            "date": {"type": "string"},
            "time": {"type": "string"},
            "topic": {"type": "string"},
        },
        "required": ["attendees", "date", "time", "topic"],
    },
}

client = genai.Client()
tools = types.Tool(function_declarations=[schedule_meeting])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="帮我约 Bob 和 Alice 明天下午三点讨论发布计划",
    config=types.GenerateContentConfig(
        tools=[tools]
    ),
)

parts = response.candidates[0].content.parts
for part in parts:
    if part.function_call:
        fn = part.function_call
        print(fn.name, fn.args)
        # 这里执行你自己的业务函数
        tool_result = {
            "status": "ok",
            "meeting_id": "mtg_123"
        }
        # 如果你手动维护历史，必须把原始 model part 一并放回，
        # 这样 thought_signature 才能保留下来
```

如果使用官方 chat/SDK，并把完整模型响应对象直接加入历史，`thought_signature` 往往会被自动处理；真正高风险的是 REST 手写历史时把 model parts 拆散了。

真实工程里，建议把执行循环写成下面这个结构：

| 步骤 | 应用要做什么 | 为什么 |
|---|---|---|
| 1 | 发送用户请求和工具声明 | 让模型知道有哪些可用能力 |
| 2 | 读取所有 `functionCall` parts | 不能只看第一个调用 |
| 3 | 并行或串行执行工具 | 取决于依赖关系 |
| 4 | 保留原始 model parts 与签名 | 为下一轮续上推理上下文 |
| 5 | 回传 `functionResponse` | 让模型继续生成最终答案 |
| 6 | 做参数校验与副作用确认 | 防止误调用真实系统 |

---

## 工程权衡与常见坑

Gemini 的优势很明显，但落地时有几个坑几乎一定会遇到。

第一，手动维护历史时漏传 `thought_signature`。这是最常见也最隐蔽的问题。很多人以为“函数名和参数拿到了，下一轮只要把 functionResponse 发回去就行”，结果 Gemini 3 在 function calling 场景直接返回 4xx。根因不是函数逻辑错，而是协议上下文断了。

第二，把 schema 设计成“数据库建模比赛”。Function Calling 的 schema 不是越完整越好，而是越能帮助模型决策越好。对模型来说，`enum`、明确的 `required`、清晰的描述都很有价值；过深的嵌套、过长的描述、模糊的可选对象，反而会增加歧义和 token 成本。

第三，把“参数结构正确”误当成“业务语义正确”。比如模型按 schema 生成了 `{"date":"2026-03-13","time":"15:00"}`，这不代表时区正确，也不代表会议室一定可用。Function Calling 保证的是接口入口更规整，不是业务结果自动正确。

第四，并行调用虽然强，但不是默认就该开。并行的前提是工具之间无依赖，且副作用可控。如果一个工具是“查数据”，另一个是“写数据库”，盲目并行可能让写入动作发生在校验之前。

第五，OpenAI strict structured outputs 与 Gemini 的目标函数不同。OpenAI 的强项是“严格按 schema 输出”，但官方文档也明确写了 strict 模式不兼容 parallel function calls，需要设 `parallel_tool_calls: false`。Gemini 更偏向 agent 流程能力，支持 parallel/compositional，但你要自己接受“协议更灵活，应用层责任更多”这个事实。

| 常见坑 | 表现 | 规避策略 |
|---|---|---|
| 漏传 `thought_signature` | Gemini 3 function calling 报 4xx | 保留完整 model response 或用官方 SDK chat |
| schema 过深过杂 | 调用不稳定、参数歧义、token 增长 | 保持扁平、短描述、少嵌套 |
| 忽略应用层校验 | 调用了错误对象或错误时间 | 对必填项、枚举、权限、时间再校验 |
| 误用并行 | 副作用顺序错乱 | 先画依赖图，再决定并发 |
| 只处理首个工具调用 | 漏掉同轮其他 calls | 遍历所有 `functionCall` parts |
| 把 SDK 自动处理当成 REST 也自动 | 手写请求时状态丢失 | 明确区分 SDK 自动循环与手动协议循环 |

---

## 替代方案与适用边界

如果你的目标是“参数必须 100% 匹配 schema”，OpenAI 的 Structured Outputs 更像一把直尺。直尺的意思是“输出边界更硬”。官方说明里写得很清楚：`strict: true` 时模型输出会匹配提供的 JSON Schema，但它和 parallel function calls 不兼容，需要关闭并行。

如果你的目标是“一个模型自己协调多个工具，甚至做多步链式调用”，Gemini 更自然。因为它支持 parallel 和 compositional，而且还能和 Google Search、Code Execution 等原生能力组合。

可以把两者理解成两种工程取向：

| 维度 | Gemini Function Calling | OpenAI Structured Outputs |
|---|---|---|
| 核心目标 | 工具选择 + 多步调用 + 状态闭环 | 严格 schema 匹配 |
| 并行工具调用 | 支持 | strict 模式下不兼容 |
| 组合调用 | 支持 | 需要更多外层编排 |
| thinking / 推理签名 | 有 `thought_signature` 约束 | 无对应机制 |
| 手动历史管理难度 | 更高 | 相对更低 |
| 适合场景 | agent、多工具流水线、原生工具协作 | 高确定性字段抽取、格式化输出 |

一个简单判断标准是：

- 如果你要做表单抽取、工单字段生成、数据库过滤条件生成，且格式正确率优先，OpenAI strict 更直接。
- 如果你要做“搜索 + 执行 + 业务 API”混合流水线，且希望模型自己决定工具顺序，Gemini 更合适。

真实工程例子：

“同时拉取新闻、运行脚本、写入内部数据库，并最终生成摘要”。

这类需求通常包含三类动作：

1. 外部信息获取
2. 中间计算
3. 有副作用的写入

在 OpenAI strict 路线里，你通常会把“每一步的结构化结果”做得很稳，再由外层 orchestrator 串起来；在 Gemini 路线里，则更容易把多工具协作交给模型，但数据库写入这类高风险动作仍然建议要求显式确认，或者让写入函数只产出 dry-run 计划。dry-run 的意思是“先给执行计划，不真正落库”。

因此，Gemini 的适用边界不是“万能 agent”，而是“在工具定义清晰、依赖关系明确、应用层校验充分时，能显著减少外层流程代码”。

---

## 参考资料

- [Gemini API: Function calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Gemini API: Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)
- [Gemini API: Thinking](https://ai.google.dev/gemini-api/docs/thinking)
- [OpenAI: Introducing Structured Outputs in the API](https://openai.com/index/introducing-structured-outputs-in-the-api/)
