## 核心结论

函数调用（Function Calling）可以理解为：模型不再用自然语言“描述它想做什么”，而是直接按约定好的结构化格式提出工具调用请求。这个结构化格式通常由 JSON Schema 定义。JSON Schema 是一种“参数说明书”，用来约束字段名、类型、必填项和枚举值。

它解决的不是“模型会不会思考”，而是“模型如何把思考结果稳定交给程序执行”。在工程上，这条链路可以写成：

$$
Intent \rightarrow JSON\ Schema \rightarrow tool\_calls(name, arguments)
$$

也就是：用户意图先被映射到某个工具定义，再由模型输出 `tool_calls`，最后由宿主程序执行。

一个最小玩具例子是“计算两个数之和”。开发者先定义工具 `sum_calc`，要求参数里必须有 `x` 和 `y`。模型如果判断用户是在请求计算，就不该输出“3 加 5 等于 8”这种自由文本，而应该输出类似：

```json
{
  "tool_calls": [
    {
      "name": "sum_calc",
      "arguments": {
        "x": 3,
        "y": 5
      }
    }
  ]
}
```

平台执行后，把结果 `{"result": 8}` 作为工具返回消息再发给模型，模型再决定是否向用户组织最终答案。这样就形成了“意图识别 -> 工具执行 -> 结果回流”的闭环。

下面这张表能看出自然语言和函数调用的本质差异：

| 形式 | 输出示例 | 机器可执行性 | 稳定性 |
|---|---|---:|---:|
| 自然语言输出 | “我建议你调用加法工具，参数是 3 和 5” | 低 | 低 |
| 函数调用 JSON | `{"name":"sum_calc","arguments":{"x":3,"y":5}}` | 高 | 高 |

结论可以压缩成三点：

| 结论 | 含义 | 工程价值 |
|---|---|---|
| Schema 是边界 | 模型只能在给定字段和类型里填值 | 降低解析歧义 |
| `strict` 是约束 | 强制输出满足 schema 的 JSON | 提高可预测性 |
| 并行调用是优化 | 一轮同时发起多个工具请求 | 降低整体延迟 |

---

## 问题定义与边界

问题定义很明确：如何让开放式对话系统可靠地调用预定义外部工具，而不是把“调用意图”混在自然语言里。

这里的“可靠”至少包含三层含义：

| 问题 | 限制 | 成功条件 |
|---|---|---|
| 模型是否选对工具 | 只能在已注册工具集合内选择 | 工具名匹配正确 |
| 模型是否传对参数 | 参数必须满足 JSON Schema | 参数合法且完整 |
| 系统是否能执行 | 工具真实存在且后端能处理 | 返回可消费结果 |

因此，函数调用不是“让模型自动连任何系统”，而是“让模型在一组预定义接口里做受控选择”。这就是它的边界。

如果一个 schema 写成：

```json
{
  "type": "object",
  "properties": {
    "city": { "type": "string" }
  },
  "required": ["city"]
}
```

那么 `city` 就是必填字段。必填字段的白话解释是：少了这个字段，工具就没有足够信息执行。模型如果输出 `{}` 或 `{ "country": "CN" }`，这次调用就应该被判为非法，不能进入真实执行。

这件事可以形式化为：

$$
success = valid\_json(tool\_calls) \land parameters \models schema
$$

含义是：成功不仅要求输出是合法 JSON，还要求参数满足 schema 约束。只满足其中一个都不够。

函数调用还有几个常见边界，初学者容易混淆：

| 边界 | 说明 |
|---|---|
| 不是自动联网 | 模型只提出调用请求，真正联网的是你的程序 |
| 不是数据库查询语言 | schema 定义的是接口参数，不是任意查询语法 |
| 不是万能流程编排器 | 多步任务仍需宿主程序控制状态和重试 |
| 不是安全边界本身 | 参数合法不等于业务安全，还要做服务端校验 |

真实工程里，模型和外部系统之间至少隔着两层防线：一层是 schema 校验，一层是业务校验。比如“转账金额”即使类型正确、字段齐全，也不能跳过权限、额度、风控检查。

---

## 核心机制与推导

函数调用的核心对象通常有三个：`tools`、模型响应里的 `tool_calls`、以及工具执行结果回传。

`tools` 可以理解为“给模型看的接口目录”。每个工具通常包含三部分：

| 字段 | 作用 | 白话解释 |
|---|---|---|
| `name` | 工具唯一标识 | 模型调用时写哪个名字 |
| `description` | 工具用途说明 | 告诉模型什么时候该用它 |
| `parameters` | 参数 schema | 告诉模型参数该怎么填 |

模型的推导过程可以抽象成：

$$
tool\_calls = model(response \leftarrow conversation, tools, strict)
$$

这里 `strict` 的意思是“严格约束模式”。它不是让模型更聪明，而是让输出空间更窄。输出空间越窄，违反 schema 的概率通常越低。

`strict=false` 和 `strict=true` 的差异可以概括为：

| 模式 | 模型自由度 | schema 违反概率 | 典型问题 |
|---|---:|---:|---|
| `strict=false` | 高 | 较高 | 字段缺失、类型漂移、额外字段 |
| `strict=true` | 低 | 较低 | schema 过复杂时更容易直接失败 |

这里还涉及一个重要机制：Constrained Decoding，通常可理解为“受约束解码”。白话说，就是生成时不是任由模型写任何 token，而是只允许它写出符合某种语法或 schema 的 token 序列。Grammar Sampling 也是类似思想，即按语法规则采样，而不是完全自由生成。它们的工程意义是：JSON 合法性不再主要依赖模型“记住 JSON 长什么样”，而是由解码约束直接保证。

一个直观推导是：

1. 对话上下文告诉模型用户意图。
2. `description` 帮模型做工具路由。
3. `parameters` 帮模型做参数填充。
4. `strict` 和约束解码帮模型把输出压进合法结构。
5. 宿主程序执行工具并返回结果。

真实工程例子可以看多城市天气查询。用户问：“比较 1900 年代三个城市今天的天气风险，给我一个概览。”这里的“1900s”如果产品定义为三个预置城市集合，例如 London、New York、Paris，那么同一轮模型可以返回三个并行工具调用：

```json
{
  "tool_calls": [
    {"name":"get_weather","arguments":{"city":"London"}},
    {"name":"get_weather","arguments":{"city":"New York"}},
    {"name":"get_weather","arguments":{"city":"Paris"}}
  ]
}
```

这就是 `parallel_tool_calls`。它的白话解释是：一轮里允许模型同时开出多个工具请求，后端并行执行，减少串行等待时间。

如果三个天气 API 各耗时 700ms，那么串行大约要 $3 \times 700 = 2100ms$，并行在理想情况下接近：

$$
T_{parallel} \approx \max(T_1, T_2, T_3)
$$

也就是约 700ms 到 900ms 量级，而不是 2100ms。对智能体系统，这种优化非常直接，因为外部 I/O 往往比模型推理更慢。

---

## 代码实现

先看一个可运行的 Python 玩具实现，它不依赖真实 API，只演示“schema 校验 -> 工具执行 -> 返回结果”的闭环。

```python
import json

TOOLS = {
    "sum_calc": {
        "required": ["x", "y"],
        "types": {"x": int, "y": int},
    }
}

def validate_arguments(tool_name, arguments):
    schema = TOOLS[tool_name]
    for key in schema["required"]:
        assert key in arguments, f"missing required field: {key}"
    for key, expected_type in schema["types"].items():
        assert isinstance(arguments[key], expected_type), f"{key} type error"
    return True

def execute_tool(tool_name, arguments):
    validate_arguments(tool_name, arguments)
    if tool_name == "sum_calc":
        return {"result": arguments["x"] + arguments["y"]}
    raise ValueError(f"unknown tool: {tool_name}")

# 模拟模型返回的 tool call
tool_call_json = """
{
  "tool_calls": [
    {
      "name": "sum_calc",
      "arguments": {
        "x": 3,
        "y": 5
      }
    }
  ]
}
"""

payload = json.loads(tool_call_json)
tool_call = payload["tool_calls"][0]
result = execute_tool(tool_call["name"], tool_call["arguments"])

assert result == {"result": 8}
assert validate_arguments("sum_calc", {"x": 1, "y": 2}) is True
print(result)
```

这个例子里最关键的不是“做加法”，而是调用链结构是稳定的：

1. 模型返回 `tool_calls`
2. 程序校验参数
3. 程序执行工具
4. 程序把结果再送回模型

下面是更接近实际接入方式的 JS/TS 伪代码。重点看 `tools`、`strict`、`parallel_tool_calls` 和工具结果回传。

```ts
const tools = [
  {
    type: "function",
    function: {
      name: "sum_calc",
      description: "计算两个整数之和。仅当用户明确要求加法计算时调用。",
      parameters: {
        type: "object",
        properties: {
          x: { type: "integer", description: "第一个加数" },
          y: { type: "integer", description: "第二个加数" }
        },
        required: ["x", "y"],
        additionalProperties: false
      },
      strict: true
    }
  },
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "返回指定 city 当前天气和未来3小时降雨概率。",
      parameters: {
        type: "object",
        properties: {
          city: { type: "string", description: "城市名，使用英文标准名" }
        },
        required: ["city"],
        additionalProperties: false
      },
      strict: true
    }
  }
];

const response = await client.responses.create({
  model: "gpt-4.1",
  input: [
    { role: "user", content: "帮我算 3 + 5，并顺便查一下 Shanghai 的天气" }
  ],
  tools,
  parallel_tool_calls: true,
  response_format: { type: "json_schema" }
});

for (const item of response.output || []) {
  if (item.type === "tool_call") {
    const { name, arguments: args, call_id } = item;

    let toolResult;
    if (name === "sum_calc") {
      toolResult = { result: args.x + args.y };
    } else if (name === "get_weather") {
      toolResult = await weatherService.getByCity(args.city);
    }

    await client.responses.create({
      model: "gpt-4.1",
      input: [
        {
          role: "tool",
          tool_call_id: call_id,
          content: JSON.stringify(toolResult)
        }
      ]
    });
  }
}
```

参数字段的职责可以汇总为：

| 字段 | 是否关键 | 作用 |
|---|---:|---|
| `name` | 是 | 模型选择哪个工具 |
| `description` | 是 | 帮助模型判断使用时机 |
| `parameters` | 是 | 定义参数结构 |
| `required` | 是 | 定义哪些字段不能省 |
| `strict` | 强烈建议 | 提高结构输出稳定性 |
| `parallel_tool_calls` | 按需 | 允许单轮多调用 |

实现时要记住一点：工具调用结果通常不是最终用户答案，而是“给模型的中间事实”。模型基于这些事实再生成最终答复，用户体验会更自然。

---

## 工程权衡与常见坑

函数调用最容易被低估的部分，不是 JSON，而是“描述设计”。`description` 写得越含糊，模型越容易路由错工具。这个规律可以粗略写成：

$$
error\_rate \propto ambiguity
$$

也就是歧义越大，错误率越高。

一个典型坑是把天气工具写成“查询天气”。这句话太宽，模型很可能把“新闻里的天气影响”“网页搜索天气”“长期气候统计”都混进来，最后错选 `search_web` 之类的工具。更好的写法是：“返回指定 city 当前温度、湿度和未来 3 小时降雨概率，仅用于实时天气查询。”这时工具边界就清楚得多。

常见坑和处理方式如下：

| 坑 | 现象 | 解决策略 |
|---|---|---|
| 描述过于笼统 | 模型误选工具 | 在 `description` 中写清输入、输出、适用场景 |
| schema 过复杂 | 模型频繁生成失败或漏字段 | 拆小工具，减少嵌套层级 |
| 未启用 `strict` | 参数类型漂移 | 对关键工具启用严格模式 |
| 必填字段过多 | 模型不敢调用或反复追问 | 只保留执行真正必要的字段 |
| 并行调用后处理混乱 | 多工具结果难以聚合 | 为每个调用保留 `call_id` 并建立聚合层 |
| 把 schema 当安全校验 | 非法业务仍被执行 | 服务器端再做权限、范围、风控校验 |

这里有一个很实际的权衡：schema 越详细，不一定越好。对模型来说，过多字段会增加理解和填充成本。比如一个简单天气查询，如果你要求 `city`、`country_code`、`timezone`、`unit`、`geo_source`、`client_platform` 六个字段，模型可能反而更容易缺字段，调用率下降。对新手最重要的经验是：先做最小可执行 schema，再按真实失败案例补字段。

并行调用也不是免费收益。它减少了等待时间，但会提高后端编排复杂度。你需要处理：

| 并行带来的问题 | 说明 |
|---|---|
| 结果乱序 | 返回顺序不一定和发起顺序一致 |
| 部分失败 | 三个调用里可能只有两个成功 |
| 结果聚合 | 需要统一格式再交还模型 |
| 与结构化输出约束冲突 | 某些场景下需在并行和最终结构严格性之间取舍 |

真实工程里，一个常见模式是：先允许并行拉取多个外部结果，再由服务端做第一次归一化，最后把干净结果交给模型总结。不要把所有脏数据清洗工作都留给模型。

---

## 替代方案与适用边界

函数调用不是唯一方案。它适合“参数明确、接口稳定、错误成本较高”的场景，但不适合所有问题。

常见替代方案有三类：

| 方案 | 做法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| Prompt 指令 | 让模型输出约定文本，再用 regex 解析 | 简单，接入快 | 脆弱，格式漂移大 | 工具很少、容错高 |
| 函数调用 | 用 schema 定义工具，模型输出 `tool_calls` | 稳定、可校验 | 设计成本更高 | 生产级工具调用 |
| 分步 hybrid | 先选工具，再做严格参数填充 | 灵活，可扩展 | 编排复杂 | 工具很多、异构性强 |

“客服系统”是一个容易理解的例子。如果系统里只有 2 到 3 个简单动作，例如“查订单”“查退款状态”，而且参数也只有订单号，那么 prompt + 正则解析有时就够用，因为开发成本最低。

但如果进入更复杂的场景，比如企业知识助手要同时查 CRM、工单系统、监控平台和日程接口，还要支持并发请求、字段校验、错误重试，那么函数调用通常更优，因为它提供了明确的机器接口。

当工具数量非常大、schema 很难一次设计清楚时，可以考虑 hybrid。典型流程是：

1. 第一阶段让模型只做工具选择。
2. 第二阶段对选中的工具使用更严格的 schema 生成参数。
3. 第三阶段由程序做业务校验和回退处理。

这个决策思路可以写成：

$$
Choice = \arg\max(accuracy\_cost\_balance)
$$

意思是：最终方案不是看谁“最先进”，而是看准确率和工程成本的平衡点。

适用边界可以这样判断：

| 场景特征 | 更适合的方案 |
|---|---|
| 工具少、参数简单、容错高 | Prompt 指令 |
| 工具固定、参数明确、需要稳定执行 | 函数调用 |
| 工具多、流程长、不同工具约束差异大 | 分步 hybrid |

对初级工程师来说，最实用的原则是：只要你已经开始写“解析模型文本里的半结构化字段”，而且这段解析代码越来越长，就该认真考虑迁移到函数调用。

---

## 参考资料

以下资料用于建立本文中的术语、流程和限制说明。这里特别保留“2025 年 3 月”这个时间点，是为了避免把后续版本变更误当成本文前提。

| 来源名称 | 更新日期 | 主要涵盖内容 |
|---|---|---|
| OpenAI Function Calling 指南 | 2025-03 | 函数调用基本流程、tools 定义、tool message 回传 |
| OpenAI Structured Outputs 说明 | 2024-08 文中说明，本文按 2025-03 使用语境引用 | `strict`、结构化输出、约束解码思想及限制 |
| OpenAI Platform Function Calling 文档 | 2025-03 使用语境 | `parallel_tool_calls`、schema 细节、调用行为 |
| Avian Function Calling 示例文档 | 2025-03 使用语境 | 多工具并行调用的工程示例 |

各资料的主要贡献可以概括为：

| 资料 | 关键贡献 |
|---|---|
| OpenAI 帮助中心文章 | 给出函数调用的基础定义和最小闭环 |
| Structured Outputs 文章 | 解释为什么“合法 JSON”不该只靠模型记忆 |
| Platform 文档 | 给出工具字段、严格模式、并行调用等接口层细节 |
| 第三方示例文档 | 展示多城市天气这类真实工程型编排示例 |

参考链接：

| 名称 | 链接 |
|---|---|
| OpenAI Help: Function Calling in the API | https://help.openai.com/en/articles/8555517-function-calling-in-the-openai-api |
| OpenAI: Introducing Structured Outputs in the API | https://openai.com/index/introducing-structured-outputs-in-the-api/ |
| OpenAI Platform Docs: Function Calling Guide | https://platform.openai.com/docs/guides/function-calling |
| Avian Docs: Function Calling | https://docs.avian.io/get-started/function-calling |
