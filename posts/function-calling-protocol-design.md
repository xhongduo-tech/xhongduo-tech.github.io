## 核心结论

Function Calling 的本质不是“让模型会调用函数”，而是“把自然语言意图压缩进一个可验证的参数合同”。合同就是 JSON Schema。Schema 越明确，模型的输出空间越小，参数越不容易漂移。

截至 2026 年 3 月，OpenAI、Anthropic、Gemini 都已经提供了结构化输出或严格工具调用能力，但三家的协议重点不同：

| 平台 | 主要入口 | 严格约束方式 | 对工程最关键的点 | 适合场景 |
|---|---|---|---|---|
| OpenAI | `tools[].function.parameters` 或 `response_format/json_schema` | `strict: true` + 约束解码 | 强调把 schema 编译成 CFG，递归结构表达力强 | 复杂工具调用、递归 JSON、严格执行 |
| Anthropic | `tools[].input_schema` + `strict: true`，或 `output_config.format` | 编译 grammar + schema 子集 | SDK 会自动把不支持的约束下沉到描述里，并补 `additionalProperties:false` | 工具链稳定、要同时做 JSON 输出和 tool use |
| Gemini | `response_mime_type` + `response_json_schema`，Gemini 3 预览支持与工具联用 | schema 子集 + 响应顺序控制 | 输出 key 顺序更可控，但官方明确建议业务侧继续校验语义正确性 | 信息提取、分类、固定结果格式 |

真正决定准确率的，不只是模型能力，而是协议设计。可以把可靠率粗略写成：

$$
\text{Reliability} \approx \text{ConstrainedDecoding}(\text{CFG(schema)}) \times \text{Coverage}(\text{enum/oneOf/\$ref}) \times \text{Prompting}
$$

这里的 Constrained Decoding 是“约束解码”，意思是模型每吐一个 token 时，只能从“当前 schema 允许的候选”里选；Coverage 是“约束覆盖率”，意思是你到底把多少歧义写进了 `enum`、`oneOf`、`$ref`；Prompting 则负责补足字段语义。

玩具例子最能说明问题。假设你只想让模型在摘要和翻译之间二选一：

```json
{
  "type": "object",
  "properties": {
    "mode": {
      "type": "string",
      "enum": ["summary", "translate"]
    }
  },
  "required": ["mode"],
  "additionalProperties": false
}
```

这时模型只能输出 `summary` 或 `translate`。它不能偷偷发明 `summarize`、`summary_mode`、`translation`。这就是协议设计直接减少错误。

---

## 问题定义与边界

问题不是“模型会不会写 JSON”，而是“模型输出的参数，能不能稳定通过验证并安全进入执行层”。很多失败都发生在这一步：JSON 是合法的，但参数不合法。

术语先说白话：

- `required`：必填字段，不给就算失败。
- `enum`：候选值白名单，只能从固定集合里选。
- `oneOf`：多种参数形状里只能选一种。
- `$ref`：把重复结构抽出来复用，避免多处定义漂移。
- `additionalProperties:false`：禁止额外字段，防止模型自创键名。

边界也要收紧。本文只讨论“通过 schema + prompting 控制模型输出”的协议问题，不讨论三类内容：

1. 工具执行后的业务兜底逻辑。
2. 后训练、微调、RLHF 对工具调用能力的提升。
3. 单纯的“先输出自然语言，再人工解析”。

一个典型新手错误是只写提示词，不写 schema。比如让模型“返回包含 `name` 和 `age` 的 JSON”，它可能输出：

```json
{"name":"Alice","年龄":18}
```

这段 JSON 语法没错，但对后端来说键名错了。加上 schema 后，验证器才有资格拒绝它。

更进一步，很多“silent failure”来自“结构上合法，语义上错”。比如 schema 里要求：

- `age: integer`
- `country: enum["CN","US","JP"]`

模型却给出：

```json
{"age":"18","country":"China"}
```

这就是典型的 schema mismatch。它往往不会在 API 层立刻炸掉，而是在函数执行、数据库写入、下游业务规则里晚一点爆炸，所以排查成本高。

---

## 核心机制与推导

核心机制可以概括成一句话：先把“意图空间”压缩成“参数空间”，再把“参数空间”压缩成“合法 token 空间”。

OpenAI 官方公开过其关键做法：把 JSON Schema 编译成 CFG。CFG 是“上下文无关文法”，白话说就是一组能描述合法结构层级的规则。模型生成时，不是从整个词表里随便挑，而是每一步只看文法允许的候选。这样就能强制满足对象层级、数组闭合、字段名和递归结构。

为什么 `enum`、`oneOf`、`$ref` 会显著提高调用准确率？因为它们在缩小分支数。

假设没有枚举，字段 `action` 的理论候选是所有字符串，模型可能输出 `create`、`new`、`insert`、`make_record`。  
加上：

```json
"action": { "type": "string", "enum": ["create", "update"] }
```

那就只剩两个合法分支。

`oneOf` 更重要。它适合表达“参数形状互斥”。例如天气查询既支持“城市名”，也支持“经纬度”，但不能混用：

```json
{
  "oneOf": [
    {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"],
      "additionalProperties": false
    },
    {
      "type": "object",
      "properties": {
        "lat": {"type": "number"},
        "lon": {"type": "number"}
      },
      "required": ["lat", "lon"],
      "additionalProperties": false
    }
  ]
}
```

这可以理解成“先选表单模板，再填字段”。如果不用 `oneOf`，模型容易输出半套城市名加半套经纬度。

`$ref` 的价值则在于复用。白话说，它是“别把同一结构抄三遍”。一旦你把地址、联系人、分页对象等结构复用起来，协议变更时只需要改一处，减少 schema drift，也减少提示与工具定义不一致。

但要注意：结构表达力更强，不等于运行更稳。社区汇总的 JSONSchemaBench 数据显示，GPT-4 Turbo 在扁平 schema 上有效率约 82%，深嵌套结构约 54%。这个数字不是官方基准，但足够说明一个趋势：嵌套一深，模型做结构推理和长距离闭合的负担会明显上升。因此很多真实工程会把业务对象先扁平化，再由后端 mapper 还原层级。

---

## 代码实现

先看一个最小可运行的 Python 例子。它不依赖任何模型 API，只演示“协议设计怎样影响验证结果”。

```python
from jsonschema import validate, ValidationError

strict_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "action": {"type": "string", "enum": ["create", "update"]}
    },
    "required": ["name", "action"],
    "additionalProperties": False
}

good = {"name": "alice", "action": "create"}
bad_extra = {"name": "alice", "action": "create", "mode": "fast"}
bad_enum = {"name": "alice", "action": "summary_mode"}

validate(instance=good, schema=strict_schema)

try:
    validate(instance=bad_extra, schema=strict_schema)
    assert False, "bad_extra 应该失败"
except ValidationError:
    pass

try:
    validate(instance=bad_enum, schema=strict_schema)
    assert False, "bad_enum 应该失败"
except ValidationError:
    pass

assert good["action"] in ["create", "update"]
```

这个玩具例子说明两件事：

1. `enum` 把语义映射成固定动作集合。
2. `additionalProperties:false` 阻止模型塞入未声明字段。

再看三家接口的最小协议形状。下面是伪请求，重点不是 SDK 语法，而是 schema 放在哪里。

```json
{
  "openai": {
    "tools": [{
      "type": "function",
      "function": {
        "name": "upsert_user",
        "description": "Create or update a user",
        "strict": true,
        "parameters": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "action": {"type": "string", "enum": ["create", "update"]}
          },
          "required": ["name", "action"],
          "additionalProperties": false
        }
      }
    }]
  },
  "anthropic": {
    "tools": [{
      "name": "upsert_user",
      "description": "Create or update a user",
      "strict": true,
      "input_schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "action": {"type": "string", "enum": ["create", "update"]}
        },
        "required": ["name", "action"],
        "additionalProperties": false
      }
    }]
  },
  "gemini": {
    "generationConfig": {
      "responseMimeType": "application/json",
      "responseJsonSchema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "action": {"type": "string", "enum": ["create", "update"]}
        },
        "required": ["name", "action"],
        "additionalProperties": false
      }
    }
  }
}
```

真实工程例子通常不是这么简单。比如客服助手要调用退款、查订单、查物流、查地址、用户画像五个工具。如果直接把后端原始对象暴露给模型：

- `customer.profile.address.city`
- `customer.profile.address.zip_code`
- `order.fulfillment.delivery.expected_at`

模型很容易在嵌套层级里丢字段、错字段、拼错路径。更稳的办法是把工具层做成扁平参数：

- `customer_id`
- `order_id`
- `city`
- `zip_code`
- `expected_delivery_date`

然后由后端 mapper 把平铺参数映射回领域对象。模型负责“分类与填表”，业务层负责“对象组装与约束校验”。这比让模型直接操纵深对象可靠得多。

---

## 工程权衡与常见坑

Function Calling 的稳定性，通常不是毁在“大错”，而是毁在“小但持续的协议噪声”。

| 常见故障 | 触发条件 | 结果 | 规避策略 |
|---|---|---|---|
| schema mismatch | 字段类型、键名、层级与后端不一致 | 422、业务侧异常、静默脏数据 | 上线前双向校验，版本化工具 |
| 缺 `additionalProperties:false` | 模型能自由加键 | 出现垃圾字段、错误分支 | 所有 object 显式禁止额外字段 |
| 缺 `enum` | 候选动作靠自然语言理解 | 词义漂移，如 `summary_mode` | 能枚举就枚举 |
| 缺 `oneOf` | 多种调用形状混在一个对象里 | 半套参数拼接 | 用互斥分支建模 |
| schema 太深 | 4 层以上对象、数组嵌对象 | 丢字段、闭合错误、漏必填 | 扁平化、拆工具、分两步调用 |
| 工具太多 | 一次暴露 10+ 工具 | 错路由、幻觉函数名 | 每轮只暴露相关工具 |

这里有一个很典型的坑。你后端实际要求：

```json
{
  "address": {
    "city": "Shanghai"
  }
}
```

模型却返回：

```json
{
  "city": "Shanghai"
}
```

人眼看完全能理解，但验证一定失败。这不是模型“笨”，而是协议没有把层级讲清楚，或者工具定义和提示词存在冲突。

另一个权衡是“强约束 vs 灵活性”。约束越强，错误率越低，但 schema 编译成本、首请求时延、复杂度限制都会更明显。OpenAI 和 Anthropic 都明确提到新 schema 首次请求会有额外延迟，因为要先编译 grammar。Anthropic 还公开了复杂度限制，例如严格 schema 中可选参数、union 类型、strict tools 数量都有上限。这意味着设计工具协议时，不能一味追求“全表达力”，而要优先考虑“可编译、可维护、可观测”。

生产环境里更重要的一条经验是：把“错误暴露得早”。PithyCyborg 总结的失败模式很典型，很多调用失败并不会在 API 层报错，而是以错误参数、缺失字段、幻觉工具名的形式向后传播。所以必须在模型输出和函数执行之间加一层 validator，并把错误编码成机器可读反馈，而不是直接把异常吞掉。

---

## 替代方案与适用边界

严格 schema 不是唯一方案，但它通常是最稳的第一方案。它最适合三类场景：

1. 工具参数稳定，字段少且明确。
2. 动作集合有限，能枚举。
3. 错一次成本很高，比如订单、支付、权限、工单。

如果你的工具高度动态，比如字段随租户配置变化、搜索过滤条件非常自由、实时 schema 经常变，那就不要把全部复杂性直接塞给 strict schema。更现实的方案是“两阶段”：

1. 模型先输出 reasoning + 参数草稿。
2. validator/mapper 再把草稿转成最终函数调用。

伪代码如下：

```text
Step 1: LLM 输出
{
  "tool_name": "search_order",
  "draft_args": {
    "customer": "alice",
    "date_hint": "last month",
    "status_hint": "late delivery"
  }
}

Step 2: Validator / Mapper
- 解析自然语言日期 -> 2026-02-01 ~ 2026-02-28
- 规范状态 -> "delayed"
- 生成最终调用:
search_order(customer_id="u_123", from_date="2026-02-01", to_date="2026-02-28", status="delayed")
```

这种做法的优点是更灵活，容易吸收 schema 变化；缺点是准确率通常不如直接 strict function calling，因为模型第一步仍然在自由文本和半结构化参数之间活动。

因此可以把边界总结成一句话：  
如果你能把问题改写成“有限表单填写”，就用严格 schema；如果问题本质上仍是“开放式理解后再映射”，就用草稿层 + 验证层，不要假装一个 schema 能覆盖全部业务复杂性。

---

## 参考资料

| 来源 | 内容聚焦 | 链接 |
|---|---|---|
| OpenAI 官方 | Structured Outputs、`strict:true`、CFG 约束解码、递归 schema | https://openai.com/index/introducing-structured-outputs-in-the-api/ |
| OpenAI 官方文档 | strict function calling 的要求：所有属性 required、`additionalProperties:false` | https://platform.openai.com/docs/guides/function-calling/how-do-i-ensure-the-model-calls-the-correct-function |
| Anthropic 官方博客 | Claude Structured Outputs 上线与适用场景 | https://claude.com/blog/structured-outputs-on-the-claude-developer-platform |
| Anthropic 官方文档 | `output_config.format`、`strict:true`、复杂度限制、grammar cache | https://platform.claude.com/docs/en/build-with-claude/structured-outputs |
| Gemini 官方文档 | `response_json_schema`、schema 子集、property ordering、业务侧继续验证 | https://ai.google.dev/gemini-api/docs/structured-output |
| PithyCyborg | 生产中 silent failure 的四类模式与验证层建议 | https://www.pithycyborg.com/why-do-ai-function-calls-fail-silently-in-production/ |
| Codastra / Medium | 工具裁剪、幂等键、invalid-args 约 12% 降到 2.1% 的实战案例 | https://medium.com/@2nick2patel2/llm-function-calling-pitfalls-nobody-mentions-a0a0575888b1 |
| 社区汇总资料 | JSONSchemaBench 扁平 82% vs 深嵌套 54% 的二手整理，适合作为趋势参考，不宜当官方基准 | https://gist.github.com/donbr/1509eda1d753bbd25d899748a4a15a60 |
