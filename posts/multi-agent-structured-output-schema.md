## 核心结论

多 Agent 链路里，最容易丢信息的地方不是模型“不会回答”，而是上游给了自由文本，下游却想把它当成程序输入直接消费。**Schema**，即“数据长什么样的合同”，解决的是这个问题：每条消息不只传内容，还同时传“字段名、类型、必填项、约束”。

最稳的做法不是“提示模型输出 JSON”，而是建立一个三段循环：

1. 上游 Agent 声明输出 Schema。
2. 下游 Agent 或运行时先校验，再使用。
3. 如果 Schema 不兼容，先走适配器，再进入业务逻辑。

这可以写成一个最小通信模型：

$$
Agent_i \xrightarrow[]{declare\ S} D(S), \quad
Agent_j: validate(D, S)
$$

若下游期望的 Schema 为 $S'$，则：

$$
D(S) \not\models S' \Rightarrow adapt(D, S \to S') \Rightarrow validate(D', S')
$$

对工程实现来说，`Pydantic BaseModel` 更适合 Python 内部强类型开发，`JSON Schema` 更适合跨语言、跨框架、跨模型协商。前者像 Python 世界里的类型定义，后者像链路层协议。

| 传递方式 | 强类型校验 | 典型问题/收益 |
|---|---|---|
| 自由文本 | 无 | 字段缺失、命名漂移、类型错误、下游要重复解析 |
| JSON + 口头约定 | 弱 | 看起来像结构化，实则仍靠提示词自觉 |
| JSON Schema + 校验 | 强 | 可直接消费、可自动重试、可插入适配器 |
| Pydantic + Schema导出 | 强 | Python 端开发体验好，同时能向外输出标准 Schema |

---

## 问题定义与边界

这里讨论的“结构化输出与 Schema 协商”，不是让模型写得更整齐，而是让多个 Agent 之间交换**可验证对象**。可验证对象的意思是：程序可以在运行时判断“这个输出是否满足约定”。

问题边界有三条。

第一，结构化输出解决的是**格式可靠性**，不是**事实正确性**。一个字段即使类型正确，也可能内容是错的。比如 `email` 是字符串，不代表它一定是真邮箱。

第二，结构化输出要求链路参与方共享至少一份可比较的 Schema。如果 Agent A 输出的是 `count`，Agent B 要的是 `total`，两边即使都“输出 JSON”，也仍然不兼容。

第三，Schema 协商只适合“对象化任务”，不适合所有对话任务。比如诗歌创作、长篇解释、开放式讨论，本质目标不是稳定字段，就不必强行结构化。

玩具例子最容易看清这个边界：

- Agent A 输出：`{"count": 3}`
- Agent B 期待：`{"total": 3}`

这不是模型能力问题，而是接口不匹配问题。没有 Schema 时，下游常见写法是“如果有 `count` 就映射成 `total`”，这会把协议知识散落在业务代码里。随着 Agent 数量增长，映射规则会失控。

---

## 核心机制与推导

自由文本之所以在多 Agent 中容易失真，是因为它把三层信息揉在一起了：

1. 语义层：用户真正要表达什么。
2. 表示层：字段如何命名、如何嵌套。
3. 校验层：哪些字段必填、类型是什么、范围是否合法。

结构化协议的核心，就是把后两层显式化。

一个可执行的最小链路通常长这样：

1. Agent A 声明输出 Schema `S_A`
2. 运行时要求模型只生成满足 `S_A` 的数据
3. 产出对象 `D`
4. Agent B 接收前，先验证 `D ⊨ S_B`
5. 若 `S_A != S_B`，调用适配器 `A_{A\to B}`

其中“适配器”可以理解成翻译层，但它翻译的不是自然语言，而是字段结构。例如：

| 来源字段 | 目标字段 | 转换规则 |
|---|---|---|
| `count` | `total` | 重命名 |
| `price_cents` | `price` | 除以 100 |
| `tags: string` | `tags: string[]` | 按逗号切分 |
| `created_at: string` | `created_at: datetime` | 解析时间格式 |

这套机制的关键收益，不在于“模型更聪明”，而在于系统把错误前移了。原本错误是在数据库写入时报 `KeyError`、在报表聚合时报类型异常；现在错误在消息交接时就暴露。

真实工程例子：一个招聘工作流里，Agent 1 从简历 PDF 提取联系人，Agent 2 写入 CRM，Agent 3 触发邮件。若 Agent 1 返回自由文本，下游至少要做三件额外工作：切句、抽字段、兜底重试。若 Agent 1 直接返回符合 Schema 的 `ContactInfo`，Agent 2 只要读 `name/email/phone` 三个字段即可。链路从“文本理解问题”变成“对象传递问题”，复杂度显著下降。

---

## 代码实现

下面是一个可运行的玩具实现：用简化版 JSON Schema 校验消息，并在字段不兼容时自动适配。它不依赖第三方库，但足够说明“声明-校验-适配-再校验”的闭环。

```python
from typing import Any, Dict

Schema = Dict[str, Any]

source_schema: Schema = {
    "type": "object",
    "required": ["count"],
    "properties": {
        "count": {"type": "integer"}
    }
}

target_schema: Schema = {
    "type": "object",
    "required": ["total"],
    "properties": {
        "total": {"type": "integer"}
    }
}

def validate(data: Dict[str, Any], schema: Schema) -> None:
    assert schema["type"] == "object"
    for key in schema.get("required", []):
        assert key in data, f"missing required field: {key}"

    for key, rule in schema.get("properties", {}).items():
        if key not in data:
            continue
        value = data[key]
        expected = rule["type"]
        if expected == "integer":
            assert isinstance(value, int) and not isinstance(value, bool), f"{key} should be integer"
        elif expected == "string":
            assert isinstance(value, str), f"{key} should be string"
        else:
            raise AssertionError(f"unsupported type: {expected}")

def adapt_count_to_total(data: Dict[str, Any]) -> Dict[str, Any]:
    if "count" in data and "total" not in data:
        return {"total": data["count"]}
    return data

# Agent A 声明并输出
message = {"count": 3}
validate(message, source_schema)

# Agent B 直接消费会失败
failed = False
try:
    validate(message, target_schema)
except AssertionError:
    failed = True

assert failed is True

# 走适配器后再校验
adapted = adapt_count_to_total(message)
validate(adapted, target_schema)
assert adapted == {"total": 3}

print("schema negotiation passed")
```

如果进入 Python 工程栈，通常会把“内部类型”交给 Pydantic，把“跨 Agent 协议”交给 JSON Schema。一个常见模式是：

- 用 `BaseModel` 定义 `ContactInfo`
- 运行时从 `ContactInfo.model_json_schema()` 导出 JSON Schema
- 将该 Schema 交给支持结构化输出的模型/Agent SDK
- 返回结果后再反序列化成 `ContactInfo`

这样做的好处是，开发者写的是 Python 类，但链路上传递的是标准 Schema。也就是说，**Pydantic 负责开发体验，JSON Schema 负责协商兼容性**。

---

## 工程权衡与常见坑

结构化输出不是“加个 Schema 就结束”，它会引入新的工程约束。

| 坑 | 影响 | 规避 |
|---|---|---|
| Schema 频繁变化 | 首次编译/缓存失效，延迟上升 | 稳定字段，给 Schema 做版本号 |
| 嵌套过深或递归过多 | 提供方可能直接拒绝请求 | 拆小对象，限制递归层级 |
| 只校验格式不校验业务 | 数据“合法但没用” | 增加业务级二次校验 |
| 字段名漂移 | 下游大量隐式兼容逻辑 | 建显式适配器，不写魔法兜底 |
| 把 Pydantic 当协议标准 | 跨语言协作困难 | 外部接口统一落到 JSON Schema |

一个常见误区是：既然用了 Pydantic，就不需要 JSON Schema。这个结论只在“单 Python 进程内部”成立。一旦跨语言、跨服务、跨供应商，Pydantic 不再是公共协议，只是本地类型系统。

另一个常见坑是过度追求“一次定义，永远兼容”。现实里 Schema 会演进，正确做法不是阻止变化，而是显式版本化。例如 `ContactInfoV1` 到 `ContactInfoV2` 增加 `company` 字段时，下游可以声明自己接受哪个版本，适配器只处理相邻版本迁移。

---

## 替代方案与适用边界

如果模型或框架支持原生结构化输出，优先使用原生能力。原因很简单：约束发生在生成阶段，而不是生成后补救。

| 方案 | 何时用 | 优点 | 缺点 |
|---|---|---|---|
| Provider 原生结构化输出 | 模型直接支持 Schema | 最稳，失败点前移 | 依赖供应商能力 |
| Tool Strategy | 模型不原生支持，但支持工具调用 | 兼容面广 | 多一步工具协议 |
| JSON Mode | 只能要求“输出 JSON” | 接入快 | 仍需自行校验 |
| 自由文本 + 正则/解析 | 临时脚本、一次性任务 | 开发最省 | 扩展性最差 |

适用边界也要说清楚。

- 单 Agent 给人类看结果，且结果不进数据库、不进程序分支时，结构化输出不是必须。
- 多 Agent 串联，尤其是“抽取 -> 判断 -> 写库 -> 调用 API”这种链路，结构化输出几乎是基础设施。
- 当任务输出天然是表单、配置、标签、计划、工单、联系人、检索结果时，Schema 收益最高。
- 当任务输出是长文解释、创意写作、探索式推理时，结构化只应包住“最终摘要对象”，不要把整个推理过程都塞进严格 Schema。

---

## 参考资料

| 资料 | 内容亮点 |
|---|---|
| LangChain Structured Output | `response_format`、`ProviderStrategy`、`ToolStrategy`、`structured_response`/`structuredResponse` |
| LangChain Reference | `StructuredOutputValidationError` 等错误类型与处理策略 |
| Claude Agent SDK Structured Outputs | 多轮工具后返回校验过的 `structured_output` |
| Claude Structured Outputs | JSON Schema 约束、缓存、复杂 Schema 限制、与 citations/prefill 的兼容边界 |
| Agno Structured Output | `output_schema`、Pydantic 对象直返、运行时覆盖 schema |
| Gemini Structured Outputs | 原生 JSON Schema、字段顺序保持、Pydantic/Zod 兼容 |

- LangChain 文档：https://docs.langchain.com/oss/python/langchain/structured-output
- LangChain Reference：https://reference.langchain.com/python/langchain/agents/structured_output
- Claude Agent SDK：https://platform.claude.com/docs/en/agent-sdk/structured-outputs
- Claude Structured Outputs：https://docs.claude.com/en/docs/build-with-claude/structured-outputs
- Agno 文档：https://docs.agno.com/input-output/structured-output/agent
- Gemini API 文档：https://ai.google.dev/gemini-api/docs/structured-output
- Google 官方博客：https://blog.google/innovation-and-ai/technology/developers-tools/gemini-api-structured-outputs/
