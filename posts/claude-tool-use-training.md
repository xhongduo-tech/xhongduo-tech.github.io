## 核心结论

Claude 的工具使用能力，本质上是把“自然语言请求”稳定映射成“可执行的结构化调用”。这里的“结构化调用”就是符合 JSON Schema 的参数对象，而不是一段模糊文本。公开资料没有完整披露 Anthropic 的训练配方，但从官方文档、功能设计和外部评测看，可以合理推断：这类能力并不主要靠临时 prompt 拼出来，而是依赖专门面向工具调用的训练与对齐数据，让模型学会三件事：选工具、填参数、在多步流程里维持状态一致性。

最重要的工程结论有三个。

第一，`schema` 只解决“字段长什么样”，不解决“字段通常怎么填”。`input_examples` 的作用，是把格式约定、字段关联和常见默认值显式交给模型。Anthropic 新一轮工具能力文档也把 `input_examples` 定位为复杂输入场景的增强项；外部对该功能的整理引用了 Anthropic 的测试结果，称复杂参数处理正确率可从 72% 提升到 90%。

第二，复杂任务不应把每一步中间结果都塞回对话。Programmatic Tool Calling 可以理解为“让模型先写小脚本，再由脚本去调工具”。脚本在受控执行环境里串联多个工具，中间状态留在执行环境，不反复污染上下文。这对长链路、多服务、批量处理特别关键。

第三，大型工具库不能一次性全部喂给模型。Tool Search 的作用是“先找工具，再加载工具定义”。Anthropic 文档明确给出两个现实约束：工具定义会迅速膨胀上下文；当可选工具超过 30 到 50 个时，工具选择准确率会明显下降。因此，搜索式按需加载不是优化项，而是规模化所必需的基础设施。

可以把整个过程写成一个简化公式：

$$
\text{自然语言请求} + \text{tool schema} + \text{examples}
\rightarrow \text{JSON payload}
\rightarrow \text{tool result}
\rightarrow \text{最终回答}
$$

一个最小玩具例子是日历工具。工具只有 `start`、`end`、`summary` 三个字段。若只给 schema，模型知道 `start` 是字符串，却未必知道你要求 RFC3339 时间、UTC 时区，还是本地时间。再给 3 到 5 个真实例子，模型就更容易学到“日期格式”和“字段搭配关系”，这正是工具调用准确率提升的来源。

| 配置方式 | 模型知道什么 | 典型效果 |
|---|---|---|
| 仅 `schema` | 字段名、类型、必填项 | 能生成“合法 JSON”，但容易填错格式或漏语义 |
| `schema + input_examples` | 再额外知道真实填写模式 | 对复杂参数更稳定，尤其是日期、枚举、嵌套对象 |

---

## 问题定义与边界

“工具使用”不是普通问答里的知识补全，而是一类受约束的决策问题。模型要在一次或多次交互中完成如下链路：

1. 判断当前请求是否需要工具。
2. 在多个工具里选出合适工具。
3. 根据工具定义构造参数。
4. 接收返回值，并决定是否继续调用别的工具。
5. 最终把结果整理给用户。

这里的“JSON Schema”可以白话理解为“函数参数的正式说明书”，它规定字段名、字段类型、是否必填、取值范围。问题在于，说明书通常只能表达结构，难以表达习惯。例如“城市名要写成 `San Francisco, CA` 还是 `SF`”“时间是本地时区还是 UTC”，这些都不是纯结构问题。

因此，Claude 的工具使用问题边界可以拆成三层：

| 输入类型 | 作用 | 解决什么错误 |
|---|---|---|
| 自然语言请求 | 提供任务目标 | 决定是否调用工具、先做什么 |
| 工具信息：名字、描述、schema | 提供形式约束 | 防止字段缺失、类型错误 |
| 示例：`input_examples` | 提供使用习惯 | 降低格式错误、组合错误、歧义错误 |

一个新手容易忽略的边界是：工具调用能力不等于“自动完成任何业务”。如果工具返回值本身不稳定，或者工具描述模糊，模型再强也会出错。模型只能在给定接口契约下工作，不能替接口设计兜底。

再看一个具体问题。用户说：“取下周三所有在 SF 的会议，并按时间排序。”这并不是一步调用。系统可能要先找“日历查询工具”，再把“下周三”解析成精确时间窗口，把 “SF” 归一化成地点过滤条件，然后查询，再排序。若工具库很大，还可能先走 Tool Search。

所以更准确的公式是：

$$
\text{Request} \rightarrow \text{Tool Search} \rightarrow \text{JSON call} \rightarrow \text{必要时脚本编排} \rightarrow \text{合并结果}
$$

这里还有一个边界：不是所有任务都需要 Programmatic Tool Calling。单个工具、参数简单、没有中间状态时，普通 function calling 就够了。真正需要脚本编排的，是多步依赖、跨服务串联、批量循环和并行调用场景。

---

## 核心机制与推导

从机制上看，Claude 的工具使用可以分成四段。

第一段，Tool Search。它的作用不是直接执行业务，而是在大量工具里先缩小候选集。Anthropic 文档给了一个很直白的原因：多服务器工具定义可能先吃掉约 55K tokens，而 Tool Search 常能把这部分降低 85% 以上，并只加载当前真正需要的 3 到 5 个工具。这一步本质上是“检索”，不是“执行”。

第二段，schema 与 examples 联合约束。schema 解决“能不能过校验”，examples 解决“像不像真实调用”。对模型来说，这相当于把抽象规则和具体样例一起给出。若把参数生成看成条件概率问题，可简化写成：

$$
P(\text{payload} \mid \text{request}, \text{tool meta}, \text{examples})
$$

其中 `tool meta` 包含名字、描述、参数定义；`examples` 进一步收缩正确参数空间。examples 越贴近真实流量，条件分布越集中，模型乱填的概率越低。

第三段，普通工具调用或程序化调用。普通调用适合“单次判断后直接执行”。程序化调用适合“要先算、要循环、要分支、要保留中间变量”。Anthropic 的 Programmatic Tool Calling 文档明确描述了这一流程：Claude 先写 Python 代码，代码在沙箱容器里运行；当代码里调用某个工具时，执行暂停，系统返回 `tool_use`；外部提供工具结果后，代码继续跑。关键点在于，中间结果不必重新回灌到 Claude 的上下文窗口。

第四段，结果汇总。模型最后拿到的是已经执行完成、尽量压缩过的信号，而不是整个中间轨迹。这会降低上下文污染，也减少后续步骤受前面自然语言噪声干扰的风险。

完整机制可以写成：

```text
NL_request + tool_meta + examples
→ Tool Search → selected_tool
→ schema + input_examples → JSON_payload
→ (if needed) Programmatic Tool Calling → tool_result
→ Claude answer
```

玩具例子可以用一个三字段日历工具说明。用户说：“帮我约明天下午 3 点到 4 点的团队同步会。”模型若只见到 schema，可能输出：

- `start: "tomorrow 3pm"`
- `end: "tomorrow 4pm"`

这在人类看来能懂，但对 API 来说常常不可执行。若 examples 全是 RFC3339 格式，例如 `2026-03-20T15:00:00+08:00`，模型更可能对齐到该格式。也就是说，examples 并不是“多给几个例子而已”，而是在缩小输出语言的自由度。

真实工程例子是 Tines。Anthropic 客户案例显示，Tines 将 Claude 用于安全与 IT 自动化，出现过“把 120 步工作流压缩成单步 agent”的案例，并报告 100x 的 time-to-value 改善。这里的核心并不是“模型更会聊天”，而是模型在工具链里承担了编排器角色，把原先硬编码在工作流图中的逻辑，转成了按需生成的调用计划。

外部综述整理的 2024 年 9 月 BFCL 数据还显示，Claude 3.5 Sonnet 在该工具调用基准上达到 90.20%。这个数字不能被误读成“所有真实业务都有 90% 准确率”，因为基准测试是受控任务集合；但它至少说明，在“从请求到函数调用”这条狭义链路上，Claude 3.5 Sonnet 的能力已经进入可工程化使用区间。

---

## 代码实现

实现上，建议把每个工具都定义成四部分：名字、描述、JSON Schema、输入示例。名字解决可识别性，描述解决语义边界，Schema 解决结构合法性，示例解决填写习惯。

下面先给一个最小可运行的 Python 玩具实现。它不依赖 Claude API，只模拟“自然语言请求经过规则映射后，变成一个合格 payload”。重点是看结构，而不是看模型推理本身。

```python
from datetime import datetime, timedelta, timezone

def create_event_payload(start_iso: str, end_iso: str, summary: str) -> dict:
    payload = {
        "tool": "calendar.create_event",
        "arguments": {
            "start": start_iso,
            "end": end_iso,
            "summary": summary,
        }
    }
    assert payload["tool"] == "calendar.create_event"
    assert "T" in payload["arguments"]["start"]
    assert payload["arguments"]["end"] > payload["arguments"]["start"]
    assert len(payload["arguments"]["summary"]) > 0
    return payload

tz = timezone(timedelta(hours=8))
start = datetime(2026, 3, 20, 9, 0, tzinfo=tz).isoformat()
end = datetime(2026, 3, 20, 10, 0, tzinfo=tz).isoformat()

payload = create_event_payload(start, end, "Team sync")
assert payload["arguments"]["summary"] == "Team sync"
assert payload["arguments"]["start"].startswith("2026-03-20T09:00:00")
print(payload)
```

如果把它对应回 Claude 的工具定义，结构大致如下：

```python
calendar_tool = {
    "name": "calendar.create_event",
    "description": "创建日历事件。输入必须是带时区的 RFC3339 时间，summary 为面向参与者可读的标题。",
    "input_schema": {
        "type": "object",
        "properties": {
            "start": {"type": "string", "description": "RFC3339 开始时间"},
            "end": {"type": "string", "description": "RFC3339 结束时间"},
            "summary": {"type": "string", "description": "事件标题"}
        },
        "required": ["start", "end", "summary"]
    },
    "input_examples": [
        {
            "start": "2026-03-20T09:00:00+08:00",
            "end": "2026-03-20T10:00:00+08:00",
            "summary": "Team sync"
        },
        {
            "start": "2026-03-21T14:00:00+08:00",
            "end": "2026-03-21T14:30:00+08:00",
            "summary": "1:1 with Alice"
        }
    ]
}
```

对应的 payload 会长这样：

```python
payload = {
    "tool": "calendar.create_event",
    "arguments": {
        "start": "2026-03-20T09:00:00Z",
        "end": "2026-03-20T10:00:00Z",
        "summary": "Team sync"
    }
}
```

这个例子很小，但已经能说明为什么 examples 重要。因为 `string` 这个类型太宽，不能告诉模型该填哪种时间格式。

| 字段 | 含义 | 示例值 |
|---|---|---|
| `name` | 工具标识，等价于函数名 | `calendar.create_event` |
| `description` | 业务语义说明 | 必须带时区、适用场景 |
| `input_schema` | 参数结构约束 | `start/end/summary` |
| `input_examples` | 真实输入样本 | `2026-03-20T09:00:00+08:00` |

真实工程实现再往前一步，就是 Programmatic Tool Calling。比如处理 IT 工单流程：

1. 先查用户配额。
2. 若配额不足，创建审批工单。
3. 成功后通知团队频道。
4. 失败时写审计日志。

如果把每一步都当作普通聊天轮次，就会出现两个问题：中间数据越来越多；模型后续决策会被前面冗长结果干扰。程序化调用的思路是让 Claude 先生成 Python 脚本，例如先 `check_quota(user_id)`，再条件分支调用 `create_ticket(...)`，最后 `notify_team(...)`。脚本运行时保存变量，Claude 最后只接收聚合后的结果。

---

## 工程权衡与常见坑

最常见的误区，是把“工具会调用”误认为“系统已经稳了”。真正难的是稳定性，而稳定性取决于上下文管理和接口设计，不只是模型本身。

第一个坑是结果消息分散。Anthropic 文档对工具结果格式约束得很严：普通工具调用时，`tool_result` 必须紧跟相应的 `tool_use`；程序化调用时，等待中的响应消息甚至只能包含 `tool_result`，不能夹带别的文本。原因很简单，消息格式一乱，模型就难以恢复调用状态。

第二个坑是工具描述太短。很多团队只写一句“get weather”。这类描述对人或许够，对模型远远不够。官方建议恰恰相反：优先把描述写细，再考虑 examples。因为模型需要知道“不该什么时候用这个工具”，否则会误选。

第三个坑是没有示例。schema 能防止“结构错误”，但防不住“语义错填”。像日期格式、ID 样式、可选字段组合，都是 examples 更擅长表达的内容。

第四个坑是把所有工具一次性塞进 prompt。Tool Search 文档明确指出，大工具库会造成上下文膨胀，并且一旦可选工具超过 30 到 50 个，选错工具的概率会明显上升。很多人以为“给得越全越安全”，实际往往相反。

第五个坑是把程序化调用当成通用默认方案。它确实强，但也更复杂。要处理容器生命周期、工具返回格式、超时、代码注入风险，还要为 `allowed_callers`、容器过期时间等机制留出工程处理。

| 常见坑 | 典型后果 | 规避策略 |
|---|---|---|
| 结果消息分散或顺序错误 | 状态错乱、调用失败、400 错误 | 严格遵循 `tool_use → tool_result` 配对格式 |
| 只有 schema 没有示例 | 参数合法但语义不对 | 为复杂工具补 1 到 5 个真实例子 |
| 工具描述过短 | 误选工具、乱填参数 | 描述里写清用途、边界、返回值 |
| 一次加载全部工具 | token 浪费、选错率上升 | 用 Tool Search 按需加载 |
| 盲目上程序化调用 | 复杂度和安全风险升高 | 只在多步编排、批量、并行时使用 |

简化流程可以记成：

```text
多个工具
→ Tool Search
→ Load / Call
→ Single-result message
```

---

## 替代方案与适用边界

并不是所有系统都值得上“schema + examples + Tool Search + Programmatic Tool Calling”全套。正确做法是按复杂度分层。

第一层，prompt-only 或 schema-only。适合单一工具、字段很少、调用链极短的场景。比如天气查询：`location` 一个字段，加一个单位枚举，通常只靠 schema 和清晰描述就能跑通。这一层成本最低，但对复杂参数和多工具决策不稳。

第二层，schema + examples。适合参数存在格式约束、嵌套对象或可选字段组合的场景。典型如日历、工单、支付、报表筛选。它的优点是改动小，收益高，通常是最划算的增强手段。

第三层，Tool Search + examples。适合工具数量大、来源多、上下文预算敏感的系统。工具库一旦扩展到几十个以上，不做搜索式缩减，模型容易“看花眼”。

第四层，Programmatic Tool Calling。适合真实工程里的长链路、多服务、批处理和条件分支。Tines 这种安全自动化平台属于这一层，因为它处理的不是“一次函数调用”，而是整条操作链。

| 方案 | 适用条件 | 优点 | 限制 |
|---|---|---|---|
| prompt-only / schema-only | 单工具、参数简单 | 实现最快 | 容易在复杂参数上出错 |
| schema + examples | 格式敏感、嵌套参数 | 成本低，提升明显 | 仍不解决长链路状态管理 |
| Tool Search + examples | 大型工具库 | 节省上下文，提升选工具准确率 | 要维护工具索引 |
| Programmatic Tool Calling | 多步、跨服务、批量流程 | 中间状态不污染上下文，适合复杂编排 | 实现复杂，需要安全控制 |

一个简单对照是：

- 玩具例子：天气查询接口，只需 `location`，通常用 schema + description 就够。
- 真实工程例子：安全平台里的 IT 自动化，可能涉及用户目录、审批系统、消息系统、审计系统，且步骤可达几十到上百，这时更适合程序化编排。

所以边界很清楚。短链路问题，优先减少系统复杂度；长链路问题，优先减少上下文污染。不要把两类问题混成一种。

---

## 参考资料

| 标题 | 类型 | 关键贡献 |
|---|---|---|
| [How to implement tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use) | Claude 官方文档 | 定义 `tools`、`input_schema`、`input_examples`、并行调用与结果格式要求 |
| [Tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) | Claude 官方文档 | 说明按需加载工具、减少上下文占用、提升大工具库下的工具选择准确率 |
| [Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling) | Claude 官方文档 | 说明 Claude 写 Python 代码、在沙箱容器中串联工具调用的机制与限制 |
| [Claude can now use tools](https://claude.com/blog/tool-use-ga) | Anthropic 官方博客 | 给出 Claude 工具使用的产品定位与基础能力边界 |
| [Tines transforms workflow automation with Claude in Amazon Bedrock](https://claude.com/customers/tines) | Anthropic 客户案例 | 提供 120 步工作流压缩为单步 agent、10 到 100 倍效率提升的真实工程案例 |
| [A Comprehensive Survey of Benchmarks for Evaluating Tool and Function Calling in Large Language Models](https://huggingface.co/datasets/tuandunghcmut/BFCL_v4_information/blob/main/A%20Comprehensive%20Survey%20of%20Benchmarks%20for%20Evaluating%20Tool%20and%20Function%20Calling%20in%20Large%20Language%20Models.md) | 基准综述 | 汇总 BFCL 等工具调用评测，记录 Claude 3.5 Sonnet 在 BFCL 上的 90.20% 成绩 |
| [Anthropic's Claude Adds Three Beta Tools to Cut Context and Boost Accuracy](https://augmenter.dev/articles/anthropics-claude-adds-three-beta-tools-to-cut-context-and-boost-accuracy-1764110439156/) | 行业技术报道 | 整理 `input_examples`、Tool Search、Programmatic Tool Calling 的工程收益，并引用 72% 到 90% 的复杂参数准确率提升 |
