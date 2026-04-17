## 核心结论

Claude 的 `system prompt` 可以理解为“在用户提问之前，先塞给模型的一段最高优先级背景说明”。它不是回答时临时参考的便签，而是每次生成下一个 token 时都会一起进入条件上下文的前缀。

公开文档能直接确认三件事。第一，Anthropic 官方要求用 `system` 参数设定角色，把任务细节放在 `user` 消息里，这说明 `system` 不是普通备注，而是专门的控制位。第二，prompt caching 的缓存顺序明确是 `tools -> system -> messages`，说明 `system` 处在前缀结构的固定层。第三，Agent SDK 里 `systemPrompt`、`CLAUDE.md`、output styles 都是在回答前先拼进系统侧指令，再影响整轮行为。

因此，准确的工程结论不是“Claude 内部有一套神秘的 system 专用注意力层”，而是两层机制叠加：

| 机制 | 作用 |
| --- | --- |
| Transformer 前缀上下文 | `system` 作为前缀 token，被后续生成持续看到 |
| 指令跟随训练 | 模型被专门训练去更稳定地遵循系统级规则 |

这也是为什么当 `system prompt` 和用户消息冲突时，Claude 通常更倾向于服从 `system`。这里的“通常”很重要，它表示高概率行为与官方设计方向，不等于数学上的绝对保证。

---

## 问题定义与边界

先定义边界。`system prompt` 是开发者写给模型的全局约束，用来规定角色、风格、权限边界和冲突处理规则。`user prompt` 是当前用户的任务输入。`context window` 是“模型这次能看到的全部文本窗口”，白话讲，就是一次回答时能一起拿来参考的文本总预算。

如果把一次生成写成条件概率，形式是：

$$
p(x_t \mid S, M_{<t})
$$

其中 $x_t$ 是当前要生成的第 $t$ 个 token，$S$ 是 system prompt，$M_{<t}$ 是此前已经进入上下文的消息历史。它表达的意思很直接：Claude 不是先忘掉 `system` 再回答用户，而是在预测每个 token 时都把 `system` 当成条件的一部分。

这里有三个常见误解需要先排除。

| 误解 | 更准确的说法 |
| --- | --- |
| system prompt 只在开头生效一次 | 它作为前缀进入整次生成，不是“一次性开关” |
| system prompt 一定压过一切 | 它有更高优先级，但最终行为仍受模型训练和具体表述影响 |
| 只要写很长的 system prompt 就更稳定 | 过长、矛盾、无结构，反而会降低稳定性 |

玩具例子最容易说明边界。假设系统提示是：

> 你只回答 Python 基础语法，且必须使用简短中文。

这时用户问“推荐几款机械键盘”。更符合机制预期的行为，不是直接开始推荐键盘，而是说明这个请求超出当前系统设定范围。这说明 `system` 的职责是先定义“允许回答什么、如何回答”，再轮到用户具体提问。

---

## 核心机制与推导

核心机制可以拆成“结构位置”和“训练偏好”两部分。

第一部分是结构位置。Claude 底层仍是 Transformer。Transformer 的 self-attention 可以理解为“当前 token 回头看前面哪些 token 更相关”。因为 `system prompt` 在输入序列最前面，所以后续 token 理论上都能对它分配注意力权重。白话讲，后面每写一个字，都有机会重新参考前面那段系统说明。

如果把输入抽象成：

$$
[S \, | \, U_1 \, | \, A_1 \, | \, U_2 \, | \cdots ]
$$

那么生成第 $t$ 个输出 token 时，注意力并不会只看最近的用户句子，而是会在整个可见前缀上分配权重。`system` 之所以稳定，不是因为它在数学上“永远最大”，而是因为它总在前缀里、每轮都在、而且训练中被赋予更高指令地位。

第二部分是训练偏好。Anthropic 文档长期强调用 `system` 设角色、用结构化标签组织规则、在冲突时优先使用系统级说明。这说明模型不仅“能看到” system prompt，而且被训练成“更应该听它”。

可以把这个过程近似写成一个打分模型：

$$
\text{score}(token) = f(\text{semantic relevance}, \text{instruction priority}, \text{safety policy})
$$

这里 `instruction priority` 就是指令优先级，白话讲，就是“哪类命令更该被听从”。`system` 在这项上通常比分散在用户消息里的临时要求更高。

再看 prompt caching，就更容易理解“前缀地位”。官方文档明确说缓存前缀按 `tools -> system -> messages` 组织。缓存的不是一句自然语言摘要，而是这段静态前缀对应的已处理结果。工程上这意味着：如果 `system` 不变，Claude 不必每次都从头处理整段固定规则，延迟和成本都更可控。

一个最小数值例子：

| 组成 | token 数 |
| --- | --- |
| system prompt | 10 |
| 用户消息 | 30 |
| 历史助手消息 | 20 |

总上下文是 60 token。生成第 61 个 token 时，模型仍然是在这 60 个 token 上做条件预测，其中开头那 10 个 system token 并没有“失效”，只是它们的实际影响大小取决于表述质量、相关性和训练偏好。

---

## 代码实现

下面先用一个可运行的 Python 玩具程序模拟“系统规则优先于用户规则”的选择逻辑。它当然不是 Claude 的真实实现，但足够帮助初学者建立正确心智模型。

```python
from dataclasses import dataclass

@dataclass
class Rule:
    source: str   # system / user
    key: str
    value: str
    priority: int

def resolve_rules(rules):
    chosen = {}
    for rule in sorted(rules, key=lambda r: r.priority, reverse=True):
        if rule.key not in chosen:
            chosen[rule.key] = rule.value
    return chosen

rules = [
    Rule(source="system", key="language", value="zh-cn", priority=100),
    Rule(source="system", key="style", value="concise", priority=100),
    Rule(source="user", key="language", value="en", priority=10),
    Rule(source="user", key="format", value="bullet-list", priority=10),
]

resolved = resolve_rules(rules)

assert resolved["language"] == "zh-cn"
assert resolved["style"] == "concise"
assert resolved["format"] == "bullet-list"

print(resolved)
```

这个例子表达的不是“Claude 内部就是一个 if-else 排序器”，而是一个工程抽象：当同一维度的规则冲突时，系统侧规则应该先被解释和保留。

再看一个接近真实 API 的例子。真实工程里，你通常不会把全部要求都塞进 `system`，而是把稳定规则放系统侧，把当前任务放用户侧。

```python
import anthropic

client = anthropic.Anthropic()

resp = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=400,
    system="""
你是一名面向初级工程师的代码审查助手。
要求：
1. 优先指出确定性 bug，再谈风格问题
2. 输出中文
3. 如果信息不足，直接说“不足以判断”
""".strip(),
    messages=[
        {
            "role": "user",
            "content": "请审查下面这个 Python 函数是否有边界条件问题：..."
        }
    ],
)

print(resp.content)
```

真实工程例子是 Claude Code 或 Agent SDK。团队会把“代码风格、测试要求、目录约定、输出习惯”放进 `CLAUDE.md` 或 `systemPrompt` 预设中。这样不同开发者发起请求时，模型先读到的是同一套团队规则，而不是每个人都重新手写一遍“请按我们项目规范回答”。

---

## 工程权衡与常见坑

最常见的坑不是“没写 system prompt”，而是“把它写成一锅粥”。

| 风险 | 具体表现 | 更稳妥的做法 |
| --- | --- | --- |
| 规则过长 | 模型抓不住重点，回答忽左忽右 | 只保留稳定、全局、必须执行的规则 |
| 规则冲突 | 一会儿要简洁，一会儿要详细，一会儿只输出表格 | 显式写优先级，例如“若冲突，以准确性优先于格式” |
| 把任务细节塞进 system | 每次任务变更都要改系统前缀，缓存命中下降 | 把变化大的任务描述放进 `user` |
| 边界不清 | 模型不知道何时拒答、何时澄清 | 在 system 里写明超范围处理方式 |

一个典型错误写法是：

- 永远简短
- 必须覆盖全部细节
- 只输出 JSON
- 用自然语言解释原因

这四条里至少有两组天然冲突。系统提示不是愿望清单，而是可同时满足的约束集合。

另一个坑是把“遵循 system”理解成“绝对服从”。如果 `system` 本身模糊，比如“尽量专业，同时像朋友一样随意，同时必须极其正式”，模型只能在冲突里猜测主目标。很多看起来像“模型不听话”的问题，本质上是提示词设计不自洽。

---

## 替代方案与适用边界

`system prompt` 适合放稳定规则，但不适合承担全部控制需求。工程上至少有三类替代或补充方案。

| 方案 | 适合什么问题 | 不适合什么问题 |
| --- | --- | --- |
| 用户消息中的临时指令 | 本轮任务特有要求 | 需要跨多轮保持的一致规范 |
| `CLAUDE.md` / output styles | 团队共享、项目长期规则 | 单次临时实验 |
| 工具层约束 | 输出必须可执行、可验证 | 纯风格和语气控制 |

如果你只是临时想让 Claude “这一轮多给一个 SQL 版本”，写在 `user` 里更合理；如果你要它“所有代码审查都先报 bug 再报样式”，放 `system` 或项目级 `CLAUDE.md` 更合理；如果你要求“输出必须是合法 JSON 且字段齐全”，那往往应该再加解析器、schema 校验或工具调用，而不是只靠 prompt。

适用边界也要说清楚。`system prompt` 能强烈影响 Claude，但它不能替代模型能力本身。它能改变角色、格式、优先级和安全边界，不能凭空让一个不擅长的模型获得不存在的领域知识，也不能保证 100% 消除幻觉。

---

## 参考资料

- Anthropic, Giving Claude a role with a system prompt  
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/system-prompts
- Anthropic, Context windows  
  https://docs.anthropic.com/en/docs/build-with-claude/context-windows
- Anthropic, Prompt caching  
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Claude API Docs, Modifying system prompts  
  https://platform.claude.com/docs/en/agent-sdk/modifying-system-prompts
- Claude Docs, Modifying system prompts  
  https://docs.claude.com/en/docs/agent-sdk/modifying-system-prompts
- Anthropic, System Prompts release notes  
  https://docs.anthropic.com/en/release-notes/system-prompts
- Anthropic, Use XML tags  
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags
