## 核心结论

先给结论：如果你讨论的是 **AutoGen v0.4 的自定义 Agent**，主线接口是 `BaseChatAgent` 的 `on_messages`、`on_reset`、`produced_message_types`，而不是把 `register_reply` 当成新版本的中心扩展点。`register_reply`、`register_function`、`system_message` 这组接口主要属于 **v0.2 的 `ConversableAgent` 风格**；在 2024 年发布的 v0.4 重写版里，官方迁移指南明确建议，原来依赖 `register_reply` 的逻辑应迁移为自定义 `BaseChatAgent`。这一步必须先分清，否则会把两代 API 混在一起。

但工程上，很多资料和旧代码仍然围绕 `ConversableAgent` 展开，所以理解它仍然有价值。可以把它看成一套“能力暴露协议”：

| 能力层 | 作用 | 典型接口 | 核心问题 |
|---|---|---|---|
| 行为注册 | 决定收到什么消息时如何回复 | `register_reply(...)` | 谁触发、顺序如何、何时终止 |
| 边界声明 | 告诉模型“你能做什么，不能做什么” | `system_message` | 防止模型越权编造或乱用工具 |
| 工具暴露 | 把 Python 函数变成可调用工具 | `register_function(...)` | 谁发起、谁执行、返回什么 |

对初学者最重要的不是“能不能注册成功”，而是三件事同时成立：

1. **可宣称**：能力边界必须在 `system_message` 中写清楚。
2. **可触发**：回复链必须通过触发器和顺序显式定义。
3. **可恢复**：状态在 `reset` 后应回到初始配置，不能把上轮对话残留带进下一轮。

一个最直观的“限制型 Agent”例子是：你只想让 Agent 处理提醒请求，不允许它调工具，也不允许它临时扩权。那么 `system_message` 就要写成类似：

`仅响应提醒类询问，不调用任何工具，不回答与提醒无关的问题。`

这类文本不是装饰，而是能力边界声明。它的作用是把模型的自由度压缩到你允许的任务范围内。

---

## 问题定义与边界

“自定义 Agent 的注册与能力声明”本质上是在回答三个边界问题。

第一，**它处理什么消息**。这里的“消息”可以理解成“外部送进来的输入对象”。在 v0.4 自定义 Agent 里，这体现在 `on_messages()`；在 v0.2 `ConversableAgent` 里，则常通过 `register_reply(trigger, ...)` 表达“谁发来的消息会进入哪条回复逻辑”。

第二，**它重置后恢复到什么状态**。这里的“重置”是指把会话内状态清空，回到初始化配置。v0.4 用 `on_reset()` 明确实现；v0.2 中很多回复链和配置也需要在 `reset` 后恢复，否则下一轮对话可能沿用旧规则。

第三，**它能产生什么输出类型**。这里的“输出类型”就是最终响应的形态，例如纯文本、工具调用总结、停止信号。v0.4 用 `produced_message_types` 明确列出允许的消息类型，这其实是在做一层输出约束。

下面用一个“只处理提醒”的玩具例子说明边界怎么定。

| 能力 | 触发条件 | 可调用接口 | 重置逻辑 |
|---|---|---|---|
| 提醒回复 | 来自普通用户的提醒请求 | 文本回复函数 | 清空历史提醒状态 |
| 默认兜底 | sender 为 `None` 或无法识别 | 默认提醒文本 | 恢复默认兜底逻辑 |
| 工具调用 | 禁止 | 无 | 始终保持禁用 |

如果按 v0.2 风格写，它的意图大致是：

- `system_message="仅响应提醒类询问，不调用任何工具"`
- `register_reply(trigger="RemindSender", reply_func=handle_remind, position=0)`
- `register_reply(trigger=None, reply_func=send_default_reminder, position=10)`

这套约束的关键不是“让它更聪明”，而是“让它更窄”。对零基础读者来说，可以把这理解成：你先圈定工作范围，再决定触发方式，最后才谈模型生成。

从形式上看，能力边界也可以写成一个简洁模板：

| 能力 | 接口 | 限制 |
|---|---|---|
| 回答提醒请求 | `register_reply` 或 `on_messages` | 仅处理提醒语义 |
| 生成最终文本 | `produced_message_types` | 只输出文本 |
| 调用外部工具 | `register_function` | 当前 Agent 禁止调用 |

这个表比长段文字更适合做工程文档，因为团队在加新功能时能快速检查“是否越界”。

---

## 核心机制与推导

先看旧版 `ConversableAgent` 的核心机制。`register_reply(trigger, reply_func, position=0, config=None, reset_config=None, ...)` 可以理解成一条**回复链注册表**。回复链就是“匹配到消息后，按顺序试哪些处理函数”。官方文档说明两个关键点：

1. **后注册优先检查**。
2. `position` 可调整顺序，通常越靠前越早执行。

也就是说，它不是一个简单的“覆盖”，而是一条有序规则链。可以用文字流程表示成：

`收到消息 -> 依次检查 reply_func 是否命中 -> 某个 reply 返回 final=True -> 终止 -> 否则继续下一条`

在很多 v0.2 实现中，还会与默认流程拼接，常见顺序可抽象为：

`终止判定 -> 工具/函数执行 -> 代码执行 -> LLM 文本回复`

为什么“`final=True` 才中断”重要？因为这意味着你可以把回复链拆成多层：

- 第一层做强约束，比如安全拦截、任务分类。
- 第二层做工具路由。
- 第三层再把剩余请求交给 LLM。

这其实像一个分层过滤器。数学上可以把它看成一个有序判定函数集合 $\{f_1, f_2, \dots, f_n\}$。输入消息 $m$ 后，系统求第一个满足：

$$
f_i(m) = (final=True, reply=r)
$$

的函数；若不存在，再落到默认生成逻辑。这个顺序决定了控制权属于谁。

下面给一个玩具例子。假设我们先注册一个默认兜底，再注册一个提醒专用处理器：

- `register_reply(None, fallback_reply, position=10)`
- `register_reply(UserQuerySender, reminder_reply, position=0)`

因为提醒处理器位置更前，所以普通用户消息会优先命中 `reminder_reply`；只有没匹配上时，才落到 `fallback_reply`。这就是“有序触发”。

触发器 `trigger` 也不是只支持一个字符串。旧版接口允许的类型包括：

| trigger 类型 | 白话解释 | 适用场景 |
|---|---|---|
| 类 | 发送者属于某类对象时触发 | 区分 `UserProxyAgent`、`AssistantAgent` |
| 字符串 | 发送者名字匹配时触发 | 小型脚本快速路由 |
| 实例 | 只对某个具体发送者生效 | 固定双 Agent 对话 |
| callable | 你自己写布尔判定函数 | 复杂过滤逻辑 |
| `None` | 没有 sender 时触发 | 系统触发、默认兜底 |

这里还要补一个经常被忽略的边界：**异步回复函数**。官方文档明确说明，`async` 版 reply 只会在异步 chat 中被调用；如果在同步 chat 中注册了异步 reply，默认会报错，除非你设置 `ignore_async_in_sync_chat=True` 让它被忽略。这不是语法细节，而是运行时契约。

再看 v0.4。迁移指南的核心意思是：不要再把“复杂行为”堆到 `register_reply` 上，而应直接写自定义 `BaseChatAgent`。这意味着控制流从“注册函数链”转向“显式实现生命周期方法”。也就是：

- `on_messages()` 决定收到新消息后怎么处理。
- `on_reset()` 决定状态如何归零。
- `produced_message_types` 决定输出边界。

所以如果你在写新项目，正确理解应该是：

- `system_message` 仍然是能力声明的文本边界。
- `register_function` 仍然是工具暴露的方式之一。
- 但“自定义 Agent 的主体行为”在 v0.4 中更推荐写进 `BaseChatAgent`，而不是继续依赖 v0.2 的 `register_reply` 编排。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，不依赖 AutoGen 包，也能把“注册回复链”和“能力边界”机制跑通。目的是先把机制讲清楚，再看真实工程例子。

```python
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple

@dataclass
class Message:
    sender: Optional[str]
    content: str

@dataclass
class ReplyResult:
    final: bool
    content: str

class MiniAgent:
    def __init__(self, system_message: str):
        self.system_message = system_message
        self._replies: List[Tuple[int, Optional[str], Callable[[Message], ReplyResult]]] = []

    def register_reply(self, trigger: Optional[str], reply_func, position: int = 0):
        self._replies.append((position, trigger, reply_func))
        self._replies.sort(key=lambda x: x[0])

    def reset(self):
        # 玩具实现只重置运行时状态，不清空注册表
        pass

    def receive(self, message: Message) -> str:
        for _, trigger, func in self._replies:
            if trigger is None or trigger == message.sender:
                result = func(message)
                if result.final:
                    return result.content
        return "未命中任何规则"

def handle_remind(message: Message) -> ReplyResult:
    if "提醒" in message.content:
        return ReplyResult(True, "已记录提醒请求，只返回提醒相关文本。")
    return ReplyResult(False, "")

def default_reply(message: Message) -> ReplyResult:
    return ReplyResult(True, "该 Agent 仅处理提醒类询问，不调用任何工具。")

agent = MiniAgent(system_message="仅响应提醒类询问，不调用任何工具")
agent.register_reply(trigger="RemindSender", reply_func=handle_remind, position=0)
agent.register_reply(trigger=None, reply_func=default_reply, position=10)

r1 = agent.receive(Message(sender="RemindSender", content="请提醒我明天交周报"))
r2 = agent.receive(Message(sender="OtherSender", content="帮我查天气"))

assert r1 == "已记录提醒请求，只返回提醒相关文本。"
assert r2 == "该 Agent 仅处理提醒类询问，不调用任何工具。"
print(r1)
print(r2)
```

这个例子对应的思想很直接：

- `system_message` 声明边界。
- `register_reply` 决定触发规则。
- 默认规则放后面做兜底。
- 一旦命中 `final=True`，后面的规则不再执行。

再看“真实工程例子”：把 Python 函数暴露为 Agent 可调用工具。这里以最小计算器为例，因为它最容易验证调用链。

```python
def calculator(operator: str, a: int, b: int) -> int:
    table = {
        "+": a + b,
        "-": a - b,
    }
    if operator not in table:
        raise ValueError("unsupported operator")
    return table[operator]

assert calculator("+", 3, 4) == 7
assert calculator("-", 10, 6) == 4
```

在 AutoGen 旧版 `ConversableAgent` 风格中，注册工具的关键三元组是：

| 参数 | 白话解释 | 作用 |
|---|---|---|
| `caller` | 谁可以发起工具调用 | 通常是 `assistant_agent` |
| `executor` | 谁真正执行 Python 函数 | 通常是 `user_proxy_agent` |
| `description` | 给模型看的工具说明 | 影响模型是否正确选择该工具 |

对应代码通常写成：

```python
assistant_agent.register_function(
    calculator,
    caller=assistant_agent,
    executor=user_proxy_agent,
    description="支持 +/- 的整型计算"
)
```

这段注册背后的流程是：

1. `assistant_agent` 看到用户请求，比如“计算 3 + 4”。
2. 模型根据 `description` 判断“应调用 calculator”。
3. `user_proxy_agent` 作为 `executor` 真正执行 `calculator("+", 3, 4)`。
4. 执行结果 `7` 回到对话链，再由 assistant 组织成自然语言回答。

这里 `caller` 和 `executor` 分离很重要。它把“决策权”和“执行权”拆开了。前者负责判断什么时候该用工具，后者负责在 Python 侧真正落地。这是多 Agent 工具编排的基本形态。

如果你按 v0.4 思路实现，则更推荐把“收到消息后的路由逻辑”放进 `BaseChatAgent`，而把工具作为模型可用资源注入。下面给一个概念化骨架，重点看三个生命周期钩子：

```python
from typing import Sequence

class ReminderAgent:  # 这里简化展示，真实代码应继承 BaseChatAgent
    def __init__(self):
        self.system_message = "仅响应提醒类询问，不调用任何工具"
        self.history = []

    async def on_messages(self, messages):
        last = messages[-1]
        self.history.append(last)
        if "提醒" in last.content:
            return "已处理提醒请求"
        return "仅支持提醒类问题"

    async def on_reset(self):
        self.history = []

    @property
    def produced_message_types(self):
        return ("text",)
```

这里的 `produced_message_types` 可以理解成“输出白名单”。如果你只想产出文本，就不要让它返回结构化工具消息。这种约束比“靠提示词说不要输出 JSON”要可靠。

---

## 工程权衡与常见坑

实际做项目时，最常见的问题不是“不会注册”，而是“注册了但行为失控”。下面按影响排序。

| 问题 | 影响 | 解决办法 |
|---|---|---|
| 混用 v0.2 与 v0.4 心智模型 | 代码能跑但架构混乱 | 新项目以 `BaseChatAgent` 为主，旧项目再理解 `register_reply` |
| 未设置 `None` 触发器 | 系统触发或无 sender 消息漏处理 | 增加 `register_reply(None, fallback_reply)` |
| 异步 reply 放进同步 chat | 运行时直接报错 | 改用 async chat，或显式设 `ignore_async_in_sync_chat=True` |
| `system_message` 过长或重复 | 模型忽略约束、角色混淆 | 只保留必要边界，写短而明确 |
| `description` 写得含糊 | 模型不会选工具或错误选工具 | 用输入输出约束描述工具能力 |
| `reset` 不彻底 | 上轮状态污染下一轮 | 在 `on_reset` 中恢复初始状态与配置 |

这里重点展开三个坑。

第一，**把 `system_message` 当万能权限系统**。它不是强制沙箱，而是模型侧约束。也就是说，它能显著影响模型行为，但不能替代代码层面的权限控制。真正的限制应该是“双层”的：

- 文本层：`system_message` 写清边界。
- 代码层：不要注册不该给它的工具，不要暴露不该执行的 executor。

第二，**忘了默认兜底**。很多人只注册“命中的正常路径”，却没处理“未命中时怎么办”。结果是模型直接落到自由生成，输出超出授权范围。一个稳妥模式是同时保留：

- `register_reply(UserProxySender, tool_reply, position=0)`
- `register_reply(None, default_reply, position=10)`

前者负责正常流程，后者负责兜底收口。

第三，**重置逻辑只清历史，不恢复规则**。如果你的回复顺序、配置对象或运行态开关在会话中发生过变化，那么 `reset` 只清空消息列表是不够的。应把“初始配置快照”恢复回来。否则会出现上一轮临时放开的能力，在下一轮仍然保留的问题。

真实工程里，一个典型场景是“天气查询助手”：

- `assistant` 决定什么时候调用 `get_weather`
- `user_proxy` 执行 Python 函数或外部 API
- 返回结果后 assistant 重新组织自然语言回复

这个模式适合因为天气查询本身是明确、结构化、易验证的工具调用。如果没有明确工具边界，而是把所有问题都丢给 LLM 自行发挥，结果通常更不稳定。

---

## 替代方案与适用边界

工程上至少有三种可选结构，不同场景选型不同。

| 方案 | 适用场景 | 限制 |
|---|---|---|
| `BaseChatAgent` 自定义主逻辑 | 新项目、v0.4、需要明确生命周期 | 需要自己写状态管理 |
| `ConversableAgent + register_reply` | 维护旧项目、快速拼装回复链 | 容易把规则堆乱，迁移成本迟早出现 |
| `AssistantAgent + 工具注册` | 工具调用明确、文本生成为主 | 不适合复杂状态机或重控制流 |

如果需求只是“限制回答范围，不需要工具调用”，最简单的做法反而不是多 Agent，而是：

- 用一个 Agent
- 写清楚 `system_message`
- 在 `on_messages` 或简单回复链里实现窄任务逻辑

这适合“提醒 Agent”“FAQ Agent”“固定格式改写 Agent”这类任务。

如果需求是“需要真实工具调用”，例如天气查询、数据库只读查询、计算器，那么推荐结构是：

1. `AssistantAgent` 负责理解用户请求。
2. 通过 `register_function(get_weather, caller=tool_caller, executor=tool_executor, ...)` 暴露工具。
3. 由 `UserProxyAgent.initiate_chat(...)` 或等价入口启动会话。
4. executor 执行函数并把结果送回。

它的优点是职责清楚：模型负责决策，代理负责执行，结果可追踪。

如果需求再进一步，变成“多 Agent 协同且每个 Agent 有独立权限边界”，那就不该继续依赖零散的 `register_reply` 规则，而应转向 v0.4 的自定义 Agent 组合、团队编排和显式状态管理。因为这时问题已从“回复顺序”升级成“系统设计”。

所以可以用一句话概括适用边界：

- 旧代码理解 `register_reply`
- 新代码优先 `BaseChatAgent`
- 需要工具时再引入 `register_function`
- `system_message` 始终只做边界声明，不做唯一安全保证

---

## 参考资料

1. AutoGen v0.4 Custom Agents: https://microsoft.github.io/autogen/0.4.5/user-guide/agentchat-user-guide/tutorial/custom-agents.html
2. AutoGen v0.4 Migration Guide: https://microsoft.github.io/autogen/0.4.2/user-guide/agentchat-user-guide/migration-guide.html
3. AutoGen `BaseChatAgent` API: https://microsoft.github.io/autogen/0.4.9/reference/python/autogen_agentchat.agents.html
4. AutoGen `ConversableAgent` / `register_reply` 参考: https://autogenhub.github.io/autogen/docs/reference/agentchat/conversable_agent/
5. AutoGen Tool Use 教程: https://autogenhub.github.io/autogen/docs/tutorial/tool-use/
