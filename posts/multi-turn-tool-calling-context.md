## 核心结论

多轮工具调用里，真正被模型“看到”的不是“用户问题 + 最终答案”，而是整段输入序列。只要你的系统把工具结果重新喂回下一次模型调用，这些结果就成了正式上下文的一部分。在传统 chat-style 接口里，它常表现为 `role=tool` 消息；在更细粒度的接口里，它也可能表现为 `function_call_output` 之类的输入项。形式不同，机制相同：工具返回值不是旁路数据，而是下一轮推理的直接输入。

这件事有两个直接后果。第一，真正吃掉上下文预算的，往往不是用户问题，而是工具返回值。用户一句话可能只有几十个 token，一次搜索结果、数据库查询、网页正文、日志片段却可能是几千个 token。第二，长上下文不是“装得下就等于用得好”。历史越长，模型越容易把真正重要的约束和大量低价值观测混在一起处理，导致后续推理、工具选择和答案质量一起下降。

所以工程重点不是“尽量保留全部历史”，而是“保留后续推理真正需要的状态”。常见且有效的三类策略是：摘要压缩、选择性保留、滑动窗口。经验上，不要等到窗口快满才处理，应该在利用率到 70% 到 80% 左右时就开始做软触发压缩；到了 85% 到 90%，再进入硬限制或拒绝继续扩写。

---

## 问题定义与边界

这个问题的准确表述是：在多轮 agent 系统中，工具调用结果如何回传到模型，以及这些结果如何持续占用上下文窗口，最终影响后续推理、工具选择和回答质量。

先把“工具回传”说清楚。一个最小闭环通常是下面这样：

| 步骤 | 发生了什么 | 会不会进入下一轮上下文 |
| --- | --- | --- |
| 1 | 用户提出任务 | 会 |
| 2 | 模型决定调用工具 | 会 |
| 3 | 应用执行工具，拿到结果 | 原始结果先在应用侧产生 |
| 4 | 应用把工具结果作为消息或输入项回传给模型 | 会 |
| 5 | 模型基于“历史消息 + 工具结果”继续推理 | 会 |

OpenAI 的函数调用与 Responses 文档都把这个流程写得很清楚：模型先产出工具调用，应用执行，再把工具输出作为新的输入发回模型。换句话说，工具结果回传本质上是“把结果重新追加进下一次调用的输入”，而不是“模型自己偷偷记住了”。

这里要区分三个边界：

| 概念 | 白话解释 | 关注点 |
| --- | --- | --- |
| 上下文窗口 | 模型单次调用最多能读入多少 token | 会不会装不下 |
| 上下文质量 | 当前窗口里高价值信息的占比够不够高 | 即使装得下，是否还能推理对 |
| 工具回传 | 工具结果被重新作为输入送回模型 | 结果会不会把预算挤爆 |

一个常见误区是：只要模型支持 128K、200K，甚至 1M 上下文，就不需要做上下文管理。这个判断不成立。长窗口只能推迟溢出，不能自动去噪，也不能自动把“重要信息”排到前面。窗口越大，系统越容易产生另一种错觉：反正还能塞，就继续塞。最后不是“装不下”，而是“信息密度越来越差”。

可以把上下文压力写成一个很简单的式子：

$$
u = \frac{T_{\text{input}}}{T_{\text{max}}}
$$

其中，$T_{\text{input}}$ 是这次请求真正送进模型的总 token，$T_{\text{max}}$ 是模型支持的最大上下文窗口。工程上更关心的不是它是否等于 1，而是它什么时候开始逼近风险区。

如果把多轮流程的单步保真度粗略记为 $p$，经过 $n$ 步后的整体成功率可以近似写成：

$$
success\_rate \approx p^n
$$

当 $p = 0.95$ 时，20 步后的估算值是：

$$
0.95^{20} \approx 0.358
$$

这不是自然定律，只是工程上的直觉模型。它想说明一件事：多轮系统的失败，很多时候不是某一步突然完全错了，而是每一步都丢一点，最后累计成明显偏差。

看一个最小例子更直观：

| 项目 | 近似 token |
| --- | ---: |
| 用户提问：“查一下上海天气” | 20 ~ 50 |
| 模型发起工具调用 | 50 ~ 150 |
| 搜索或天气接口返回原始结果 | 2,000 ~ 4,000 |
| 模型生成自然语言答案 | 100 ~ 300 |

只看用户和答案，会觉得任务很短；一旦把工具结果算进去，下一轮调用前的上下文可能已经是几千 token。再来两三轮类似操作，真正占空间的就不是问题本身，而是旧工具结果。

---

## 核心机制与推导

多轮工具调用的上下文管理，可以拆成三层来看：消息层、注意力层、位置层。

第一层是消息层。模型发起工具调用以后，应用要把工具结果再送回来，模型才有条件继续推理。这意味着“工具使用次数越多”，通常也就意味着“被送入上下文的外部文本越多”。这些文本可能是：

| 工具类型 | 常见返回内容 | 容易膨胀的原因 |
| --- | --- | --- |
| Web 搜索 | 搜索结果、网页正文、来源列表 | 重复段落多，正文很长 |
| 数据库查询 | JSON、表格行、日志 | 字段名重复，结构噪声大 |
| 文件检索 | 多个文件片段 | 文件头、注释、重复上下文多 |
| 终端执行 | 命令输出、报错栈 | 错误日志常常远大于真正有用的信息 |

第二层是注意力层。对 Transformer 来说，长上下文不是“白送的记忆条”。模型要在所有 token 之间分配注意力，而它并不会天然知道哪些旧工具结果已经没有用了。结果就是，高价值约束和低价值噪声会一起竞争。

可以把“上下文有效性”粗略理解成下面这个比值：

$$
q = \frac{T_{\text{useful}}}{T_{\text{input}}}
$$

这里的 $T_{\text{useful}}$ 不是客观真值，而是“这一步推理真正还需要的 token 数”。当你把整页网页、整个 JSON、整段日志都保留时，$T_{\text{input}}$ 会快速变大，但 $T_{\text{useful}}$ 未必同步增长，所以 $q$ 往往在下降。

第三层是位置层。大量实践都观察到：模型通常对开头和结尾更敏感，对中间长段文本更容易忽略。这意味着“还在窗口里”不等于“仍然好用”。早期系统约束、关键决策、边界条件，即使没有被截断，也可能因为被夹在长工具结果中间而失去稳定性。

所以压缩的目标不是“让文本变短”，而是“把后续仍然需要的状态提炼出来”。这个状态通常包括：

| 必须尽量保留的信息 | 为什么重要 |
| --- | --- |
| 当前目标 | 决定系统下一步做什么 |
| 已完成事项 | 防止重复查询、重复调用工具 |
| 关键决策及原因 | 防止后续推理和前面结论冲突 |
| 错误码、文件路径、变量名、ID | 这些是后续可定位、可复现的锚点 |
| 最近几轮消息 | 最近状态最容易影响下一步动作 |

对应地，下面这些内容通常应该优先压缩：

| 优先压缩对象 | 压缩时保留什么 |
| --- | --- |
| 长网页正文 | 标题、来源、发布时间、核心结论 |
| 大块 JSON | 关键字段、状态码、主键、异常字段 |
| 长日志 | 首个报错点、堆栈顶部、退出码 |
| 已完成且不会再次细读的旧对话 | 决策、结论、待办 |

推荐压缩比可以这样理解：

| 内容类型 | 建议压缩比 | 说明 |
| --- | --- | --- |
| 旧对话历史 | 3:1 ~ 5:1 | 保留目标、决策、未完成事项 |
| 工具输出 / 观测 | 10:1 ~ 20:1 | 只保留结论、关键字段、错误码 |
| 最近消息（5~7 条） | 不压缩或轻压缩 | 这是当前工作区 |
| 系统提示 / 开发者约束 | 不压缩 | 这是行为锚点 |

新手最容易犯的错误，是把“所有原始工具结果都保留”误当成“信息更完整”。实际上，很多原始结果的作用只是帮助模型在那一轮完成一次判断。一旦判断已经形成，后续需要的是“判断结果和锚点”，而不是整份原始材料继续常驻上下文。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它只依赖标准库，演示四件事：

1. 估算消息 token 数
2. 在达到软阈值时触发压缩
3. 优先保留系统消息和最近消息
4. 在仍然过长时，对超长 `tool` 结果做硬压缩

这个实现不绑定任何具体模型，重点是把“上下文管理器”这个中间层讲清楚。

```python
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Message:
    role: str
    content: str
    meta: Dict[str, str] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """
    玩具估算器。
    真实系统应替换成 provider 的 token counting API 或官方 tokenizer。
    这里用“约 2 个字符 ~= 1 token”做粗略近似。
    """
    if not text:
        return 0
    return max(1, (len(text) + 1) // 2)


def message_tokens(msg: Message) -> int:
    """
    给每条消息增加少量固定开销，模拟 role / 包装结构的成本。
    """
    overhead = 6
    meta_cost = sum(estimate_tokens(f"{k}={v}") for k, v in msg.meta.items())
    return overhead + estimate_tokens(msg.content) + meta_cost


def summarize_tool_output(text: str, max_lines: int = 3, max_chars: int = 180) -> str:
    """
    从工具输出里抽取少量可继续推理的内容。
    真实系统里，这里可以换成更可靠的摘要器或规则提取器。
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "[tool-summary] empty output"

    selected = lines[:max_lines]
    summary = " | ".join(selected)

    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."

    return f"[tool-summary] {summary}"


def clip_text(text: str, keep: int = 80) -> str:
    """
    对普通 user / assistant 文本做轻裁剪，避免旧消息过长。
    """
    if len(text) <= keep:
        return text
    return text[: keep - 3] + "..."


def compress_message(msg: Message) -> Message:
    """
    不同角色使用不同压缩策略。
    关键思想：
    - system 尽量原样保留
    - tool 转成短摘要，避免原始结果继续常驻
    - 普通对话只保留短版本
    """
    if msg.role == "system":
        return msg

    if msg.role == "tool":
        return Message(
            role="system",
            content=summarize_tool_output(msg.content),
            meta={"source": "tool-pruned"},
        )

    if msg.role == "assistant":
        return Message(
            role="assistant",
            content=clip_text(msg.content, keep=60),
            meta={"source": "assistant-brief"},
        )

    return Message(
        role="user",
        content=clip_text(msg.content, keep=60),
        meta={"source": "user-brief"},
    )


def total_tokens(messages: List[Message]) -> int:
    return sum(message_tokens(m) for m in messages)


def compact_messages(
    messages: List[Message],
    max_context: int,
    soft_ratio: float = 0.75,
    hard_ratio: float = 0.90,
    keep_recent: int = 6,
) -> List[Message]:
    """
    两级策略：
    1. 低于软阈值：不动
    2. 超过软阈值：先压缩旧消息，保留最近消息
    3. 如果仍高于硬阈值：从前往后继续压缩，直到回到安全区
    """
    total = total_tokens(messages)
    if total <= max_context * soft_ratio:
        return messages

    recent = messages[-keep_recent:]
    old = messages[:-keep_recent]

    compacted = [compress_message(m) for m in old]
    compacted.extend(recent)

    i = 0
    while total_tokens(compacted) > max_context * hard_ratio and i < len(compacted):
        current = compacted[i]

        # 第一条 system 往往是行为锚点，尽量不要碰
        if i == 0 and current.role == "system":
            i += 1
            continue

        compacted[i] = compress_message(current)
        i += 1

    return compacted


def pretty_print(messages: List[Message]) -> None:
    for i, msg in enumerate(messages, start=1):
        preview = msg.content.replace("\n", " ")
        print(f"{i:02d}. {msg.role:<9} tokens={message_tokens(msg):<4} {preview[:90]}")


if __name__ == "__main__":
    messages = [
        Message(
            "system",
            "你是一个天气助手。回答必须以最新 tool 结果为准，并明确标注城市。",
        ),
        Message("user", "先查上海天气，再和北京、广州做对比。"),
        Message("assistant", "我先调用 weather_search 查询上海。"),
        Message(
            "tool",
            "city=Shanghai\ncondition=Sunny\ntemp_c=18\nhumidity=41%\nwind=3m/s\n" * 80,
        ),
        Message("assistant", "上海当前晴，18 摄氏度。接着查询北京和广州。"),
        Message(
            "tool",
            "city=Beijing\ncondition=Cloudy\ntemp_c=12\nhumidity=35%\nwind=5m/s\n" * 80,
        ),
        Message(
            "tool",
            "city=Guangzhou\ncondition=Rain\ntemp_c=24\nhumidity=88%\nwind=2m/s\n" * 80,
        ),
        Message("assistant", "已拿到三个城市的原始天气结果，准备汇总。"),
    ]

    before = total_tokens(messages)
    compacted = compact_messages(
        messages,
        max_context=1200,
        soft_ratio=0.60,
        hard_ratio=0.75,
        keep_recent=4,
    )
    after = total_tokens(compacted)

    assert after < before
    assert any(m.content.startswith("[tool-summary]") for m in compacted)
    assert compacted[0].role == "system"

    print(f"before={before}")
    print(f"after={after}")
    pretty_print(compacted)
```

这段代码能直接运行。它不是生产级方案，但足够把机制讲明白。你可以把它理解成一个位于“模型调用前”的上下文清理器。

运行后的结果会类似下面这样：

```text
before=7670
after=255
01. system    tokens=24   你是一个天气助手。回答必须以最新 tool 结果为准，并明确标注城市。
02. user      tokens=24   先查上海天气，再和北京、广州做对比。
03. assistant tokens=30   我先调用 weather_search 查询上海。
04. system    tokens=44   [tool-summary] city=Shanghai | condition=Sunny | temp_c=18
05. assistant tokens=29   上海当前晴，18 摄氏度。接着查询北京和广州。
06. system    tokens=44   [tool-summary] city=Beijing | condition=Cloudy | temp_c=12
07. system    tokens=44   [tool-summary] city=Guangzhou | condition=Rain | temp_c=24
08. assistant tokens=16   已拿到三个城市的原始天气结果，准备汇总。
```

这个例子里最值得注意的不是“压缩后更短”，而是“压缩后还保留了下一步真正需要的状态”：

| 被保留的信息 | 用途 |
| --- | --- |
| 系统消息 | 约束回答行为 |
| 用户目标 | 保证任务没有跑偏 |
| 上海 / 北京 / 广州的摘要结果 | 支撑后续比较 |
| 最近 assistant 状态 | 保持当前执行节奏 |

生产环境里，通常还会再做三件事。

第一，给消息打元数据。比如记录 `msg_id`、`token_count`、`source`、`tool_name`、`created_at`。这样做的目的不是给人看，而是让系统知道“哪些消息最贵、最旧、最适合裁剪”。

第二，保留“裁剪痕迹”，而不是直接硬删。比如把一大段网页正文删掉后，补一条：

```text
[pruned:search-17] 已提取来源、发布时间、核心结论；原始正文已移出主会话。
```

这样模型会知道这段历史存在过，而且知道自己当前拿到的是提炼后的结果，不会把摘要误当成完整原文。

第三，把摘要写成固定结构，而不是一段散文。比如：

```json
{
  "goal": "比较三个城市天气",
  "done": [
    "已查询上海",
    "已查询北京",
    "已查询广州"
  ],
  "facts": {
    "Shanghai": {"condition": "Sunny", "temp_c": 18},
    "Beijing": {"condition": "Cloudy", "temp_c": 12},
    "Guangzhou": {"condition": "Rain", "temp_c": 24}
  },
  "next_step": "生成对比回答"
}
```

这种结构化摘要对新手尤其重要。原因很简单：模型在下一轮最需要的不是“漂亮总结”，而是“稳定、可引用、可继续计算的状态”。

---

## 工程权衡与常见坑

三种常见策略各有代价：

| 策略 | 优点 | 代价 |
| --- | --- | --- |
| 滑动窗口 | 实现简单，延迟最低 | 容易丢掉中间的重要约束 |
| 摘要压缩 | 适合长任务，能保留长期状态 | 需要额外调用摘要器，且可能失真 |
| 选择性保留 | 对超长工具结果最有效 | 需要额外的“重要度判断”逻辑 |

真正难的地方不是“知道有这三种策略”，而是知道它们分别会在哪里失效。

第一个常见坑是静默截断。很多系统在上下文过长时不会明确报错，而是悄悄丢掉一部分输入。表面上看，模型还能继续回答；实际上，系统提示、旧决策或关键工具结果可能已经不在了。最麻烦的地方在于，这种失败经常伪装成“模型推理差”“工具路由错”“检索质量差”，排查成本很高。

第二个坑是把原始工具 JSON 全量保留。新手经常在 prompt 上下很多功夫，却忽略真正占空间的不是 prompt，而是工具返回值。下面这个对比很典型：

| 输入项 | 看起来是否短 | 实际 token 风险 |
| --- | --- | --- |
| 用户问题 | 是 | 低 |
| 系统提示 | 中等 | 中 |
| 10 行函数 schema | 中等 | 中 |
| 200 行 JSON 查询结果 | 否 | 高 |
| 2 页网页正文 | 否 | 极高 |

第三个坑是摘要过度。摘要不是越短越好。若你把错误码、文件路径、主键、变量名、时间戳都压掉，模型下轮虽然“记得做过这件事”，却失去了继续执行的锚点。很多“摘要后模型开始反复问已经问过的问题”，本质上就是锚点被删光了。

第四个坑是只保留“最近几条消息”，却忘了最近几条里可能正好包含超长工具结果。也就是说，滑动窗口并不自动等于上下文健康。若最近 4 条里有 2 条是 5,000 token 的搜索结果，那你保留“最近消息”本身就在制造风险。

第五个坑是忽略工具定义和系统固定前缀。真正的输入不仅包括聊天历史，还包括：

| 固定输入部分 | 常被忽略的原因 |
| --- | --- |
| 工具 schema / JSON schema | 平时看起来“不长” |
| 系统提示 / 开发者提示 | 默认总在那，不容易警觉 |
| 输出保留预算 | 很多人只算输入，不留输出空间 |
| provider 包装开销 | 本地字符串长度不等于真实 token |

所以更稳妥的预算方式是：

$$
T_{\text{available-history}}
= T_{\text{max}}
- T_{\text{system}}
- T_{\text{tools}}
- T_{\text{reserved-output}}
- T_{\text{safety-margin}}
$$

不是“窗口有 128K，所以历史能放 128K”，而是“除去固定前缀和输出预留后，历史真正能用的只剩多少”。

一个实用阈值逻辑可以写成这样：

```python
usage = context_tokens / max_context

if usage >= 0.90:
    print("硬阻断：停止继续扩写，必须压缩或拒绝")
elif usage >= 0.80:
    print("强提醒：压缩旧历史，过滤长 tool 输出")
elif usage >= 0.70:
    print("软触发：开始生成结构化摘要")
```

这个阈值不是标准答案，但它比“等满了再说”稳得多。

---

## 替代方案与适用边界

如果任务很短，比如 3 到 5 轮问答，滑动窗口通常够用。因为长期依赖弱，中间状态也不复杂，维护一套摘要系统的收益不一定高于复杂度。

如果任务很长，比如代码修复、报表分析、客服工单追踪、多步检索再汇总，摘要压缩更合适。这里更推荐“锚定式迭代摘要”，也就是持续维护固定结构的会话状态，例如“目标、已完成、关键事实、关键决策、待办、待验证”。这种写法更接近状态机，不容易在多轮后漂成一段谁也不敢依赖的长散文。

如果工具输出特别长，选择性保留通常比“先全量保留，再统一压缩”更有效。原因很直接：很多工具结果从一开始就没必要进入主会话。比如网页正文只提取标题、来源、发布日期和结论；数据库查询只回传关键字段；日志只回传首个错误点和退出码。你不是在“丢信息”，而是在阻止垃圾信息进入高价值上下文。

如果你的系统支持更细的架构拆分，还可以把上下文压力转移到系统设计层。典型做法有两种：

| 方案 | 思路 | 适用场景 |
| --- | --- | --- |
| 子代理 / 子会话隔离 | 让检索、分析、执行在独立上下文里完成，只把结果返回主会话 | 多工具、长流程、主代理需要保持干净 |
| Prompt caching | 把稳定前缀缓存，降低重复传输成本和延迟 | 工具定义长、系统提示长、重复调用多 |

这里要特别说明：`prompt caching` 主要解决的是重复前缀的成本和延迟问题，不等价于“自动解决上下文污染”。它能减少你反复发送同样的大前缀，但不能替你判断哪些历史该删、哪些工具结果该摘要。

如果不想自己从零实现，也可以先利用 provider 已经提供的原生能力：

| 原生能力 | 已验证作用 | 边界 |
| --- | --- | --- |
| Token counting API | 在请求前估算输入规模，做预警和路由 | 只能测量，不能自动治理 |
| Prompt caching | 复用长前缀，降低成本和延迟 | 不会自动压缩脏历史 |
| 独立子代理上下文 | 把长任务拆成多个干净窗口 | 需要额外编排逻辑 |
| 工具输出警告阈值 | 对超长工具返回做运行时提醒 | 提醒不等于自动修复 |

最终可以按下面这个表来选：

| 方案 | 适用场景 | 不适合的场景 |
| --- | --- | --- |
| 滑动窗口 | 短会话、实时聊天 | 需要保留长链决策 |
| 选择性保留 | 工具输出极长、JSON 很多 | 很难判断重要度的开放任务 |
| 锚定式摘要 | 长流程、多阶段任务 | 对逐字事实一致性要求极高的原文处理 |
| 子代理隔离 | 多工具并行、任务可拆解 | 流程非常短，拆分收益低 |
| Prompt caching | 固定前缀很长、重复请求多 | 脏历史主要来自动态工具结果 |

结论很简单：没有一种方案能单独解决所有问题。真正稳定的多轮 agent，通常都是“选择性保留 + 结构化摘要 + 阈值监控 + 架构隔离”的组合。

---

## 参考资料

- OpenAI Function Calling Guide：函数调用的标准闭环，明确要求应用执行工具后，把结果再次作为输入发回模型  
  https://platform.openai.com/docs/guides/function-calling

- OpenAI Responses API Reference：说明 conversation / input items 会被持续加入会话状态，适合理解“为什么工具结果会继续占上下文”  
  https://platform.openai.com/docs/api-reference/responses

- OpenAI Model Spec (2025-10-27)：模型行为与消息链条的官方定义，适合理解消息角色和链式调用的基本边界  
  https://model-spec.openai.com/2025-10-27

- Anthropic Token Counting Docs：在请求前估算 token，用于做压缩前预警和路由决策  
  https://docs.anthropic.com/en/docs/build-with-claude/token-counting

- Anthropic Prompt Caching Docs：解释长前缀缓存的适用方式和边界，适合和“上下文压缩”对照理解  
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

- Anthropic Claude Code MCP Docs：给出超长 MCP 工具输出的警告阈值，能帮助理解“工具结果过大”在工程上如何暴露  
  https://docs.anthropic.com/en/docs/claude-code/mcp

- Anthropic Subagents Docs：说明子代理使用独立上下文窗口，适合放在“架构级隔离”这一替代方案里看  
  https://docs.anthropic.com/en/docs/claude-code/sub-agents

- Factory.ai, “Compressing Context” (2025-07-21)：锚定式迭代摘要的工程思路，强调维护长期状态而非每轮重写全文  
  https://factory.ai/news/compressing-context

- Design for Uptime, “Alert When Context Window Usage Exceeds 90%”：上下文利用率监控与静默截断告警的工程建议  
  https://design-for-uptime.com/blog/ai-observability/context-window-90-percent-alert

- Praetorian, “Deterministic AI Orchestration: A Platform Architecture for Autonomous Development”：从多代理编排角度讨论上下文膨胀与上下文隔离  
  https://www.praetorian.com/blog/deterministic-ai-orchestration-a-platform-architecture-for-autonomous-development/
