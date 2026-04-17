## 核心结论

Structured Outputs 和 Function Calling 不应看成两套独立能力。更准确的说法是：它们共享同一套“受限解码”机制。受限解码可以先理解成“模型每走一步，系统先检查这一步是否合法，再允许它继续生成”。

当 `strict: true` 开启后，JSON Schema 会先被转换成一套上下文无关文法，简称 CFG。上下文无关文法可以先理解成“描述合法字符串结构的一组规则”。生成时，系统不再只是“让模型猜下一个 token”，而是先根据当前历史输出计算合法 token 集合，再把非法 token 全部屏蔽。可以写成：

$$
valid\_tokens\_{next} = mask(CFG(schema), history)
$$

这意味着 strict 模式不是“更努力地遵守格式”，而是“从采样阶段就不允许违法格式”。因此它能保证输出 100% 符合 schema，而普通 best-effort 模式只是“尽量靠近 schema”，仍可能漏字段、错类型、输出多余字段。

下面这个最小玩具例子最能说明问题。假设 schema 是：

```json
{
  "type": "object",
  "properties": {
    "score": { "type": "number" }
  },
  "required": ["score"],
  "additionalProperties": false
}
```

这个 schema 的含义很简单：必须返回一个对象，而且对象里只能有 `score`，它必须是数字。进入 strict 模式后，模型生成的第一步实际上已经被 CFG 限死到了对象结构。对这个例子，合法路径几乎只有：

```json
{"score": 0.82}
```

不会出现 `{"score":"high"}`，因为 `"high"` 是字符串，不是数字；也不会出现 `{}`，因为缺少 required 字段；也不会出现 `{"score":0.82,"extra":1}`，因为 `additionalProperties: false` 禁止额外字段。

但严格控制不是免费获得的。代价主要有两类：一类是 schema 首次编译带来的额外延迟；另一类是逐 token 做合法性筛选带来的运行时开销。

| 维度 | strict CFG | best-effort |
|---|---|---|
| 生成方式 | 每步先算合法 token，再采样 | 直接按模型概率采样 |
| Token 约束 | 非法 token 直接 mask | 不做硬约束 |
| Schema 满足度 | 100%，可控到 required 和额外字段 | 可能漏字段、错类型 |
| 首次请求延迟 | 需要 schema 预处理与编译 | 基本没有 |
| 首 token 延迟 | 往往更高，常见多出约 0.5 到 1 秒 | 更低 |
| 后端接入方式 | 可直接进入确定性流程 | 常要补解析、补校验、补重试 |
| 典型适用场景 | 工具调用、数据库写入、订单流 | 原型验证、探索式抽取 |

结论可以压缩成一句话：如果你要的是“格式可靠性”，strict 是语法层保证；如果你要的是“响应灵活性”，best-effort 往往更轻。

---

## 问题定义与边界

问题本身不是“怎么让模型输出 JSON”，而是“怎么让模型输出的 JSON 在进入工程系统前就已经合法”。这里的“合法”不是人眼看着像 JSON，而是严格满足既定 schema 的全部约束。

这件事通常发生在两类场景。

第一类是结构化响应。比如你希望模型返回一个意图识别结果：

```json
{
  "intent": "book_meeting",
  "date": "2026-03-08",
  "duration": 30
}
```

第二类是工具调用。工具调用可以先理解成“模型不是直接回答，而是构造一个函数参数对象交给程序执行”。例如预约电话函数只接受：

```json
{
  "date": "2026-03-08",
  "duration": 30
}
```

如果 `date` 缺失、`duration` 变成 `"half hour"`、或者多出一个后端不认识的字段，业务系统就会失败，甚至产生错误副作用。

对新手来说，这里要分清三层概念：

| 层次 | 关注点 | 例子 |
|---|---|---|
| JSON 语法合法 | 能不能被解析器读出来 | `{"duration":30}` |
| Schema 结构合法 | 字段名、类型、必填项是否满足约束 | `duration` 必须是 number |
| 业务语义合法 | 值在业务上是否合理 | `duration` 不能是 `-100` |

strict 模式只解决第二层，不自动解决第三层。

它保证的是“结构合法”，不是“业务真值正确”。比如 schema 规定 `duration` 是 number，那么 `-100` 只要仍被 schema 接受，结构上就是合法的；但业务上它可能毫无意义。因此 strict 替代的是一部分解析校验，不是全部业务规则。

它还要求约束在生成期间就参与决策。原因很直接：如果等模型生成完再检查，错误路径已经被走完了，只能重试或补救；而 strict 的目标是让错误路径在采样前就被剪掉。于是每一步都要做：

$$
P(token \mid history, schema) = 0,\quad token \notin valid\_tokens\_{next}
$$

也就是说，任何会导致 future JSON 非法的 token，都直接把概率置为 0。

可以把这个过程理解成一棵搜索树：

$$
\text{所有候选分支} \xrightarrow{\text{schema 剪枝}} \text{仅保留合法分支} \xrightarrow{\text{模型采样}} \text{选择下一步}
$$

用一个新手向的预约例子看更直观。假设函数要求：

```json
{
  "type": "object",
  "properties": {
    "date": { "type": "string" },
    "duration": { "type": "number" }
  },
  "required": ["date", "duration"],
  "additionalProperties": false
}
```

普通模式下，模型可能输出 `{"duration":30}`，然后忘了 `date`。strict 模式下，只要当前分支最终无法满足 required，系统就不会让这条分支继续走到完成状态。这样 Agent 收到结果后，可以直接交给日程系统，而不是再写一堆容错逻辑。

边界还要再补一条：strict 也不保证“工具一定被调用”或者“调用一定符合真实用户意图”。它只保证“如果给出了该 schema 下的结构化结果，这个结果在结构上合法”。意图识别错、工具选择错、业务时机错，仍然要靠提示词、流程设计和业务规则处理。

---

## 核心机制与推导

核心机制可以拆成三步：schema 预处理、CFG 构建、逐 token 受限采样。

先说 schema 预处理。JSON Schema 是一种声明式约束语言，声明式可以先理解成“只描述结果必须满足什么，不描述具体怎么生成”。系统会把这种声明转成 CFG。CFG 里的终结符可以理解成“最终会出现在输出中的 JSON token”，非终结符可以理解成“还没展开完的结构规则”。

例如这个对象：

```json
{
  "user": {
    "name": "Ada",
    "age": 18
  }
}
```

如果 schema 规定：

- 顶层必须有 `user`
- `user` 必须有 `name` 和 `age`
- `name` 是字符串
- `age` 是整数

那么当生成到：

```json
{"user":{"name":"
```

此时合法的后续 token 只可能是字符串内容相关 token。模型不能突然跳去输出 `"age"`，因为当前字符串还没闭合；也不能直接结束 `user`，因为 `age` 还没满足 required。也就是说，CFG 不只是检查“最后结果像不像”，而是在每一步维护“当前还欠哪些结构义务”。

可以把“还欠哪些结构义务”理解成一个运行中的栈或状态集合：

| 当前历史 | 已满足约束 | 仍未满足约束 | 下一步允许什么 |
|---|---|---|---|
| `{` | 顶层对象开始 | 必须出现 `score` | 只能生成 `"score"` |
| `{"score":` | key 已出现 | 需要一个 number 值 | 只能生成数字起始 token |
| `{"score":0.8` | 数值已完成 | 顶层对象需闭合 | 只能生成 `}` |

伪代码可以写成：

```python
cfg = compile_schema(schema)
history = ""
while not done(history):
    valid = cfg.next_terminals(history)
    next_token = sample_from_model(valid_only=valid)
    history += next_token
```

这里的 `compile_schema` 表示把 schema 变成可增量匹配的文法结构，`next_terminals` 表示“基于当前历史，下一步哪些 token 合法”。

如果写得更完整一点，受限采样本质上是在做下面这件事：

```python
logits = model(history)
masked_logits = apply_mask(logits, valid_tokens)
next_token = sample(masked_logits)
```

其中：

- `logits` 是模型原始预测分数
- `valid_tokens` 是 CFG 算出的合法 token 集
- `apply_mask` 会把非法 token 的分数改成负无穷
- `sample` 再只从剩下的合法 token 里采样

玩具例子最适合看 mask 的效果。还是用 `score:number` 的 schema：

1. 历史为空时，合法起始只能是 `{`
2. 生成 `{` 后，合法键只剩 `"score"`
3. 生成 `"score":` 后，合法值只能是 number 开头
4. 数字结束后，合法后继只能是 `}`，因为没有别的字段允许出现

这就不是“模型恰好懂规则”，而是“系统只给它合法选项”。

如果继续追问“为什么 Function Calling 和 Structured Outputs 是同一种机制”，可以这样理解：

| 能力表面形态 | 用户看到的结果 | 底层约束对象 |
|---|---|---|
| Structured Outputs | 直接得到结构化 JSON | 响应 schema |
| Function Calling | 得到 `function_name + arguments` | 工具参数 schema |

两者的区别主要在“结果被谁消费”：

- Structured Outputs 通常由调用方直接消费 JSON 结果
- Function Calling 通常由框架或程序继续把 `arguments` 交给某个函数执行

但在生成阶段，二者都要回答同一个问题：当前历史下，哪些 token 还能保证最终满足 schema。底层机制因此统一。

更真实一点的工程例子是 RAG Agent。RAG 可以先理解成“先检索资料，再让模型基于资料回答”。假设 Agent 有三个工具：

- `query_db(query: string)`
- `create_event(date: string, duration: number)`
- `notify_user(channel: string, message: string)`

如果这些工具都用 strict schema 定义，那么模型每次产出的函数参数都天然满足结构约束。数据库工具不会收到数组而不是字符串，日程工具不会收到缺失 `date` 的对象，通知工具不会收到额外垃圾字段。这样控制流就从“模型输出文本，后端猜它是什么意思”，变成“模型在受约束的状态机里选择下一步动作”。

从工程角度看，这种机制的价值不是“更像 JSON”，而是“把非确定性语言模型嵌入一个确定性执行边界里”。

---

## 代码实现

在实现层面，重点不是把 schema 写出来，而是把 schema 写得适合 strict。最常见的要求有两个：

- 所有业务必需字段都要出现在 `required`
- 对象通常要加 `additionalProperties: false`

原因很简单。strict 的价值在于“边界封死”。如果你允许额外字段，或者必需字段不完整，模型虽然仍会生成合法 JSON，但对后端来说仍可能是不稳定输入。

先给一个真正可运行的 Python 教学示例。它不是 OpenAI SDK 的真实实现，而是一个简化版状态机，用来模拟“下一步只能从合法 token 中选”。

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecoderState:
    history: str = ""


def next_valid_tokens(history: str) -> list[str]:
    transitions = {
        "": ["{"],
        "{": ['"score"'],
        '{"score"': [":"],
        '{"score":': ["0", "1"],
        '{"score":0': [".", "}"],
        '{"score":1': [".", "}"],
        '{"score":0.': ["0", "5", "8"],
        '{"score":1.': ["0"],
        '{"score":0.0': ["}"],
        '{"score":0.5': ["}"],
        '{"score":0.8': ["}"],
        '{"score":1.0': ["}"],
    }
    return transitions.get(history, [])


def choose_token(valid_tokens: list[str]) -> str:
    if not valid_tokens:
        raise ValueError("No valid tokens left. History is on an illegal branch.")
    # 教学目的：固定选择第一个合法 token，模拟“只能在合法集合内选择”
    return valid_tokens[0]


def greedy_decode() -> str:
    state = DecoderState()

    while True:
        valid = next_valid_tokens(state.history)
        if not valid:
            break
        token = choose_token(valid)
        state.history += token
        if state.history.endswith("}"):
            return state.history

    raise ValueError(f"Decode failed. Current history: {state.history}")


if __name__ == "__main__":
    result = greedy_decode()
    assert result == '{"score":0.0}'
    print(result)
```

这个脚本可以直接运行，输出结果是：

```json
{"score":0.0}
```

这个例子说明的不是“怎么手写解析器”，而是 strict 运行时本质上就在做类似事情：它维护合法后继集合，然后只在集合内部采样。你可以自己改坏几处状态，观察程序如何立刻走到“无合法 token”分支。

如果想更贴近真实业务，可以看一个带校验函数的预约示例。这个例子同样可以直接运行。

```python
from __future__ import annotations

import json


appointment_schema = {
    "name": "appointment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string"},
            "duration": {"type": "number"}
        },
        "required": ["date", "duration"],
        "additionalProperties": False
    }
}


def validate_appointment(payload: dict) -> None:
    allowed_keys = {"date", "duration"}
    required_keys = {"date", "duration"}

    if set(payload.keys()) != required_keys:
        raise ValueError(f"keys must be exactly {required_keys}, got {set(payload.keys())}")

    if not isinstance(payload["date"], str):
        raise TypeError("date must be a string")

    if not isinstance(payload["duration"], (int, float)):
        raise TypeError("duration must be a number")

    if payload["duration"] <= 0:
        raise ValueError("duration must be > 0")


def create_event(date: str, duration: float) -> str:
    return f"event(date={date}, duration={duration})"


if __name__ == "__main__":
    # 这里模拟 strict 模式已经产出了结构合法的 arguments
    arguments_json = '{"date":"2026-03-08","duration":30}'
    arguments = json.loads(arguments_json)

    validate_appointment(arguments)
    result = create_event(**arguments)

    print("validated:", arguments)
    print("created:", result)
```

这个示例故意分成两层：

- 第一层是 schema 负责的结构约束：字段齐不齐、类型对不对、有没有额外字段
- 第二层是业务代码负责的语义约束：`duration > 0`

这是工程里最常见也最容易混淆的边界。

如果是实际接入大模型接口，schema 通常会长这样：

```json
{
  "name": "appointment",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "date": { "type": "string" },
      "duration": { "type": "number" }
    },
    "required": ["date", "duration"],
    "additionalProperties": false
  }
}
```

这个定义的含义是：函数参数必须是一个对象，且只有 `date` 和 `duration` 两个字段。只要模型成功给出调用结果，后端就可以把 `arguments` 直接交给业务函数，而不用先写一轮脆弱的字符串解析。

真实工程里还要加 schema 缓存或预热。预热可以先理解成“在真实流量到来前，先让系统把 schema 编译一遍”。这样能把首次请求的编译成本移出用户路径。

实践建议可以总结成下表：

| 实现要点 | 建议 | 原因 |
|---|---|---|
| Schema 设计 | 先压平对象层级，减少不必要嵌套 | 减少文法状态数 |
| 字段命名 | 用稳定、短小、业务含义明确的 key | 降低提示词和后端分歧 |
| 必填字段 | 所有生产必需字段都进 `required` | 防止结果“看似成功，实际缺参” |
| 额外字段 | 默认使用 `additionalProperties: false` | 封死后端不认识的输入 |
| 枚举值 | 能枚举就枚举，如 `status`、`channel` | 进一步缩小合法空间 |
| Schema 缓存 | 相同 schema 复用，避免重复编译 | 降低首次延迟抖动 |
| 错误处理 | 对 `refusal`、schema 合法但业务不合法分别处理 | 两类错误来源不同 |
| 预热策略 | 启动时预编译高频 schema | 把冷启动成本移出用户路径 |
| 调用编排 | strict 工具尽量串行控制 | 简化状态管理与失败恢复 |

再补一个新手常忽略的点：不要把 schema 写成“什么都能接收”的宽松版本，然后指望 strict 自动带来稳定性。strict 只能严格执行你写下的约束，不能替你补全缺失的约束。Schema 写得松，输出就会“合法但没用”。

---

## 工程权衡与常见坑

strict 的主要收益是稳定，主要代价是延迟和灵活性下降。不要把它理解成“永远更高级”，而应理解成“在可靠性优先时更合适”。

第一个权衡是首次延迟。schema 首次使用时要预处理并编译成 CFG，这一步可能从几秒到更长；即使 schema 已编译，首 token 延迟通常也会比普通模式高一些，常见可感知增量约在 0.5 到 1 秒。对聊天类产品，这可能直接影响体感；对数据库写入、工单创建、支付编排这类流程，这点延迟通常可以接受。

第二个权衡是并行能力。strict 场景下，工具调用往往更适合串行编排，因为你追求的是“每一步参数都绝对合法，再进入下一步”。如果你一开始就把多个工具并发发散出去，虽然吞吐更高，但控制面会明显复杂。

第三个权衡是 schema 复杂度。schema 越复杂，编译和运行时状态管理越重。特别是频繁变化、每次都不同的动态 schema，可能把 strict 的优势吃掉。

可以把这个权衡写成一个更工程化的公式：

$$
总成本 = 预处理成本 + 运行时约束成本 + 失败恢复成本
$$

对 strict 来说：

$$
总成本\_{strict} \approx CFG\ 编译 + token\ masking + 少量业务校验
$$

对 best-effort 来说：

$$
总成本\_{best} \approx 自由生成 + JSON\ 解析 + schema\ 校验 + 重试 + 容错分支
$$

哪边更划算，不看单次请求速度，而看整条链路的总成本和稳定性。

常见坑与对策如下：

| 坑 | 结果 | 规避方式 |
|---|---|---|
| 忘记写 `required` | 返回对象结构不稳定 | 把业务必需字段全部列入 `required` |
| 忘记 `additionalProperties: false` | 后端收到未知字段 | 默认禁止额外字段 |
| 把结构校验当业务校验 | 合法 JSON 仍可能是错业务值 | 再加业务层验证 |
| schema 每次都变 | 编译成本反复出现 | 抽象成少量稳定 schema |
| 首次请求直接上生产 | P95 延迟抖动大 | 启动时预热高频 schema |
| 没处理 `refusal` | 消费端直接崩 | 把拒绝作为独立分支处理 |
| schema 过深过宽 | 状态复杂、排障困难 | 优先扁平化、拆小工具 |
| 一个工具塞太多职责 | 参数多、含义混杂 | 按业务动作拆成多个工具 |

下面给一个典型失败链路，能更直观看出 strict 的价值：

| 阶段 | best-effort 常见问题 | strict 的改进点 |
|---|---|---|
| 模型生成 | 少字段、错字段、值类型漂移 | 结构错误在生成阶段被剪掉 |
| 参数解析 | `json.loads` 失败、字段名拼错 | 通常可直接拿结构化结果 |
| 工具执行 | 函数签名不匹配 | 参数对象已满足 schema |
| 故障恢复 | 要重试、补默认值、写兼容层 | 只需处理业务校验和拒绝 |
| 线上排障 | 难分辨是模型错还是解析错 | 结构层与业务层边界更清楚 |

一个真实工程例子是 RAG Agent 同时要查数据库、建日程、发通知。很多团队会想把 `query_db`、`create_event`、`notify_user` 一次性都做成 strict 并并发调用。但更稳妥的做法往往是：先预热高频 schema，再让 Agent 按步骤串行执行。这样第一次大延迟不会落到真实用户请求上，执行阶段只承受较小的首 token 开销，整体 P95 反而更稳定。

对新手还有一个经验性判断标准：

- 如果一次错误输出只会让页面显示差一点，strict 不是刚需
- 如果一次错误输出会触发写库、扣费、发通知、创建工单，strict 通常值得

---

## 替代方案与适用边界

不是所有场景都该用 strict。

如果你的需求是实验性质强、schema 变化频繁、失败可以接受，那么 best-effort 加后处理往往更划算。后处理可以先理解成“让模型先自由输出，再在程序里做解析和重试”。它的好处是启动快、改 schema 成本低、对探索型开发更友好。

可以把两种成本粗略写成：

$$
strict\ 成本 = CFG\ 编译延迟 + 串行约束成本
$$

$$
best\text{-}effort\ 成本 = 运行时验证 + 解析失败重试
$$

应该选哪种，不取决于谁“更先进”，而取决于哪边总成本更低。

适合 strict 的场景：

- 数据库 CRUD
- 工单创建
- 金融或订单类操作
- 自动化工作流编排
- 任何“格式错一次就会引发后续副作用”的系统

适合 best-effort 的场景：

- 快速原型
- 一次性分析任务
- schema 高频变化的实验
- 最终仍需要人工确认的流程
- 允许重试和容错的弱约束任务

可以再用一张对比表把边界说清楚：

| 维度 | strict 更适合 | best-effort 更适合 |
|---|---|---|
| Schema 稳定性 | 稳定，长期复用 | 高频变化，临时试验 |
| 错误代价 | 高，一次错就有副作用 | 低，失败可重试 |
| 响应速度诉求 | 可接受更高首包延迟 | 更看重交互流畅 |
| 后端控制要求 | 强，需要确定性输入 | 弱，可接受后处理 |
| 开发阶段 | 上线、收敛、固化阶段 | 探索、验证、试错阶段 |

一个简单对比是：临时实验场景里，今天提取 `title` 和 `summary`，明天又换成 `keywords`、`audience`、`risk_level`，schema 每次都改，这时用普通 function calling 加 `json.loads` 和运行时校验通常更省事。等字段稳定、调用路径固定后，再切换到 strict，把格式校验前移到生成阶段。

如果要给出一个落地决策顺序，可以直接按下面判断：

1. 先问错误代价高不高。高，就优先考虑 strict。
2. 再问 schema 稳不稳定。不稳定，先用 best-effort。
3. 最后问链路里有没有写库、调用外部系统、真实副作用。有，就尽量把 strict 放到副作用前。

所以最终边界很明确：strict 适合稳定、高价值、强约束流程；best-effort 适合变化快、试错多、可重试流程。

---

## 参考资料

1. OpenAI, Introducing Structured Outputs in the API  
   https://openai.com/index/introducing-structured-outputs-in-the-api/?utm_source=openai

2. OpenAI Platform Docs, Function Calling 指南  
   https://platform.openai.com/docs/guides/function-calling/how-do-i-ensure-the-model-calls-the-correct-function?utm_source=openai

3. Vercel AI Discussion, Structured Outputs 首次 schema 延迟讨论  
   https://github.com/vercel/ai/discussions/3656?utm_source=openai

4. CodeAwake, Structured Outputs 在工程抽取场景中的实践  
   https://codeawake.com/blog/structured-outputs?utm_source=openai
