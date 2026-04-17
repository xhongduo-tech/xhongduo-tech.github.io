## 核心结论

ReAct 是一种让大语言模型在“推理”和“行动”之间交替前进的工作方式。这里的“推理”指模型先用自然语言写出当前判断，“行动”指模型决定是否调用外部工具，比如搜索、数据库、天气接口；“观测”指工具返回的结果，再被送回模型继续下一轮思考。

它的核心不是“多一步工具调用”，而是把推理和工具反馈连成闭环：

| 步骤 | 产物 | 作用 |
|---|---|---|
| 1 | Thought | 说明现在缺什么信息、下一步为什么这么做 |
| 2 | Action | 按 Thought 的判断调用工具 |
| 3 | Observation | 接收工具结果，更新上下文 |
| 4 | Thought | 利用新结果继续判断 |
| 5 | Final Answer | 信息足够时结束循环并输出答案 |

可以把它理解成一个简单循环：

$$
\text{Thought} \rightarrow \text{Action} \rightarrow \text{Observation} \rightarrow \text{Thought} \rightarrow \cdots
$$

与 Chain-of-Thought，简称 CoT，只让模型连续写推理链不同，ReAct 允许模型在推理过程中主动访问外部世界。前者适合“已知信息足够”的题，后者适合“必须查资料、用工具、分多步做”的任务。

一个最小玩具例子：

- 用户问：“纽约现在气温是多少？”
- Thought：需要实时温度，内部记忆不可靠。
- Action：调用天气 API，参数 `city=New York`
- Observation：`22°C, Sunny`
- Thought：信息足够，可以回答。
- Final Answer：纽约当前约 22°C，天气晴。

这个例子看起来简单，但它揭示了 ReAct 的本质：不是让模型“猜”，而是让模型“先判断，再查，再修正”。

---

## 问题定义与边界

ReAct 解决的是一类多步决策任务：只靠模型参数里的静态知识不够，必须把外部工具接进推理过程。输入通常是一个查询 $q$，输出不是单个答案，而是一条轨迹 $\tau$。轨迹可以理解成“做事过程的记录”，包含多轮 Thought、Action 和 Observation。

形式上，可以写成：

$$
\tau = \big[(T_1, A_1, O_1), (T_2, A_2, O_2), \dots, (T_T, A_T, O_T)\big]
$$

这里的边界很重要。不是所有任务都该用 ReAct。

| 问题类型 | 是否需要工具 | 典型终止条件 |
|---|---|---|
| 纯逻辑题 | 通常不需要 | 推理完成即可 |
| 实时查询 | 需要 | 拿到最新数据 |
| 多步检索 | 需要 | 证据足够支持结论 |
| API 工作流 | 需要 | 状态变为成功或失败 |
| 开放式写作 | 不一定 | 达到目标长度或结构 |

对初学者来说，最容易混淆的一点是：ReAct 不是“任何时候都调用工具”。它要求模型先判断“是否缺信息”。如果当前上下文已经足够，继续调用工具只会增加成本和错误面。

再看一个新手版例子。任务是“查某城市温度并告知当地时间”。这不是一步任务，因为回答需要两类外部信息：天气和时间。于是模型应先 Thought：“需要实时温度和时区信息。”然后 Action 调两个工具或分两轮调用；拿到 Observation 后，再决定信息是否齐全。如果时间接口失败，下一轮 Thought 应该转成“改用备用时间工具”，而不是直接编造。

因此 ReAct 的边界至少包含三点：

1. 模型必须有“是否要调用工具”的判断能力。
2. 工具返回必须能及时写回当前上下文。
3. 系统必须能终止，不能无限循环。

---

## 核心机制与推导

ReAct 的核心机制，是在每一轮维护一个上下文 $c_t$。上下文可以理解成“到当前为止所有已知内容”，包括用户问题、前几轮推理、动作和工具返回。

第 $t$ 轮时，模型基于 $c_t$ 生成当前令牌序列 $v_t$。这些令牌中，可能包含 Thought，也可能包含 Action 指令。若触发 Action，外部工具执行后返回 Observation，再拼回上下文，形成下一轮的 $c_{t+1}$。

因此，完整轨迹的生成概率可以写成：

$$
P(\tau \mid q) = \prod_{t=1}^{T} P(v_t \mid q, v_{<t})
$$

这里：

- $q$ 是用户问题。
- $v_{<t}$ 是之前所有轮次已经生成的内容。
- $v_t$ 可以同时覆盖“当前的推理文字”和“当前的动作指令”。

这个公式的重点不在数学复杂，而在因果关系：后一步的生成依赖前一步的结果，尤其依赖 Observation。也就是说，ReAct 不是预先写好完整计划再执行，而是边走边修正。

看一个更完整的玩具例子：

| 轮次 | 上下文关心点 | Thought | Action | Observation |
|---|---|---|---|---|
| 1 | 用户问“纽约现在气温” | 需要实时温度 | `weather("New York")` | `22°C, Sunny` |
| 2 | 已知温度，但缺时间 | 需要确认当前本地时间 | `time("America/New_York")` | `15:30` |
| 3 | 信息足够 | 组合答案并结束 | `finish(...)` | 终止 |

这里最关键的不是调用了两个工具，而是第二轮 Thought 依赖第一轮 Observation。如果第一轮返回的是“城市不存在”，第二轮 Thought 就不该继续查时间，而应改成“需要用户澄清城市名称”。这就是 ReAct 相比静态提示词更强的地方。

真实工程里，这种闭环在多跳检索中更明显。多跳检索的意思是一个问题要经过多次查询才能拼出答案。以 HotpotQA 为例，问题可能是“某位演员主演的电影，其导演出生在哪个国家？”第一轮搜索演员，第二轮搜索电影，第三轮搜索导演，第四轮搜索导演出生地。每一步都不是预先固定的，而是根据上一轮 Observation 选下一步 Action。

所以，ReAct 的价值不只是“会调工具”，而是“把工具结果纳入推理轨道”。

---

## 代码实现

下面给一个最小可运行的 Python 示例。它不是完整 LLM 系统，而是一个教学版 ReAct 循环，用规则函数模拟 Thought 生成，用本地函数模拟工具调用。重点是看循环结构，而不是模型能力。

```python
from dataclasses import dataclass

@dataclass
class Step:
    thought: str
    action: str
    observation: str

def weather_tool(city: str) -> str:
    data = {
        "new york": "22C, sunny",
        "beijing": "18C, cloudy",
    }
    return data.get(city.lower(), "city_not_found")

def time_tool(city: str) -> str:
    data = {
        "new york": "15:30",
        "beijing": "04:30",
    }
    return data.get(city.lower(), "time_not_found")

def simple_policy(question: str, history: list[Step]) -> tuple[str, str]:
    if not history:
        return ("需要先查询实时天气", "weather:new york")
    if len(history) == 1 and "22C" in history[0].observation:
        return ("已经有温度，还需要补充当地时间", "time:new york")
    return ("信息足够，结束", "finish")

def run_react(question: str, max_steps: int = 5) -> tuple[list[Step], str]:
    history: list[Step] = []

    for _ in range(max_steps):
        thought, action = simple_policy(question, history)

        if action == "finish":
            final = "纽约当前 22C，晴，本地时间 15:30。"
            return history, final

        tool_name, arg = action.split(":", 1)

        if tool_name == "weather":
            observation = weather_tool(arg)
        elif tool_name == "time":
            observation = time_tool(arg)
        else:
            observation = "unknown_tool"

        history.append(Step(thought=thought, action=action, observation=observation))

    raise RuntimeError("exceed_max_steps")

steps, answer = run_react("纽约现在气温是多少？")
assert len(steps) == 2
assert steps[0].action == "weather:new york"
assert steps[1].action == "time:new york"
assert "22C" in answer
assert "15:30" in answer
print(answer)
```

这段代码对应的逻辑可以概括为：

1. 先生成 Thought。
2. 再根据 Thought 生成 Action。
3. 执行工具，得到 Observation。
4. 把三者写入历史。
5. 直到出现 `finish` 或达到最大轮次。

如果写成伪代码，就是：

```text
context = [user_query]
while step < max_steps:
    thought = model(context)
    if thought says "enough":
        return final_answer(context)
    action = model_or_parser(context + thought)
    observation = call_tool(action)
    context += [thought, action, observation]
return fallback_or_error
```

真实工程例子通常会比上面多三个模块：

| 模块 | 作用 | 工程必要性 |
|---|---|---|
| Action 解析器 | 把模型文本转成结构化参数 | 防止工具调用格式错误 |
| 循环控制器 | 限制最大轮次、检测重复调用 | 防止成本失控 |
| 失败回退策略 | 工具超时、空结果时换路由 | 提升稳定性 |

例如一个检索增强问答系统，第一轮 Thought 可能是“问题涉及最新政策，需要搜索官方文档”，Action 调搜索 API；Observation 返回 3 条结果；第二轮 Thought 判断“结果里只有新闻摘要，没有原文，需要继续打开官网 PDF”；第三轮再做答案生成。这就是典型的 ReAct 结构。

---

## 工程权衡与常见坑

ReAct 在概念上直接，在工程上却不“白送正确”。最常见的问题不是模型不会思考，而是循环和工具接口管理不好。

| 坑 | 成因 | 缓解策略 |
|---|---|---|
| 无限循环 | 模型反复认为“还需要再查一次” | 设最大轮次，加入重复检测 |
| 幻觉行动 | 模型编造不存在的工具或参数 | 固定工具白名单，做 schema 校验 |
| 工具结果未被利用 | Observation 写回了，但提示词没要求显式使用 | 强制每轮 Thought 引用上一轮 Observation |
| 过度调用工具 | 能直接回答却还继续查 | 增加“是否真的缺信息”判断 |
| 终止条件含糊 | 模型不知道何时该结束 | 显式定义 `finish` 动作 |
| 上下文膨胀 | 轮次多导致 token 成本高 | 摘要历史，只保留关键 Observation |

一个典型坑是“格式正确但语义错误”。比如模型输出了合法 Action：

```text
Action: weather(city="New York City now")
```

语法上像是对的，但参数把“now”也塞进了城市名，工具会失败。解决方法不是只靠更强模型，而是把 Action 约束成严格结构，比如 JSON Schema、函数调用签名或枚举参数。

另一个坑是重复检索。假设搜索接口连续返回空结果，模型可能一直换措辞重试，成本很快飙升。一个实用策略是做“观测去重”：如果连续两轮 Observation 等价，就强制进入失败分支，例如“当前工具无法得到结果，需要澄清问题或切换数据源”。

还要注意 Thought 的暴露范围。训练或研究里常把完整 Thought 显示出来，但生产环境不一定需要把内部推理原样暴露给用户。很多系统会保留“内部状态”和“外部可见解释”两层表示，以兼顾可调试性和安全性。

---

## 替代方案与适用边界

ReAct 不是唯一方案，它是在“需要边推理边用工具”时最自然的一类方案。判断是否该用它，关键看任务是否依赖外部反馈。

| 方法 | 是否显式工具调用 | 适合场景 | 主要限制 |
|---|---|---|---|
| CoT | 否 | 数学推导、纯逻辑分析 | 无法获取外部实时信息 |
| ReAct | 是 | 多步检索、实时问答、Agent 工作流 | 对指令遵循和工具设计要求高 |
| Toolformer | 是 | 让模型学会在文本中插入工具调用 | 依赖训练过程，接入门槛更高 |
| SayCan | 是 | 机器人任务规划与动作选择 | 更依赖物理环境和价值函数 |
| 纯工作流编排 | 通常固定 | 明确规则、低不确定任务 | 灵活性不足，难处理开放问题 |

CoT 的含义是“思维链”，白话说就是让模型把中间推理步骤写出来。它适合“题目给的信息已经够了”的情况，比如证明题、代码阅读、公式推导。若还必须查百科、看数据库、调用浏览器，仅靠 CoT 就会出现“推理很完整，但事实是猜的”这个问题。

ReAct 更适合下面几类任务：

- 问答依赖实时数据，如天气、股价、排班、日志状态。
- 问题要多跳检索，如 HotpotQA、知识图谱问答。
- 任务本身就是操作流程，如“先查订单，再发起退款，再确认状态”。

但它也有明确边界：

1. 如果模型连工具格式都经常写错，先不要上 ReAct，应先收紧接口。
2. 如果任务路径完全固定，比如“收到订单号后永远按 3 步执行”，纯工作流可能更便宜、更稳。
3. 如果外部工具延迟高、成功率低，ReAct 会把这些不稳定性直接放大到整体系统里。

可以用一句话区分：CoT 解决“怎么想”，ReAct 解决“怎么想并在必要时去做”。

---

## 参考资料

- ReAct 原始论文：Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*  
  链接：https://arxiv.org/abs/2210.03629  
  用途：框架定义、实验设定、HotpotQA/FEVER/ALFWorld/WebShop 结果。

- ReAct 项目主页  
  链接：https://react-lm.github.io/  
  用途：查看论文配图、任务示例与实验摘要。

- Emergent Mind, *ReAct Paradigm*  
  链接：https://www.emergentmind.com/topics/react-paradigm  
  用途：补充轨迹概率视角、Thought/Action/Observation 循环解释。

- IBM Think, *What is a ReAct Agent?*  
  链接：https://www.ibm.com/think/topics/react-agent  
  用途：梳理工程实现中的循环控制、工具约束和常见失败模式。

- HotpotQA 数据集论文  
  链接：https://aclanthology.org/D18-1259/  
  用途：理解多跳检索任务为何需要分步推理与证据整合。
