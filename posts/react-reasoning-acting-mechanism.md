## 核心结论

ReAct 是一种把推理和外部交互写成固定循环的框架。它要求模型按 `Thought -> Action -> Observation` 三步反复执行：先说明下一步为什么做，再调用工具或环境动作，最后把真实反馈接回上下文。这里的 `Thought` 是“当前打算怎么推进”，`Action` 是“可执行的规范动作”，`Observation` 是“工具或环境返回的外部事实”。

---

与纯 CoT 不同，ReAct 不把全部推理都关在模型内部；与 Act-only 不同，ReAct 也不是只做动作、不解释原因。它的核心价值不是“多输出几行中间文本”，而是把“理由”和“证据校验”绑定在一起。只要动作空间设计得当，模型每一步都能被外部反馈纠偏，因此在多跳问答、网页检索、环境决策这类任务里，通常比只推理或只行动更稳。

原始论文与后续总结里，一个常见结论是：在需要外部信息验证的任务中，ReAct 能显著降低幻觉，并提高任务完成率。以公开材料中常被引用的结果为例，HotpotQA 中相对纯推理基线，幻觉率大约下降 20%；ALFWorld 中相对 Act-only 基线，成功率提升约 34%；WebShop 中也有大约 10 个百分点的收益。它不是“任何任务都更强”，但在“问题答案依赖外部观察”的场景中非常有效。

| 任务 | 基线 | ReAct | 结论 |
| --- | --- | --- | --- |
| HotpotQA | 纯 CoT | 更低幻觉率 | 适合多跳事实拼接 |
| ALFWorld | Act-only 约 45% | 约 71% | 交互式决策收益明显 |
| WebShop | 约 30.1% | 约 40% | 减少无效点击与误判 |

一个直观比喻是“侦探办案”，但要注意这只是帮助理解，不是定义本身。真正的定义是：每个推理结论都要尽快用动作去拿证据，再让证据进入下一轮推理。这样得到的不是空转的思维链，而是被环境持续校正的决策链。

## 问题定义与边界

ReAct 要解决的问题是：当任务需要多步推理，而且中途必须访问外部信息时，怎样让模型既能规划，又能基于真实反馈修正计划。

---

这类任务有两个共同点。第一，答案不完全在参数里，而是在检索系统、网页、数据库、模拟环境或工具返回值里。第二，任务必须按步推进，前一步的观察会影响后一步的判断。如果只用 CoT，模型可能会“合理地编造”；如果只用 Act-only，系统可能机械执行动作，却不知道为什么切换策略。

所以 ReAct 的适用边界很明确：

| 任务类型 | 典型动作 | 终止判据 |
| --- | --- | --- |
| 多跳 QA | `search[...]` / `lookup[...]` / `finish[...]` | 找到足够证据并给出答案 |
| 环境交互 | `go` / `open` / `take` / `finish` | 状态满足目标 |
| 事实核查 | `search` / `lookup` / `finish` | 给出支持或反驳结论 |
| API 代理 | `query_api` / `parse` / `submit` | 返回结构化结果 |

玩具例子：问“爱因斯坦出生的国家现在的首都是哪里？”  
这不是一步能答完的问题，因为它至少包含两跳：先确认爱因斯坦出生地，再确认对应现代国家，再查首都。ReAct 会先搜索出生地，再根据观察结果决定下一跳，而不是一次性猜完整答案。

真实工程例子：用户问“找出下周从伦敦到纽约最便宜、且允许免费托运行李的航班”。  
这不是静态知识问答，而是一个受实时数据和筛选条件约束的任务。系统需要先调用航班搜索，再根据返回结果筛掉不含行李额度的选项，必要时追加查询细则，最后才能生成可执行推荐。这个流程天然适合 ReAct，因为每一步都依赖上一轮观察。

形式化地看，ReAct 每轮处理的不是单个 prompt，而是累积状态 $c_t$。只要动作集合明确、终止条件明确，系统就能在有限轮次内推进；如果动作不合法、观察不稳定、终止条件含糊，循环就会失控。

## 核心机制与推导

ReAct 的状态可以写成：

$$
c_t = (x, Thought_1, Action_1, Obs_1, \dots, Thought_t, Action_t, Obs_t)
$$

其中 $x$ 是原始问题，$Obs_t$ 是第 $t$ 轮动作带回来的观察。也可以写成递推式：

$$
c_t = c_{t-1} + Thought_t + Action_t + Observation_t
$$

---

这两个式子的含义很直接：当前决策不仅依赖原始输入，还依赖之前“想过什么、做过什么、看到了什么”。Observation 的作用尤其关键，因为它把外部世界的事实写回上下文，相当于给模型增加一条“自我监督”通道。

| 步骤 | 内容 | 作用 |
| --- | --- | --- |
| Thought | 人类可读推理 | 规划下一步 |
| Action | 规范工具调用 | 触发交互 |
| Observation | 外部反馈 | 校正后续推理 |

HotpotQA 里的经典多跳过程能说明这一点。问题不是直接问一个孤立事实，而是需要先找到中间实体，再继续追问属性。比如先搜一个地质名词对应的区域，观察结果给出 `High Plains`，下一步再查这个区域的高度范围，最后才能回答原问题。核心顺序始终是“问一跳、查一跳、再问下一跳”。

这就是 ReAct 比 CoT 更稳的原因。纯 CoT 的误差会在内部链式传播；ReAct 则把一部分中间变量外包给环境验证。它也比 Act-only 更强，因为动作不是盲点按钮，而是由推理决定。可以把它理解成一个条件决策过程：在第 $t$ 轮，模型根据 $c_{t-1}$ 生成 Thought，再选 Action；环境执行后给出 $Obs_t$；新状态 $c_t$ 继续驱动下一轮。

## 代码实现

最小实现并不复杂，本质上就是一个循环：维护历史、让模型输出 Thought 与 Action、执行动作、记录 Observation，直到 `finish`。

---

下面是一个可运行的 Python 玩具版本。它不依赖真实 LLM，而是用规则函数模拟 ReAct 循环，重点是把状态更新逻辑写清楚。

```python
from dataclasses import dataclass

KB = {
    "einstein birthplace": "Ulm",
    "Ulm country": "Germany",
    "Germany capital": "Berlin",
}

@dataclass
class Step:
    thought: str
    action: str
    observation: str = ""

def env_execute(action: str) -> str:
    assert "[" in action and action.endswith("]"), "invalid action format"
    name, arg = action[:-1].split("[", 1)
    if name == "search":
        return KB.get(arg, "NOT_FOUND")
    if name == "finish":
        return arg
    raise ValueError(f"unknown action: {name}")

def policy(question: str, history: list[Step]) -> Step:
    if not history:
        return Step(
            thought="先确定爱因斯坦出生地",
            action="search[einstein birthplace]",
        )
    last_obs = history[-1].observation
    if last_obs == "Ulm":
        return Step(
            thought="需要确认 Ulm 所属国家",
            action="search[Ulm country]",
        )
    if last_obs == "Germany":
        return Step(
            thought="需要查询 Germany 的首都",
            action="search[Germany capital]",
        )
    if last_obs == "Berlin":
        return Step(
            thought="信息已足够，可以结束",
            action="finish[Berlin]",
        )
    raise AssertionError("unexpected state")

def run_react(question: str) -> str:
    history = []
    for _ in range(8):
        step = policy(question, history)
        step.observation = env_execute(step.action)
        history.append(step)
        if step.action.startswith("finish["):
            return step.observation
    raise RuntimeError("max steps exceeded")

answer = run_react("爱因斯坦出生的国家现在的首都是哪里？")
assert answer == "Berlin"
print(answer)
```

这个例子说明三点。第一，`history` 不是日志装饰，而是下一轮决策的输入。第二，动作必须是可解析的规范格式，例如 `search[...]`、`finish[...]`。第三，终止动作和普通工具调用一样，也是状态机的一部分。

真实工程里会多两层封装。  
一层是工具层：把搜索、数据库查询、网页浏览、浏览器点击、环境执行统一成 `execute(action)`。  
另一层是解析层：从模型输出中安全提取 Thought 和 Action，并处理非法动作、超时、空观察、重复观察。

一个简化的工程循环可以写成：

```python
history = [question]
while True:
    thought, action = llm(history)
    if action.startswith("finish["):
        answer = parse_finish(action)
        break
    observation = env.execute(action)
    history.extend([thought, action, observation])
```

如果任务是 ALFWorld 一类环境代理，`env.execute(action)` 返回的是房间状态、物体状态和执行结果；如果任务是检索问答，它返回的是文档片段、命中摘要或结构化字段。

## 工程权衡与常见坑

ReAct 的理论很清楚，但工程里最常见的问题不是“不会写循环”，而是循环写出来以后不稳定。

---

第一类问题是上下文漂移。轨迹一长，模型会忘记原始问题，只盯着最近几轮 Observation。解决办法通常是每轮都重申用户目标，或者把原始问题固定放在系统提示中。Focused ReAct 里的 `Reiterate` 本质上就是这个思路。

第二类问题是重复动作。模型可能连续输出 `search[同一个关键词]`，导致 token 浪费甚至死循环。这个问题常见于搜索结果稀疏、解析失败或模型不确定时。工程上应当做动作去重和早停。

| 机制 | 作用 | 效果 |
| --- | --- | --- |
| Reiterate | 每轮重申问题 | 防止偏题 |
| Early Stop | 检查重复动作 | 避免死循环 |
| Max Step | 限制轮数 | 控制成本 |
| Action Schema | 限制动作格式 | 降低解析错误 |

第三类问题是 Observation 质量差。如果搜索返回的是噪声文本，ReAct 会把噪声写进状态，后续推理一样会偏。也就是说，ReAct 不是自动纠错器，它依赖“动作可用、观察可信”。垃圾输入仍会产生垃圾输出。

第四类问题是 Thought 泄漏与安全。某些场景不适合把完整推理暴露给用户或下游系统，这时可以保留“内部短推理 + 外部结构化动作”，只输出必要解释，而不是照搬全部思维链。

一个典型坑是把 ReAct 当成“多打印几行 prompt 工程”。这会错过重点。真正的工程重点有三个：动作空间是否封闭、Observation 是否可靠、停止条件是否严格。如果这三点没做好，再长的 Thought 也只是噪声。

## 替代方案与适用边界

ReAct 不是统一答案，它只是“需要边查边做”的任务里很强的一类方案。

---

| 方法 | 适用场景 | 失效风险 |
| --- | --- | --- |
| CoT | 纯理解、数学推导、短链 reasoning | 外部事实无法验证时易幻觉 |
| Act-only | 规则明确、动作固定的自动化流程 | 缺少中间判断，遇到变化容易错 |
| ReAct | 需要工具反馈的多步任务 | 依赖动作设计与观察质量 |
| Reflexion / ReflAct | 长时程任务、需要反思与记忆 | token 开销更高，流程更复杂 |

CoT 适合“答案主要靠内部推理”的问题，例如基础数学题、代码思路说明、静态定义解释。Act-only 适合高度流程化任务，例如表单填报、固定接口编排。ReAct 适合中间状态不确定、必须与外部世界交互的任务，例如检索增强问答、网页代理、环境导航、带工具的客服系统。

进一步的扩展方法，如 Reflexion 或 ReflAct，会在 ReAct 之外增加“反思”或“目标状态检查”。它们适合更长 horizon 的任务，也就是“完成目标需要很多步，而且中途容易走错”的任务，但代价是更高的 prompt 长度、更复杂的状态管理，以及更难调试。

因此，一个实用判断标准是：如果任务在执行前就能完整规划，优先考虑 CoT 或工作流；如果必须“做一步、看一步、再决定下一步”，ReAct 通常更合适。

## 参考资料

- Yao et al. ReAct: Reasoning and Acting in Language Models. arXiv:2210.03629  
- ReAct Framework: Synergizing Reasoning and Action, Emergent Mind  
- ReAct: Reason and Act Principle in AI, Emergent Mind  
- Focused ReAct: Improving ReAct Through Reiterate and Early Stop, Emergent Mind  
- HotpotQA few-shot ReAct demonstrations, OpenReview  
- IBM Think: ReAct agent overview  
- WebShop 与 ALFWorld 相关实验总结资料
