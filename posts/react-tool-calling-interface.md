## 核心结论

ReAct 可以理解为一种把“大模型推理”和“外部工具调用”接到同一条流水线里的接口协议。它把每一轮决策固定成 `Thought → Action → Observation` 三段：`Thought` 是当前判断，白话说就是“模型先明确自己现在要做什么”；`Action` 是结构化动作，白话说就是“把调用哪个工具、传什么参数写清楚”；`Observation` 是工具返回值，白话说就是“把外部世界的新信息重新送回模型”。

这种设计的价值不在“会不会调用工具”，而在“调用后还能不能继续正确推理”。如果只有工具调用，没有显式观察反馈，模型很容易在下一步忘记刚刚拿到的结果，或者把失败调用当成成功调用继续往下编。ReAct 把每一步的输入、动作、输出都写入轨迹，因此更适合搜索、计算器、数据库、代码执行器并存的多工具任务。

从统一接口角度看，ReAct 的关键不是某个具体工具，而是把所有工具都抽象成同一种动作对象。只要工具都满足“名称 + 输入 Schema + 输出 Schema”，计算器、搜索引擎、Python 执行器、本地知识库就能挂到同一个调度循环里。这样做后，工程上可以统一记录日志、重试、限流、权限控制和审计。

最小玩具例子是复利计算后再搜索相近收益。用户问题不是一次动作能完成的，而是两个依赖顺序的动作：先算，再查。ReAct 轨迹可以写成：

1. `Thought`: 需要先计算 10,000 美元按 5% 年化复利 10 年后的金额。
2. `Action`: `calculator(expr="10000*(1+0.05)^10")`
3. `Observation`: `16288.95`
4. `Thought`: 已得到数值结果，接下来搜索常见“约 5% 年化收益”的投资或资产类别。
5. `Action`: `rag_search(query="5% annual return investment")`
6. `Observation`: 返回若干候选资料
7. `Thought`: 合并计算结果和检索结果，形成最终回答

如果把这个过程写成决策问题，状态 $s_t$ 是截至第 $t$ 步的历史轨迹，动作空间 $A$ 是所有可调用工具，则策略为：

$$
a_t \sim \pi(a \mid s_t)
$$

其中 $\pi(a \mid s_t)$ 的含义是：在当前历史上下文下，系统选择哪个动作最合适。这个式子说明 ReAct 不只是“提示词技巧”，它本质上已经接近一个显式决策器。

---

## 问题定义与边界

问题的核心不是“如何让模型学会调 API”，而是“如何在多步任务里稳定、低成本、可审计地调 API”。当工具只有 2 到 3 个时，模型直接生成调用语句往往也能工作；但当工具目录扩展到几十个，问题马上转成路由问题，白话说就是“先决定该用哪把扳手，再决定怎么拧”。

这里有两个边界必须先说清：

| 场景 | 主要风险 | 结果 |
|---|---|---|
| 工具太多，全部塞进 Prompt | 选择混乱、上下文膨胀 | 错选工具、漏选工具 |
| 工具太少，任务很简单 | 调度开销大于收益 | 直接回答更省 |
| 工具输出过长 | 轨迹变成日志堆积 | 后续推理变慢、变贵 |
| 工具不可靠 | 错误被写入 Observation | 形成级联误差 |

“过度调用”和“完全不调用”都不是好设计。前者会带来三类可量化成本：

| 成本类型 | 过度调用的表现 | 不调用的风险 |
|---|---|---|
| Token 成本 | 每步都附带工具描述和历史结果 | 模型只能凭记忆猜 |
| 延迟成本 | 外部 API 串行等待 | 回答快但事实不稳 |
| 错误重试 | 失败调用需要重试或回退 | 错误无法被外部校验 |

一个对新手很直观的类比是：把 50 个工具一次性交给模型，就像让人站在五金店门口一口气挑 50 件工具，常见结果不是更聪明，而是更犹豫、更容易拿错。更合理的做法是先检索出 3 到 5 个候选，再做正式选择。

因此本文讨论的边界很明确：ReAct 最适合“多步、可外部验证、需要中间状态同步”的任务；如果任务只需要一句常识回答，或者工具集极小且调用格式固定，ReAct 循环可能不是最优解。

---

## 核心机制与推导

ReAct 的统一接口设计可以拆成三层。

第一层是状态表示。状态 $s_t$ 不只是用户问题，还包含之前所有的 `Thought` 和 `Observation`。白话说，系统做下一步决策时，不是只看原题，而是看“到目前为止已经做过什么、看到了什么”。

第二层是动作抽象。把不同工具统一成同一种动作结构：

$$
a_t = (\text{name}, \text{args})
$$

其中 `name` 表示工具名，`args` 表示参数。这样搜索工具和计算器的差异，只体现在参数结构不同，而不会破坏主循环。

第三层是状态更新。一次工具调用后，新的状态可以写成：

$$
s_{t+1} = f(s_t, a_t, o_t)
$$

其中 $o_t$ 是 `Observation`。这表示系统每前进一步，都把“动作”和“观察”追加回轨迹。若引入动态提示状态，也可以写成：

$$
C_t = f(C_{t-1}, I_t, E_t)
$$

这里 $C_t$ 是当前提示上下文，$I_t$ 是新输入，$E_t$ 是环境反馈。它强调的是：Prompt 不是静态模板，而是随着工具结果不断演化的工作内存。

当工具数为 $N$ 时，最朴素的方法是让模型直接在全部工具中选一个。但当 $N$ 很大时，通常先做一次粗筛，取出 $\ell \ll N$ 个候选工具，再在候选集合中做 ReAct 决策。这个做法常被称为动态工具检索。白话说，就是先在“工具仓库”里找相关抽屉，再从抽屉里挑具体工具。

简化流程如下：

1. 根据用户问题检索相关工具
2. 将候选工具 Schema 放入当前 Prompt
3. 生成 `Thought`
4. 生成 `Action`
5. 执行工具得到 `Observation`
6. 判断是否终止，否则进入下一轮

玩具例子还是复利问题。第一步必须调用计算器，而不能先搜网页，因为金额计算是确定性问题。计算结果 $10000 \times (1+0.05)^{10}$ 得出后，第二步才适合调用搜索工具去查“相近收益”的现实资产类别。这正说明 ReAct 的优势不是“会并行调很多工具”，而是“能在每一步选择当前最必要的工具”。

真实工程例子通常更复杂。比如一个代码分析智能体接到任务：“找出某个仓库里最可能导致内存泄漏的模块，并给出修复建议。”合理的流程往往是：

1. 搜索器检索相关文件
2. 代码阅读器打开候选文件
3. Python 执行器跑静态分析脚本
4. 提取器整理结果
5. 最后由回答模块汇总

这里每个子模块都可以继续用 ReAct 循环，因为“打开文件后读到什么”会影响“下一步分析哪个模块”。这就是多工具协同真正需要统一接口的原因。

---

## 代码实现

工程实现的关键，不是把 `Thought` 直接展示给用户，而是把动作层做成稳定的结构化协议。最常见的做法是为每个工具定义 Schema。

| 字段 | 含义 | 示例 |
|---|---|---|
| `name` | 工具唯一标识 | `calculator` |
| `description` | 工具能力描述 | `Evaluate arithmetic expressions` |
| `input_schema` | 输入参数格式 | `{"expr": "string"}` |
| `output_schema` | 输出结果格式 | `{"value": "number"}` |

一个最小 ReAct 调度器可以写成下面这样：

```python
from dataclasses import dataclass
from typing import Any, Dict, List
import math

@dataclass
class Action:
    name: str
    args: Dict[str, Any]

def calculator(expr: str) -> Dict[str, Any]:
    value = eval(expr, {"__builtins__": {}}, {"math": math})
    return {"value": round(float(value), 2)}

def rag_search(query: str) -> Dict[str, Any]:
    docs = {
        "5% annual return investment": [
            "investment-grade bonds",
            "dividend ETFs",
            "high-yield savings in specific rate cycles",
        ]
    }
    return {"hits": docs.get(query, [])}

def reason(state: List[Dict[str, Any]]) -> str:
    if not state:
        return "先做确定性计算，再补充外部检索。"
    last_obs = state[-1]["observation"]
    if "value" in last_obs:
        return "金额已经算出，下一步搜索相近收益的现实选项。"
    return "信息足够，准备汇总。"

def plan_tool(state: List[Dict[str, Any]], thought: str) -> Action | None:
    if not state:
        return Action(name="calculator", args={"expr": "10000*(1+0.05)**10"})
    last_obs = state[-1]["observation"]
    if "value" in last_obs:
        return Action(name="rag_search", args={"query": "5% annual return investment"})
    return None

def call_tool(action: Action) -> Dict[str, Any]:
    if action.name == "calculator":
        return calculator(**action.args)
    if action.name == "rag_search":
        return rag_search(**action.args)
    raise ValueError(f"unknown tool: {action.name}")

state: List[Dict[str, Any]] = []

while True:
    thought = reason(state)
    action = plan_tool(state, thought)
    if action is None:
        break
    observation = call_tool(action)
    state.append({
        "thought": thought,
        "action": action,
        "observation": observation,
    })

assert state[0]["action"].name == "calculator"
assert state[0]["observation"]["value"] == 16288.95
assert state[1]["action"].name == "rag_search"
assert "dividend ETFs" in state[1]["observation"]["hits"]
```

这段代码展示了统一接口的两个重点。

第一，主循环不关心工具内部逻辑，只要求工具都遵守 `Action -> Observation` 的协议。  
第二，`reason` 和 `plan_tool` 是可以替换的。生产环境里它们通常由 LLM 完成，而不是手写规则。

如果进一步做工程化，常见拆分是：

1. `Planner`：根据当前状态生成候选动作
2. `Router`：在多个工具中做最终选择
3. `Executor`：执行工具并处理超时、重试、权限
4. `Memory`：压缩历史轨迹
5. `Responder`：输出最终答案

这样做后，统一接口不仅服务于“调用工具”，也服务于“观测、记录、回放和审计”。

---

## 工程权衡与常见坑

ReAct 的主要代价是显式轨迹。显式轨迹带来可解释性，也带来成本。

最常见的坑有四类：

| 陷阱 | 现象 | 规避方式 |
|---|---|---|
| 工具暴露过多 | 模型频繁选错或犹豫 | 先检索候选工具，再加载 Schema |
| Observation 过长 | 后续每轮都在处理大段 JSON | 结果压缩、字段白名单、摘要化 |
| 工具失败未建模 | 失败结果被当成正常输入 | 区分 `success/error/retryable` 状态 |
| 无终止条件 | 不断循环调用工具 | 设置最大步数、置信阈值、停止规则 |

轨迹膨胀尤其常见。比如搜索 API 一次返回完整 JSON，包含标题、摘要、URL、打分、分页信息、调试字段，总长度 5,000 token。如果你把这个 Observation 原样放回下一轮 Prompt，那么后面每一步都要重复消化这些日志。结果不是更聪明，而是越来越慢、越来越贵。工程上通常只保留“下一步决策真正需要的字段”，例如前 3 条结果的标题和摘要。

另一个常见坑是“有工具就一定要调”。这是错误的。一个成熟调度器要允许输出 `no_action`，白话说就是“当前信息已经足够，不必再查”。否则系统会为了形式完整而过度调用工具，导致延迟和成本失控。

真实工程例子里，这个问题更明显。假设企业知识助手接入了内部搜索、文档解析、SQL 查询、图表生成四个工具。用户只问“去年营收同比增长多少”。正确流程通常是：先查指标库，再计算同比，最后直接输出；不需要文档解析，更不需要图表生成。如果没有路由约束，模型很可能把所有工具都试一遍。

因此，ReAct 在工程上不是“加几个工具”这么简单，而是要补上三套基础设施：

1. 工具检索：缩小候选集合
2. 轨迹压缩：控制上下文成本
3. 错误治理：把失败当成一等公民处理

---

## 替代方案与适用边界

ReAct 不是唯一方案。至少有三类常见替代思路。

| 方案 | 核心思路 | 适用场景 | 局限 |
|---|---|---|---|
| ReAct | 显式 `Thought → Action → Observation` 循环 | 多步、多工具、需要审计 | 轨迹长，成本高 |
| Toolformer | 在生成过程中隐式学习何时插入工具调用 | 工具较少、训练可控 | 路由理由不透明 |
| Gorilla | 直接生成 API 调用语句 | API 调用格式稳定的场景 | 多步回退能力弱 |
| 直接回答 / Few-shot | 不建工具循环 | 问题简单、外部验证不重要 | 准确性受模型记忆限制 |

Toolformer 的重点是“让模型自己学会在生成中插入工具调用”，适合工具较少、训练数据可构造的任务。它的优势是流程短，缺点是中间决策不够显式。Gorilla 更偏向“把 API 文档映射成可生成的调用语句”，适合调用语法很重要、接口边界很清晰的场景，但如果任务需要多轮观察和回退，设计难度会上升。

ReAct 的优势在于轨迹清晰。你可以准确知道“为什么调用这个工具、拿到了什么结果、下一步为什么变了”。这使它特别适合以下任务：

1. 需要多种工具组合
2. 中间结果必须可追踪
3. 工具可能失败，需要回退或重试
4. 结果要经过人工审计

它不适合的边界也很明确。如果你只有 3 个公开 API，调用格式非常稳定，而且任务大多是一跳完成，那么直接函数调用或 few-shot 往往更轻。只有当系统开始出现“先搜、再算、再查库、最后汇总”的链式结构时，ReAct 的统一接口价值才会明显超过它的额外开销。

一句话判断标准是：如果你的系统需要管理“决策轨迹”，用 ReAct；如果只需要“把请求发出去”，更轻的方案通常更合适。

---

## 参考资料

- ReAct Agent 文档：<https://ragents.readthedocs.io/en/stable/agents/react/>
- ReAct Agent Framework 综述：<https://www.emergentmind.com/topics/react-agent-framework>
- Tool Use and Dynamic Prompting：<https://www.next.gr/ai/large-language-models/tool-use-and-dynamic-prompting>
- Tool Use Optimization 工程报告：<https://zylos.ai/research/2026-03-03-ai-agent-tool-use-optimization>
