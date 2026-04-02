## 核心结论

ReAct 的推理-行动循环，本质上是把大语言模型从“只会一次性吐出答案的文本生成器”，改造成“会边想、边查、边修正的执行器”。这里的“推理”是指模型先用自然语言整理下一步要做什么；“行动”是指模型调用搜索、数据库、代码执行器等工具；“观测”是工具返回的结果。三者按顺序反复迭代，形成闭环。

它为什么成立，关键不在“模型更聪明了”，而在“模型每一步都被外部事实纠偏”。单纯让模型直接回答，容易出现幻觉，也就是模型编造看似合理但并不存在的事实。ReAct 让模型在关键节点先暂停，明确自己缺什么信息，再通过工具把缺失信息补回来，因此推理链和执行链同步前进。

形式化地看，任务输入为 $q$，第 $t$ 轮决策前的状态写成：

$$
c_t=(q,\tau_1,a_1,o_1,\dots,\tau_{t-1},a_{t-1},o_{t-1})
$$

其中 $\tau_i$ 是第 $i$ 轮的 Thought，白话说就是“这一轮模型脑子里写下的工作笔记”；$a_i$ 是 Action，也就是实际调用的工具动作；$o_i$ 是 Observation，也就是工具回传结果。模型在状态 $c_t$ 上生成下一步决策：

$$
\pi(a_t \mid c_t)=\text{LLM}[\text{Prompt}(c_t)](a_t)
$$

这里的策略 $\pi$，白话说就是“当前上下文下，模型决定下一步输出什么的规则”。这个输出可能是新的 Thought，也可能是工具 Action，直到模型发出 finish 信号。

玩具例子最容易理解：

1. 用户问：“《时间简史》的作者是谁？”
2. Thought：“我需要作者名和出版年。”
3. Action：`search["《时间简史》作者"]`
4. Observation：“作者霍金，初版时间 1988”
5. Thought：“信息完整，可以回答。”
6. finish：“《时间简史》的作者是史蒂芬·霍金，初版于 1988 年。”

这个例子说明，ReAct 不是先闭门推理到底，再统一输出；而是让推理和查证交替发生。

---

## 问题定义与边界

ReAct 解决的问题不是“怎么让模型说得更多”，而是“怎么让模型在开放环境里更可靠地完成任务”。开放环境的意思是：模型不是只面对一道静态题，而是要和搜索引擎、知识库、订单系统、代码环境之类的外部工具交互。

如果没有 ReAct，一般会出现两类失真：

| 失真类型 | 表现 | 根因 | 结果 |
|---|---|---|---|
| 纯语言幻觉 | 没查资料就直接回答 | 模型把概率高的文本当成事实 | 答案看似流畅但错误 |
| 盲目工具调用 | 连问题都没拆清楚就乱查 | 缺少显式中间推理 | 工具成本高、结果还不稳定 |

所以 ReAct 的问题定义可以写得很直接：在需要工具交互的任务里，如何让模型既不过度想象，也不过度调用工具。

它的边界也必须说清。ReAct 不是所有任务都适合。它通常适用于以下场景：

| 场景 | 是否适合 ReAct | 原因 |
|---|---|---|
| 事实问答且需要外部验证 | 适合 | 可通过搜索或知识库补证据 |
| 多轮客服 | 适合 | 每轮都可能决定“回答”还是“继续查” |
| 复杂规划但几乎不需要工具 | 一般 | ReAct 的行动环节收益不大 |
| 纯数学证明或封闭题推理 | 视情况而定 | 如果没有外部工具，CoT 往往更便宜 |
| 工具调用极贵且延迟敏感 | 不一定适合 | 多轮循环会增加成本 |

新手版边界例子可以看客服系统。用户问：“我的订单已经签收，但发票还没开，怎么办？” 一个 ReAct 代理不会立刻编一套流程，而会先想：“我需要查发票政策和订单状态。” 然后它调用知识库检索政策，再调用订单 API 看状态，拿到 Observation 后再决定是直接答复，还是继续追问用户信息。

但如果你不给它边界条件，它就可能一直循环。所以工程里必须提前定义：

| 风险边界 | 典型表现 | 规避策略 |
|---|---|---|
| 无限循环 | 一直 Thought/Action 不 finish | 设置 `max_steps` 和明确终止信号 |
| 工具失败 | 工具空返回却被当成成功 | 对 Observation 做格式校验 |
| 上下文过长 | 历史轨迹越滚越长 | 裁剪状态、摘要旧轨迹 |
| 权限越界 | 模型调用不该调用的工具 | 工具白名单和参数校验 |

---

## 核心机制与推导

ReAct 的核心机制可以压缩成一句话：模型在每一轮都必须回答“我现在是继续思考，还是执行动作”。

它的状态更新过程是：

$$
c_{t+1}=
\begin{cases}
(c_t,\tau_t), & \text{若本轮只产生 Thought} \\
(c_t,\tau_t,a_t,o_t), & \text{若本轮执行了 Action 并得到 Observation}
\end{cases}
$$

这意味着状态不是“最后一个答案”，而是“到目前为止发生过什么”。这点很重要，因为很多失败案例都来自状态设计太差。比如只把最近一条 Observation 发回模型，模型就会忘掉前面已经验证过的约束，导致来回打转。

可以把它画成一个极简状态流：

| 阶段 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Thought | 当前状态 $c_t$ | $\tau_t$ | 判断缺什么、下一步干什么 |
| Action | $\tau_t$ + 状态 | $a_t$ | 调用外部工具 |
| Observation | 工具执行结果 | $o_t$ | 用外部事实更新上下文 |
| Finish | 完整状态 | 最终答案 | 在证据足够时终止 |

继续用“《时间简史》作者”这个玩具例子：

1. 初始状态只有用户问题 $q$。
2. 模型先输出 Thought：“先拿到作者和出版年。”
3. 再输出 Action：搜索对应关键词。
4. 工具返回 Observation：“霍金，1988。”
5. 模型读取新状态后，输出 Thought：“信息已齐。”
6. 最后 finish。

这个过程中，Thought 不是给用户看的花哨文案，而是内部控制信号。它帮助模型显式表达“为什么现在要调用这个工具”。这就是 ReAct 相比纯函数式工具调用更强的地方：它不仅执行，还保留了执行前的理由。

真实工程例子更能看出价值。设想一个企业客服代理，需要处理“退款规则 + 订单状态 + 优惠券补偿”三类信息：

1. Thought：“先确认用户诉求属于退款还是售后。”
2. Action：调用意图分类器。
3. Observation：返回“退款相关”。
4. Thought：“需要先查订单，再查退款政策。”
5. Action：查订单 API。
6. Observation：返回“订单已签收 5 天”。
7. Thought：“还需对照退款时限政策。”
8. Action：检索退款规则文档。
9. Observation：返回“签收后 7 天可申请无理由退款”。
10. finish：给出可执行答复。

这里 ReAct 成立，不是因为它“模仿了人类思考”，而是因为它把“状态不完整时不能贸然回答”编码进了循环结构。

---

## 代码实现

最小实现并不复杂，重点在三件事：一是让模型输出可解析的结构；二是把工具调用包起来；三是每轮都更新状态并判断是否终止。

下面是一个可运行的简化版 `python` 示例。它不用真实 LLM，而是用规则函数模拟一个 ReAct 代理，这样可以直接运行并通过 `assert` 验证循环逻辑。

```python
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Step:
    kind: str   # "thought" | "action" | "finish"
    content: str


def fake_llm(state: List[Dict[str, Any]]) -> Step:
    question = state[0]["question"]
    has_obs = any(x["type"] == "observation" for x in state)

    if "时间简史" in question and not has_obs:
        return Step("thought", "我需要作者名和出版年")
    if state[-1]["type"] == "thought" and state[-1]["content"] == "我需要作者名和出版年":
        return Step("action", 'search["时间简史 作者 出版年"]')
    if has_obs:
        obs = next(x["content"] for x in state if x["type"] == "observation")
        if "霍金" in obs and "1988" in obs:
            return Step("finish", "《时间简史》的作者是史蒂芬·霍金，初版于1988年")
    return Step("finish", "信息不足，无法回答")


def execute_tool(action: str) -> str:
    if action == 'search["时间简史 作者 出版年"]':
        return "作者：史蒂芬·霍金；出版年：1988"
    return "ERROR: unknown action"


def react_loop(question: str, max_steps: int = 5) -> str:
    state: List[Dict[str, Any]] = [{"type": "question", "question": question}]

    for _ in range(max_steps):
        step = fake_llm(state)

        if step.kind == "thought":
            state.append({"type": "thought", "content": step.content})
            continue

        if step.kind == "action":
            state.append({"type": "action", "content": step.content})
            obs = execute_tool(step.content)
            state.append({"type": "observation", "content": obs})
            continue

        if step.kind == "finish":
            return step.content

    raise RuntimeError("max steps exceeded")


answer = react_loop("《时间简史》的作者是谁？")
assert "霍金" in answer
assert "1988" in answer
print(answer)
```

这个示例虽然简单，但把 ReAct 最关键的骨架保留了下来：

| 模块 | 作用 | 工程上要注意什么 |
|---|---|---|
| `fake_llm` | 生成 Thought / Action / finish | 真实环境中必须输出结构化格式 |
| `execute_tool` | 执行动作 | 需要超时、重试、错误码 |
| `state` | 保存轨迹 | 不能无限增长 |
| `max_steps` | 防止死循环 | 必须有硬上限 |

如果换成真实 LLM，最常见做法是要求它输出 JSON，比如：

```json
{"type":"action","tool":"search","input":"时间简史 作者 出版年"}
```

这样做的原因很现实：自然语言标签容易解析错，JSON 更适合接工程系统。真正稳定的 ReAct，不是“Prompt 写得像论文”，而是“输出格式、工具协议、终止条件都写死”。

---

## 工程权衡与常见坑

ReAct 在方法上很直观，但工程上并不轻松。它的好处是透明，代价是链路更长、故障点更多。

最常见的坑有三类。

| 坑 | 具体表现 | 后果 | 解决方式 |
|---|---|---|---|
| 无限循环 | 模型一直说“再查一下” | 成本失控、延迟飙升 | `max_steps`、终止模板、重复动作检测 |
| 工具静默失败 | API 返回空结构但没报错 | 模型误以为拿到证据 | 校验 Observation 字段完整性 |
| 上下文溢出 | 历史记录过长 | 模型忘重点、费用上升 | 摘要旧轨迹，只保留关键状态 |

先看无限循环。模型很容易进入这种模式：Thought 说需要更多信息，Action 又查到一堆边缘信息，Observation 不够直接，于是继续查。解决办法不是只靠 prompt 说“请简洁”，而是工程上加硬约束。比如超过 6 步必须 finish，或者同一个工具参数连续调用两次就触发降级处理。

再看工具静默失败。这在真实系统里比“明确报错”更危险。明确报错至少能让模型知道失败了，但静默失败会让模型基于空结果继续推理。例如客服检索 API 返回：

```json
{"result": []}
```

如果代理只看到“有返回对象”，就可能误判为“知识库没有这条规则”，然后瞎回答。正确做法是定义 Observation 合法性，比如必须满足 `status == "ok"` 且 `result` 非空；否则把错误也写回状态，让模型知道“这一步没有成功”。

真实工程例子是企业客服。在订单检索链路里，模型可能先查订单，再查物流，再查退款政策。如果订单工具偶发超时，但 SDK 把超时吞掉返回空字典，模型就可能跳过订单状态确认，直接根据政策给出错误建议。这类问题靠“更强的模型”解决不了，只能靠工具层的结构化校验。

第三个坑是上下文爆炸。ReAct 天生会积累轨迹，步数越多，prompt 越大。一个简单方法是做状态抽象，只保留“已确认事实、待解决问题、最近一次工具结果”三类摘要，而不是把所有 Thought 原文都塞回去。否则模型后面会把早期试探性推理误当最终结论。

工程上还要权衡日志透明度与安全性。Thought 日志对调试很有用，但如果直接暴露给用户，可能泄露内部策略或中间不确定信息。所以生产系统常见做法是区分两层输出：内部保留完整轨迹，外部只展示必要解释。

---

## 替代方案与适用边界

ReAct 不是唯一方案。理解它的边界，必须拿替代方案对比。

| 方案 | 是否有显式行动 | 透明度 | 成本 | 适合任务 |
|---|---|---|---|---|
| CoT | 否 | 中 | 低 | 不需要工具的推理题 |
| ReAct | 是 | 高 | 中到高 | 多工具、多轮交互任务 |
| Plan-Act | 有，但先规划后执行 | 中到高 | 中 | 结构稳定、步骤较长的任务 |

CoT，Chain of Thought，白话说就是“把中间推理写出来再给答案”。它适合不需要外部工具的场景，比如一道数学应用题或代码阅读题。它的问题在于，一旦任务依赖外部事实，CoT 也只是把“猜测过程”写得更长，并不会自动查证。

Plan-Act 则是“先做总计划，再按计划执行”。它适合任务结构比较稳定的场景，比如固定流程的工单处理、固定模板的数据清洗。相比 ReAct，它减少了每一步都重新决策的开销，但弱点是计划一旦基于错误前提，后续执行会整体偏掉。

新手对比可以这样理解：

1. CoT：先想完，再说。
2. ReAct：想一步，做一步，再根据结果继续想。
3. Plan-Act：先列完整计划，再按计划做。

如果问题只是“解释 TCP 三次握手”，优先用 CoT 就够了，因为没有工具调用需求。如果问题是“帮我确认某个订单能否退款”，ReAct 更合适，因为它需要查订单、查政策、看观察结果再决定。如果工具调用非常贵，比如每次都要访问付费外部服务，那么可以考虑混合方案：先用 Plan 做粗规划，再只在必要节点进入 ReAct 循环。

所以适用边界可以总结为：

| 条件 | 更推荐的方案 |
|---|---|
| 无工具、封闭问题 | CoT |
| 多工具、开放问题、要求过程透明 | ReAct |
| 流程稳定、希望减少在线决策次数 | Plan-Act |
| 工具很贵但仍需局部查证 | 计划 + ReAct 混合 |

ReAct 的强项不是“理论上最优”，而是“在不确定环境里，能把推理和证据绑定起来”。它尤其适合需要审计日志、错误回放、人工接管的系统，因为每一步 Why 和 What 都能追溯。

---

## 参考资料

1. AgentWiki: ReAct 框架综述  
   价值：适合先建立整体概念，理解 Thought、Action、Observation 的基本闭环。  
   链接：https://agentwiki.org/react_framework?utm_source=openai

2. EmergentMind: ReAct Loop Architecture / React-based Agent Architecture  
   价值：给出了状态 $c_t$、策略 $\pi(a_t\mid c_t)$ 这类更正式的机制表达，适合理解为什么这个循环可以被当成一个策略过程。  
   链接：https://www.emergentmind.com/topics/reason-act-reflect-react-architectures?utm_source=openai  
   链接：https://www.emergentmind.com/topics/react-based-agent-architecture?utm_source=openai

3. IBM Think: What is a ReAct Agent?  
   价值：更接近企业落地视角，适合把方法和客服、检索、工作流编排联系起来看。  
   链接：可在 IBM Think 站点检索对应主题文章

4. Arun Baby: ReAct Pattern Deep Dive  
   价值：对无限循环、工具失败、上下文管理这些工程坑讲得更直接，适合做实现前的风险检查。  
   链接：https://arunbaby.com/ai-agents/0014-react-pattern-deep-dive/?utm_source=openai
