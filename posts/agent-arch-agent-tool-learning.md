## 核心结论

“工具学习”不是给大模型多接几个 API，而是让智能体在完整任务链路里持续做三层判断：是否调用工具、调用哪个工具、如何调用工具。这里的“工具”可以先理解成外部能力接口，比如搜索、天气、数据库查询、代码执行器。大模型擅长语言归纳，但不天然擅长访问最新信息、执行精确计算、操作真实系统，所以需要把外部工具纳入决策闭环。

对真实系统来说，最稳的做法通常不是“一个模型既推理又直接调工具”，而是把职责拆开：控制器负责分解任务和推进计划，感知器读取环境状态，检索器缩小可用工具集合，执行者实际调用工具，验证器检查结果是否可接受。这样形成“感知 - 计划 - 调用 - 反馈”的环形循环。它成立的原因很简单：任务不是一次生成文本，而是在不完整观察下逐步逼近目标。

新手最容易误解的一点是，工具学习的重点不是“会不会调 API”，而是“能不能在成本、错误率、上下文长度、环境噪声约束下稳定完成任务”。因此，工具学习本质上是一个受约束的序列决策问题，而不是单次文本生成问题。

---

## 问题定义与边界

先把问题边界说清楚。一个带工具的 Agent，可以形式化成带部分可观测的决策过程。所谓“部分可观测”，白话说就是模型每一步只能看到局部信息，看不到完整真实世界。

设时刻 $t$ 的隐藏状态为 $s_t$，观测为 $o_t$，记忆为 $m_t$，可选工具集合为 $\mathcal{T}$，动作 $a_t$ 可以是两类：

$$
a_t \in \{\text{text response}, \text{tool invocation}\}
$$

如果是工具调用，还要进一步包含：

$$
a_t = (\text{whether}, \text{which}, \text{how})
$$

其中：

- `whether`：这一步要不要调用工具
- `which`：从候选工具里选哪个
- `how`：传什么参数、按什么顺序调用、是否重试

记忆更新可以写成：

$$
m_{t+1} = f(m_t, o_t, a_t, r_t)
$$

这里的“记忆”就是系统保留的中间结论、摘要、历史调用日志。它的作用不是模仿人脑，而是避免每一轮都从零开始推理。

把五个模块放进这个框架里，边界就明确了：

| 模块 | 输入 | 输出 | 主要职责 |
|---|---|---|---|
| 感知器 Perceiver | 页面、文本、接口回包、环境状态 | 结构化观察 `obs` | 把原始输入整理成可推理信息 |
| 控制器 Controller | 用户目标、记忆、验证结果 | 下一步计划 | 决定当前子任务与推进顺序 |
| 检索器 Retriever | 当前子任务、工具描述 | 候选工具列表 | 缩小工具搜索空间 |
| 执行者 Executor | 工具名、参数 | 工具返回值 | 真实发起调用并记录日志 |
| 验证器 Validator | 返回值、目标约束 | 通过/失败/修正建议 | 检查格式、完整性、一致性 |

这五个模块不是为了“架构好看”，而是为了把错误定位在具体环节。比如天气 API 返回 401，这是执行问题；返回字段缺失，这是验证问题；本来该先查城市再查天气却直接调用天气接口，这是控制问题；选错了汇率 API，这是检索问题。

一个玩具例子能把边界看得更清楚。用户说：“帮我查明天北京会不会下雨。”  
流程不是一句话生成完，而是：

1. 控制器把任务定成“获取城市天气预报”。
2. 感知器检查上下文里是否已有“北京”和“明天”。
3. 检索器从工具池里选天气 API。
4. 执行者调用天气 API。
5. 验证器确认返回里有日期、降水概率、单位。
6. 控制器决定是否直接回答，还是需要再补充温度或出行建议。

这说明工具学习的边界在“任务完成链路”，不在“模型文本输出”本身。

---

## 核心机制与推导

工具学习为什么要分层？因为推理和调用工具的信号性质不同。推理需要稳定、抽象、低噪声的上下文；工具调用会带来大量临时结果、格式差异和失败回包。如果让一个模型同时承担两者，推理链容易被噪声污染。

一种代表性做法是 `Planner + Toolcaller` 分层。Planner 负责多步结构化推理，只输出类似 `<think>` 和 `<tool calling>` 的控制信号。Toolcaller 负责把调用真正落地，并把结果包装成 `<obs>` 返回。白话说，Planner 像总调度，Toolcaller 像执行工位。

这种设计的核心收益有两个：

1. Planner 的上下文更干净，因为它不必直接吸收所有工具原始返回。
2. 工具调用行为可以单独优化，比如做参数校验、失败重试、成本统计。

如果再往前走到训练层，目标就不是“让模型写得更像人”，而是“让 Planner 更会选工具链”。这时会用到类似 GRPO 的强化学习目标。一个典型形式是：

$$
J(\Theta)=\mathbb{E}_{x\sim\mathcal{D},y_i\sim\pi_{old}}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(r_iA_i,\operatorname{clip}(r_i,1-\epsilon,1+\epsilon)A_i\right)-\beta D_{KL}(\pi_\Theta\|\pi_{ref})\right]
$$

其中：

$$
r_i=\frac{\pi_\Theta(y_i|x)}{\pi_{old}(y_i|x)}
$$

这个式子可以直接按工程直觉理解：

- $A_i$ 是优势，表示这条轨迹比平均水平好多少。
- `clip` 限制每次更新别跳太大，避免策略发散。
- $D_{KL}$ 正则约束新策略不要离参考策略太远，避免训练后工具调用风格崩掉。

为什么要对 `<obs>` 做 masking，也就是训练时弱化或屏蔽部分工具观测？原因是奖励应该尽量落在“推理决策”而不是“背诵回包格式”。否则模型容易学到一种伪能力：只要看见某种回包模板就复制，而不是学会在何时、为何调用工具。

用“找票 + 查天气”的玩具例子看这套机制更直观。用户说：“下周去上海出差，帮我找便宜机票并看当天是否下雨。”  
合理链路应当是：

1. 控制器拆成两个子任务：订票信息、天气信息。
2. 检索器先选机票搜索工具，再选天气工具。
3. 执行者调用机票 API 得到候选航班。
4. 验证器检查日期、价格、起降地是否齐全。
5. 控制器把最优航班日期传给天气任务。
6. 执行者调用天气 API。
7. 验证器确认天气结果对应的是同一天、同一城市。
8. 控制器合成最终答复。

这里“工具学习”学的不是单个 API 文档，而是跨工具传递约束。例如天气查询里的日期，不是用户原话里的“下周”，而是机票结果里选中的具体出发日期。这就是工具链推导。

真实工程例子比玩具例子更复杂。比如多跳问答系统要回答：“某人物毕业学校所在城市的人口规模是多少？”  
它通常需要：

1. 搜索人物信息。
2. 从搜索结果抽取毕业学校。
3. 搜索学校所在城市。
4. 再查该城市人口。
5. 验证每一步实体是否一致。

这个过程中，任何一步实体漂移都会把后续答案带偏。因此，分层架构能把“实体跟踪”和“工具调用”拆开处理，减少错误传播。

---

## 代码实现

工程里最先要做的不是训练，而是把执行面做稳。没有稳定执行器，再好的 Planner 也会被脏回包和随机失败拖垮。一个最小可运行实现如下：

```python
from dataclasses import dataclass, field

@dataclass
class SafeToolExecutor:
    tools: dict
    max_retries: int = 3
    total_cost: float = 0.0
    call_log: list = field(default_factory=list)

    def execute(self, tool_name, params):
        if tool_name not in self.tools:
            return {"ok": False, "error": f"tool_not_found:{tool_name}"}

        spec = self.tools[tool_name]
        missing = [p for p in spec["required_params"] if p not in params]
        if missing:
            return {"ok": False, "error": f"missing_params:{missing}"}

        for attempt in range(1, self.max_retries + 1):
            try:
                result = spec["function"](**params)
                self.total_cost += spec.get("cost", 0.0)
                record = {
                    "tool": tool_name,
                    "params": params,
                    "attempt": attempt,
                    "status": "success",
                    "cost": spec.get("cost", 0.0),
                }
                self.call_log.append(record)
                return {"ok": True, "result": result, "attempt": attempt}
            except Exception as e:
                record = {
                    "tool": tool_name,
                    "params": params,
                    "attempt": attempt,
                    "status": "failed",
                    "error": str(e),
                    "cost": spec.get("cost", 0.0),
                }
                self.call_log.append(record)
                if attempt == self.max_retries:
                    return {"ok": False, "error": str(e), "attempt": attempt}

def fake_weather(city, date):
    if city != "Shanghai":
        raise ValueError("unsupported_city")
    return {"city": city, "date": date, "rain_prob": 0.7}

def fake_flight(origin, dest, date):
    return [
        {"flight_no": "MU123", "price": 520, "date": date, "origin": origin, "dest": dest},
        {"flight_no": "CA987", "price": 680, "date": date, "origin": origin, "dest": dest},
    ]

tools = {
    "weather_api": {
        "function": fake_weather,
        "required_params": ["city", "date"],
        "cost": 0.02,
    },
    "flight_api": {
        "function": fake_flight,
        "required_params": ["origin", "dest", "date"],
        "cost": 0.15,
    },
}

executor = SafeToolExecutor(tools=tools)

flight_ret = executor.execute("flight_api", {"origin": "Beijing", "dest": "Shanghai", "date": "2026-04-10"})
assert flight_ret["ok"] is True
best = min(flight_ret["result"], key=lambda x: x["price"])
assert best["price"] == 520

weather_ret = executor.execute("weather_api", {"city": "Shanghai", "date": best["date"]})
assert weather_ret["ok"] is True
assert weather_ret["result"]["rain_prob"] == 0.7

bad_ret = executor.execute("weather_api", {"city": "Hangzhou"})
assert bad_ret["ok"] is False
assert "missing_params" in bad_ret["error"]

assert round(executor.total_cost, 2) == 0.17
assert len(executor.call_log) == 3
```

这段代码有几个关键点：

- 工具注册表显式声明必填参数和调用成本。
- 执行器统一返回 `ok/result/error/attempt`，让 Planner 不用解析杂乱异常文本。
- 每次调用都记日志，后续才能做训练样本、失败分析和成本控制。
- `assert` 不只是测试语法正确，它在这里承担“协议正确性”的作用。

如果往上接 Planner，一轮运行通常是：

| 阶段 | 输入 | 输出 |
|---|---|---|
| Planner | 用户目标、记忆摘要 | `<think>` 与 `<tool calling>` |
| Toolcaller | 工具名、参数 | 调用执行器 |
| Executor | 工具规范、参数 | 统一格式结果 |
| Validator | 执行结果、目标约束 | `pass/fail/fix` |
| Planner | 结果摘要、验证反馈 | 下一轮决策 |

一个简化的训练循环可以写成概念流程：

1. 从样本集中取一个任务。
2. Planner 对同一任务采样多条 rollout，例如 12 条。
3. 每条 rollout 在执行工具时统一走 `SafeToolExecutor`。
4. 根据是否完成目标、是否超预算、是否参数错误计算 reward。
5. 用 reward 形成优势估计，更新 Planner。
6. 若工具调用成功率低，优先修工具协议和验证器，不要急着加训练轮数。

这里的工程常识很重要：如果执行层不稳定，强化学习会把系统噪声当成策略信号，最后学到“保守不调用”或者“重复乱调用”两种坏策略。

---

## 工程权衡与常见坑

工具学习落地时，最大的约束通常不是“模型够不够聪明”，而是预算、稳定性和上下文。尤其在生产环境，调用贵工具的代价会迅速累积，所以常把工具选择看成一个背包问题。所谓“背包问题”，白话说就是在有限预算内挑一组收益最大的动作。

例如预算为 10，工具 A 成本 3、价值 50，工具 B 成本 8、价值 90，那么最优解是选 B，而不是“都试一下”。形式化地说：

$$
\max \sum_i v_i x_i \quad \text{s.t.} \quad \sum_i c_i x_i \le B,\; x_i \in \{0,1\}
$$

这反映的是现实系统中的成本约束：高质量搜索、长上下文模型、代码沙箱、数据库聚合都可能是昂贵工具，不能无限重试。

常见失败模式可以直接列成检查表：

| 失败模式 | 典型表现 | 根因 | 修复措施 |
|---|---|---|---|
| reasoning error | 任务拆分顺序错 | 计划能力不足 | 增加中间检查点，强制子目标显式化 |
| tool use error | 参数缺失、格式错 | 工具协议不明确 | 用 schema 校验，返回统一错误码 |
| context overflow | 历史太长被截断 | 重复写入原始回包 | 摘要压缩，长期记忆与短期上下文分层 |
| loop | 重复调用同一工具 | 缺少停止条件 | 设最大步数和重复调用惩罚 |
| goal drift | 答着答着偏题 | 中间目标覆盖主目标 | 每轮重申 task id 与 success condition |
| prompt injection | 工具返回中夹带恶意指令 | 未做信任边界隔离 | 将工具结果视为数据而非指令 |
| hallucination | 工具没返回也编结果 | 结果未绑定证据 | 强制答案引用对应 observation |
| cost runaway | 成本快速失控 | 缺少预算感知 | 预算计数器 + 动态工具剪枝 |

真实工程例子可以看“汇总门店销量”。用户要的是“本周华东区所有门店销量总和，并给出环比变化”。一个不成熟 Agent 常见失败路径是：

1. 先用搜索工具找“华东区门店列表”。
2. 对每个门店逐个调用报表接口。
3. 每次都把完整 JSON 回包塞进上下文。
4. 最后上下文溢出，或者重复汇总。

更稳的做法是：

1. 控制器先确定主目标是“总销量 + 环比”。
2. 检索器优先选内部报表 API，而不是通用 web search。
3. 执行者分页查询，每页结果只抽取门店 ID、销量、周期。
4. 感知器把原始回包转成紧凑表结构。
5. 验证器检查是否覆盖所有门店、周期是否一致。
6. 若上下文接近上限，触发 `summary + memory` 管线，只保留统计摘要和可追溯引用。

这里的关键权衡是：保留可追溯性会增加 token，过度压缩会损失验证能力。工程上通常保留“摘要 + 原始日志索引”，而不是把全部原始数据一直塞在 prompt 里。

---

## 替代方案与适用边界

不是所有场景都需要分层 Agent 和强化学习。如果工具池很小、调用顺序固定、错误代价低，更简单的方法更合适。

先看几类常见替代方案：

| 方案 | 核心思路 | 适合场景 | 边界 |
|---|---|---|---|
| 词汇扩展式工具调用 | 给模型加入专用 `tool token` | 工具少、协议稳定 | 难处理多步规划和复杂约束 |
| 多标签分类式选工具 | 先分类选工具，再单步调用 | FAQ、固定流程表单 | 无法处理多跳决策 |
| 编译式方案 | 把任务编译成程序或执行图 | 可验证、确定性强的任务 | 对开放环境和噪声输入不稳 |
| 成本感知搜索 | 用 A* 或树搜索提前剪枝 | 工具成本高、路径多 | 实现复杂，状态设计要求高 |
| 分层 RL | Planner 学策略，执行器单独落地 | 多步、噪声大、需长期优化 | 数据和训练成本高 |

一个适合新手理解的判断规则是：

- 如果任务像“查快递、查天气、算税率”这种固定输入到固定工具的映射，优先用轻量工具路由。
- 如果任务像“多跳问答、跨系统操作、开放环境网页代理”，优先考虑分层 Agent。
- 如果每一步都可编译成稳定程序，编译式方案通常比自由推理更稳。
- 如果单次调用费用很高，必须先做预算感知和工具剪枝，再谈更复杂的训练。

再看一个具体对比。  
网页信息固定、流程简单的客服机器人，如果只需要“根据意图调用退款接口或查询物流”，用多标签分类就够了。因为任务结构短，策略空间小，没必要引入复杂 RL。  
但在多跳 QA 或开放网页环境中，工具输出噪声大、每一步观察都不完整，这时分层 `Planner + Toolcaller` 更稳，因为它能把“思考”和“动手”拆开。

所以，Agent 的工具学习不是通用银弹。它适用于任务链长、工具异构、外部环境不确定、且系统愿意承担更高设计复杂度的场景；不适用于工具极少、路径固定、成本敏感到不允许多轮探索的场景。

---

## 参考资料

- Weikai Xu 等，《LLM-Based Agents for Tool Learning: A Survey》：系统总结工具学习中的 `whether / which / how` 三层决策，以及执行者、感知器、验证器、控制器、检索器五组件闭环。  
  https://link.springer.com/article/10.1007/s41019-025-00296-9

- Agent-as-Tool hierarchical RL 综述：介绍 Planner 与 Toolcaller 分离、对 `<obs>` 做 masking、以及基于 GRPO 的训练目标和实验结果。  
  https://www.themoonlight.io/en/review/agent-as-tool-a-study-on-the-hierarchical-decision-making-with-reinforcement-learning

- Xiaofang Yang 等，《Toward Efficient Agents: Memory, Tool learning, and Planning》：从效率视角讨论工具选择、成本约束、背包式预算优化和搜索策略。  
  https://www.researchgate.net/publication/399953342_Toward_Efficient_Agents_Memory_Tool_learning_and_Planning

- AgentWiki，《Common Agent Failure Modes》：整理推理链错误、工具协议错误、上下文溢出、循环调用、目标漂移、提示注入、幻觉和成本失控等典型失效模式。  
  https://agentwiki.org/common_agent_failure_modes
