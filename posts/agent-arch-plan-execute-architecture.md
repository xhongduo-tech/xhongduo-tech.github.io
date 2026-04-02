## 核心结论

Plan-and-Execute 架构的核心不是“让模型多想几步”，而是把“想什么”和“做什么”拆成两个职责明确的层：Planner 负责规划，也就是先给出高层步骤；Executor 负责执行，也就是按步骤调用工具、收集结果、返回观察。这样做的价值在于，复杂任务不再依赖一次性生成完整答案，而是变成一个可检查、可暂停、可重规划的闭环。

它成立的原因很直接。很多真实任务不是一句话能完成，而是由多个依赖步骤组成：先拿信息，再判断，再执行，再验证。如果把这些动作都压进单轮推理，大模型很容易在中途漏步骤、误判前提、忘记回滚条件，或者把尚未验证的中间结果当成事实继续往下推。Plan-and-Execute 通过显式计划，把“接下来为什么要做这一步”暴露出来；通过执行反馈，把“这一步到底有没有成功”变成可计算状态，而不是模型的主观感觉。

一个新手能立刻理解的玩具例子是“提交日报”。目标不是直接输出一段日报，而是先写出四步计划：收集数据、分析指标、写总结、发布。Executor 执行第一步时去读昨天的工单与提交记录，执行第二步时发现数据缺失，Planner 就不会盲目继续写总结，而是插入“补数据”这一步，再回到分析。整个过程像一组复选框：每完成一步，就依据真实结果决定下一步，而不是假设前面都顺利完成。

可以把它压缩成一个最小闭环：

`Plan Step -> Executor -> Observation -> Planner Decision`

这里的 `Observation` 是执行后的客观观察，白话说就是“机器实际看到了什么结果”。`Planner Decision` 则表示 Planner 根据观察决定继续、跳过、插入新步骤，或者整体重规划。只要这个闭环存在，计划就不是静态清单，而是一个能根据现实修正的控制系统。

---

## 问题定义与边界

Plan-and-Execute 要解决的问题，是单轮推理在长期、多依赖任务中的不稳定性。所谓单轮推理，就是模型一次性根据目标生成完整方案或最终输出。它在短问答、轻写作、简单分类上通常够用，但一旦任务需要跨多个步骤、多个工具、多个中间状态，它就会暴露三个问题：容易漏掉关键依赖、难以追踪错误来源、很难让人类审查执行路径。

“上线新功能”就是典型场景。这个目标看起来很明确，但实际至少包含写代码、跑测试、部署到测试环境、验证结果、准备回滚方案、再上线生产。如果让模型一次性决定所有动作，它可能直接跳到“部署”，也可能忽略回滚策略，甚至把测试通过当成默认前提。问题不在于模型完全不会，而在于没有强制结构来约束它按依赖关系做事。

Plan-and-Execute 的边界也很明确。它适合满足下面三个条件的任务：

| 条件 | 白话解释 | 例子 |
|---|---|---|
| 目标明确 | 知道最后要交付什么 | 提交日报、发布版本、生成周报 |
| 可分解成动作 | 能拆成多个可执行步骤 | 查询数据、调用接口、运行测试 |
| 需要可控路径 | 人希望审查“怎么做到的” | 运维、财务、审批、医疗辅助 |

如果任务本身很模糊，比如“想点有创意的产品方向”，或者根本不需要工具调用，只是做一次简短解释，那么引入 Planner 和 Executor 可能反而增加成本。因为计划本身也要消耗推理资源，而且还要维护状态、处理失败、决定是否重规划。

下面这张表可以直接看出它和单轮推理的差异：

| 维度 | 单轮推理 | Plan-and-Execute |
|---|---|---|
| 鲁棒性 | 中间出错后常继续胡推 | 可在每步后纠偏 |
| 可审查性 | 主要看最终答案 | 可检查完整计划与每步结果 |
| 工具调用 | 常混在文本里，约束弱 | 步骤级绑定工具调用 |
| 长任务一致性 | 易丢上下文或跳步 | 通过状态与计划维持一致性 |
| 实现成本 | 低 | 较高，需要状态机与反馈回路 |

所以它不是“更高级的默认架构”，而是“面向长任务控制”的架构。当你需要审查路径、管理依赖、吸收执行反馈时，它有明显价值；当任务足够短、足够确定时，单轮推理往往更便宜。

---

## 核心机制与推导

从形式化角度看，一个计划可以表示为有向图

$$
G=(V,E)
$$

其中节点集合 $V$ 表示动作，白话说就是“要做的具体步骤”；边集合 $E$ 表示依赖关系，白话说就是“哪一步必须在另一部之前完成”。如果把“提交日报”建模为图，那么“分析指标”依赖“收集数据”，“发布”依赖“写总结”。

在最简单情形下，计划也可以写成线性序列：

$$
\pi=[t_1,t_2,t_3,t_4]
$$

例如：

$$
\pi=[\text{收集数据},\text{分析指标},\text{写总结},\text{发布}]
$$

但真实系统里，计划通常不只是线性清单，因为有些步骤可能并行，有些步骤可能在失败后分叉。例如“部署 staging”和“准备监控面板”可以并行，但“推生产”必须依赖“回归通过”。

运行时的关键不是计划本身，而是状态更新。假设第 $k$ 步执行后得到观察 $o_k$，系统状态从 $s_k$ 更新为：

$$
s_{k+1}=f(s_k,o_k)
$$

这里的 $s_k$ 是当前世界状态，白话说就是系统现在掌握的事实集合；$o_k$ 是执行结果，比如“接口返回 500”“测试通过”“缺少字段”；$f$ 是状态更新函数，也就是把观察写回系统上下文的规则。这个式子的含义是：后续决策不能只看原计划，还必须看刚刚执行后真实发生了什么。

“日报”例子可以写成一个最小重规划过程。初始计划为：

$$
\pi=[t_1,t_2,t_3,t_4]
$$

其中：

- $t_1=\text{收集数据}$
- $t_2=\text{分析指标}$
- $t_3=\text{写总结}$
- $t_4=\text{发布}$

如果执行 $t_2$ 后观察到“数据异常”，那么 Planner 不应该继续执行 $t_3$，而应该生成新计划：

$$
\pi'=[\text{补数据},t_2,t_3,t_4]
$$

这一步很关键，因为它说明 Executor 没有自主改目标。Executor 只对当前计划项负责，发现异常后把观察返回；真正决定是否插入“补数据”的，是 Planner。这样能防止执行层边做边猜，最终偏离原始目标。

可以用简化伪代码描述“偏差检测 -> 重规划”的逻辑：

```text
state = init_state(goal)
plan = planner.plan(state)

while not goal_finished(state):
    step = plan.next_open_step()

    observation = executor.run(step, state)
    state = update_state(state, observation)

    if deviation_detected(step, observation, state):
        plan = planner.replan(state, current_plan=plan)
    else:
        plan = planner.confirm_or_continue(state, current_plan=plan)
```

这里的 `deviation_detected` 是偏差检测，白话说就是判断“结果是否偏离了该步骤的预期后条件”。例如预期是“拿到完整数据表”，结果却是“缺三天数据”，那就是偏差；预期是“回归测试全部通过”，结果却有 2 个关键用例失败，也属于偏差。

真正有效的 Plan-and-Execute 系统，通常都要明确这三件事：

| 组件 | 负责内容 | 如果缺失会怎样 |
|---|---|---|
| Planner | 生成步骤、决定是否重规划 | 系统只会机械执行，遇错僵住 |
| Executor | 调工具、返回客观结果 | 计划无法落地 |
| State/Validator | 记录状态、核对后条件 | 模型会把“想象成功”当“真实成功” |

这也是它比“多轮聊天”更严格的地方。多轮聊天只是消息在来回传递；Plan-and-Execute 则要求每一轮都围绕计划项、观察结果和状态更新来组织，因此更像一个显式控制回路，而不是自由对话。

---

## 代码实现

代码实现的关键只有两点。第一，Planner 和 Executor 的接口必须分离，不能让一个对象既负责想计划又负责直接调用工具，否则失败时你很难知道是“计划错了”还是“执行错了”。第二，每一步执行都要把结构化状态传回 Planner，至少包含当前步骤、是否成功、工具输出、错误原因和剩余计划。

下面是一个可运行的 Python 玩具实现。它没有接入真实大模型，只用规则函数模拟 Planner，但已经包含 Plan、Execute、Observation、Replan 的完整闭环。

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class StepResult:
    step: str
    success: bool
    observation: str


@dataclass
class State:
    goal: str
    data_ready: bool = False
    analysis_ready: bool = False
    summary_ready: bool = False
    published: bool = False
    history: List[StepResult] = field(default_factory=list)


class Planner:
    def plan(self, state: State) -> List[str]:
        plan = []
        if not state.data_ready:
            plan.append("collect_data")
        if state.data_ready and not state.analysis_ready:
            plan.append("analyze")
        if state.analysis_ready and not state.summary_ready:
            plan.append("write_summary")
        if state.summary_ready and not state.published:
            plan.append("publish")
        return plan

    def replan(self, state: State, failed_step: str, observation: str) -> List[str]:
        if failed_step == "analyze" and "missing data" in observation:
            return ["backfill_data", "analyze"] + self.plan(state)
        return self.plan(state)


class Executor:
    def execute_step(self, step: str, state: State) -> StepResult:
        if step == "collect_data":
            state.data_ready = True
            return StepResult(step, True, "raw data collected")

        if step == "backfill_data":
            state.data_ready = True
            return StepResult(step, True, "missing data repaired")

        if step == "analyze":
            if not state.data_ready:
                return StepResult(step, False, "missing data")
            # 模拟第一次分析发现原始数据不完整
            analyzed_before = any(item.step == "analyze" for item in state.history)
            if not analyzed_before:
                return StepResult(step, False, "missing data: 3 records absent")
            state.analysis_ready = True
            return StepResult(step, True, "analysis complete")

        if step == "write_summary":
            if not state.analysis_ready:
                return StepResult(step, False, "analysis not ready")
            state.summary_ready = True
            return StepResult(step, True, "summary drafted")

        if step == "publish":
            if not state.summary_ready:
                return StepResult(step, False, "summary not ready")
            state.published = True
            return StepResult(step, True, "report published")

        return StepResult(step, False, f"unknown step: {step}")


def run_agent(goal: str) -> State:
    state = State(goal=goal)
    planner = Planner()
    executor = Executor()

    plan = planner.plan(state)
    while plan:
        step = plan.pop(0)
        result = executor.execute_step(step, state)
        state.history.append(result)

        if result.success:
            plan = planner.plan(state)
        else:
            plan = planner.replan(state, step, result.observation)

        if state.published:
            break

    return state


state = run_agent("submit daily report")

assert state.data_ready is True
assert state.analysis_ready is True
assert state.summary_ready is True
assert state.published is True
assert any(item.step == "backfill_data" for item in state.history)
assert state.history[-1].step == "publish"
```

这个例子展示了两个核心点。

第一，Planner 没有碰执行细节。它只根据 `State` 决定计划，或者在失败时重规划。第二，Executor 不替 Planner 做策略判断。它只负责尝试执行当前步骤，并返回 `StepResult`。如果 `analyze` 失败，Executor 不会擅自跳去 `write_summary`，也不会自己决定“那我先补数据”，而是把失败原因返回给 Planner，让 Planner 生成新计划。

如果换成真实工程例子，可以把这个骨架套到 DevOps 部署助手上。Planner 先输出：验证代码、部署 staging、运行回归、推 production。Executor 每一步调用真实工具，例如 CI、测试平台、部署系统和监控接口。若“运行回归”失败，观察结果会写入状态，例如“支付链路 2 个关键用例失败”。Planner 再根据策略插入“回滚 staging”“创建修复任务”“重新触发回归”，而不是继续执行推生产。这个设计直接把“不能盲推生产”编码进流程结构，而不是寄希望于模型每次都自觉谨慎。

---

## 工程权衡与常见坑

Plan-and-Execute 提高了可控性，但代价是系统复杂度明显增加。你不再只是在调用一个模型，而是在维护计划、状态、执行结果、验证规则和重规划策略。真正踩坑的地方通常不在“能不能生成计划”，而在“计划与执行之间是否绑定得足够紧”。

最常见的问题是 Plan drift，也就是计划漂移。白话说，就是执行器做着做着偏离了原计划。典型原因包括：Executor 自作主张跳步骤、模型把类似步骤混为一谈、状态更新不完整导致 Planner 误判已经完成。漂移一旦发生，系统表面上还在运行，实际上已经在错误路径上继续积累动作。

第二类问题是工具结果幻觉。所谓幻觉，这里不是指编造常识，而是模型把工具输出解读错了，或者把“部分成功”误当成“完全成功”。例如接口返回了 206 部分内容，系统却把它当成“数据已完整下载”；或者测试报告只显示 smoke test 通过，系统却写成“所有回归通过”。

第三类问题是计划中毒，也就是 Planner 生成了危险或错误计划，后续执行器严格照做。因为 Plan-and-Execute 的执行器通常更服从计划，所以一旦 Planner 早期出错，错误会更稳定地传播下去。比如部署助手一开始就输出“直接推 prod 再做验证”，如果没有审查机制，整个流程会高效地做错事。

下面这张表是工程上最有用的最小检查表：

| 问题类型 | 触发条件 | 缓解措施 |
|---|---|---|
| Plan drift | 执行器跳过计划项，或步骤与工具调用未绑定 | 每次行动必须附带 `current_step_id`，执行后校验是否满足该步后条件 |
| Hallucination | 模型主观总结工具结果，未做结构化解析 | 工具输出先转结构化字段，再由规则校验成功/失败 |
| Poisoning | Planner 生成危险步骤，执行器无条件照做 | 引入 plan review，关键动作需策略检查或人工批准 |
| Infinite replanning | 每次失败都重规划，但永不收敛 | 设置最大重规划次数和失败升级策略 |
| Context bloat | 历史状态越积越大，Planner 被噪声干扰 | 使用摘要状态，只保留决策必要字段 |

真实工程里的 DevOps 助手很能说明这一点。假设 Planner 产出计划：验证、部署 staging、回归、推 prod。Executor 执行到“回归”时收到结果：支付服务回归失败。如果系统没有严格后条件校验，模型可能把“90% 用例通过”解释成“基本通过”，于是继续推生产。这不是模型不会看报告，而是系统没有把“是否允许进入下一步”的判断显式结构化。正确做法是把关键后条件写成规则，例如“关键业务链路必须全部通过，否则禁止进入生产步骤”。

另一个常见坑是把 Planner 设计得过细。比如让它生成几十个极小步骤，每个步骤都要调用模型重新判断。这样理论上更可控，实际上成本很高，而且状态噪声会迅速膨胀。工程上更稳妥的做法通常是让 Planner 负责中粒度任务，例如“收集数据”“运行回归”“部署 staging”，而把步骤内部的机械细节交给确定性程序处理。

---

## 替代方案与适用边界

Plan-and-Execute 不是唯一方案，也不是所有任务都该上它。是否采用，核心看任务长度、不确定性来源，以及你是否真的需要“计划可审查”。

第一种替代方案是单轮大模型决策。它适合短任务、实时应答、低风险生成，例如写一段解释、总结一篇文档、从固定模板中补全文本。它的优点是快、便宜、实现简单；缺点是长期任务中可审查性弱，中间出错后没有显式纠偏点。

第二种替代方案是反复提示控制，也就是每轮都让模型继续思考，但不维护一个显式计划结构。这比单轮稍强，因为它至少能多轮修正；但它的问题在于“每一轮到底围绕哪个计划项在推进”并不总是清楚，因此更像松散对话，而不是可验证流程。

第三种是层级强化学习。强化学习的白话解释是：系统通过反复试错学习策略，而不是只靠提示词。层级强化学习会把高层目标和低层动作分层处理，所以概念上和 Plan-and-Execute 有相似之处。但它更适合未知环境中的长期探索，例如机器人控制、游戏代理、复杂资源调度。对于很多企业工作流，环境并不需要持续探索，反而更需要清晰、可审查、易接规则引擎的步骤计划，这时 Plan-and-Execute 更直接。

可以用表格看边界：

| 方案 | 适用条件 | 优点 | 缺点 |
|---|---|---|---|
| 单轮大模型决策 | 任务短、风险低、不依赖工具链 | 便宜、快、实现简单 | 长任务易漏步，难审查 |
| 反复提示控制 | 有一定迭代需求，但流程不严格 | 比单轮更灵活 | 状态与步骤边界模糊 |
| Plan-and-Execute | 目标明确、工具丰富、需要可控路径 | 可审查、可重规划、适合长任务 | 设计与维护成本高 |
| 层级强化学习 | 环境未知、需要长期探索和策略学习 | 能学习复杂策略 | 训练成本高，系统复杂 |

因此，合理的工程策略通常不是“一上来就做 Plan-and-Execute”，而是逐级升级：

1. 任务短而稳定，先用单轮模型。
2. 任务开始跨多轮，但仍不需要严格审查，用反复提示控制。
3. 任务依赖工具、步骤多、风险高，再上 Plan-and-Execute。
4. 如果任务属于持续探索环境，再考虑层级强化学习。

换句话说，Plan-and-Execute 适合“我知道目标，也知道动作类型，但不确定每一步执行后会发生什么，因此需要边做边校正”的任务。它不适合“目标本身还没定义清楚”的问题，因为那时最大的难点不是执行，而是目标形成。若目标模糊，先做澄清或探索，再引入计划执行，会更符合成本收益比。

---

## 参考资料

- Ronnie Huss, *Plan-and-execute AI agents*
- Meta Intelligence, *Plan-and-Execute: Layered Planning Architecture*
- EmergentMind, *Planned Execution Agent Architecture*
- C# Corner, *Autonomous AI in the Real World*
- Hugging Face 安全指南, *Agent Architecture Patterns*
