## 核心结论

Agent 的执行监控，本质上不是“多记几条日志”，而是把智能体每一步为什么这样做、做了什么、后果如何、何时应该被拦下，变成可重构、可判断、可干预的运行闭环。

对初学者先下定义：

- 执行监控：指在 agent 运行过程中持续采集状态，并据此判断是否偏离预期。
- 观测：指记录事实，回答“发生了什么”。
- 操作控制：指触发动作，回答“现在要不要拦、降级、重试或交给人”。

这两个层次不能混为一谈。只做观测，系统会进入“看见问题但不阻止”的状态；只做控制，没有因果证据，系统会频繁误杀正常流程。真实系统需要的是“观测支撑控制，控制反过来校正执行”。

可以把它想成“智能体操作系统”的仪表盘：计划步骤、工具调用、记忆更新、成本变化、失败重试都要留下结构化记录，并且每条记录要能和策略规则对应。一旦出现偏离，比如重复调用同一工具、成本在短时间内飙升、写入了不该写的记忆，就触发限流、回滚、人工审核或保险箱模式。

下面这张表用于区分“解释行为的数据”和“纠正行为的动作”。

| 维度 | 观测关注什么 | 控制关注什么 | 典型触发策略 | 执行者 |
|---|---|---|---|---|
| 计划执行 | 当前计划、步号、分支选择 | 是否允许继续执行下一步 | 关键步骤偏离预设流程 | 调度器 / 监督 agent |
| 工具调用 | 输入、输出、错误、延迟 | 是否重试、熔断、改走备用工具 | 连续失败、参数异常 | 运行时框架 |
| 成本与 token | 单步 token、累计成本、调用频率 | 是否限流、终止、降级模型 | 单轮或单位任务成本超阈值 | 成本守卫器 |
| 记忆更新 | 新增、修改、删除的内容差异 | 是否允许写入长期记忆 | 低置信度信息写入 | 记忆策略模块 |
| 安全与权限 | 实际动作与授权范围 | 是否强制人工审核 | 越权读写、敏感动作 | 策略引擎 / 人类审核 |

结论可以压缩成一句话：Agent 执行监控的目标不是“把 trace 存下来”，而是基于因果路径持续判断“是否还应继续自治执行”。

---

## 问题定义与边界

问题定义要尽量收紧，否则“监控”会变成什么都想管、最后什么都管不好。

本文讨论的对象是：自治或半自治 agent 在执行任务时的运行监控。它关注的是 agent 内部决策链条是否健康，包括计划、工具、记忆、成本、权限、恢复动作。它不等价于传统服务器监控，也不等价于只看某一次 LLM 请求的日志。

边界先说清楚。

| 在监控 scope 内 | 为什么在 scope 内 | 不在本文 scope 内 | 为什么不在 scope 内 |
|---|---|---|---|
| 计划步骤与状态迁移 | 决定 agent 是否按设计推进 | 机房断电、宿主机硬件损坏 | 属于基础设施运维 |
| LLM 调用输入输出 | 影响后续推理与工具参数 | 第三方系统整体宕机修复 | 属于外部系统治理 |
| 工具调用链与重试 | 错误会沿因果链放大 | 数据库索引设计优化 | 属于业务系统性能工程 |
| 记忆写入与上下文变化 | 会长期污染后续决策 | 公司组织审批流程本身 | 属于组织流程设计 |
| 成本、token、循环次数 | 决定是否失控 | 模型训练质量提升 | 属于模型研发 |

为什么要强调“因果依赖链”？因为 agent 失败通常不是单点事故，而是链式放大。一次工具参数错误，可能先导致结果为空；空结果又写入记忆；错误记忆再被后续步骤当成事实；最后系统给出“看似完整、实则建立在错误上下文上的”答案。如果只看某次 trace 中单个节点是否成功，就会漏掉真正的问题。

玩具例子很简单：一个“查天气并提醒带伞”的 agent，步骤可能是“解析城市 -> 调天气 API -> 解释结果 -> 生成提醒”。如果 API 返回空值，但 agent 把空值解释成“晴天”，从单次模型调用看它没有报错；但从因果链看，错误已经从工具层传到了用户输出层。这就是执行监控的对象。

真实工程例子更典型。金融场景常把任务拆成指令解析 agent、工具调用 agent、监督审计 agent 三层。监控层不只记录“模型回了什么”，还记录“是谁授权的、调用了哪个工具、花了多少成本、是否越过审批边界”。一旦任一 agent 的动作与安全策略不一致，就不能再让系统自由推进，而是必须降级到“人类审核”模式。这说明执行监控的边界是“自治执行链条是否仍然可信”，不是“服务器还活着就算健康”。

---

## 核心机制与推导

执行监控可以抽象成一个四段循环：

$$
\text{感知} \rightarrow \text{残差/路径检测} \rightarrow \text{报警} \rightarrow \text{约束/恢复}
$$

这里几个术语先白话解释：

- 感知：就是把运行中的事实采上来。
- 残差：就是“实际发生”和“预期应该发生”之间的差。
- 路径检测：就是判断 agent 是沿着哪条执行分支走下去的。
- 恢复：就是偏离后如何把系统拉回安全区域。

### 1. 感知层为什么必须结构化

若只记录自然语言日志，后续很难做规则判断。最小可用结构通常包括：

- `session_id`：同一任务的全局编号。
- `step_id`：当前执行到第几步。
- `span_type`：planning、llm、tool、memory、guardrail 等。
- `input/output`：输入输出摘要。
- `latency_ms`：延迟。
- `prompt_tokens/completion_tokens`：token 消耗。
- `cost`：成本估算。
- `error`：错误类型。
- `parent_span_id`：因果父节点。

有了这些字段，才可能问出工程上关键的问题：这次错误来自哪个上游步骤？是模型规划错了，还是工具参数错了？是偶发失败，还是进入循环？

### 2. 为什么单一 trace 不够

很多团队会说：“我们已经有 trace 了。”但 trace 常常默认展示一条实际走过的路径，而不是展示“本可以走的所有关键路径”。Agent 的监控难点在于分支。

路径总数可以写成：

$$
P = \prod_{i=1}^{n} b_i
$$

其中 $b_i$ 表示第 $i$ 步的候选分支数，$P$ 是总路径数。

如果一个 10 步任务中，第 3、6、9 步各有 2 个可选分支，其余步骤只有 1 个分支，那么：

$$
P = 1 \times 1 \times 2 \times 1 \times 1 \times 2 \times 1 \times 1 \times 2 \times 1 = 8
$$

这意味着什么？意味着“某一次成功执行”只覆盖了 8 条路径中的 1 条。你不能因为这 1 条成功，就假设系统整体健康。监控必须认识“路径集合”，否则会把大量潜在失效藏在未观测分支里。

### 3. 残差检测为什么是核心

真正让监控有价值的，不是采集数据，而是检测残差。常见残差包括：

- 计划残差：实际执行步骤和原计划不一致。
- 成本残差：某一步 token 或费用远高于历史分位数。
- 工具残差：工具输出格式与预期 schema 不匹配。
- 记忆残差：低置信度结论写入了长期记忆。
- 时间残差：主循环长期无进展，但进程还活着。

玩具例子可以继续扩展。一个“订会议室” agent，本该执行“查空闲 -> 预定 -> 发确认”。如果它在“查空闲”失败后开始反复重试同一个 API，CPU 和进程都正常，底层监控也显示服务在线，但主循环其实已经失控。这里的关键残差不是机器资源，而是“同一状态无进展地重复”。

### 4. 报警不是终点，恢复才是闭环

报警只回答“出事了”，恢复回答“怎么收场”。恢复动作通常包括：

- 停止当前分支。
- 切换到保守模型或只读工具。
- 清空本轮临时记忆。
- 切换备用工具。
- 进入人工审核。
- 回滚到上一个稳定检查点。

所以完整逻辑不是“观测到异常 -> 发消息”，而是“观测到异常 -> 判断异常等级 -> 执行对应约束”。这也是执行监控和普通日志系统的根本区别。

---

## 代码实现

下面给一个最小可运行的 Python 玩具实现。它不依赖外部框架，但保留了真实系统最重要的结构：span 记录、成本检测、循环检测、恢复动作。

```python
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class Span:
    session_id: str
    step_id: int
    span_type: str
    name: str
    input_summary: str
    output_summary: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    latency_ms: int
    error: Optional[str] = None
    parent_step_id: Optional[int] = None


@dataclass
class MonitorState:
    spans: List[Span] = field(default_factory=list)
    repeated_actions: int = 0
    halted: bool = False
    reason: Optional[str] = None

    def record(self, span: Span) -> None:
        self.spans.append(span)

    def total_cost(self) -> float:
        return round(sum(s.cost for s in self.spans), 4)

    def last_n_same_action(self, n: int) -> bool:
        if len(self.spans) < n:
            return False
        names = [s.name for s in self.spans[-n:]]
        return len(set(names)) == 1


def guardrail_check(state: MonitorState, max_cost: float = 1.0) -> None:
    if state.total_cost() > max_cost:
        state.halted = True
        state.reason = "cost_limit_exceeded"
        return

    if state.last_n_same_action(3):
        state.halted = True
        state.reason = "stuck_in_retry_loop"
        return


def fake_tool_call(action_name: str, step_id: int) -> Span:
    start = time.time()
    # 模拟一个反复失败的工具调用
    output = "timeout"
    latency_ms = int((time.time() - start) * 1000) + 120
    return Span(
        session_id="sess-001",
        step_id=step_id,
        span_type="tool",
        name=action_name,
        input_summary="GET /weather?city=shanghai",
        output_summary=output,
        prompt_tokens=0,
        completion_tokens=0,
        cost=0.42,
        latency_ms=latency_ms,
        error="timeout",
        parent_step_id=step_id - 1 if step_id > 0 else None,
    )


state = MonitorState()

for i in range(1, 4):
    span = fake_tool_call("weather_api_retry", i)
    state.record(span)
    guardrail_check(state, max_cost=2.0)
    if state.halted:
        break

assert state.halted is True
assert state.reason == "stuck_in_retry_loop"
assert len(state.spans) == 3
assert state.total_cost() == 1.26
```

这段代码展示了一个最小原则：监控逻辑必须放在主循环里，而不是只在外层统计日志。否则你只能在任务结束后复盘，无法在失控中止损。

如果进一步贴近真实工程，通常会这样分层：

| 层级 | 负责内容 | 典型实现 |
|---|---|---|
| Span 采集层 | 记录决策、工具、记忆、错误 | OpenTelemetry 风格 tracer 或自定义事件总线 |
| 规则检测层 | 检测成本异常、重复路径、越权动作 | 阈值规则、状态机、简单统计模型 |
| 策略执行层 | 限流、熔断、回滚、人工接管 | guardrail 服务、策略引擎 |
| 查询分析层 | 追踪 session、聚类失败模式 | 日志检索、trace UI、审计报表 |

真实工程例子可以是一个客服退款 agent。它的主循环可能是：

1. 解析用户请求。
2. 查询订单。
3. 判断是否符合退款条件。
4. 调用退款 API。
5. 写入审计记录。
6. 回复用户。

在这里，至少要监控三类关键 span：

- 决策 span：为什么判断该订单可退款。
- 工具 span：调用了哪个订单系统、哪个支付接口。
- 审计 span：是否记录了权限、金额、审批来源。

如果系统检测到“同一订单被重复尝试退款”“金额超过阈值”“模型置信度低但仍试图调用支付接口”，就不能再继续，而要切换成“待人工审核”。这就是观测数据驱动操作逻辑，而不是“事后查日志”。

---

## 工程权衡与常见坑

执行监控的难点不在概念，而在取舍。你不可能记录一切，也不应该对所有异常都立即终止。下面是常见坑和对应对策。

| 常见坑 | 失败模式 | 直接后果 | 对策 |
|---|---|---|---|
| 只看进程心跳 | 进程活着但主循环卡死 | 任务长时间无进展却不报警 | 心跳嵌入主循环内部，按“状态推进”而非“进程存活”判断 |
| 只看 LLM 调用日志 | 工具失败、记忆污染看不见 | 根因定位错误 | 统一记录 planning / tool / memory span |
| 不监控 token 与成本 | 重试环路悄悄烧钱 | 账单失控 | 每轮成本、累计成本、单位成功成本都设阈值 |
| 不记录父子因果关系 | 看到错误但找不到上游原因 | 恢复动作粗暴、误杀正常流 | 强制带 `parent_span_id` 或等价因果链字段 |
| 把报警当成终点 | 发现问题但无法自动止损 | 运营依赖人工盯盘 | 设计降级、熔断、回滚、人工审核闭环 |
| 观测字段全是自由文本 | 后续无法规则化判断 | 只能人工读日志 | 关键字段结构化，文本仅作补充 |
| 把所有异常都一刀切中断 | 正常波动也被误报 | 可用性下降 | 区分 info / warn / critical 三档策略 |

其中两个坑最容易被低估。

第一，僵尸状态。很多系统只做进程级心跳，比如“每 30 秒上报一次 agent 还活着”。这对 Web 服务可能够用，但对 agent 不够。因为 agent 可能卡在外部 HTTP 请求、锁等待、错误重试或者某个永远满足不了的计划条件里。此时进程还活着，CPU 也不一定高，但业务上已经死了。正确做法是让心跳跟着主循环推进，比如“最近 60 秒内 step_id 是否变化”“最近一次状态转移是否完成”。

第二，成本失控。Agent 的很多事故不是模型答错，而是模型一直在“认真地错”，并且每次都消耗更多 token。只监控成功率会漏掉这类问题，因为系统表面上还在工作。应当把 token 和 cost 当成健康指标，而不是财务指标。单位任务成本突然上升，通常意味着上下文膨胀、重试环路、工具反馈异常或计划发散。

实际落地时还有一个权衡：监控越细，开销越大。记录完整输入输出最利于审计，但会增加存储成本，也可能涉及隐私数据。工程上常用折中方案是：

- 默认记录结构化摘要。
- 对高风险步骤记录脱敏原文。
- 对低风险步骤只记录哈希、长度和关键统计值。
- 将全量原文采集限制在抽样或故障重现模式。

---

## 替代方案与适用边界

执行监控不是唯一方案。在更高风险或更复杂的系统里，常常要和其他机制组合。

| 方案 | 核心思想 | 适用场景 | 成本 | 局限 |
|---|---|---|---|---|
| 传统执行监控 | 采集运行状态并触发控制 | 大多数单 agent 或弱协作系统 | 中 | 对输出正确性判断有限 |
| 独立验证 agent / judge | 用独立判定链校验输出或动作 | 多 agent 协作、高风险输出 | 中到高 | 增加延迟与系统复杂度 |
| 结构化协议与 schema | 用 JSON schema、工具合同约束交互 | 需要严格接口稳定性的生产系统 | 中 | 灵活性下降 |
| 人工审核关口 | 高风险节点必须人确认 | 金融、医疗、法务等 | 高 | 吞吐下降 |
| 全局沙箱 / 权限最小化 | 即使判断错也限制损害范围 | 有外部写操作的系统 | 中 | 不能替代正确判断 |

独立验证 agent 的思路很直接：不要让执行者自己给自己打分。比如代理 A 生成一段 API 调用说明，再让独立 judge agent 检查是否符合 schema、是否越权、是否遗漏必填字段。如果 judge 判为不合格，就中止下游 agent，避免错误继续串联。这种方案适合高安全要求场景，但代价是更高延迟和更多调用成本。

结构化协议则是另一个方向。与其依赖自由文本里“希望你遵守规则”，不如直接要求 agent 输出可校验的 JSON，并用 schema 验证。这样一来，很多偏离会在接口层就被发现，而不是进入后续执行链才暴露。对生产系统来说，这通常比“靠 prompt 约束行为”更稳定。

不过这些方案都有边界：

- 若系统主要是内容生成、风险低、无需调用外部写接口，轻量监控就足够。
- 若系统涉及支付、删改数据、跨系统写操作，必须把监控和权限控制结合。
- 若多 agent 协作复杂，单纯监控难以覆盖语义错误，这时 judge 或 schema 更重要。
- 若业务要求极低延迟，过度验证会拖慢响应，需要只在高风险节点启用。

最终可以这样理解：执行监控回答“现在系统是否仍在可控地运行”，独立验证回答“当前结果是否可接受”，结构化协议回答“交互是否满足合同”。三者不是替代关系，而是风险等级不同的组合件。

---

## 参考资料

- ServicesGround，《Agentic AI Observability & Operations》（2025）：https://servicesground.com/blog/agentic-ai-observability-ops/?utm_source=openai
- Latitude，《Detecting AI Agent Failure Modes in Production: A Framework for Observability-Driven Diagnosis》（2026）：https://latitude.so/blog/ai-agent-failure-detection-guide?utm_source=openai
- Deloitte Insights，《Unlocking exponential value with AI agent orchestration》（2025）：https://www.deloitte.com/us/en/insights/industry/technology/technology-media-and-telecom-predictions/2026/ai-agent-orchestration.html?icid=tmt-predictions_click&utm_source=openai
- Augment Code，《Why Multi-Agent LLM Systems Fail (and How to Fix Them)》（2025）：https://www.augmentcode.com/guides/why-multi-agent-llm-systems-fail-and-how-to-fix-them?utm_source=openai
- Reddit / r/LangChain，《Three AI agent failure modes my old monitoring never caught》（2026）：https://www.reddit.com/r/LangChain/comments/1s5j8rn/three_ai_agent_failure_modes_my_old_monitoring/?utm_source=openai
