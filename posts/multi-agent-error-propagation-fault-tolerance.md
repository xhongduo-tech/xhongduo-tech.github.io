## 核心结论

多 Agent 系统的主要风险，不是某个 Agent 偶尔答错，而是这个错误会被当成“上游事实”继续传下去，最后变成系统级失败。这里的“级联传播”指局部错误沿着调用链扩散，影响越来越大的过程。对串行流水线来说，只要前面一环把错误写进上下文，后面的 Agent 即使执行完全正常，也可能稳定地产生错误结果。

最直接的结论可以用 Lusser 法则表示。它原本用于串联系统可靠性估计，这里可以把每个 Agent 看成一个串行组件。若第 $i$ 个 Agent 的单次成功率为 $r_i$，则整体成功率近似为：

$$
R_{\text{sys}}=\prod_i r_i
$$

整体失效率则为：

$$
P(\text{fail}) = 1-\prod_i r_i
$$

玩具例子：3 个 Agent 串行执行，若每个 Agent 单次成功率都是 $0.9$，则整体成功率只有 $0.9^3=0.729$，失效率约为 $27.1\%$。这说明即使每一环“看起来还行”，串起来之后也会明显变脆。

因此，多 Agent 容错不能只靠“失败了再试一次”。更有效的方案是三级防线：

| 层级 | 目标 | 核心动作 |
|---|---|---|
| Agent 级 | 截断局部幻觉 | 输出校验、有限重试、自我纠错 |
| 流水线级 | 阻止错误继续扩散 | checkpoint、阶段验真、失败回滚 |
| 系统级 | 控制爆炸半径 | 断路器、降级路径、人工接管 |

新手版理解可以用“传水桶”来记。三名搬砖工人依次传水桶，第一人漏水，相当于上游 Agent 幻觉；第二、第三人只是机械搬运，相当于下游 Agent 按错误输入继续决策；最后水桶虽然被送到了终点，但已经空了，相当于系统完成了流程，却失败了目标。要避免这种情况，必须做到三件事：每人先看桶里有没有水，途中设临时缓冲区，最后准备备用水源或人工接手。

---

## 问题定义与边界

本文讨论的对象，是“一个 Agent 的输出会直接成为下游 Agent 输入”的多 Agent 流水线。这里的流水线，不要求一定是严格同步的，但至少满足两个条件：一是存在明显的上游和下游依赖；二是下游会基于上游输出来做下一步判断、检索或执行。

问题定义可以写得更精确一些：当上游 Agent 产生幻觉、格式合法但语义错误、遗漏关键约束，或者调用工具后解释错误时，这个输出会被下游当成可信上下文使用，导致后续决策偏移。若缺乏验证和隔离，这种偏移会逐步放大，最终触发 cascading failure，也就是级联失效。

“幻觉”一词首次出现时可以简单理解为：模型给出了看似完整、实则不受真实证据支持的输出。它最危险的地方不是错，而是“像真的”。

记事纸接力的例子很适合说明传播机制。第一人把“今天库存 300”误写成“今天库存 3000”，第二人据此安排发货，第三人据此生成补货计划。每个人都忠实执行自己的局部职责，但系统整体已经偏离真实世界。这里没有任何一环主动作恶，错误来自“错误被当成事实”。

下面这个表可以用来划定边界：

| 幻觉来源 | 传播路径 | 容易恶化的边界条件 |
|---|---|---|
| 无依据推断 | 上游文本输出进入下游提示词 | 全串行结构、无中间验证 |
| 工具调用结果误读 | Agent 误解 API 返回 | 输出格式合法，语义错误难发现 |
| 上下文截断 | 关键信息丢失后继续推理 | 缺乏 trace，无法追溯 |
| 盲目重试 | 同一错误反复提交给下游 | 无断路器、无预算上限 |
| 状态污染 | 错误写入共享 memory | 多 Agent 共享黑板但无版本控制 |

本文不展开两个边界外的话题。第一，不讨论完全并行、互不依赖的 Agent 集群，因为那类系统的主要问题是聚合冲突，而不是串行传播。第二，不讨论训练阶段的模型对齐，只讨论运行时的工程容错，因为大多数系统失败发生在编排、校验和执行层，而不是权重本身。

---

## 核心机制与推导

串行多 Agent 系统为什么脆弱，核心原因有两个：一是概率乘积效应，二是错误语义会向后继承。

先看概率乘积。若三个串联灯泡必须同时亮，整条电路才算成功，那么任一灯泡坏掉，整体就灭。多 Agent 流水线也是同样结构：规划 Agent 出错，执行 Agent 和汇总 Agent 就算各自成功，也是在执行错误目标。形式上：

$$
R_{\text{sys}}=\prod_{i=1}^{n}r_i
$$

若每个 Agent 成功率相同，记为 $r$，则：

$$
R_{\text{sys}}=r^n
$$

所以链路越长，整体可靠度下降越快。例如 $r=0.95$ 时，2 步任务成功率约为 $90.25\%$，8 步任务就降到约 $66.34\%$。这还没有把“错误会放大”这件事算进去，因此真实情况通常更差。

再看语义传播。传统服务链路中，失败往往表现为报错码、超时、异常；而 Agent 系统中，很多失败表现为“内容错误但结构正常”。这类错误最难，因为 JSON schema 可以通过，字段也齐全，下游照样能消费。结果是验证只挡住了格式错，没有挡住语义错。

所以容错要分层做。

Agent 级的目标是“不要把明显错误交出去”。这里通常做三件事：第一，输出前验证，例如检查证据引用、字段完整性、关键约束；第二，有限重试，例如最多 2 次；第三，自我纠错，即让同一 Agent 用独立提示检查自己的输出是否违背约束。它不保证绝对正确，但能拦住一部分低级错误。

流水线级的目标是“就算上游错了，也别让整条链一起陪葬”。checkpoint 就是运行时快照，白话讲就是“到某一步先存档”。一旦后续验证失败，可以回到最近一个可信状态，而不是从头再跑，也不是带着污染状态继续执行。

系统级的目标是“连续失败时主动切断传播”。这里最典型的是断路器。断路器是一种失败隔离模式，可以简单理解为“某条路径连续失败太多次，就先别走了”。它通常有三个状态：

| 状态 | 含义 | 触发条件 | 下一步 |
|---|---|---|---|
| Closed | 正常放行 | 失败率未超阈值 | 继续请求 |
| Open | 熔断拒绝 | 连续失败达到阈值 | 直接降级或返回默认结果 |
| Half-open | 半开探测 | 冷却时间到 | 放少量探测请求判断是否恢复 |

状态转换可以记成：

`Closed -> Open -> Half-open -> Closed/Open`

若系统连续三次验证失败，就从 `Closed` 进入 `Open`；在冷却窗口内不再调用该链路，而是改走备用模型、缓存结果，或者转人工。冷却时间到后进入 `Half-open`，只放少量测试流量。如果恢复，再回 `Closed`；如果仍失败，重新 `Open`。

真实工程例子可以看高性能生物计算中的容错调度。那类系统会同时监控 job 和 compute core。检测到执行核心异常时，任务不会一直在故障核心上重试，而是迁移到可靠核心，或者结合 checkpoint 从最近安全状态恢复。论文中的结论是，这类分层容错相比集中式 checkpoint 增加的时间开销大约在 10% 左右，但能显著减少人工干预和数据丢失。这个结论对多 Agent 系统同样适用：适度增加校验和快照成本，通常比一次完整系统失效便宜得多。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，演示三层策略如何组合：Agent 输出校验与重试、流水线 checkpoint 回滚、系统级断路器与人工降级。代码不依赖外部库，可以直接运行。

```python
from dataclasses import dataclass, field
from copy import deepcopy

def verify(output: dict) -> bool:
    # 最小验证：必须有值，且显式标记为可信
    return bool(output.get("value")) and output.get("trusted") is True

@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    reset_timeout_ticks: int = 2
    state: str = "CLOSED"
    failures: int = 0
    open_until_tick: int = -1

    def allow(self, tick: int) -> bool:
        if self.state == "OPEN" and tick >= self.open_until_tick:
            self.state = "HALF_OPEN"
        return self.state != "OPEN"

    def record_success(self):
        self.state = "CLOSED"
        self.failures = 0

    def record_failure(self, tick: int):
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            self.open_until_tick = tick + self.reset_timeout_ticks

@dataclass
class PipelineState:
    data: dict = field(default_factory=dict)

    def snapshot(self):
        return deepcopy(self.data)

    def rollback(self, checkpoint):
        self.data = deepcopy(checkpoint)

def flaky_agent(input_data: dict, attempt: int) -> dict:
    # 第一次故意返回“结构合法但语义不可信”的结果
    if attempt == 0:
        return {"value": "inventory=3000", "trusted": False}
    return {"value": "inventory=300", "trusted": True}

def run_agent_with_retry(input_data: dict, max_retries: int = 2) -> dict:
    last = None
    for attempt in range(max_retries + 1):
        candidate = flaky_agent(input_data, attempt)
        last = candidate
        if verify(candidate):
            return candidate
    raise ValueError(f"agent output invalid after retries: {last}")

def run_pipeline(tick: int, breaker: CircuitBreaker):
    if not breaker.allow(tick):
        return {"mode": "degraded", "result": "handoff_to_human"}

    state = PipelineState({"query": "check inventory"})
    checkpoint = state.snapshot()

    try:
        result = run_agent_with_retry(state.data)
        state.data["agent_output"] = result["value"]

        # 红绿校验：绿色表示通过，红色表示回滚
        if state.data["agent_output"] != "inventory=300":
            state.rollback(checkpoint)
            breaker.record_failure(tick)
            return {"mode": "rollback", "result": state.data}

        breaker.record_success()
        return {"mode": "normal", "result": state.data}
    except Exception:
        state.rollback(checkpoint)
        breaker.record_failure(tick)
        return {"mode": "rollback", "result": state.data}

# 单次运行：第一次失败后会自我纠正成功
breaker = CircuitBreaker()
out = run_pipeline(tick=0, breaker=breaker)
assert out["mode"] == "normal"
assert out["result"]["agent_output"] == "inventory=300"
assert breaker.state == "CLOSED"

# 连续制造失败，观察断路器打开
breaker2 = CircuitBreaker()
for t in range(3):
    breaker2.record_failure(tick=t)
assert breaker2.state == "OPEN"
assert breaker2.allow(0) is False

# 冷却窗口后进入半开
assert breaker2.allow(5) is True
assert breaker2.state == "HALF_OPEN"
```

这段代码表达了几个关键点。

第一，`verify(output)` 必须存在，而且要在交给下游之前执行。这里用了最简化的规则，但在真实工程里应该验证 schema、证据引用、业务约束、置信度和 trace id。

第二，`snapshot()` 和 `rollback()` 不是可选优化，而是污染控制。没有 checkpoint，错误会被写进共享状态，之后哪怕修复某一步，也很难证明系统已经恢复一致性。

第三，断路器不等于简单重试。盲目重试只是把同一个错误更快地重复更多次；断路器的作用是，当失败超过阈值时，主动拒绝继续扩散，并切到降级路径。

真实工程里，常见实现会再加三样东西：一是 `trace_id`，用于追踪错误从哪一跳开始出现；二是输出哈希，用于快速判断下游是否在消费旧状态；三是 idempotency key，也就是幂等键，白话讲是“同一个动作重复提交也只执行一次”的标识，避免重试把外部副作用放大。

---

## 工程权衡与常见坑

容错不是越多越好，而是在可靠性、延迟、成本之间找平衡。最大的问题通常不是“不会做”，而是“做了以后太慢、太贵、太复杂”。因此需要先看失败代价，再决定投入多少防线。

常见坑和规避方式可以整理成表格：

| 风险 | 规避手段 | 成本影响 |
|---|---|---|
| 盲目重试导致 API 冲击波 | 指数退避、断路器、预算上限 | 增加调度复杂度，明显降低异常成本 |
| 缺乏可观测性，错误静默传播 | trace、step 日志、输出 hash、质量评分 | 增加存储和日志处理开销 |
| 只有格式验证，没有语义验证 | 交叉校验、引用检查、 judge agent 或规则校验 | 增加延迟与模型调用次数 |
| checkpoint 太少，回滚代价大 | 关键节点快照，而非每步全量快照 | 约增加一定存储与序列化开销 |
| checkpoint 太频繁 | 分层快照，只在高价值节点存档 | 牺牲部分恢复粒度，减少开销 |
| 降级路径未验证 | 预演故障演练和 canary 验证 | 需要额外测试预算 |

一个常见误区是把“多做几次”当成容错。对确定性 bug、脏上下文、配额耗尽、工具 500 错误来说，重试往往不会变好，只会更贵。Reddit 上不少一线经验都提到，Agent 一旦进入重复动作循环，成本会迅速失控。更有效的做法是给重试加条件，例如只对可恢复错误重试，对重复状态直接终止。

另一个误区是过度依赖 schema。schema 很重要，但它只回答“长得对不对”，不回答“内容真不真”。多 Agent 场景里最危险的是“看上去对的错答案”。所以至少要在高风险节点加语义验证，例如检查引用是否能回到原始证据、关键数值是否和上游工具结果一致、业务规则是否被违反。

再看性能权衡。频繁 checkpoint 的确会增加开销，但高代价流程通常值得。比如一个自动化审批链有 8 个 Agent，每次完整失败都要人工复盘 30 分钟，那么哪怕每次多花 10% 延迟去保存关键快照，也通常是划算的。高性能生物计算里的经验也是类似逻辑：轻微时间开销换来更少的全局中断和人工干预。

---

## 替代方案与适用边界

不是所有多 Agent 系统都需要完整的三级容错。设计要看链路长度、外部副作用、可接受延迟和人工介入能力。

先看短链路。只有两步或三步的小任务，例如“检索 Agent -> 汇总 Agent”，往往可以用简化方案：每步做一次输出校验，失败后整体重跑一次。如果总成本低、无外部副作用，这种做法足够实用。

再看长链路或高风险链路。比如“规划 -> 检索 -> 分析 -> 执行 -> 审计”这种五步以上流程，或者涉及发邮件、下单、改数据库等外部动作，就不适合整体重跑。这里更稳妥的策略是局部回滚、局部隔离和系统降级。原因很简单：越往后，重跑成本越高，副作用越难撤销。

下面给一个策略对比表：

| 策略 | 适用情景 | 限制 |
|---|---|---|
| 单步自检 + 整体重跑 | 短链路、低成本、无副作用任务 | 长链路成本高，容易重复污染 |
| Agent 级重试 + 语义校验 | 输出经常有轻微漂移，但大体可恢复 | 对系统级故障无能为力 |
| checkpoint + 局部回滚 | 中长链路、状态复杂、需要恢复一致性 | 增加状态管理和存储复杂度 |
| 断路器 + 降级路径 | 外部依赖不稳定、失败会放大成本 | 需要预先准备备用路径 |
| 人工接管 | 高安全、高合规、高成本任务 | 吞吐下降，依赖人工排班 |
| 冗余 Agent 交叉投票 | 高价值决策节点 | 成本高，且可能出现群体性偏差 |

新手版可以这样记：小任务像两步接力，纸条写错了就整条重跑；多步且时间敏感的任务，不能每次都回到起点，而是要回滚最后一跳，必要时换备用资源；涉及钱、数据删除、对外动作的任务，宁可降级或转人工，也不要“相信模型大概没问题”。

因此，替代方案不是谁优谁劣，而是谁和失败代价匹配。低风险任务追求简单，关键流程追求可控，安全敏感任务追求可证明的隔离和接管。

---

## 参考资料

1. [Zylos Research, *Graceful Degradation Patterns in AI Agent Systems*, 2026-02-20](https://zylos.ai/research/2026-02-20-graceful-degradation-ai-agent-systems)  
   贡献：系统化总结 graceful degradation、断路器、fallback chain、bulkhead 和 checkpoint 等运行时容错模式，适合作为系统级降级设计的参考。

2. [MMNTM, *The 5 Agent Failure Modes (And How to Prevent Them)*, 2025-11-27](https://www.mmntm.net/articles/agent-failure-modes)  
   贡献：明确提出 multi-agent 中的 cascade failure、infinite loop、confidence hallucination 等常见失败模式，帮助界定错误传播的表现形式。

3. [Microsoft Learn, *Circuit Breaker pattern*](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)  
   贡献：给出 `Closed -> Open -> Half-open` 状态机及触发逻辑，适合作为断路器实现时的工程参考。

4. [Wikipedia, *Lusser's law*](https://en.wikipedia.org/wiki/Lusser%27s_law)  
   贡献：提供串联系统可靠性乘积公式 $R_{\text{sys}}=\prod_i r_i$，可用于解释多 Agent 串行链路为何会随环节增多而快速变脆。

5. [ScienceDirect, *Automating fault tolerance in high-performance computational biological jobs using multi-agent approaches*](https://www.sciencedirect.com/science/article/abs/pii/S0010482514000365)  
   贡献：展示在真实计算任务中通过 agent/core 双层调度与容错迁移降低人工干预的思路，可类比到多 Agent 执行链的局部转移与恢复。

6. [Reddit, *After 3 months of running multi-agent orchestration in production I finally solved the cascading failure problem. Here is the pattern.*, 2026-03-13](https://www.reddit.com/r/aiagents/comments/1rt1kjh/after_3_months_of_running_multiagent/)  
   贡献：提供一线工程经验，指出“格式校验不足以解决语义错误”，强调 staging buffer、checkpoint hashing、rollback 在生产中的价值。

7. [Reddit, *anyone else's agent get stuck in infinite retry loops or is my ReActAgent just broken*, 2026-02-06](https://www.reddit.com/r/LangChain/comments/1qxgdkz/anyone_elses_agent_get_stuck_in_infinite_retry/)  
   贡献：补充盲目重试的成本风险，说明状态去重、循环检测和预算上限在 Agent 级重试中是必要约束。
