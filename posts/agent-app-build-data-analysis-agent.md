## 核心结论

数据分析 Agent 不是“会写 SQL 的聊天机器人”，而是一个能把自然语言目标拆成多步执行计划、调用真实数据工具、持续维护上下文、并在关键节点接受校验的执行系统。这里的 Agent，可以先理解为“会自己分步骤做事的软件代理”。

它成立的前提有三个。

第一，它必须知道你到底在什么数据环境里工作。只会生成通用代码没有意义，真正有用的是能看到表结构、字段类型、业务元数据、当前 Notebook 状态，再据此生成可执行的 SQL、Python 或 PySpark。AWS 在 2025 年 11 月发布 SageMaker Data Agent，2026 年 1 月进一步公开了一个上下文感知的数据分析流程，核心就是让 Agent 直接利用 Glue Data Catalog、DataZone 和 Notebook 上下文来缩短“提问”到“跑出结果”的距离。

第二，它必须把多步执行做成受控流水线，而不是一次性吐出一大段代码。受控，意思是每一步都可看、可验、可停。比如先找相关表，再确认字段，再生成查询，再执行，再把结果解释成图表或结论。中间任一步如果发现字段不存在、数据类型不匹配、结果为空或偏离目标，就要暂停，而不是把错误继续传下去。

第三，它必须把治理和权限放进系统内部。治理，就是“谁能看什么、谁能跑什么、哪些结果能被接受”的规则集合。数据分析 Agent 一旦能直接访问生产数据，它就不再只是交互体验问题，而是权限边界、审计、成本和错误传播问题。

一个初学者能立刻理解的例子是：在 Notebook 里问“计算纽约出租车平均车费”。合格的 Agent 不该直接猜表名，而是先列出候选数据表，确认 fare_amount 字段和时间范围，再生成 SQL，执行后返回结果，并把每一步展示给你审批。你只写一句话，但系统内部完成的是“检索元数据 → 规划 → 执行 → 校验 → 汇报”的闭环。

---

## 问题定义与边界

先把问题说准：数据分析 Agent 的目标，不是替代数据团队做所有决策，而是在一个给定数据域内，把“自然语言需求”稳定转换成“可执行分析动作”。

这里有四个边界。

| 维度 | 数据分析 Agent 应该做什么 | 不应该假装做到什么 |
| --- | --- | --- |
| 数据域 | 读取受权表、字段、业务术语和上下文 | 凭空知道未授权或未登记的数据 |
| 工具链 | 调 SQL、Python、PySpark、可视化工具 | 脱离工具直接编造分析结果 |
| 状态保持 | 记住目标、已完成步骤、失败原因 | 在长流程里反复忘记最初约束 |
| 输出形式 | 生成可执行脚本、查询结果、分析报告 | 只给“看起来合理”的自然语言总结 |

所以，一个数据分析 Agent 至少要回答四个问题：

1. 用户真正要解决的业务问题是什么？
2. 哪些数据对象与这个问题相关？
3. 该用哪类工具完成当前步骤？
4. 当前结果是否足够可信，能否进入下一步？

“问题定义”之所以重要，是因为很多失败都不是模型不会推理，而是系统边界没画清。比如用户说“展示销售增长”，这句话本身至少有四个未定项：销售额还是订单数、按日还是按月、同比还是环比、全站还是某区域。Agent 如果直接生成图表，往往会把模糊需求误当确定需求。

玩具例子很简单。假设数据库里有 `orders` 和 `refunds` 两张表。用户问“这个月收入怎么样”。如果 Agent 没有边界意识，可能只查 `orders.amount`，完全忽略退款。结果不是“算错了一点”，而是业务定义根本错了。

真实工程里，失败通常更隐蔽。比如第二步 SQL 把 `sales_growth` 需求错映射成不存在的字段 `growth_rate`，但系统没有做语义校验，第三步还继续基于空结果生成图表，最后给出一张“看起来正常”的折线图。这类错误最危险，因为它不会报错，只会悄悄把错误包装成结果。Latitude 在 2026 年 3 月总结生产环境 Agent 失败模式时，把这类问题归类为工具误用、上下文丢失、目标漂移和级联失败。

因此，数据分析 Agent 的边界可以压缩成一句话：它负责执行受控分析，不负责替用户发明业务定义。

---

## 核心机制与推导

理解数据分析 Agent，关键不是“它会多少工具”，而是“长链路为什么容易失败”。

设第 $i$ 步出错概率为 $\varepsilon_i$，那么一条包含 $n$ 步的分析流水线整体成功率是：

$$
P_{\text{succ}}=\prod_{i=1}^{n}(1-\varepsilon_i)
$$

这条公式的含义很直接：每一步都要成功，整条链才算成功。哪怕单步错误率很低，步骤一多，整体成功率也会快速下降。

如果每一步错误率都是 $1\%$，一共 100 步，那么：

$$
P_{\text{succ}}=(0.99)^{100}\approx 0.366
$$

也就是说，整条链只有大约 36.6% 的概率完全成功。问题不在某一步“特别差”，而在于错误会累积。

这解释了为什么数据分析 Agent 必须在每一步插入 guard。guard，可以先理解为“防止错误继续传播的检查点”。典型 guard 包括：

| 能力 | 具体动作 | 对应验证手段 |
| --- | --- | --- |
| 上下文理解 | 从 Catalog 和 Notebook 提取相关表与字段 | 元数据匹配、权限检查 |
| 工具切换 | 在 SQL、Python、PySpark 之间选工具 | 语义一致性检查、执行前预览 |
| 中间审查 | 展示计划、代码、结果摘要 | 人工审批、阈值告警 |
| 状态保持 | 保存计划、步骤结果、失败原因 | 快照、可恢复状态仓库 |
| 治理闭环 | 所有操作都跑在受控环境 | IAM、审计日志、数据边界 |

再看一个玩具例子。用户说：“找出上周订单转化率下降的原因。”

一个稳健的 Agent 通常不会一步到位，而会拆成：

1. 明确“转化率”定义，是支付订单数 / 访问数，还是加购数 / 访问数。
2. 找到订单表与流量表，并确认时间字段。
3. 计算本周与上周指标。
4. 如果下降成立，再分渠道、地区、设备做切片。
5. 返回最显著的变化因素，并给出可复现查询。

关键点在第 3 步之后。假设流量表里的日期字段是 `event_date`，订单表里是 `created_at`。如果 Agent 在 JOIN 前没有检查时间粒度和时区，后面的所有归因都会失真。一个好的系统会在“准备 JOIN”时做字段与类型校验，发现不一致就回退并要求人工确认。

真实工程例子更能说明机制。AWS 展示的 SageMaker Data Agent 场景里，用户在 Unified Studio Notebook 中提问，Agent 先利用 Glue 和 DataZone 元数据定位资产，再结合当前 Notebook 状态生成计划与代码。这里真正有价值的不是“它写了 SQL”，而是“它拿着你当前环境的上下文写了能跑的 SQL”，并且每一步都有可见的中间结果。这样做的意义是把错误尽量截断在早期，而不是让一条错误 SQL 继续污染后续 Python 分析和报告生成。

所以，数据分析 Agent 的核心机制不是“大模型一次答对”，而是“分步执行 + 中间验证 + 状态传播 + 出错即停”。

---

## 代码实现

最小实现不需要复杂框架，但一定要有四个组件：计划器、工具执行器、验证器、状态仓库。

计划器负责把用户目标拆成步骤。工具执行器负责调用 SQL 或 Python。验证器负责判断当前输出是否满足进入下一步的条件。状态仓库负责保存“目标是什么、做到了哪一步、哪里失败过”。

下面是一个可运行的玩具实现。它不连接真实数据库，但把数据分析 Agent 最关键的控制流写全了。

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Step:
    name: str
    tool: str
    args: Dict[str, Any]


@dataclass
class AgentState:
    goal: str
    completed: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    halted: bool = False
    reason: str = ""


def make_plan(question: str) -> List[Step]:
    return [
        Step("discover_schema", "catalog", {"question": question}),
        Step("generate_query", "planner", {"metric": "avg_fare"}),
        Step("run_query", "sql", {"sql": "SELECT AVG(fare_amount) AS avg_fare FROM nyc_taxi"}),
    ]


def call_tool(step: Step, state: AgentState) -> Dict[str, Any]:
    if step.name == "discover_schema":
        return {"tables": ["nyc_taxi"], "columns": ["fare_amount", "pickup_datetime"]}
    if step.name == "generate_query":
        if "fare_amount" not in state.context.get("columns", []):
            return {"ok": False, "error": "missing required column"}
        return {"ok": True, "preview_sql": "SELECT AVG(fare_amount) FROM nyc_taxi"}
    if step.name == "run_query":
        return {"ok": True, "rows": [{"avg_fare": 17.35}]}
    return {"ok": False, "error": "unknown step"}


def validate(step: Step, result: Dict[str, Any]) -> bool:
    if step.name == "discover_schema":
        return "nyc_taxi" in result.get("tables", []) and "fare_amount" in result.get("columns", [])
    if step.name == "generate_query":
        return result.get("ok") is True and "AVG(fare_amount)" in result.get("preview_sql", "")
    if step.name == "run_query":
        rows = result.get("rows", [])
        return result.get("ok") is True and len(rows) == 1 and rows[0]["avg_fare"] > 0
    return False


def run_agent(question: str) -> AgentState:
    state = AgentState(goal=question)
    for step in make_plan(question):
        result = call_tool(step, state)
        if not validate(step, result):
            state.halted = True
            state.reason = f"validation failed at {step.name}"
            break

        # 把本步产出的关键信息写入上下文，供后续步骤继续使用
        state.context.update(result)
        state.completed.append(step.name)

    return state


state = run_agent("计算纽约出租车平均车费")
assert state.halted is False
assert state.completed == ["discover_schema", "generate_query", "run_query"]
assert abs(state.context["rows"][0]["avg_fare"] - 17.35) < 1e-9
```

这段代码里最重要的不是 `AVG(fare_amount)`，而是循环内部这件事：

1. 先执行工具。
2. 再校验结果。
3. 通过后才写入上下文。
4. 任何一步失败就中止。

如果把这套结构映射到真实系统，通常会演变成下面的工程形态：

- `planner`：把请求拆成可执行 DAG 或顺序计划。DAG 就是“带依赖关系的步骤图”。
- `tool runner`：封装 Athena、Spark、Notebook kernel、可视化引擎等工具。
- `validator`：检查字段存在性、SQL 是否越权、结果是否为空、分布是否异常。
- `state store`：保存 plan、已完成步骤、快照、预算消耗和人工审批记录。

真实工程例子可以是“分析某电商最近 30 天退款率变化”。系统可能先在 Catalog 中找到 `orders`、`payments`、`refunds` 三张表，再生成 SQL 统计退款率，接着切换到 Python 画趋势图。这里的难点不是单个工具调用，而是跨工具时如何保留语义一致性。比如 SQL 输出的 `refund_rate` 到 Python 绘图阶段必须仍然指向同一个定义，而不是在中途被误解释成“退款金额占比”。

---

## 工程权衡与常见坑

工程上最大的权衡，是自动化程度和审查频率之间的平衡。

自动化越高，用户体验越流畅，但 silent failure 风险越高。silent failure，就是“系统表面完成了任务，实际上结果已经偏了”。审查越频繁，准确性越高，但速度变慢、交互中断更多。

常见失败模式可以直接列成表：

| 失效类型 | 典型症状 | 缓解策略 |
| --- | --- | --- |
| 工具误用 | SQL 参数错、表名错、空响应未处理 | 执行前 lint，执行后结果断言 |
| 上下文丢失 | 后续步骤忘了最初业务定义 | 保存目标快照，阶段性重述目标 |
| 目标漂移 | 从“分析原因”变成“生成图表” | 每隔若干步与初始目标做语义比对 |
| 级联失败 | 第 2 步错了，第 5 步才暴露 | 每步校验，不允许未验证结果入状态 |
| 成本失控 | 长流程不断堆上下文，token 暴涨 | 预算阈值、上下文压缩、超额人工接管 |
| 权限越界 | 访问不该看的表或字段 | IAM、白名单工具、审计日志 |

一个非常实用的策略是“阶段快照”。比如每 5 步保存一次状态摘要，包括初始目标、当前子目标、最近结果、剩余预算。如果快照和初始目标的语义差异超过阈值，就自动暂停。这种做法本质上是在检测目标漂移。

新手最容易忽视的坑有两个。

第一个坑是把“代码能运行”误当成“分析正确”。SQL 能跑通，只说明语法对，不说明业务定义对。比如平均客单价是否要排除退款订单，只有业务约束能决定。

第二个坑是把“模型知道很多”误当成“系统知道你的数据”。模型知道什么是销售额，不等于它知道你库里的 `net_revenue` 和 `gross_revenue` 分别代表什么。没有元数据、没有权限、没有上下文，Agent 只能猜。

因此，工程上真正该优化的，不是让 Agent “更像人”，而是让它“更像一个带审计和断言的执行器”。

---

## 替代方案与适用边界

不是所有分析任务都值得上“完整数据分析 Agent”。选择架构要看任务复杂度、数据风险和协作需求。

| 任务特征 | 推荐架构 |
| --- | --- |
| 单步查询、定义清晰、低风险 | 单 Agent 顺序 Pipeline |
| 多步依赖、需跨 SQL/Python 切换 | 单 Agent + 计划执行器 |
| 子任务可并行、角色明显不同 | Orchestrator + Specialist 多 Agent |
| 多团队共享状态、长流程协作 | 分层编排 + 共享黑板 |

单 Agent 顺序 Pipeline 最简单，适合“查一个指标、画一张图”这类任务。优点是实现快、调试容易，缺点是上下文一长就容易腐化，复杂任务下也难拆分权限边界。

Orchestrator + Specialist 的意思是“一个总调度 Agent 负责拆任务，多个专业 Agent 分别负责 schema 检索、SQL 生成、可视化、解释”。这种结构适合复杂任务，但通信设计必须清晰，否则 handoff 时容易丢信息。handoff 可以理解为“一个 Agent 把部分工作结果交给另一个 Agent”。

共享黑板模式适合更复杂的协作。黑板，就是一块所有 Agent 都能读写的共享状态区。举个真实工程风格的例子：

- `Schema Reader Agent` 负责从 Catalog 中取表结构、字段注释、权限信息。
- `Query Builder Agent` 根据黑板上的结构信息写 SQL。
- `Visualization Agent` 读取查询结果快照生成图表。
- `Supervisor Agent` 检查是否偏离用户目标。

相比“一个全能 Agent 什么都做”，这种拆法在复杂系统里更稳定，因为每个 Agent 的职责更窄，校验点更清楚。但代价是延迟更高、调试链路更长、上下文同步更复杂。

所以适用边界很明确：

如果你的任务只是“帮我写个 SQL 草稿”，没必要上多 Agent。
如果你的任务涉及真实数据资产、跨工具链、长步骤执行和权限治理，那么数据分析 Agent 才有充分价值。
如果你的任务已经复杂到需要不同安全域、不同专业角色和并行执行，再考虑多 Agent 编排。

---

## 参考资料

1. AWS Big Data Blog. *Accelerate context-aware data analysis and ML workflows with Amazon SageMaker Data Agent*. 2026-01-22. https://aws.amazon.com/blogs/big-data/accelerate-context-aware-data-analysis-and-ml-workflows-with-amazon-sagemaker-data-agent/
2. AWS What's New. *Introducing Amazon SageMaker Data Agent for analytics and AI/ML development*. 2025-11-21. https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-sagemaker-data-agent-analytics-ai-ml-development
3. AWS What's New. *Announcing notebooks with a built-in AI agent in Amazon SageMaker*. 2025-11-21. https://aws.amazon.com/about-aws/whats-new/2025/11/notebooks-built-in-ai-agent-amazon-sagemaker/
4. Latitude. *Detecting AI Agent Failure Modes in Production: A Framework for Observability-Driven Diagnosis*. 2026-03-26. https://latitude.so/blog/ai-agent-failure-detection-guide
5. Kenaz. *Production AI Patterns: Multi-Agent Systems, Long-Running Agents, and Failure Modes*. 2026. https://kenaz.ai/wiki/practical-patterns
6. Chanl. *The Multi-Agent Pattern That Actually Works in Production*. 2026-03-20. https://www.chanl.ai/blog/multi-agent-orchestration-patterns-production-2026
