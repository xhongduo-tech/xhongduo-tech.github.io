## 核心结论

研究助手 Agent 不是“一个会聊天的大模型”，而是一个被编排过的任务系统。更准确地说，它是一组分工明确的子 Agent 与工具节点：先把用户问题转成结构化任务，再分别做数据发现、分析执行、结果验证和报告生成，最后由协调器统一收口。这里的“协调器”就是总控模块，负责决定谁先做、谁并行做、失败后怎么回退。

它之所以成立，不是因为多 Agent 听起来更高级，而是因为开放式研究任务天然包含多种不同推理模式。找数据、写 SQL、跑统计检验、解释结果、补引用，这些步骤对上下文的要求不同，失败模式也不同。把它们塞进同一个长提示里，常见结果是注意力稀释、错误传播、重试失控。PublicAgent 这类工作给出的核心启发是：把意图澄清、数据查找、分析、报告拆开，每一步都只处理当前最相关的信息，并把输出交给下一步验证。

但多 Agent 不是默认更可靠。真实工程里，可靠性、成本和可观测性才是硬指标。Redis 的工程总结很直接：如果单步失败率是 5%，20 步链路的整体成功率会迅速下降；Kalvium 的经验也说明，系统提示过长、工具部分失效没有 fallback、每步不计成本，都会让 Agent 系统在生产环境里失控。研究助手 Agent 的正确目标不是“尽量自动”，而是“在预算内、可审计、可中断地自动”。

| 模块 | 主要职责 | 控制点 |
| --- | --- | --- |
| 意图澄清 | 把模糊自然语言转换为结构化任务要求 | 一致性检验、确认提问 |
| 数据查找 | 搜索并评估公开数据集、论文、API | 可用性评分、调用限额 |
| 分析执行 | 用代码、统计或模拟完成验证 | 步骤预算、结果校验 |
| 报告生成 | 组织结论、引用与局限性 | 模板约束、来源可追踪 |
| Judge/协调器 | 串联各层并决定重试或回退 | 预算表、超时、人工接管 |

---

## 问题定义与边界

研究助手 Agent 处理的是“开放式科研任务”。所谓开放式，就是用户给出的不是一个固定输入到固定输出的题目，而是一个方向，例如“分析气候政策对私营车销量的影响”。这类任务的问题不在于模型会不会生成文字，而在于它必须先补全任务边界：研究对象是什么，时间和地域范围是什么，目标变量是什么，允许使用哪些数据源，最后产出是图表、备忘录还是可复现实验。

术语首次说明如下。

“意图解析”就是把一句人话拆成机器可检查的字段。
“记忆”就是保存前面已经确认过的约束，避免后面步骤反复改口。
“规划”就是把大任务拆成有顺序和依赖关系的小任务。
“验证”就是检查上一层输出是否足够可信，决定继续、重试还是停止。

玩具例子可以非常小。假设用户说：“我想评估 A/B 测试在不同城市的效果。”这句话至少缺四个信息：指标是什么、城市样本是否可比、实验周期多长、显著性标准是什么。一个合格的研究助手不会直接输出“结论”，而是先形成如下结构化任务：

| 字段 | 示例 |
| --- | --- |
| 目标 | 评估不同城市中实验组相对对照组的转化提升 |
| 指标 | 次日留存率、付费转化率 |
| 范围 | 北京、上海、成都；2026-01 至 2026-03 |
| 数据源 | 实验日志、订单表、城市画像表 |
| 输出 | 统计检验结果 + 业务解释 + 风险说明 |

真实工程例子更复杂。比如“分析气候政策对私营车销量的影响”，系统不能把“政策”“销量”“影响”混成一个大问题，而应拆成至少四层：政策事件抽取、销量数据发现、因果识别或准实验分析、结论生成。每个 Agent 都应声明自己的边界，例如：

- 意图 Agent 只能澄清字段，不能伪造数据源。
- Discovery Agent 只能返回数据集候选与元信息，不能直接得出政策效果。
- Analysis Agent 只能基于已确认数据运行代码，不能私自扩大研究范围。
- Report Agent 只能总结已验证结果，不能补造引用。

这类边界的意义是防止越权。越权的白话解释是：前一个模块把后一个模块该做的事提前“脑补”了，系统看起来更快，但错误更隐蔽。

---

## 核心机制与推导

研究助手 Agent 的核心机制可以概括成一句话：把长链任务拆成短链闭环，并在每个闭环上加控制点。这里的“控制点”就是显式的停顿位置，例如预算检查、输出校验、人工确认、失败上报。

为什么必须这样做，可以用一个简单公式说明。设单步失败率为 $p$，总步数为 $n$，若把每一步近似看成独立，则端到端成功率为：

$$
P_{\text{success}} = (1-p)^n
$$

推导并不复杂。第一步成功概率是 $(1-p)$，第二步也成功仍是 $(1-p)$，连续 $n$ 步都成功，就把这些概率相乘，得到 $(1-p)^n$。

如果 $p=0.05$，$n=20$，则：

$$
P_{\text{success}} = 0.95^{20} \approx 0.358
$$

意思很直接：即使每一步只有 5% 的失败概率，20 步长链最后成功的概率也只有约 35.8%。这就是为什么“一个模型自己想 20 步再给我答案”通常不可靠。问题不在单次推理不够聪明，而在错误会沿链路累积。

下面给一个可运行的最小示例，同时展示可靠性估算和预算控制：

```python
from dataclasses import dataclass

class BudgetExceeded(Exception):
    pass

budgets = {
    "intent": 2.0,
    "discovery": 5.0,
    "analysis": 10.0,
    "report": 3.0,
}

def enforce_budget(agent_id: str, tool_cost: float) -> float:
    budget = budgets[agent_id]
    if budget - tool_cost < 0:
        raise BudgetExceeded(f"{agent_id} 无足够预算")
    budgets[agent_id] -= tool_cost
    return budgets[agent_id]

def success_probability(step_fail_rate: float, steps: int) -> float:
    return (1 - step_fail_rate) ** steps

# 预算检查
left = enforce_budget("discovery", 1.5)
assert abs(left - 3.5) < 1e-9

# 可靠性估算
p = success_probability(0.05, 20)
assert 0.35 < p < 0.36

# 超预算应抛错
raised = False
try:
    enforce_budget("intent", 5.0)
except BudgetExceeded:
    raised = True

assert raised
print("ok")
```

这个玩具代码反映了两个工程事实。

第一，Judge 层必须先看预算再允许工具调用，否则 Agent 会把“继续尝试”当成默认策略。
第二，长链任务必须分段验证。例如把“跨语言语料对齐”拆成“数据收集、预处理、对齐算法、人工抽检”四段，并给每段单独统计失败率；如果累计成功概率跌到某个阈值，例如 60% 以下，就直接触发人工介入，而不是继续盲跑。

PublicAgent 的价值就在这里：不是让模型更会思考，而是让不同思考方式在不同阶段发生，并且每阶段都可验证。

---

## 代码实现

最小可用实现不需要上来就做复杂框架。对新手来说，先把“状态、输出、反馈”三元组写清楚，比追求炫目的多 Agent 图更重要。

“状态”是当前已确认的信息。
“输出”是每个 Agent 的结构化结果。
“反馈”是下一层或 Judge 对输出的检查结论。

下面是一个足够接近真实系统的最小 orchestrator：

```python
from dataclasses import dataclass, field

@dataclass
class TaskSpec:
    question: str
    target: str
    scope: str
    data_requirements: list[str]

@dataclass
class AgentResult:
    ok: bool
    payload: dict
    reason: str = ""

@dataclass
class RunState:
    cost: float = 0.0
    steps: int = 0
    trace: list[str] = field(default_factory=list)

class IntentAgent:
    def refine(self, query: str) -> AgentResult:
        if "影响" not in query and "评估" not in query:
            return AgentResult(False, {}, "问题目标不清晰")
        spec = TaskSpec(
            question=query,
            target="私营车销量变化",
            scope="按国家/地区与年份分层",
            data_requirements=["政策时间线", "销量数据", "控制变量"],
        )
        return AgentResult(True, {"task_spec": spec.__dict__})

class DiscoveryAgent:
    def find(self, task_spec: dict) -> AgentResult:
        if "政策时间线" not in task_spec["data_requirements"]:
            return AgentResult(False, {}, "缺少关键数据需求")
        return AgentResult(True, {
            "datasets": [
                {"name": "policy_events", "quality": 0.82},
                {"name": "vehicle_sales", "quality": 0.91},
            ]
        })

class AnalysisAgent:
    def execute(self, datasets: list[dict]) -> AgentResult:
        if len(datasets) < 2:
            return AgentResult(False, {}, "数据不足，无法分析")
        return AgentResult(True, {
            "method": "difference_in_differences",
            "effect": -0.07,
            "confidence_note": "需要进一步做平行趋势检查"
        })

class ReportAgent:
    def summarize(self, analysis: dict) -> AgentResult:
        return AgentResult(True, {
            "summary": f"估计政策后私营车销量变化约为 {analysis['effect']:.0%}",
            "method": analysis["method"],
            "limitations": analysis["confidence_note"],
        })

class Judge:
    def validate(self, stage: str, result: AgentResult) -> AgentResult:
        if not result.ok:
            return result
        if stage == "discovery":
            qualities = [x["quality"] for x in result.payload["datasets"]]
            if min(qualities) < 0.6:
                return AgentResult(False, {}, "发现的数据质量过低")
        return result

class ResearchCoordinator:
    def __init__(self):
        self.judge = Judge()

    def run(self, query: str) -> dict:
        intent = self.judge.validate("intent", IntentAgent().refine(query))
        if not intent.ok:
            return {"status": "need_clarification", "reason": intent.reason}

        discovery = self.judge.validate("discovery", DiscoveryAgent().find(intent.payload["task_spec"]))
        if not discovery.ok:
            return {"status": "fallback", "reason": discovery.reason}

        analysis = self.judge.validate("analysis", AnalysisAgent().execute(discovery.payload["datasets"]))
        if not analysis.ok:
            return {"status": "human_review", "reason": analysis.reason}

        report = ReportAgent().summarize(analysis.payload)
        return {"status": "done", "report": report.payload}

result = ResearchCoordinator().run("分析气候政策对私营车销量的影响")
assert result["status"] == "done"
assert "report" in result
print(result["report"]["summary"])
```

这个实现的重点不是模型调用，而是回退路径。

- `intent` 失败，返回澄清问题。
- `discovery` 失败，返回 fallback，不假装有数据。
- `analysis` 失败，进入人工审核。
- `report` 只消费已经验证过的分析结果。

真实工程例子可以是一个政策情报系统。用户输入“比较欧洲 2020 年后新能源补贴变化与私营车销量走势”，系统流程如下：

1. IntentAgent 解析出研究对象、时间范围、地区粒度。
2. DiscoveryAgent 去公开数据平台和统计年鉴目录找数据，并生成字段摘要。
3. AnalysisAgent 生成并执行 Python 或 SQL，做趋势图、回归或双重差分。
4. Judge 检查缺失值比例、样本覆盖、异常系数。
5. ReportAgent 产出可审阅报告，并带上数据来源和局限性。

真正可用的研究助手，不是“最后答案写得像论文”，而是中间每一步都能被人检查。

---

## 工程权衡与常见坑

工程权衡的第一条是：不要把多 Agent 当成默认答案。协调开销、状态同步、日志追踪、成本归因都会增加。只有当任务天然可分解，并且每个阶段确实存在不同失败模式时，多 Agent 才值回复杂度。

第二条是提示词越长不一定越好。Kalvium 的经验很实用：一个试图覆盖所有边角情况的 3000 词系统提示，往往不如一个明确写出角色、工具、约束和输出模板的短提示。原因不是模型“读不完”，而是有效注意力被无关规则稀释。

第三条是一定要处理“部分失效”。部分失效的白话解释是：不是整个系统挂了，而是某一个工具这次没结果、超时或返回半截数据。若没有明确 fallback，Agent 很容易开始无限改写查询、重复调用同一 API，最后同时损失预算和上下文。

| 常见坑 | 结果 | 规避策略 |
| --- | --- | --- |
| 过长系统提示 | 角色不清、注意力稀释 | 固定“角色 + 工具指南 + 输出模板” |
| 工具部分失效未处理 | 幻觉补结果或无限重试 | 明确 fallback：上报无结果、停止重试 |
| 不做预算控制 | 每次查询成本飙升 | 为任务和子 Agent 分配预算 |
| 不做中间校验 | 错误一路传到最终报告 | 每层输出进入下一层前先验证 |
| 不做评测集 | 改 prompt 后性能退化不自知 | 维护 50 到 100 条基准任务做回归测试 |

一个典型失败模式是：政策数据搜索 API 连续超时，但 DiscoveryAgent 没有调用上限。它会不断改写关键词、重复请求，协调器又把这些重试都记进同一上下文，最后既烧钱，又让后续 AnalysisAgent 拿到一堆互相矛盾的“候选数据源”。正确做法是把“最多重试 2 次”“超时则上报空结果”“空结果触发人工补数据源”写成硬规则，而不是寄希望于模型自己克制。

---

## 替代方案与适用边界

研究助手 Agent 并不是唯一方案。对于很多团队，正确路线是从简单方案起步，等失败模式出现后再升级。

| 方案 | 优点 | 限制 | 适用场景 |
| --- | --- | --- | --- |
| Plan-and-Execute | 结构清晰、成本低 | 计划一旦错，适应性差 | 环境稳定、步骤已知 |
| 单 Agent + RAG | 搭建快、维护简单 | 易注意力稀释，错误不易定位 | 资料问答、轻量分析 |
| 多 Agent 协作 | 专责明确、便于插入验证 | 编排复杂、监控要求高 | 开放式研究、长链分析 |
| 人机协作半自动 | 风险低、结论可控 | 自动化程度较低 | 高风险、高价值研究任务 |

如果你是新手，最合理的路线通常不是直接做四五个 Agent，而是：

1. 先做单 Agent + 检索，把“找公开数据 + 生成图表”打通。
2. 当你发现同一上下文里既要找数据又要跑分析，错误开始增多时，再把 Discovery 和 Analysis 拆开。
3. 当你发现系统经常因为问题表述模糊而走错方向时，再加入 IntentAgent。
4. 当你需要审计、预算和人工接管时，再加入 Judge 和完整 trace。

这也是研究助手 Agent 的适用边界。若任务只是“根据给定文档回答问题”，用高级 RAG 就够了。若任务是“围绕模糊目标，持续找资料、写代码、检验假设、更新结论”，多 Agent 才有明显收益。换句话说，复杂架构应该由问题逼出来，而不是由流行词驱动。

---

## 参考资料

| 参考资料 | 核心贡献 |
| --- | --- |
| [Cai et al. (2025), Designing LLM-based Multi-Agent Systems for Software Engineering Tasks](https://www.researchgate.net/publication/397522452_Designing_LLM-based_Multi-Agent_Systems_for_Software_Engineering_Tasks_Quality_Attributes_Design_Patterns_and_Rationale) | 总结 LLM 多 Agent 的质量属性、角色协作模式与设计动机 |
| [Montazeri et al. (2025), PublicAgent](https://huggingface.co/papers/2511.03023) | 给出意图澄清、数据发现、分析、报告四阶段分工与验证思路 |
| [Redis Blog (2026), AI Agent Architecture](https://redis.io/blog/ai-agent-architecture/) | 讨论可靠性、控制点、Plan-and-Execute 与多 Agent 的适用边界 |
| [Kalvium Labs (2026), Building AI Agents](https://www.kalviumlabs.ai/blog/building-ai-agents-architecture-tradeoffs/) | 提供生产环境中的提示词、fallback、评测与成本控制经验 |
