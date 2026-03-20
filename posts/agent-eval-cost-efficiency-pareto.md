## 核心结论

Agent 评测不能只看成功率。成功率只回答“能不能做成”，但生产环境真正关心的是“要花多少资源做成”。这里的资源至少包括 Token、API 调用次数和延迟。Token 是模型输入输出的文本计量单位，通常直接映射为 API 账单；延迟是从发起任务到完成任务的时间，直接影响用户等待体验；API 调用次数会放大失败重试、工具调用和外部服务费用。

在公开资料里，这个判断已经越来越明确。CalmOps 在 2026 年 3 月的综述里，把 `Success Rate`、`Cost Efficiency`、`Latency`、`Token Usage` 并列为 Agent benchmark 的核心维度。SWE-agent 文档也直接提示，如果不设成本或回合上限，平均成本会“趋向无穷”，因为 Agent 会持续迭代而不停止。

更关键的是，前沿模型之间已经出现“性能收敛、成本分化”。AgentPMT 在 2026 年 2 月 20 日汇总的 SWE-Bench Verified 公开数据中写到：Claude Opus 4.6 为 80.8%，Gemini 3.1 Pro 为 80.6%，MiniMax M2.5 为 80.2%，前四名只差 0.8 个百分点；但同文又给出 MiniMax M2.5 输入价格 $0.15/1M tokens，Claude Opus 4.6 为 $5.00/1M tokens，形成约 20 倍价差。这个结构说明，单看“谁最高”已经不够，必须问：这 0.6 个百分点是否值得付出 20 倍 Token 单价。

因此，评测应从“排行榜思维”切换到“帕累托前沿思维”。帕累托前沿可以理解为预算边界上的最优集合：如果一个 Agent 的成功率更低、成本更高，那它就被另一个 Agent 完全压制，没有继续讨论的必要。

---

## 问题定义与边界

本文讨论的是 Agent 评测中的成本效率，而不是单次模型问答质量。对象主要是会多轮思考、调用工具、读写代码库、执行命令的 Agent，典型场景包括 SWE-bench 这类软件工程任务评测。

为了让结论可落地，先把边界说清楚。

第一，至少跟踪四类指标：

| 指标 | 白话解释 | 作用 |
| --- | --- | --- |
| 成功率 `success_rate` | 一批任务里做成了多少 | 衡量结果质量 |
| Token 消耗 `tokens` | 模型总共读写了多少文本 | 近似映射 LLM 成本 |
| API 调用次数 `api_calls` | 调了多少次模型或外部工具 | 衡量系统开销与失败重试 |
| 延迟 `latency` | 完成一个任务花了多久 | 衡量用户体验与吞吐 |

第二，评测必须有预算守门条件。守门条件就是先把“不允许超过的上限”写死，否则不同 Agent 的比较没有公平基础。SWE-agent 官方文档给出的保守建议是：对 Claude 3.7 一类模型，可设置每实例 `$1` 成本上限和 `50` 回合上限。这个思想比具体数字更重要，因为它把“无限尝试”变成了“可控搜索”。

第三，本文不把“官方 API 单价”直接等同于“单任务真实总成本”。真实成本还会受到输入输出比例、缓存、工具调用、失败重试、并发调度影响。所以本文强调的是评测框架：先标准化记录 `success/tokens/api_calls/latency`，再做相对比较，而不是给出一个放之四海而皆准的美元数。

一个玩具例子最容易说明问题：

| Agent | 成功率 | 总 Token | CE: 每千 Token 成功率 |
| --- | --- | --- | --- |
| A | 70% | 200K | 0.35 |
| B | 75% | 2,000K | 0.0375 |

B 的成功率高 5 个百分点，但 Token 用量是 10 倍，因此单位 Token 产出反而更差。对预算有限的团队，A 往往更接近可部署方案。

---

## 核心机制与推导

CRAB benchmark 给出过一个非常直接的思路：`Cost Efficiency = Completion Ratio / Total Tokens`，记作

$$
CE = \frac{CR}{T}
$$

其中 `CR` 是完成比例，意思是任务整体完成到了什么程度；`T` 是总 Token。若任务定义只有成败两种结果，也可以把它近似为：

$$
CE \approx \frac{success\_rate}{tokens / 1000}
$$

也就是“每千 Token 产生多少成功率”。这个指标的价值在于归一化。归一化的意思是把不同规模的消耗拉回同一单位上比较，从而回答：同样花 1000 Token，哪个 Agent 产出更多有效结果。

如果要把 API 调用次数和延迟也纳入，可以做加权扩展：

$$
CE_w = \frac{success\_rate}{tokens/1000} \cdot \frac{1}{1+\lambda \cdot api\_calls+\mu \cdot latency}
$$

这里 $\lambda,\mu$ 是惩罚系数。惩罚系数的作用是把“过多调用工具”和“过慢”显式扣分。Brenndoerfer 在讨论 Agent 效率时提出过效率比 `optimal steps / actual steps`，本质也是在说：正确不等于高效，走弯路应该被惩罚。

帕累托前沿的判定也可以正式写出来。若 Agent $i$ 和 Agent $j$ 满足：

$$
success_i \ge success_j,\quad cost_i \le cost_j
$$

且至少有一个严格不等号成立，那么称 $i$ 支配 $j$。这里的 `cost` 可以是单一 Token，也可以是加权成本：

$$
cost = \alpha \cdot tokens + \beta \cdot api\_calls + \gamma \cdot latency
$$

这样就能把“多目标问题”压缩成可比较的成本轴。

真实工程例子更直观。按 2026 年 2 月 20 日 AgentPMT 汇总的 SWE-Bench Verified 公开数据，Claude Opus 4.6 为 80.8%，MiniMax M2.5 为 80.2%，仅差 0.6 个百分点；但同文给出的输入 Token 单价分别是 `$5.00/1M` 和 `$0.15/1M`。如果你的后端团队预算只有几千美元，且评测目标是“在固定预算内尽量多跑题”，那么你首先应该看的是预算边界内谁最优，而不是榜首是谁。很多时候，真正该进产线的是“略低一点但便宜很多”的那个点。

---

## 代码实现

下面给一个最小可运行的 Python 例子。它读取评测结果，计算成本效率，并找出在“成功率越高越好、Token 越低越好”定义下的帕累托前沿。

```python
from typing import List, Dict

runs: List[Dict] = [
    {"name": "Agent A", "success_rate": 0.70,  "tokens": 200_000,  "api_calls": 18, "latency": 24.0},
    {"name": "Agent B", "success_rate": 0.75,  "tokens": 2_000_000, "api_calls": 42, "latency": 51.0},
    {"name": "Agent C", "success_rate": 0.802, "tokens": 900_000,  "api_calls": 26, "latency": 31.0},
    {"name": "Agent D", "success_rate": 0.808, "tokens": 8_000_000, "api_calls": 54, "latency": 57.0},
]

def cost_efficiency(run: Dict) -> float:
    return run["success_rate"] / (run["tokens"] / 1000)

def weighted_ce(run: Dict, lam: float = 0.01, mu: float = 0.005) -> float:
    base = cost_efficiency(run)
    penalty = 1 + lam * run["api_calls"] + mu * run["latency"]
    return base / penalty

def within_budget(run: Dict, max_tokens: int = 1_000_000, max_latency: float = 40.0) -> bool:
    return run["tokens"] <= max_tokens and run["latency"] <= max_latency

def dominates(a: Dict, b: Dict) -> bool:
    return (
        a["success_rate"] >= b["success_rate"]
        and a["tokens"] <= b["tokens"]
        and (a["success_rate"] > b["success_rate"] or a["tokens"] < b["tokens"])
    )

def pareto_frontier(items: List[Dict]) -> List[str]:
    frontier = []
    for x in items:
        if not any(dominates(y, x) for y in items if y is not x):
            frontier.append(x["name"])
    return frontier

for run in runs:
    run["ce"] = round(cost_efficiency(run), 6)
    run["weighted_ce"] = round(weighted_ce(run), 6)

frontier = pareto_frontier(runs)

assert round(cost_efficiency(runs[0]), 2) == 0.35
assert round(cost_efficiency(runs[1]), 4) == 0.0375
assert "Agent B" not in frontier
assert within_budget(runs[2]) is True
assert within_budget(runs[3]) is False

for run in runs:
    status = "within budget" if within_budget(run) else "over budget"
    print(run["name"], run["ce"], run["weighted_ce"], status)

print("Pareto frontier:", frontier)
```

这段代码里有三个关键点。

第一，`cost_efficiency` 用的是最简单定义：成功率除以千 Token。它适合做第一轮筛选。

第二，`weighted_ce` 把 `api_calls` 和 `latency` 也纳入惩罚。这样可以避免某些 Agent 靠大量重试“堆”出成功率。

第三，`pareto_frontier` 不做加权，直接比较“成功率更高、Token 更低”的支配关系。这种做法更透明，适合画散点图。

如果把上面的示例数据整理成表，结果大致如下：

| Agent | success_rate | tokens | CE | 预算状态 |
| --- | --- | --- | --- | --- |
| Agent A | 0.70 | 200,000 | 0.35 | 通过 |
| Agent B | 0.75 | 2,000,000 | 0.0375 | 淘汰 |
| Agent C | 0.802 | 900,000 | 0.0891 | 通过 |
| Agent D | 0.808 | 8,000,000 | 0.1010 | 淘汰 |

这就是新手最容易忽略的一点：`D` 的 CE 看起来不差，但在预算条件下仍然可能被淘汰。评测不是只算一个分，而是先过预算门，再谈谁更好。

---

## 工程权衡与常见坑

最常见的坑，是把“更多回合”误当成“更强能力”。多轮搜索确实可能提升成功率，但也会同时推高 Token、API 调用次数和延迟。SWE-agent 文档明确提醒：多次尝试配置会非常昂贵；如果不设 per-instance cost limit 或 turn limit，平均成本会发散。

第二个坑，是把“模型单价便宜”误当成“任务一定便宜”。单价低只说明每个 Token 便宜，不代表 Agent 不会疯狂生成上下文、重复调用工具、反复读取文件。真正该盯的是任务级日志：每题花了多少 Token、调用了多少次、跑了多久、最终是否成功。

第三个坑，是只比较平均值，不比较分布。一个 Agent 平均 CE 很高，但如果长尾问题会无限重试，它在生产上仍然危险。对真实系统，应至少看 P95 延迟、P95 Token 消耗、超预算比例。

第四个坑，是把不同实验设置混在一起。比如一个 Agent 用 1 次尝试，另一个 Agent 用 5 次尝试再做投票，表面上都叫“成功率”，但成本结构完全不同。没有统一预算和回合限制，所谓“领先”往往不可比。

实践里，一个更稳妥的配置是：

| 配置项 | 作用 | 典型用途 |
| --- | --- | --- |
| `per_instance_cost_limit` | 限制单题花费 | 防止个别题目烧穿预算 |
| `turn_limit` / `call_limit` | 限制单题迭代次数 | 防止无限重试 |
| `max_tokens` | 限制上下文膨胀 | 控制长上下文成本 |
| `p95_latency` 告警 | 监控慢任务尾部 | 防止线上超时 |

真实工程里，后端团队常见做法不是“选榜首”，而是分层：默认流量走便宜且 CE 高的 Agent；只有高价值、疑难任务才升级到更贵模型。这样才能把预算用于真正有边际收益的部分。

---

## 替代方案与适用边界

CE 很有用，但不是唯一答案。它最适合“任务成败明确、Token 账单明确、预算压力真实存在”的场景，比如代码修复、工单处理、自动化工具调用。

如果你的任务更偏业务侧，也可以用更直观的替代指标。

| 替代指标 | 定义 | 适用场景 |
| --- | --- | --- |
| 每成功任务成本 | `total_cost / successes` | 财务和采购最容易理解 |
| 相对成本效率 | `agent_cost / baseline_cost` | 和基线模型比较 |
| 延迟阈值达标率 | 满足 SLA 的请求占比 | 用户交互场景 |
| Throughput | 单位时间完成任务数 | 并发生产场景 |

例如，Sendbird 的思路更接近业务评估：除了成功率，还看 latency 和 cost efficiency。Agent CI 的性能评测则更工程化，直接给 token、latency、throughput 设阈值。这两类方法都不冲突，它们只是把同一个问题换成更便于落地的表达。

可以给新手一个非常容易执行的评分卡：

1. `success_rate >= 0.70`
2. `cost_per_success <= $0.05`
3. `p95_latency <= 15s`
4. `over_budget_ratio <= 1%`

满足就进“高效队列”，不满足就继续调 prompt、减回合、换模型、缩工具链。这个方法虽然没有帕累托前沿那么数学化，但足够实用，适合团队第一次建立 Agent 评测纪律。

最后要注意适用边界。若任务没有清晰成功标准，只能靠主观评审，那么 CE 会失真，因为分子本身不稳定。若模型价格变化很快，美元口径会过时，这时最好回到更稳的物理量，比如 Token、API 次数和延迟。若任务是高风险领域，还必须把安全性、合规性单独列出来，不能用成本效率掩盖安全风险。

---

## 参考资料

- [CalmOps: AI Agent Evaluation Benchmarks 2026](https://calmops.com/ai/ai-agent-evaluation-benchmarks-2026/)
- [AgentPMT: Twelve Frontier Models. 0.8 Points Apart. The Moat Moved.](https://www.agentpmt.com/articles/twelve-frontier-models-0-8-points-apart-the-moat-moved-2)
- [CRAB Benchmark 论文 PDF](https://aclanthology.org/2025.findings-acl.1113.pdf)
- [CRAB 指标说明摘要](https://www.camel-ai.org/blogs/crab-cross-platform-agent-benchmark)
- [SWE-agent: Competitive runs](https://swe-agent.com/latest/usage/competitive_runs/)
- [SWE-agent: Batch mode](https://swe-agent.com/latest/usage/batch_mode/)
- [SWE-agent: Model config](https://swe-agent.com/latest/reference/model_config/)
- [Michael Brenndoerfer: Agent Evaluation Metrics, Benchmarks and Safety Standards](https://mbrenndoerfer.com/writing/agent-evaluation-metrics-benchmarks-safety)
- [Sendbird: The Complete Guide to AI Agent Evaluation](https://sendbird.com/blog/ai-agent-evaluation-guide)
- [Agent CI: Performance Evaluations](https://agent-ci.com/docs/evaluations/performance)
