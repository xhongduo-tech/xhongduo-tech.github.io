## 核心结论

多 Agent 评估不能只看“最后答对没有”。对零基础读者来说，可以把它理解为：不仅要看结果，还要看这群 Agent 为了得到结果花了多少步、多少钱、多久。更准确地说，应同时跟踪四个维度：

| 维度 | 它回答的问题 | 常见指标 | 为什么不能缺 |
|---|---|---|---|
| 结果 | 任务有没有完成 | 成功率、Goal Success Rate | 只看过程不看结果没有意义 |
| 过程 | 协作是否高效 | 消息轮次、动作数、效率比 | 能发现互相踢皮球、重复规划 |
| 成本 | 为结果付出了多少资源 | Token 成本、工具调用次数 | 成功但太贵，生产上不可用 |
| 体验 | 用户要等多久 | 平均延迟、P95 延迟 | 用户感知首先来自等待时间 |

如果把多 Agent 系统看成一个“会开会的软件团队”，那么成功率是交付是否完成，消息轮次是开了多少轮会，Token 成本是会议成本，延迟是用户等了多久。

一个新手友好的判断标准是：结果、过程、成本、体验必须一起看。比如下面这个“改进前后”对比中，成功率从 70% 提到 78%，同时消息轮次、Token 和延迟都下降，这才说明优化是完整的，不是靠堆资源硬刷出来的。

| 版本 | 任务成功率 | 平均消息轮次 | 平均 Token 成本 | 平均延迟 |
|---|---:|---:|---:|---:|
| 改进前 | 70% | 12 | 8k | 3.2s |
| 改进后 | 78% | 9 | 6.1k | 2.4s |

结论可以压缩成一句话：多 Agent 评估本质上是“结果 + 过程 + 成本 + 体验”的统一度量体系。

---

## 问题定义与边界

先定义什么叫“评估基准”。基准就是一组固定场景，用来让不同系统在相同条件下重复做题。场景通常包含三部分：

1. 用户意图，比如“帮我在电商网站找到最便宜且满足条件的商品”。
2. 环境约束，比如只能点击、搜索、读取页面，不能直接知道答案。
3. 断言集合，意思是预先写好的判定条件，例如“商品价格低于 50 美元”“颜色为黑色”“最终提交了订单草稿”。

对新手来说，最重要的一句是：成功不等于“看起来像成功”，而等于“所有断言都通过”。

把单次任务是否成功记为随机变量 $X_u^s$。这里 $s$ 表示场景，$u$ 表示用户输入或用户变体。它只有两个值：成功记 1，失败记 0，所以可写成：

$$
X_u^s \sim \mathrm{Bernoulli}(p)
$$

整体基准关心的是平均成功概率：

$$
p_{\text{success}}=\mathbb{E}_{s,u}[X_u^s]
$$

白话解释：从所有场景和所有用户输入中平均来看，这个系统有多大概率把事情做成。

边界也要说清楚。多 Agent 评估通常不只评最终答案，还会覆盖三类对象：

| 评估对象 | 具体内容 | 作用 |
|---|---|---|
| 行为轨迹 | 每一步消息、计划、工具调用 | 判断协作是否绕路 |
| 模拟用户与动作 | 统一复现场景交互 | 保证不同版本在同样条件下比较 |
| LLM 评审器 | 对语义类断言自动打分 | 处理“字面不同但意思正确”的情况 |

这里的 LLM 评审器可以理解为“自动阅卷老师”，它不负责执行任务，只负责根据评分规则判定输出是否满足语义要求。

因此，本文讨论的边界不是“开放世界里万能评价 Agent”，而是“在固定场景、固定断言、固定模拟器下，对不同多 Agent 系统做可复现对比”。

---

## 核心机制与推导

评估的核心机制是先把“成功”定义成断言集合，再用模拟器和评审器稳定地复现执行过程。

一个典型流程可以写成：

```text
场景定义
  ↓
用户模拟器生成输入
  ↓
多 Agent 系统执行
  ↓
动作模拟器返回环境反馈
  ↓
记录消息/Token/延迟/路由
  ↓
断言检查 + LLM 评审
  ↓
产出 success、成本、效率、稳定性指标
```

这里有两个关键量。

第一是成功率，刚才已经给出。它解决“做没做成”。

第二是效率比。效率比用于衡量系统有没有绕远路，可以定义为：

$$
\text{Efficiency Ratio}=\frac{\text{理想动作数}}{\text{实际动作数}}
$$

如果一个任务理想情况下只要 5 步，但系统实际走了 10 步，那么：

$$
\text{Efficiency Ratio}=\frac{5}{10}=0.5
$$

白话解释：系统只达到了 50% 的动作效率，多走了一倍路径。若每步都会产生额外推理和上下文，那么 Token 与延迟通常也会近似放大。

### 玩具例子

任务：让“规划 Agent”和“执行 Agent”协作，在一个商品库里找出最低价的黑色键盘。

理想路径：
1. 规划 Agent 确定筛选条件
2. 执行 Agent 搜索商品
3. 执行 Agent 过滤颜色
4. 执行 Agent 比较价格
5. 返回答案

实际路径如果变成 10 步，常见原因是：
- 规划 Agent 反复重写计划
- 搜索结果已足够，但执行 Agent 仍重复搜索
- 两个 Agent 对“最低价”是否含运费理解不一致

这时即使最终答对，效率比依然只有 0.5。工程上这往往意味着：
- Token 多耗接近一倍
- 延迟显著增加
- 出错传播概率更高，因为每多一步就多一次失败机会

### 真实工程例子

在企业场景里，AWS 公开的断言式 Pipeline 思路是：先从多个业务域收集场景描述、用户输入和断言，再由用户模拟器与动作模拟器驱动多 Agent 执行，最后用规则和 LLM 评审器共同判定用户侧目标与系统侧目标是否达成，同时记录延迟、Token、路由成本等运行信息。

这类 Pipeline 的价值不是“测一次分数”，而是让同一批场景能够反复回放，用来比较：
- 单 Agent vs 多 Agent
- 改路由策略前后
- 不同模型版本
- 是否引入专门化子 Agent

所以，多 Agent 评估不是简单排行榜，而是一个可回放、可定位、可对比的实验系统。

---

## 代码实现

工程上最容易落地的方式，是像维护 `posts.json` 一样维护 `scenarios.json`。每个场景都是一条结构化记录，包含输入、期望断言和理想步数。

```json
{
  "id": "buy-keyboard-001",
  "task": "找到最便宜的黑色机械键盘",
  "user_input": "帮我挑一个预算 50 美元以内的黑色机械键盘",
  "ideal_steps": 5,
  "assertions": [
    {"type": "field_eq", "field": "color", "value": "black"},
    {"type": "field_lte", "field": "price", "value": 50},
    {"type": "semantic", "rule": "final answer mentions the selected product and price"}
  ]
}
```

然后评测主循环做四件事：加载场景、运行 Agent、记录轨迹、打分。

```python
from dataclasses import dataclass

@dataclass
class RunStats:
    success: int
    messages: int
    tokens: int
    latency_s: float
    ideal_steps: int
    actual_steps: int

def efficiency_ratio(ideal_steps: int, actual_steps: int) -> float:
    assert ideal_steps > 0
    assert actual_steps > 0
    return ideal_steps / actual_steps

def score_run(stats: RunStats) -> dict:
    ratio = efficiency_ratio(stats.ideal_steps, stats.actual_steps)
    return {
        "success": stats.success,
        "messages": stats.messages,
        "tokens": stats.tokens,
        "latency_s": stats.latency_s,
        "efficiency_ratio": round(ratio, 4),
    }

# 玩具样例：理想 5 步，实际 10 步
sample = RunStats(
    success=1,
    messages=12,
    tokens=8000,
    latency_s=3.2,
    ideal_steps=5,
    actual_steps=10,
)

result = score_run(sample)

assert result["success"] == 1
assert result["efficiency_ratio"] == 0.5
assert result["messages"] == 12
assert result["tokens"] == 8000
assert abs(result["latency_s"] - 3.2) < 1e-9

print(result)
```

更接近真实系统的伪代码如下：

```text
load scenarios
for scenario in scenarios:
    state = simulator.reset(scenario)
    trace = []
    start = now()

    while not state.done:
        agent_msg = controller.dispatch(state)
        tool_result = action_simulator.step(agent_msg)
        trace.append({
            "agent_msg": agent_msg,
            "tool_result": tool_result,
            "tokens": usage(agent_msg),
            "latency_ms": elapsed_step(),
            "route": route_name(agent_msg)
        })
        state = update_state(state, tool_result)

    judge_result = judge_assertions(
        final_output=state.final_output,
        trace=trace,
        assertions=scenario.assertions
    )

    emit_metrics(
        success=judge_result.all_passed,
        message_rounds=len(trace),
        token_cost=sum(x.tokens for x in trace),
        latency=now() - start,
        efficiency=scenario.ideal_steps / max(len(trace), 1)
    )
```

这和 AgentBench、BOLAA 常见的源码思路是一致的：场景驱动循环、控制器分发、环境反馈、统一打分。区别只在于环境类型不同。AgentBench 更偏多种交互环境上的统一 benchmark，BOLAA 更强调控制器编排多个专门化 Agent。

---

## 工程权衡与常见坑

多 Agent 评估最大的问题不是“不会算平均数”，而是系统天然不稳定。LLM 输出有随机性，路径长了以后误差会累积，多个 Agent 之间还会出现涌现行为。涌现行为的意思是：单个 Agent 看不出问题，但放在一起会出现新的整体行为，比如无限来回确认。

某些企业实践会在云上搭评测 Pipeline，收集约 90 个场景，由 LLM 评判器做自动化评分，再对两个版本做 3 到 5 次重复 A/B 测试，最后用非参数检验判断差异是否可靠。这么做的原因很简单：单次跑分不可信，必须比较分布。

| 常见坑 | 典型表现 | 规避手段 |
|---|---|---|
| 非确定性太强 | 同一输入每次得分波动大 | 固定场景、固定模拟器、同任务重复多次 |
| 只看成功率 | 成功率升了，但 Token 暴涨 | 同时报告成功率、成本、延迟、P95 |
| 路径过长 | 两个 Agent 相互确认、重复检索 | 增加理想步数基线与效率比监控 |
| 错误传播 | 上游路由错一次，下游全错 | 分层记录中间状态与路由决策 |
| LLM 评审漂移 | 评分标准随提示词变化 | 保留评审 prompt、理由、样本抽检 |
| 数据污染 | 基准被训练集见过，分数虚高 | 区分开发集、测试集与私有回归集 |

A/B 检验可以按下面步骤做：

1. 让版本 A 和版本 B 跑完全相同的场景集。
2. 每个场景重复 3 到 5 次，得到成功率、Token、延迟分布。
3. 比较中位数、分位数，而不是只看平均值。
4. 对成功率差异或成本差异使用 Mann-Whitney U 检验或 bootstrap 置信区间。
5. 如果差异显著，再决定是否发布。

这里用非参数检验，是因为多 Agent 指标经常不是正态分布。比如延迟常常右偏，少数超慢任务会把均值拉歪。

---

## 替代方案与适用边界

不同方案适合的目标不同，关键不是谁“更先进”，而是谁更匹配你的任务形态。

| 方案 | 更适合什么场景 | 优点 | 局限 |
|---|---|---|---|
| AgentBench 类通用基准 | 想横向比较模型或代理能力 | 环境覆盖广，适合公开对比 | 贴近你业务的程度有限 |
| 断言式 Pipeline | 企业内部回归、发布前验收 | 可复现、可自动化、便于接 CI | 前期要写场景、模拟器、断言 |
| BOLAA 式多架构评测 | 研究多 Agent 编排收益 | 能比较控制器和专门化 Agent 组合 | 对工程接入要求更高 |
| 单 Agent 离线评测 | 任务短、流程稳定 | 成本低，搭建快 | 很难暴露协作问题 |

对初学者，可以这样理解：

- AgentBench 更像“标准化考试”，重点是看模型在多种交互环境里的通用能力。
- AWS 式断言 Pipeline 更像“企业内部验收系统”，重点是让同一业务场景可重复回放、可自动打分。
- BOLAA 更像“多岗位协作实验台”，重点是比较不同编排架构是否真的带来收益。

适用边界也要注意。如果你的任务本质上只有一步检索或一步分类，多 Agent 往往不值得，评估体系也没必要搞得太重。只有当任务具备以下特征时，多 Agent 评估才真正必要：
- 任务需要分工，如规划、检索、执行、审查
- 任务链路长，容易出现错误传播
- 成本和延迟对生产可用性有直接影响
- 你需要频繁比较版本、模型、路由策略

换句话说，评估体系的复杂度应该跟系统复杂度匹配。不要用多 Agent 的方法去评一个本来单 Agent 就能稳定解决的问题。

---

## 参考资料

下面这些资料分别对应“公开 benchmark”“多 Agent 编排”“企业评测 Pipeline”“工程评估方法”四个方向，适合按需深入：

- AgentBench: Evaluating LLMs as Agents. arXiv / GitHub: https://github.com/THUDM/AgentBench
- AgentBench 论文索引页: https://arxiv.org/abs/2308.03688
- BOLAA: Benchmarking and Orchestrating LLM-augmented Autonomous Agents. GitHub: https://github.com/salesforce/BOLAA
- BOLAA Workshop 页面: https://iclr.cc/virtual/2024/22222
- Towards Effective GenAI Multi-Agent Collaboration: Design and Evaluation for Enterprise Applications. Amazon Science: https://www.amazon.science/publications/towards-effective-genai-multi-agent-collaboration-design-and-evaluation-for-enterprise-applications
- 上述论文 PDF: https://assets.amazon.science/bc/1e/1202475d44a6842a065dd4adf9b9/towards-effective-genai-multi-agent-collaboration-design-and-evaluation-for-enterprise-applications.pdf
- Microsoft Multi-agent Reference Architecture: Evaluation: https://microsoft.github.io/multi-agent-reference-architecture/docs/evaluation/Evaluation.html
- Grizzly Peak Software, Agent Evaluation Methods and Benchmarks: https://www.grizzlypeaksoftware.com/library/agent-evaluation-methods-and-benchmarks-9pkzw04n
