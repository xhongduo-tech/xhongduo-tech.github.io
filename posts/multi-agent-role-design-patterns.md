## 核心结论

多 Agent 系统的角色分工，本质上是在设计一张“谁负责什么、谁不能做什么、信息怎么传、谁有最终裁决权”的协作图。这里的 Agent 可以先理解成“带特定职责和提示词的智能体进程”，不是人格，也不是抽象概念，而是可调用、可约束、可验证的软件组件。

在代码生成任务里，常见的四类分工模式是：

| 模式 | 一句话定义 | 典型角色 | 适合任务 |
| --- | --- | --- | --- |
| 专家型 | 按领域拆分，每个 Agent 只处理自己擅长的部分 | 算法专家、API 专家、测试专家 | 多约束、多知识域任务 |
| 流水线型 | 按步骤串起来，上一步输出喂给下一步 | 读题、规划、编码、调试 | 有明显阶段顺序的任务 |
| 辩论型 | 多个 Agent 并行给方案，再互相质疑后汇总 | 正方、反方、裁判 | 高风险、易出错推理任务 |
| 层级型 | 一个上层负责调度，下层只执行子任务 | Supervisor、Worker、Critic | 多任务路由、复杂流程控制 |

工程上最重要的结论不是“Agent 越多越强”，而是“边界越清楚，系统越稳定”。角色分工的收益主要来自三件事：先分解任务，再限制越权，最后对中间结果做验证。

公开结果已经说明结构设计会放大准确率。Blueprint2Code 把代码生成拆成 Preview、Blueprint、Coding、Debugging 四个阶段，在 GPT-4o 上 HumanEval Pass@1 为 96.3%，直接生成基线为 80.1%；在 GPT-4o-mini 上，同一框架为 89.1%，直接生成为 84.7%。这说明“先规划再编码再修复”的流水线，对复杂任务比“单次直接出答案”更稳。题设给出的专家型 85% 对辩论型 91%，也符合这个方向：纠错回路更强的拓扑，通常在复杂代码任务上更有优势。

---

## 问题定义与边界

多 Agent 不是把一个问题交给多个模型就结束。真正的问题是：如何让“任务分解、角色匹配、结果融合”形成闭环，而不是形成新的混乱。

先把边界说清楚。本文讨论的是“面向代码生成或软件任务的多 Agent 分工设计”，重点不是底层模型训练，而是上层协作架构。也就是说，我们关注的是：

1. 角色怎么定义。
2. 角色之间怎么交接。
3. 什么情况下需要争论，什么情况下只需要串行执行。
4. 如何防止角色越权、重复劳动和错误传播。

这里有一个常见误解：多 Agent 不是并发越多越好。对于简单、边界清楚、验收标准明确的任务，一个 Direct 单 Agent 往往已经足够。比如“写一个把摄氏度转华氏度的 Python 函数”，再加 Supervisor、Critic、Debater，多数时候只是在增加延迟和成本。

真正需要多 Agent 的，是这类任务：

| 任务特征 | 为什么单 Agent 容易出错 | 更合适的分工 |
| --- | --- | --- |
| 需求长、约束多 | 容易漏条件 | 层级型或流水线型 |
| 方案空间大 | 容易早早锁定错误路线 | 辩论型 |
| 涉及多知识域 | 一个提示词难同时覆盖 | 专家型 |
| 需要强审计 | 需要知道错在何处 | 流水线型或层级型 |

一个新手能立刻理解的玩具例子是：让系统“写一个判断回文字符串的函数并带测试”。如果只有一个 Agent，它可能直接给代码，但忘记空字符串、大小写、空格处理。若改成四段流水线：

1. Preview Agent 先读需求，提炼边界。
2. Blueprint Agent 先列出案例。
3. Coding Agent 再写实现。
4. Debug Agent 再执行测试并指出失败样例。

这时错误更容易被前一阶段暴露，而不是等到最后统一爆炸。

Loadsys 的文章把大量失败归因到规划阶段，核心不是模型弱，而是任务定义弱：上下文不够、范围模糊、验收标准缺失。对多 Agent 系统尤其如此，因为一旦第一个 Agent 把含糊信息传下去，后面所有 Agent 都会把错误当成前提。

因此，角色定义至少要覆盖下面五个维度：

| 维度 | 要回答的问题 | 典型角色 |
| --- | --- | --- |
| 当前任务 | 你这轮到底负责什么 | 所有角色 |
| 前置条件 | 你依赖什么输入才能开始 | Worker、Coder |
| 允许工具 | 你能读什么、写什么、调用什么 | Worker、Tool Agent |
| 输出格式 | 你必须产出什么结构 | Planner、Critic |
| 不可越权 | 哪些决定不是你做的 | Supervisor、Worker |

---

## 核心机制与推导

角色分工为什么有效，可以从“降低联合搜索空间”来理解。联合搜索空间，白话说，就是“系统需要同时猜对的东西太多”。单 Agent 要同时理解需求、设计方案、写代码、找 bug、判断边界条件，任何一步错了，都会污染后续推理。多 Agent 的价值，是把一个大搜索问题拆成多个局部搜索问题。

### 1. 流水线型为什么能提高稳定性

流水线型的关键不是“串行”，而是“接口化”。上一阶段必须输出明确结构，下一阶段只能在这个结构上继续工作。

例如 Blueprint2Code 的四段式流程，本质是：

- Preview 负责压缩题意，识别输入输出和隐藏约束。
- Blueprint 负责形成可执行计划。
- Coding 负责按计划实现。
- Debugging 负责把失败样例重新反馈给实现层。

这等价于把一次生成拆成四次受限生成。每一层都在减少自由度，从而减少跑偏概率。

### 2. 辩论型为什么有时比专家型更强

辩论型不是让 Agent “聊天”，而是把错误暴露出来。它利用了一个不对称事实：很多任务里，“找错”比“首次答对”更容易。

可以把辩论型最后的融合过程写成加权投票：

$$
\hat{y}=\arg\max_{y\in\mathcal{Y}}\sum_{i=1}^{N}\mathbb{I}\{a_i^{(R)}=y\}\cdot w_{E,i}\cdot w_{conf,i}
$$

其中：

- $a_i^{(R)}$ 是第 $i$ 个 Agent 在第 $R$ 轮后的答案。
- $w_{E,i}$ 是专长权重，白话说就是“这个角色在这个问题上该不该更有话语权”。
- $w_{conf,i}$ 是置信权重，白话说就是“它自己对当前答案有多确定”。

这个公式的直觉很简单：不是票数最多就赢，而是“由更相关、且更有把握的角色投出的票”更重。

一个玩具例子：

- Agent A 是算法专家，选方案 X，置信度 0.9。
- Agent B 是测试专家，选方案 Y，置信度 0.8。
- Agent C 是性能专家，选方案 X，置信度 0.6。

如果专长权重分别是 1.0、0.9、0.7，那么 X 的总分就是 $1.0\times0.9+0.7\times0.6=1.32$，Y 的总分是 $0.9\times0.8=0.72$。最终系统选 X，但这个决策是可解释的，不是黑箱拍脑袋。

### 3. 层级型为什么适合真实工程

层级型里，Supervisor 可以理解成“调度器”，白话说就是只负责派单，不直接干活。它最重要的职责是：

- 解析用户意图。
- 选择下游 Worker。
- 记录上下文。
- 检查是否需要再次分解或进入审查环节。

Oracle 的实践建议很明确：Supervisor 不应该自己执行领域任务，Worker 也不应该自行路由。这个限制非常关键，因为一旦 Supervisor 又调度又编码，或者 Worker 又执行又改需求，系统就失去可测性。

### 4. 专家型的本质不是“知识多”，而是“损失函数不同”

算法专家关心复杂度，API 专家关心接口契约，测试专家关心边界覆盖，安全专家关心输入校验。这些角色之所以要分开，不是因为它们知道不同知识，而是因为它们优化的是不同目标。

真实工程例子：让系统生成“一个带 JWT 鉴权、限流、审计日志的用户登录接口”。如果只有一个通用 Coding Agent，它容易把“功能能跑”当成成功。但专家型系统会天然拆成：

- 安全专家检查 token 生命周期、刷新策略、暴力破解防护。
- API 专家检查状态码、字段契约、幂等语义。
- 测试专家补齐错误密码、过期 token、并发登录等边界场景。
- 性能专家评估 Redis 限流和数据库写日志的热点。

这时系统不是更“聪明”，而是看问题的角度更完整。

---

## 代码实现

下面给一个最小可运行示例，用 Python 模拟“Supervisor + Specialist + Debate Aggregator”的流程。它不依赖真实模型，只演示角色边界和加权融合逻辑。

```python
from dataclasses import dataclass

@dataclass
class Vote:
    agent: str
    answer: str
    expertise_weight: float
    confidence_weight: float

def weighted_vote(votes):
    scores = {}
    for v in votes:
        scores.setdefault(v.answer, 0.0)
        scores[v.answer] += v.expertise_weight * v.confidence_weight
    best = max(scores, key=scores.get)
    return best, scores

def supervisor_route(task: str) -> str:
    task = task.lower()
    if "api" in task or "http" in task:
        return "specialist"
    if "optimize" in task or "tradeoff" in task:
        return "debate"
    return "pipeline"

def run_pipeline(task: str):
    preview = {
        "goal": "实现函数",
        "constraints": ["通过测试", "处理边界条件"],
    }
    blueprint = {
        "steps": ["定义输入输出", "列边界样例", "写实现", "补测试"]
    }
    coding = {
        "result": "def solve(x): return x[::-1]"
    }
    debug = {
        "issues": ["若输入不是字符串会失败"],
        "status": "needs_revision"
    }
    return {"preview": preview, "blueprint": blueprint, "coding": coding, "debug": debug}

def run_specialist():
    return [
        Vote("api_expert", "方案A", 1.0, 0.8),
        Vote("security_expert", "方案A", 1.1, 0.9),
        Vote("test_expert", "方案B", 0.9, 0.7),
    ]

def run_debate():
    return [
        Vote("proposer", "方案B", 0.9, 0.95),
        Vote("critic", "方案A", 1.0, 0.75),
        Vote("judge", "方案B", 1.2, 0.85),
    ]

# toy example
mode = supervisor_route("Optimize an API handler with clear tradeoff discussion")
assert mode == "specialist" or mode == "debate"

pipeline_result = run_pipeline("write palindrome checker")
assert pipeline_result["debug"]["status"] == "needs_revision"

best_specialist, specialist_scores = weighted_vote(run_specialist())
assert best_specialist == "方案A"
assert specialist_scores["方案A"] > specialist_scores["方案B"]

best_debate, debate_scores = weighted_vote(run_debate())
assert best_debate == "方案B"
assert debate_scores["方案B"] > debate_scores["方案A"]
```

这个例子有两个重点。

第一，Supervisor 只决定走哪种协作拓扑，不直接给最终答案。第二，最终融合不是简单少数服从多数，而是按角色权重和置信度加权。

如果把它写成更接近真实 LLM 系统的伪代码，流程通常是这样：

```python
def solve_with_agents(user_request):
    plan = preview_agent(user_request)
    topology = supervisor.select_topology(plan)

    if topology == "pipeline":
        blueprint = blueprint_agent(plan)
        code = coding_agent(blueprint)
        report = debug_agent(code, blueprint)
        return supervisor.finalize(code, report)

    if topology == "specialist":
        algo = algorithm_agent(plan)
        api = api_agent(plan)
        tests = test_agent(plan)
        return supervisor.merge([algo, api, tests])

    if topology == "debate":
        round1 = [agent.respond(plan) for agent in debaters]
        round2 = [agent.critique(round1) for agent in debaters]
        return judge.weighted_consensus(round2)
```

真实工程里，最关键的不是提示词写得多华丽，而是每个 Agent 的输入输出结构要固定。建议至少固定成 JSON 风格字段，例如：

| 角色 | 输入 | 输出 |
| --- | --- | --- |
| Preview | 原始任务、上下文 | goal、constraints、unknowns |
| Blueprint | Preview 输出 | steps、risks、tests |
| Coding | Blueprint 输出 | code、assumptions |
| Debug | code、tests | failures、fixes、status |
| Critic | 任一候选方案 | defects、severity、evidence |

这样做的原因很简单：下游 Agent 读的是结构，不是散文。

---

## 工程权衡与常见坑

第一个坑是角色边界模糊。模糊的后果不是“风格不好”，而是系统行为不可预测。Supervisor 如果开始自己写代码，Worker 如果开始自行改需求，Review Agent 如果既提问题又偷偷修代码，整个链路会失去审计能力。

一个可直接复用的角色模板是：

| 字段 | 模板内容 |
| --- | --- |
| 角色定义 | 你是 `X`，只负责 `Y` |
| 当前任务 | 本轮你只需要完成 `task` |
| 前置条件 | 仅基于 `inputs` 工作 |
| 允许工具 | 你只能使用 `tools` |
| 输出格式 | 你必须输出 `schema` |
| 禁止事项 | 你不能做 `forbidden_actions` |
| 失败处理 | 输入不足时返回 `missing_info`，不要猜 |

第二个坑是把所有问题都升级成辩论。辩论型的成本高，延迟高，还可能产生“多数压制少数”的假共识。Emergent Mind 总结的研究里也提到，很多场景下静态投票已经拿走了大部分收益，增加过多轮辩论未必继续提升正确率。

第三个坑是中间状态不做验证。多 Agent 最大的风险不是最终答案错，而是“错误的中间状态被当成事实继续传播”。例如 Blueprint 阶段误判“允许修改数据库结构”，后续 Coding 和 Debug 都会在一个错误前提上工作。解决方法不是让模型更谨慎，而是在每个交接点加契约校验。

第四个坑是工具权限过宽。ctimes.tech 把常见失败总结为四类：意图模糊、工具误用、记忆污染、循环失控。这些问题在多 Agent 系统中更严重，因为任何一个角色的越权调用都可能污染共享上下文。

一个实用的排障模板可以直接这样写：

| 排障项 | 内容 |
| --- | --- |
| 当前 Prompt | 粘贴当前角色完整提示词 |
| 错误场景 | 给出一条会失败的具体输入 |
| 当前错误输出 | 粘贴真实错误行为 |
| 期望输出 | 说明理想行为 |
| 约束冲突 | 标出可能互相矛盾的规则 |
| 建议修改 | 只改必要字段，不整体重写 |

Oracle 给出的做法很直接：把“当前 prompt、错误例子、期望输出”一起交给另一个 LLM，让它指出冲突、遗漏和歧义。这相当于给提示词做一次 code review。

---

## 替代方案与适用边界

不是所有任务都要上四种模式。一个更实用的做法，是把协作强度看成一个连续谱：

| 方案 | 协调成本 | 准确率潜力 | 适用场景 |
| --- | --- | --- | --- |
| Direct | 低 | 低到中 | 简单、单步、低风险 |
| 加权投票 | 低到中 | 中 | 客观问答、轻量校验 |
| 流水线 | 中 | 中到高 | 步骤清晰的代码任务 |
| 专家型 | 中到高 | 高 | 多知识域工程任务 |
| 辩论型 | 高 | 高 | 高风险、易分歧推理任务 |
| 深度共识 | 很高 | 不稳定 | 只有高价值任务才值得 |

一个新手容易忽略的事实是：多 Agent 的收益并不线性。简单 FAQ 查询，用三个 Agent 各自回答再做加权投票，通常就足够稳定。再往上加多轮辩论，多半只是增加成本。只有当问题包含明显对抗观点、多阶段推理或高失败代价时，辩论型才值得。

回到代码生成场景，可以这样选：

- 写单个函数、边界明确：Direct 或轻量流水线。
- 写完整模块、需要设计接口：流水线型。
- 改跨模块系统、涉及安全与性能：专家型。
- 需求含糊、方案争议大、错误代价高：辩论型或层级 + 辩论混合。

因此，角色分工设计的核心不是追求最复杂的拓扑，而是让拓扑复杂度和任务复杂度匹配。协作强度过低会漏错，协作强度过高会拖慢系统。

---

## 参考资料

- Cognition Commons, [Multi-Agent Coordination](https://cognitioncommons.org/research/multi-agent-coordination)
- Frontiers in Artificial Intelligence, [Blueprint2Code: a multi-agent pipeline for reliable code generation via blueprint planning and repair](https://www.frontiersin.org/articles/10.3389/frai.2025.1660912/full)
- Emergent Mind, [Multiagent Debate Framework](https://www.emergentmind.com/topics/multiagent-debate-framework)
- Emergent Mind, [Multi-Agent Debate Strategies](https://www.emergentmind.com/topics/multi-agent-debate-mad-strategies)
- Loadsys, [82% of Agent Failures Start Before the First Line of Code](https://www.loadsys.com/blog/ai-coding-agent-failure-rate/)
- Oracle Fusion CoE, [Best Practices for Prompts in AI Agent Studio](https://blogs.oracle.com/fusioncoe/best-practices-for-prompts-in-ai-agent-studio)
- ctimes.tech, [AI Agent Failures: Prompt Design Fixes 4 Common Issues](https://ctimes.tech/en/2026/01/08/ai-agent-failures-prompt-design-fixes-4-common-issues/)
