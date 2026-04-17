## 核心结论

红队测试不是“多跑几组攻击样例”，而是把系统当成真实会被攻击的目标，先画清楚攻击面，再按目标设计攻击路径，执行后把漏检、误判、响应迟缓等结果回写到规则、流程和模型更新里，形成闭环。

这里的“攻击面”是系统所有可能被外部影响的入口，白话讲，就是攻击者能碰到的一切输入点和相邻系统。对智能体系统来说，它通常不只包括用户提示词，还包括 API 参数、检索文档、工具返回结果、插件回调、消息系统、工单系统和人工客服接口。

对初学者最重要的判断标准只有一个：一次红队是否真正改变了后续防御。如果结果只停留在报告里，没有进入检测规则、评测集、对齐训练或响应流程，它更像一次展示，不像工程化安全流程。

快照式红队和持续红队的差别，不在“谁更高级”，而在节奏与闭环密度。前者适合上线前验收、季度审计、合规场景；后者适合云服务、Agent 平台、提示模板频繁变化的产品。

| 模式 | 测试频率 | 适用场景 | 反馈形式 |
| --- | --- | --- | --- |
| 快照红队 | 间歇，常见为季度或版本发布前 | 典型部署、上线验收、合规检查 | 演练报告、风险清单、整改建议 |
| 持续红队 | 滚动执行，按周或按变更触发 | 云服务、Agent 平台、快速迭代产品 | 趋势指标、规则自动更新、样本持续回灌 |

一个新手版定义可以记成一句话：把系统所有入口画成图，按图依次用自动脚本和人工尝试突破，最后写清楚“哪里没拦住、为什么没拦住、下一轮怎么拦”。

---

## 问题定义与边界

红队测试的目标不是证明“系统绝对安全”，而是在明确边界内，尽可能逼近真实攻击者的行为，检验三件事：

1. 能不能打进去。
2. 打进去后能不能被检测到。
3. 被检测到后，团队和系统能不能及时响应。

这和传统渗透测试不同。渗透测试更偏向“找漏洞”；红队测试更偏向“验证整条攻击链能否成立，以及防守链是否有效”。

先看一个玩具例子。假设一个问答型智能体只有三个输入来源：

```text
[用户提示]
   |
   v
[Agent 编排器] -----> [外部 API]
   |
   v
[检索文档库]
```

这时攻击面至少包括三类：

- 用户提示：直接 prompt injection，即恶意提示注入，白话讲就是把“坏指令”伪装成正常输入。
- API 返回：上游系统被污染后，把恶意字段塞回模型上下文。
- 外部文档：检索到的网页、PDF、知识库条目里夹带隐藏指令。

边界必须先写明，否则红队结果不可比。一个简单的边界表如下：

| 维度 | 可攻范围 | 不可攻范围 |
| --- | --- | --- |
| 资产边界 | 测试环境中的 Agent、检索库、工具调用链 | 生产数据库、真实用户账户、第三方付费接口 |
| 攻击方式 | 自动脚本、人工构造样本、社工演练、跨系统链路验证 | 破坏性攻击、真实数据外泄、超出授权的外部扫描 |
| 评估目标 | 漏检率、误报率、攻击成功率、响应时间 | 法务裁定、业务 KPI 本身 |
| 证据采集 | 日志、模型输出、工具调用记录、告警记录 | 未经批准的终端取证 |

如果你不先写这张表，后面几乎一定会出现两种误判：

- 把“没测到”误当成“没有风险”。
- 把“单点没问题”误当成“整条链路安全”。

对初学者更实用的做法，是把红队闭环拆成四段：

```text
建模 -> 攻击 -> 复盘 -> 更新
```

“更新”要明确落点。最常见的两个位置是：

- 写入检测规则的位置：WAF/提示拦截器/工具调用策略/告警规则/审计看板。
- 需要人工验证的边界：社工链路、跨团队审批流、异常工单处理、模型输出语义判定。

---

## 核心机制与推导

红队流程之所以需要系统化，不是因为步骤多，而是因为攻击是多轮决策问题。多轮决策的意思是：前一轮看到什么，会影响下一轮打哪里、怎么打。

可以把攻击路径抽象成：

$$
P_t = f(S_t, A_t, O_t)
$$

其中：

- $S_t$ 是当前攻击面状态，白话讲，就是此刻系统暴露了哪些入口、权限和上下游关系。
- $A_t$ 是可选策略集合，白话讲，就是当前能用哪些打法，例如自动模糊测试、恶意文档投毒、人工社工、跨工具跳转。
- $O_t$ 是上一轮观测，白话讲，就是系统刚才给出的响应、日志、告警、失败信息和副作用。
- $P_t$ 是下一步攻击计划。

这个式子说明一个事实：红队不是预先写好 100 条攻击再机械执行，而是要根据观测动态改计划。

执行后，反馈回路可以近似写成：

$$
D_{t+1} = \min\left(1,\; D_t + \alpha \cdot \mathrm{Severity}(F_t)\right)
$$

其中：

- $D_t$ 是当前检测覆盖率，白话讲，就是系统能拦住或识别多少高风险样本。
- $F_t$ 是本轮发现的问题样本，例如漏检、误判、告警延迟、工具误调用。
- $\mathrm{Severity}(F_t)$ 是严重性评分，越危险、越可复现、越容易扩散，分值越高。
- $\alpha$ 是更新强度，表示团队把发现真正转化为规则或训练样本的效率。

玩具例子如下。系统有三个输入渠道：用户提示、外部 API、外部文档；两类攻击策略：自动模糊测试和人工社工；初始检测覆盖率为 60%。五次攻击后，发现三个漏检，经过规则补丁和评测集补充，覆盖率上升到 76%。

| 回合 | 输入渠道 | 策略 | 结果 | 严重性增量 | 覆盖率 |
| --- | --- | --- | --- | --- | --- |
| 1 | 用户提示 | 自动模糊 | 被拦截 | 0.00 | 0.60 |
| 2 | 外部文档 | 自动模糊 | 漏检 | 0.08 | 0.68 |
| 3 | API 返回 | 人工构造链路 | 漏检 | 0.04 | 0.72 |
| 4 | 用户提示 | 人工社工 | 漏检 | 0.04 | 0.76 |
| 5 | 外部文档 | 自动复测 | 被拦截 | 0.00 | 0.76 |

这个例子表达的不是“数学上一定这样涨”，而是工程上要把每次发现量化，避免复盘只停在口头结论。

真实工程例子可以看 NIST CAISI 于 2025 年 1 月 17 日发布的 AgentDojo 评估博客。它在 Workspace、Travel、Slack、Banking 四类 Agent 环境里测试劫持风险，并通过与英国 AI Security Institute 协作的红队演练，把某组模型上的攻击成功率从基线攻击的 11% 拉高到新攻击的 81%。这个案例说明两点：

- 新模型对旧攻击更稳，不代表对新攻击就稳。
- 红队必须持续适配目标系统，而不是永远复用旧题库。

---

## 代码实现

工程上建议把实现拆成三层：

- 攻击场景描述层：定义入口、资产、权限、禁止操作、成功条件。
- 任务队列执行层：把自动攻击、人工复核、复测任务统一排队。
- 反馈评分层：对每个结果打分，并决定写入规则、评测集还是训练数据。

下面是一个可运行的最小 Python 例子，模拟“5 次攻击后覆盖率从 0.60 提升到 0.76”的过程：

```python
from dataclasses import dataclass

@dataclass
class AttackResult:
    channel: str
    strategy: str
    detected: bool
    severity: float  # 仅对漏检样本计分

def plan(state, actions, observation):
    # 简化版 P_t = f(S_t, A_t, O_t)
    if observation.get("last_miss_channel"):
        return {"channel": observation["last_miss_channel"], "strategy": "retest"}
    for ch in state["channels"]:
        if ch not in observation["tested"]:
            return {"channel": ch, "strategy": actions[ch][0]}
    return {"channel": state["channels"][0], "strategy": actions[state["channels"][0]][0]}

def update_coverage(detection_coverage, result, alpha=1.0):
    # 简化版 D_{t+1} = min(1, D_t + alpha * Severity(F_t))
    if result.detected:
        return detection_coverage
    return min(1.0, detection_coverage + alpha * result.severity)

state = {"channels": ["prompt", "document", "api"]}
actions = {
    "prompt": ["fuzz", "social"],
    "document": ["fuzz"],
    "api": ["chain_abuse"],
}

scripted_results = [
    AttackResult("prompt", "fuzz", True, 0.00),
    AttackResult("document", "fuzz", False, 0.08),
    AttackResult("api", "chain_abuse", False, 0.04),
    AttackResult("prompt", "social", False, 0.04),
    AttackResult("document", "retest", True, 0.00),
]

coverage = 0.60
observation = {"tested": set(), "last_miss_channel": None}
history = []

for i, result in enumerate(scripted_results, start=1):
    task = plan(state, actions, observation)
    coverage = update_coverage(coverage, result)
    history.append((i, task["channel"], task["strategy"], result.detected, round(coverage, 2)))

    observation["tested"].add(result.channel)
    observation["last_miss_channel"] = None if result.detected else result.channel

assert round(coverage, 2) == 0.76
assert sum(1 for _, _, _, detected, _ in history if not detected) == 3

print(history)
```

这段代码对应的机制如下：

- `state` 对应攻击面状态 $S_t$。
- `actions` 对应策略集合 $A_t$。
- `observation` 对应上一轮观测 $O_t$。
- `plan()` 表示根据状态和观测生成下一步攻击计划。
- `update_coverage()` 表示把漏检样本转成覆盖率改进。
- `history` 相当于统一反馈仓库，后续可接日志平台或评测集。

如果放到真实系统里，还要补四类接口：

- 日志接口：记录 prompt、工具调用、检索结果、告警事件。
- 评分接口：给每条发现打严重性、可复现性、影响范围分数。
- 规则更新接口：把高置信发现写入拦截器、策略引擎或检测规则。
- 数据回灌接口：把高风险样本加入后续评测集或对齐训练集。

一个常见的工程结构是：

```text
场景定义 YAML/JSON -> 调度器 -> 执行器 -> 日志汇聚 -> 评分器 -> 反馈仓库
                                                     |
                                                     v
                                           规则更新 / 评测集更新 / 训练样本更新
```

---

## 工程权衡与常见坑

第一类坑是把一次性红队当成全局结论。快照式红队能回答“这个版本上线前是否存在明显缺口”，但不能回答“未来三个月攻击面变化后是否仍有效”。

第二类坑是过度自动化。自动化脚本擅长覆盖广、复现快，但对复杂社工、跨系统权限继承、审批流绕过这类问题很容易漏掉。白话讲，脚本善于扫平面，不擅长理解组织行为。

第三类坑是只看模型输出，不看工具副作用。对 Agent 来说，最危险的结果常常不是一句不当回答，而是“发了一封邮件”“执行了一条命令”“下载了一段脚本”“把文档共享给了错误的人”。

第四类坑是没有统一评分口径。没有口径时，团队会把“有趣的攻击”误判成“高风险攻击”，导致修复顺序失真。

| 风险 | 典型表现 | 缓解方式 |
| --- | --- | --- |
| 把快照结果当长期结论 | 一次演练后半年不复测 | 按版本、按能力变更触发持续红队 |
| 过度依赖自动化 | 只测 prompt injection，不测社工和跨系统链路 | 预留人工分析时段，蓝红联合复盘 |
| 只看模型文本输出 | 忽略工具调用、副作用和审计链路 | 采集工具调用日志和告警链路 |
| 没有评分标准 | 修复优先级全靠感觉 | 统一严重性、可复现性、影响范围指标 |
| 结果不回写 | 报告发完即结束 | 建立规则更新、评测集更新、训练回灌三个动作 |

一个新手常见错误是：只刷一轮自动化提示注入，发现多数样本被拦住，就下结论“模型已经安全”。这个结论通常不成立，因为它没有覆盖社工、恶意文档、上游 API 污染、重复尝试、跨工具跳转这些真实攻链。

更合理的改法是：

- 自动化先做广覆盖，找到低成本、可批量复现的问题。
- 人工再做深链路，验证复杂流程、角色切换和跨系统攻击。
- 红队、蓝队、产品、合规一起复盘，明确谁改规则、谁改流程、谁改产品。

---

## 替代方案与适用边界

红队不是唯一评估方式，也不应该替代所有安全工作。它适合回答“攻击者是否能在真实约束下完成目标”，但不适合单独承担所有漏洞发现、合规审计和形式化证明任务。

一个便于初学者理解的对比是：

- 红队像模拟真实入侵，关注整条攻击链和防守链。
- 渗透测试像系统查已知漏洞，关注某个范围内的技术缺陷。
- 基准评测像标准化考试，关注一组固定题上的分数。
- 对齐训练像事前教育，关注让模型更少输出危险行为。

| 方法 | 主要目标 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- | --- |
| 红队测试 | 模拟真实攻击与防守闭环 | 能验证检测、响应、跨系统攻链 | 成本高，覆盖永远不完整 | 高风险服务、Agent 平台、关键业务 |
| 渗透测试 | 发现技术漏洞 | 结构清晰，复现标准化 | 不一定覆盖检测和响应 | 上线前检查、合规、固定系统 |
| 基准评测 | 量化模型在固定任务上的表现 | 易比较、易自动化 | 容易被题库过拟合 | 模型迭代、回归测试 |
| 对齐训练 | 降低危险输出概率 | 能系统性改善模型行为 | 不能代替实战验证 | 模型训练和上线前安全收敛 |

它们的关系不是“谁替代谁”，而是分工：

- 低敏感度内部工具，可以先做渗透测试和固定题库评测。
- 有工具调用、外部数据接入、自动执行能力的 Agent，至少需要红队加持续评估。
- 面向公众、高权限、高数据敏感度系统，红队之外还需要审计、权限隔离、最小授权、人工确认和速断机制。

还要明确一个边界：红队与对齐循环能显著降低风险，但不能给出绝对安全保证。Springer 2025 年关于 AI 开发安全实践的研究明确指出，当前 alignment-red teaming 循环仍无法提供可靠的完全安全保证。工程上应把它理解为“持续降低风险的机制”，而不是“拿到认证后永久安全”。

---

## 参考资料

- IBM, “What is red teaming?”：对红队测试的基础定义，强调模拟攻击与安全改进闭环。[https://www.ibm.com/think/topics/red-teaming](https://www.ibm.com/think/topics/red-teaming)
- APXML, “The LLM Red Teaming Lifecycle”：给出面向新手的 planning and scoping 等生命周期分段。[https://apxml.com/courses/intro-llm-red-teaming/chapter-1-foundations-llm-red-teaming/llm-red-teaming-lifecycle/](https://apxml.com/courses/intro-llm-red-teaming/chapter-1-foundations-llm-red-teaming/llm-red-teaming-lifecycle/)
- Wang et al., “A Red Team automated testing modeling and online planning method for post-penetration”, *Computers & Security*, 2024：讨论自动化红队建模、规划、状态与观测驱动的攻击过程。[https://www.sciencedirect.com/science/article/pii/S0167404824002505](https://www.sciencedirect.com/science/article/pii/S0167404824002505)
- Spelda and Stritecky, “Security practices in AI development”, *AI & Society*, 2025：讨论 alignment-red teaming 循环的作用与边界。[https://link.springer.com/article/10.1007/s00146-025-02247-4](https://link.springer.com/article/10.1007/s00146-025-02247-4)
- NIST CAISI, “Technical Blog: Strengthening AI Agent Hijacking Evaluations”, 2025-01-17：AgentDojo 场景、攻击成功率提升与持续适配的重要案例。[https://www.nist.gov/news-events/news/2025/01/technical-blog-strengthening-ai-agent-hijacking-evaluations](https://www.nist.gov/news-events/news/2025/01/technical-blog-strengthening-ai-agent-hijacking-evaluations)
- Accorian, “Red Teaming and Continuous Red Teaming for Cybersecurity”：用于理解一次性红队与持续红队的节奏差异及常见工程问题。[https://www.accorian.com/why-do-you-need-red-teaming/](https://www.accorian.com/why-do-you-need-red-teaming/)
