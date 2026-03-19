## 核心结论

多 Agent 的“共识达成”本质上不是“谁票多听谁的”，而是“如何把多份局部判断，变成一个可复查的整体结论”。这里的 Agent 指能独立给出判断的自动体，可以是不同角色的 LLM、规则系统，或带工具调用能力的审查模块。对代码审查这类任务，简单多数投票实现最便宜，但最容易被共享偏差放大；加权投票能表达“谁更可靠”；Borda 计数能保留排序信息；LLM-as-Judge 则直接让一个裁判模型综合证据打分。

当前更重要的工程结论有两点。第一，结构化验证比裸投票更可靠。已有研究里，传统多数投票与人工结论的一致率大约在 72% 左右，而经过结构化验证的 LLM Judge 能把 hit-agreement 推到 89% 以上；进一步引入 AgentAuditor 这类“推理树审计器”后，整体准确率还能相对多数票再提高约 5%，相对 Judge 再提高约 3%。第二，多 Agent 链路的可靠性不是相加，而常常接近乘积衰减。Lusser’s law 用一句话解释就是：串联系统里只要有一个薄弱环节，整体可靠性会被它拉低。若每个环节可靠性分别是 $r_1,r_2,\dots,r_n$，则系统可靠性近似为：

$$
R_s=\prod_{i=1}^{n} r_i
$$

这意味着“每个 Agent 看起来都还行”并不等于“整个审查系统可靠”。

一个必须记住的玩具例子是：三个候选结论 A、B、C，三名 Agent 的选择分别不同。多数票可能选 A，加权票仍可能选 A，但 Borda 或 Judge 可能因为“B 的证据更完整”而选 B。一个必须记住的真实工程例子是：安全、性能、测试三个审查 Agent 都接入了同一份带偏差的数据源，对一个危险的 `override` 配置文件做出了错误判断，结果 2:1 放行。多数票看到的是“通过票更多”，AgentAuditor 看到的却是“关键分歧点上的证据分支没有对齐”，因此会拒绝自动合并并要求人工复查。

---

## 问题定义与边界

在代码审查场景中，多 Agent 共识的目标是：把多个审查者的输出汇总成一个可执行结论，例如“合并”“阻塞”“需要补测”。这里的“共识”不是哲学意义上的完全一致，而是工程上的决策协议，也就是一套明确规则，规定输入格式、角色责任、聚合方式、冲突处理和升级条件。

边界必须先划清，否则讨论算法没有意义。一个最小可控系统至少要回答四个问题：

| 边界项 | 必须明确的内容 | 不明确会发生什么 |
|---|---|---|
| 角色 | 安全、性能、测试、架构各自检查什么 | 角色重叠或遗漏，输出互相污染 |
| 输入 | 看哪些文件、是否能跑工具、是否有 benchmark | 结论不可比，票数失真 |
| 输出 | 统一成 `issue / severity / evidence / confidence` | 只能做文本拼接，无法可靠聚合 |
| 校验 | 何时重跑、何时人工介入、何时拒绝合并 | 低质量结论直接进入主流程 |

对初学者，一个直观定义是：把三个机器人 reviewer 的意见汇总成一个合并决定。问题不在于“机器人会不会投票”，而在于“它们是否在同一任务空间内说话”。如果 Security Agent 检查静态分析，Performance Agent 只看 benchmark，Test Agent 只看覆盖率，但协调者只统计“通过/拒绝”，那么大量上下文会被抹平。这样做的后果是，强风险的少数意见很容易被多数的弱意见吞掉。

下面这个文字流程图可以帮助理解问题边界：

```text
PR/补丁
  -> 角色分发
     -> Security: 漏洞/配置/依赖
     -> Performance: 热路径/复杂度/回归
     -> Test: 覆盖率/用例缺口/可复现性
  -> 结构化输出
     -> issue
     -> severity
     -> evidence
     -> confidence
     -> provenance
  -> 聚合器
     -> 投票 / 排名 / Judge / 审计器
  -> 不确定性闸门
     -> 直接通过
     -> 重跑复审
     -> 人工升级
```

真实工程里，很多多 Agent 系统失败，不是因为模型完全不够强，而是因为缺少契约。契约可以理解为“角色之间先说清楚规则”。例如，Test Agent 必须复述它覆盖了哪些模块；Security Agent 必须给出触发风险的配置键和值；Performance Agent 必须标明基线版本。没有这些约束，后面的投票只是把不可验证的文本变成一个看似客观的数字。

---

## 核心机制与推导

最常见的四类共识机制可以写成统一的“聚合算子”。

### 1. 多数投票

多数投票只统计谁拿到的票最多：

$$
o^*=\arg\max_o \sum_i \mathbb{1}(vote_i=o)
$$

这里 $\mathbb{1}$ 是示性函数，白话解释就是“如果第 $i$ 个 Agent 投给了选项 $o$，就记 1 分，否则记 0 分”。它的优点是简单、快、可解释；缺点是丢失理由强弱。

### 2. 加权投票

加权投票在多数票上再加一层“谁更可信”的权重：

$$
o^*=\arg\max_o \sum_i w_i \mathbb{1}(vote_i=o)
$$

其中 $w_i$ 是第 $i$ 个 Agent 的权重。白话解释是：不是每张票都一样重，历史表现更稳的 Agent 权重大一些。它比多数票更接近工程现实，但前提是权重本身可信。

### 3. Borda 计数

Borda 计数不只看第一名，而是让每个 Agent 给候选项排个序。若一共 $n$ 个候选项，某个选项 $o$ 的分数为：

$$
s_o=\sum_i (n-rank_i(o))
$$

这里 `rank` 是名次，白话解释就是“排得越靠前，分越高”。它能保留“第二选择”信息，适合意见并不完全对立的任务。

### 4. LLM-as-Judge

Judge 机制不先投票，而是直接让裁判模型读入多份证据，对每个候选结论给出分数：

$$
o^*=\arg\max_o score(o)
$$

`score(o)` 可以理解为“这个结论在当前证据下有多站得住”。它的核心优势不是神秘，而是能直接处理文本证据、上下文关联和非功能性信息，比如“覆盖率说明不完整”“风险描述前后矛盾”。

下面用一个玩具例子对比四种机制。候选结论为 A、B、C，三位 Agent 的排序如下：

| Agent | 第1名 | 第2名 | 第3名 |
|---|---|---|---|
| 安全 | A | B | C |
| 性能 | B | A | C |
| 测试 | A | C | B |

若只看第一名票数，A 获得 2 票，B 获得 1 票，C 为 0，故多数票选 A。若安全、性能、测试的权重分别是 $3,2,1$，则 A 的加权分为 $3+1=4$，B 为 $2$，C 为 $0$，仍选 A。若按 Borda 计数，第一名得 2 分、第二名得 1 分、第三名得 0 分，则：

- A: $2+1+2=5$
- B: $1+2+0=3$
- C: $0+0+1=1$

仍选 A，但 B 作为“稳定的第二选择”被保留下来。若 Judge 读完证据后打分为 A:0.58, B:0.62, C:0.33，则最终选 B，因为它认为 B 的证据链更完整。这里的关键不是“Judge 更聪明”，而是它利用了投票机制没有保存下来的证据结构。

把它们放在一张速查表里更直观：

| 聚合策略 | 输入形式 | 保留的信号 | 典型优点 | 典型风险 |
|---|---|---|---|---|
| 多数投票 | 单一选择 | 票数 | 最便宜、最快 | 多数共谋，忽略少数高风险 |
| 加权投票 | 单一选择 + 权重 | 票数 + 历史可靠性 | 可表达专家优先 | 权重难维护，易过拟合历史 |
| Borda | 完整排序 | 第一选择 + 次优偏好 | 冲突较柔和时更稳 | 不能直接处理复杂证据 |
| LLM Judge | 多份文本证据 | 理由、上下文、完整性 | 能看非结构化信息 | 对 prompt、域外样本敏感 |
| AgentAuditor | 推理树/证据分支 | 分歧点、证据链、一致性 | 可发现“表面一致、理由冲突” | 实现复杂、成本更高 |

从拜占庭容错的角度看，这些方法都在回答同一个问题：当部分节点不可靠时，系统还能否输出可信结果。这里不需要把多 Agent 代码审查直接等同于区块链共识，但启示是相同的：如果多个 Agent 共享偏差、共享工具缺陷、共享 prompt 污染，那么它们不是“独立裁判”，而是“复制的同一错误”。

---

## 代码实现

工程实现的关键不是先选算法，而是先定义结构化输出。下面给出一个可运行的 Python 玩具实现，展示多数投票、加权投票、Borda 计数，以及一个最小版的“证据闸门”。`assert` 用来直接校验结论。

```python
from collections import Counter, defaultdict

reviews = [
    {
        "agent": "security",
        "vote": "approve",
        "ranking": ["approve", "revise", "reject"],
        "weight": 3,
        "confidence": 0.55,
        "evidence": ["no critical vuln found", "override file ignored"],
        "high_risk_issue": False,
    },
    {
        "agent": "performance",
        "vote": "approve",
        "ranking": ["approve", "reject", "revise"],
        "weight": 2,
        "confidence": 0.60,
        "evidence": ["benchmark stable"],
        "high_risk_issue": False,
    },
    {
        "agent": "test",
        "vote": "reject",
        "ranking": ["reject", "revise", "approve"],
        "weight": 1,
        "confidence": 0.92,
        "evidence": ["override file resets test config to zero", "coverage bypass"],
        "high_risk_issue": True,
    },
]

def majority_vote(items):
    return Counter(x["vote"] for x in items).most_common(1)[0][0]

def weighted_vote(items):
    scores = defaultdict(int)
    for x in items:
        scores[x["vote"]] += x["weight"]
    return max(scores, key=scores.get)

def borda_count(items, options):
    scores = defaultdict(int)
    n = len(options)
    for x in items:
        for rank, option in enumerate(x["ranking"], start=1):
            scores[option] += n - rank
    return max(scores, key=scores.get), dict(scores)

def uncertainty_gate(items, consensus):
    for x in items:
        if x["high_risk_issue"] and x["confidence"] >= 0.9:
            return "needs_manual_review"
    return consensus

assert majority_vote(reviews) == "approve"
assert weighted_vote(reviews) == "approve"

winner, borda_scores = borda_count(reviews, ["approve", "revise", "reject"])
assert winner == "approve"
assert borda_scores["reject"] > 0

final_decision = uncertainty_gate(reviews, majority_vote(reviews))
assert final_decision == "needs_manual_review"

print("majority =", majority_vote(reviews))
print("weighted =", weighted_vote(reviews))
print("borda =", winner, borda_scores)
print("final =", final_decision)
```

这个例子故意模拟一个“2:1 通过，但仍应阻塞”的情形。多数票、加权票、Borda 都可能给出 `approve`，因为它们看的是聚合后的表面结论；不确定性闸门看到的是：有一个高置信度、高风险拒绝，而且给出了具体证据，因此自动升级为人工复查。

真实工程实现应至少补齐五个字段：

| 字段 | 作用 | 为什么必须有 |
|---|---|---|
| `issue` | 问题描述 | 没有它就无法对齐“在说哪件事” |
| `severity` | 严重级别 | 决定少数意见是否可触发阻塞 |
| `evidence` | 证据片段 | 决定能否审计与复现 |
| `confidence` | 置信度 | 决定是否进入不确定性闸门 |
| `provenance` | 来源记录 | 用于追踪 prompt、工具、版本、输入 |

若进一步实现 AgentAuditor，核心逻辑不是再投一轮票，而是比较“推理树”。推理树可以理解为“从结论回溯到证据的路径”。一个简化伪代码如下：

```text
for each agent_output:
    parse into reasoning_trace, claims, evidence_hits

build reasoning_tree from claims -> evidence

find divergence_points:
    if two agents reach same conclusion with different unsupported evidence:
        mark as suspicious
    if one agent raises high-severity issue and others never inspect same artifact:
        mark as incomplete consensus

aggregate_hits:
    consensus_hit = overlap(evidence_hits)
    support_ratio = supported_claims / total_claims

if suspicious or support_ratio < threshold:
    trigger uncertainty_gate
    rerun judge or request human review
else:
    emit final decision
```

这正是“reasoning trace -> divergence detection -> aggregated hit”的核心链路。它比多数票慢，但优势很明确：多数票只知道谁赢，审计器知道为什么赢、赢得是否站得住。

---

## 工程权衡与常见坑

多 Agent 共识系统常见的问题，不是算法不会写，而是把错误的假设带进了生产。

第一类坑是共享偏差。三个 Agent 看起来独立，实际上都吃同一份数据、同一版 prompt、同一套工具封装。这样做出来的 3 票，并不是三份独立证据，而是一份偏差的三次复读。前面那个 `override` 文件的例子就是典型：安全和性能 Agent 根本没检查到关键配置，测试 Agent 检查到了，但被 2:1 吞没。降级策略不是简单“给测试 Agent 更高权重”，而是要求高严重级别问题触发二次验证。

第二类坑是把 Judge 当作黑箱真理。Judge 的强项是综合复杂文本，但弱项也明确：受 prompt drift、verbosity、领域迁移影响大。prompt drift 可以理解为“提示词版本轻微变化，行为却明显变了”。如果不做版本控制与回归测试，Judge 一次升级就可能把整个审查口径改掉。

第三类坑是忽略可靠性乘积。假设检索正确率 0.95、角色执行正确率 0.92、聚合正确率 0.96、输出格式化正确率 0.98，那么整条链路的可靠性近似为：

$$
R_s = 0.95 \times 0.92 \times 0.96 \times 0.98 \approx 0.822
$$

单看每一步都不差，但串起来后只有约 82.2%。这就是为什么工程上必须插入契约和校验，而不是只看最终模型分数。

下面这张表可以直接用于排坑：

| 坑位 | 典型表现 | 降级策略 |
|---|---|---|
| 数据偏差 | 多个 Agent 同时漏掉同类问题 | 用 AgentAuditor 比对证据分支，不只比结论 |
| 多数共谋 | 2:1 放过高风险改动 | 高严重级别少数意见可触发阻塞或重跑 |
| Prompt drift | 同一任务不同日期结果漂移 | Judge 做版本管理和回归样本测试 |
| 证据缺失 | 结论有了，但找不到依据 | 强制输出 `evidence + provenance` |
| 可靠性衰减 | 每步都“还行”，整体常失效 | 在每个环节加 assert、schema 校验、人工闸门 |
| 角色错位 | 性能 Agent 在评价安全问题 | 明确角色边界，不允许越权结论 |

真实工程例子可以这样理解。一个组织搭了三层审查链：静态分析 Agent、LLM Reviewer、LLM Judge。上线后发现漏检率仍高。排查发现不是 Judge 不够强，而是前面的 Reviewer 输出没有把“问题文件路径”和“触发条件”结构化，导致 Judge 在文本里找不到稳定锚点。修复方式不是继续加模型，而是先修输出协议。这种场景非常常见。

---

## 替代方案与适用边界

如果目标是“最低成本上线”，多数投票仍然有价值，但只能用于低风险、可回滚、证据结构简单的任务，例如格式建议、轻量风格审查。只要任务涉及安全、配置、生效路径、复杂依赖，裸多数票通常不够。

如果目标是“尽快得到一个总体判断”，LLM Judge 适合上下文稳定、prompt 已验证、输入格式统一的场景。它尤其适合把多个 reviewer 的自然语言结论压成一个总评。但它必须配 uncertainty gate，也就是不确定性闸门。白话解释是：当模型自己也拿不准，或发现高风险冲突时，不允许它继续自动决策。

如果目标是“尽量提高 recall”，也就是尽量少漏问题，那么 Multi-Agg 和 Self-Agg 更合适。Multi-Agg 可以理解为“多模型多轮汇总”，Self-Agg 则是“同一模型多次独立复审再聚合”。前者能引入异质性，后者实现简单、成本可控。已有代码审查研究显示，多轮聚合能明显抬高 F1，最高提升可达 43.6%。这类方案的核心不是票数，而是“持续命中的问题更可信”。

如果目标是“覆盖率 + 证据链 + 可审计”，AgentAuditor 一类结构化审计器更合适。它特别适用于 evidence 分歧多、单次 Judge 不稳定、需要复盘责任链的团队。

最后给出一张选择表：

| 方案 | 适用边界 | 优势 | 主要代价 |
|---|---|---|---|
| 多数投票 | 低风险、结论简单 | 快、便宜、易解释 | 容易吞没少数高风险意见 |
| 加权投票 | 有稳定历史统计 | 能利用专家/历史表现 | 权重维护复杂 |
| Borda | 需要保留排序偏好 | 比单票更细腻 | 仍不看深层证据 |
| LLM Judge | 上下文稳定、格式规范 | 可综合文本与非功能信息 | 受 prompt 和域迁移影响 |
| Self-Agg | 单模型可重复调用 | 成本低于多模型 | 异质性不足 |
| Multi-Agg | 追求更高召回 | 对单次波动更稳 | 成本和治理复杂度更高 |
| AgentAuditor | 需要证据链审计 | 能识别“理由不一致的表面共识” | 实现成本最高 |

一个实用决策原则是：先看风险，再看证据密度。低风险低证据密度任务，用多数或加权；高风险高证据密度任务，用 Judge 或 AgentAuditor；如果你不知道自己属于哪类，默认加一个 uncertainty gate，而不是默认自动通过。

---

## 参考资料

- Auditing Multi-Agent LLM Reasoning Trees and Improving Consensus via Anti-Consensus Search（AgentAuditor、相对多数票与 Judge 的提升）
- Benchmarking and Studying the LLM-based Code Review（代码审查中的 hit-agreement、Multi-Agg 与 Self-Agg）
- LLM-as-a-Judge for Software Engineering: Literature Review, Vision and the Road Ahead（Judge 在软件工程任务中的优势与限制）
- Why do most Multi-Agent systems fail?（多 Agent 失败的工程原因、契约与协作问题）
- Borda count（Borda 计数的基本定义与公式）
- Lusser’s law（串联系统可靠性乘积）
- 关于 Claude Code 多 Agent 审查实践的公开博客与行业案例（多角色审查对评论深度、合规率和合并效率的影响）
