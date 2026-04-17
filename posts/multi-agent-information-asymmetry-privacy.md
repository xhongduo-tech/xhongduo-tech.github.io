## 核心结论

多 Agent 协作里，最常见的安全误区不是“模型会不会胡说”，而是“默认让所有 Agent 看见全部历史”。这会把系统变成一个高耦合、低审计、易泄露的单体。

更稳妥的做法是把“谁能看到什么”定义成一个显式策略：由角色、当前任务、消息敏感级别、字段级过滤规则共同决定。角色是“你是谁”，任务是“你现在为什么需要这段信息”。两者缺一不可。只有满足授权条件的 Agent，才应拿到原文；其余 Agent 只能拿到裁剪后的上下文、脱敏字段，或者一段目的明确的摘要。

对初级工程师来说，可以先记住一句话：多 Agent 的上下文不是广播，而是分发。分发前必须先过过滤器。第一层是消息过滤，第二层是字段脱敏，第三层是上下文裁剪，第四层是审计留痕。医疗、金融、客服这类场景里，这不是“可选优化”，而是合规和事故边界的一部分。

---

## 问题定义与边界

这里讨论的“信息可见性控制”，指系统在多 Agent 协作时，决定每个 Agent 可以读取哪些消息块、哪些字段、哪些摘要，以及是否允许继续传播这些内容。

它至少包含四个维度：

| 维度 | 含义 | 典型值 |
|---|---|---|
| 角色 `role` | Agent 的职责身份 | `doctor`、`scheduler`、`auditor`、`support` |
| 任务 `task` | 当前正在执行的工作目标 | 问诊、排班、退款、合规审查 |
| 敏感级别 `sensitivity` | 数据泄露后的风险等级 | 公开、内部、PII、PHI、财务机密 |
| 裁剪策略 `policy` | 允许保留、屏蔽或摘要化的规则 | 全文、字段掩码、摘要、拒绝 |

白话解释一下几个术语：

- `PII`：个人身份信息，能直接或间接识别一个人，比如手机号、身份证号。
- `PHI`：受保护健康信息，通常指医疗相关的个人敏感数据。
- `RBAC`：基于角色的访问控制，意思是先按职责分配权限，而不是每条请求都手写规则。

一个简单定义可以写成：

$$
visibleset(r, m, t)=
\begin{cases}
1, & r \in m.requiredRoles \land m.sensitivity \le clearance(r) \land taskMatch(r,t,m) \\
0, & \text{otherwise}
\end{cases}
$$

其中，$r$ 是角色，$m$ 是消息或消息块，$t$ 是当前任务。这个公式表达的不是“能不能调用模型”，而是“这段上下文是否允许进入该 Agent 的可见集合”。

玩具例子：

一个客户对话里有四个字段：`问题类别`、`订单号`、`手机号`、`付款卡后四位`。客服 Agent 为了处理退款，需要看到前 3 项；审计 Agent 为了做流程检查，只需要看到“退款争议”这个问题类别和一段摘要，不需要手机号；推荐 Agent 只需要知道“用户在咨询售后”，完全没必要看到订单号。

这说明“非特权 Agent 不能读写全部历史”不是抽象原则，而是具体到字段和任务的控制要求。

---

## 核心机制与推导

实现上，最稳的做法不是按“整条消息”授权，而是按 chunk 授权。chunk 可以理解为“消息块”，也就是把一段输入拆成更小的可控片段，例如“病历摘要”“预约时间”“患者手机号”分别作为不同块处理。

可见性判断通常分两步：

1. 先过滤：对 chunk 做脱敏、删字段、摘要化。
2. 再判定：检查角色权限、任务匹配、敏感级别。

于是，一个 Agent 最终看到的响应可以写成：

$$
R(r,t)=\sum_{i=1}^{n} visibleset(r,m_i,t)\cdot filter(m_i, r, t)
$$

其中：

| 符号 | 含义 |
|---|---|
| $m_i$ | 第 $i$ 个消息块 |
| $filter(m_i,r,t)$ | 针对角色和任务处理后的块 |
| $visibleset$ | 该块是否允许进入可见集的掩码，取值为 0 或 1 |
| $R(r,t)$ | 给角色 $r$ 在任务 $t$ 下生成的上下文 |

“掩码”可以理解成一个开关，1 表示保留，0 表示丢弃。这样做的好处是，运行时计算非常快，很多系统甚至会把敏感等级直接编码成整数，减少动态分支。

医疗玩具例子：

- `Doctor`，`clearance = 3`
- `Scheduler`，`clearance = 1`

两段 chunk：

- `chunk1 = {type: diagnosis, level: 3}`，包含病名与用药
- `chunk2 = {type: schedule, level: 1}`，包含复诊时间

则：

- 对 `Doctor`：$1 \cdot chunk1 + 1 \cdot chunk2$
- 对 `Scheduler`：$0 \cdot chunk1 + 1 \cdot chunk2$

如果 `chunk2` 里还有患者编号，那么在进入 `Scheduler` 可见集前，还应先做字段脱敏，例如 `patient_id -> ***8127`。这体现了一个关键点：是否可见，与可见后显示成什么样，是两个独立决策。

下面这个简化流程足够表达工程思路：

`输入消息 -> 分类与打标签 -> 过滤/脱敏 -> 计算可见集 -> 响应合成 -> 审计记录`

真实工程例子：

养老医疗多 Agent 系统里，问诊 Agent、排班 Agent、解释 Agent、监督 Agent 会共享一部分任务状态，但不应共享全部临床上下文。排班 Agent 只需要“患者需要 3 天内复诊”和“时间偏好”；解释 Agent 可能需要看到“系统为何触发复诊建议”的理由链；监督 Agent 需要访问调用日志与授权记录，但不一定需要看到完整病历正文。这样才能同时满足协作效率、最小可见原则和 HIPAA/GDPR 一类要求。

---

## 代码实现

下面给出一个可运行的 Python 最小实现。它演示四件事：角色定义、chunk 结构、字段过滤、可见集计算。

```python
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class Role:
    name: str
    clearance: int
    tasks: Set[str]


@dataclass
class Chunk:
    kind: str
    level: int
    required_roles: Set[str]
    data: Dict[str, str]


FIELD_POLICY = {
    "doctor": {"patient_name", "diagnosis", "appointment_time", "patient_id"},
    "scheduler": {"appointment_time", "patient_id"},
    "auditor": {"summary", "ticket_type"},
}

MASK_FIELDS = {"patient_id", "patient_name"}


def mask_value(key: str, value: str) -> str:
    if key == "patient_id":
        return "***" + value[-4:]
    if key == "patient_name":
        return value[0] + "**"
    return value


def apply_filter(role: Role, chunk: Chunk) -> Dict[str, str]:
    allowed_fields = FIELD_POLICY.get(role.name, set())
    filtered = {}
    for key, value in chunk.data.items():
        if key not in allowed_fields:
            continue
        filtered[key] = mask_value(key, value) if key in MASK_FIELDS and role.name != "doctor" else value
    return filtered


def visible(role: Role, task: str, chunk: Chunk) -> bool:
    return (
        role.name in chunk.required_roles
        and chunk.level <= role.clearance
        and task in role.tasks
    )


def build_context(role: Role, task: str, chunks: List[Chunk]) -> List[Dict[str, str]]:
    visible_chunks = []
    for chunk in chunks:
        if visible(role, task, chunk):
            filtered = apply_filter(role, chunk)
            if filtered:
                visible_chunks.append(filtered)
    return visible_chunks


doctor = Role("doctor", clearance=3, tasks={"triage", "review_case"})
scheduler = Role("scheduler", clearance=1, tasks={"schedule_visit"})
auditor = Role("auditor", clearance=1, tasks={"audit_refund", "audit_case"})

chunks = [
    Chunk("clinical", 3, {"doctor"}, {
        "patient_name": "Zhang San",
        "diagnosis": "atrial fibrillation",
        "patient_id": "CN202603198127",
    }),
    Chunk("schedule", 1, {"doctor", "scheduler"}, {
        "appointment_time": "2026-03-21 09:30",
        "patient_id": "CN202603198127",
    }),
    Chunk("audit", 1, {"auditor"}, {
        "summary": "refund dispute escalated",
        "ticket_type": "after_sales",
        "patient_name": "Zhang San",
    }),
]

doctor_ctx = build_context(doctor, "triage", chunks)
scheduler_ctx = build_context(scheduler, "schedule_visit", chunks)
auditor_ctx = build_context(auditor, "audit_case", chunks)

assert len(doctor_ctx) == 2
assert doctor_ctx[0]["patient_name"] == "Zhang San"
assert len(scheduler_ctx) == 1
assert scheduler_ctx[0]["patient_id"].startswith("***")
assert "diagnosis" not in scheduler_ctx[0]
assert len(auditor_ctx) == 1
assert "patient_name" not in auditor_ctx[0]
```

字段级过滤通常要有一张明确矩阵，不要把规则散落在 prompt 里：

| chunk 字段 | doctor | scheduler | auditor | 过滤操作 |
|---|---|---|---|---|
| `patient_name` | 保留 | 删除 | 删除 | 字段移除 |
| `patient_id` | 保留 | 掩码 | 删除 | 部分脱敏 |
| `diagnosis` | 保留 | 删除 | 删除 | 高敏感屏蔽 |
| `appointment_time` | 保留 | 保留 | 删除 | 任务必要保留 |
| `summary` | 可选 | 删除 | 保留 | 摘要转发 |

工程上建议把“权限判断”和“过滤动作”拆开：前者负责决定看不看，后者负责决定怎么看。否则很容易出现一个问题：虽然权限没开，但摘要器先把敏感字段带进了下游上下文。

---

## 工程权衡与常见坑

最小可见性会降低泄露面，但也可能降低协作效率。关键不是“让不让看更多”，而是“是否能证明这些额外信息确实是任务所需”。

常见坑和缓解方式如下：

| 风险 | 典型表现 | 后果 | 缓解措施 |
|---|---|---|---|
| 中央协调器权限过大 | Orchestrator 默认能看全部上下文 | 变成单点观测器 | 协调器只持路由元数据，不持明文正文 |
| 摘要器泄露敏感信息 | 摘要中保留病名、手机号 | 间接泄露 | 摘要前先脱敏，再做摘要 |
| 静态白名单失效 | 任务变化后权限仍沿用旧模板 | 越权访问 | 把任务、项目、时间窗纳入实时判断 |
| 日志记录过多 | 调试日志写入原始提示词 | 二次泄露 | 日志只存哈希、元数据、最小必要片段 |
| 只控读取不控写回 | 低权限 Agent 在输出里重组敏感信息 | 数据外泄 | 对输出同样做策略校验和 DLP 扫描 |

新手最容易忽略的是“摘要也算数据复制”。如果原始文本里有患者姓名、银行卡尾号、地址，那么“帮我总结一下问题”这个动作本身就可能把敏感信息重写一遍。因此，顺序必须是：先筛，再总结；不能先总结，再祈祷模型别提到敏感字段。

另一个常见问题是把 RBAC 当成完整答案。RBAC 只能解决“这个角色通常可以看什么”，但很多真实系统还要求“这个角色在当前项目、当前时间窗口、当前案件状态下是否还可以看”。这已经是上下文感知控制，不再是纯静态授权。

---

## 替代方案与适用边界

实际系统里常见三类方案：

| 方案 | 特性 | 优点 | 局限 | 适用场景 |
|---|---|---|---|---|
| 纯静态 RBAC | 只按角色授权 | 简单、便于上线 | 无法应对任务变化 | 内部知识库、低风险客服 |
| 端到端加密 + 少量解密节点 | 强保护传输与存储 | 降低明文暴露 | 解密节点容易集中化 | 医疗、金融核心链路 |
| 动态角色/任务掩码 + 实时过滤 | 按角色、任务、字段联合控制 | 精细、可审计、适合多 Agent | 设计复杂、测试成本高 | 多团队协作、强合规系统 |

如果系统很小、数据敏感度低、Agent 数量少，纯 RBAC 往往已经够用。比如一个站内知识助手，只有“编辑”和“访客”两类角色，不涉及身份证、病历、账户资产，这时引入复杂的动态掩码未必划算。

但只要进入下面这些场景，就应考虑多层过滤：

- 医疗：需要 PHI 保护、操作审计、解释留痕。
- 金融：需要账户字段隔离、操作追踪、合规审查。
- 企业客服：不同团队只应看到工单必需字段，不能跨租户串看。
- 多租户 SaaS：同一模型服务多个客户，必须做租户级数据隔离。

可以把边界理解成一句话：纯 RBAC 适合“角色稳定、数据简单”；动态可见掩码适合“任务频繁变化、字段敏感度高、审计要求强”。

---

## 参考资料

1. Microsoft Multi-agent Reference Architecture, Security  
   贡献：给出企业多 Agent 安全基线，包括身份认证、RBAC、记忆范围控制、PII 脱敏、工具调用审计，以及在 RAG 场景中按敏感度和权限过滤检索结果。  
   链接：https://microsoft.github.io/multi-agent-reference-architecture/docs/security/Security.html

2. Protecto, Role-Based Access Control for AI Agents  
   贡献：强调“实时角色权限 + 属性级解掩码 + 上下文感知 masking”的组合，适合说明为什么多 Agent 不能只靠静态白名单。  
   链接：https://www.protecto.ai/solutions/agentic-ai-rbac-for-agents/

3. MDPI, Towards Trustworthy AI Agents in Geriatric Medicine: A Secure and Assistive Architectural Blueprint  
   贡献：提供医疗多 Agent 的真实工程语境，覆盖 GDPR、HIPAA、审计日志、角色权限、加密通信、可解释性与人工监督，说明高风险行业为什么必须把权限与审计做成架构的一部分。  
   链接：https://www.mdpi.com/1999-5903/18/2/75
