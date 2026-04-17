## 核心结论

Agent 的越权行为，本质上不是“模型做错了一步”，而是“系统允许它把模糊意图翻译成了超出授权的动作”。零基础可以先记住一句话：用户说“帮我查”，系统最后却执行了“删、改、转发、提交”，这就是越权。

要把这类风险压住，单靠提示词里的“你只能做什么”不够，至少要有三层防护：

| 防护层 | 解决的问题 | 责任方 | 典型手段 |
|---|---|---|---|
| 行为规范/策略层 | 哪些工具和动作天生不该放行 | 策略管理模块、API Gateway | 白名单、默认拒绝、审计日志 |
| 预操作意图推理层 | 当前动作是否真的符合用户目标 | LLM 判定器、语义校验模块 | 意图抽取、一致性打分、阈值拦截 |
| 后验影响评估层 | 动作执行后是否产生超范围影响 | 审计模块、约束执行器 | 结果校验、敏感字段检查、回滚/告警 |

这三层里，第一层是硬边界，第二层是语义边界，第三层是结果边界。少任何一层，都会留下明显缺口。

一个必须先建立的原则是：所有动作都要先经过确定性的授权检查，再做语义一致性判断。原因很简单。`delete_customer` 这种动作，即使模型声称“为了排查问题”，也不能先执行再解释。先把根本不允许的动作挡在门外，后续推理才有意义。

玩具例子可以直接看客服机器人。策略只允许 `customer_db.read_customer`。当 Agent 说“我要读取客户资料来排查订单状态”，白名单先通过，再做意图一致性校验；如果它尝试 `customer_db.delete_customer`，Gateway 直接拒绝并写 deny log，这一步连语义分析都不必进入。这里的“deny log”就是拒绝日志，白话讲就是“系统把被拦下的危险动作记账，便于追责和复盘”。

真实工程里，越权通常不是这么直白。更常见的是“为了提高效率”而扩大 blast radius。blast radius 可以理解为“一个操作出问题时，可能波及多大范围”。例如运维 Agent 本来只该查看某个租户的配置，却顺手扫描了整个环境、写入了全局缓存，或者调用了不在用户授权里的批量修复接口。表面看都是“排障”，实质上已经脱离原始授权意图。

所以结论可以压缩成一句工程规则：先用确定性策略决定“能不能做”，再用意图对齐决定“该不该现在做”，最后用影响评估验证“做完之后有没有超界”。

---

## 问题定义与边界

“越权行为检测”不是泛指所有错误调用，而是特指 Agent 执行的动作超出了用户授权范围、业务边界或系统策略边界。

这里要分清三个概念：

| 概念 | 白话解释 | 例子 |
|---|---|---|
| 授权范围 | 用户明确允许系统做的事 | “查询客户信息” |
| 意图 | 用户真正想达成的目标 | “为了回答客户问题，需要看到订单状态” |
| 动作 | Agent 最终调用的具体接口 | `customer_db.read_customer` |

很多系统的错误在于只校验动作，不校验意图；或者只保留意图描述，不把它绑定到可执行动作上。结果就是系统“看起来合规”，但执行路径已经和原始目标脱钩。这种现象可以理解为 operational decoupling，白话讲就是“表面在干一件事，实际执行链已经跑偏”。

边界至少要落在两层。

第一层是“工具 + 动作”边界。也就是某个 Agent、某个角色、某个会话，到底能调用哪些 API。这里要求是细粒度，而不是笼统地写“允许访问数据库”。因为 `db.read` 和 `db.delete` 的风险完全不同。

第二层是“意图语义”边界。也就是即使动作本身在白名单内，也要判断它当前是否服务于用户意图。比如用户让系统“查找张三的联系方式”，而 Agent 选择了“批量导出所有客户联系人”，虽然都属于读取类动作，但后者明显超范围。

下面这个表可以把边界划清：

| 用户请求 | 允许操作 | 被拒动作 | 触发的防护层 |
|---|---|---|---|
| 查询客户资料 | `customer_db.read_customer` | `customer_db.delete_customer` | 策略层直接拒绝 |
| 查询单个订单状态 | `order_db.read_order` | `order_db.export_all_orders` | 意图层拒绝 |
| 修改用户昵称 | `profile.update_nickname` | `profile.update_role` | 策略层直接拒绝 |
| 生成日报 | `report.generate_daily` | `mail.send_external_batch` | 后验层校验外发影响 |

新手最容易忽略的是：边界不是“我相信模型会听话”，而是“我先定义它不可能越过的线”。如果一个系统只有提示词，没有网关，没有审计，那么它并没有真正的边界，只有希望。

一个可操作的定义是：只有当动作同时满足以下条件时，才算“授权内执行”：

$$
\text{Authorized}(a) = \text{PolicyAllow}(a) \land \text{IntentAligned}(i, a) \land \text{ImpactAcceptable}(r)
$$

其中，$a$ 是动作，$i$ 是意图，$r$ 是执行结果。也就是说，动作必须被策略允许、与当前意图一致、执行结果没有产生超范围影响。三者缺一不可。

---

## 核心机制与推导

三层防护可以写成一个串行决策链：

1. 行为规范层先做确定性白名单检查。
2. 预操作层再做意图和动作的一致性判断。
3. 执行后做影响评估和审计归档。

这里的关键，是把原本模糊的“用户想要什么”变成可计算对象。常见方法是把意图和动作描述都编码成向量，再计算余弦相似度：

$$
S = \cos(\text{Enc}(intent), \text{Enc}(action))
$$

这里的 `Enc` 是编码器，白话讲就是“把一句话变成一串能算距离的数字”。$S$ 越接近 1，说明语义越接近。系统可以设定阈值 $\theta$，只有当：

$$
S > \theta
$$

并且动作已经通过白名单检查时，才允许执行。

这套公式的意义不是让模型“自己判断自己对不对”，而是把“意图对齐”从纯文本描述变成可审计的数值决策。数值并不代表绝对正确，但它能形成稳定的门槛、日志和复盘依据。

看一个最小例子。

玩具例子里，用户意图是：“获取客户信息用于排查投诉”。系统有两个候选动作：

- `customer_db.read_customer`
- `customer_db.delete_customer`

假设编码后得到：

- 与 `read_customer` 的相似度为 $0.82$
- 与 `delete_customer` 的相似度为 $0.34$
- 阈值设为 $\theta = 0.75$

那么结论是：

| 动作 | 白名单是否允许 | 相似度分数 | 结果 |
|---|---|---|---|
| `read_customer` | 是 | 0.82 | 允许执行 |
| `delete_customer` | 否 | 0.34 | 直接拒绝 |

注意顺序。`delete_customer` 其实在第一层就该被拒绝，根本不应该靠低相似度来“补救”。这是工程上非常重要的设计原则：语义判断是补充，不是替代权限控制。

真实工程例子可以看企业知识库助手。用户说“帮我找出某个客户最近三次工单的根因”。Agent 可能有这些工具：

- `ticket.read_ticket`
- `ticket.search_all`
- `crm.read_customer`
- `crm.export_customers`
- `storage.write_report`

策略上，`crm.export_customers` 虽然对某些后台流程是合法工具，但对这个用户会话未授权；即使模型认为“导出后更方便分析”，也必须先被挡掉。然后在剩余允许动作里，再判断是读取单个客户、读取相关工单，还是误用全量搜索接口。最后执行后还要检查结果里是否写出了超范围数据，比如把整个客户列表写进报告文件。

因此，三层链路实际上在回答三个不同问题：

| 问题 | 判断方式 | 失败后处理 |
|---|---|---|
| 这个动作是否被允许存在 | 确定性策略 | 立即拒绝并记录 |
| 这个动作是否符合当前意图 | 相似度或 LLM 判定 | 拒绝、降级、人工确认 |
| 这个动作执行后是否产生超界结果 | 审计与结果验证 | 告警、回滚、冻结能力 |

如果系统只做第一层，会挡住明显越权，但挡不住“白名单内误用”；如果只做第二层，会被高风险动作绕过；如果不做第三层，则会漏掉“执行前看起来没问题，执行后结果超范围”的情况。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它不依赖外部模型，而是用手工定义的向量来模拟“意图编码”和“动作编码”。重点不是向量本身，而是三层链路怎么串起来。

```python
from math import sqrt
from dataclasses import dataclass
from typing import Dict, List


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Action:
    name: str
    tool: str
    vector: List[float]
    effect: str  # read / write / delete / export


class PolicyGateway:
    def __init__(self, allowlist: Dict[str, List[str]]):
        self.allowlist = allowlist
        self.deny_logs = []

    def allow(self, tool: str, action: str, intent_artifact_id: str) -> bool:
        allowed_actions = self.allowlist.get(tool, [])
        ok = action in allowed_actions
        if not ok:
            self.deny_logs.append({
                "tool": tool,
                "action": action,
                "intent_artifact_id": intent_artifact_id,
                "reason": "policy_denied",
            })
        return ok


class IntentActionJudge:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def is_aligned(self, intent_vec: List[float], action_vec: List[float]) -> bool:
        return cosine(intent_vec, action_vec) > self.threshold


class ImpactAuditor:
    def __init__(self):
        self.audit_logs = []

    def verify(self, action: Action, result: Dict, intent_artifact_id: str) -> bool:
        bad = False

        # 真实工程里这里通常会检查敏感字段、跨租户访问、不可变字段改写等
        if action.effect in {"delete", "export"}:
            bad = True

        if result.get("touched_sensitive_service", False):
            bad = True

        self.audit_logs.append({
            "action": action.name,
            "effect": action.effect,
            "intent_artifact_id": intent_artifact_id,
            "impact_ok": not bad,
        })
        return not bad


def run_guarded_action(
    intent_artifact_id: str,
    intent_vec: List[float],
    action: Action,
    gateway: PolicyGateway,
    judge: IntentActionJudge,
    auditor: ImpactAuditor,
):
    # Step 1: deterministic policy check
    if not gateway.allow(action.tool, action.name, intent_artifact_id):
        return "DENY_POLICY"

    # Step 2: intent-action alignment
    if not judge.is_aligned(intent_vec, action.vector):
        return "DENY_ALIGNMENT"

    # Step 3: execute + post-check
    fake_result = {
        "status": "ok",
        "touched_sensitive_service": False,
    }
    if not auditor.verify(action, fake_result, intent_artifact_id):
        return "DENY_IMPACT"

    return "ALLOW"


read_customer = Action(
    name="read_customer",
    tool="customer_db",
    vector=[0.9, 0.8, 0.1],
    effect="read",
)

delete_customer = Action(
    name="delete_customer",
    tool="customer_db",
    vector=[0.1, 0.2, 0.95],
    effect="delete",
)

intent_vec = [0.88, 0.84, 0.05]  # “获取客户信息用于排查问题”

gateway = PolicyGateway({
    "customer_db": ["read_customer"]
})
judge = IntentActionJudge(threshold=0.75)
auditor = ImpactAuditor()

assert run_guarded_action(
    "intent-001", intent_vec, read_customer, gateway, judge, auditor
) == "ALLOW"

assert run_guarded_action(
    "intent-001", intent_vec, delete_customer, gateway, judge, auditor
) == "DENY_POLICY"

assert len(gateway.deny_logs) == 1
assert gateway.deny_logs[0]["action"] == "delete_customer"
assert auditor.audit_logs[0]["impact_ok"] is True
```

这段代码对应的实现思路很直接：

| 步骤 | 作用 | 为什么不能省 |
|---|---|---|
| `PolicyGateway.allow` | 检查动作是否在白名单内 | 防止未授权工具直接穿透 |
| `IntentActionJudge.is_aligned` | 检查当前动作是否符合当前意图 | 防止白名单内误用 |
| `ImpactAuditor.verify` | 检查执行结果有没有超范围影响 | 防止“执行前正常、执行后越界” |

如果要把它放进真实工程，结构通常会更像下面这样：

1. API Gateway 接收 Agent 工具调用请求，先根据身份、会话、租户、能力策略做 allow/deny。
2. 通过后，系统读取 `intent_artifact_id` 对应的意图摘要，调用意图-动作一致性判定器。
3. 一致性通过后才真正访问下游服务。
4. 返回结果前，审计模块检查是否触及敏感资源、是否扩大数据范围、是否违反不可变约束。
5. 最终把“请求、动作、分数、结果、审计结论”写入统一日志。

真实工程例子：客服 Agent 查询用户退款状态。它被允许读取订单、读取支付状态、写入一条客服备注，但不允许删用户、导出支付表、修改订单金额。流程是：

- Gateway 先限制工具范围。
- 判定器确认“查询退款状态”与“读取订单详情”一致，而与“批量导出支付流水”不一致。
- 审计模块再检查返回结果是否包含完整银行卡号、是否写入了不该修改的账务字段。

这样做的价值是，即使模型给出一段貌似合理的解释，系统也不会因为“它说得通”就放行。

---

## 工程权衡与常见坑

第一类常见坑，是把 `allowed-tools` 当成最终安全机制。`allowed-tools` 可以理解为“模型被告知你应该用哪些工具”，但这更像提示，不是强制执行。只要没有位于执行链路上的 Gateway，提示就可能被绕过、误解或忽略。

下面这个对比非常关键：

| 方案 | 可执行强制性 | 语义防误用 | 审计能力 | 风险 |
|---|---|---|---|---|
| 只靠 `allowed-tools` | 弱 | 弱 | 弱 | 模型可能仍尝试危险动作 |
| Gateway + 意图校验 | 强 | 中到强 | 中 | 能挡住大部分前置风险 |
| Gateway + 意图校验 + 后验评估 | 强 | 强 | 强 | 成本更高，但闭环更完整 |

第二类坑，是以为“白名单已经足够细”，所以不做语义判定。问题在于很多越权不是调用了禁用动作，而是在允许动作里做了不该做的事。例如允许 `search_documents`，但 Agent 为了回答一个单用户问题，搜索了整个公司的人事档案库。这类风险只看 ACL 很难挡住。

第三类坑，是忽略后验影响评估。执行前的所有检查，都只能保证“计划看起来合理”，不能保证“结果确实没超界”。例如：

- 本来只读单个客户，却因为查询条件错误读到了多个租户。
- 本来允许生成报告，却把敏感字段原样输出到可下载文件。
- 本来允许写客服备注，却误改了用户权限字段。

这些问题只有看结果才能发现。

第四类坑，是阈值设计过于粗糙。意图一致性分数不是越高越好，也不是一个固定阈值适合所有动作。读取、写入、删除、转账的风险等级不同，阈值应该分层。高风险动作不仅阈值更高，还可能要求人工确认或双重审批。

第五类坑，是日志不绑定意图工件。意图工件可以理解为“把用户目标压缩成可追踪对象”。如果日志里只有“谁调用了什么 API”，却没有“它当时声称是为了解决什么问题”，就很难复盘模型是否发生了语义漂移，也很难优化策略。

一个典型误区是：系统发现 `db.delete` 没在白名单，就拒绝了，于是团队觉得“我们已经安全”。其实这只能说明“最粗暴的越权被挡住了”。更难的问题是，Agent 用 `db.read` 读了一大批不相关数据，然后在后续步骤里继续扩散。真正成熟的系统，不会把“没删库”当成合格线。

---

## 替代方案与适用边界

除了“策略网关 + 意图判定 + 后验审计”这条主线，工程上还有两类常见替代或补充方案。

第一类是 Capability Token，也就是能力令牌。白话讲，它像一张写明“你现在能做什么”的临时通行证。ACT 这类方案会把权限编码进 token，例如只允许某个 Agent 在某个时间窗口里读取指定资源。这种方式的优点是权限边界清晰、适合跨系统传递、支持撤销；缺点是它主要解决“有没有权限”，不能天然解决“当前动作是否符合当前意图”。

第二类是 Token Vault 或 OAuth 编排平台。它们更关注授权生命周期管理，比如用户授权、token 托管、失效刷新、跨平台接入。这对多系统 Agent 非常重要，因为如果凭证管理混乱，再好的意图对齐也落不了地。但这类平台同样不直接判断“模型是不是在借合法凭证做不合意图的事”。

因此可以做一个并列比较：

| 方案 | 授权策略能力 | 语义验证能力 | 适用场景 | 局限 |
|---|---|---|---|---|
| ActionAuth 风格 Gateway + AgentArmor/三层链路 | 强，细粒度动作级控制 | 强，可叠加意图与后验验证 | 高风险 Agent、需要审计闭环的业务 | 系统设计更复杂 |
| ACT 类能力令牌 | 强，适合跨系统传递与撤销 | 弱，需要额外接入判定器 | 多服务、多平台、动态授权 | 只解决权限，不直接解决意图漂移 |
| Scalekit 类授权栈 | 强于凭证和授权编排 | 弱到中，通常需外接 | 需要 OAuth、Token Vault、统一授权流 | 本身不是越权检测器 |

初学者可以这样理解三者关系：

- ACT 负责“给 Agent 发什么钥匙”。
- Token Vault / OAuth 平台负责“钥匙怎么保管、怎么发、怎么撤销”。
- 意图对齐与后验审计负责“有钥匙也不能乱开门”。

所以替代方案并不是真的替代三层防护，而更像是补齐不同层面的基础设施。对高风险业务，最稳妥的组合通常是：能力令牌控制长期权限，Gateway 强制执行动作级规则，意图判定器负责当前任务语义，后验审计负责收口。

适用边界也要说明白。如果你的系统只是一个只读 FAQ 助手，没有写操作、没有敏感外部工具，那么只做轻量策略检查可能已经足够。但只要 Agent 能调用数据库、文件系统、支付接口、消息发送、代码执行或外部 SaaS，三层链路就应该视为默认配置，而不是高级选项。

---

## 参考资料

- 火山引擎开发者社区，《为 AI Agent 行为立“规矩”——Jeddak AgentArmor 智能体安全框架》：https://developer.volcengine.com/articles/7599494081718583302
- Wandoo Systems, “Why Your AI System Looks Fine, But Is Not / Reasoning Integrity”：https://www.wandoosystems.com/articles/reasoning-integrity
- ActionAuth Agent Access Control Docs：https://www.actionauth.com/docs
- ACT: Agent Capability Tokens：https://www.acttokens.com/
- Scalekit agentic action stack：https://www.scalekit.com/agentic-action
- LLM Council, “ADR-034 Agent Skills Verification”：https://llm-council.dev/adr/ADR-034-agent-skills-verification/
