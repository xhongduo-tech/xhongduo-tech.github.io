## 核心结论

在金融、医疗这类高敏感场景里，大模型应用不是“把模型接进来”就结束，而是要把“谁可以用、用什么数据、调用哪个模型、输出给谁看、谁最终负责”写成一条可审计链路。这里的“可审计”可以简单理解为：事后能查清楚每一步是谁做的、为什么做、做了什么、结果流向哪里。

如果只关注回答质量，不把权限、留痕、脱敏、人工复核一起设计进去，系统上线后很容易在两个地方出问题：一是违规处理敏感数据，二是出了错误结论却无法追责。真正可落地的方案通常有两个共同点：

| 设计点 | 必须回答的问题 | 工程含义 |
| --- | --- | --- |
| 身份与权限 | 谁发起请求，是否有权访问这类数据 | 角色校验、最小权限、审批流 |
| 数据处理 | 输入里有没有身份证号、病历、账户流水等敏感信息 | 脱敏、分级、出境控制 |
| 模型调用 | 调的是本地模型还是外部 API | 在线/离线路径、边界隔离 |
| 审计留痕 | 事后如何证明过程合规 | append-only 日志、时间戳、版本号 |
| 责任闭环 | 模型输出谁负责确认 | 人工复核、签名确认、用途限制 |

一个新手也能理解的最小流程可以写成：

请求人提交任务 -> 审批人确认角色与用途 -> 系统脱敏并调用模型 -> 写入审计链 `L` -> 输出进入人工复核 -> 复核通过后才可用于业务动作。

这里的审计链 `L` 就是“全过程账本”，用来记录请求 ID、时间、角色、模型 ID、Prompt 版本、脱敏策略、输出结果、复核人等字段。没有这条链，合规基本无从谈起。

---

## 问题定义与边界

本文讨论的是高敏感行业中的大模型应用设计，重点是金融、医疗、政务等对数据处理和责任追踪要求严格的场景。这里的“高敏感”指一旦处理不当，会带来监管处罚、隐私泄露、业务误判或重大法律责任的业务。

边界先划清楚：

1. 本文讨论的是“应用设计”，不是模型训练算法本身。
2. 本文关注的是“调用链路合规”，不是泛泛而谈的 AI 伦理。
3. 本文默认企业已经有基本账号体系、权限体系和日志平台，但这些能力需要为大模型场景重新加字段和流程。
4. 本文只讨论“模型作为辅助判断或生成工具”的场景，不讨论让模型自动做不可逆决策的极端方案。

最关键的边界，是区分在线调用和离线批处理。在线调用就是用户发起请求后几秒内返回结果，白话说就是“人在等结果”；离线批处理是系统定时处理成百上千条任务，白话说就是“人不盯着屏幕等，系统先跑完再复核”。

| 维度 | 在线服务 | 离线批处理 |
| --- | --- | --- |
| 目标 | 低延迟响应 | 大规模处理 |
| 延迟要求 | 高，常见为秒级 | 低，分钟到小时级 |
| 权限检查 | 每次请求实时校验 | 批次启动前预审，单条可抽检 |
| 日志写入频率 | 每次调用必写 | 每批次汇总 + 每条明细 |
| 复核方式 | 高风险结果实时拦截 | 批量抽样或重点复核 |
| 典型场景 | 智能客服、辅助问答、在线审批建议 | 病历摘要、批量报告脱敏、贷前材料整理 |

再往前一步，合规边界还包括数据出境和共享审批。数据出境可以简单理解为：数据是否离开本机构、本地域或本监管边界；共享审批则是：数据是否被允许给另一个部门、供应商或模型服务商使用。只要调用路径中出现外部模型 API，这两个问题就必须先回答，不能等上线后补流程。

一个玩具例子是：公司内部员工让模型总结一份包含客户手机号的 Excel。技术上很简单，但如果请求人只是实习生、模型又是外部云服务、日志里还没记用途，那么这个请求就不该直接放行。问题不是“能不能调模型”，而是“谁有权在什么条件下调模型”。

---

## 核心机制与推导

合规设计的中心对象是审计链 `L`。可以把它理解为一次模型交互的结构化证据。最小形式可以写成：

$$
L = \{id, t, user, role, data\_class, model, prompt\_ver, input\_hash, output\_hash, scope, mask\_policy, reviewer, status\}
$$

其中：

- `id`：请求唯一标识，白话说就是“这次调用的编号”。
- `t`：请求时间。
- `role`：请求人角色，比如风控专员、医生助理、审计员。
- `data_class`：数据分级，比如公开、内部、敏感、核心敏感。
- `scope`：授权范围，表示输出能用于什么业务，不能无限扩散。
- `mask_policy`：脱敏策略 ID，说明输入和输出按哪条规则处理。
- `status`：当前状态，比如待审批、已执行、待复核、已归档。

为什么 `L` 是追责基石？因为事后所有判断都依赖它。系统是否允许回显，不是只看“这人登录了没有”，而是看角色、用途、数据等级、审批状态是否同时满足。可以写成一个简化判定：

$$
allow = match(L.role, requester) \land approve(L.scope, purpose) \land safe(L.data\_class, channel)
$$

这里：

- `match` 表示角色匹配，白话说就是“是不是该你看”。
- `approve` 表示用途是否获批，白话说就是“你拿它干这个事是否被允许”。
- `safe` 表示数据与通道是否匹配，白话说就是“这类数据能不能走这条调用链”。

如果输入包含敏感字段，还要先做脱敏：

$$
mask(x, n) = x_{1:k} + \underbrace{* \cdots *}_{n} + x_{m:r}
$$

这个公式表示保留少量前后缀，中间用掩码替代。比如身份证号只显示前 6 位和后 4 位，中间隐藏。重点不是掩码形式有多复杂，而是要把“采用了哪条掩码规则”记进 `L.mask_policy`，否则审计时只能看到结果，看不到规则。

下面给一个新手能读懂的 `L` 结构示意：

```json
{
  "request_id": "req-20240825-140300-001",
  "request_time": "2024-08-25T14:03:00+08:00",
  "user_id": "u1024",
  "role": "risk_officer",
  "purpose": "pre_loan_review",
  "data_class": "sensitive",
  "model_id": "internal-llm-v3",
  "prompt_version": "loan-risk-prompt-v12",
  "input_ref": "sha256:...",
  "output_ref": "sha256:...",
  "mask_policy": "cn-id-mask-v2",
  "authorization_scope": "read_summary_only",
  "reviewer": "rev-77",
  "status": "pending_human_review"
}
```

玩具例子可以进一步说明机制。假设有一条请求：实习生上传客户名单，想让模型按违约风险排序。系统应先判断：

1. 角色是否允许接触这类名单。
2. 名单里是否有身份证号、住址、病史等敏感字段。
3. 调用的是内网模型还是外部 API。
4. 输出是否直接用于审批，还是仅供人工参考。

如果第 2 步发现有敏感字段，第 3 步又是外部模型，那么即使模型很强，也可能必须拦截或改成离线脱敏后再调用。这里的结论不是“外部模型不能用”，而是“高敏感原文不能默认直接送出去”。

真实工程例子更能说明问题。比如 2024-08-25 14:03，某银行风控专员批量处理 100 份贷前报告，报告中有身份证号和家庭住址。合理链路应该是：

提交批次 -> 系统识别敏感字段 -> 执行脱敏策略 `cn-id-mask-v2` -> 调用内部模型生成摘要与风险提示 -> 将输入摘要哈希、输出哈希、Prompt 版本、用途“贷前审批”、复核账号一起写入 `L` -> 复核人确认后才允许进入评分卡或人工面审页面。

这说明，大模型在高敏感领域通常不是终审者，而是“辅助判断器”。真正负责业务动作的是规则引擎、评分卡或人工复核节点。

---

## 代码实现

下面用一个可运行的 Python 玩具实现，把权限校验、脱敏、模型调用、审计链写入和人工复核串起来。这里不接真实模型，用规则函数代替，重点看结构。

```python
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json

@dataclass
class AuditLog:
    request_id: str
    request_time: str
    user_id: str
    role: str
    purpose: str
    data_class: str
    model_id: str
    prompt_version: str
    input_hash: str
    output_hash: str
    mask_policy: str
    authorization_scope: str
    reviewer: str | None
    status: str

APPEND_ONLY_LOG = []

ROLE_POLICY = {
    "risk_officer": {"sensitive", "internal"},
    "doctor_assistant": {"sensitive", "internal"},
    "intern": {"internal"},
}

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def mask_cn_id(text: str) -> str:
    # 玩具规则：把 18 位身份证号中间 8 位替换为 *
    if len(text) == 18 and text[:-1].isdigit():
        return text[:6] + "*" * 8 + text[-4:]
    return text

def authorize(role: str, data_class: str, channel: str) -> bool:
    allowed = ROLE_POLICY.get(role, set())
    return data_class in allowed and channel in allowed

def toy_model_call(masked_input: str, prompt_version: str) -> str:
    # 用简单规则模拟模型输出
    if "逾期" in masked_input or "欠款" in masked_input:
        return f"[{prompt_version}] 建议人工重点复核"
    return f"[{prompt_version}] 风险信号较弱"

def append_only_write(log: AuditLog) -> None:
    APPEND_ONLY_LOG.append(json.dumps(asdict(log), ensure_ascii=False))

def handle_request(user_id: str, role: str, raw_text: str, purpose: str, channel: str = "internal"):
    data_class = "sensitive" if "身份证" in raw_text or len(raw_text) >= 18 else "internal"
    if not authorize(role, data_class, channel):
        raise PermissionError("role or channel not allowed")

    masked_text = raw_text.replace("身份证:", "身份证:").replace("110101199001011234", mask_cn_id("110101199001011234"))
    output = toy_model_call(masked_text, "prompt-v12")

    log = AuditLog(
        request_id="req-001",
        request_time=datetime.now().isoformat(timespec="seconds"),
        user_id=user_id,
        role=role,
        purpose=purpose,
        data_class=data_class,
        model_id="internal-llm-v3",
        prompt_version="prompt-v12",
        input_hash=sha256_text(masked_text),
        output_hash=sha256_text(output),
        mask_policy="cn-id-mask-v2",
        authorization_scope="read_summary_only",
        reviewer=None,
        status="pending_human_review",
    )
    append_only_write(log)
    return masked_text, output, log

masked_text, output, log = handle_request(
    user_id="u1001",
    role="risk_officer",
    raw_text="客户A，身份证:110101199001011234，存在逾期记录",
    purpose="pre_loan_review"
)

assert "********" in masked_text
assert "人工重点复核" in output
assert len(APPEND_ONLY_LOG) == 1
assert log.status == "pending_human_review"
```

这个示例省略了很多生产细节，但保留了最关键的五步：

| 步骤 | 示例动作 | 设计目的 |
| --- | --- | --- |
| 权限校验 | `authorize(role, data_class, channel)` | 防止未授权访问 |
| 输入脱敏 | `mask_cn_id(...)` | 减少敏感原文暴露 |
| 模型调用 | `toy_model_call(...)` | 生成辅助结论 |
| 审计写入 | `append_only_write(log)` | 保留可追责证据 |
| 人工复核 | `status="pending_human_review"` | 阻断自动直出风险 |

生产实现通常还要再补四类字段：

1. Prompt 版本号和模板 ID，确保事后知道模型当时被怎样引导。
2. 脱敏算法 ID 和配置版本，避免“看得见结果、看不见规则”。
3. 输出用途标签，比如“仅参考”“不可直连放款”“仅供病历摘要”。
4. 复核账号与复核时间，形成责任闭环。

如果对“不可篡改存证”有更高要求，append-only 日志可以进一步落到对象存储 WORM、数据库审计表、消息队列顺序日志，甚至区块链存证。这里的关键不是追新技术，而是保证“写进去后不能被静默改掉”。

---

## 工程权衡与常见坑

高敏感场景的核心权衡，是延迟与审计深度之间的平衡。在线系统追求秒级返回，但不能为了快，把日志简化到只剩一句“调用成功”。一旦缺字段，审计就出现断点。

常见坑可以直接列出来：

| 坑点 | 影响 | 缓解策略 |
| --- | --- | --- |
| 只记录部分字段 | 无法追溯输入、Prompt、输出版本 | 强制审计链最小字段集，缺字段拒绝落库 |
| 忽视数据出境审批 | 触发监管问责或合同违约 | 外部模型必须绑定审批单和数据分级 |
| 输出可直接驱动业务动作 | 模型误判直接放大成业务事故 | 高风险场景改为“模型建议 + 人工复核” |
| 日志可改写 | 事后无法证明过程真实 | append-only、哈希校验、定期审计 |
| 只做输入脱敏，不管输出 | 输出可能重新拼出隐私信息 | 输出侧再做敏感检测与回显控制 |
| 权限只看登录态 | 同一系统内越权访问 | 角色、用途、数据等级三元校验 |

有三个坑尤其常见。

第一，忽视数据出境审批。很多团队以为“调用的是知名云厂商 API，所以默认合规”，这是错误的。是否可用，取决于数据类型、合同边界、地域、监管要求，而不是供应商名气。

第二，黑盒输出无法解释。比如模型给出“建议拒贷”，但说不清依据是收入异常、历史逾期还是材料缺失，这时就很难通过审计。解决方法通常不是强行要求模型“绝对可解释”，而是引入检索增强、规则命中回显、知识图谱校验，把输出锚定到可验证证据。

第三，日志可改写。很多系统把日志写进普通业务表，管理员有更新权限，结果等于没有可信审计。日志设计必须默认“写多次可以，回写覆盖不行”。

真实工程里，常见做法是把链路拆成三层：

1. 数据治理层：分级分类、脱敏、跨境审批。
2. 模型编排层：Prompt 模板、模型路由、输出校验。
3. 审计复核层：结构化日志、告警、人工复核、归档。

如果拿前面的银行例子说，2024-08-25 那次“100 份报告脱敏处理”，真正决定系统是否合规的，不是摘要模型写得多漂亮，而是有没有把“谁发起、脱敏用哪条规则、何时调用、输出给谁、谁复核”完整记下来。

---

## 替代方案与适用边界

不是所有业务都要上最重的存证和复核。合规设计要按风险分层，否则系统会被流程拖死。

可以用下面这张决策表判断：

| 场景 | 风险级别 | 推荐方案 |
| --- | --- | --- |
| 内部公开知识问答 | 低 | 普通日志 + 定时审计 |
| 客服话术润色 | 低到中 | 日志 + 敏感词检测 + 抽样复核 |
| 合同摘要、病历整理 | 中到高 | 结构化审计链 + 输出脱敏 + 人工复核 |
| 贷前审批建议、医疗辅助判断 | 高 | 双轨架构：传统规则/评分卡 + LLM 辅助 + 强复核 |
| 涉及跨境或外部模型处理核心敏感数据 | 很高 | 优先本地化部署，外部调用需单独审批甚至禁止 |

所谓“双轨架构”，白话说就是两套判断并行运行：一套是传统规则、评分卡、检索证据，另一套是大模型生成摘要和提示。最终是否提交业务动作，不由大模型单独决定，而由人工或规则主链决定。

可以画成文字草图：

传统评分卡/规则引擎 -> 给出基础风险结论  
LLM 辅助链路 -> 生成摘要、解释、补充关注点  
人工复核 -> 对比两条结果 -> 决定是否进入下一步

这种架构特别适合金融风控和医疗辅助，因为它把大模型放在“增强调研和解释”位置，而不是“最终裁决”位置。

对于低风险业务，完全没必要上区块链存证。只要能做到：

1. 日志字段完整。
2. 权限有约束。
3. 能定期审计。
4. 输出不会直接伤害用户权益。

那用普通 append-only 日志就足够。区块链、WORM 存储、硬件时间戳这些手段，主要适用于审计要求极高、争议成本很高的场景。

---

## 参考资料

| 来源 | 核心贡献 | 适用章节 |
| --- | --- | --- |
| [cnblogs 文章](https://www.cnblogs.com/lyh-001/p/18939622?utm_source=openai) | 总结合规场景中的数据出境、审计留痕、人工复核要求 | 核心结论、问题定义 |
| [CSDN 文章](https://blog.csdn.net/sinat_28461591/article/details/147314529?utm_source=openai) | 给出审计链 `L`、在线/离线路径、双轨风控思路 | 核心机制、替代方案 |
| [Bestcoffer 资料](https://www.bestcoffer.com/how-bestcoffer-ai-redaction-solves-financial-data-compliance-issues/?utm_source=openai) | 脱敏在金融合规中的工程实践，适合作为真实案例参考 | 代码实现、工程权衡 |
| [Fanruan 资料](https://www.fanruan.com/finepedia/article/690dce58f7a2e712975aa6a8?utm_source=openai) | 梳理日志不可篡改、共享审批、解释性缺口等常见问题 | 工程权衡 |

新手可读顺序建议是：先看合规目标，再看审计链字段，再看在线/离线架构差异，最后理解为什么高风险场景必须保留人工复核。
