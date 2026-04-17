## 核心结论

Agent 执行链的异常，适合先按“错误发生在系统哪一层”来分类，再决定恢复动作。一个实用的四层框架是：

| 异常层级 | 白话解释 | 常见信号 | 首选恢复策略 | 误判代价 |
|---|---|---|---|---|
| 环境异常 | 外部世界坏了，不是模型先做错 | 429、500、超时、网络抖动 | 重试，通常配指数退避 | 误判成推理问题会白白重规划，浪费 token |
| 工具异常 | 调了工具，但参数或返回格式不合法 | 参数缺失、字段类型不对、schema 校验失败 | 参数修正、结构修复、重发工具调用 | 误判成环境问题会重复提交，可能写出脏数据 |
| 推理异常 | 模型“想错了”，比如幻觉、循环、误读结果 | 连续重复调用、解释自信但无证据、validation 连续失败 | 反思、重规划、切换更强模型或验证器 | 误判成环境问题会无限 retry |
| 目标异常 | 任务本身不可行，或前提已经失效 | 权限不足、资源不存在、业务规则禁止 | 终止、转人工、要求补充条件 | 误判成可恢复异常会持续烧钱 |

把这个框架写成公式，就是：

$$
C \in \{\text{环境}, \text{工具}, \text{推理}, \text{目标}\}, \quad S=f(C)
$$

这里 $C$ 是异常类别，$S$ 是恢复策略。核心不是“有错就重试”，而是“先分类，再恢复”。

对环境异常，常用指数退避：

$$
Wait_k = Base \times 2^k + Jitter
$$

其中 $Base$ 是初始等待时间，$k$ 是第几次失败，$Jitter$ 是随机抖动，作用是避免一批 Agent 同时重试，形成重试风暴。

玩具例子：天气查询 Agent 调城市 API，连续收到 429。若设 `Base=2s`，三次等待分别是 2、4、8 秒。第三次还失败，就不应继续盲目重试，而应进入“终止 + 人工介入”或“稍后重跑”。

真实工程例子：客服 Agent 调用工单系统创建 ticket。接口偶发 502，且返回 JSON 字段有过 schema 变更。如果系统只会 retry，可能创建重复工单；如果系统把返回结果先做 schema 校验、写操作再加幂等键，就能把“环境抖动”和“工具格式错误”分开处理。

---

## 问题定义与边界

“异常分类”讨论的不是 Python 里 `try/except` 那种语言级异常，而是 Agent 执行链里的运行失败。执行链指“模型思考 + 工具调用 + 状态更新 + 验证 + 下一步决策”这一串动作。

边界必须先说清：

1. 这里讨论的是生产 Agent，不是单轮聊天。
2. 重点是“恢复策略选择”，不是“如何让模型永远不犯错”。
3. 分类依据是可观测信号，也就是日志、状态码、校验结果、调用轨迹，而不是人的主观感觉。

很多初学者会把所有失败都看成“模型没理解”。这是错的。因为不同异常的入口信号完全不同。

| 观测信号 | 更可能的分类 | 为什么 |
|---|---|---|
| HTTP 429 / 503 / timeout | 环境异常 | 外部依赖暂时不可用 |
| `JSONDecodeError` / schema mismatch | 工具异常 | 输出结构不符合约束 |
| 同一工具同参数重复 3 次 | 推理异常 | 模型卡在循环里 |
| API 明确返回 `permission_denied` | 目标异常 | 任务前提不成立，不是重试能解决 |
| 输出很自信但拿不出来源 | 推理异常 | 典型幻觉信号 |
| 写操作成功但下游读回校验失败 | 工具异常或推理异常 | 需要看是返回格式变了，还是模型误读结果 |

判断边界的关键，是把“外部失败”和“模型造错”分开。外部失败通常有硬信号，例如 429、500、超时。幻觉没有硬错误码，它常常表现为“系统看起来成功，但答案不成立”。

所以生产系统必须有一层观测面。可以把它理解为“给 Agent 装仪表盘”。最小链路是：

`输入 -> 工具调用 -> 输出校验 -> 轨迹监控 -> 分类决策 -> 恢复/终止`

如果没有这条链路，系统只能靠 prompt 猜自己出了什么错。那不是恢复机制，而是碰运气。

---

## 核心机制与推导

四层分类之所以有价值，不是因为名字整齐，而是因为它把恢复动作收敛成有限集合。

### 1. 环境异常：先重试，不先重想

环境异常的假设是：当前执行逻辑没明显错，错在外部依赖暂时不可用。典型动作是指数退避。

若 `Base=2` 秒，失败次数从 0 开始，则：

$$
Wait_0=2 \times 2^0=2
$$

$$
Wait_1=2 \times 2^1=4
$$

$$
Wait_2=2 \times 2^2=8
$$

这不是数学装饰，而是工程节奏。它表达的是：失败越多，越不该立刻重试。因为系统已经给出“现在环境不适合继续”的信号。

但退避必须带跳出条件。一个最小状态机如下：

1. 收到 429/503/timeout。
2. 检查是否在 retryable 集合内。
3. 若重试次数未超阈值，则保存当前进度，等待后重跑该步骤。
4. 若超阈值，则标记为环境失败，进入人工或备用通道。
5. 不回滚到整条任务起点，只恢复失败节点。

这里“保存当前进度”很重要。否则一次 API 超时就要整条执行链重来，成本会被放大。

### 2. 工具异常：把错误反馈给模型，不要直接放过

工具异常的本质是“调用契约被破坏”。契约可以是输入参数，也可以是输出结构。最常见的修复模式是 Try-Rewrite-Retry，也就是“先试，出错后把具体错误回写，让模型重写，再重试”。

伪代码可以写成：

```text
call tool(args)
if tool_error:
    inject("参数错误：缺少 user_id，允许字段为 [user_id, email]")
    ask model to rewrite tool call
    retry once
```

这个模式有效，是因为模型对“具体错误提示”比对“重新想想”反应更好。前者缩小了解空间，后者只是重新采样。

### 3. 推理异常：不要拿 retry 假装修复

推理异常包括两类常见问题：

1. 幻觉：模型说出了没有证据支持的话。
2. 循环：模型不断重复相同策略，没有真实进展。

这类问题如果继续 retry，通常不会好，只会更贵。正确动作是触发反思、切换路线，或者直接让第二个 Validator Agent 做事实核查。Validator Agent 可以理解成“专门负责挑错的第二个 Agent”。

### 4. 目标异常：承认任务不可行

目标异常最容易被忽略。它不是系统坏了，而是当前目标在现有条件下不成立。例如：

- 用户要查的数据系统里根本没有。
- 当前账号无权限执行写入。
- 业务规则要求人工审批。
- 计划依赖的前提资源已被删除。

这种情况下，最优动作通常不是“更努力地试”，而是及时终止，并把缺失条件明确返回给人。

所以完整的恢复流，不是单一循环，而是分叉树：

| 分类结果 | 下一步 |
|---|---|
| 环境 | backoff retry |
| 工具 | 修正参数或修结构 |
| 推理 | 反思、重规划、验证 |
| 目标 | 终止并转人工 |

---

## 代码实现

下面给一个可运行的最小实现。它演示三件事：

1. 用明确分类把异常映射到策略。
2. 对可重试环境错误做指数退避。
3. 对工具输出做严格校验，避免“假成功”。

```python
from dataclasses import dataclass
from typing import Literal
import random


RetryableStatus = {429, 503, 504}


@dataclass
class TicketOutput:
    ticket_id: str
    status: Literal["created", "pending"]
    url: str


def validate_ticket_output(raw: dict) -> TicketOutput:
    required = {"ticket_id", "status", "url"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"schema_error: missing={sorted(missing)}")
    if raw["status"] not in {"created", "pending"}:
        raise ValueError(f"schema_error: bad_status={raw['status']}")
    return TicketOutput(
        ticket_id=str(raw["ticket_id"]),
        status=raw["status"],
        url=str(raw["url"]),
    )


def classify_error(signal: dict) -> str:
    if signal.get("http_status") in RetryableStatus or signal.get("timeout"):
        return "environment"
    if signal.get("schema_error") or signal.get("param_error"):
        return "tool"
    if signal.get("loop_detected") or signal.get("hallucination"):
        return "reasoning"
    if signal.get("permission_denied") or signal.get("impossible_goal"):
        return "goal"
    return "unknown"


def choose_strategy(category: str) -> str:
    mapping = {
        "environment": "retry",
        "tool": "rewrite_and_retry",
        "reasoning": "replan_or_validate",
        "goal": "terminate",
        "unknown": "terminate",
    }
    return mapping[category]


def backoff_seconds(base: int, attempt: int, jitter: bool = False) -> float:
    wait = base * (2 ** attempt)
    if jitter:
        wait += random.random()
    return wait


# 玩具例子：天气 API 连续 429
assert classify_error({"http_status": 429}) == "environment"
assert choose_strategy("environment") == "retry"
assert backoff_seconds(2, 0) == 2
assert backoff_seconds(2, 1) == 4
assert backoff_seconds(2, 2) == 8

# 真实工程例子：工单创建返回结构异常
bad_output = {"ticket_id": "t_1", "status": "ok", "url": "/tickets/t_1"}
try:
    validate_ticket_output(bad_output)
    raise AssertionError("should fail closed")
except ValueError as e:
    assert "schema_error" in str(e)

good_output = {"ticket_id": "t_2", "status": "created", "url": "/tickets/t_2"}
validated = validate_ticket_output(good_output)
assert validated.status == "created"
```

这段代码很小，但表达了工程上的几个硬规则：

| 规则 | 作用 |
|---|---|
| `RetryableStatus` 明确列出可重试状态 | 防止“所有错误都重试” |
| `validate_ticket_output` 失败即拒绝 | 防止工具异常悄悄传到下游 |
| `classify_error -> choose_strategy` | 把恢复逻辑写成可测的映射 |
| `assert` 覆盖关键路径 | 保证策略不是凭感觉改出来的 |

在真实工程里，还应再补三层：

1. 幂等键。幂等的意思是“同一请求重复执行，结果只算一次”。写操作如创建 ticket、发送邮件、退款，都要有 `idempotency_key`。
2. 电路断路器。断路器指“达到阈值就强制停止”的保护机制，例如 `max_steps`、`max_spend`、`same_tool_same_args >= 3`。
3. 结构化日志。至少记录 `run_id`、`step`、`tool_name`、`args_hash`、`error_tag`、`retry_count`。

---

## 工程权衡与常见坑

最大的坑，不是不会写重试，而是分不清什么时候不该重试。

### 坑 1：把环境问题当成推理问题

例如合同查询 Agent 调用内部 API 超时，却立刻让模型“重新规划”。结果通常是：

- token 增长；
- 上下文更长；
- 重复走到同一个坏掉的 API；
- 最后比简单 backoff 更慢、更贵。

正确顺序应该是：先看状态码和工具错误，再决定是否进入推理层反思。

### 坑 2：把工具错误当成环境问题

如果返回体 schema 已经变了，你继续 retry 只会重复收到同样的不合法结构。此时应优先校验并阻断，而不是假设“再来一次也许就正常”。

### 坑 3：写操作没有幂等，重试制造重复副作用

这是最贵的坑之一。比如工单 API 偶发 502，客户端以为没成功，连续 retry，后台其实已经创建成功，于是出现 34 张重复工单。系统表面上是在“恢复”，本质上是在放大损失。

### 坑 4：没有停止条件

如果没有断路器，推理异常很容易演变成无限循环。最少应限制：

| 指标 | 推荐用途 |
|---|---|
| `max_steps` | 防止执行链无限延长 |
| `max_retry_per_tool` | 防止单个工具被打爆 |
| `max_spend` | 防止 token 成本失控 |
| `same_call_repeats` | 防止逻辑循环 |
| `validation_fail_count` | 防止错误被无限改写 |

### 坑 5：目标已不可行，还在“努力恢复”

例如用户要求删除一个已归档且无权限访问的资源。此时最合理的动作是终止并说明原因，而不是继续尝试不同 prompt。目标异常不是模型能力问题，而是业务边界问题。

从收益看，恢复策略是一个成本选择题：

| 策略 | 好处 | 风险 |
|---|---|---|
| 持续 retry | 对短暂抖动有效 | 容易烧预算、打爆依赖 |
| 参数修正 | 对工具调用错误高效 | 需要明确错误信息 |
| 重规划 | 可跳出错误路线 | 成本高，可能掩盖环境问题 |
| 终止 + 人工 | 最稳妥 | 自动化率下降 |

工程上没有“永远正确”的默认动作。只有“当前证据下最便宜且最安全”的动作。

---

## 替代方案与适用边界

四层分类不是唯一方案，但它适合大多数带工具的 Agent。若任务特征不同，可以直接换更保守的系统设计。

### 1. 只读检索优先，不必强上行动 Agent

如果任务只是知识问答、流程说明、文档总结，通常用 RAG 就够。RAG 指“检索增强生成”，白话就是先查资料，再回答。此时系统边界更窄，异常面更小，也不涉及写操作副作用。

适用边界：

- 不需要创建、修改、删除数据；
- 输出可附来源；
- 错误代价主要是回答不准，而不是执行错动作。

### 2. 高风险写操作前置人工确认

如果任务涉及退款、账号变更、工单关闭、发布配置，不应让 Agent 直接提交。更稳的做法是“两阶段提交”：

1. Agent 只生成执行提案。
2. 人类或策略引擎批准后，工具才真正写入。

### 3. 引入 Validator Agent

当系统经常遇到“看起来成功，但结果不可信”时，可以加一个 Validator Agent。它不负责完成任务，只负责核验事实、检查证据、复核工具返回。这个方案适合高风险问答、合规、金融、医疗等场景。

### 4. 何时直接终止 Agent

下面这些情况，通常应直接终止，而不是继续恢复：

| 情况 | 建议 |
|---|---|
| 连续多次 validation fail 且错误模式相同 | 终止，排查 schema 或 prompt |
| 同一写操作因外部错误重试已达上限 | 停止自动重试，转人工 |
| 目标依赖的权限或资源不存在 | 终止并返回缺失条件 |
| 任务跨越高风险边界 | 切换到人工审批流 |

最终决策树可以压缩成一句话：

能确定是暂时性外部抖动，就重试；能确定是契约错误，就修参数或修结构；能确定是模型思路错了，就反思或重规划；能确定目标不成立，就停止。

---

## 参考资料

- Arun Baby, “Error Handling and Recovery”, 2026: https://www.arunbaby.com/ai-agents/0033-error-handling-recovery/
- StackAI, “Prevent AI Agent Hallucinations in Production Environments”, 2026-02-06: https://www.stackai.com/insights/prevent-ai-agent-hallucinations-in-production-environments
- Agent Patterns, “Why Agents Fail in Production (And How to Prevent It)”, 更新于 2026-03-07: https://www.agentpatterns.tech/en/failures/why-agents-fail
