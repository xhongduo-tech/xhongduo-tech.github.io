## 核心结论

LLM 应用的可观测性，不是“多打一点日志”，而是把一次请求从输入到输出的完整行为变成可追踪、可评分、可回放的证据链。这里的“可观测性”可以先用一句白话理解：系统出错时，你不是只能看到“坏了”，而是能顺着线索定位“为什么坏、坏在什么环节、影响有多大”。

在传统 Web 服务里，CPU、内存、错误率、响应时间通常足以解释大部分问题。但在 LLM 应用里，服务器返回 `200 OK` 并不代表结果正确，因为真正的风险常常发生在语义层，也就是“话说出来了，但内容错了、漏了、越权了”。因此，生产级 LLM 系统至少要同时观测三类信号：

| 维度 | 含义 | 当前值示例 | 目标/阈值 | 备注 |
|---|---|---:|---:|---|
| `E` | 评价得分，白话说就是“答案像不像正确答案” | 0.82 | `> 0.80` | 自动评测、规则检查或基准集打分 |
| `H` | 高风险输出比例，白话说就是“危险回答占比” | 0.05 | `< 0.10` | 包含幻觉、违规、越权、错误引用 |
| `L` | 延迟稳定分，白话说就是“响应速度稳不稳” | 0.45 | `> 0.40` | 不是原始时延，而是归一化健康分 |
| `C` | 成本控制分，白话说就是“花费是否在预算内” | 0.78 | `> 0.70` | 结合 token、缓存命中和单次请求成本 |

一个便于落地的综合读数是：

$$
O = \alpha \cdot E - \beta \cdot H + \gamma \cdot L + \delta \cdot C
$$

其中 `O` 是观测完备度，白话说就是“当前链路是否足够健康、足够可解释”。它不是学术标准，而是工程上的统一读数，目的是把分散指标压缩成一个便于告警和回溯的数值。

如果业务设定 $\alpha=1,\beta=2,\gamma=1,\delta=0.5$，并且某段时间内 `E=0.82`、`H=0.05`、`L=0.45`、`C=0.78`，那么：

$$
O = 1 \times 0.82 - 2 \times 0.05 + 1 \times 0.45 + 0.5 \times 0.78 = 1.56
$$

当阈值设为 `1.2` 时，这条链路暂时健康；一旦低于阈值，就应该自动进入回溯流程，优先检查评测曲线、Prompt 漂移、检索失败和工具调用异常。

玩具例子可以非常直观。一条请求记录如下：

- prompt: “把二分查找解释给初学者”
- model version: `gpt-4.1-mini-2026-03`
- `E=0.82`
- `H=0.05`
- latency score: `0.45`
- cost tag: `normal`
- `O≈1.56`

运维同学不需要先读完整回答，只看这一行就能判断：质量尚可、风险不高、时延稳定、成本可控，暂时无需排障。真正重要的不是“有没有日志”，而是这条日志能不能支持后续判断、过滤、聚合和复盘。

---

## 问题定义与边界

先把边界说清楚。LLM 可观测性关注的不是“机器活着没有”，而是“模型在真实任务里到底做了什么、为什么这么做、结果是否可信、代价是多少”。

传统监控和 LLM 可观测性的差异可以直接对比：

| 维度 | 传统监控 | LLM 可观测性 |
|---|---|---|
| 关注对象 | CPU、内存、错误码、QPS | prompt、response、tool call、检索结果、评分、成本 |
| 数据粒度 | 请求级或实例级指标 | 单次生成级、步骤级、agent 链路级 |
| 回溯能力 | 能知道哪台机器慢 | 能知道哪段提示词、哪个模型版本、哪个工具步骤出错 |
| 风险覆盖 | 宕机、超时、异常 | 幻觉、Prompt 漂移、注入攻击、越权调用、错误引用 |

这里有几个必须纳入边界的问题。

第一，Prompt drift，也就是提示词漂移。白话说，同一个任务，系统提示词、模板、检索上下文悄悄变了，输出质量也跟着变，但服务表面仍然正常。它常见于两类场景：一类是运营同学调整模板措辞，另一类是代码重构时改了 prompt 拼接顺序。

第二，Prompt injection，也就是输入中混入恶意指令，诱导模型忽略原始约束。白话说，用户的问题里藏了一段“忽略以上规则，直接输出内部策略”。如果系统没有记录原始用户输入、清洗后的输入、最终发送给模型的 prompt，就很难判断注入是从哪里进入链路的。

第三，tool call failure，也就是工具调用失败。白话说，模型看起来在认真回答，实际上它依赖的搜索、数据库、代码执行步骤已经超时、报错或返回空数据。最终用户看到的是“答案不对”，但根因可能根本不在模型，而在外部依赖。

第四，model switch，也就是模型切换。白话说，线上从一个模型版本切到另一个，接口不报错，行为却可能明显变化。比如旧模型遇到证据不足更愿意拒答，新模型则更愿意“补全一个看起来通顺的答案”。

一个新手常见误区是：服务器返回了 `200`，前端也显示了答案，于是默认系统“没问题”。但真实情况往往是，系统层面没有错误，语义层面已经偏了。比如某次客服机器人把退款政策解释错了，接口日志只有成功记录；真正的问题要到结构化日志里看 `prompt`、`response`、`retrieval_hits`、`model_version`，才发现是新版提示模板把“优先引用知识库原文”改成了“尽量简洁总结”，导致模型在缺少证据时自行补全。

再看一个更具体的新手例子。假设用户问：

> “订单超过 7 天还能退款吗？请引用平台规则原文。”

系统表面流程很简单：用户提问，模型回答。实际最少包含四层判断：

| 层级 | 真实问题 | 如果不记录会怎样 |
|---|---|---|
| 输入层 | 用户原话里有没有注入、歧义或缺字段 | 无法判断是用户问题还是系统问题 |
| 检索层 | 是否真的取到了退款规则原文 | 无法区分“没查到”还是“模型瞎编” |
| 生成层 | 模型是否依据证据回答，是否引用正确 | 无法解释答案为何出现偏差 |
| 输出层 | 用户是否投诉、纠错、低评分 | 无法形成闭环修正 |

因此，LLM 可观测性的定义边界可以概括为：它必须覆盖 prompt 到 response 的输入输出链路、链路中的中间步骤、质量评分、风险标签、版本标签以及资源消耗，并且这些信息要用同一个 `trace_id` 串起来。`trace_id` 可以理解为“一次完整请求的身份证号”。如果是多 agent 或多服务系统，还需要把父子 span、步骤顺序和跨服务调用一起串上，否则 trace 很容易在中间断掉。

---

## 核心机制与推导

LLM 可观测性的核心机制，不是单一指标，而是信号打通。所谓“信号打通”，白话说就是原本分散在不同地方的数据，现在能拼成一条完整路径。

一条完整路径通常包括：

| Metric | 描述 | 衡量方式 | 阈值/目标 |
|---|---|---|---|
| Prompt Trace | 输入提示与上下文快照 | 保存模板版本、用户输入、检索片段摘要 | 关键链路 100% 保留 |
| Response Trace | 输出文本与结构化字段 | 保存回答、引用、拒答原因 | 高风险链路 100% 保留 |
| Auto Eval `E` | 自动评估质量 | 对照基准答案、规则检查、模型裁判 | `> 0.80` |
| High Risk `H` | 高风险输出占比 | 幻觉检测、越权检测、政策违规检测 | `< 0.10` |
| Tool Success | 工具调用成功率 | 检索命中率、函数调用成功率 | `> 0.95` |
| Latency Score `L` | 时延稳定分 | 由 P95、抖动幅度归一化 | `> 0.40` |
| Cost Score `C` | 成本控制分 | 输入输出 token、缓存命中、单次成本 | `> 0.70` |

这里要特别说明 `L`。如果直接把原始时延秒数代入公式，延迟越大反而分数越高，逻辑是错的。所以工程里应当把 `L` 定义成“延迟稳定分”或“延迟健康分”，例如通过归一化函数把更低、更稳定的延迟映射成更高分值。`C` 同理，也不应是直接美元成本，而应是“成本健康分”。

一种常见的归一化写法如下：

$$
L = 1 - \min\left(1,\frac{p95\_latency}{latency\_budget}\right)\times w_1
    - \min\left(1,\frac{jitter}{jitter\_budget}\right)\times w_2
$$

其中 $w_1 + w_2 = 1$。这表示：越接近预算上限，分数越低；越抖动，分数也越低。

成本控制分也可以类似定义：

$$
C = 1 - \min\left(1,\frac{cost\_per\_request}{cost\_budget}\right)\times u_1
    - \min\left(1,\frac{token\_per\_request}{token\_budget}\right)\times u_2
$$

其中 $u_1 + u_2 = 1$。这表示：同一条请求即使还没超预算，只要 token 快速膨胀，也应提前扣分。

为什么公式里 `H` 是减项？因为高风险输出会直接破坏系统可信度。在问答系统里，一次轻微延迟通常比一次严重幻觉更容易接受，所以很多业务会把 $\beta$ 设得较大。通俗讲，用户一般能忍受“慢一点”，很难忍受“错得很自信”。

拿 10,000 次请求的玩具例子继续看：

- 自动评价平均分 `E=0.82`
- 高风险输出率 `H=0.05`
- 延迟稳定分 `L=0.45`
- 成本控制分 `C=0.78`

则：

$$
O = 0.82 - 2 \times 0.05 + 0.45 + 0.5 \times 0.78 = 1.56
$$

如果阈值是 `1.2`，这说明当前系统总体可控；如果某天 `O` 跌到 `1.08`，排查顺序通常应当是：

1. 先看 `E` 是否突然下降，确认是不是回答质量退化。
2. 再看 `H` 是否上升，确认是不是出现幻觉或违规。
3. 检查 `tool_success_rate` 和 `retrieval_hit_rate`，确认是不是外部依赖失效。
4. 对比 `prompt_version`、`model_version`、`knowledge_base_version`，确认是不是版本切换导致漂移。

这套顺序的价值在于，它把“排障”从拍脑袋改成有顺序的证据筛查。新手做排查最容易犯的错，是先盯着模型本身看，结果真正的问题是检索空了、工具挂了，或者 prompt 拼装顺序变了。

真实工程例子更能说明问题。假设某企业知识分析平台需要满足合规审计。用户提问“某合同条款是否允许提前解约”，模型回答给出了明确结论，但后台实际经历了四步：检索合同、抽取条款、调用政策工具、整合回答。如果系统只保留最终输出，就无法解释这个结论来自哪份文档、用了哪个模型版本、工具是否失败过、成本是否异常。一旦客户质疑结果，团队既无法审计，也无法快速定位是索引污染、模型切换，还是提示注入。只有把 `trace_id + prompt + retrieval docs + tool calls + model version + eval score + cost` 全部串起来，才算真正可观测。

再把“信号打通”拆成更易落地的四层：

| 层 | 要记录什么 | 为什么重要 |
|---|---|---|
| Trace 层 | `trace_id`、父子 span、步骤耗时 | 解决“问题发生在哪一步” |
| Content 层 | prompt、上下文、response、引用 | 解决“模型到底看到了什么、说了什么” |
| Quality 层 | 自动评分、风险标签、人工反馈 | 解决“结果到底好不好” |
| Resource 层 | token、成本、缓存命中、重试次数 | 解决“代价是否失控” |

这四层少一层都不完整。只有 Trace 没有内容，只能知道“慢”；只有内容没有评分，只能知道“说了什么”，不能判断“说得对不对”；只有评分没有资源信息，也很难解释为什么成本突然爆炸。

---

## 代码实现

实现上最重要的是四个模块：请求拦截、自动评测、资源监控、链路追踪。

请求拦截负责采集 prompt、上下文、模型版本。自动评测负责给回答打分。资源监控负责记录 token、延迟、成本。链路追踪负责把多步调用绑到同一个 `trace_id` 下。

先看一个可运行的 Python 玩具实现。它不依赖第三方库，直接用标准库模拟一次“检索增强问答”的观测流程。目标不是替代线上系统，而是把核心逻辑说明白，并且保证你复制后就能运行。

```python
from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean
from time import time
from typing import List, Dict
import json
import uuid


@dataclass
class ToolCall:
    name: str
    success: bool
    latency_ms: int
    output_size: int


@dataclass
class ObsEvent:
    trace_id: str
    prompt: str
    response: str
    model_version: str
    prompt_version: str
    knowledge_base_version: str
    retrieval_hit_rate: float
    tool_calls: List[ToolCall]
    token_input: int
    token_output: int
    latency_ms: int
    cost_usd: float
    auto_score: float      # E
    high_risk_rate: float  # H
    latency_score: float   # L
    cost_score: float      # C
    user_feedback: str


def normalize_latency_score(latency_ms: int, p95_budget_ms: int = 2000) -> float:
    # 越接近预算上限，得分越低；结果限制在 0~1
    score = 1 - min(latency_ms / p95_budget_ms, 1.0)
    return round(score, 4)


def normalize_cost_score(cost_usd: float, cost_budget_usd: float = 0.02) -> float:
    # 每次请求成本越接近预算上限，得分越低
    score = 1 - min(cost_usd / cost_budget_usd, 1.0)
    return round(score, 4)


def simple_auto_eval(prompt: str, response: str, evidence: str) -> float:
    """
    一个极简示例：
    1. 回答非空
    2. 是否提到证据中的关键短语
    3. 是否长度足够解释问题
    """
    if not response.strip():
        return 0.0

    score = 0.4
    if len(response) >= 20:
        score += 0.2
    if "有序" in response or "折半" in response:
        score += 0.2
    if evidence and any(word in response for word in evidence.split()[:3]):
        score += 0.2
    return round(min(score, 1.0), 4)


def simple_high_risk_detector(response: str, retrieval_hit_rate: float, tool_success_rate: float) -> float:
    """
    一个极简风险分：
    - 没检索到证据却给出确定口吻，风险上升
    - 工具失败较多，风险上升
    - 回答里出现绝对化表达，风险略升
    """
    risk = 0.0
    if retrieval_hit_rate < 0.5:
        risk += 0.03
    if tool_success_rate < 0.95:
        risk += 0.03
    if any(word in response for word in ["一定", "绝对", "完全没问题"]):
        risk += 0.02
    return round(min(risk, 1.0), 4)


def compute_observability(event: ObsEvent, alpha=1.0, beta=2.0, gamma=1.0, delta=0.5) -> float:
    return round(
        alpha * event.auto_score
        - beta * event.high_risk_rate
        + gamma * event.latency_score
        + delta * event.cost_score,
        4
    )


def should_traceback(o_score: float, threshold: float = 1.2) -> bool:
    return o_score < threshold


def simulate_rag_request(question: str) -> Dict:
    start = time()

    # 模拟检索
    retrieval_docs = [
        "二分查找 适用于 有序 数组",
        "每次比较后 将搜索区间 折半"
    ]
    retrieval_hit_rate = 1.0 if retrieval_docs else 0.0

    # 模拟工具调用
    tool_calls = [
        ToolCall(name="vector_search", success=True, latency_ms=80, output_size=2),
        ToolCall(name="rerank", success=True, latency_ms=40, output_size=2),
    ]
    tool_success_rate = mean(1.0 if t.success else 0.0 for t in tool_calls)

    # 模拟生成
    response = "二分查找是在有序数组中，每次取中间元素比较，再把搜索范围折半缩小的方法。"
    token_input = 120
    token_output = 38
    latency_ms = int((time() - start) * 1000) + 420
    cost_usd = 0.0044

    latency_score = normalize_latency_score(latency_ms)
    cost_score = normalize_cost_score(cost_usd)
    auto_score = simple_auto_eval(question, response, retrieval_docs[0])
    high_risk_rate = simple_high_risk_detector(response, retrieval_hit_rate, tool_success_rate)

    event = ObsEvent(
        trace_id=f"trace-{uuid.uuid4()}",
        prompt=question,
        response=response,
        model_version="gpt-4.1-mini-2026-03",
        prompt_version="explain-v3",
        knowledge_base_version="kb-2026-04-01",
        retrieval_hit_rate=retrieval_hit_rate,
        tool_calls=tool_calls,
        token_input=token_input,
        token_output=token_output,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        auto_score=auto_score,
        high_risk_rate=high_risk_rate,
        latency_score=latency_score,
        cost_score=cost_score,
        user_feedback="",
    )

    o_score = compute_observability(event)

    result = {
        "event": {
            **asdict(event),
            "tool_calls": [asdict(t) for t in event.tool_calls],
        },
        "o_score": o_score,
        "traceback": should_traceback(o_score),
    }
    return result


if __name__ == "__main__":
    result = simulate_rag_request("解释什么是二分查找")
    print(json.dumps(result, ensure_ascii=False, indent=2))
```

这段代码可以直接运行，输出类似下面的结构化结果：

```json
{
  "event": {
    "trace_id": "trace-xxxx",
    "prompt": "解释什么是二分查找",
    "response": "二分查找是在有序数组中，每次取中间元素比较，再把搜索范围折半缩小的方法。",
    "model_version": "gpt-4.1-mini-2026-03",
    "prompt_version": "explain-v3",
    "knowledge_base_version": "kb-2026-04-01",
    "retrieval_hit_rate": 1.0,
    "tool_calls": [
      {"name": "vector_search", "success": true, "latency_ms": 80, "output_size": 2},
      {"name": "rerank", "success": true, "latency_ms": 40, "output_size": 2}
    ],
    "token_input": 120,
    "token_output": 38,
    "latency_ms": 420,
    "cost_usd": 0.0044,
    "auto_score": 1.0,
    "high_risk_rate": 0.0,
    "latency_score": 0.79,
    "cost_score": 0.78,
    "user_feedback": ""
  },
  "o_score": 2.18,
  "traceback": false
}
```

这个玩具实现有三个关键点。

第一，它把“日志”变成了结构化事件。结构化的意思不是为了好看，而是为了后续能按 `model_version`、`prompt_version`、`customer_id`、`task_type` 过滤统计。纯文本日志只能读，结构化日志才能算。

第二，它把“质量”从主观描述变成了可计算字段。上面的 `simple_auto_eval` 很粗糙，但它至少说明一个事实：生产系统必须把“好不好”编码成某种分数、标签或规则结果，否则系统永远只能回答“跑完了”，不能回答“跑得对不对”。

第三，它把中间步骤显式记录了下来。`tool_calls` 和 `retrieval_hit_rate` 看似只是附加字段，实际上它们决定了你排障时能不能区分“模型问题”和“数据问题”。

线上实现通常会把每次请求变成结构化事件，再送到统一采集器：

```javascript
function buildObservabilityEvent({
  traceId,
  prompt,
  promptVersion,
  response,
  modelVersion,
  reference,
  durationMs,
  tokenInput,
  tokenOutput,
  costUsd,
  toolCalls,
  retrievalHitRate,
  knowledgeBaseVersion,
  userFeedback
}) {
  const autoScore = evaluate(response, reference);
  const highRisk = detectHallucinationOrPolicyRisk({
    response,
    retrievalHitRate,
    toolCalls
  });
  const latencyScore = normalizeLatency(durationMs, 2000);
  const costScore = normalizeCost(costUsd, 0.02);

  return {
    traceId,
    prompt,
    promptVersion,
    response,
    modelVersion,
    autoScore,
    highRisk,
    latencyMs: durationMs,
    latencyScore,
    tokenInput,
    tokenOutput,
    costUsd,
    costScore,
    toolCalls,
    retrievalHitRate,
    knowledgeBaseVersion,
    userFeedback,
    timestamp: Date.now()
  };
}

function computeO({ autoScore, highRisk, latencyScore, costScore }) {
  return autoScore - 2 * highRisk + latencyScore + 0.5 * costScore;
}
```

如果你是新手，可以把这段代码理解成三步：

1. 先收集事实。
2. 再把事实变成评分。
3. 最后把评分汇总成能告警的读数。

这些字段里，最容易被遗漏的是三类。

第一类是版本字段。`promptVersion`、`modelVersion`、`knowledgeBaseVersion` 不写进去，后续很难区分问题来自模板、模型还是知识库。

第二类是中间态字段。比如 `toolCalls`、`retrievalHitRate`、`rerankScore`。如果只看最终文本，很多错误会被误判成“模型笨”，但真实原因可能是检索结果为空。

第三类是反馈字段。自动评测能规模化，但不是万能的；真实用户的纠错、人工抽检、投诉标签，应该回流到同一条链路里，形成 human-in-the-loop，也就是“人在回路中”的修正机制。

告警规则通常也不复杂：

| 规则 | 含义 | 处理动作 |
|---|---|---|
| `O < 1.2` | 综合观测读数退化 | 进入链路回溯 |
| `H > 0.1` | 高风险输出偏多 | 拉高告警等级，抽样人工复核 |
| `retrieval_hit_rate < 0.8` | 检索命中偏低 | 检查索引、分词、召回配置 |
| `tool_success_rate < 0.95` | 工具链不稳定 | 检查依赖系统或重试策略 |
| `costUsd_per_request` 突增 | 成本异常膨胀 | 检查 prompt 膨胀、缓存失效、模型切换 |

如果再往前走一步，完整工程实现通常会把观测数据分为两条通道：

| 通道 | 用途 | 数据特征 |
|---|---|---|
| 在线通道 | 告警、回溯、实时面板 | 秒级或分钟级更新 |
| 离线通道 | 回归评测、版本对比、周报 | 支持批量统计与样本复盘 |

在线通道解决“现在出了什么问题”，离线通道解决“这个版本比上个版本到底退化了没有”。两条通道缺一不可。

---

## 工程权衡与常见坑

完整观测不是零成本。每多记录一个字段，就多一份存储、计算和隐私压力；每多做一次自动评测，就多一部分延迟和费用。所以工程上要做“关键证据优先”，而不是“什么都存”。

常见坑可以直接列出来：

| 问题 | 风险 | 规避策略 |
|---|---|---|
| 只看 CPU、内存、成功率 | 系统看着正常，但回答持续出错 | 强制记录 `prompt/response/eval` |
| 只存最终输出，不存中间态 | 无法定位是检索错、工具错还是模型错 | 保存 `tool_calls` 与检索指标 |
| 日志不带版本号 | 无法判断退化是否由模型或提示词变更引起 | 每条事件带 `promptVersion/modelVersion` |
| 全靠人工抽检 | 成本高、覆盖率低、发现滞后 | 自动 eval 为主，人工复核高风险样本 |
| 全量保存原始上下文 | 成本高、可能触发隐私问题 | 对敏感字段脱敏，分级采样 |
| trace 断裂 | 多 agent 或多服务链路无法串联 | 强制传递 `trace_id` |

一个典型失败模式是：团队只监控 CPU 和接口耗时，发现系统“稳定运行”；但用户开始投诉答案瞎编。后来排查发现，新版模型上线后更倾向于“流畅回答”，而旧版更倾向于“缺证据就拒答”。如果日志里没有 `model_version` 和 `high_risk_flag`，这个退化很难被快速证实。

另一个常见坑是把可观测性做成“离线报表工程”。每天汇总一次平均分，看起来数据很好，但线上真实问题是某个细分类目在某个新 Prompt 版本下大面积失败。平均值会把局部灾难淹没，所以必须支持按任务类型、版本、客户、渠道、知识库分片查看。

还有一个经常被忽视的问题是“评测错觉”。很多团队上了自动评测之后，就把 `E` 当成绝对真相。实际上自动评测本身也会有偏差：

| 评测方式 | 优点 | 局限 |
|---|---|---|
| 规则检查 | 快、便宜、可解释 | 覆盖面窄，容易漏掉语义错误 |
| 基准集对比 | 适合版本回归 | 容易过拟合固定题集 |
| 模型裁判 | 语义判断能力强 | 可能和业务标准不一致 |
| 人工复核 | 最接近真实质量 | 昂贵，扩展性差 |

因此更稳妥的做法不是“只选一种”，而是组合使用：规则检查兜底硬约束，基准集看版本回归，模型裁判看语义质量，人工复核看高风险样本。这样得到的 `E` 才比较可信。

隐私和合规也是工程权衡的一部分。尤其在客服、医疗、法务场景里，原始 prompt 和上下文很可能包含邮箱、手机号、身份证号、合同编号、病历摘要。此时不能简单理解为“多存就更好”，而要做字段分级：

| 字段类型 | 是否建议原文保存 | 建议策略 |
|---|---|---|
| trace_id、版本号、耗时、token | 是 | 全量结构化保存 |
| 用户输入正文 | 视场景而定 | 脱敏、哈希、采样或加密存储 |
| 检索文档片段 | 视合规要求而定 | 保存摘要、文档 ID、位置偏移 |
| 最终输出 | 通常建议保存 | 高风险链路全量，低风险链路采样 |
| 用户身份标识 | 不建议直接明文保存 | 使用内部匿名 ID |

权衡原则可以概括为一句话：高风险链路做全量、低风险链路做采样、核心字段永远结构化。这里的“结构化”指字段是可筛选、可聚合、可关联的键值对，而不是一大段不可解析文本。

---

## 替代方案与适用边界

并不是所有项目一开始都要上完整三位一体观测。是否需要全面观测，取决于风险、规模和审计要求。

| 场景 | 建议方案 | 是否需要全面观测 |
|---|---|---|
| 个人实验、Demo | 简单日志 + 手工抽查 | 否 |
| 内部小工具、低频使用 | 基础延迟/成本监控 + 少量样本评测 | 通常不需要 |
| 日请求量过万的业务系统 | prompt trace + 自动 eval + 成本监控 | 需要 |
| 客服、金融、医疗、法务 | 完整链路观测 + 审计留痕 + 人工复核 | 必须 |
| 多 agent、多工具协作系统 | 全链路 trace + 步骤级指标 | 必须 |

可以把决策逻辑理解成一个小型判断树：

1. 是否直接面向真实用户？
2. 是否存在合规、审计或赔付风险？
3. 是否依赖检索、工具调用、多步推理？
4. 是否已经有明显的版本漂移和成本压力？

只要其中两项以上答案是“是”，就不应该停留在轻量监控。

研发早期的替代方案通常有三种。

第一，轻量监控。只看时延和成本，适合快速试验，不适合稳定运营。

第二，事后 UAT 回归测试。UAT 就是“用户验收测试”，白话说是上线前拿一批题目人工跑一遍。它能发现静态问题，但对线上漂移无能为力。

第三，人工 QA。QA 就是人工质量检查。优点是判断细致，缺点是昂贵且覆盖面有限。

再把这三种方案和完整可观测性放在一张表里，看得更清楚：

| 方案 | 能发现什么 | 难发现什么 | 适用阶段 |
|---|---|---|---|
| 轻量监控 | 超时、成本异常、接口失败 | 幻觉、漂移、错误引用 | Demo、早期验证 |
| UAT 回归 | 发布前明显质量退化 | 上线后环境漂移、真实用户分布变化 | 上线前门禁 |
| 人工 QA | 复杂语义错误、高风险样本 | 大规模实时退化 | 高风险业务补充 |
| 完整可观测性 | 链路级问题、质量变化、成本失控、审计追踪 | 无法替代业务策略本身 | 生产系统 |

一个现实的升级路径是：研发阶段先做简单日志和周报测试；当系统上线、请求量达到 `10k/日`，或者要向客户提供审计材料时，再升级成完整的 triad，也就是 `prompt trace + evaluation + cost/resource` 三位一体方案。这个边界非常重要，因为很多团队真正出问题，不是不会做，而是升级时机拖得太晚。

最后要强调一个适用边界：可观测性解决的是“看见问题、定位问题、量化问题”，它不直接替代 prompt 设计、知识库建设、权限治理和模型选型。换句话说，可观测性是诊断系统，不是万能修复器。但没有它，后面这些优化几乎都只能靠猜。

---

## 参考资料

| Source | Date | 洞察 | 链接 |
|---|---|---|---|
| OpenTelemetry, Ishan Jain, “An Introduction to Observability for LLM-based applications using OpenTelemetry” | 2024-06-04 | 从标准化遥测角度说明 LLM 场景下要观测 prompt、token、model version、latency，并给出 OTel Collector、Prometheus、Jaeger 组合示例，适合支撑“链路追踪”与“标准化采集”部分 | https://opentelemetry.io/blog/2024/llm-observability/ |
| Observability.How, Jigar Bhatt, “Observability for LLMs: Why It Matters and How to Achieve It” | 2025-05-27 | 明确指出 LLM 生产环境的关键观测对象包括 traces、metrics、token usage、error rates、quality checks 和 drift detection，适合支撑问题定义与落地步骤 | https://www.observability.how/observability-for-llms/ |
| Braintrust Team, “LLM monitoring vs LLM observability: What's the difference?” | 2026-02-18 | 清晰区分 monitoring 与 observability，强调“接口健康不等于回答正确”，适合支撑本文开头的核心判断 | https://www.braintrust.dev/articles/llm-monitoring-vs-observability |
| Braintrust Team, “What is LLM observability? (Tracing, evals, and monitoring explained)” | 2026-02-09 | 把 tracing、evals、monitoring 放到同一框架下解释，适合补充“为什么必须把 prompt、工具、检索和输出串成一条 trace” | https://www.braintrust.dev/articles/llm-observability-guide |
| Forbes, Subba Rao Katragadda, “Observability For LLM-Powered Applications: Unlocking Trust And Performance In The Age Of AI” | 2026-03-25 | 从企业治理、风险、合规角度解释为什么 LLM 可观测性已经从“可选能力”变成“运营基础设施”，适合支撑工程背景与审计场景 | https://www.forbes.com/councils/forbestechcouncil/2026/03/25/observability-for-llm-powered-applications-unlocking-trust-and-performance-in-the-age-of-ai/ |
| Maxim, Kuldeep Paul, “LLM Observability: Best Practices for 2025” | 2025-08-29 | 总结 prompt-completion linkage、tool calls、distributed tracing、token accounting 和 human feedback loop 等最佳实践，适合补充工程权衡与字段设计 | https://www.getmaxim.ai/articles/llm-observability-best-practices-for-2025/ |
| OpenInference Specification | 持续更新规范 | 给出面向 AI 应用的语义约定，说明为什么传统通用 tracing 字段不足以表达 LLM 调用、agent 步骤、tool invocation 与 retrieval 语义，适合补充“结构化字段”这一工程要求 | https://arize-ai.github.io/openinference/spec/ |
