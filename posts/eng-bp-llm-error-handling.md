## 核心结论

LLM 应用里的“错误处理”，不是只处理接口报错，而是把“这次回答到底对不对、能不能用、是否安全”变成可诊断、可追责、可修复的工程系统。这里的可观测性，白话讲，就是系统在出错时能把“错在哪里”主动暴露出来，而不是靠人猜。

核心结论有三条。

第一，必须把错误定义扩展到“语义错误”。语义错误，白话讲，就是程序返回了一个看起来正常的答案，但答案本身不扎根、不符合规则，或者引用了错误依据。对传统 Web 服务，`200 OK` 往往意味着请求成功；对 LLM 应用，`200 OK` 只说明模型输出了一段文本，不说明文本正确。

第二，错误处理必须覆盖全链路，而不是只盯模型调用。一个问答系统通常至少有这些阶段：输入预处理、检索、提示构建、模型生成、工具执行、结果后处理。用户看到的一次“答错”，可能根因在检索没召回文档，也可能在提示词丢了约束，还可能在工具返回脏数据。新手最容易犯的错，是把所有问题都归到模型“不稳定”。

第三，真正有效的诊断闭环来自三类观测柱：趋势指标 $M$、请求级 trace $T$、故障分类 $F$。趋势指标，白话讲，就是看一段时间内整体表现有没有退化；trace，白话讲，就是给一次请求留下完整执行轨迹；故障分类，白话讲，就是把“错了”继续细分成可行动的问题类型。三者共享同一个请求 ID，才能从“某个用户说答错了”一路定位到“这次检索 miss 导致模型 unsupported”。

先看一个新手版玩具例子。假设一个问答请求先检索文档，再调用模型回答。如果你只监控接口是否超时，那么用户投诉“答案胡编”时，你只能翻海量日志；如果你在检索阶段记录命中文档数和分数，在模型阶段记录 token、模型版本和输出标签，那么你很快就能判断：是检索没找到内容，还是模型没按要求引用来源，还是后处理错误裁剪了答案。

下表可以把三类观测柱放在同一张图里看：

| 观测柱 | 关注对象 | 可采集数据 | 解决的问题 |
| --- | --- | --- | --- |
| $M$ 指标 | 整体趋势 | 有据回答率、延迟、token 成本、拒答率、安全拦截率 | 是否出现大面积退化 |
| $T$ trace | 单次请求过程 | request_id、阶段耗时、召回结果、prompt、模型版本、工具输入输出 | 具体哪一步出错 |
| $F$ 故障分类 | 错误类型 | retrieval miss、hallucination、tool bug、policy violation、guardrail fail | 应该改检索、改提示、改工具还是改策略 |

如果只能记住一句话，那就是：LLM 应用的错误处理，本质上是把“模型调用”升级成“答案正确性系统”。

---

## 问题定义与边界

要讨论错误处理，先要定义什么叫“错”。在 LLM 应用里，错误不等于 HTTP 5xx。5xx 是服务失败，白话讲，就是系统根本没把请求处理完；而 LLM 错误更多是“处理完了，但结果不能用”。

一个可操作的定义是：只要输出不满足预先定义的可接受条件，就算错误。常见条件有三类。

第一类是扎根性。扎根，白话讲，就是回答能对应到输入材料、检索文档或工具返回值，而不是模型自己补全。RAG 问答里，回答没有依据，就算语言流畅，也应视为错误。

第二类是合法性。合法，白话讲，就是输出满足格式、字段、调用协议等硬约束。比如要求模型输出 JSON，结果多了一段解释文字；要求工具参数是整数，结果给了自然语言，这也是错误。

第三类是合规性。合规，白话讲，就是输出不能违反安全、隐私和业务规则。一个客服机器人泄露内部政策，或者一个代码助手建议危险命令，即使“技术上回答对了”，工程上也应算失败。

因此，问题边界不是“模型有没有返回文本”，而是“文本是否达到业务可用标准”。这意味着每一层都要打点和评估，而不是只在最后看用户反馈。可以把简化流程理解为：

输入 → 检索 → 提示构建 → 模型 → 工具 → 输出校验

这条链上的每一层，都可能引入自己的错误类型：

| 阶段 | 常见错误 | 典型后果 |
| --- | --- | --- |
| 输入 | 用户意图解析错、上下文丢失 | 后续阶段全部建立在错误问题上 |
| 检索 | 文档没召回、召回错文档 | 模型“有理有据地答错” |
| 提示构建 | 系统约束缺失、变量拼接错 | 模型跑偏或忽略引用要求 |
| 模型 | 幻觉、格式错误、过度自信 | 输出看似完整但不可信 |
| 工具 | API 参数错、返回脏数据 | 模型基于错误外部事实生成 |
| 输出校验 | 没拦截敏感内容、没检查结构 | 错误直接暴露给用户 |

为了描述这个覆盖范围，可以引入一个诊断覆盖度模型：

$$
C = M + T + F
$$

这里的 $C$ 不是严格数学意义上的加法，而是工程视角下的覆盖组合。它表达的是：如果只有指标 $M$，你只能知道“整体坏了”；如果只有 trace $T$，你能看到过程，但不知道哪些错误最常见；如果只有故障分类 $F$，你知道错因名称，却难以回放上下文。三者结合，才覆盖了趋势、过程、归因三个维度。

所以边界要画清楚：错误处理不只是异常捕获，而是围绕“可接受输出”的全链路诊断设计。

---

## 核心机制与推导

三类观测柱为什么必须同时存在，可以从它们解决的问题不同来推导。

先看 $M$。趋势指标 $M$ 监控的是一段时间内的整体状态，例如有据回答率、平均延迟、P95 延迟、单请求 token 成本、安全拦截率。它的价值是预警。预警，白话讲，就是系统还没完全崩，但已经出现可检测的退化信号。比如本周“有据回答率”从 96% 掉到 89%，就算接口成功率仍是 99.9%，也说明系统质量在变差。

再看 $T$。trace $T$ 记录的是单次请求的执行轨迹。轨迹，白话讲，就是把一次请求拆成多个阶段，每阶段发生了什么都能还原。它的价值是定位。没有 trace，你只能看到最后错误；有 trace，你能看到“检索只命中 1 篇低分文档”“prompt 拼接遗漏了 policy block”“模型输出前工具超时 fallback 到缓存”。

最后看 $F$。故障分类 $F$ 是把错误映射到稳定标签。稳定标签，白话讲，就是不同人看同一种错，会归到同一类。它的价值是治理。因为工程团队不是为“某次错误”修系统，而是为“某类错误反复出现”改机制。比如把错误统一分成 `retrieval_miss`、`unsupported_claim`、`tool_failure`、`policy_violation`，你才能知道预算该投在 reranker、prompt、工具重试，还是 guardrail。

这三者组合后，就形成闭环：

1. 指标发现退化。
2. trace 还原具体请求。
3. 故障分类聚合根因。
4. 改动系统。
5. 再由指标验证是否修复。

看一个最小数值例子。假设团队定义“有据回答率”阈值为 92%。某天抽样 100 次请求，其中 12 次被标记为 `unsupported`，也就是答案里的关键结论没有被检索证据支撑。那么：

$$
\text{有据回答率} = \frac{100 - 12}{100} = 88\%
$$

88% 低于 92%，因此 $M$ 报警。接着工程师查看这 12 次请求的 trace，发现它们都出现在“长尾问题 + 旧版 reranker”场景里，且检索阶段 top-3 文档分数普遍偏低，这说明根因更像 `retrieval_miss`，而不是模型随机幻觉。于是团队替换 reranker，并增加“低召回时触发拒答”的策略。修复后再次抽样 100 次，只剩 4 次 unsupported：

$$
\text{修复后有据回答率} = \frac{100 - 4}{100} = 96\%
$$

指标回到阈值之上，闭环完成。

这个例子说明两个工程事实。第一，很多“模型答错”其实是上游供给错。第二，如果没有共享请求 ID，把指标、trace、分类串起来，你只能看到 88% 这个数字，却不知道为什么是 88%。

真实工程例子更明显。假设你做一个基于内部文档的企业问答助手。一次用户查询会经历：文本嵌入、向量检索、重排、拼接 prompt、调用模型、必要时调用权限检查工具。用户最后收到一句错误答案：“公司报销政策允许个人采购后全额报销。”如果系统只记录模型输出，那么问题像是“模型瞎说”；但如果你有 span 级 trace，可能会发现是检索命中了两份过期制度文档，重排器把新版政策排到了第五，prompt 截断时新版文档又被切掉。根因其实是“过期知识 + 截断策略”，不是单纯模型幻觉。

为了方便落地，可以用下表区分三者各自追踪什么：

| 维度 | 追踪数据 | 典型问题 | 代表性警报 |
| --- | --- | --- | --- |
| $M$ | groundedness、延迟、token、成本、安全命中率 | 大面积质量退化、成本暴涨 | “近 1 小时 unsupported 超过阈值” |
| $T$ | request_id、span、模型版本、召回文档、prompt、工具输入输出 | 单请求根因不清 | “本次请求检索 hit=0” |
| $F$ | retrieval_miss、hallucination、tool_bug、policy_fail | 无法做错误治理 | “retrieval_miss 连续三天占比第一” |

---

## 代码实现

代码层面的关键原则很简单：在请求入口生成统一请求 ID，把每一层都记录成结构化事件。结构化事件，白话讲，就是不是随手打印一行字符串，而是把状态、原因、指标、token、耗时这些字段按固定结构写下来，方便后续检索和统计。

下面先看一个可运行的 Python 玩具实现。它不依赖任何外部库，只模拟三个阶段：检索、模型、故障分类。

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any
import uuid


@dataclass
class Event:
    request_id: str
    stage: str
    status: str
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class TraceStore:
    def __init__(self):
        self.events: List[Event] = []

    def record_trace(self, request_id: str, stage: str, status: str, reason: str, **metrics):
        self.events.append(
            Event(
                request_id=request_id,
                stage=stage,
                status=status,
                reason=reason,
                metrics=metrics,
            )
        )

    def by_request(self, request_id: str) -> List[Event]:
        return [e for e in self.events if e.request_id == request_id]


def classify_failure(events: List[Event]) -> str:
    for e in events:
        if e.stage == "retrieval" and e.metrics.get("hits", 0) == 0:
            return "retrieval_miss"
        if e.stage == "model" and e.reason == "unsupported_claim":
            return "unsupported_claim"
    return "ok"


def handle_query(query: str, docs: List[str], trace: TraceStore):
    request_id = str(uuid.uuid4())

    hits = [d for d in docs if query.lower() in d.lower()]
    trace.record_trace(
        request_id,
        stage="retrieval",
        status="ok",
        reason="retrieval_done",
        hits=len(hits),
        top_score=0.91 if hits else 0.0,
    )

    if not hits:
        trace.record_trace(
            request_id,
            stage="model",
            status="degraded",
            reason="unsupported_claim",
            input_tokens=120,
            output_tokens=38,
        )
        return request_id, "我没有足够依据回答这个问题。"

    trace.record_trace(
        request_id,
        stage="model",
        status="ok",
        reason="grounded_answer",
        input_tokens=180,
        output_tokens=56,
    )
    return request_id, "答案基于已检索文档生成。"


trace = TraceStore()

rid1, ans1 = handle_query("报销上限", ["差旅报销上限为 2000 元"], trace)
rid2, ans2 = handle_query("年假规则", ["差旅报销上限为 2000 元"], trace)

assert ans1 == "答案基于已检索文档生成。"
assert classify_failure(trace.by_request(rid1)) == "ok"
assert ans2 == "我没有足够依据回答这个问题。"
assert classify_failure(trace.by_request(rid2)) == "retrieval_miss"
```

这个例子体现了两个最低要求。

第一，请求 ID 必须贯穿所有阶段。否则你没法把检索事件和模型事件关联起来。

第二，故障分类要建立在 trace 之上，而不是独立写死。因为 `unsupported_claim` 有时来自模型胡编，有时来自检索为空；只有结合前序阶段，分类才有意义。

把它翻译成更接近生产环境的伪代码，大致会是这样：

```python
def serve(request):
    request_id = new_request_id()

    record_trace(request_id, stage="input", status="ok", reason="accepted",
                 user_id=request.user_id, query_len=len(request.query))

    docs = retrieve(request.query)
    record_trace(request_id, stage="retrieval", status="ok", reason="done",
                 hits=len(docs), top_score=docs[0].score if docs else 0.0)

    prompt = build_prompt(request.query, docs)
    record_trace(request_id, stage="prompt", status="ok", reason="built",
                 prompt_tokens=count_tokens(prompt), doc_count=len(docs))

    output = call_llm(prompt)
    record_trace(request_id, stage="model", status="ok", reason=output.label,
                 input_tokens=output.input_tokens, output_tokens=output.output_tokens,
                 model=output.model_name)

    if output.need_tool:
        tool_result = call_tool(output.tool_args)
        record_trace(request_id, stage="tool", status=tool_result.status,
                     reason=tool_result.reason, latency_ms=tool_result.latency_ms)

    failure_type = classify(request_id)
    emit_metric("failure_type", failure_type)
    return finalize(output, failure_type)
```

新手最该注意的是，不要把 trace 做成“只有出错才打印”。因为很多 LLM 错误不是异常，而是低质量成功。正确方式是每层都产出事件，无论成功、降级还是失败，都记录 `status`、`reason`、`metrics`、`tokens`。这样你既能看坏样本，也能看正常样本的分布变化。

真实工程里，结构化事件至少应包含这些字段：

| 字段 | 作用 |
| --- | --- |
| `request_id` | 串联整次请求 |
| `stage` | 标识输入、检索、提示、模型、工具等阶段 |
| `status` | 区分成功、降级、失败 |
| `reason` | 给出稳定原因标签 |
| `latency_ms` | 判断性能瓶颈 |
| `input_tokens` / `output_tokens` | 统计成本与异常长度 |
| `model_version` | 排查版本回归 |
| `metadata` | 存放命中文档、工具名、分数、规则结果 |

---

## 工程权衡与常见坑

工程上最常见的误区，是把 LLM 应用当成普通 API 服务治理。普通 API 的核心问题往往是可用性，LLM 应用则同时要治理正确性、稳定性、成本和安全。

第一类坑，只看延迟和成功率，不看语义质量。语义质量，白话讲，就是回答在意思上是否正确、有依据、符合任务目标。一个系统可以做到 99.99% 成功返回，同时持续输出 unsupported 答案。这时 SRE 看面板会觉得一切正常，业务方却说“系统一直答错”。补救方法是增加 groundedness、task success、人工抽样评分或自动评估。

第二类坑，没有共享 trace，团队只能靠猜。检索团队说“我们召回了文档”，模型团队说“模型按上下文生成了”，工具团队说“API 正常响应了”，最后谁都说不清为什么用户拿到错答案。补救方法是每次请求共享同一个 request ID，并把 span 级事件放进统一后端。

第三类坑，故障分类过粗。很多团队只分“成功”和“失败”，结果无法决定下一步优化点。把 `hallucination`、`retrieval_miss`、`tool_timeout`、`policy_violation` 混在一起，后续统计没有行动价值。补救方法是先定义少量稳定、高频、可执行的类别，再逐步细化。

第四类坑，监控只在模型调用点。实际上，输入污染、文档过期、prompt 截断、工具返回脏数据都可能是主因。只监控模型阶段，相当于把多数根因藏起来。

第五类坑，成本监控后置。LLM 系统常见问题不是“服务挂了”，而是“还能跑，但 token 成本翻倍”。尤其是检索文档过长、prompt 模板膨胀、多工具重试时，成本会静悄悄地上涨。

可以把这些坑和补救方式总结成表：

| 常见坑 | 后果 | 补救方法 |
| --- | --- | --- |
| 只看延迟、成功率 | 错题大量存在却无报警 | 增加语义质量、 groundedness、拒答率等指标 |
| 没有 trace | 无法定位根因，团队互相推诿 | 全链路共享 `request_id`，记录 span 事件 |
| 分类过粗 | 无法决定优化优先级 | 建立稳定故障标签体系 |
| 只监控模型调用 | 上游问题被误判为模型问题 | 输入、检索、prompt、工具全部打点 |
| 不看 token 与成本 | 预算失控或上下文截断 | 记录 token、上下文长度、重试次数 |
| 无 guardrail 分类 | 敏感输出直接暴露 | 增加安全拦截与违规类型统计 |

一个真实工程例子很典型。某团队把主要精力都放在把平均延迟从 3.2 秒压到 2.1 秒，面板很漂亮，但上线后用户仍频繁投诉“答案没有依据”。复盘后发现，系统为了追求低延迟，把 reranker 关掉了，导致 unsupported 回答大幅增加。也就是说，性能优化伤害了质量，但因为没有质量指标，这种退化长期没被看见。工程上的正确做法不是“只追单指标最优”，而是在延迟、质量、成本之间找可接受区间。

---

## 替代方案与适用边界

不是所有项目一开始都要上完整的全链路可观测体系。方案要和系统复杂度匹配。

如果你现在只是一个小规模 POC，例如“检索 + 单模型回答”的内部演示系统，那么轻量方案就够用。轻量方案，白话讲，就是先只抓最容易出问题、最有价值的几个点：统一请求 ID、检索命中数、模型 token、最终故障标签。这样投入低，能快速建立第一版诊断能力。它的局限也很明显：只能覆盖主路径，对复杂工具链和跨模型协作的可见度不足。

当系统升级到多工具、多模型、内部知识库、权限控制、异步任务这些场景时，就需要更完整的方案。比如采用异构 trace + eval + monitor 的体系，把模型 span、检索 span、工具 span、用户反馈 span 串起来，再配合离线评测和在线报警。异构 trace，白话讲，就是同一次请求里允许不同类型阶段都按统一结构记录和关联，不要求所有节点都长得一样，但要求它们能被同一个请求上下文串起来。

下面给出一个选择表：

| 方案 | 适用场景 | 投入 | 可观测能力 |
| --- | --- | --- | --- |
| 轻量监听 | 单模型、单检索、POC、流量较小 | 低 | 能看主路径问题，适合快速起步 |
| 全链路 trace + eval + monitor | 多工具、多模型、内部知识库、正式生产 | 高 | 能做跨阶段关联、在线报警、离线评估与回归分析 |

可以再看两个边界例子。

玩具例子：一个只有“检索 + 模型”的 FAQ 机器人。你完全可以用轻量 trace，加上手工故障标签，例如 `retrieval_miss`、`unsupported`、`format_error`。这时最重要的不是系统多复杂，而是先把“错在哪里”记录下来。

真实工程例子：企业内部知识助手，接入文档库、工单系统、权限校验和多个模型路由。此时如果还靠零散日志，你几乎不可能定位跨阶段错误。更合适的做法是用完整的 trace 体系把检索、重排、提示、模型、工具、权限判断全部串起来，再结合在线 monitor 和离线 eval 做回归控制。

所以替代方案的核心不是“选最先进的”，而是“选能覆盖当前主要失败模式的”。系统复杂度一旦上升，错误处理必须从“日志排错”升级成“可观测性工程”。

---

## 参考资料

1. OptyxStack, *AI Observability for Production LLM Systems*, 2026. https://optyxstack.com/llm-audit/ai-observability-production-llm-systems  
2. Braintrust, *What is LLM observability?*, 2026. https://www.braintrust.dev/articles/llm-observability-guide  
3. TechTarget, *Understanding the fundamentals of LLM observability*, 2025. https://www.techtarget.com/searchsoftwarequality/tip/Understanding-the-fundamentals-of-LLM-observability
