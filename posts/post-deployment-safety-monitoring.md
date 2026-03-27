## 核心结论

上线后的安全监控，本质上不是“看服务有没有挂”，而是持续观察模型在真实流量中的行为是否偏离预期。这里的“真实流量”指用户在生产环境里实际发出的请求，它通常比离线测试更脏、更杂、更容易触发长尾问题。对大模型系统，最关键的不是单一准确率，而是违规率、拒答率、人工介入比例、抽样复核结果是否稳定。

安全监控必须同时覆盖三条链路。第一条是自动指标链，负责把违规、拒答、延迟、版本差异变成数字。第二条是人工审查链，负责补自动规则看不见的长尾样本。第三条是控制执行链，负责在发现异常后真正减量、切流、回滚。只有“看见问题”而不能“立刻动作”，监控就只是报表系统，不是安全系统。

一个实用判断标准是：上线方案必须回答四个问题。出了什么问题、问题影响谁、能否在分钟级发现、能否在分钟级止损。如果这四个问题答不清，说明监控和回滚方案还没形成闭环。

| 信号 | 监测通道 | 触发动作 |
| --- | --- | --- |
| 违规率上升 | 安全分类器、策略命中日志 | 告警、暂停灰度、升级人审 |
| 拒答率异常 | 输出标签、拒答模板统计 | 检查提示词、防护阈值、模型版本 |
| 人工介入比例上升 | 工单系统、审核台 | 挂起高风险流量、补充样本复核 |
| 高风险样本抽检命中 | 抽样任务、红队回放 | 更新规则、封禁路径、准备回滚 |
| 延迟或超时恶化 | APM、网关指标 | 降级、熔断、切旧版本 |

---

## 问题定义与边界

本文讨论的“上线后的安全监控”，指的是大模型已经开始服务真实用户之后，对其输出安全性和控制能力进行持续观测与处置的体系。这里的“安全性”不是泛指所有安全，而是狭义的模型行为安全，包括违规内容、错误拒答、风险放行、敏感信息泄露、明显幻觉传播，以及由此引发的业务事故。

边界要先划清，否则指标没有意义。至少要明确五件事：

1. 监控对象是谁。是基础模型、带检索的 RAG 系统，还是多步骤 Agent。
2. 监控粒度是什么。是按全站汇总，还是按模型版本、语言、地区、用户等级、业务场景分别统计。
3. 监控窗口有多长。是 1 分钟告警、1 小时趋势，还是 1 天复盘。
4. 触发阈值怎么定。是绝对阈值，还是相对基线的偏移。
5. 发现异常后谁有权切流、降级、回滚。

传统 Web 服务常看错误率和延迟。大模型系统不够。原因是模型输出具有非确定性，也就是同类输入不一定得到同类输出。于是“服务 200 OK”不等于“回答安全”。一个请求可能在技术上成功返回，但内容已经越过风险边界。

可以把监控覆盖度写成一个简化指标：

$$
Coverage\ Score = \frac{A + H + S}{3}
$$

其中，$A$ 是自动告警覆盖率，表示有多少高风险请求会被自动规则或分类器观察到；$H$ 是人工审查比例，表示实际流量中有多少会进入人工复核；$S$ 是抽样率，表示系统对常规流量做随机或分层抽检的比例。这个公式不是行业标准，而是一个便于工程落地的简化视角：如果三项里有一项接近 0，监控一定有盲区。

玩具例子可以说明问题。假设一个问答机器人每天处理 1000 条请求，平均违规率只有 0.3%。看上去很低。但如果这 3 条违规都集中在“未成年人”“医疗建议”“仇恨表达”三类高风险主题，且正好被投诉到社交平台，业务影响会远大于一般的接口报错。因此，安全监控看的是分布，不只是均值。

真实工程例子通常更复杂。企业客服系统里，同一个模型会同时面对普通用户、VIP 用户、夜间咨询、跨语言问题、促销高峰等不同流量切片。如果只看全量汇总，某个小语种市场的风险激增很容易被主流流量稀释掉。边界没分清，监控就会“总体正常，局部失控”。

---

## 核心机制与推导

完整机制可以拆成“观测层、判断层、控制层”三层。

观测层负责采集。每次请求至少要记录输入摘要、模型版本、策略命中、输出标签、人工处理结果、链路延迟。这里的“输入摘要”是经过脱敏后的关键信息，不是简单全量落库；“策略命中”是指内容过滤器、提示词护栏、关键词规则、安全分类器是否触发。

判断层负责把原始日志转成可执行信号。最基础的三个安全指标是：

1. 违规率：被判定为违规或高风险输出的比例。
2. 拒答率：模型主动拒绝、被系统拦截、被模板替换的比例。
3. 人工介入比例：请求进入人工审核、人工接管、人工复写的比例。

为什么这三项必须一起看？因为它们反映了三个不同失效模式。

- 违规率上升，说明风险放行变多，系统变“松”了。
- 拒答率上升，说明系统可能过度防御，业务可用性变差，系统变“紧”了。
- 人工介入比例上升，说明自动策略看不准，或者异常分布变化太快，系统变“乱”了。

可以把它理解为一个简单张力关系。若放宽策略，违规率可能上升，拒答率下降；若收紧策略，拒答率可能上升，违规率下降。工程目标不是把单一指标压到最低，而是在风险成本和业务可用性之间找到可控区间。

一个常见推导方式是设定基线与漂移阈值。假设稳定期基线为：

- 违规率基线 $v_0 = 0.8\%$
- 拒答率基线 $r_0 = 2.0\%$
- 人工介入比例基线 $h_0 = 0.4\%$

灰度期间观测到新版本指标为 $v_1, r_1, h_1$。可以定义一个风险得分：

$$
RiskScore = w_v \cdot \frac{v_1-v_0}{v_0+\epsilon}
+ w_r \cdot \frac{r_1-r_0}{r_0+\epsilon}
+ w_h \cdot \frac{h_1-h_0}{h_0+\epsilon}
$$

其中 $\epsilon$ 用来避免分母为 0，$w_v,w_r,w_h$ 是权重。若系统更在意风险放行，就让 $w_v$ 更高；若更在意业务连续性，就提高 $w_r$ 或单独设置硬阈值。这个公式仍然是工程简化版，但它说明了一件重要的事：是否回滚，不应只靠“某一项超过固定值”，而应结合基线偏移和业务优先级。

灰度发布和蓝绿切换就是控制层的核心机制。

灰度发布，也叫 Canary，意思是只让一小部分流量进入新版本，先观测再放量。它的价值是把问题影响范围限制在 1% 到 5% 这类可控区间。蓝绿部署指同时保留两套可工作的环境，一套线上服务，一套候命。发现问题时直接把流量切回旧环境，而不是边修边扛。

一个常见的控制闭环是：

1. 新版本先接入 2% 流量。
2. 连续观测 15 到 30 分钟。
3. 若违规率、拒答率、人工介入比例都在阈值内，再升到 5%、10%、25%。
4. 任一关键指标在连续窗口中越界，立刻暂停放量。
5. 若越界达到事故阈值，自动或人工触发回滚。

玩具例子：假设旧版本拒答率稳定在 1%，新版本灰度 2% 流量后，60 秒窗口里拒答率升到 4%，人工审核队列也显著增长。这时即使没有明显违规投诉，也应该先暂停灰度。因为这说明新版本虽然可能更保守，但已经影响正常业务，系统进入了不稳定状态。

真实工程例子：新闻摘要、客服回复、搜索答案生成这类面向大量真实用户的功能，风险不在“偶尔答错一道题”，而在“高传播内容被系统自信地说错”。一旦错误内容进入通知、摘要、自动外发邮件等高曝光通道，事故会被迅速放大。所以这些场景通常要求蓝绿切换能力，而不是只依赖手工下线。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它不依赖外部服务，只演示三件事：记录请求、统计灰度指标、在阈值越界时触发回滚。这里的“回滚”是把流量标签从 `green` 切回 `blue`，也就是从新版本切回旧版本。

```python
from dataclasses import dataclass
from typing import List


@dataclass
class Event:
    version: str
    violation: bool
    refusal: bool
    human_review: bool
    latency_ms: int


class Monitor:
    def __init__(self):
        self.active_target = "green"

    def summarize(self, events: List[Event]) -> dict:
        n = len(events)
        assert n > 0, "events cannot be empty"
        violations = sum(e.violation for e in events)
        refusals = sum(e.refusal for e in events)
        human_reviews = sum(e.human_review for e in events)
        p95_latency = sorted(e.latency_ms for e in events)[int(0.95 * (n - 1))]
        return {
            "violation_rate": violations / n,
            "refusal_rate": refusals / n,
            "human_review_rate": human_reviews / n,
            "p95_latency": p95_latency,
        }

    def should_rollback(self, baseline: dict, canary: dict) -> bool:
        violation_jump = canary["violation_rate"] > baseline["violation_rate"] + 0.01
        refusal_jump = canary["refusal_rate"] > baseline["refusal_rate"] + 0.02
        review_jump = canary["human_review_rate"] > baseline["human_review_rate"] + 0.01
        latency_jump = canary["p95_latency"] > baseline["p95_latency"] + 300
        return violation_jump or refusal_jump or review_jump or latency_jump

    def rollback(self):
        self.active_target = "blue"


baseline_events = [
    Event("blue", False, False, False, 300),
    Event("blue", False, False, False, 320),
    Event("blue", False, True,  False, 340),
    Event("blue", False, False, False, 310),
    Event("blue", True,  False, True,  360),
]

canary_events = [
    Event("green", False, True,  False, 350),
    Event("green", True,  True,  True,  420),
    Event("green", False, True,  False, 390),
    Event("green", True,  False, True,  410),
    Event("green", False, True,  True,  405),
]

monitor = Monitor()
baseline = monitor.summarize(baseline_events)
canary = monitor.summarize(canary_events)

if monitor.should_rollback(baseline, canary):
    monitor.rollback()

assert baseline["violation_rate"] == 0.2
assert canary["refusal_rate"] == 0.6
assert monitor.active_target == "blue"
```

这段代码体现了最小闭环：

1. 每个请求都落成结构化事件。
2. 汇总时输出违规率、拒答率、人工介入比例、P95 延迟。
3. 将新版本与旧版本基线比较。
4. 达到条件后触发回滚。

真实工程里，这个闭环会扩展成四层组件。

| 组件 | 作用 | 关键字段 |
| --- | --- | --- |
| 网关与日志层 | 记录请求入口与版本路由 | request_id、user_segment、model_version |
| 安全判定层 | 给输出打安全标签 | safety_score、policy_hits、refusal_type |
| 审核与工单层 | 接住高风险样本 | reviewer_id、review_result、escalation_level |
| 控制面 | 执行切流与回滚 | rollout_percent、feature_flag、rollback_reason |

实际系统里，每次调用通常至少要有一个统一 `trace_id`。它是请求追踪 ID，也就是把一次对话涉及的检索、模型推理、过滤器、人审动作串起来的唯一编号。没有它，事故发生后只能看到零散日志，无法快速复原问题链路。

一个真实工程例子是企业内部知识助手。请求先经过身份校验，再查向量库，再调用模型，最后经过输出过滤器。如果用户投诉“系统泄露了不该看到的内部信息”，排查时必须知道：泄露内容来自检索结果、提示词拼接、模型自行编造，还是过滤器漏拦。没有完整 trace，就无法判断是该调检索权限、改提示词，还是直接回滚模型版本。

---

## 工程权衡与常见坑

第一个权衡是安全性与可用性。策略越严格，违规率可能越低，但拒答率通常会上升。对高风险场景，例如金融投顾、医疗建议、未成年人相关内容，宁可更保守；对低风险场景，例如内部文档检索，可以接受更高的人工复核比例，换取更好的可用性。权衡的前提不是拍脑袋，而是事先定义业务损失函数，也就是“放过一次违规”和“多拒答一条正常请求”哪个代价更高。

第二个权衡是采样率与成本。抽样审查越多，越容易发现长尾问题，但人工成本越高。常见做法不是全量人工看，而是分层抽样：高风险场景高抽样，低风险场景低抽样；新版本高抽样，稳定旧版本低抽样；命中过滤器的样本全审，未命中的样本按比例抽查。

第三个权衡是自动回滚与人工确认。自动回滚响应最快，但阈值设太敏感会造成误回滚；人工确认更稳，但速度慢。常见做法是把严重风险设计成自动回滚，把边界模糊问题设计成人工确认。例如“违规率突然翻倍”可以自动回滚，而“拒答率轻微升高但投诉未增”可以先暂停灰度，等待值班人员确认。

常见坑主要有以下几类：

| 常见坑 | 影响 | 规避方式 |
| --- | --- | --- |
| 只看平均值 | 长尾高风险样本被均值掩盖 | 统计分桶、看 P95/P99、按场景切片 |
| 只监控违规率，不监控拒答率 | 系统可能因过度保守而不可用 | 同时设业务可用性阈值 |
| 没有保留旧版本环境 | 出事后只能热修，无法分钟级止损 | 保留最近 2 到 3 个可回退版本 |
| 只有告警，没有 Runbook | 告警触发后团队不知道谁做决定 | 写清责任人、操作步骤、升级路径 |
| 日志没有脱敏 | 监控系统本身成为数据泄露源 | 输入摘要化、字段分级、最小落库 |
| 抽样不分层 | 高风险流量被低风险流量稀释 | 按国家、语言、用户群、场景分层 |
| 演练缺失 | 真事故时切流和回滚不熟练 | 定期做回滚演练和告警演练 |

这里的 Runbook 指事故操作手册，也就是“什么情况下谁执行什么动作”的标准流程。没有 Runbook，事故处理会变成多人同时判断、重复下指令、互相等待，恢复时间会显著拉长。

一个真实工程里的典型坑是：团队上线前做了离线评测，也设置了内容过滤器，但没有设计“人工介入比例”指标。结果新模型并没有明显增加违规内容，却大量生成模糊、难判定、需要客服手工改写的回答。最终业务侧感受到的是处理效率下降，而技术侧报表仍然显示“安全指标正常”。这就是指标设计不完整导致的盲区。

---

## 替代方案与适用边界

灰度和蓝绿不是唯一方案，但它们是最常见、最稳妥的上线控制手段。不同业务阶段，适合的方案不同。

| 方案 | 适用边界 | 主要优点 | 主要限制 |
| --- | --- | --- | --- |
| Shadow 模式 | 早期探索、先看行为不对外返回 | 不影响真实用户 | 看不到真实交互闭环 |
| Canary 灰度 | 新版本渐进放量 | 风险隔离好、便于比较 | 需要精细流量切分 |
| Blue/Green 蓝绿 | 高可用、要求快速回退 | 切换快、恢复明确 | 双环境成本更高 |
| Feature Flag | 功能按用户或场景开关 | 控制细、适合业务联动 | 状态管理复杂 |
| Circuit Breaker 熔断 | 下游异常或风险暴涨时止损 | 反应快、保护系统 | 只能止损，不能定位根因 |

Shadow 模式，也叫影子模式，指新版本接收真实请求但不直接影响用户结果，只用于记录和对比。这适合上线前最后一轮验证，尤其适合高风险场景。它的弱点是，用户不会真的依赖影子结果，因此无法完整暴露交互反馈链路。

Canary 适合“我想知道新版本在真实流量里是否更好，但又不敢一次性全量上线”的场景。Blue/Green 适合“我必须保证分钟级恢复”的场景，例如首页搜索摘要、对外自动通知、关键客服入口。Feature Flag 更偏业务控制，例如只对内部员工开放新模型，或只对英语用户开启新策略。Circuit Breaker 则更像保险丝，一旦风险过阈值，立即拒绝部分功能、切到模板回复、或者关闭高风险生成能力。

一个推荐顺序通常是：先 Shadow，再 Canary，最后进入 Blue/Green 可回退生产。对零基础到初级工程师来说，可以记成一句话：先偷偷看，再小范围试，再准备好秒切回旧版。这个顺序不是规则，而是风险逐步暴露、控制能力逐步增强的过程。

---

## 参考资料

- EvalOps, Production LLM Monitoring: Beyond Uptime and Latency, 2025-09-28, https://www.evalops.dev/blog/production-llm-monitoring
- APXML, Monitoring LLMs in Production, https://apxml.com/courses/llm-alignment-safety/chapter-6-interpretability-monitoring-safety/monitoring-llms-production-safety
- Braintrust, What is LLM monitoring?, 2026-02-09, https://www.braintrust.dev/articles/what-is-llm-monitoring
- Amit Kothari, LLM deployment pipeline, 2025-11-04, https://amitkoth.com/llm-deployment-pipeline
- TechCrunch, Apple pauses AI notification summaries for news after generating false alerts, 2025-01-16, https://techcrunch.com/2025/01/16/apple-pauses-ai-notification-summaries-for-news-after-generating-false-alerts/
- APXML, Blue/Green and Canary Deployments, https://apxml.com/courses/langchain-production-llm/chapter-7-deployment-strategies-production/blue-green-canary-deployments
- Preploop, Blue Green and Canary Deployment Patterns for Model Rollout, https://preploop.io/learn/ml-model-serving/model-versioning-rollback/blue-green-and-canary-deployment-patterns-for-model-rollout
- OptyxStack, LLM Reliability Checklist Before Enterprise Rollout, https://optyxstack.com/llm-evaluation/llm-reliability-checklist-before-enterprise-rollout
