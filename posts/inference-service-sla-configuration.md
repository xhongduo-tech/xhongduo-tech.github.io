## 核心结论

推理服务的 SLA 不能从系统现状直接抄出来，必须先从业务承诺倒推。业务承诺说的是“用户能否用、要等多久、出问题谁负责”，系统指标说的是“怎么测、怎么告警、怎么处置”。如果顺序反过来，最后写出来的往往是“GPU 利用率”“QPS”“显存占用”这类内部观测值，而不是用户真正感知到的服务质量。

对大模型在线推理，延迟目标通常至少要拆成 `TTFT`、`TPOT`、`E2E`。`TTFT` 是首 token 时间，白话说就是“用户点发送后，多久看到模型开始说话”；`TPOT` 是每个输出 token 的平均间隔，白话说就是“模型后续说话的流畅度”；`E2E` 是总响应时间，白话说就是“从发请求到完整答完一共多久”。这三个指标描述的是不同故障面，不能互相替代。

公式可以直接写成：

$$
TTFT = t_{first\_token} - t_{request\_start}
$$

$$
TPOT = \frac{t_{final\_token} - t_{first\_token}}{N_{out} - 1},\quad N_{out}\ge 2
$$

$$
E2E = t_{final\_token} - t_{request\_start} = TTFT + (N_{out}-1)\cdot TPOT
$$

可用性也不能只写一句“99.9% 可用”。必须说明统计窗口、有效请求、排除项、故障判定方法。常见写法是：

$$
A_W = 1 - \frac{D_{eligible}}{T_{eligible}}
$$

其中 $W$ 是统计窗口，$D_{eligible}$ 是应计入 SLA 的停机时间，$T_{eligible}$ 是窗口内应计量的总时间或总请求量。

下面这张表是最核心的映射关系：

| 业务承诺 | SLI（服务指标） | SLO（内部目标） | 告警 | 处置 | 赔付/后果 |
|---|---|---:|---|---|---|
| 首 token 要快 | `ttft_seconds` p95 | `<= 1.5s` | 1h fast-burn | 降批大小、切备用集群 | 月度 SLA 未达标触发赔付 |
| 输出要稳定 | `tpot_seconds` p95 | `<= 120ms/token` | 24h slow-burn | 限流低优先级、扩容 | 同上 |
| 总耗时可接受 | `e2e_seconds` p95 | `<= 8s` | 尾延迟异常告警 | 排查排队/解码瓶颈 | 同上 |
| 服务整体可用 | 成功请求率或月度可用性 | `>= 99.9%` | burn-rate 超阈值 | 故障切流、熔断、回退模型 | 同上 |

一个真实工程例子：在线聊天服务对外承诺“首 token 快、整体回答稳定、月度可用性达标”，对应的 SLA 就更接近下面这种形式，而不是“GPU 利用率低于 80%”：

- `TTFT p95 <= 1.5s`
- `TPOT p95 <= 120ms/token`
- `E2E p95 <= 8s`
- `月可用性 >= 99.9%`
- 排除计划维护、客户网络故障、客户错误配置

没有明确告警阈值和自动化处置，SLA 只是文档，不会转化成运行约束。

---

## 问题定义与边界

先把四个容易混淆的词分开：

| 术语 | 定义 | 面向对象 | 解决什么问题 |
|---|---|---|---|
| SLA | 服务等级协议，带违约后果的对外承诺 | 客户、业务、法务 | 没做到会发生什么 |
| SLI | 服务等级指标，可测量的量化指标 | 研发、运维、平台 | 到底怎么测 |
| SLO | 服务等级目标，SLI 的目标值 | 团队内部 | 我们希望达到多少 |
| Error Budget | 错误预算，允许消耗的失败空间 | 研发、运维、管理 | 还能冒多大风险 |

白话说，SLA 是合同，SLI 是尺子，SLO 是及格线，Error Budget 是“还能花掉多少容错额度”。

推理服务做 SLA，边界必须先定义清楚，否则数字没有约束力。最重要的边界有四类：

1. 统计窗口是什么  
`calendar month` 是自然月统计，适合对账和赔付；`rolling 30d` 是滚动 30 天，适合持续观察，但不适合直接当财务结算口径。两者会让同一故障落在不同周期里，结论可能不同。

2. 什么叫有效请求  
通常要限定为“符合 API 文档、身份合法、请求格式正确、流量进入服务边界后由平台处理的请求”。如果连请求都没构成有效调用，就不该进入 SLA 分母。

3. 哪些故障计入 SLA  
这是责任边界问题。平台自己的无响应、5xx 激增、调度拥塞、集群切流失败，通常应计入。客户本地网络、防火墙阻断、错误 SDK 使用、无效参数，通常不计入。

4. 第三方依赖怎么处理  
如果你的推理服务强依赖对象存储、鉴权网关、外部向量库，SLA 必须说明这些依赖失效时如何归责。否则用户看到的是“服务不可用”，团队内部却会争论“不是我们挂了，是依赖挂了”。

常见的计入与排除可以写成表：

| 场景 | 是否计入 SLA | 说明 |
|---|---|---|
| 服务端 5xx 大量上升 | 计入 | 平台自身故障 |
| 推理队列堆积导致超时 | 计入 | 容量或调度问题 |
| 首 token 长时间不返回 | 计入 | 用户实际不可用 |
| 计划维护窗口 | 排除 | 需提前声明且受控 |
| 客户网络/防火墙问题 | 排除 | 不在平台控制域 |
| 客户错误配置或错误调用 | 排除 | 无效请求或客户责任 |
| 第三方依赖故障 | 视合同定义 | 必须预先写清 |

玩具例子：某团队写“服务月可用性 99.9%”。这个表述看起来完整，其实没有落地能力，因为至少缺了三件事：

- 统计自然月还是滚动 30 天
- 计划维护算不算停机
- 客户侧错误算不算失败

只要这三条没写，同一个月内 20 分钟故障，销售、客户成功、SRE、法务可能会得到四个不同结论。

---

## 核心机制与推导

在线大模型服务为什么要拆 `TTFT`、`TPOT`、`E2E`？因为用户体感是分阶段发生的。

第一阶段是“模型有没有开始说”。这对应 `TTFT`。它常受排队、请求预处理、prefill、调度、KV cache 命中情况影响。  
第二阶段是“模型说得顺不顺”。这对应 `TPOT`。它常受 decode 吞吐、批处理策略、抢占、显存压力影响。  
第三阶段是“整段回答多久结束”。这对应 `E2E`。它是前两者与输出长度共同作用的结果。

一个最小玩具例子：

- `TTFT = 0.9s`
- `TPOT = 0.10s/token`
- 输出 `30` 个 token

那么：

$$
E2E \approx 0.9 + 29 \times 0.10 = 3.8s
$$

这个例子说明，总耗时不是一个孤立指标。首 token 慢 500ms，和后续每 token 慢 20ms，都会把 `E2E` 拉长，但根因完全不同。

故障类型和指标的映射大致如下：

| 现象 | 更可能异常的指标 | 常见根因 |
|---|---|---|
| 用户迟迟看不到回复开始 | `TTFT` 高 | 排队、prefill 慢、调度拥塞 |
| 回复开始了但一个字一个字蹦得很慢 | `TPOT` 高 | decode 吞吐不足、批过大、显存抖动 |
| 总体超时 | `E2E` 高 | `TTFT` 或 `TPOT` 任一异常，或输出太长 |
| 请求经常失败 | 可用性下降、错误率上升 | 服务故障、依赖故障、限流配置错误 |

可用性和错误预算本质上是同一件事的两个角度。SLO 是“目标线”，错误预算是“允许失败空间”。

如果月度可用性目标为：

$$
S = 99.9\% = 0.999
$$

则错误预算为：

$$
B = 1 - S = 0.001
$$

30 天按分钟计算总量是：

$$
30 \times 24 \times 60 = 43200
$$

可容忍停机时间约为：

$$
43200 \times 0.001 = 43.2\ \text{分钟}
$$

如果本月已经有 25 分钟应计入停机，那么还剩 18.2 分钟预算。此时再发生一次 30 分钟故障，就会直接违约。

`burn rate` 是预算消耗速度，白话说就是“照当前坏下去，多久会把整月容错额度烧光”。可以写成：

$$
burn\_rate = \frac{B_{consumed}}{B_{total}}
$$

更工程化一点的理解是：如果某个短窗口内 `burn_rate > 1`，说明按当前错误速度持续下去，会在合规周期内超预算；如果远大于 1，就需要立刻处置，而不是等月末算账。

真实工程例子：聊天服务在晚高峰流量暴涨，GPU 利用率从 65% 升到 92%。如果此时 `TTFT p95` 仍稳定在 1.2s、`TPOT p95` 仍是 95ms/token、错误率不升，那么“GPU 利用率高”本身不是 SLA 问题。反过来，GPU 利用率只有 55%，但调度器配置错误导致请求排队，`TTFT p95` 从 1.1s 飙到 4.0s，这才是用户真实感知到的失约。

---

## 代码实现

SLA 落地不是写一页文档，而是把文档变成埋点、聚合、告警、自动处置。

先定义最小监控面：

| 指标名 | 单位 | 采样点 | 用途 | 是否进入 SLA |
|---|---|---|---|---|
| `ttft_seconds` | 秒 | 首 token 返回时 | 监控首响应体验 | 是 |
| `tpot_seconds` | 秒/token | 每次生成结束后 | 监控输出流畅度 | 是 |
| `e2e_seconds` | 秒 | 请求完成时 | 监控总时延 | 是 |
| `request_total` | 次 | 请求进入服务边界时 | 统计分母 | 是 |
| `request_success_total` | 次 | 请求成功完成时 | 统计成功率 | 是 |
| `queue_wait_seconds` | 秒 | 请求出队时 | 诊断排队瓶颈 | 否，通常只做内部诊断 |
| `batch_size` | 个 | 调度周期内 | 诊断吞吐策略 | 否 |
| `gpu_utilization` | 百分比 | 节点级采样 | 容量观察 | 否 |

一个最小配置片段可以是：

```yaml
slo:
  ttft_p95_ms: 1500
  tpot_p95_ms: 120
  e2e_p95_ms: 8000
  availability: 0.999
  window: calendar_month
  exclusions:
    - planned_maintenance
    - customer_network_failure
    - customer_misconfig
alerts:
  fast_burn:
    window: 1h
    threshold: 2.0
  slow_burn:
    window: 24h
    threshold: 1.0
actions:
  - reduce_batch_size
  - failover_to_secondary_cluster
  - throttle_low_priority_traffic
```

埋点逻辑可以用很小的 Python 例子说明：

```python
from dataclasses import dataclass

@dataclass
class RequestMetrics:
    ttft_seconds: float
    tpot_seconds: float
    e2e_seconds: float
    success: bool

def calc_request_metrics(t_request_start, t_first_token, t_final_token, n_out, success=True):
    assert t_first_token >= t_request_start
    assert t_final_token >= t_first_token
    assert n_out >= 1

    ttft = t_first_token - t_request_start
    e2e = t_final_token - t_request_start

    if n_out == 1:
        tpot = 0.0
    else:
        tpot = (t_final_token - t_first_token) / (n_out - 1)

    assert ttft >= 0
    assert e2e >= ttft
    assert tpot >= 0

    return RequestMetrics(
        ttft_seconds=ttft,
        tpot_seconds=tpot,
        e2e_seconds=e2e,
        success=success,
    )

m = calc_request_metrics(0.0, 0.9, 3.8, 30, True)
assert round(m.ttft_seconds, 2) == 0.90
assert round(m.tpot_seconds, 2) == 0.10
assert round(m.e2e_seconds, 2) == 3.80
assert m.success is True
```

再看一个月度可用性和 burn-rate 的最小实现：

```python
def availability(total_minutes, eligible_downtime_minutes):
    assert total_minutes > 0
    assert 0 <= eligible_downtime_minutes <= total_minutes
    return 1 - eligible_downtime_minutes / total_minutes

def error_budget(slo):
    assert 0 < slo < 1
    return 1 - slo

def burn_rate(consumed_budget, total_budget):
    assert total_budget > 0
    assert consumed_budget >= 0
    return consumed_budget / total_budget

month_minutes = 30 * 24 * 60
a = availability(month_minutes, 25)
assert round(a, 6) == round(1 - 25 / 43200, 6)

budget = error_budget(0.999)
assert round(budget, 4) == 0.0010

consumed = 25 / month_minutes
br = burn_rate(consumed, budget)
assert br > 0
```

告警规则的关键不是“阈值越多越专业”，而是“触发后能自动执行动作”。比如：

- `fast-burn`：1 小时窗口内预算消耗速度过快，立即切备用集群
- `slow-burn`：24 小时窗口内持续偏离，说明容量规划或模型版本有系统性问题
- `TTFT p95` 单独异常：优先检查排队、prefill、路由
- `TPOT p95` 单独异常：优先检查 decode 吞吐、批大小、显存压力

如果告警触发后只有“发一条群消息”，没有 runbook、没有自动切流、没有降级策略，那么 SLA 仍然没有真正进入工程系统。

---

## 工程权衡与常见坑

第一个常见坑是用平均值描述延迟。平均值会掩盖尾延迟，也就是少数特别慢的请求。用户抱怨往往来自尾部，而不是均值。

| 指标 | 适合什么 | 不适合什么 |
|---|---|---|
| 平均值 | 粗看整体趋势、容量变化 | 对外 SLA 承诺 |
| p95 | 在线服务主流延迟目标 | 极端长尾分析 |
| p99 | 严格时延场景、关键客户保障 | 样本量很小的低频流量 |

第二个坑是只看 `E2E`。如果用户 8 秒后收到完整答案，看起来总耗时合规，但前 4 秒完全没有任何输出，用户体感依然很差。对流式聊天，这种“首 token 卡住”的问题只能通过 `TTFT` 暴露。

第三个坑是把内部资源指标当成对外承诺。GPU 利用率高、显存占用高、batch 大，并不等于用户体验差；它们最多是诊断信号，不是 SLA 本身。

第四个坑是只写可用性数字，不写窗口和排除项。比如“99.9% 可用”如果不写是自然月统计，团队月末结算就没有统一口径；如果不写计划维护是否排除，就会在每次升级后争议是否违约。

第五个坑是没有定义“有效请求”。如果客户传了非法参数、错误认证头、过期 token，这些请求到底算不算失败？不先定义，分母分子都会失真。

常见坑可以汇总成表：

| 常见坑 | 造成的问题 | 规避方式 |
|---|---|---|
| 从系统指标正推 SLA | 指标与用户体验脱节 | 先写业务承诺，再定义 SLI/SLO |
| 只看平均延迟 | 尾部慢请求被掩盖 | 对外目标优先用 p95/p99 |
| 只看 E2E | 漏掉首 token 卡顿 | 拆 TTFT、TPOT、E2E |
| 不写统计窗口 | 结算口径不一致 | 明确 `calendar month` 或 `rolling 30d` |
| 不写排除项 | 故障归责争议 | 明确维护、客户侧、第三方依赖边界 |
| 有告警无动作 | 问题发现了也止不住 | 告警绑定 runbook 和自动化处置 |

真实工程里还要做一个权衡：目标越细，运维成本越高。你把 SLA 拆成 10 个指标，每个指标 3 个分位数、4 个客户等级、2 个区域维度，系统会变得很难维护。对初期团队，通常先抓住 3 到 5 个真正决定用户体验的指标，比铺满仪表盘更有价值。

---

## 替代方案与适用边界

不是所有推理服务都必须采用同一套 SLA 结构。指标必须跟服务形态对齐。

| 服务类型 | 推荐 SLA 指标 | 不推荐作为主指标 |
|---|---|---|
| 流式 LLM 在线聊天 | `TTFT`、`TPOT`、`E2E`、可用性 | 只看吞吐量 |
| 非流式文本生成 | `E2E`、成功率、超时率 | 过度强调 `TTFT` |
| 批处理推理 | 作业完成率、队列等待时间、吞吐量 | `TTFT` |
| 离线任务 | 日完成率、截止时间达成率、失败重试率 | p95 首响应时间 |

可以用一个简式判断：

- 如果用户会实时盯着流式输出，就必须拆 `TTFT` 和 `TPOT`
- 如果服务一次性返回完整结果，没有流式体验，通常只看 `E2E` 就够
- 如果是批处理或离线任务，重点往往是吞吐、队列等待、完成率，而不是首 token

再看适用边界：

| 场景 | 是否需要细粒度 SLA | 原因 |
|---|---|---|
| 对外商业化 API | 需要 | 有明确承诺和违约后果 |
| 内部实验环境 | 通常不需要 | 重点是迭代速度，不是对外保障 |
| 低频离线评测 | 不需要拆 TTFT | 没有实时交互体验 |
| 多租户在线推理平台 | 强烈需要 | 需要分层限流、赔付和隔离策略 |

一个典型替代方案是：内部环境只保留两条约束，“成功率 >= 99%”和“E2E p95 <= 20s”。这种方案不够细，但维护成本低，适合实验期。等进入正式商用，再把指标拆细到 `TTFT`、`TPOT`、区域级可用性和错误预算告警。

核心原则不变：指标不是越多越好，而是要和承诺对象、服务形态、运维能力匹配。

---

## 参考资料

下表说明每类来源主要支持哪部分内容：

| 来源名称 | 适用内容 | 可支持章节 |
|---|---|---|
| Google SRE Book | SLI/SLO/SLA 与错误预算方法论 | 问题定义、核心机制 |
| Google Cloud Burn Rate 文档 | burn-rate 告警机制与告警窗口 | 核心机制、代码实现 |
| Google Cloud SLA 文档 | 月度统计窗口、有效请求、计划维护排除项 | 问题定义、工程权衡 |
| vLLM Metrics 文档 | TTFT、TPOT、E2E 等 LLM serving 指标 | 核心机制、代码实现 |
| SOLA 论文 | LLM serving 中 SLO 达成问题 | 核心机制、替代方案 |

1. [Google SRE Book: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)
2. [Google Cloud: Alerting on your burn rate](https://docs.cloud.google.com/stackdriver/docs/solutions/slo-monitoring/alerting-on-budget-burn-rate)
3. [Google Cloud: Cloud Observability SLA](https://cloud.google.com/operations/sla)
4. [vLLM Docs: Metrics](https://docs.vllm.ai/en/stable/design/metrics/)
5. [OpenReview: SOLA, Optimizing SLO Attainment for Large Language Model Serving](https://openreview.net/forum?id=ubIvpetAd6)
