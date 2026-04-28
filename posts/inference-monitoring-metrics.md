## 核心结论

推理监控的目标不是证明“模型能跑”，而是判断“线上服务是否持续满足用户体验与容量目标”。因此不能只看平均延迟，必须把一次请求拆成 `排队 - prefill - decode - 返回 - 失败` 五段分别观测，再把这些观测与业务 `SLA` 绑定。

这里先解释几个术语。`TTFT` 是 *Time To First Token*，白话就是“用户多久看到第一个字”。`TPOT` 是 *Time Per Output Token*，白话就是“后面每个字平均要等多久”。`QPS` 是 *Queries Per Second*，白话就是“系统每秒接住多少次请求”。队列长度表示“门口排了多少请求”，显存水位表示“显存用了多少比例”，`GPU` 利用率表示“算力忙不忙”。

核心判断不是单看某一个值，而是看组合关系。典型结论如下：

| 现象 | 更可能的瓶颈 | 不应先下的结论 |
|---|---|---|
| `TTFT` 高，`TPOT` 正常，队列长 | 排队、限流、网络首包 | 模型解码慢 |
| `TTFT` 正常，`TPOT` 高 | decode 慢、批处理策略不合理、`GPU` 算力紧张 | 网关抖动 |
| `TTFT` 和 `TPOT` 都高，显存水位高 | 资源吃满、`KV cache` 紧张、并发过高 | 只是偶发网络波动 |
| `QPS` 下降且 `429/503/RESOURCE_EXHAUSTED` 增多 | 系统已进入拒绝或退化阶段 | 只是“有点慢” |

实际排障时，建议顺序是：先看错误率，再看 `TTFT` 与 `TPOT`，再看队列长度，再看显存水位与 `GPU` 利用率，最后回到 `QPS` 和业务 `SLO` 判断容量是否足够。原因很直接：用户首先感知的是“是否失败”和“多久开始回复”，不是显卡曲线好不好看。

---

## 问题定义与边界

推理监控的对象是“线上推理服务”，不是单个模型算子，也不是训练系统。对线上系统，真正重要的问题只有三个：

1. 用户多久看到第一个 token。
2. 用户整段回复多久结束。
3. 请求在负载上升时是否开始失败或退化。

这就要求先划清统计边界。否则同样一个“慢”，可能来自完全不同的地方。

| 统计边界 | 观测对象 | 常见慢点 | 适合记录的时间点 |
|---|---|---|---|
| 客户端 | 用户真实体验 | 本地网络、浏览器渲染、移动端抖动 | 发起请求、收到首包、收到末包 |
| 网关 | 接入层转发 | TLS、鉴权、限流、反向代理 | 到达网关、转发后端、回写客户端 |
| 服务端 | 调度与排队 | 并发控制、队列堆积、实例不足 | 到达服务、入队、出队 |
| 模型内部 | prefill / decode 计算 | prompt 长、`KV cache` 紧张、批处理失衡 | 开始 prefill、首 token、结束 |

一个玩具例子：用户点击按钮后等了 1 秒才看到第一个字。这个 1 秒不一定是模型算得慢。可能是：

- 客户端到网关网络抖动了 300 ms。
- 网关因为限流排了 200 ms。
- 服务端队列等了 350 ms。
- 模型 prefill 真正只花了 150 ms。

如果只保留“端到端总耗时”这一列日志，后续就无法知道该扩容、改网关策略，还是优化 prompt。

这里还要区分 `SLI / SLO / SLA`。`SLI` 是 *Service Level Indicator*，白话就是“你拿来衡量服务表现的指标”。`SLO` 是“该指标要达到什么目标”，例如“`TTFT P95 < 800 ms`”。`SLA` 是“对外承诺”，通常带业务后果。监控设计应从 `SLA` 倒推 `SLO`，再决定记录哪些 `SLI`，而不是先收一堆指标再猜它们有没有用。

---

## 核心机制与推导

一次推理请求至少可以抽象为下面几个关键时间点：

- `t_arr`：请求到达服务端。
- `t_q_start`：进入队列。
- `t_pf_start`：开始 prefill。`prefill` 是“先把输入上下文跑一遍，建立后续生成需要的状态”。
- `t_1`：首个输出 token 返回。
- `t_end`：最后一个 token 返回或请求结束。
- `n_out`：输出 token 数。

基于这些时间点，可以得到一组最小但有解释力的指标：

$$
TTFT = t_1 - t_{arr}
$$

$$
TTFT \approx t_q + t_{pf} + t_{net,1}
$$

这里 `t_q` 是排队时间，`t_pf` 是 prefill 时间，`t_{net,1}` 是首包的序列化与网络传输时间。

$$
TPOT = \frac{t_{n\_out} - t_1}{n_{out} - 1}, \quad n_{out} > 1
$$

$$
QPS = \frac{N_{ok}}{\Delta t}
$$

$$
E2E = TTFT + (n_{out} - 1) \cdot TPOT
$$

$$
Memory\ Level = \frac{M_{used}}{M_{total}}
$$

$$
Error\ Rate = \frac{N_{err}}{N_{req}}
$$

这些式子重要的地方不在数学本身，而在“把慢拆开”。`TTFT` 更接近首包体验，`TPOT` 更接近持续生成速度。两者必须分开看，因为它们对应不同机制。

先看一个玩具例子。某次请求：

- 排队 `120 ms`
- prefill `280 ms`
- 首包网络与序列化 `40 ms`
- 共输出 `4` 个 token
- 后续每个 token 间隔 `15 ms`

则：

- `TTFT = 120 + 280 + 40 = 440 ms`
- `TPOT = 15 ms`
- `E2E = 440 + 3 × 15 = 485 ms`

这个例子里，用户觉得“开口有点慢，但说起来很顺”。因此主要矛盾在首包，而不在持续生成。

再看一个真实工程例子。午高峰时，某个大模型服务的平均总延迟只上涨了约 `20%`，表面上像可接受波动；但拆开后发现：

- `TTFT P95` 从 `700 ms` 涨到 `1.6 s`
- `TPOT P95` 基本不变
- 队列长度从 `3` 升到 `60`
- 显存水位长期 `94%+`
- `RESOURCE_EXHAUSTED` 与 `503` 同时增多

这组组合信号说明：问题不是 decode 单步太慢，而是系统容量边界被顶住，`KV cache` 和并发调度开始挤压新请求，导致新请求先排队、再失败。此时如果只看平均延迟，很容易误判为“还能扛”；如果只看 `GPU` 利用率，还可能误判为“卡利用率不算满，应该没问题”。

可以把常见判读规则整理成表：

| 指标组合 | 解释 | 首先检查 |
|---|---|---|
| `TTFT` 高，`TPOT` 正常 | 首包前流程慢 | 队列、网关、网络、鉴权 |
| `TTFT` 正常，`TPOT` 高 | 生成阶段慢 | decode、批大小、采样参数、GPU 算力 |
| `TTFT` 和 `TPOT` 都高 | 全链路资源紧张 | 显存、`KV cache`、并发上限 |
| `QPS` 持平但错误率升高 | 请求被拒绝或超时 | 限流策略、实例数、下游可用性 |
| `GPU` 利用率高但 `QPS` 不升 | 算力未转成有效吞吐 | 队列堆积、显存碎片、批处理失衡 |

---

## 代码实现

工程实现的关键不是“最后算几个数”，而是把每次请求的重要时间点结构化记录下来。最低限度可以先定义一条请求记录：

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class RequestTrace:
    t_arr: float
    t_q_start: float
    t_pf_start: float
    t_1: Optional[float]
    t_end: float
    n_out: int
    status: str
    error_code: Optional[str] = None

    @property
    def queue_time(self) -> float:
        return self.t_pf_start - self.t_q_start

    @property
    def prefill_time(self) -> float:
        if self.t_1 is None:
            return 0.0
        return self.t_1 - self.t_pf_start

    @property
    def ttft(self) -> Optional[float]:
        if self.t_1 is None:
            return None
        return self.t_1 - self.t_arr

    @property
    def tpot(self) -> Optional[float]:
        if self.t_1 is None or self.n_out <= 1:
            return None
        return (self.t_end - self.t_1) / (self.n_out - 1)

    @property
    def e2e(self) -> float:
        return self.t_end - self.t_arr


def error_rate(traces: list[RequestTrace]) -> float:
    if not traces:
        return 0.0
    return sum(1 for x in traces if x.status != "ok") / len(traces)


def qps(success_count: int, window_seconds: float) -> float:
    return success_count / window_seconds


# 玩具例子
req = RequestTrace(
    t_arr=0.000,
    t_q_start=0.000,
    t_pf_start=0.120,
    t_1=0.440,
    t_end=0.485,
    n_out=4,
    status="ok",
)

assert round(req.queue_time, 3) == 0.120
assert round(req.prefill_time, 3) == 0.320
assert round(req.ttft, 3) == 0.440
assert round(req.tpot, 3) == 0.015
assert round(req.e2e, 3) == 0.485
assert qps(30, 1.0) == 30.0
assert error_rate([req, RequestTrace(0, 0, 0.1, None, 0.2, 0, "error", "503")]) == 0.5
```

这段代码可以直接运行。它做了三件事：

1. 为每个请求保存统一字段。
2. 把 `TTFT / TPOT / E2E` 的计算逻辑集中起来。
3. 用 `assert` 固化最小正确性，避免后续埋点改动把公式算错。

在服务里，埋点通常至少分四类：

| 指标类型 | 适合上报什么 | 例子 |
|---|---|---|
| Histogram | 时延分布 | `ttft_seconds`、`tpot_seconds`、`e2e_seconds` |
| Counter | 次数累计 | `requests_total`、`errors_total{code="503"}` |
| Gauge | 瞬时水位 | `queue_length`、`gpu_memory_used_bytes` |
| Info / Label | 上下文标签 | 模型名、实例 ID、路由版本 |

如果用 Prometheus 思路，最常见的做法是：

- `TTFT / TPOT / E2E` 用 histogram，看 `P50/P95/P99` 和桶分布。
- 错误码用 counter，至少按 `429`、`503`、`UNAVAILABLE`、`RESOURCE_EXHAUSTED` 分层。
- 队列长度、显存水位、`GPU` 利用率用 gauge。
- 客户端、网关、服务端分别打时间点，避免边界混淆。

一个实用原则是：埋点字段宁可少，但必须语义稳定。不要今天把 `TTFT` 算到“收到第一个 token”，明天又算到“开始 decode”，否则所有历史对比都会失真。

---

## 工程权衡与常见坑

监控里最容易犯的错，是把“看起来简单”误当成“足够诊断”。

第一类坑是只看平均值。平均值会掩盖长尾，而推理系统的痛点恰恰常在长尾。一次短 prompt 和一次超长上下文请求，平均后可能都显得“还行”，但用户真正抱怨的通常是那批最慢请求。因此至少要看 `P95 / P99`，并尽量保留 histogram。

第二类坑是把 `QPS` 当目标。`QPS` 更像负载输入，不是用户承诺。用户不会因为你扛住了 `40 QPS` 就满意，用户只关心是否在约定时间内成功返回。因此 `SLO` 应该优先绑定“时延分位数 + 成功率 + 错误率”。

第三类坑是只看 `GPU` 利用率。高利用率不等于健康，低利用率也不等于没问题。比如请求被队列压住时，`GPU` 可能阶段性空闲，但用户已经在排队；相反，显存几乎打满时，`GPU` 还没到 100%，系统却已开始 `RESOURCE_EXHAUSTED`。

常见坑可以整理为：

| 常见坑 | 错误原因 | 更合理做法 |
|---|---|---|
| 只看平均延迟 | 掩盖长尾 | 看 `P95/P99` 和 histogram |
| 只看端到端耗时 | 无法定位慢在哪段 | 拆 `TTFT`、`TPOT`、队列 |
| 只看 `QPS` | 吞吐不等于体验 | 绑定 `SLO/SLA` |
| 只看 `GPU` 利用率 | 资源忙不等于用户好 | 联合显存、队列、错误码 |
| 不分错误码 | 容量问题与服务故障混在一起 | 按协议语义分层告警 |
| 时间边界不统一 | 指标横向不可比 | 固定埋点定义 |

错误码也应分层看：

| 错误码 | 常见语义 | 监控上的含义 |
|---|---|---|
| `429` | 被限流 | 接入层容量控制触发 |
| `503` | 服务暂不可用 | 实例不可用或过载退化 |
| `UNAVAILABLE` | 下游暂时不可达 | 网络、依赖、实例抖动 |
| `RESOURCE_EXHAUSTED` | 资源耗尽 | 显存、配额、连接池或并发耗尽 |

一个经验判断是：当 `TTFT P95` 上升而 `TPOT P95` 稳定，优先排队；当 `TPOT P95` 持续上升且 `GPU` 算力、显存水位同步走高，优先算力与缓存；当错误码开始抬头，说明问题已经从性能退化跨进可用性退化。

---

## 替代方案与适用边界

不是所有系统一开始都需要全套推理监控。关键在于系统处于什么阶段。

早期小系统，请求量低、用户少、模型固定，往往用“端到端耗时 + 成功率 + 基础日志”就能发现主要问题。因为那时最重要的是“能不能稳定提供服务”，不是精细容量优化。

但随着请求量上升、模型变大、并发变高，只看端到端就不够了。因为不同故障表面上都表现为“慢”，而治理动作完全不同。排队问题要扩实例或调并发；decode 问题要改批策略、采样或模型配置；显存问题要控制上下文长度、`KV cache` 或换卡型。

可以把方案对比成表：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 只做端到端监控 | 简单、接入快 | 定位能力弱 | 早期、小流量 |
| 只做 `GPU` 监控 | 资源视角清楚 | 看不到用户体验 | 单机压测、底层调优 |
| 只做日志分析 | 细节全 | 实时告警弱，汇总成本高 | 事后排障 |
| 分段时延 + 资源指标 + 错误码 | 定位强，可做容量判断 | 埋点设计更复杂 | 中大型线上推理服务 |

边界判断可以这样定：

| 情况 | 监控要求 |
|---|---|
| 日请求量低、无流式输出、业务不敏感 | 端到端耗时和错误率通常够用 |
| 有流式输出，用户关心“多久开口” | 必须拆 `TTFT` |
| 生成长度差异大，用户关心“说得顺不顺” | 必须拆 `TPOT` |
| 有明显高峰、排队、扩缩容需求 | 必须监控队列长度与 `QPS` |
| 模型大、上下文长、显存常逼近上限 | 必须监控显存水位与缓存利用率 |

所以，替代方案不是“谁对谁错”，而是“在哪个阶段足够”。真正危险的是把早期的粗监控沿用到规模化系统里，然后在告警响起时才发现没有证据判断瓶颈在哪一段。

---

## 参考资料

1. [vLLM Metrics](https://docs.vllm.ai/en/stable/design/metrics.html)：用于支持 `TTFT`、`TPOT`、请求队列、`KV cache` 使用率等推理服务指标定义。
2. [NVIDIA Triton Inference Server Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2610/user-guide/docs/user_guide/metrics.html)：用于支持队列时延、推理时延、`GPU` 利用率、显存使用等资源与服务指标。
3. [Prometheus Histograms and Summaries](https://prometheus.io/docs/practices/histograms/)：用于支持为什么不能只看平均值，而应保留分布、桶和分位数。
4. [Google SRE: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)：用于支持 `SLI/SLO/SLA` 的关系，以及监控应从用户目标倒推。
5. [gRPC Status Codes](https://grpc.io/docs/guides/status-codes/)：用于支持 `UNAVAILABLE`、`RESOURCE_EXHAUSTED` 等错误码语义与告警分层。
