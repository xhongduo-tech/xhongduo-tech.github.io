## 核心结论

可观测性，白话说，就是只看系统对外暴露出来的信号，也能反推出系统内部发生了什么。它不等于“多画几个监控图”，而是要把三类信号接起来：

| 支柱 | 主要回答的问题 | 典型载体 | 最适合做什么 |
|---|---|---|---|
| 指标 Metrics | 系统整体是否变差了 | 延迟、QPS、错误率、GPU 利用率 | 看趋势、设阈值、做告警 |
| 日志 Logs | 某一时刻具体发生了什么 | 结构化 JSON 日志、异常堆栈 | 查单次事件、复盘错误 |
| 追踪 Traces | 一次请求到底慢在哪一段 | Trace ID、Span、调用链 | 找瓶颈、定位跨服务问题 |

只依赖单一信号很容易误判。比如延迟升高，不一定是 CPU 满了，也可能是下游数据库变慢，或者消息队列堆积。正确做法是：

1. 用指标先发现“哪里异常”。
2. 用日志确认“异常时发生了什么事件”。
3. 用追踪还原“请求经过了哪些服务，慢在了哪一跳”。

对大模型服务尤其如此。因为它同时受应用逻辑、模型推理、GPU 资源、外部依赖四层因素影响。真正有价值的监控，至少要覆盖推理延迟、吞吐量、错误率、内存使用、GPU 利用率，并且让日志和追踪都带上同一个 Trace ID。

一个真实工程例子：订单服务的吞吐量突然下降。指标显示 QPS 没变，但 p95 延迟从 800ms 升到 3s；日志里出现大量 `db_timeout=true`；追踪中可以看到 `payment-service -> db` 这一段 span 延迟陡增。三者合并后，故障范围会从“订单系统变慢”收敛到“支付链路数据库超时”，排障时间通常能从几十分钟降到几分钟。

---

## 问题定义与边界

监控，白话说，是“提前定义好要看哪些数”；可观测性则更进一步，是“即使没提前想到具体故障，也能靠输出信号推断内部状态”。这就是两者的边界差异。

对初学者最重要的边界，不是“工具用哪个”，而是先知道系统至少要回答四个问题，也就是常说的四大黄金信号：

| 黄金信号 | 定义 | 常见指标 | 典型告警/看板场景 |
|---|---|---|---|
| Latency 延迟 | 请求从进入到返回花了多久 | p50/p95/p99 延迟、推理耗时 | 用户体验变差、尾延迟飙升 |
| Traffic 流量 | 系统每秒在处理多少请求 | QPS、tokens/s、消息消费速率 | 容量评估、突发流量识别 |
| Errors 错误 | 请求失败的比例 | 5xx 比例、超时率、模型调用失败率 | 服务可用性下降 |
| Saturation 饱和度 | 资源是否接近上限 | CPU、内存、队列长度、GPU SM 利用率 | 容量瓶颈、雪崩前兆 |

这四类信号不是随便列的，而是覆盖了“慢、忙、错、满”四种核心故障模式。

玩具例子：一个只有 `/predict` 接口的小服务，平时每秒处理 100 个请求。某天延迟从 200ms 升到 1.5s，但 QPS 基本不变。这个现象说明问题大概率不在“流量突然变大”，而在“单位请求的处理链变慢”。如果同时看到错误率没升，说明还没有完全失败，更像是资源饱和或下游变慢。

在工程落地里，还要再加一层业务边界：不是所有异常都值得半夜把人叫起来。这里就引出 SLO 和 SLA。

SLA，白话说，是对外承诺；SLO，白话说，是团队内部盯住的服务目标。比如“99.9% 请求在 2 秒内完成”就是一个典型 SLO。可观测性的目标不是把所有波动都报警，而是围绕这些目标判断：哪些异常真的影响用户，哪些只是噪声。

如果一天允许 0.1% 的失败率，那么错误预算就是：

$$
\text{Error Budget} = 1 - \text{SLO}
$$

当 SLO 为 $99.9\%$ 时，错误预算就是 $0.1\%$。这个预算不是让系统“允许出错”，而是给团队一个工程决策边界：只有当预算消耗过快时，告警和人工介入才更合理。

---

## 核心机制与推导

一个可用的可观测性系统，通常按下面的链条工作：

1. 采集信号：应用暴露指标，打印结构化日志，生成 trace/span。
2. 聚合存储：Prometheus 抓指标，ELK 或 Loki 收日志，Jaeger 或 Tempo 收追踪。
3. 关联分析：用请求 ID、Trace ID 把三类信号串起来。
4. 告警决策：根据阈值、趋势、SLO、错误预算触发不同级别的告警。

为什么新手常觉得“图很多但没法排障”？核心原因通常不是采集不够，而是关联不够。指标是聚合后的结果，日志是离散事件，追踪是单次请求的路径。只有三者共享上下文，系统才真正可问答。

尾延迟是这里的关键概念。所谓 p95 延迟，白话说，就是 95% 的请求都比这个值快，只剩最慢的 5% 更慢。它比平均值更适合发现用户真实感知的问题，因为平均值会被大量正常请求“冲淡”。

在 Prometheus 里，尾延迟通常用直方图近似计算。典型表达式是：

```promql
histogram_quantile(
  0.95,
  sum(rate(llm_request_latency_seconds_bucket[5m])) by (le)
) > 2
```

它的含义是：最近 5 分钟内，系统的 p95 请求延迟是否超过 2 秒。

如果再配合告警条件：

```yaml
for: 5m
```

完整意思就变成：只有当 p95 延迟连续 5 分钟都超过 2 秒，才触发告警。这样做的原因很直接：瞬时抖动不值得打断人，持续恶化才值得升级。

进一步地，告警不应只盯单个阈值，而要结合错误预算的消耗速度。例如某服务每天允许 0.1% 的失败率，如果当前 30 分钟内失败率快速冲到 1%，那不是“超一点点”，而是在高速度消耗预算，这种情况优先级应明显高于偶发的 0.12%。

可以把这套逻辑理解成一个简化推导：

$$
\text{告警强度} \propto \text{影响范围} \times \text{持续时间} \times \text{预算消耗速度}
$$

也就是说，真正该报警的不是“有波动”，而是“波动正在持续影响用户，并快速吃掉系统容错空间”。

真实工程例子：一个 LLM 推理服务部署在 EKS 上。应用层指标显示 `llm_request_latency_seconds` 上升，硬件层指标显示 `gpu_memory_usage_bytes` 接近上限，日志里出现 `oom_retry=true`，追踪里 `embedding-service` 正常但 `inference-worker` span 明显变长。这个时候根因不是“服务整体慢”，而是“推理工作进程因显存紧张发生重试，导致尾延迟扩大”。

---

## 代码实现

下面用一个简化的 Python 示例，把指标、结构化日志和 Trace ID 串起来。它不是完整生产实现，但逻辑可以直接运行。

```python
import json
import time
import uuid
from collections import defaultdict

class Counter:
    def __init__(self):
        self.value = 0

    def inc(self, n=1):
        self.value += n

class Histogram:
    def __init__(self):
        self.values = []

    def observe(self, value):
        self.values.append(value)

    def percentile(self, p):
        assert self.values, "histogram is empty"
        data = sorted(self.values)
        idx = int((len(data) - 1) * p)
        return data[idx]

request_total = Counter()
error_total = Counter()
latency_hist = Histogram()
status_count = defaultdict(int)

def log_event(trace_id, event, **fields):
    record = {"trace_id": trace_id, "event": event, **fields}
    print(json.dumps(record, ensure_ascii=False))

def run_inference(x):
    # 玩具推理逻辑：输入过大时变慢，模拟真实服务中的“慢请求”
    time.sleep(0.01 if x < 5 else 0.05)
    if x < 0:
        raise ValueError("invalid input")
    return x * 2

def handle_request(x):
    trace_id = str(uuid.uuid4())
    start = time.time()
    request_total.inc()
    try:
        log_event(trace_id, "request_start", input=x)
        result = run_inference(x)
        latency = time.time() - start
        latency_hist.observe(latency)
        status_count[200] += 1
        log_event(trace_id, "request_ok", result=result, latency_ms=round(latency * 1000, 2))
        return {"trace_id": trace_id, "result": result}
    except Exception as e:
        latency = time.time() - start
        latency_hist.observe(latency)
        error_total.inc()
        status_count[500] += 1
        log_event(trace_id, "request_error", error=str(e), latency_ms=round(latency * 1000, 2))
        return {"trace_id": trace_id, "error": str(e)}

# 运行几个请求
ok = handle_request(3)
slow = handle_request(8)
bad = handle_request(-1)

# 基本断言
assert "result" in ok and ok["result"] == 6
assert "result" in slow and slow["result"] == 16
assert "error" in bad
assert request_total.value == 3
assert error_total.value == 1
assert status_count[200] == 2
assert status_count[500] == 1
assert latency_hist.percentile(0.95) >= latency_hist.percentile(0.5)
```

这个例子里每一类信号的作用是明确的：

| 代码元素 | 产出的信号 | 用途 |
|---|---|---|
| `request_total.inc()` | 请求计数指标 | 看吞吐量、算 QPS |
| `error_total.inc()` | 错误计数指标 | 算错误率 |
| `latency_hist.observe()` | 延迟直方图指标 | 算 p95/p99 |
| `log_event(...trace_id...)` | 结构化日志 | 记录单次事件细节 |
| `trace_id` | 关联字段 | 串起日志、指标、追踪上下文 |

如果要接到真实栈里，通常会做三件事：

1. 用 `prometheus_client` 暴露 `/metrics`。
2. 用 OpenTelemetry 给 HTTP 请求、数据库调用、模型推理各自创建 span。
3. 在日志格式里强制输出 `trace_id`、`span_id`、`service_name`、`status_code`。

一个简化的 FastAPI 方向示意如下：

```python
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import time

app = FastAPI()
REQ = Counter("llm_request_total", "Total requests", ["endpoint", "status"])
LAT = Histogram("llm_request_latency_seconds", "Request latency", ["endpoint"])

@app.get("/predict")
def predict():
    start = time.time()
    status = "200"
    try:
        time.sleep(0.1)
        return {"output": "ok"}
    except Exception:
        status = "500"
        raise
    finally:
        REQ.labels("/predict", status).inc()
        LAT.labels("/predict").observe(time.time() - start)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

在生产里，再加 OpenTelemetry 后，一次请求就会同时生成：

1. Prometheus 可抓取的计数器和直方图。
2. 带 `trace_id` 的 JSON 日志。
3. 进入 Jaeger 或 Tempo 的调用链。

这样 Grafana 面板上看到 p95 异常时，可以直接点进 trace，再反查对应日志，而不是在三个系统里盲搜。

---

## 工程权衡与常见坑

可观测性系统最大的成本，不是工具 license，而是“采多少、存多久、告多勤、关联多深”的权衡。

常见坑可以直接总结成下面这张表：

| 常见坑 | 具体表现 | 后果 | 改进策略 |
|---|---|---|---|
| 只看单一指标 | 只盯 CPU 或 QPS | 根因判断失真 | 用黄金信号做最小覆盖 |
| 日志不结构化 | 全是字符串拼接 | 无法聚合、难检索 | 输出 JSON，固定字段名 |
| 没有 Trace ID | 指标、日志、追踪互相断开 | 排障靠猜 | 全链路透传 `trace_id` |
| 追踪采样过低 | 异常请求恰好没被采到 | 瓶颈不可复现 | 错误请求全采样，正常请求按比例采样 |
| 告警太多 | 每逢流量高峰就狂响 | 团队对告警失去信任 | 以 SLO 和用户症状为中心 |
| 标签维度爆炸 | `user_id`、`session_id` 进 metrics label | Prometheus 基数失控 | 高基数字段放日志，不放指标 |
| 只监控应用，不监控 GPU/队列 | LLM 服务看起来“代码正常” | 硬件瓶颈无法解释 | 同时采应用层和资源层指标 |

这里最容易被忽略的是“标签基数”。基数，白话说，就是一个指标会拆成多少种标签组合。比如 `request_total{user_id="..."}` 如果把每个用户都放进去，Prometheus 就会存下海量时间序列，内存和查询性能都会恶化。规则很简单：高基数字段进日志，不进指标。

另一个高频误区是告警泛滥。很多团队刚接入监控时，会把 CPU、内存、磁盘、网络、线程数、容器重启次数全配阈值告警，最后的结果通常是每天几十条噪声，工程师学会的不是响应，而是静音。

更稳妥的做法是分层：

1. 用户感知告警：SLO、错误率、尾延迟，直接触发值班通知。
2. 症状支撑告警：队列堆积、GPU OOM、数据库连接池耗尽，进工作群或工单。
3. 诊断看板：节点 CPU、磁盘、单机内存等，只上 dashboard，不直接 page。

真实工程例子：某 API 新部署后建了 30 多条基础设施告警。晚高峰时 CPU 短时到 85%，磁盘 IO 抖动，线程数上升，连续打出十几条通知，但用户请求成功率仍在 99.95% 以上。团队最后把大量告警静音。调整后只保留“5xx 错误率持续 5 分钟超过阈值”“p95 超过 SLO”“队列延迟持续积压”三类症状告警，噪声立刻大幅下降，真正报警时也更可信。

---

## 替代方案与适用边界

如果只记一个原则，就是不要先问“最强方案是什么”，而要先问“当前阶段最小可用方案是什么”。

常见路线大致有三类：

| 方案 | 覆盖能力 | 成本/维护 | 适合场景 |
|---|---|---|---|
| Prometheus + Grafana + Loki/ELK + Jaeger | 指标、日志、追踪全覆盖 | 自建成本较高，灵活性最高 | 有运维能力，想自主管控 |
| CloudWatch / Container Insights + Grafana | 云资源和应用指标接入快 | 托管方便，但定制度受限 | AWS 上快速落地 |
| Langfuse / Helicone 等 AI 平台 | AI tracing、token 成本、会话分析更强 | 平台依赖更高 | LLM 产品、Prompt 调优场景 |
| 只做基础监控 | 仅少量指标和日志 | 最低成本 | 单体应用、早期项目 |

对于 LLM 推理场景，云原生方案有一个明显优势：它能从硬件层直接拿到 GPU 指标，例如显存占用、功耗、SM 活动率。这些指标对解释“为什么模型服务变慢”很关键，因为很多问题不发生在 Python 代码里，而发生在显存压力、批处理配置、设备饱和度上。

一个实际可行的演进路径是：

1. 初期：先有 Prometheus 指标和结构化日志，确保能看延迟、吞吐量、错误率。
2. 中期：接入 OpenTelemetry 和 Jaeger，把关键链路追起来。
3. LLM 场景增强：补 GPU 指标、tokens/s、每请求 token 数、模型调用失败分类。
4. 成熟期：用 SLO、错误预算、告警分级替代“见数就报”。

玩具例子是一个单体 Flask 服务。它只需要请求总数、错误数、延迟分位数，外加 JSON 日志，追踪甚至可以先不做。因为链路很短，靠日志就能定位大部分问题。

真实工程例子则完全不同。一个在线问答系统包含 API 网关、检索服务、重排服务、向量库、推理服务、计费系统。这里如果没有 trace，单看日志几乎无法回答“本次请求慢在检索、重排还是推理”。如果没有 GPU 指标，也无法判断是模型批处理配置问题，还是外部依赖问题。此时标准的 Prometheus 栈加追踪系统是下限，AI 专用平台则适合在此基础上补 token 成本、会话级分析和 prompt 质量观测。

结论不是“所有系统都要上全家桶”，而是：系统复杂度一旦跨过单体应用边界，只靠传统监控就不够了；而一旦进入大模型服务场景，只看应用指标也不够了，必须把资源层和推理层一起纳入。

---

## 参考资料

- Hakia, *Observability: Logs, Metrics, and Traces*  
  https://www.hakia.com/engineering/observability/
- Amazon EKS Best Practices, *AI/ML Observability*  
  https://docs.aws.amazon.com/eks/latest/best-practices/aiml-observability.html
- Cloudraft, *LLM Observability*  
  https://www.cloudraft.io/blog/llm-observability
- Grafana Labs, *What is observability? Best practices, key metrics, methodologies, and more*  
  https://grafana.com/blog/2022/07/01/what-is-observability-best-practices-key-metrics-methodologies-and-more/
