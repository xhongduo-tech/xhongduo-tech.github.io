## 核心结论

模型推理性能回归，不是“模型答错了”，而是**在相同或可比负载下，推理服务的性能指标相对稳定基线出现了可重复劣化**。这里的“基线”可以理解为团队认可的正常版本；“可重复劣化”指多次测量都能看到同方向变差，而不是一次偶然波动。

判断回归时，优先看三类指标：**尾延迟、吞吐、错误率**。尾延迟是大多数请求之外那部分慢请求的表现，常用 P95 或 P99；吞吐是单位时间内处理多少请求或多少 token；错误率是超时、失败、取消等异常请求占比。只看平均延迟，常常会把真正的风险藏起来。

下面给一个最小判断表。假设压测负载完全一致：

| 指标 | 基线 | 当前 | 相对变化 | 是否回归 |
| --- | ---: | ---: | ---: | --- |
| P95 延迟 | 120 ms | 150 ms | +25.0% | 是 |
| 吞吐 | 400 req/s | 340 req/s | -15.0% | 是 |
| 错误率 | 0.10% | 0.10% | 0 | 否 |
| 综合结论 | - | - | - | 是 |

这个例子里，即使模型还能正常返回结果，也应判定为性能回归。原因很直接：同样的请求，现在更慢，而且单位时间能处理的请求更少。

---

## 问题定义与边界

要谈“性能回归”，必须先把边界定清楚。否则很多团队会把“线上今天变慢了”直接叫回归，结论往往不可靠。

一个严格定义至少包含五个条件：

| 条件 | 含义 | 为什么必须有 |
| --- | --- | --- |
| 基线版本 | 作为对照的稳定版本 | 没有对照，就没有回归 |
| 当前版本 | 被评估的新版本 | 要知道是谁变了 |
| 相同或可比负载 | 请求类型、输入长度、并发度、持续时间一致 | 否则数值不可比 |
| 稳定采样窗口 | 排除冷启动和瞬时抖动后再统计 | 避免把噪声当结论 |
| 明确阈值 | 例如 P95 上升超 10% 算回归 | 避免主观争论 |

反过来看，下面这些情况通常**不属于严格意义上的回归**：

| 不可直接比较的情况 | 为什么不能直接下结论 |
| --- | --- |
| 请求类型变了 | 业务流量结构已经不是同一组工作负载 |
| prompt 长度分布变了 | 长上下文会显著改变 prefill 和 KV cache 压力 |
| 机器规格变了 | GPU、CPU、内存、网络变化都会影响结果 |
| 部署拓扑变了 | 单机、多机、跨可用区的排队和通信成本不同 |
| 压测窗口太短 | 一次短测可能只采到偶然状态 |

这里最容易混淆的，是“现象”和“回归”的区别。

玩具例子：线上某天晚高峰响应变慢，这只是现象。可能是流量涨了，也可能是下游 Redis 抖动了，还可能是某个节点刚重启。只有当你把**新版本**和**旧版本**放到**相同压测条件**下比较，并且看到新版本稳定更差，才叫性能回归。

真实工程里，很多“回归事故”其实是基线没定义好。例如旧版本跑在 A100，新版本跑在 L40S；或者旧版本用 2k prompt 基准，新版本已经支持 32k 上下文，结果团队还想直接对比总吞吐。这种比较在方法上就不成立。

---

## 核心机制与推导

先给最常用的回归度量公式。设某项指标的基线值为 $M_b$，当前值为 $M_c$，则相对变化率是：

$$
\Delta m = \frac{M_c - M_b}{M_b}
$$

这个公式的作用是把“绝对差”变成“相对变化”，便于跨版本和跨指标比较。比如 P95 从 120 ms 变到 150 ms，绝对差是 30 ms，但更有意义的是相对上升了 25%。

对推理服务，常看的不是一个指标，而是一组指标：

| 指标 | 看什么 | 回归方向 |
| --- | --- | --- |
| P95 延迟 | 大多数用户在高位分位上的体验 | 越高越差 |
| P99 延迟 | 最慢那部分请求是否被放大 | 越高越差 |
| 吞吐 | 单位时间服务能力 | 越低越差 |
| 错误率 | 超时、失败、取消是否上升 | 越高越差 |

对不同方向的指标，计算方式略有不同。延迟和错误率变大是坏事，吞吐变小是坏事，因此常写成：

$$
\Delta P95 = \frac{P95_c - P95_b}{P95_b}
$$

$$
\Delta X = \frac{X_b - X_c}{X_b}
$$

其中 $X$ 表示吞吐。这样写的好处是：正值统一表示“变差”。

为什么尾延迟会比平均值更早恶化？核心是排队。排队论里最常用的关系之一是 Little's Law：

$$
N \approx X \times R
$$

其中：

- $N$ 是系统中的在途请求数，可以白话理解成“正在系统里排队或处理的请求总量”
- $X$ 是吞吐，可以白话理解成“每秒处理多少请求”
- $R$ 是平均响应时间，可以白话理解成“一个请求从进来到结束平均要等多久”

这个式子告诉我们：如果吞吐 $X$ 不变，而响应时间 $R$ 变长，那么系统里的在途请求数 $N$ 会增加。请求一多，队列更长，排队等待更久，尾部那批请求会被放大得最明显，所以 P95 和 P99 往往先坏掉。

可以看一个玩具例子。

假设基线时吞吐 $X=400$ req/s，平均响应时间 $R=0.10$ s，那么：

$$
N \approx 400 \times 0.10 = 40
$$

如果某次升级后，服务单次处理慢了一点，平均响应时间升到 $0.13$ s，而入口流量没有下降，那么：

$$
N \approx 400 \times 0.13 = 52
$$

在途请求从 40 涨到 52，队列更深。平均延迟只涨了 30 ms，但尾延迟往往不止涨 30 ms，因为新来的请求会被前面的积压连带拖慢。这就是“服务稍慢一点，尾部恶化很多”的原因。

真实工程例子更典型。一个 LLM 服务升级成 continuous batching 后，GPU 利用率上去了，短 prompt benchmark 看起来更快；但生产流量中混有大量长 prompt 和长输出，decode 阶段共享批次更复杂，长请求在批里待得更久，于是 P99 反而升高。这不是公式失效，而是公式揭示了同一个事实：**局部效率提升，不等于全局排队代价下降**。

---

## 代码实现

最小可用的回归检测流程，不是“把两个数字一减”这么简单。工程里至少要考虑四件事：warmup、重复采样、阈值判断、结果落盘。warmup 是预热，意思是让服务先跑到稳定状态；重复采样是多跑几轮，降低偶然误差；阈值判断是把“变差多少算回归”写成规则；结果落盘是把数据保存下来，方便后续追查。

先看一个新手能读懂的伪代码：

```text
1. 启动待测服务
2. 先进行 warmup，不记录结果
3. 在固定负载下重复压测 N 轮
4. 汇总每轮的 P95、P99、吞吐、错误率
5. 计算当前值相对基线的变化率
6. 按阈值判断 is_regression
7. 将结果写入 JSON 或数据库
8. 如有回归，报警或阻止发布
```

下面给一个可运行的 Python 最小实现。它不依赖具体压测工具，但保留了实际流程中的关键结构。

```python
from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import median
from pathlib import Path
import json


@dataclass
class Metrics:
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    error_rate: float


def relative_increase(current: float, baseline: float) -> float:
    assert baseline > 0
    return (current - baseline) / baseline


def relative_drop(current: float, baseline: float) -> float:
    assert baseline > 0
    return (baseline - current) / baseline


def aggregate_runs(runs: list[Metrics]) -> Metrics:
    assert runs, "runs must not be empty"
    return Metrics(
        p95_ms=median([r.p95_ms for r in runs]),
        p99_ms=median([r.p99_ms for r in runs]),
        throughput_rps=median([r.throughput_rps for r in runs]),
        error_rate=median([r.error_rate for r in runs]),
    )


def detect_regression(
    baseline_metrics: Metrics,
    current_metrics: Metrics,
    thresholds: dict[str, float],
) -> dict:
    deltas = {
        "p95_increase": relative_increase(current_metrics.p95_ms, baseline_metrics.p95_ms),
        "p99_increase": relative_increase(current_metrics.p99_ms, baseline_metrics.p99_ms),
        "throughput_drop": relative_drop(current_metrics.throughput_rps, baseline_metrics.throughput_rps),
        "error_rate_increase": current_metrics.error_rate - baseline_metrics.error_rate,
    }

    is_regression = (
        deltas["p95_increase"] > thresholds["p95_increase"]
        or deltas["p99_increase"] > thresholds["p99_increase"]
        or deltas["throughput_drop"] > thresholds["throughput_drop"]
        or deltas["error_rate_increase"] > thresholds["error_rate_increase"]
    )

    return {
        "baseline_metrics": asdict(baseline_metrics),
        "current_metrics": asdict(current_metrics),
        "thresholds": thresholds,
        "deltas": deltas,
        "is_regression": is_regression,
    }


def save_result(path: str, result: dict) -> None:
    Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    # warmup 之后的多轮稳定结果
    baseline_runs = [
        Metrics(118, 155, 402, 0.0010),
        Metrics(121, 160, 398, 0.0010),
        Metrics(120, 158, 401, 0.0010),
    ]
    current_runs = [
        Metrics(149, 214, 342, 0.0010),
        Metrics(151, 220, 338, 0.0012),
        Metrics(150, 218, 340, 0.0011),
    ]

    baseline_metrics = aggregate_runs(baseline_runs)
    current_metrics = aggregate_runs(current_runs)
    thresholds = {
        "p95_increase": 0.10,
        "p99_increase": 0.15,
        "throughput_drop": 0.05,
        "error_rate_increase": 0.0010,
    }

    result = detect_regression(baseline_metrics, current_metrics, thresholds)
    save_result("/tmp/model_regression_result.json", result)

    assert round(result["deltas"]["p95_increase"], 3) == 0.25
    assert round(result["deltas"]["throughput_drop"], 3) == 0.15
    assert result["is_regression"] is True

    print(json.dumps(result, indent=2, ensure_ascii=False))
```

这个实现里，`baseline_metrics`、`current_metrics`、`thresholds`、`is_regression` 四个核心变量都明确出现了。中位数聚合的目的，是降低单轮抖动对结论的影响。真实系统中，你还可以把 `aggregate_runs` 换成更稳健的统计方法，比如直接比较每轮分布，或者用置信区间约束误判。

如果要接入现有压测工具，通常会采用“外部压测 + 内部判定”的方式。比如：

```bash
perf_analyzer -m my_model --concurrency-range 32:32:1 --measurement-interval 10000
python detect_regression.py --baseline baseline.json --current current.json
```

真实工程例子是这样的：你在 CI 里固定一套 prompt 数据集，先对主分支 nightly 产出基线，再对待发布版本跑同样压测。如果 P95 超阈值或吞吐降幅超阈值，CI 直接失败。这比发布后再看监控更便宜，因为问题在预发布阶段就被挡住了。

---

## 工程权衡与常见坑

性能回归检测不是只靠公式，还要接受工程权衡。很多优化并不是单向收益，而是“某个指标变好，另一个指标变坏”。

一个常见权衡表如下：

| 方案 | 可能收益 | 可能代价 |
| --- | --- | --- |
| 更大 batch | 吞吐提升 | 单请求等待变长，尾延迟变差 |
| 更激进连续批处理 | GPU 利用率更高 | 混合长度请求下 P99 更不稳定 |
| 更低精度量化 | 显存占用下降、成本降低 | 某些模型路径更慢，或输出质量受影响 |
| 更长上下文支持 | 功能边界更强 | prefill 更慢，KV cache 压力上升 |
| 更少副本数 | 成本下降 | 抖动放大，故障冗余变差 |

最常见的坑不是“不会测”，而是“测错了问题”。

| 坑点 | 为什么错 | 如何规避 |
| --- | --- | --- |
| 只看平均值 | 平均值会掩盖慢请求，用户经常感知的是尾延迟 | 必看 P95、P99 |
| 只测短输入 | 短输入主要测到轻载路径，长 prompt 的 prefill 压力没暴露 | 基准集必须覆盖长短分布 |
| 只跑一次 | 一次结果可能只是随机抖动 | 至少多轮重复并聚合 |
| 不做 warmup | 模型加载、CUDA 图编译、缓存建立会污染结果 | 单独预热，稳定后再记数 |
| 新旧负载不一致 | 比较对象已经变了 | 固定数据集、并发、持续时间和环境 |
| 只看离线 benchmark | 离线环境没有线上排队和混合流量 | 离线结果要用线上灰度补充验证 |
| 忽略错误率 | 有时系统靠超时和失败“换吞吐” | 延迟、吞吐、错误率必须一起看 |

continuous batching 是一个典型坑。它常让短压测更漂亮，因为 GPU 空闲时间减少了；但如果真实流量里有很多长 prompt，批次会被长请求拖住，后面的短请求也要一起等，结果 P99 变坏。这里的教训不是“continuous batching 不好”，而是**实验室指标不能代替生产流量**。

另一个真实工程坑，是把模型升级和环境升级绑在一起。比如同时改了模型量化方案、TensorRT 版本、CUDA 驱动、容器基础镜像。结果一旦变慢，根因就不清楚了。更稳妥的方法是单变量推进：先锁环境测模型，再锁模型测引擎，再锁引擎测部署参数。

---

## 替代方案与适用边界

性能回归检测不是万能工具。它解决的是“新旧版本在可比条件下是否变差”，但不是所有性能问题都适合这样处理。

下面把几类常见方法放在一起看：

| 方法 | 解决什么问题 | 适用场景 | 不适合什么 |
| --- | --- | --- | --- |
| 性能回归检测 | 新版本是否比基线更差 | 发布前验证、CI 守门 | 业务流量已完全变化 |
| 灰度发布 | 新版本在真实流量下是否安全 | 发布阶段小流量试运行 | 没有回滚能力的系统 |
| 线上监控 | 系统现在是否健康 | 持续运营、告警 | 不能单独证明“是版本回归” |
| profiler 分析 | 慢在哪里 | 定位根因、性能优化 | 不负责判断业务影响 |

适用边界的核心是：**基线必须代表当前业务场景**。如果负载分布、部署拓扑、业务目标变化很大，旧基线就失效了。

举两个边界例子。

玩具例子：原来你的基准集全部是 512 token 输入，现在产品改成了文档问答，大量请求是 8k token 输入。即使新版本的 P95 比旧版本高很多，也不能直接说“回归”，因为工作负载已经变了。正确做法是重建新的基准集。

真实工程例子：系统从单机单卡升级成多机多卡，服务链路里新增了调度器和远端 KV 传输。此时延迟构成已经不同，直接拿单机基线对比总响应时间没有意义。更合理的方法是分层评估：分别看单卡算力基准、跨机通信开销、端到端 P95，再决定是“架构切换带来的合理变化”，还是“某个版本的异常退化”。

所以，替代方案不是互斥关系，而是组合关系。常见顺序是：

1. 离线 benchmark 建立稳定基线。
2. 回归检测拦截明显变差的版本。
3. 灰度发布验证真实流量表现。
4. 线上监控持续观察长期漂移。
5. profiler 在发现问题后做根因定位。

这套链路比只做单点压测稳健得多，因为它把“是否变差”“是否可发布”“哪里变慢”拆成了不同层次的问题。

---

## 参考资料

| 资料名 | 说明 | 对应正文章节 |
| --- | --- | --- |
| The Tail at Scale | 解释为什么尾延迟在大规模系统中会被放大 | 核心机制与推导 |
| Little's Law | 解释在途请求、吞吐、响应时间的基本关系 | 核心机制与推导 |
| Triton Perf Analyzer | 说明如何用稳定窗口做推理压测 | 代码实现 |
| TensorRT-LLM Benchmarking | 说明 LLM benchmark 与 batching 的关键约束 | 工程权衡与常见坑 |
| Kubernetes Probes | 说明健康探针与性能问题不是同一类机制 | 替代方案与适用边界 |

1. [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)
2. [Little's Law](https://mathworld.wolfram.com/LittlesLaw.html)
3. [NVIDIA Triton Performance Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/README.html)
4. [TensorRT-LLM Performance Benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html)
5. [Kubernetes: Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)
