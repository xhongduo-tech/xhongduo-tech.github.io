## 核心结论

性能测试的目标不是“把系统打满”，而是回答一个更具体的问题：在给定负载下，系统能否以可接受的延迟、吞吐量、错误率和资源利用率满足 SLA。SLA 可以理解为服务承诺，例如“95% 请求在 300 ms 内完成，错误率低于 0.1%”。

从实验室到生产，通常不是一次测试，而是一条连续链路：

| 阶段 | 主要目标 | 常见负载级别 | 关注指标 | 输出结果 |
| --- | --- | ---: | --- | --- |
| 实验室环境 | 找到基本瓶颈，校验脚本与口径 | 低到中 | 平均延迟、吞吐、CPU、功能正确性 | 初始基线 |
| 预生产环境 | 验证上线风险，逼近真实容量 | 中到高 | P95/P99 延迟、错误率、资源、依赖服务状态 | 发布前结论 |
| 生产灰度/小流量探针 | 验证真实流量下无回归 | 低 | 真实延迟、异常、饱和度、告警情况 | 最终上线判断 |

一个面向新手的最小例子：

假设你在预生产环境模拟 100 个并发用户，持续 10 分钟执行“登录 -> 首页 -> 推荐列表”链路，测得平均响应时间 200 ms，P95 延迟 320 ms，吞吐量 50 r/s，错误率 0%，而历史版本基线是 45 r/s。结论不是“现在更快了”这么简单，而是“在相同环境、相同数据规模、相同用户路径、相同统计窗口下，本次没有性能退化，并且吞吐高于历史基线”。

真正有价值的性能测试，必须同时满足三件事：

1. 场景明确。要知道测的是哪条业务链路，而不是把所有接口混成一组数字。
2. 基线稳定。要能与历史稳定版本比较，而不是只看当前这一轮结果。
3. 生产可验证。要在真实环境中做小流量确认，证明实验室与预生产结论没有失真。

同一组结果，通常至少要落到下面这样的表里：

| 指标 | 当前版本 | 历史版本 | 目标 SLA | 是否达标 |
| --- | ---: | ---: | ---: | --- |
| 平均延迟 | 200 ms | 210 ms | < 250 ms | 是 |
| P95 延迟 | 320 ms | 340 ms | < 400 ms | 是 |
| P99 延迟 | 410 ms | 460 ms | < 500 ms | 是 |
| 吞吐量 | 50 r/s | 45 r/s | > 40 r/s | 是 |
| 错误率 | 0.05% | 0.08% | < 0.1% | 是 |
| CPU 利用率 | 68% | 70% | < 80% | 是 |
| 内存利用率 | 61% | 58% | < 75% | 是 |

如果没有明确场景、没有稳定基线、没有生产侧验证，性能测试结果通常只能说明“某次机器上跑出了一组数字”，不能说明系统真的可上线。

---

## 问题定义与边界

性能测试首先要定义边界。边界的意思是：这次实验到底测什么、不测什么。否则同一个系统，换一个数据集、换一个网络、换一个数据库配置，结论就会失真。

最核心的四个问题如下：

| 问题 | 需要明确的内容 | 典型错误 |
| --- | --- | --- |
| 测什么 | 登录、首页、搜索、下单、写库接口 | 把冷门接口当核心链路 |
| 在哪测 | 实验室、预生产、生产灰度 | 在本地电脑上得出上线结论 |
| 负载多大 | 并发用户数、请求速率、持续时间、升压方式 | 只说“压 500 并发”但不说请求类型 |
| 用什么指标判定 | 延迟、吞吐、错误率、资源利用、依赖服务状态 | 只看平均延迟，不看长尾延迟 |

这里先引入一个基础公式。Little’s Law 是排队论中的基本关系，可以把“系统里有多少在途请求”“单位时间处理多少请求”“每个请求平均停留多久”联系起来：

$$
N \approx X \times R
$$

其中：

- $N$ 是平均并发数，也就是系统内同时存在的请求数
- $X$ 是吞吐量，也就是每秒完成多少请求
- $R$ 是平均响应时间，也就是一个请求从发出到完成花了多久

白话解释：系统里同时“堆着”的请求数量，大致等于“每秒处理速度”乘以“每个请求平均要待多久”。

例如目标是每秒处理 200 个请求，平均响应时间控制在 250 ms，也就是 0.25 秒，那么系统内部平均并发大约是：

$$
N \approx 200 \times 0.25 = 50
$$

这不是容量上限，而是一个设定测试目标的起点。它告诉你：如果你希望达到这样的吞吐和延迟，系统至少要能稳定承受约 50 个在途请求。

还可以再补一个常用量，用来理解“离饱和还有多远”。如果平均每个请求真正占用服务时间为 $S$，请求到达速率为 $\lambda$，那么资源利用率常写成：

$$
\rho \approx \lambda \times S
$$

其中 $\rho$ 越接近 1，排队越容易急剧放大。工程里不一定精确按这个公式计算，但它提供了一个重要直觉：吞吐不是无限上升的，资源接近饱和后，先恶化的通常不是吞吐，而是尾延迟。

一个玩具例子：

你写了一个只有“加法计算”的 HTTP 服务，本地单机测试时每个请求只用 5 ms。你以为系统很快，但上线后发现平均响应时间变成 120 ms。原因不是算法退化，而是边界变了：生产环境多了鉴权、日志、数据库连接池、负载均衡、跨机网络、缓存未命中、序列化开销。这说明实验室环境只能用于定位基础问题，不能直接替代生产结论。

一个更接近工程实际的例子：

在预生产环境中，用 Locust 模拟 500 名用户执行“登录 -> 加载首页 -> 拉取推荐列表”。如果预生产的 CPU、内存、网络带宽、数据库规格与生产不一致，那么测到的吞吐量和错误率无法用于发布决策。比如生产是 8 核 16G，而预生产只有 2 核 4G，结果很可能把“环境瓶颈”误判成“代码瓶颈”。

因此，问题定义阶段至少要冻结以下条件：

| 维度 | 至少要写清楚什么 | 不写清楚会导致什么问题 |
| --- | --- | --- |
| 接口或页面范围 | 哪些 URL、哪些方法、哪些页面事件 | 数据不可复现 |
| 用户行为路径 | 登录后点什么、读多还是写多、任务权重 | 场景失真 |
| 流量模型 | 突发、均匀、阶梯上升、波峰波谷 | 结果不能映射真实业务 |
| 环境配置 | CPU、内存、数据库版本、缓存开关、网络条件 | 环境噪声掩盖代码问题 |
| 成功标准 | P95 延迟、吞吐、错误率、资源水位 | 无法判定是否达标 |
| 基线版本 | 与哪一个历史版本比较 | 回归判断失效 |
| 数据规模 | 表大小、缓存热度、索引状态、样本分布 | 小数据快，大数据慢 |

---

## 核心机制与推导

性能测试常见类型有四类：

| 类型 | 定义 | 主要问题 |
| --- | --- | --- |
| 负载测试 | 在预期业务负载下运行 | 正常业务量下是否稳定 |
| 压力测试 | 持续加压直到失败或明显退化 | 极限容量在哪里 |
| 稳定性测试 | 中高负载长时间运行 | 是否有内存泄漏、连接泄漏、资源漂移 |
| 并发测试 | 多用户同时访问关键资源 | 锁竞争、队列堆积、热点冲突是否严重 |

判断瓶颈时，不要只盯一个指标。原因很简单：延迟、吞吐、错误率、资源利用率是联动的。

核心观察关系可以概括为：

1. 吞吐上升、延迟稳定、错误率低，说明系统仍有余量。
2. 吞吐继续上升，但 P95/P99 延迟明显拉长，说明排队开始出现。
3. 吞吐不再增长，而 CPU、内存、数据库连接、磁盘 IO 或下游 QPS 接近上限，说明进入饱和区。
4. 吞吐下降且错误率上升，说明系统已过载。
5. 平均延迟看起来正常，但 P99 急剧恶化，说明少数慢请求已先出问题，常见原因是锁等待、连接池耗尽、热点 key、慢 SQL、GC 抖动。

吞吐量的基本计算公式是：

$$
X = \frac{\text{总完成请求数}}{\text{测试持续时间}}
$$

例如 60 秒内成功完成 1200 个请求，则：

$$
X = \frac{1200}{60} = 20 \text{ r/s}
$$

错误率通常写成：

$$
error\_rate = \frac{\text{失败请求数}}{\text{总请求数}} \times 100\%
$$

例如总请求 1200 个，其中失败 3 个，则：

$$
error\_rate = \frac{3}{1200} \times 100\% = 0.25\%
$$

如果此时平均响应时间 $R = 150\text{ ms} = 0.15\text{ s}$，那么根据 Little’s Law：

$$
N \approx 20 \times 0.15 = 3
$$

这表示系统平均只需要处理大约 3 个在途请求，就能维持 20 r/s 的吞吐。这个例子非常小，但它能帮助新手建立直觉：吞吐并不只由“并发配置”决定，还取决于每个请求在系统里停留多久。

再看一个更接近工程的例子。某接口在 100 r/s 时：

| 指标 | 数值 |
| --- | ---: |
| 平均延迟 | 80 ms |
| P95 延迟 | 130 ms |
| P99 延迟 | 190 ms |
| CPU | 45% |
| 错误率 | 0% |

提升到 200 r/s 后：

| 指标 | 数值 |
| --- | ---: |
| 平均延迟 | 140 ms |
| P95 延迟 | 480 ms |
| P99 延迟 | 930 ms |
| CPU | 78% |
| 错误率 | 0.2% |

这里最值得警惕的不是平均延迟翻倍，而是 P95、P99 长尾明显拉大。长尾延迟的意思是少数慢请求，它通常比平均值更早暴露锁竞争、连接池耗尽、缓存穿透、磁盘抖动等问题。对用户而言，决定“卡不卡”的往往不是平均值，而是这部分慢请求。

为了让新手更容易读懂，可以把常见指标理解为下面四类：

| 指标 | 直白解释 | 为什么重要 |
| --- | --- | --- |
| 平均延迟 | 大多数请求的平均耗时 | 能描述总体趋势，但容易掩盖慢请求 |
| P95/P99 延迟 | 最慢的 5% 或 1% 请求有多慢 | 能最早暴露排队与竞争 |
| 吞吐量 | 单位时间完成多少请求 | 反映系统产出能力 |
| 错误率 | 请求有多少比例失败 | 决定系统是否仍可用 |

性能优化通常形成一个闭环：

1. 观察：收集延迟、吞吐、错误率、CPU、内存、IO、数据库等待事件、缓存命中率。
2. 定位：确认瓶颈在应用层、数据库层、缓存层、消息队列还是网络层。
3. 调整：修改代码、索引、缓存策略、线程池、连接池、限流策略、批处理方式。
4. 验证：重跑同一场景，与基线比较，确认收益和副作用。
5. 更新基线：把新的稳定结果记为后续回归基准。

这个闭环必须滚动执行。因为性能不是“优化一次就结束”的属性，而是随着代码、依赖、数据规模和业务路径持续变化的。

---

## 代码实现

下面给出两个代码例子。第一个是可运行的 Python 基准分析脚本，用来演示吞吐、错误率、Little’s Law 和 SLA 判定。第二个是 Locust 脚本，用来模拟真实用户行为。

先看纯 Python 脚本。它不依赖第三方库，保存为 `perf_baseline_demo.py` 后可以直接运行。

```python
from dataclasses import dataclass


@dataclass
class PerfSample:
    name: str
    total_requests: int
    failed_requests: int
    duration_seconds: float
    avg_response_ms: float
    p95_response_ms: float
    cpu_percent: float


@dataclass
class SLOTarget:
    max_avg_ms: float
    max_p95_ms: float
    max_error_rate: float
    min_throughput_rps: float
    max_cpu_percent: float


def throughput_rps(sample: PerfSample) -> float:
    return sample.total_requests / sample.duration_seconds


def error_rate(sample: PerfSample) -> float:
    if sample.total_requests == 0:
        return 0.0
    return sample.failed_requests / sample.total_requests


def estimated_concurrency(sample: PerfSample) -> float:
    return throughput_rps(sample) * (sample.avg_response_ms / 1000.0)


def slo_report(sample: PerfSample, target: SLOTarget) -> dict[str, bool]:
    return {
        "avg_latency_ok": sample.avg_response_ms <= target.max_avg_ms,
        "p95_ok": sample.p95_response_ms <= target.max_p95_ms,
        "error_rate_ok": error_rate(sample) <= target.max_error_rate,
        "throughput_ok": throughput_rps(sample) >= target.min_throughput_rps,
        "cpu_ok": sample.cpu_percent <= target.max_cpu_percent,
    }


def compare_to_baseline(current: PerfSample, baseline: PerfSample) -> dict[str, float]:
    current_rps = throughput_rps(current)
    baseline_rps = throughput_rps(baseline)

    return {
        "avg_latency_delta_ms": current.avg_response_ms - baseline.avg_response_ms,
        "p95_delta_ms": current.p95_response_ms - baseline.p95_response_ms,
        "throughput_delta_rps": current_rps - baseline_rps,
        "cpu_delta_percent": current.cpu_percent - baseline.cpu_percent,
        "error_rate_delta": error_rate(current) - error_rate(baseline),
    }


def print_summary(current: PerfSample, baseline: PerfSample, target: SLOTarget) -> None:
    report = slo_report(current, target)
    delta = compare_to_baseline(current, baseline)

    print(f"scenario={current.name}")
    print(f"throughput_rps={throughput_rps(current):.2f}")
    print(f"error_rate={error_rate(current) * 100:.3f}%")
    print(f"estimated_concurrency={estimated_concurrency(current):.2f}")
    print("slo_report=", report)
    print("baseline_delta=", delta)

    all_ok = all(report.values())
    no_regression = (
        delta["avg_latency_delta_ms"] <= 0
        and delta["p95_delta_ms"] <= 0
        and delta["throughput_delta_rps"] >= 0
        and delta["error_rate_delta"] <= 0
    )

    print(f"slo_pass={all_ok}")
    print(f"no_regression_vs_baseline={no_regression}")


if __name__ == "__main__":
    baseline = PerfSample(
        name="login-home-recommend",
        total_requests=27000,
        failed_requests=24,
        duration_seconds=600,
        avg_response_ms=210,
        p95_response_ms=340,
        cpu_percent=70,
    )

    current = PerfSample(
        name="login-home-recommend",
        total_requests=30000,
        failed_requests=15,
        duration_seconds=600,
        avg_response_ms=200,
        p95_response_ms=320,
        cpu_percent=68,
    )

    target = SLOTarget(
        max_avg_ms=250,
        max_p95_ms=400,
        max_error_rate=0.001,   # 0.1%
        min_throughput_rps=40,
        max_cpu_percent=80,
    )

    assert round(throughput_rps(current), 2) == 50.00
    assert round(error_rate(current) * 100, 3) == 0.050
    assert round(estimated_concurrency(current), 2) == 10.00

    print_summary(current, baseline, target)
```

如果运行这段脚本，得到的关键结论是：

| 输出项 | 含义 |
| --- | --- |
| `throughput_rps` | 当前版本每秒实际完成多少请求 |
| `error_rate` | 当前版本失败比例是否低于 SLA |
| `estimated_concurrency` | 根据吞吐和平均延迟估算的在途请求数 |
| `slo_pass` | 当前版本是否满足既定服务目标 |
| `no_regression_vs_baseline` | 与历史稳定版本相比是否退化 |

这段代码的用途不是替代压测工具，而是把公式落成可以验证的数字，避免“只记概念，不会算”。

下面是一个最小 Locust 示例，用于模拟登录和首页访问。保存为 `locustfile.py`，安装 `locust` 后即可运行。

```python
from locust import HttpUser, task, between


class BlogUser(HttpUser):
    host = "http://127.0.0.1:8000"
    wait_time = between(1, 3)

    def on_start(self):
        response = self.client.post(
            "/api/login",
            json={"username": "demo", "password": "demo123"},
            name="login",
        )
        if response.status_code != 200:
            raise RuntimeError(f"login failed: {response.status_code}")

    @task(3)
    def open_homepage(self):
        with self.client.get("/", name="homepage", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"unexpected status: {response.status_code}")
            elif "博客" not in response.text:
                response.failure("homepage content mismatch")
            else:
                response.success()

    @task(1)
    def load_recommendations(self):
        with self.client.get("/api/recommend", name="recommend", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"recommend failed: {response.status_code}")
                return

            try:
                data = response.json()
            except Exception:
                response.failure("recommend returned invalid json")
                return

            if "items" not in data:
                response.failure("recommend payload missing items")
            else:
                response.success()
```

最小运行方式如下：

```bash
python3 -m pip install locust
locust -f locustfile.py --headless --users 100 --spawn-rate 10 --run-time 10m
```

这个脚本包含几个初学者最容易忽略的点：

| 点 | 作用 | 为什么重要 |
| --- | --- | --- |
| `on_start` | 每个虚拟用户先登录 | 避免未登录流量污染结果 |
| `@task(3)` 和 `@task(1)` | 控制行为比例 | 更接近真实用户路径 |
| `catch_response=True` | 自定义成功失败判定 | `200` 不等于业务成功 |
| `host` | 固定目标地址 | 便于在实验室与预生产切换 |
| `response.json()` 校验 | 断言响应内容结构 | 防止接口返回空壳成功 |

如何在不同阶段复用它：

| 阶段 | 建议跑法 | 目标 |
| --- | --- | --- |
| 实验室环境 | 20 到 50 用户，5 到 10 分钟 | 校验脚本、数据、监控口径 |
| 预生产环境 | 200、500 或更高用户，做阶梯升压 | 找饱和点，做回归比较 |
| 稳定性测试 | 中高负载，持续 1 到 4 小时 | 发现内存泄漏、连接泄漏、资源漂移 |
| 生产小流量探针 | 极低比例、短时间 | 做 smoke test，确认真实环境无明显回归 |

这里的 smoke test 可以理解为最小可用验证，只回答“基本链路是否仍然健康”，不回答“极限容量到底是多少”。

一个更完整的工程例子：

某团队上线新的推荐接口前，在预生产用 Locust 模拟“登录 -> 首页 -> 推荐流”。实验室中 300 用户没有问题，但预生产 500 用户时 P99 延迟突然拉高。继续排查发现不是 Python 应用进程 CPU 高，而是数据库连接池只有 50，且推荐 SQL 缺少联合索引。修正索引并扩大连接池后，吞吐恢复，错误率下降。这个例子说明，性能测试的价值不在“跑出高并发”，而在“把瓶颈定位到可修改的工程对象”。

---

## 工程权衡与常见坑

性能测试最常见的问题，不是工具不会用，而是工程口径不一致。

| 常见坑 | 具体表现 | 规避方法 |
| --- | --- | --- |
| 环境不一致 | 本地压测结论与生产完全不同 | 尽量对齐 CPU、内存、网络、数据库、缓存配置 |
| 只看平均值 | 平均延迟不错，但用户仍感觉卡顿 | 同时看 P95、P99 和错误率 |
| 基线不更新 | 拿一年前的数据做对比 | 每个稳定版本都更新滚动基线 |
| 负载模型失真 | 只压首页，不压登录和写接口 | 按真实流量比例建模 |
| 预热不足 | 刚启动就开始统计 | 预热缓存、连接池、JIT、热点数据后再计数 |
| 忽略数据规模 | 小数据下很快，大数据下退化严重 | 使用接近生产的数据体量 |
| 只测应用不测依赖 | 应用没问题，数据库先崩 | 把数据库、缓存、消息队列一并纳入观测 |
| 压测机太弱 | 客户端自己先打满 CPU 或网络 | 单独监控压测机并做分布式压测 |
| 断言太松 | 接口返回空数据也被记为成功 | 把业务正确性写进脚本 |
| 监控粒度太粗 | 1 分钟平均值掩盖尖峰 | 提高采样频率并保留明细日志 |

一个典型误判是：在本地 2 核 4G 机器上压测，结果吞吐很低，于是得出“系统不达标”的结论；但生产实际是 8 核 16G。这个结论的错误在于，你测到的是“本地机器上限”，不是“生产系统能力”。

另一个常见坑是把“历史回归”误当成“优化收益”。比如当前版本吞吐量从 40 r/s 提升到 48 r/s，看起来进步很大；但如果历史稳定版本本来就是 50 r/s，那么这次其实仍然退化了。基线必须是滚动维护的，而不是随手找一个旧版本作为参照。

还有一个新手容易忽略的问题是测试过程本身也会引入噪声：

- 压测机性能不足，导致客户端发不出足够流量。
- 日志级别过高，把 IO 放大成瓶颈。
- 监控采集频率过低，看不到瞬时尖峰。
- CDN、缓存、读写分离、连接池参数在测试与生产不一致。
- 垃圾回收、JIT 预热、文件缓存还没稳定就开始统计。
- 测试数据过于理想，没有热点 key、脏数据、长列表、冷缓存。

工程上更可靠的做法是分阶段验证：

1. 实验室先确认功能链路、统计口径、断言逻辑没问题。
2. 预生产做主要决策，包括容量、长尾延迟、资源饱和点。
3. 生产只做小流量验证，不把未知风险直接带到全量流量。

如果要把经验收敛成可执行规则，可以记下面这张表：

| 现象 | 更可能的原因 | 先查什么 |
| --- | --- | --- |
| 吞吐不升，CPU 很高 | 应用层计算、锁竞争、GC | 火焰图、线程栈、GC 日志 |
| 吞吐不升，CPU 不高，延迟变长 | 排队、连接池、下游阻塞 | 连接池、线程池、队列长度 |
| 错误率突然上升 | 超时、依赖失败、限流触发 | 错误码分布、下游日志、网关告警 |
| 平均正常但 P99 很差 | 少数请求慢、热点冲突 | 慢 SQL、热点 key、锁等待 |
| 应用指标正常但用户仍慢 | 网络、CDN、前端资源、浏览器渲染 | RUM、CDN 命中率、前端 waterfall |

---

## 替代方案与适用边界

工具不是越多越好，而是要与问题匹配。Locust、JMeter、Gatling 都能做性能测试，但适用边界不同。

| 方案 | 适用场景 | 优点 | 边界 |
| --- | --- | --- | --- |
| Locust | 需要自定义复杂用户行为 | Python 脚本灵活，易表达状态流转 | 团队需要基本 Python 能力 |
| JMeter | HTTP 接口、协议覆盖广、传统企业场景 | 生态成熟，图形化配置丰富 | 场景复杂时配置维护成本高 |
| Gatling | 需要高性能压测引擎和清晰报告 | 报告与指标结构清楚，适合回归比对 | 需要熟悉其 DSL |
| wrk / wrk2 | 快速测单接口吞吐与延迟 | 启动快，适合接口基准 | 用户行为建模能力弱 |
| k6 | API 测试与 CI 集成 | 脚本化较清晰，适合自动化回归 | 复杂会话建模不如 Locust 灵活 |
| 生产小流量探针 | 上线后低风险验证 | 最接近真实用户 | 不能替代完整容量评估 |
| 仅做监控观察 | 业务较稳、变更很小 | 成本低 | 无法提前暴露容量上限 |

如果暂时无法完整仿真高并发，可以先采用“轻量方案 + 小流量验证”的组合。例如先用 Locust 或 k6 模拟 100 个用户的核心路径，确认没有明显退化，再在生产环境用 1% 灰度流量对照真实指标。这种做法不能回答“系统极限容量是多少”，但可以回答“这次改动是否引入明显风险”。

适用边界可以这样理解：

- 如果你要测复杂登录态、购物车、推荐流，Locust 更直接。
- 如果你要快速搭很多 HTTP 接口基准，JMeter 更省时间。
- 如果你要持续做版本间回归对比，Gatling 或 k6 的报告与自动化集成通常更顺手。
- 如果你只想知道某个单接口在固定参数下的极限吞吐，`wrk` 这类工具更高效。
- 如果你连测试环境都不稳定，那么先把监控、日志、追踪系统补齐，比盲目压测更重要。

不要把生产小流量探针当成主压测工具。它的作用是验证“真实环境下没有回归”，不是探索极限容量。极限探索仍然应该放在可控的预生产环境完成。

---

## 参考资料

| 来源 | 关键词 | 用途 |
| --- | --- | --- |
| Neil J. Gunther, *Guerrilla Capacity Planning* | Little’s Law、容量规划、排队直觉 | 用于建立并发、吞吐、响应时间之间的基础关系 |
| Raj Jain, *The Art of Computer Systems Performance Analysis* | 基准设计、统计口径、实验方法 | 用于理解如何设计可复现的性能实验 |
| [Locust 官方文档](https://docs.locust.io/) | `HttpUser`、任务建模、分布式压测 | 用于编写真实用户行为脚本 |
| [Apache JMeter 官方文档](https://jmeter.apache.org/usermanual/index.html) | 吞吐量、组件参考、报告口径 | 用于理解指标定义和测试组件 |
| [Gatling 官方文档](https://docs.gatling.io/) | 场景 DSL、报告、回归对比 | 用于持续做性能回归 |
| [wrk2 项目说明](https://github.com/giltene/wrk2) | 固定速率压测、尾延迟 | 用于理解开环负载与长尾观测 |
| [Google SRE Book](https://sre.google/sre-book/table-of-contents/) | SLA、SLO、错误预算、生产验证 | 用于把性能测试与上线决策连接起来 |
| [USE Method](http://www.brendangregg.com/usemethod.html) | Utilization、Saturation、Errors | 用于建立资源观测框架 |
| [Martin Fowler: Canary Release](https://martinfowler.com/bliki/CanaryRelease.html) | 灰度发布、小流量验证 | 用于说明生产探针的工程边界 |
