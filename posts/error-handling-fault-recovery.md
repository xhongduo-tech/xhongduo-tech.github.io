## 核心结论

高可用服务不是“把服务多部署几台”这么简单，而是把故障处理拆成一条可执行链路：先做错误分类，再决定是否重试、多久超时、何时降级，最后用熔断、自动重启、故障转移和数据恢复把系统拉回可用状态。这里的“熔断”可以先理解为：当下游已经明显不稳定时，主动停止继续调用，避免把故障放大。

对大模型服务尤其如此。一次请求可能失败在四个层面：模型推理本身、网络链路、算力资源、业务参数。只有把“短暂故障”和“永久错误”分开，系统才能既不误伤正常流量，也不在错误请求上浪费机器。工程上常见的目标是把这条链路和多副本、多可用区、多区域、负载均衡、健康检查、监控一起工作，才有机会接近 $99.99\%$ 级别可用性。

最重要的判断标准不是“有没有报错”，而是“这个错值不值得再试一次”。如果错误是 `Timeout`、连接中断、瞬时 5xx，通常先限时再重试；如果错误是 400、401、404 这类明确业务问题，应立即失败并返回清晰提示。重试不是默认正确动作，错误分类才是。

| 核心能力 | 作用 | 典型实现 |
| --- | --- | --- |
| 错误分类 | 区分可重试与永久错误 | HTTP 5xx/超时 可重试；4xx 多数不可重试 |
| 超时控制 | 限制单次与整体请求时长 | 单次调用超时 + 外层总超时 |
| 指数退避 | 避免重试风暴 | 重试间隔按 1s、2s、4s 增长并加入抖动 |
| 降级方案 | 主路径失败时维持业务连续性 | 返回缓存、切轻量模型、走排队 |
| 熔断机制 | 防止故障扩散 | 连续失败后短时间拒绝新请求 |
| 多区域容灾 | 区域故障时切流 | NLB/GSLB 摘除异常 VIP，DNS 收敛到备区 |
| 健康检查与监控 | 发现故障并触发动作 | `/health` 探针、成功率、延迟、CPU/GPU 监控 |

---

## 问题定义与边界

本文讨论的是在线服务可用性，不讨论模型训练阶段的容错，也不讨论数据库事务恢复的全部细节。范围聚焦在“请求进入服务之后，如何判断错误、控制重试、切换流量、恢复服务”。

先定义四类常见错误。

“推理错误”指模型在执行时失败，白话说就是模型服务收到了请求，但内部没能把结果算出来，例如进程崩溃、CUDA OOM、推理超时、模型实例异常退出。

“网络错误”指请求在链路上丢失或等待过久，白话说就是服务本身也许没坏，但包没有按时到达，例如 DNS 解析异常、连接被重置、网关超时。

“资源错误”指 CPU、内存、显存、线程池、连接池等资源耗尽，白话说就是机器还活着，但已经没有足够容量继续接单。

“业务逻辑错误”指请求内容本身不合法，白话说就是系统按设计拒绝这个请求，例如参数缺失、鉴权失败、引用了不存在的资源。

新手最容易混淆的是：不是所有失败都应该重试。一个简单的玩具例子是交通信号灯。红灯代表 401、403、404、参数错误，这类错误再试十次也不会变绿；黄灯代表网络抖动或上游短暂超时，稍后再试可能恢复；真正需要系统自动处理的是黄灯，不是红灯。

边界要靠判据明确。常用判据包括：

| 故障类型 | 诊断特征 | 处理策略 |
| --- | --- | --- |
| 模型推理失败 | HTTP 5xx、进程异常、CUDA OOM | 限时 + 重试 + 切备用模型 |
| 网络抖动 | `Timeout`、`ConnectionError` | 指数退避重试 + 熔断保护 |
| 资源耗尽 | CPU/GPU 利用率过高、队列积压 | 限流 + 扩容 + 降级 |
| 业务参数错误 | 400/401/404 | 立即失败 + 返回提示 |
| 区域级故障 | 健康探针连续失败 | 负载均衡摘除区域，流量切备区 |

这些判据决定后续动作。比如“连续 3 次健康探针失败”可以被定义为实例不可用；“95 分位延迟持续 5 分钟超过阈值”可以被定义为性能退化；“同一错误码连续超过阈值”可以触发熔断。没有这些边界，系统就会陷入两种极端：要么过早切流，误杀健康实例；要么切得太晚，让故障实例继续接收流量。

---

## 核心机制与推导

高可用链路的核心不是某个单点技巧，而是几个机制的配合。

第一是超时。超时就是“等多久算失败”，白话说就是不给一次请求无限等待的权利。超时必须至少分两层：单次调用超时，限制一次 RPC 或 HTTP 调用；整体请求超时，限制一次用户请求从开始到结束的总预算。只设单次超时不设整体超时，会让多次重试叠加成“连环等待”。

第二是指数退避。指数退避就是“失败越多，下一次等待越久”，白话说是为了避免所有客户端同时立刻重试，把已经不稳定的服务彻底压垮。常见公式是：

$$
wait\_time = \min(max\_backoff,\; base\_delay \times 2^{attempt-1} + jitter)
$$

其中：

| 参数 | 含义 | 效果 |
| --- | --- | --- |
| `base_delay` | 初始等待时间 | 控制第一次重试节奏 |
| `attempt` | 第几次尝试，从 1 开始 | 决定指数增长幅度 |
| `max_backoff` | 最大等待上限 | 防止等待时间无限增大 |
| `jitter` | 随机抖动 | 打散同一时间的重试洪峰 |

假设 `base_delay=1s`，最大 3 次重试，不考虑上限时等待时间是 $1s, 2s, 4s$。如果再乘上 `0.5 \sim 1.5` 的随机抖动，不同实例不会在同一时刻同时发起下一轮请求，这能明显降低重试风暴风险。

第三是熔断。熔断的逻辑是：如果下游在一个窗口内持续失败，就先短暂“断开”，不再把流量送过去，等冷却时间结束后再试少量探测请求。它解决的是“明知道下游坏了，还继续打过去”的问题。熔断不是替代重试，而是当重试已经无意义时的保护层。

第四是降级。降级就是“接受结果变差，但先保住服务可用”。例如主模型失败时切换到轻量模型、直接返回缓存摘要、只返回基础字段、或者把请求转入异步队列。降级的目标不是保持所有功能，而是保住关键功能。

第五是自动容灾。一个真实工程例子是：大模型网关在华东和华北各部署一套推理集群，前面接全球负载均衡。主区域健康探针连续失败后，控制面把该区域 VIP 从负载池中摘除，DNS 只返回备用区域地址，新的请求在秒级收敛到备区。用户看到的效果不是“系统挂了”，而是个别请求变慢、少量请求失败后迅速恢复。这就是多区域故障转移的意义。

监控是这条链路的闭环。只看成功率不够，还要同时看延迟、CPU、内存、显存、连接数、线程池排队、区域健康状态。因为“成功率下降”只说明表象，不说明原因。成功率下降且 GPU OOM 上升，意味着资源问题；成功率下降且网络超时增加，意味着链路问题；成功率正常但 P99 延迟激增，意味着系统正在接近失效边缘。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它演示四件事：错误分类、整体超时预算、指数退避、降级返回。代码不是完整生产实现，但结构已经接近真实服务。

```python
import random
from dataclasses import dataclass


class RetryableError(Exception):
    pass


class PermanentError(Exception):
    pass


@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 1.0
    max_backoff: float = 4.0
    total_timeout: float = 10.0


def compute_backoff(attempt: int, base_delay: float, max_backoff: float, jitter_factor: float) -> float:
    delay = min(max_backoff, base_delay * (2 ** (attempt - 1)))
    return delay * jitter_factor


def is_retryable_error(exc: Exception) -> bool:
    return isinstance(exc, RetryableError)


def call_with_retry(fn, policy: RetryPolicy, jitter_values=None):
    elapsed = 0.0
    waits = []
    jitter_values = jitter_values or [1.0] * policy.max_retries

    for attempt in range(1, policy.max_retries + 1):
        try:
            result = fn(attempt)
            return {
                "ok": True,
                "result": result,
                "attempt": attempt,
                "waits": waits,
                "elapsed": elapsed,
            }
        except Exception as exc:
            if not is_retryable_error(exc):
                return {
                    "ok": False,
                    "result": "permanent_error",
                    "attempt": attempt,
                    "waits": waits,
                    "elapsed": elapsed,
                }

            if attempt >= policy.max_retries:
                return {
                    "ok": False,
                    "result": "fallback_cache",
                    "attempt": attempt,
                    "waits": waits,
                    "elapsed": elapsed,
                }

            jitter_factor = jitter_values[attempt - 1]
            wait = compute_backoff(attempt, policy.base_delay, policy.max_backoff, jitter_factor)

            if elapsed + wait > policy.total_timeout:
                return {
                    "ok": False,
                    "result": "timeout_fallback",
                    "attempt": attempt,
                    "waits": waits,
                    "elapsed": elapsed,
                }

            waits.append(wait)
            elapsed += wait

    raise AssertionError("unreachable")


# 玩具例子：前两次网络超时，第三次成功
def flaky_service(attempt: int):
    if attempt < 3:
        raise RetryableError("timeout")
    return "success"


# 永久错误：参数不合法，不能重试
def bad_request_service(attempt: int):
    raise PermanentError("400 bad request")


policy = RetryPolicy(max_retries=3, base_delay=1.0, max_backoff=4.0, total_timeout=10.0)

r1 = call_with_retry(flaky_service, policy, jitter_values=[1.0, 1.0, 1.0])
assert r1["ok"] is True
assert r1["result"] == "success"
assert r1["waits"] == [1.0, 2.0]

r2 = call_with_retry(bad_request_service, policy)
assert r2["ok"] is False
assert r2["result"] == "permanent_error"
assert r2["waits"] == []

print("all assertions passed")
```

这段代码有几个实现重点。

第一，`is_retryable_error` 先于重试循环中的等待逻辑执行。这样 400、401、404 这类永久错误会立刻失败，不会进入无意义重试。

第二，`total_timeout` 放在外层。它表示整个请求生命周期的预算，而不是单次尝试的预算。真实服务中通常还要在单次 HTTP/RPC 调用上再包一层更短的 timeout。

第三，失败后的结果不只有“抛异常”。代码里演示了两种降级结果：`fallback_cache` 和 `timeout_fallback`。真实系统里这可能对应读取缓存答案、切到小模型、进入异步队列，或者直接返回“系统繁忙，请稍后重试”。

如果把这个逻辑放进真实工程，结构通常是：

1. API 网关接收请求并完成鉴权、参数校验。
2. 调度层根据模型、租户、区域选择目标集群。
3. 调用层设置单次超时和整体超时。
4. 错误分类层判断是否可重试。
5. 重试层执行指数退避。
6. 连续失败后触发熔断、降级或跨区域切流。
7. 监控与告警记录成功率、延迟、资源和错误码。

---

## 工程权衡与常见坑

最常见的坑是把“失败”当成同一种东西。实际上，错误处理最大的成本不是实现逻辑，而是错误分类做错后的副作用。

| 坑 | 影响 | 规避 |
| --- | --- | --- |
| 误重试 400/401/404 | 放大无效流量，增加用户等待 | 先按状态码和异常类型分类 |
| 不设整体超时 | 多次重试后用户仍长时间卡住 | 外层加总超时预算 |
| 重试没有抖动 | 多实例同时重放，打爆下游 | 引入 `jitter` |
| 健康探针过紧 | 误判健康实例为故障 | 提高 `failureThreshold`，检查轻量化 |
| 健康探针过松 | 故障实例持续收流 | 缩短检查周期，结合业务指标 |
| 降级未预演 | 真故障时切换路径也失败 | 平时演练缓存、轻量模型、只读模式 |
| 只监控成功率 | 无法定位真实瓶颈 | 同时看延迟、资源、链路、错误码 |

健康检查尤其容易出错。一个常见误区是把 `/health` 写成“顺便连数据库、顺便查远程配置、顺便跑一次模型推理”。这会让探针本身变成慢请求源头。更稳妥的做法是把探针拆层：`liveness` 只判断进程是否存活；`readiness` 判断实例是否可以接流量；深度业务检查交给单独监控任务，而不是放进高频探针。

一个简单配置例子是：每 10 秒做一次 readiness 检查，超时 5 秒，连续 3 次失败才从负载池摘除。这个策略的含义不是“任何异常立刻切流”，而是承认系统存在短暂抖动，只有连续失败才定义为不可用。

另一个常见坑是把降级理解成“随便返回个默认值”。降级必须有业务边界。例如搜索服务可以返回缓存结果，客服机器人可以切轻量模型，支付服务则不能“随便降级”为忽略扣款失败。高可用不是让所有业务都永远成功，而是在正确的边界内优先保住核心路径。

---

## 替代方案与适用边界

不是所有服务都需要同等级别的高可用架构。低频后台任务和面向用户的在线请求，策略应该不同。

如果业务是低频、可补偿、对实时性不敏感，替代方案可以是“失败后入队重放”。这类方式实现更简单，成本更低，但不适合强交互接口。

如果业务对正确性要求高于实时性，可以把“同步返回结果”改成“同步接单，异步完成”。例如大模型长文本分析请求先写入任务队列，再由后端异步执行。这减少了前台超时压力，但用户体验从实时响应变成了任务式查询。

如果业务对可用性极高，例如在线问答、网关、实时客服，就需要多副本、多可用区、最好再加跨区域容灾。一个典型决策流程是：

1. 主路径失败。
2. 判断是否为永久错误。
3. 如果是永久错误，立即返回明确失败。
4. 如果是短暂错误，在总超时预算内做指数退避重试。
5. 如果局部实例持续失败，熔断该实例或该下游。
6. 如果主区域探针连续失败，负载均衡摘除主区，DNS 收敛到备区。
7. 如果备区压力升高或能力不足，启用降级策略。
8. 主区恢复后，先小流量探测，再逐步回切。

这里也有适用边界。跨区域多活能提升可用性，但会带来更高成本、更复杂的数据一致性问题，以及更难调试的链路。对很多中小系统来说，先把单区域多副本、限流、重试、熔断、监控做好，收益往往比直接上全球多活更高。

结论很直接：高可用不是“堆功能”，而是按故障层次设计动作。能局部解决的，不要上升到区域切换；能立即判定为永久错误的，不要进入重试；能通过降级保住主流程的，不要把所有功能绑在同一条链路上。

---

## 参考资料

- [火山引擎《AI 大模型调用全流程》](https://developer.volcengine.com/articles/7533207360601653286)
- [PHP中文网《Python API 请求失败重试策略》](https://www.php.cn/faq/2186037.html)
- [博客园《指数退避策略》](https://www.cnblogs.com/wsx2019/p/19202772)
- [阿里云《NLB产品高可用能力介绍》](https://help.aliyun.com/zh/slb/network-load-balancer/product-overview/nlb-product-high-availability-capability-introduction)
- [CSDN《高可用推理平台架构实战》](https://blog.csdn.net/sinat_28461591/article/details/147590713)
- [CSDN《健康检查间隔数据驱动》](https://blog.csdn.net/PixelShoal/article/details/154734045)
