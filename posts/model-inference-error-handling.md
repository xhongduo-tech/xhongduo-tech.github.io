## 核心结论

模型推理错误处理的核心不是“失败后自动再打一遍”，而是先把错误分型，再决定是重试、熔断、降级还是摘流。这里的“分型”可以理解为：先判断错误属于哪一类，以及它对系统会造成什么影响。

同样是一次调用失败，`400` 和 `503` 的处理方式通常完全不同。`400 Bad Request` 表示请求本身不合法，继续重试大概率还是错；`503 Service Unavailable` 表示服务暂时不可用，短时间限次重试可能恢复。判断标准不是错误名字是否“严重”，而是三个问题：

1. 这个错误是否可重试。
2. 这个错误是否会扩散到更多实例或更多下游。
3. 继续尝试是否还在延迟与成本预算内。

下面这张总表可以先记住。它不是唯一答案，但足够覆盖大多数在线推理链路。

| 错误类型 | 常见含义 | 是否重试 | 是否熔断 | 是否降级 | 是否摘流 |
| --- | --- | --- | --- | --- | --- |
| `400` | 参数错误、请求格式错 | 否 | 否 | 否 | 否 |
| `401` | 未认证 | 否 | 否 | 否 | 否 |
| `403` | 无权限 | 否 | 否 | 否 | 否 |
| `404` | 模型路由或资源不存在 | 否 | 否 | 可选 | 可选 |
| `429` | 被限流 | 是，短暂限次 | 可选 | 是 | 否 |
| `500` | 服务内部错误 | 视实现而定，少量 | 可选 | 是 | 可选 |
| `503` | 服务不可用或过载 | 是，短暂限次 | 是 | 是 | 可选 |
| `timeout` | 超时 | 是，但必须看剩余预算 | 是 | 是 | 可选 |
| `GPU OOM` | 显存不足，实例已不健康 | 否，别打同一实例 | 否 | 是 | 是 |
| `模型未加载` | 实例已启动但模型还没准备好 | 否，别打同一实例 | 否 | 是 | 是 |

一句话总结：错误处理是资源分配问题，不是语法糖。你不是在“修复一次失败”，而是在控制失败不要把整个推理系统拖垮。

---

## 问题定义与边界

本文讨论的是**在线推理链路**中的错误处理。在线推理就是：请求实时进来，系统要在较短时间内返回结果，比如聊天接口、向量检索重排、图像分类、广告排序、语音转写。这和训练阶段、离线批处理、模型调参不是同一个问题。

这里的“错误”至少要拆成四类：

| 错误来源 | 典型现象 | 是否通常可重试 | 是否需要降级 | 是否需要摘流 |
| --- | --- | --- | --- | --- |
| 请求侧 | 参数缺失、类型错误、token 超长 | 否 | 否 | 否 |
| 模型侧 | 模型未加载、版本不兼容、输出校验失败 | 视情况 | 是 | 常常需要 |
| 运行时侧 | 进程卡死、线程池耗尽、超时、GPU OOM | 部分可重试 | 是 | 常常需要 |
| 基础设施侧 | 网络抖动、DNS 问题、实例过载、LB 路由异常 | 部分可重试 | 是 | 视情况 |

新手常把问题理解成“接口没有成功返回”。但工程上，“没返回成功”只是表面现象，底层原因可能完全不同：

1. 参数错了，属于请求侧错误。
2. 依赖超时了，属于运行时或基础设施错误。
3. GPU 显存爆了，属于运行时资源错误。
4. 实例虽然端口通，但模型根本没加载完，属于模型侧可用性错误。

这四种情况如果统一用“重试三次”处理，系统会出现两类坏结果：第一类是无效重试，延迟白白增加；第二类是放大故障，把原本局部错误打成全局过载。

本文不讨论以下范围：

| 不讨论范围 | 原因 |
| --- | --- |
| 训练任务失败重跑 | 它关注吞吐和作业完成率，不是低延迟在线返回 |
| 离线批推理失败补数 | 它可以做更长时间窗口的补偿，不受在线预算约束 |
| 模型效果错误 | 比如答案不准、幻觉、召回差，这属于质量治理，不是基础错误处理 |
| 调试阶段的开发异常 | 比如本地环境缺包、脚本写错，这不是生产链路策略 |

边界要清楚，因为在线推理错误处理的关键约束是两个词：**时延预算**和**连锁影响**。前者决定你还能试几次，后者决定你该不该继续把流量送进一个可能已经坏掉的地方。

---

## 核心机制与推导

先定义统一符号。

- $E$：一次失败事件，也就是一次调用没有成功返回。
- $r(E)$：对错误 $E$ 是否允许继续重试。
- $B$：总预算，也就是整条请求链允许消耗的最大时间。
- $t_i$：第 $i$ 次调用本身花掉的时间。
- $d_i$：第 $i$ 次失败后、下一次重试前的等待时间。
- $d_0$：初始退避时间。
- $d_{max}$：最大退避上限。
- $j_i$：jitter，中文常说“抖动因子”，作用是把不同请求的重试时间打散，避免同时冲回去。

在线推理里，最常见的退避策略是指数退避加 jitter：

$$
d_n = \min(d_{max}, d_0 \cdot 2^{n-1}) \cdot j_n
$$

它的含义是：失败次数越多，下一次等待越久，但等待不能无限增大，要受 $d_{max}$ 限制；同时乘上一个随机因子 $j_n$，避免所有客户端在完全一样的时刻重试。

真正的决策条件不是“报错了就试”，而是：

$$
r(E) = 1 \land \sum (t_i + d_i) \le B
$$

这句话可以直接翻译成工程规则：**只有当错误可重试，并且累计执行时间加等待时间仍未超出总预算，才允许继续重试。**

玩具例子：

设初始等待 $d_0 = 200ms$，上限 $d_{max} = 800ms$，总预算 $B = 1s$。

假设每次调用本身耗时大约 $150ms$，并先忽略 jitter，按最简单的指数退避：

- 第 1 次失败后等待 $200ms$
- 第 2 次失败后等待 $400ms$
- 第 3 次失败后等待理论上 $800ms$

累计预算怎么变：

| 次数 | 调用耗时 | 等待时间 | 累计消耗 |
| --- | --- | --- | --- |
| 第 1 次失败后 | 150ms | 200ms | 350ms |
| 第 2 次失败后 | 150ms | 400ms | 900ms |
| 若再发第 3 次 | 至少再加 150ms | 还没算后续 | 已超过 1s |

所以第三次实际上不该继续。虽然从“次数”上看你还没达到某个最大重试数，但从“预算”上看已经没资格再试。在线系统里，预算约束通常比重试次数更重要。

再看决策表：

| 错误类别 | 可重试？ | 预算足够？ | 首选动作 |
| --- | --- | --- | --- |
| 不可重试 | 否 | 不重要 | 直接失败或返回明确错误 |
| 可重试 | 是 | 是 | 限次重试 |
| 可重试但预算不足 | 是 | 否 | 停止重试，走降级或失败 |
| 过载/不可用 | 视情况 | 常常不重要 | 熔断、摘流、降级，不应持续打同一实例 |

“熔断”可以理解为：观察到某个服务持续失败后，短时间内主动不再调用它。“摘流”可以理解为：把不健康实例从流量池里拿掉，不再分配新请求。

真实工程例子：

一个聊天服务前面挂了网关，后面有 8 个推理实例。某个实例进程还活着，但模型权重加载失败，健康检查只看端口，于是负载均衡仍把请求打过去。结果这个实例持续返回 `503` 或超时。若客户端、网关、服务端都各自重试一次，那么单次真实请求最多会放大成 $2 \times 2 \times 2 = 8$ 次尝试。高峰期下，这种“错误处理”本身就会制造过载。正确做法不是一层层加重试，而是尽快识别实例不可用，并摘流。

---

## 代码实现

实现重点只有两件事：**错误分类**和**策略执行**分离。前者回答“这是什么错”，后者回答“遇到这种错该做什么”。

下面给一个最小可运行的 Python 例子。它演示：

1. 错误分类函数。
2. 指数退避 + jitter。
3. 超时预算控制。
4. 失败后 fallback，也就是降级到备用模型或兜底响应。
5. 用 `assert` 验证关键行为。

```python
import random
from dataclasses import dataclass


@dataclass
class RetryPolicy:
    max_retries: int = 2
    base_delay_ms: int = 200
    max_delay_ms: int = 800
    total_budget_ms: int = 1000


def classify_error(err):
    code = err.get("code")
    if code in (400, 401, 403, 404):
        return "non_retryable"
    if code in (429, 500, 503, "timeout"):
        return "retryable"
    if code in ("gpu_oom", "model_not_ready"):
        return "drain_or_degrade"
    return "unknown"


def backoff_delay_ms(attempt, policy, jitter_low=0.8, jitter_high=1.2):
    raw = min(policy.max_delay_ms, policy.base_delay_ms * (2 ** (attempt - 1)))
    jitter = random.uniform(jitter_low, jitter_high)
    return int(raw * jitter)


def should_retry(category, attempt, spent_ms, next_delay_ms, call_timeout_ms, policy):
    if category != "retryable":
        return False
    if attempt > policy.max_retries:
        return False
    projected = spent_ms + next_delay_ms + call_timeout_ms
    return projected <= policy.total_budget_ms


def fallback_response():
    return {"source": "fallback", "answer": "system busy, degraded response"}


def infer_with_policy(simulated_errors, policy, call_timeout_ms=150):
    spent_ms = 0
    attempt = 0

    for err in simulated_errors:
        attempt += 1
        spent_ms += call_timeout_ms
        category = classify_error(err)

        if category == "drain_or_degrade":
            return fallback_response()

        delay_ms = backoff_delay_ms(attempt, policy)
        if should_retry(category, attempt, spent_ms, delay_ms, call_timeout_ms, policy):
            spent_ms += delay_ms
            continue

        if category == "retryable":
            return fallback_response()

        return {"source": "fail_fast", "error": err["code"]}

    return {"source": "primary", "answer": "ok"}


policy = RetryPolicy()

# 400 不应重试，应直接失败
result_400 = infer_with_policy([{"code": 400}], policy)
assert result_400["source"] == "fail_fast"

# 503 可重试，但预算不足时应停止并降级
random.seed(0)
result_503 = infer_with_policy([{"code": 503}, {"code": 503}, {"code": 503}], policy)
assert result_503["source"] in ("fallback", "primary")

# GPU OOM 不应继续打同一实例，应降级或摘流
result_oom = infer_with_policy([{"code": "gpu_oom"}], policy)
assert result_oom["source"] == "fallback"
```

这个例子故意保持简单，但结构是对的。真正工程里，通常会放在三层中的某一层：

| 层级 | 适合做什么 | 不适合做什么 |
| --- | --- | --- |
| 客户端 | 短暂重试、超时预算、幂等读请求的兜底 | 不了解实例健康，不适合做实例摘流 |
| 网关/中间层 | 统一策略、熔断、限流、实例级摘流 | 不适合处理业务级参数错误 |
| 服务端 | 最了解模型状态，可报告 OOM、未加载、队列拥塞 | 不应偷偷无限自重试，容易放大负载 |

一个更接近生产的配置可能长这样：

```yaml
retry:
  retryable_codes: [429, 500, 503, timeout]
  max_retries: 2
  base_delay_ms: 200
  max_delay_ms: 800
  total_budget_ms: 1000

degrade:
  enable_fallback_model: true
  enable_cached_response: true

health:
  readiness_requires_model_loaded: true
  startup_probe_enabled: true
```

注意 `readiness_requires_model_loaded: true` 很关键。`readiness` 可以理解为“这个实例是否准备好接流量”。如果它只检查端口通不通，而不检查模型是否已加载完成，那么系统会把流量送进一个“活着但不能服务”的实例。

---

## 工程权衡与常见坑

工程难点不在“判断一次是否重试”，而在多个组件一起做保护时，如何避免把局部故障放大成级联故障。级联故障就是：一个地方出问题，后续组件因为补救动作过猛，也跟着出问题。

最常见的事故是多层重复重试。假设客户端重试 2 次，网关重试 2 次，服务端内部再试 2 次。一次原始请求在最坏情况下会变成 $2 \times 2 \times 2 = 8$ 次尝试。如果失败原因本来就是过载，这 8 次尝试只会让过载更严重。

常见坑与规避如下：

| 常见坑 | 结果 | 规避方式 |
| --- | --- | --- |
| 所有错误都重试 | 参数错也被反复打，白白增加延迟 | 先分类，`4xx` 多数直接失败 |
| readiness 只看端口不看模型状态 | 模型没加载好也接流量 | readiness 必须包含模型就绪条件 |
| 没有 `startup probe` | 冷启动阶段被过早探活，反复重启或误接流量 | 长加载模型必须配 `startup probe` |
| 多层重复重试 | 请求数被指数放大，制造级联故障 | 规定“谁负责重试”，其余层尽量透明 |
| 重试不加 jitter | 所有请求同时回冲，形成重试风暴 | 指数退避必须带随机抖动 |
| 非幂等写操作被重复执行 | 订单、计费、日志写入重复 | 写请求默认不自动重试，或加幂等键 |

这里补一个真实工程例子。

某推理平台部署大模型，单实例冷启动需要 90 秒。团队只配了 liveness probe 和一个非常简单的 readiness probe：端口通就算成功。结果发布新版本时，Kubernetes 看到端口已起，就把请求送过去；但模型权重还在加载，实例持续返回超时。网关把这些超时当临时错误继续重试，最终把老实例也拖慢。这个故障的根因不是“模型慢”，而是健康检查语义错了：`startup probe`、`readiness probe` 和重试策略没有分工清楚。

实践上可以记一个原则：

1. 参数错，尽快失败。
2. 瞬时抖动，限次重试。
3. 实例过载，熔断或限流。
4. 实例不健康，摘流。
5. 主模型不可用，降级。

这五件事看起来都像“容错”，但它们解决的是不同故障模式，混着用就会出问题。

---

## 替代方案与适用边界

不是所有场景都适合“重试优先”。很多场景里，重试只是选项之一，甚至不是首选。

先看常见状态的边界：

- `429` 和 `503`：通常可以短暂重试，但要限次，并且必须受预算约束。
- `400`、`401`、`403`、`404`：通常不应重试，因为错误不在瞬时抖动，而在请求本身或资源配置。
- `GPU OOM`、模型加载失败：更适合摘流或降级，而不是继续打同一实例。
- 低延迟服务：比如 200ms 以内的检索重排，预算很小，重试空间也很小。
- 写请求或强一致请求：默认不应该自动重试，除非你有明确的幂等保证。

下面做一个方案对比。

| 方案 | 适用场景 | 优点 | 风险 | 失效边界 |
| --- | --- | --- | --- | --- |
| 客户端重试 | 幂等读请求、短时网络抖动 | 简单直接，离用户最近 | 不知道后端实例健康 | 多客户端同时重试会放大流量 |
| 网关重试 | 多服务统一治理 | 策略集中、便于观测 | 可能掩盖真实故障 | 若与客户端叠加会重复放大 |
| 服务端重试 | 可精确知道内部依赖状态 | 最了解业务和模型状态 | 容易变成黑盒重试 | 若外层也重试，风险更高 |
| 熔断 | 连续失败、过载明显 | 保护下游，阻止雪崩 | 误判会损失可用流量 | 依赖阈值配置是否合理 |
| 降级 | 主模型不可用但可接受次优结果 | 保持服务连续性 | 质量下降 | 有些核心场景不能接受降级 |
| 摘流 | 个别实例不健康 | 快速隔离坏节点 | 健康判定错误会误摘 | 若整体容量不足，摘流后更紧张 |
| 异步化 | 可接受延迟、不要求实时返回 | 把峰值摊平 | 产品形态改变 | 对强实时接口不适用 |
| 限流 | 系统接近容量上限 | 防止整体被打穿 | 会直接拒绝部分请求 | 若阈值太死，会牺牲可用性 |

可以看到，重试只是其中一项。很多场景下更好的答案是：

1. 用限流保护整体容量。
2. 用熔断隔离持续失败的下游。
3. 用摘流移除坏实例。
4. 用降级维持最小可用服务。

如果一定要用一句话概括适用边界，那就是：**重试适合处理“暂时失败”，不适合处理“必然失败”或“继续尝试会更糟”的情况。**

---

## 参考资料

1. [Kubernetes: Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)
2. [AWS Prescriptive Guidance: Retry with backoff pattern](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/retry-backoff.html)
3. [AWS Architecture Blog: Exponential Backoff And Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
4. [NVIDIA Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
5. [Envoy: Circuit Breaking](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/circuit_breaking)
6. [Envoy: Outlier Detection](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/outlier)
