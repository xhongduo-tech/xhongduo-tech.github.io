## 核心结论

灰度监控的目标，不是证明新版本“能启动”或“能返回结果”，而是用少量真实线上流量，判断它是否满足继续放量的条件。这里的“放量”就是把更多用户请求逐步切给新版本；“监控”就是在每个流量阶段，用统一窗口比较新旧版本的关键指标。

对推理服务来说，放量判断至少要同时覆盖三类指标：

| 维度 | 典型指标 | 为什么必须看 |
| --- | --- | --- |
| 可用性 | `5xx`、超时率、空响应率 | 模型再准，服务报错也不能上线 |
| 性能 | `p95`、`p99` 延迟 | 推理服务常见问题不是平均慢，而是尾部请求很慢 |
| 成本 | GPU 显存、单请求 token 成本、实例 CPU/GPU 利用率 | 新版如果更贵，可能业务上不成立 |

因此，灰度决策不该是“大家看着差不多”。它应该是一个明确判定：

$$
promote \iff n_c \ge n_{min} \land \Delta m \le \varepsilon \land h_c \le T
$$

其中：

- $n_c$ 是 canary 样本量，也就是灰度版本拿到的请求数
- $\Delta m$ 是相对 stable 的退化量，也就是新旧指标差值
- $h_c$ 是硬门限指标，比如 `5xx`、超时、OOM
- $T$ 是业务允许的上限

结论只有两类：继续放量，或立即回滚。所谓“再等等看”“先全量再观察”，本质上是在放弃灰度的安全边界。

一个最小判断表可以写成这样：

| 指标 | 门限 | 是否通过 | 动作 |
| --- | --- | --- | --- |
| `5xx` 错误率 | `<= 0.5%` | 否 | 立即回滚 |
| 超时率 | `<= 1%` | 是 | 继续看其他指标 |
| `p99` 延迟退化 | `<= +15%` | 否 | 停止放量 |
| GPU 显存峰值 | `<= 90%` | 是 | 可继续评估 |

新手可以先记住一句话：先给新模型 5% 流量，在一个固定时间窗里同时观察错误率、超时率和 `p95/p99`；只要任一硬门限越线，就不要继续放量。

---

## 问题定义与边界

推理服务灰度监控，是指在已有稳定版本兜底的前提下，把一小部分真实用户请求路由到新模型或新推理栈上，并用线上指标决定是否继续放量。这里的“兜底”就是旧版本仍然承接大多数流量，出问题时可以马上切回去。

它容易和几个概念混在一起，必须先分清：

| 方案 | 输入流量 | 是否直接返回用户 | 主要目的 | 是否适合做放量决策 |
| --- | --- | --- | --- | --- |
| 灰度监控 | 真实线上流量的一部分 | 是 | 判断新版本能否继续放量 | 是 |
| 压测 | 人工构造流量 | 否 | 测极限吞吐和容量 | 否 |
| 影子流量 `shadow` | 复制真实流量 | 否 | 看兼容性和行为差异 | 不能单独决定 |
| A/B 测试 | 真实线上流量分组 | 是 | 比较业务效果，如点击率、转化率 | 可以，但目标不同 |

边界也要明确：

1. 它适用于“旧版本稳定在线，新版本需要用真实流量验证”的场景。
2. 它不适合替代离线评估、单元测试、接口联调。
3. 它也不适合完全没有回滚手段的高风险变更，比如数据库不可逆 schema 改动。
4. 对极低流量业务，灰度样本可能长期不够，结论会失真。

玩具例子很简单。你有一个文本分类模型 `v1`，线上稳定运行；你把新模型 `v2` 接入同一个接口，只让 5% 用户请求打到 `v2`。如果 `v2` 的错误率更高、响应更慢，或者显存占用顶满导致 Pod 被驱逐，那么即使它在离线评测集上更准，也不能继续放量。

真实工程里，这种边界更明显。比如一个大模型推理网关后面挂着两套服务：`stable` 用 INT8 量化，`canary` 用 FP16 新权重。新权重的业务指标可能更好，但如果它把 `p99` 从 1.8 秒拉到 4.5 秒，用户侧体验会先坏掉。灰度监控要解决的是“线上系统还能不能承受”，不是“论文指标是不是更高”。

---

## 核心机制与推导

灰度监控可以拆成三层机制：流量切分、同窗观测、门限决策。

### 1. 流量切分

“流量切分”就是把一部分请求送到 canary，其他请求继续走 stable。假设总请求数是 $N$，灰度比例是 $p$，那么 canary 样本量是：

$$
n_c = pN
$$

这条式子很重要，因为很多误判都来自样本量不够。总请求 10,000 次，灰度 5%，canary 只有 500 次。如果你在 5 分钟内只拿到 47 个样本，就谈不上稳定判断。

### 2. 同窗观测

“同窗”就是新旧版本必须在同一个时间窗口里比较，否则结论不可信。比如 stable 是昨天晚高峰的数据，canary 是今天凌晨的数据，流量结构已经变了，不能直接比。

设某个指标在 canary 上的值为 $m_c$，stable 上的值为 $m_s$，那么退化量定义为：

$$
\Delta m = m_c - m_s
$$

如果这个指标是延迟，也常用相对退化：

$$
\Delta_{rel} = \frac{m_c - m_s}{m_s}
$$

例如 stable 的 `p99` 是 1200ms，canary 的 `p99` 是 1440ms，那么：

$$
\Delta_{rel} = \frac{1440 - 1200}{1200} = 20\%
$$

如果你的放量规则要求 `p99` 退化不超过 15%，那这一步就该终止。

### 3. 门限决策

灰度不应该只看“比 stable 差多少”，还要看硬门限。因为有些故障即使 stable 本身也一般，也不代表 canary 可以跟着坏。

所以最小决策式是：

$$
promote \iff n_c \ge n_{min} \land \Delta m \le \varepsilon \land h_c \le T
$$

可以把它读成一句白话：样本量够、退化没越线、硬故障没触发，才允许继续放量。

### 为什么不能只看平均值

平均值会掩盖尾部问题。推理服务尤其如此，因为少量超长请求就能拖垮用户体验和连接池。比如 100 个请求里，95 个是 200ms，5 个是 8 秒，平均值只有 590ms，看起来还行，但这 5 个超长请求已经足够让用户觉得服务坏了。

所以延迟通常看分位数。分位数就是把请求按延迟从小到大排序后，看某个位置的值；`p99` 的意思是 99% 请求不超过这个时间。

Prometheus 常见写法是：

```promql
histogram_quantile(
  0.99,
  sum by (le, version) (
    rate(http_request_duration_seconds_bucket{service="inference-gateway"}[5m])
  )
)
```

这里的 `histogram_quantile` 会从直方图桶估算分位数。直方图可以理解为“把请求按延迟区间计数”的数据结构，所以它能算出 `p95/p99`，而平均值只能反映总和除以总数。

### 一个玩具例子

总请求 $N = 10000$，灰度比例 $p = 5\%$，那么：

$$
n_c = pN = 500
$$

stable 的指标：

- `5xx = 0.2%`
- `p99 = 1000ms`

canary 的指标：

- `5xx = 1.0%`
- `p99 = 1180ms`

那么：

- 错误率退化：`+0.8 个百分点`
- 延迟退化：`+18%`

如果门限是：

- `5xx <= 0.5%`
- `p99` 退化 `<= 15%`

那 canary 同时触发两条失败条件，结论只能是回滚，而不是“先放到 20% 再看看”。

### 一个真实工程例子

假设你在做大模型问答服务升级，从 `stable` 的 TensorRT-LLM 推理栈切到新版运行时，目标是减少首 token 时间。但上线后出现两类风险：

1. 新 runtime 首次加载权重更慢，Pod 刚起来时延迟抖动明显。
2. 新 tokenizer 在少量输入上触发异常，错误率只在特定请求模式下升高。

这时灰度监控必须按 `version`、`model`、`route` 甚至 `tenant` 打标签，否则你可能只看到全局均值没变化，却漏掉了某个高价值租户的异常尖峰。换句话说，灰度不是“看一个总面板”，而是“在正确维度下看同一窗口的差异”。

---

## 代码实现

工程实现不是一条 PromQL，也不是一个发布按钮，而是三部分组合：流量路由、指标采集、决策控制。

最小闭环可以拆成四步：

1. 用网关或服务网格把 5% 流量切到 canary。
2. 在固定观测窗口内，分别采集 stable 和 canary 的指标。
3. 用规则判断是否满足继续放量条件。
4. 通过则推进到下一个权重，否则自动回滚。

一个典型链路可以表示为：

`入口网关 -> 路由到 stable/canary -> 推理服务 -> Prometheus 采集指标 -> 决策器 -> promote/rollback`

下面给一个最小化的 `Argo Rollouts` 灰度配置片段：

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: inference-service
spec:
  replicas: 10
  selector:
    matchLabels:
      app: inference-service
  template:
    metadata:
      labels:
        app: inference-service
    spec:
      containers:
        - name: app
          image: registry.example.com/inference:v2
          ports:
            - containerPort: 8080
  strategy:
    canary:
      stableService: inference-stable
      canaryService: inference-canary
      trafficRouting:
        istio:
          virtualService:
            name: inference-vs
            routes:
              - primary
      steps:
        - setWeight: 5
        - pause:
            duration: 10m
        - setWeight: 20
        - pause:
            duration: 10m
        - setWeight: 50
        - pause:
            duration: 10m
```

对应的 Istio 路由思想，是把一个服务拆成两个目标版本，再按权重分流：

```yaml
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: inference-vs
spec:
  hosts:
    - inference.example.com
  http:
    - name: primary
      route:
        - destination:
            host: inference-stable
          weight: 95
        - destination:
            host: inference-canary
          weight: 5
```

Prometheus 查询可以至少有三条：

```promql
sum(rate(http_requests_total{service="inference",version="canary",code=~"5.."}[5m]))
/
sum(rate(http_requests_total{service="inference",version="canary"}[5m]))
```

```promql
histogram_quantile(
  0.99,
  sum by (le) (
    rate(http_request_duration_seconds_bucket{service="inference",version="canary"}[5m])
  )
)
```

```promql
avg_over_time(gpu_memory_used_ratio{service="inference",version="canary"}[5m])
```

再给一个可运行的 Python 玩具决策器。它不依赖外部库，用最简单的规则模拟“继续放量或回滚”：

```python
from dataclasses import dataclass

@dataclass
class Metrics:
    samples: int
    err_rate: float      # 0.01 means 1%
    timeout_rate: float  # 0.01 means 1%
    p99_ms: int
    gpu_mem_ratio: float # 0.90 means 90%

def should_promote(
    stable: Metrics,
    canary: Metrics,
    min_samples: int = 300,
    max_err_rate: float = 0.005,
    max_timeout_rate: float = 0.01,
    max_p99_regression: float = 0.15,
    max_gpu_mem_ratio: float = 0.90,
) -> bool:
    if canary.samples < min_samples:
        return False
    if canary.err_rate > max_err_rate:
        return False
    if canary.timeout_rate > max_timeout_rate:
        return False
    if canary.gpu_mem_ratio > max_gpu_mem_ratio:
        return False

    p99_regression = (canary.p99_ms - stable.p99_ms) / stable.p99_ms
    if p99_regression > max_p99_regression:
        return False

    return True

stable = Metrics(samples=5000, err_rate=0.002, timeout_rate=0.004, p99_ms=1000, gpu_mem_ratio=0.72)
good_canary = Metrics(samples=500, err_rate=0.003, timeout_rate=0.006, p99_ms=1100, gpu_mem_ratio=0.81)
bad_canary = Metrics(samples=500, err_rate=0.010, timeout_rate=0.008, p99_ms=1180, gpu_mem_ratio=0.84)

assert should_promote(stable, good_canary) is True
assert should_promote(stable, bad_canary) is False
```

这段代码表达的不是“最佳统计方法”，而是最小工程闭环：有流量、有窗口、有阈值、有动作。对零基础读者，先把这四件事建立起来，比一开始就引入复杂统计检验更重要。

真实工程例子里，你通常还会再加两层：

1. 预热保护：canary 刚起时不计入决策窗口。
2. 自动分析：Prometheus 指标由发布控制器自动查询，不靠人盯盘手工点回滚。

---

## 工程权衡与常见坑

灰度监控不是“指标越多越安全”。指标太多，团队会在面板上迷路；指标太少，又会漏掉关键退化。真正要对齐的是四件事：观测窗口、样本量、标签维度、实例状态。

常见坑和规避方式如下：

| 问题 | 错误原因 | 规避方法 |
| --- | --- | --- |
| 只看均值 | 平均延迟掩盖尾部抖动 | 至少看 `p95/p99` 和超时率 |
| 样本太少 | 5% 流量下请求数不足，波动被当成趋势 | 设 `n_min`，不够样本不推进 |
| 标签混乱 | stable/canary 指标没有按 `version` 正确区分 | 强制按 `version`、`model`、`route` 打标签 |
| readiness 不完整 | Pod 已接流量，但模型未加载完或缓存未热 | 用 `startupProbe + readinessProbe`，预热前不接流量 |
| 镜像流量冒充灰度 | `shadow` 请求不影响用户，不能代表真实背压和超时链路 | 把影子流量和真实灰度分开看 |
| 只盯服务指标 | 忽略 GPU 显存、批大小、队列长度 | 同时采集系统资源指标 |
| 窗口没对齐 | stable 看 30 分钟，canary 看 5 分钟 | 使用同一时间窗、同一聚合口径 |
| 放量步子太大 | 5% 直接跳 50%，出问题放大太快 | 用多阶段权重推进，如 `5 -> 20 -> 50 -> 100` |

这里最容易被低估的是 warmup。warmup 就是模型刚加载完成、缓存未热、JIT 或 kernel 还没稳定时的启动阶段。很多团队看到 canary 延迟高，就断言新模型退化了；实际上只是 Pod 刚起来，首批请求在帮系统完成初始化。如果不把 warmup 单独隔离，灰度结论会天然偏向悲观。

另一个高频坑是标签串了。比如 stable 和 canary 都写到同一个 `service="inference"` 指标上，却没有 `version` 标签。这样你算出来的是混合结果，放量越大，stable 越“被污染”，最后你根本不知道是谁出了问题。

还有一个典型误区：把 shadow 当成灰度。影子流量会复制真实请求，但它通常不返回用户，也可能绕开真实超时、重试、连接池争用和客户端取消请求等链路。所以 shadow 很适合先验验证兼容性，却不能替代灰度放量决策。

---

## 替代方案与适用边界

灰度监控不是万能策略，它只是“真实流量下的渐进验证”方案。不同问题，要用不同工具。

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 灰度监控 | 直接反映真实线上表现，可逐步回滚 | 需要稳定兜底版本和足够流量 | 新模型上线、运行时升级 |
| 影子流量 `shadow` | 不直接影响用户，适合先做兼容验证 | 不能单独决定放量 | 请求格式变化、日志与行为对比 |
| A/B 测试 | 能比较业务收益，如点击率、停留时长 | 实现更复杂，周期更长 | 模型效果优劣比较 |
| 离线评估 | 成本低、可重复、便于大规模回放 | 分布可能与线上不一致 | 上线前筛选候选模型 |
| 全量回滚策略 | 故障处理快，动作简单 | 只能止损，不能提前发现渐进退化 | 已经触发事故后的应急 |

可以把它们理解成不同阶段的工具链：

1. 离线评估先筛掉明显不行的模型。
2. 影子流量检查协议兼容、日志、资源趋势。
3. 灰度监控判断能否承接真实用户流量。
4. A/B 测试再决定业务指标是否值得长期保留。

所以一句结论是：灰度监控适合真实流量验证，不适合替代离线验证。

它的适用边界也很明确：

- 如果业务流量很低，`5%` 灰度一天都积累不到足够样本，就该拉长观测窗口，或者先做离线回放。
- 如果反馈周期很长，比如模型输出要几小时后才知道对错，灰度更适合看系统指标，效果评估要交给离线或 A/B。
- 如果风险极高，比如金融风控、医疗辅助等场景，一次错误成本很大，就不该只依赖简单阈值，而要叠加更严格的审批、回放和隔离机制。

对初级工程师来说，最实用的判断标准是：当你关心的是“新版本能不能承受真实生产流量”，用灰度监控；当你关心的是“模型效果是不是更好”，灰度只能回答一部分，不能替代完整评估链路。

---

## 参考资料

1. [Kubernetes: Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)
2. [Prometheus: Query functions `histogram_quantile`](https://prometheus.io/docs/prometheus/3.4/querying/functions/)
3. [Argo Rollouts: Canary Deployment Strategy](https://argo-rollouts.readthedocs.io/en/stable/features/canary/)
4. [Argo Rollouts: Rollout Specification](https://argo-rollouts.readthedocs.io/en/latest/features/specification/)
5. [Istio: Request Routing](https://istio.io/latest/docs/tasks/traffic-management/request-routing/)
