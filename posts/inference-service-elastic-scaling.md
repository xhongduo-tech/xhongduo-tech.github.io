## 核心结论

推理服务弹性伸缩的本质，不是“自动加机器”，而是根据真实压力动态调整副本数，让延迟、吞吐和成本同时落在可接受范围内。这里的“副本”可以简单理解为一份独立运行的服务实例；“伸缩”就是增减这些实例的数量。

对推理服务，尤其是大模型推理服务，最关键的不是通用资源指标，而是能直接反映排队、并发和模型资源占用的指标。CPU 使用率高，不一定说明推理快到极限；CPU 使用率低，也不代表服务很空。真正决定用户体验的，往往是请求有没有排队、单副本能同时处理多少请求、KV cache 是否接近耗尽。KV cache 可以理解为模型为“正在生成中的请求”保留的一块上下文缓存，满了以后就很难继续稳定接单。

最直观的新手版本可以概括成三句话：

1. 流量上来就加副本，流量下去就减副本。
2. 如果支持 scale-to-zero，空闲时可以缩到 0。
3. 缩到 0 的代价是首个请求会变慢，因为模型需要重新拉起、加载并预热。

下表先把“目标、指标、动作、风险”放在一起看：

| 目标 | 常看指标 | 伸缩动作 | 主要风险 |
|---|---|---|---|
| 延迟稳定 | 每副本并发、排队长度、p95 延迟 | 压力升高时扩容 | 扩容慢于拥塞形成 |
| 吞吐提升 | 每副本吞吐、运行中请求数 | 增加副本分摊请求 | 指标选错导致扩容无效 |
| 成本控制 | 空闲副本数、平均利用率 | 低负载时缩容 | 缩太快导致抖动或冷启动 |
| 冷启动节省 | 空闲时间、最小副本数 | 支持时缩到 0 | 首请求延迟显著变差 |

结论可以再压缩成一句：弹性伸缩能否成立，不取决于“有没有自动扩缩组件”，而取决于“指标是否代表真实压力”以及“扩缩动作是否快于系统恶化”。

---

## 问题定义与边界

本文讨论的是推理服务的水平扩缩容，也就是调整副本数 $r$。不讨论修改模型结构、不讨论量化压缩，也不讨论单机 CUDA 调优。这些问题当然也重要，但它们属于“把单副本做强”；本文讨论的是“副本数如何随负载变化”。

边界要先说清楚，因为很多误解都来自边界混淆。

第一，弹性伸缩只解决“容量跟着负载变化”的问题，不保证单个请求一定更快。假设单副本本来就很慢，扩容只能减少排队，不能直接改变一次推理的内部计算时间。

第二，弹性伸缩只在指标能够代表真实压力时成立。对传统 Web 服务，CPU、内存、QPS 有时够用；对 LLM 推理，CPU 往往不是主信号。更接近真实瓶颈的，通常是运行中请求数、等待中请求数、KV cache 占用、单副本并发和 token 生成速率。

第三，伸缩不是瞬时完成的。即使控制器判断“该加副本了”，新 Pod 也需要调度、拉镜像、加载模型、探针就绪。这段时间里，请求仍然会继续排队。所以工程上不能只问“能不能扩”，而要问“扩容完成得是否足够快”。

一个新手能直接理解的边界例子是：如果一个副本能同时稳定处理 10 个请求，而现在平均每个副本来了 20 个请求，就该扩容；但如果这 20 个请求只是持续 3 秒的短暂抖动，立刻扩很多副本，可能等新副本起来时流量已经回落，钱花了，效果却不明显。

核心名词先统一：

| 名词 | 符号 | 含义 |
|---|---|---|
| 副本数 | $r$ | 当前服务实例数量 |
| 度量值 | $m$ | 当前观察到的压力指标，如每副本并发 |
| 目标值 | $m^*$ | 希望维持的目标压力 |
| 硬上限 | - | 单副本绝不能超过的并发或资源边界 |
| 软目标 | - | 希望接近的理想负载，不是绝对限制 |
| 稳定窗口 | - | 缩容前额外观察的一段时间，用于防抖 |

这里还要强调一个常被忽略的点：指标最好归一化到“每副本”。因为控制器最终决定的是“副本数要变成多少”，如果你直接用总请求数、总队列长度去做判断，不同副本规模下这个值的含义并不一致。总队列长度 100，在 2 个副本和 20 个副本下，代表的压力完全不同。

---

## 核心机制与推导

很多自动伸缩系统都可以抽象成一个简单公式：

$$
d = \left\lceil r \times \frac{m}{m^*} \right\rceil
$$

其中：

- $r$ 是当前副本数
- $m$ 是当前观测到的度量值
- $m^*$ 是目标值
- $d$ 是建议副本数

直白解释就是：当前副本数乘以“当前压力 / 目标压力”的比例，得到一个新的建议规模。如果当前压力是目标的 2 倍，那副本数大致也该翻倍。

玩具例子先看最小推导：

- 当前副本数 $r = 2$
- 每副本平均并发 $m = 35$
- 目标并发 $m^* = 20$

则：

$$
d = \left\lceil 2 \times \frac{35}{20} \right\rceil = \lceil 3.5 \rceil = 4
$$

意思是：当前两台不够，要扩到四台，才能把每台的平均负载拉回目标附近。

为什么强调“每副本平均并发”？因为如果你只看总并发 70，而不做归一化，控制器看不出“70 是压在 2 台上，还是压在 7 台上”。伸缩的本质是控制单副本承受的工作量，而不是盯着全局总量发呆。

再往前一步，要区分软目标和硬上限。很多新手会把这两个概念混为一谈。

| 概念 | 典型配置 | 作用 | 超过后会怎样 |
|---|---|---|---|
| 软目标 | `target` | 希望单副本平均承接的负载 | 可以短时间超过，系统通常尝试扩容 |
| 硬上限 | `containerConcurrency` 等 | 单副本允许接收的绝对上限 | 超过后进入排队、缓冲，或直接拒绝 |

在 Knative 或 KServe 语境里，`target` 更像“理想工作点”，不是硬闸门。比如你设置目标并发为 20，不代表第 21 个请求一定不能进来；它只是告诉控制器，平均负载高于 20 时，应考虑扩容。真正限制单副本极限承接能力的，往往是 `containerConcurrency`、GPU 显存边界、KV cache 上限等更硬的约束。

这也解释了一个常见现象：扩容判断已经发生，但延迟还是先涨。原因不是公式错了，而是“控制动作有滞后”。请求先排队，副本后补上，这是分布式系统的常态。

真实工程例子更明显。假设一个 vLLM 服务部署了 4 个副本，Prometheus 观察到：

- `num_requests_running` 总和为 24
- `num_requests_waiting` 总和为 40
- `kv_cache_usage_perc` 平均 88%

如果只看 CPU，也许只有 45%，你会误以为“资源还够”；但从运行中请求、等待请求和 KV cache 看，系统已经接近瓶颈。此时更合理的策略是：当等待请求持续增加且 KV cache 长时间高于阈值时扩容，而不是等 CPU 到 80% 才动。

缩容还必须配稳定窗口。稳定窗口可以理解为“先别急着减机器，再观察一会儿”。原因很简单：负载天然有波动，如果一掉就缩、一涨就扩，副本会来回抖动，既不省钱，也不稳定。很多平台默认把缩容观察窗口设为几分钟，常见值是 300 秒左右。它不是浪费，而是用时间换稳定性。

---

## 代码实现

工程里通常不是只写一个 HPA 就结束，而是把 KServe、Knative、KEDA、Prometheus 串成一条链路：

```text
读取指标 -> 计算每副本压力 -> 判断是否超过目标 -> 应用稳定窗口 -> 调整副本数 -> 等待观察下一轮指标
```

核心思路是：不要只盯 CPU，而是把业务上真正代表拥塞的指标接进伸缩控制器。

下面先给一个可运行的 Python 玩具实现，用来模拟建议副本数计算：

```python
import math

def desired_replicas(current_replicas: int, metric_per_replica: float, target_metric: float) -> int:
    assert current_replicas > 0
    assert metric_per_replica >= 0
    assert target_metric > 0
    return math.ceil(current_replicas * metric_per_replica / target_metric)

# 玩具例子：2 个副本，每副本平均并发 35，目标并发 20
d = desired_replicas(2, 35, 20)
assert d == 4

# 压力刚好命中目标，不需要扩
assert desired_replicas(4, 20, 20) == 4

# 压力下降时可能缩容
assert desired_replicas(4, 8, 20) == 2

print("ok")
```

这个例子没有实现容忍区间、稳定窗口和冷启动时间，但它把最核心的比例关系说明白了。

再看一个更接近真实系统的 KEDA + Prometheus 配置示意。这里不是盯 CPU，而是盯等待请求数：

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-inference-autoscaler
spec:
  scaleTargetRef:
    name: llm-inference-service
  minReplicaCount: 1
  maxReplicaCount: 10
  cooldownPeriod: 300
  pollingInterval: 15
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc.cluster.local:9090
        metricName: vllm_waiting_requests_per_replica
        query: |
          sum(vllm:num_requests_waiting) / max(kube_deployment_status_replicas{deployment="llm-inference-service"}, 1)
        threshold: "5"
```

这个配置表达的是：如果每副本平均等待请求数持续超过 5，就开始扩容。它比 CPU HPA 更贴近真实压力，因为它直接看“有没有人在排队”。

再给一个 KServe/Knative 风格的并发配置示意：

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llm-service
  annotations:
    autoscaling.knative.dev/metric: "concurrency"
    autoscaling.knative.dev/target: "20"
    autoscaling.knative.dev/minScale: "1"
    autoscaling.knative.dev/maxScale: "8"
    autoscaling.knative.dev/scaleDownDelay: "5m"
spec:
  predictor:
    containerConcurrency: 40
    model:
      modelFormat:
        name: huggingface
```

这里要读懂两层含义：

- `target: 20` 是软目标，表示希望平均并发控制在 20 附近。
- `containerConcurrency: 40` 是更硬的边界，表示单副本最多承接到 40 个并发请求，再高就会排队或缓冲。

在 LLM 场景里，还常会同时关注三个指标：

| 指标 | 代表什么 | 适合做什么判断 |
|---|---|---|
| `num_requests_running` | 正在执行的请求数 | 当前负载是否已接近单副本处理能力 |
| `num_requests_waiting` | 正在排队的请求数 | 是否已经开始拥塞 |
| `kv_cache_usage_perc` | KV cache 占用比例 | 是否接近显存/缓存瓶颈 |

一个真实工程判断常常不是“某个指标超过阈值就扩”，而是组合规则，例如：等待请求持续增长且 KV cache 超过 85%，说明不是短抖动，而是副本真的不够了。

---

## 工程权衡与常见坑

推理服务的核心矛盾通常不是“能不能扩”，而是“扩得够不够快、缩得够不够稳、成本会不会失控”。这三个目标往往彼此冲突。

最典型的新手误区，是把通用 Web 服务的经验直接搬到 LLM 推理上。LLM 的瓶颈经常在 KV cache、显存、内存带宽和排队，而不是 CPU。只看 CPU，很容易得出错误结论：明明用户已经超时，监控面板却显示 CPU 不高，于是控制器迟迟不扩容。

下表列几个高频坑：

| 坑点 | 后果 | 规避方法 |
|---|---|---|
| 只看 CPU | 压力失真，扩容滞后 | 优先接入并发、排队、KV cache 指标 |
| 把软目标当硬上限 | 对平台行为预期错误 | 区分 `target` 与 `containerConcurrency` |
| 缩到 0 太激进 | 首请求延迟恶化，p99 变差 | 保留最小副本或预热池 |
| 启动期指标污染 | 新 Pod 还没准备好就被统计 | 配好 `startupProbe` 和 `readinessProbe` |
| 缺少稳定窗口 | 副本来回抖动，成本上升 | 给缩容加观察窗口和冷却时间 |

“缩到 0”为什么危险，可以用一个新手场景理解：晚上系统几乎没流量，平台把副本全关了。凌晨突然来了第一批请求，系统要先调度 Pod、拉镜像、加载模型权重、初始化 CUDA、建立 KV cache，然后探针通过后才能接流量。这样虽然白天前的空闲时间节省了成本，但首批请求的延迟会明显变差。对聊天机器人、实时助手这类对首包延迟敏感的业务，这个代价可能无法接受。

`readinessProbe` 和 `startupProbe` 也不能当成运维细节忽略。它们直接影响 HPA 或上游流量系统看到的“有效副本数”。如果 Pod 还在启动，探针未通过，它不应该被当成可承压实例；如果探针过于宽松，未准备好的实例提前接流量，延迟会被污染；如果探针过于严格，扩容刚加出的 Pod 很久不计入可用容量，控制器又会误判“还是不够”，继续扩，最终造成过冲。

实践里，比较稳妥的流程通常是：

1. 先压测，找出单副本在不同输入长度下的稳定并发区间。
2. 再把这个区间映射成 `target`、硬上限和告警阈值。
3. 最后给缩容加稳定窗口，给启动加探针保护。

---

## 替代方案与适用边界

弹性伸缩不是唯一方案，也不是任何场景都最优。它适合“负载波动明显、指标可观测、模型可较快拉起”的系统；不适合“启动极慢、突发极强、对 p99 极度敏感”的系统。

先做一个对比：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| HPA | 机制通用，接入简单 | 默认更偏资源指标 | 传统推理服务、指标较简单 |
| KServe / Knative 自动扩缩 | 支持并发和 scale-to-zero | 需理解软目标与平台行为 | 服务化推理、流量波动明显 |
| 固定副本 + 限流 | 行为稳定，易控成本 | 高峰期可能排队 | 流量稳定、SLO 明确 |
| 预热池 + 手动扩容 | 冷启动风险低 | 资源利用率较低 | 模型加载慢、首延迟敏感 |
| 分层队列调度 | 保护核心请求 | 系统复杂度高 | 多租户、优先级差异大 |

什么时候适合自动伸缩：

- 请求波动明显，白天和夜间差距大。
- 指标能稳定反映真实压力，比如排队长度和 KV cache 占用。
- 模型副本能在可接受时间内完成启动。

什么时候不太适合纯自动伸缩：

- 模型加载非常慢，扩容生效时间长于拥塞恶化时间。
- 流量是“极端突发”，几秒内打满系统。
- 业务对 p99 延迟极敏感，不能接受冷启动或扩容滞后。

一个真实工程上的折中方案是：保留少量常驻副本，叠加自动伸缩，再给高优先级请求单独队列。这样做不是最省钱，但常常比“全部交给自动伸缩”更稳。因为自动伸缩解决的是长期容量匹配，不擅长独自处理毫秒级尖峰和严格优先级。

所以，自动伸缩更像容量调节器，不是性能万能药。单副本性能差、模型加载慢、调度链路长，这些问题不会因为“开了 HPA”就自动消失。

---

## 参考资料

新手的阅读顺序建议是：先看 HPA 的计算逻辑，再看 Knative 的 `target` 与并发模型，最后看 KServe 和 vLLM 的业务指标如何映射到真实负载。本文中关于公式、软目标、稳定窗口等基础机制，主要来自官方文档；关于“LLM 更应关注排队、KV cache、冷启动”的部分，是基于官方指标定义和工程实践的归纳，不应当被理解为对所有系统都绝对成立。

1. [Kubernetes Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/)
2. [Knative Serving Autoscaling](https://knative.dev/v1.19-docs/serving/autoscaling/)
3. [Knative Concurrency-based Autoscaling](https://knative.dev/v1.20-docs/serving/autoscaling/concurrency/)
4. [Knative Target Burst Capacity](https://knative.dev/docs/serving/load-balancing/target-burst-capacity/)
5. [KServe KPA Autoscaler](https://kserve.github.io/website/docs/model-serving/predictive-inference/autoscaling/kpa-autoscaler)
6. [KServe Generative Inference Autoscaling](https://kserve.github.io/website/docs/model-serving/generative-inference/autoscaling)
7. [vLLM Metrics Design](https://docs.vllm.ai/en/stable/design/metrics.html)
