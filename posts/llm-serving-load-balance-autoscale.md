## 核心结论

LLM 服务的低延迟弹性，不是单靠“多开几个副本”，而是靠三件事同时成立：请求路由要聪明、扩缩容指标要接近用户等待、版本发布要能控流量。这里的“路由”就是决定一个请求该进哪个 Pod；“扩缩容”就是自动加减副本；“控流量”就是新旧模型并行时按比例放量。

对多实例模型服务，普通 Round-Robin 只是平均分流，适合无状态、无缓存复用的服务；一旦引入 Prefix Cache，也就是“把相同前缀的中间计算结果缓存起来”，就必须考虑 Prefix-aware 路由或 Sticky Session。Sticky Session 的白话解释是：同一段对话尽量回到同一台实例。否则同样的上下文会在不同 Pod 上重复预填充，TTFT，也就是“首个 token 返回时间”，会明显变差。

扩缩容也不能只看 GPU 利用率。GPU 利用率高，只说明卡很忙，不说明用户一定在排队。对 LLM，更接近真实体验的指标通常是队列深度、批大小、KV/Prefix Cache 命中率这类“服务内指标”。工程上常见做法是：HPA 负责持续调节，KEDA 负责把 Prometheus 等外部指标接进 HPA，最终按多个指标中要求副本数最大的那个扩容，避免只看单一信号导致误判。

---

## 问题定义与边界

这篇文章讨论的是“多副本 LLM 在线推理服务”的流量分配与自动扩缩容，不讨论模型训练，也不讨论单机脚本推理。

目标通常有两个，而且经常互相拉扯：

| 目标 | 白话解释 | 常见指标 |
|---|---|---|
| 低延迟 | 用户发请求后尽快看到第一个字，并且整段输出别太慢 | TTFT、TPOT、P95 latency |
| 高利用率 | GPU 不要长期空转，否则成本很高 | GPU duty cycle、吞吐、tokens/s |

问题在于，LLM 不是普通 Web 服务。请求处理常分成 prefill 和 decode 两段。prefill 是“先把整段上下文过一遍并建立 KV cache”，decode 是“基于已有缓存一个个生成 token”。如果 3 个有相同前缀的请求被 Round-Robin 分到 3 个不同 Pod，那么 3 台机器都要各自重建缓存，缓存命中率接近 0。改成 Prefix-aware 或 Sticky Session 后，同一前缀会尽量落在同一 Pod，TTFT 通常立刻下降。

这里有一个玩具例子。假设有 3 个 Pod：A、B、C，连续来了 3 个请求，前缀都相同，都是“总结下面这段合同”。  
Round-Robin 会分成 A、B、C，各自做一次 prefill。  
Prefix-aware 会优先把 3 个请求都打给已经持有该前缀缓存的 A。  
前者更“平均”，后者更“省计算”。对 LLM，后者通常更快。

三种常见路由策略可以直接对比：

| 策略 | 怎么做 | 缓存命中率 | 延迟表现 | 实现复杂度 |
|---|---|---|---|---|
| Round-Robin | 按顺序轮流分发 | 低 | 稳定但不一定低 | 低 |
| 最短队列 | 发给当前等待最少的 Pod | 中 | 通常优于 RR | 中 |
| Prefix-aware / Sticky | 优先发给已有前缀缓存的 Pod | 高 | 对对话和长上下文最优 | 高 |

边界也要说清楚：如果你的服务没有会话、多数请求上下文都不同、也没有 Prefix Cache，那么 Sticky Session 的收益会明显下降，甚至可能让负载分布变差。

---

## 核心机制与推导

先看扩缩容的核心公式。无论 HPA 还是 KEDA，底层思路都可以抽象成：

$$
desired = clamp(minReplicas,\ \lceil \frac{currentMetric}{targetMetric} \rceil,\ maxReplicas)
$$

这里 `clamp` 的意思是“把结果夹在最小和最大副本之间”。

如果按队列深度扩容：

$$
desired_{queue} = clamp(min,\ \lceil queueDepth / targetQueueDepth \rceil,\ max)
$$

如果按 GPU 利用率扩容：

$$
desired_{gpu} = clamp(min,\ \lceil gpuUtil / targetGpuUtil \rceil,\ max)
$$

当同时使用多个指标时，HPA 会按“需要副本数最大的那个指标”来决定。这一点很关键，因为它本质上是在保守地保证容量，而不是在多个指标之间做平均。

给一个最小数值例子。设 `min=1, max=8`，目标 GPU 利用率 70%，当前平均 85%。则：

$$
\lceil 85/70 \rceil = 2
$$

所以仅从这个指标出发，HPA 会希望至少扩到 2 个副本。  
如果此时队列深度是 45，而队列阈值是 20，那么：

$$
\lceil 45/20 \rceil = 3
$$

最终就会取 3，而不是 2，因为真实排队更严重。

这也是为什么只看 GPU 利用率经常不够。Google Cloud 的实践里明确提到，GPU duty cycle 与请求延迟没有稳定的一一对应关系；而队列深度和批大小与延迟更直接相关。换句话说，GPU 忙，不代表用户正在等；队列长，则几乎肯定有人在等。

可以把流程理解成 3 步：

1. 采集指标：队列深度、批大小、GPU 利用率、KV cache 利用率或命中率。
2. 计算目标副本：HPA/KEDA 分别按各自公式算 `desired`，最后取最大值。
3. 调整流量：负载均衡器或网关把新流量导向新增副本；模型升级时再配合 Istio 逐步切流。

真实工程例子：一个客服对话系统白天高峰期多为“同一会话连续追问”。这类流量对 Prefix Cache 极度敏感。如果入口只做最短队列而不做会话亲和，负载可能看起来很平均，但 TTFT 会偏高，因为大量请求在做重复 prefill。更合理的策略通常是“两阶段”：先按 prefix/session 找候选 Pod，再在候选 Pod 里挑最短队列。

---

## 代码实现

下面先用一个可运行的 Python 玩具程序，把“按多个指标算副本数”和“前缀感知路由”写清楚。

```python
import math
from dataclasses import dataclass

def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(x, hi))

def desired_replicas(current: float, target: float, min_r: int, max_r: int) -> int:
    assert target > 0
    return clamp(math.ceil(current / target), min_r, max_r)

def merged_desired(metrics, min_r: int, max_r: int) -> int:
    # metrics: [(current, target), ...]
    wants = [desired_replicas(current, target, min_r, max_r) for current, target in metrics]
    return max(wants)

@dataclass
class Pod:
    name: str
    queue_depth: int
    prefixes: set

def pick_pod(prefix: str, pods: list[Pod]) -> str:
    # 先找已有前缀缓存的 Pod，再选最短队列
    matched = [p for p in pods if prefix in p.prefixes]
    pool = matched if matched else pods
    best = min(pool, key=lambda p: p.queue_depth)
    return best.name

# 扩缩容例子
gpu_based = desired_replicas(current=85, target=70, min_r=1, max_r=8)
queue_based = desired_replicas(current=45, target=20, min_r=1, max_r=8)
final_replicas = merged_desired([(85, 70), (45, 20)], min_r=1, max_r=8)

assert gpu_based == 2
assert queue_based == 3
assert final_replicas == 3

# 路由例子
pods = [
    Pod("pod-a", queue_depth=3, prefixes={"chat:alice"}),
    Pod("pod-b", queue_depth=1, prefixes=set()),
    Pod("pod-c", queue_depth=2, prefixes={"chat:bob"}),
]

assert pick_pod("chat:alice", pods) == "pod-a"   # 命中缓存优先
assert pick_pod("chat:carol", pods) == "pod-b"   # 无缓存时选最短队列

print("ok")
```

在 Kubernetes 里，HPA 适合做稳定的自动调节。下面是一个基于队列深度的 `autoscaling/v2` 示例：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-server
  minReplicas: 1
  maxReplicas: 8
  metrics:
  - type: Pods
    pods:
      metric:
        name: prometheus.googleapis.com|tgi_queue_size|gauge
      target:
        type: AverageValue
        averageValue: "20"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0
    scaleDown:
      stabilizationWindowSeconds: 300
```

这里的 `AverageValue: "20"` 可以理解成“每个 Pod 平均允许约 20 个等待请求”，超过就扩。

KEDA 更适合把 Prometheus、自定义事件源接进来。下面是一个用 `vllm:num_requests_waiting` 的例子：

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-keda
spec:
  scaleTargetRef:
    name: llm-server
  pollingInterval: 15
  cooldownPeriod: 300
  minReplicaCount: 1
  maxReplicaCount: 5
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-operated.monitoring.svc:9090
      metricName: vllm:num_requests_waiting
      query: vllm:num_requests_waiting
      threshold: "5"
      activationThreshold: "1"
```

如果当前队列是 25，阈值是 5，那么直觉上就是 `ceil(25/5)=5` 个副本。`activationThreshold` 的意思是“低于这个值时别急着激活扩容”，常用于减少轻微抖动。

模型更新时，蓝绿部署和金丝雀发布要和流量治理一起用。蓝绿部署是“新旧两套完整环境并存，切换一次完成”；金丝雀发布是“先给新版本极少量流量，再逐步增加”。Istio 的权重路由是一个常见做法，例如先 `95/5`，观察 TTFT、错误率、P95 latency，再决定是否扩大新版本比例。

---

## 工程权衡与常见坑

最常见的误区，是把 GPU 利用率当成唯一主指标。它的优点是通用，缺点是和用户等待不直接对应。尤其在 decode 阶段，GPU 可能持续很忙，但队列其实已经清空了；这时继续保留很多副本，只会浪费钱。

另一个常见坑，是“看到了 Prefix Cache，却没做请求亲和”。缓存不是自动生效的，前提是相同前缀要尽量进同一实例。没有这层路由，即使模型框架支持 KV/Prefix Cache，收益也会被入口层冲掉。

下面是一个风险表：

| 指标或策略 | 潜在误判 | 结果 | 缓解手段 |
|---|---|---|---|
| 只看 GPU 利用率 | GPU 忙但无人排队 | 过度扩容，成本高 | 加入 queue depth 作为主触发 |
| 只看队列深度 | 队列短但 batch 已接近上限 | 延迟突然恶化 | 对低延迟业务加入 batch/KV 指标 |
| Round-Robin | 相同前缀被打散 | Prefix Cache 命中率低 | 改为 prefix-aware 或 Sticky Session |
| 激进缩容 | 刚空闲就回收副本 | 抖动、冷启动增加 | 设置 scaleDown stabilization window |
| 金丝雀直接放量 | 新模型性能退化未被发现 | 大面积延迟或错误 | 先 95/5，再逐步放量 |

还有一个容易忽略的问题：Sticky Session 不是永远越强越好。若某个超长对话持续粘在单 Pod，上下文很长、输出也很长，这台机器会明显更忙。工程上不能只做“硬亲和”，还要结合队列、KV 使用率做兜底，避免热点 Pod 被长期压垮。

---

## 替代方案与适用边界

如果业务流量是明显的突发型，比如整点批量请求、营销活动瞬时洪峰，KEDA 往往比单纯 HPA 更合适，因为它天然适合接外部事件源，也支持更灵活的激活和 scale-to-zero。代价是系统链路更长，排障复杂度更高。

如果业务流量规律稳定，比如工作日早 9 点固定高峰，可以考虑预测式扩缩容。预测式扩缩容的白话解释是：不是等队列已经涨起来再扩，而是根据历史到达率提前准备容量。它适合规律强、成本敏感的系统，不适合事件驱动特别强、波动模式经常变化的系统。

如果发布的是新模型权重、新推理后端，或者改了缓存策略，那么蓝绿和金丝雀比“原地滚动更新”更稳，因为你需要验证的不只是可用性，还包括 TTFT、TPOT、cache hit、吞吐这些性能指标。

| 方案 | 适用场景 | 边界 |
|---|---|---|
| HPA + 队列深度 | 稳定在线服务，目标是吞吐与成本平衡 | 对超低延迟场景未必够快 |
| HPA + batch/KV 指标 | 延迟敏感场景 | 阈值更依赖实验调参 |
| KEDA + Prometheus | 突发流量、自定义指标丰富 | 链路复杂，观测要求更高 |
| Predictive Scaling | 到达率稳定、可预测高峰 | 对随机流量效果有限 |
| Blue/Green | 大版本切换、回滚要求高 | 资源成本更高 |
| Canary | 渐进发布、验证新模型表现 | 需要完善监控与流量治理 |

所以，一个实用的默认组合通常是：入口用“Prefix-aware + 最短队列”，扩缩容用“HPA/KEDA 多指标并行”，发布用“蓝绿保底、金丝雀放量”。这不是唯一方案，但对大多数多实例 LLM 服务，是目前工程上最稳的起点。

---

## 参考资料

- Google Cloud: [Best practices for autoscaling large language model (LLM) inference workloads with GPUs on GKE](https://cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling)
- Google Cloud: [Configure autoscaling for LLM workloads on GPUs with GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/inference/autoscaling)
- Google Cloud Blog: [Tuning the GKE HPA to run inference on GPUs](https://cloud.google.com/blog/products/containers-kubernetes/tuning-the-gke-hpa-to-run-inference-on-gpus)
- Google Cloud Blog: [GKE Inference Gateway and Quickstart are GA](https://cloud.google.com/blog/products/ai-machine-learning/gke-inference-gateway-and-quickstart-are-ga)
- Google Cloud Docs: [About GKE Inference Gateway](https://cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway)
- vLLM Docs: [Autoscaling with KEDA](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html)
- KEDA Docs: [ScaledObject specification](https://keda.sh/docs/2.19/reference/scaledobject-spec/)
- Istio Docs: [Canary Deployments using Istio](https://istio.io/latest/blog/2017/0.1-canary/)
- Istio Docs: [Traffic Management Concepts](https://preliminary.istio.io/latest/docs/concepts/traffic-management/)
