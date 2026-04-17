## 核心结论

基于指标的弹性伸缩，本质上是一个**控制回路**。控制回路的白话解释是：系统每隔一段时间看一次“现在有多忙”，再和“我希望它忙到什么程度”做比较，然后决定是否增减实例。

它的价值不在“自动”两个字，而在三个更具体的结果：

1. 让资源跟着负载走。高峰时扩容，低谷时缩容，避免长期按峰值买机器。
2. 把性能目标转成配置。比如“单个实例 CPU 长期别超过 60%”或者“消息队列积压别超过 1000 条”。
3. 把人工值班变成参数调优。真正难的不是会不会扩，而是何时扩、扩多少、何时缩、缩多快。

对初级工程师来说，最重要的结论只有一句：**自动扩缩容不是“看见 CPU 高就加机器”，而是“围绕目标指标做受约束的副本调整”。** 这里的“受约束”包括最小副本数、最大副本数、容忍区间、稳定窗口、扩缩速率限制。

一个最常用的推导公式是：

$$
desiredReplicas = \left\lceil currentReplicas \times \frac{currentMetricValue}{desiredMetricValue} \right\rceil
$$

它表示：当前副本数乘以“当前指标 / 目标指标”的比值，再向上取整。

例如，当前有 3 个副本，平均 CPU 利用率是 75%，目标是 50%，那么：

$$
\left\lceil 3 \times \frac{75}{50} \right\rceil = \lceil 4.5 \rceil = 5
$$

系统就会倾向扩到 5 个副本。这个公式简单，但工程上必须配合稳定机制，否则很容易抖动。

---

## 问题定义与边界

自动扩缩容要解决的问题，不是“怎么把副本数调大调小”，而是：

**在负载波动下，如何同时满足可用性、延迟目标和成本约束。**

这里有三个边界必须先定义清楚。

| 边界项 | 作用 | 常见错误 |
| --- | --- | --- |
| `minReplicas` | 保证低谷时仍有基础处理能力 | 设成 0 或 1，导致冷启动太慢 |
| `maxReplicas` | 控制峰值成本和资源上限 | 设太大，突发流量时成本失控 |
| 指标目标值 | 定义“多忙算合适” | 目标值拍脑袋，和真实性能无关 |
| 稳定窗口/冷却时间 | 防止频繁来回扩缩 | 不设置，系统出现抖动 |
| 节点容量边界 | 保证 Pod 扩出来后有地方跑 | 只配 HPA，不配节点扩容 |

这里的 **HPA**，白话解释是：Kubernetes 里专门管“横向加减 Pod 副本”的控制器。  
这里的 **Pod**，白话解释是：Kubernetes 中承载容器的最小运行单元。  
这里的 **Cluster Autoscaler**，白话解释是：当 Pod 太多、节点放不下时，负责给集群加节点的组件。

所以，自动扩缩容有明确边界：**它只能解决“已经定义好可扩展副本”的工作负载问题。** 对于单体数据库、强状态服务、启动很慢的模型实例，扩缩容未必直接有效。

### 玩具例子

假设你有一个小型 API 服务：

- 每个实例稳定处理能力约为 100 QPS
- 目标 CPU 利用率是 60%
- 最小副本数 2
- 最大副本数 8

如果业务从 150 QPS 上升到 380 QPS，2 个实例显然扛不住。系统看到 CPU 升高后，会把副本数逐步拉高到 4 或 5。  
如果夜间负载降回 80 QPS，系统再把副本数缩回 2。

这个例子里，扩缩容解决的是“波动”，不是“无限增长”。因为一旦流量超过 `maxReplicas` 对应的总处理能力，系统还是会超时。

---

## 核心机制与推导

自动扩缩容通常依赖三层机制。

### 1. 指标采集

**指标**，白话解释是：能反映系统忙闲程度的数字。  
最常见的指标分为四类：

| 指标类别 | 代表例子 | 适合场景 | 局限 |
| --- | --- | --- | --- |
| 资源指标 | CPU、内存 | Web 服务、普通 API | 不一定直接反映业务压力 |
| 流量指标 | QPS、RPS、并发连接数 | 网关、HTTP 服务 | 需要额外采集系统 |
| 积压指标 | 队列长度、消息堆积、Lag | 异步消费、批处理 | 只适合队列型系统 |
| 业务指标 | 平均推理时延、活跃会话数 | 模型服务、实时系统 | 定义和采集成本更高 |

为什么不能只看 CPU？因为 CPU 高不一定等于业务忙，CPU 低也不一定等于系统空。  
例如，一个消费者服务被外部接口阻塞，CPU 可能很低，但队列已经堆积。

### 2. 目标比值计算

HPA 的核心算法是目标比值法。  
如果当前指标高于目标，说明每个实例太忙，需要扩容；如果低于目标，说明每个实例太闲，可以缩容。

公式是：

$$
desiredReplicas = \left\lceil currentReplicas \times \frac{currentMetricValue}{desiredMetricValue} \right\rceil
$$

### 数字推导

假设当前有 4 个副本，平均 CPU 利用率 90%，目标值 60%。

$$
desiredReplicas = \left\lceil 4 \times \frac{90}{60} \right\rceil = \lceil 6 \rceil = 6
$$

所以系统建议扩到 6 个副本。

如果当前平均 CPU 降到 30%：

$$
desiredReplicas = \left\lceil 4 \times \frac{30}{60} \right\rceil = \lceil 2 \rceil = 2
$$

系统建议缩到 2 个副本。

### 3. 稳定机制

如果只有公式，没有稳定机制，系统会来回抖动。  
典型稳定机制有三种：

1. **容忍度**。白话解释是：差一点点就不动，别对小波动过度反应。
2. **稳定窗口**。白话解释是：看最近一段时间的建议值，不要只看这一瞬间。
3. **扩缩速率限制**。白话解释是：一次最多加多少、一次最多减多少。

Kubernetes 默认控制循环不是连续运行，而是周期性检查。官方文档中默认同步周期通常是 15 秒。  
这意味着扩缩容本来就不是“瞬时生效”，而是“定期观测，逐次逼近目标”。

### 缺失指标和未就绪实例

工程上还有两个常见细节：

- **缺失指标**：有些 Pod 还没上报指标，系统会保守处理，避免误判。
- **未就绪 Pod**：新 Pod 刚启动时还不能接流量，不能立刻按正常实例计算能力。

这也是为什么模型服务、JVM 服务、冷启动慢的服务，必须把启动期算进扩容策略中，否则会出现“还没准备好就继续加，越加越乱”。

### 真实工程例子

一个在线推理服务在白天稳定 20 RPS，晚上直播时冲到 200 RPS。  
如果只按平均负载准备 2 个实例，直播开始后延迟会立刻飙升。  
合理做法通常是：

- HPA 监控 CPU 和请求速率
- `minReplicas` 设为 3，避免低谷全缩没
- `maxReplicas` 设为 20，控制预算
- `scaleUp` 允许每 30 秒最多翻倍或增加 4 个
- `scaleDown` 设置 5 分钟稳定窗口

这样做的目的不是“尽快扩到最多”，而是“尽快扩到够用，缩容时更谨慎”。

---

## 代码实现

下面先用一个可运行的 Python 玩具实现，模拟“按目标指标计算副本数”。

```python
import math

def desired_replicas(current_replicas, current_metric, desired_metric, tolerance=0.1,
                     min_replicas=1, max_replicas=100):
    ratio = current_metric / desired_metric

    # 容忍区间内不扩不缩，避免抖动
    if abs(ratio - 1.0) <= tolerance:
        return current_replicas

    replicas = math.ceil(current_replicas * ratio)
    replicas = max(min_replicas, min(max_replicas, replicas))
    return replicas

# 玩具例子：3 个副本，CPU 75%，目标 50%
assert desired_replicas(3, 75, 50) == 5

# 低于目标很多，触发缩容
assert desired_replicas(4, 30, 60) == 2

# 接近目标值，落在容忍区间内，不调整
assert desired_replicas(5, 52, 50, tolerance=0.1) == 5

# 受最小副本数约束
assert desired_replicas(2, 10, 60, min_replicas=2) == 2

# 受最大副本数约束
assert desired_replicas(10, 300, 50, max_replicas=20) == 20
```

这个代码只模拟最核心的数学关系，没有实现指标缺失处理、启动期保护、历史推荐值稳定窗口，也没有实现多指标取最大值。但它已经足够说明 HPA 的第一性原理。

如果落到 Kubernetes，常见配置是 `autoscaling/v2`，因为它支持多指标和 `behavior`。

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 30
        - type: Pods
          value: 4
          periodSeconds: 30
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 20
          periodSeconds: 60
      selectPolicy: Max
```

关键字段含义如下：

| 字段 | 含义 | 设计意图 |
| --- | --- | --- |
| `scaleTargetRef` | 绑定要扩缩的 Deployment | 指定控制对象 |
| `minReplicas` | 最少保留副本数 | 保证基础可用性 |
| `maxReplicas` | 最多扩到多少副本 | 限制成本和资源 |
| `metrics` | 扩缩容依据的指标 | 定义“忙”和“闲” |
| `behavior.scaleUp` | 扩容速度规则 | 防止扩得过猛或过慢 |
| `behavior.scaleDown` | 缩容速度规则 | 防止刚降下来又涨回去 |

对于新手，最容易忽略的一点是：**HPA 只负责改副本数，不负责保证有足够节点。**  
如果 20 个 Pod 需要 5 台节点，而集群只有 2 台，Pod 会 Pending。这时要让 Cluster Autoscaler 负责节点层扩容。

---

## 工程权衡与常见坑

自动扩缩容最难的部分不是“写出配置”，而是“选对指标并约束行为”。

### 常见坑 1：只看 CPU

这是最常见也最危险的简化。  
CPU 适合 CPU 密集型服务，不适合所有服务。

例如，一个消息消费服务真正的瓶颈是外部数据库写入速度。  
这时 CPU 可能只有 35%，但队列已经积压 10 万条。  
如果仍然按 CPU 扩缩容，系统会误以为“还很闲”。

更稳妥的做法如下：

| 场景 | 首选指标 | 备注 |
| --- | --- | --- |
| 普通 Web API | CPU + 请求速率 | 兼顾资源和流量 |
| 异步消息消费 | 队列长度 + 消费时延 | 直接反映堆积 |
| 模型推理服务 | GPU/CPU + P95 延迟 + 并发数 | 资源和用户体验都要看 |
| 批处理任务 | 待处理任务数 | 比 CPU 更直接 |

### 常见坑 2：缩容太快

缩容比扩容更危险。  
扩容太慢通常是延迟变高；缩容太快可能直接把刚稳定的服务又缩回去。

尤其是在线推理场景，实例启动慢、模型加载重，缩容后再扩回来成本很高。  
因此很多生产系统会采用“扩容激进，缩容保守”的策略：

- 扩容允许快速加副本
- 缩容使用更长稳定窗口
- 缩容比例受限，比如每分钟最多减 10% 到 20%

### 常见坑 3：忽略冷启动

**冷启动**，白话解释是：新实例从创建到真正能接流量，中间需要的准备时间。  
模型服务的冷启动尤其明显，因为要拉镜像、加载模型、初始化权重、建立缓存。

如果冷启动要 90 秒，而流量峰值在 30 秒内打满，那么单靠实时 HPA 往往来不及。  
这时要配合：

- 更高的 `minReplicas`
- 预热实例
- 定时扩容
- 基于请求趋势的提前扩容

### 常见坑 4：Pod 扩出来了，节点没跟上

HPA 和 Cluster Autoscaler 解决的是两个不同层级的问题：

- HPA 负责 Pod 数量
- Cluster Autoscaler 负责节点数量

如果只配 HPA，不配节点扩容，结果是 HPA 计算出“应该从 10 扩到 30”，但集群没有足够节点，20 个新 Pod 都处于 Pending。  
这类故障表面上看像“扩容失效”，本质上是容量规划断层。

### 常见坑 5：和 VPA 混用不当

**VPA**，白话解释是：不加副本，而是调整单个 Pod 的资源请求和限制。  
如果 HPA 和 VPA 同时都根据 CPU/内存做决策，就可能互相打架：

- HPA 觉得应该加副本
- VPA 觉得应该给单 Pod 更多资源

结果是系统行为不可预测。  
更常见的做法是：HPA 用业务指标或外部指标扩副本，VPA 用历史资源使用做资源定型。

---

## 替代方案与适用边界

基于指标的弹性伸缩不是唯一方案。不同负载形态，对应的最优方案并不一样。

| 方案 | 解决的问题 | 适合场景 | 不适合场景 |
| --- | --- | --- | --- |
| HPA | 调整副本数 | 无状态服务、Web API、推理服务 | 单实例强状态系统 |
| VPA | 调整单 Pod 资源 | 负载变化慢、资源请求不准 | 需要快速应对流量峰值 |
| Cluster Autoscaler | 调整节点数 | Pod 扩容后节点不足 | 不能替代 Pod 层扩缩 |
| KEDA | 基于外部事件扩缩 | 消息队列、流处理、事件驱动 | 纯 CPU 型服务不一定需要 |
| 定时扩缩容 | 已知固定峰谷 | 工作日白天高峰、夜间低谷 | 流量不可预测场景 |

**KEDA**，白话解释是：把队列长度、消息积压、流式 Lag 等外部事件接入 Kubernetes 扩缩容的组件。  
它特别适合“压力不体现在 CPU，而体现在外部事件数量”的系统。

### 什么时候优先选 HPA

- 服务是无状态的
- 可以横向复制多个副本
- 有明确、可采集、能代表压力的指标
- 需要分钟级自动响应流量变化

### 什么时候 HPA 不够

- 启动极慢，扩容来不及
- 单实例持有大量状态，难以复制
- 性能瓶颈不在 Pod，而在数据库或外部依赖
- 流量峰值可预测，提前扩容比实时扩容更有效

一个典型真实工程判断是：

- Web 前端、API 网关、在线推理入口：优先 HPA
- RabbitMQ/Kafka 消费者：优先 KEDA 或队列指标 HPA
- 后台任务执行器：任务数驱动扩容
- 数据库、缓存、状态服务：更依赖垂直扩容、分片或专用架构

所以，**自动扩缩容不是通用解，而是“可横向复制系统”的效率工具。**

---

## 参考资料

- Kubernetes 官方文档：Horizontal Pod Autoscaling
- Kubernetes 官方文档：HorizontalPodAutoscaler Walkthrough
- Kubernetes 官方文档：Vertical Pod Autoscaling
- Microsoft Learn：AKS Cluster Autoscaler Overview
- KEDA 官方文档：Concepts
