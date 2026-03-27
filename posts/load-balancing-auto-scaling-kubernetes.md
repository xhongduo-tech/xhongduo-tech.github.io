## 核心结论

Kubernetes 里的“负载均衡”不是一个单点组件，而是一组分层抽象。

`Service` 是服务发现与四层转发抽象。白话说，它给一组会变化的 Pod 一个相对稳定的访问入口。  
`Ingress` 是七层流量入口抽象。白话说，它负责把不同域名、不同路径的 HTTP 请求分到不同 Service。  
Pod 调度是资源落点机制。白话说，它决定新 Pod 具体放到哪台节点上运行。

自动扩缩容也不是一个按钮，而是三个层级协同：

| 组件 | 作用层级 | 主要解决问题 | 常见指标 |
| --- | --- | --- | --- |
| HPA | Pod 数量 | 单个服务副本不够 | CPU、内存、自定义指标、外部指标 |
| VPA | 单 Pod 资源请求/限制 | 单个 Pod 规格不合理 | CPU 使用、内存使用、OOM 历史 |
| CA | 节点数量 | Pod 想扩但集群没地方放 | Pending Pod、节点利用率 |
| Karpenter | 节点数量与生命周期 | 更灵活地自动建节点 | Pending Pod、约束条件、节点寿命 |

实践上要记住一句话：先把入口稳定，再让扩缩容跟着真实瓶颈走。对于 Web 服务，常从 CPU 或请求速率起步；对于队列消费、日志查询、GPU 推理，更可靠的指标常常是队列长度、批大小、尾延迟，而不是 CPU。

---

## 问题定义与边界

这类问题的核心不是“怎么把副本数调大”，而是“在 Pod 和节点不断变化时，如何让外部请求持续进来，并且以合理成本撑住峰值流量”。

先看边界：

| 观察层 | 典型信号 | 说明 | 常见响应方式 |
| --- | --- | --- | --- |
| 流量入口 | 404、502、握手失败、RT 飙升 | 入口规则或后端不可达 | 检查 Ingress、Service、探针 |
| Pod 层 | CPU 90%、重启、OOM、Pending | 单 Pod 过热或副本不足 | HPA 扩 Pod、VPA 调请求 |
| 节点层 | 节点 CPU/内存打满、Pod 无法调度 | 集群容量不够 | CA 或 Karpenter 扩节点 |
| 队列层 | backlog 增长、等待时长升高 | 系统吞吐跟不上 | 用外部指标驱动 HPA |
| 业务层 | P95/P99 延迟上升、错误率升高 | 用户感知已受影响 | 联动告警、限流、扩容 |

玩具例子：  
一个 `frontend` Deployment 有 3 个 Pod，前面挂一个 `Service`，再由 `Ingress` 把 `/api` 路由到这个 Service。此时某个 Pod 被重建，Pod IP 会变，但 Service 名字和 ClusterIP 不变，Ingress 仍然把请求送到同一个逻辑后端。对外入口稳定，对内后端可变，这就是 Kubernetes 负载均衡的基本边界。

真实工程例子：  
一个在线推理服务白天流量平稳，晚上活动开始后请求量 10 分钟内上涨 6 倍。若只盯 CPU，扩容可能偏慢，因为 GPU 推理常先在请求队列里积压，CPU 还没打满，延迟已经恶化。此时真正该看的边界不是“CPU 有没有超过 80%”，而是“队列长度是否跨过可接受阈值”“P95 延迟是否逼近 SLA”。

---

## 核心机制与推导

### 1. Service、Ingress、调度三层如何配合

Service 维护 Pod 集合的稳定访问点。它背后依赖标签选择器和 Endpoint/EndpointSlice，把流量转发到当前健康 Pod。  
Ingress 只做 HTTP/HTTPS 规则匹配，本身不直接转发包，真正执行流量控制的是 Ingress Controller。  
调度器在新 Pod 出现时，根据资源请求、亲和性、污点容忍等规则选择节点。

因此一次请求的大致路径是：

外部请求 -> Ingress Controller -> Service -> 某个健康 Pod -> 节点上的容器进程

只要这条链路清楚，Pod 替换、节点增减都不会要求客户端改地址。

### 2. HPA 的计算公式

HPA 是 Horizontal Pod Autoscaler，意思是“横向 Pod 自动扩缩容”，即通过增减 Pod 个数分摊负载。

它的核心计算可写成：

$$
desiredReplicas = \left\lceil currentReplicas \times \frac{currentMetricValue}{desiredMetricValue} \right\rceil
$$

玩具例子：  
当前有 3 个副本，平均 CPU 使用率是 90%，目标值是 75%，则

$$
\left\lceil 3 \times \frac{90}{75} \right\rceil
=
\left\lceil 3.6 \right\rceil
= 4
$$

所以 HPA 会建议把副本数扩到 4。

如果同时配置多个指标，比如 CPU 和请求队列长度，HPA 会分别算出目标副本数，取其中最大的那个。这么做的理由很直接：任何一个瓶颈先爆，都值得扩容。

缩容比扩容更敏感，因为如果刚降副本，流量又回升，就会来回抖动。Kubernetes 默认对缩容使用稳定窗口，常见默认值是 300 秒。白话说，控制器会回看最近一段时间的建议值，避免因为一次短暂回落就立刻缩容。

### 3. VPA 与 CA 的职责

VPA 是 Vertical Pod Autoscaler，意思是“纵向 Pod 自动扩缩容”，即不改副本数，而是改每个 Pod 的资源请求与限制。它更像“给单个工人换更大的工位”。

CA 是 Cluster Autoscaler，意思是“集群自动扩缩容”，即当 Pod 因资源不足调度不上去时，自动增加节点；当节点长期空闲时，再尝试回收节点。

一个典型链路是：

1. 流量上涨，HPA 决定把 Pod 从 4 个扩到 8 个。
2. 调度器发现只有 5 个 Pod 能放下，另外 3 个处于 `Pending`。
3. CA 看到有放不下的 Pod，给节点池加机器。
4. 新节点加入后，剩余 Pod 被调度成功。

### 4. 指标为什么不能只看 CPU

| 指标 | 优点 | 缺点 | 适合场景 |
| --- | --- | --- | --- |
| CPU | 易获取、配置简单 | 对 IO 等待和队列积压不敏感 | 普通 Web、无状态 API |
| 内存 | 能发现缓存型或泄漏型问题 | 很多语言不会及时释放内存，缩容不灵敏 | Java、缓存服务 |
| 请求量/QPS | 贴近业务流量 | 不等于处理压力，单请求成本可能差异很大 | API 网关、均匀请求服务 |
| 队列长度 | 对积压很敏感 | 需要自定义指标链路 | 消费者、推理、异步任务 |
| 延迟 | 直接反映用户体验 | 容易受外部依赖影响，调参较难 | 严格 SLA 服务 |
| 批大小 | 能捕捉推理吞吐边界 | 通用性差 | GPU/LLM 推理 |

真实工程里，指标选择常遵循一个顺序：先选最接近瓶颈的指标，再考虑实现复杂度。CPU 最容易，但不总是最对。

---

## 代码实现

下面给一个最小可用示例：`Deployment + Service + Ingress + HPA`。它表达的是最常见的生产结构，而不是最复杂的配置。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: 200m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
---
apiVersion: v1
kind: Service
metadata:
  name: frontend
spec:
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: frontend-ingress
spec:
  ingressClassName: nginx
  rules:
  - host: demo.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: frontend
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: frontend
  minReplicas: 1
  maxReplicas: 10
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 75
```

应用与观察命令：

```bash
kubectl apply -f frontend.yaml
kubectl get ingress,svc,deploy,hpa
kubectl describe hpa frontend
kubectl top pods
```

如果你只是跟教程走，`autoscaling/v1` 也能用，但它主要面向 CPU；新建 HPA 更推荐 `autoscaling/v2`，因为可以同时接 CPU、内存、自定义指标和外部指标。

下面用一个可运行的 Python 片段模拟 HPA 公式：

```python
import math

def desired_replicas(current_replicas: int, current_metric: float, target_metric: float) -> int:
    assert current_replicas > 0
    assert current_metric >= 0
    assert target_metric > 0
    return math.ceil(current_replicas * current_metric / target_metric)

# 玩具例子：3 个副本，CPU 平均 90%，目标 75%
assert desired_replicas(3, 90, 75) == 4

# 负载下降时，理论值会下降；实际缩容还会受稳定窗口影响
assert desired_replicas(8, 30, 75) == 4

print("ok")
```

真实工程例子：  
如果你做的是推理服务，可以把 Prometheus 中的队列长度暴露成外部指标，再让 HPA 或 KEDA 驱动扩容。比如某类查询系统会用下面这类 PromQL 聚合最近 2 分钟的在途请求：

```promql
sum(max_over_time(loki_query_scheduler_inflight_requests{namespace="loki-cluster", quantile="0.75"}[2m]))
```

这个指标的意义很直接：最近一段时间真正积压了多少请求。比 CPU 更接近“用户还在等多久”。

---

## 工程权衡与常见坑

| 问题 | 常见原因 | 排查命令/工作流 | 缓解方式 |
| --- | --- | --- | --- |
| HPA 不生效 | 没有 `metrics.k8s.io` 或自定义指标适配器 | `kubectl get apiservices` | 安装 metrics-server 或 Prometheus Adapter |
| Ingress 404 | host/path/backend 写错 | `kubectl describe ingress` | 先确认规则，再确认 Service 名称与端口 |
| Ingress 502 | Pod 未就绪、Service 后端为空 | `kubectl get endpoints` | 配好 readinessProbe，确认 selector |
| HPA 扩了 Pod 但仍 Pending | 节点容量不足 | `kubectl get pods -A | grep Pending` | 开启 CA/Karpenter，预留节点余量 |
| HPA 与 VPA 打架 | 同时基于 CPU 自动调节 | 看 HPA 目标与 VPA policy | 通常让 HPA 管副本，VPA 只给建议或只管内存 |
| 频繁抖动 | 阈值太敏感、冷却时间太短 | 看 HPA events 和 Grafana 曲线 | 配容差、稳定窗口、分级策略 |
| `LoadBalancer` 没有外网 IP | 本地集群/裸金属不支持云 LB | `kubectl get svc` | 用 Ingress Controller、MetalLB 或云环境 |

最常见的误区有三个。

第一，把 HPA 当成性能优化工具。  
HPA 只能增加副本，不会自动修复慢 SQL、错误缓存策略、单请求算法过重这类问题。

第二，把 CPU 当成唯一指标。  
对于 Python worker、Java 服务、消息消费、GPU 推理，这经常会错。CPU 低不代表系统闲，可能只是请求都堵在队列里。

第三，忽略资源请求 `requests`。  
调度器是按 `requests` 放 Pod 的，不是按实际瞬时使用放。如果 `requests` 设得离谱，HPA 和 CA 都会被误导。  
设得太小，Pod 容易被挤爆；设得太大，节点利用率会虚高，扩节点过早。

Prometheus 和 Grafana 在这里的价值不是“画图好看”，而是让你把扩缩容从事后反应变成事前判断。至少应覆盖这些告警：Pod CPU/内存、Pod Pending、节点可分配资源、HPA 目标与当前值、核心队列长度、P95/P99 延迟。

---

## 替代方案与适用边界

Ingress 不是唯一七层入口。若系统需要更复杂的流量治理，例如更细粒度的路由、流量镜像、权重切换，可以考虑 Gateway API 或服务网格。但对初级工程师而言，Ingress 仍是最常见起点。

节点扩容也不是只有 CA。

| 方案 | 特点 | 适用边界 |
| --- | --- | --- |
| Cluster Autoscaler | 依赖预配置节点组，成熟稳定 | 标准云上集群、传统节点池 |
| Karpenter | 支持自动建节点，管理生命周期更灵活 | 云上、实例类型选择复杂的场景 |
| KEDA | 事件驱动扩缩容，直接接队列/外部系统 | 消费者、批处理、推理、日志查询 |

适用场景矩阵可以简化成下面这样：

| 场景 | 首选指标 | 首选扩缩容方式 | 说明 |
| --- | --- | --- | --- |
| 普通无状态 API | CPU + QPS | HPA | 成本低，上手快 |
| 缓存/内存敏感服务 | 内存 + OOM 历史 | VPA + HPA | 注意内存回收行为 |
| 队列消费者 | backlog/lag | KEDA 或 HPA 外部指标 | CPU 往往不敏感 |
| GPU 推理 | 队列长度、batch size、延迟 | HPA/KEDA + CA/Karpenter | 节点启动慢，要提前量 |
| 大规模日志查询 | inflight requests | 事件驱动扩容 | 请求突刺明显 |

真实工程例子：  
GPU 推理集群里，单卡可以稳定处理 8 个并发批次，但流量突刺时请求先堆在 scheduler 队列。此时若只按 CPU 扩容，GPU 节点可能还没打满，HPA 不动作，P95 延迟已经恶化。更合理的做法是：以队列长度或批大小作为触发器先扩 Pod，再由 CA 或 Karpenter 补节点。这样扩容信号更早，延迟更稳。

---

## 参考资料

1. Kubernetes 官方文档：Horizontal Pod Autoscaling  
   https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/

2. Kubernetes 官方文档：Service  
   https://kubernetes.io/docs/concepts/services-networking/service/

3. Kubernetes 官方文档：Ingress  
   https://kubernetes.io/docs/concepts/services-networking/ingress/

4. Kubernetes 官方文档：Node Autoscaling（含 Cluster Autoscaler 与 Karpenter 对比）  
   https://kubernetes.io/docs/concepts/cluster-administration/node-autoscaling/

5. Kubernetes 官方文档：Vertical Pod Autoscaling  
   https://kubernetes.io/docs/concepts/workloads/autoscaling/vertical-pod-autoscale/

6. Kubernetes 官方博客：Kubernetes v1.33 HPA Configurable Tolerance  
   https://kubernetes.io/blog/2025/04/28/kubernetes-v1-33-hpa-configurable-tolerance/

7. Google Cloud 文档：Configuring horizontal Pod autoscaling  
   https://docs.cloud.google.com/kubernetes-engine/docs/how-to/horizontal-pod-autoscaling

8. Google Cloud 文档：Best practices for autoscaling LLM inference workloads with GPUs  
   https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling

9. Grafana 文档：Autoscaling queriers with Prometheus/KEDA  
   https://grafana.com/docs/enterprise-logs/latest/manage/autoscaling_queriers/
