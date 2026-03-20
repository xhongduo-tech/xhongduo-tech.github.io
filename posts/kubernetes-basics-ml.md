## 核心结论

Kubernetes 基础里最重要的一句是：**Pod 是最小的部署与调度单位**。调度单位的意思是，调度器不会把一个 Pod 拆开分配到两台机器上，而是把整个 Pod 一次性放到某个节点。一个 Pod 里的多个容器共享网络和卷，可以把它理解成“一台逻辑上的小服务器”。

这直接解释了一个初学者常见误区：为什么 LLM 训练通常是多个 Pod，而不是一个 Pod 里塞很多训练容器。原因不是“官方推荐”这么简单，而是 **Pod 不能跨节点拆开**。如果训练任务需要多张 GPU、多个节点协同，正确做法通常是“每个训练副本一个 Pod”，让调度器为每个 Pod 独立找节点、分配 CPU、内存、GPU，再通过网络通信组成分布式训练。

Deployment、Service、资源配额机制是围绕 Pod 工作的三层基础设施：

| 对象 | 白话解释 | 核心职责 | 是否直接影响调度 |
| --- | --- | --- | --- |
| Pod | 一台逻辑小主机 | 运行容器、声明资源请求 | 是 |
| Deployment | Pod 的批量管理员 | 保证副本数、滚动更新 | 间接影响 |
| Service | Pod 的稳定入口 | 负载均衡与服务发现 | 否 |
| ResourceQuota | namespace 级配额门卫 | 限制资源总量、强制声明 requests/limits | 间接影响 |

滚动更新可以用一个简单范围理解：

$$
availableReplicas \in [replicas - maxUnavailable,\ replicas + maxSurge]
$$

比如 `replicas=3, maxUnavailable=1, maxSurge=1`，更新期间可用 Pod 数通常被控制在 2 到 4 之间。`maxSurge` 可以理解成“升级时短期多开一份”，`maxUnavailable` 可以理解成“最多允许少几份”。

---

## 问题定义与边界

本文只讨论 Kubernetes 的基础部署模型，不讨论 Operator、复杂 CNI 网络插件、跨集群调度、Service Mesh，也不展开 GPU 算子、存储编排这类高级主题。边界限定在一个集群内、一个 namespace 内，理解 Pod 如何被调度、Deployment 如何滚动更新、Service 如何暴露访问、ResourceQuota 如何约束资源。

几个基本对象先定清楚：

- **Pod**：容器运行载体，里面可以有一个或多个容器，共享 IP 和卷。
- **Deployment**：无状态应用控制器，负责“应该有几个 Pod、如何升级”。
- **Service**：给一组 Pod 提供稳定访问入口，不直接运行容器。
- **requests/limits**：`requests` 是“至少要给我多少资源”，`limits` 是“最多不准超过多少资源”。
- **ResourceQuota**：namespace 里的总量限制规则，常用来要求所有 Pod 必须声明资源。

玩具例子：一个最小的 `nginx` 网站，可以用 1 个 Deployment 管 3 个 Pod，再用 1 个 ClusterIP Service 暴露给集群内部访问。每个 Pod 声明 `100m CPU` 和 `128Mi` 内存请求，调度器据此选节点。

这个例子里，新手最该记住的不是 YAML 语法，而是职责拆分：**Deployment 管副本，Pod 管资源，Service 管入口**。三者混在一起看会觉得 Kubernetes 很复杂，拆开以后其实很机械。

---

## 核心机制与推导

调度发生在 Pod 层，不发生在 Deployment 层。Deployment 先根据模板创建 Pod，真正决定“放到哪台节点”的是调度器。调度器主要看 `requests`，因为它需要判断某台节点是否还有足够的可分配资源。节点上真正执行和约束容器的是 kubelet，它结合 `limits` 通过 cgroup 做运行时限制。白话说，**scheduler 决定去哪里，kubelet 决定到了以后怎么管**。

可以把资源机制写成一个简化关系：

$$
node\_allocatable \ge \sum requests
$$

调度器关心的是“节点剩余可分配资源是否能容纳这个 Pod 的请求总和”。如果一个 Pod 请求 `cpu: 100m, memory: 128Mi`，那它不是“最多占这么多”，而是“至少得给我留出这么多”。

滚动更新的核心推导也很直接：

$$
availableReplicas = runningPods - unavailablePods
$$

$$
minAvailable = replicas - maxUnavailable
$$

$$
maxTotal = replicas + maxSurge
$$

因此更新过程中，Deployment 会尽量让：

$$
availableReplicas \ge minAvailable,\quad totalPods \le maxTotal
$$

数值例子：`replicas=3, maxUnavailable=1, maxSurge=1`。

- 最少可用副本数：`3 - 1 = 2`
- 最多总 Pod 数：`3 + 1 = 4`

这意味着升级时，系统可以先额外拉起 1 个新 Pod，总数到 4；也允许最多有 1 个旧 Pod 暂时不可用，所以可用副本通常维持在 2 到 4 之间。这里的 `surge` 本质上是“用短期额外资源换更平滑升级”。

真实工程例子：LLM 训练通常不会把 8 个训练进程塞进 1 个 Pod 的 8 个容器里，然后希望它自动扩展到多节点。因为一个 Pod 只能落在一个节点，里面的容器天然共享同一台机器的网络命名空间和本地卷，跨节点资源无法通过“多容器 Pod”获得。更常见的设计是每个训练 worker 一个 Pod，每个 Pod 独立申请 GPU、CPU、内存，再通过 Service 或 Headless Service 做发现与通信。这样某个 Pod 宕机时，只需要重建这个副本，不会把一整个大 Pod 全部拖垮。

---

## 代码实现

下面给一个对新手足够有代表性的实现：Deployment 管 3 个 `nginx` Pod，设置滚动更新参数，并通过 ClusterIP Service 暴露集群内访问。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1   # 更新时最多允许 1 个副本不可用
      maxSurge: 1         # 更新时最多临时多开 1 个 Pod
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: nginx
          image: nginx:1.27
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "100m"     # 调度器据此找节点
              memory: "128Mi"
            limits:
              cpu: "200m"     # kubelet/cgroup 据此限制运行时上限
              memory: "256Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  type: ClusterIP
  selector:
    app: web
  ports:
    - port: 80
      targetPort: 80
```

如果要给集群外访问，才考虑 `NodePort` 或 `LoadBalancer`。基础阶段优先把 `ClusterIP` 理清楚，因为它是默认类型，也是大多数服务之间互调的基础。

下面用一个可运行的 Python 小程序模拟滚动更新边界：

```python
def rolling_window(replicas: int, max_unavailable: int, max_surge: int):
    min_available = replicas - max_unavailable
    max_total = replicas + max_surge
    return min_available, max_total

min_available, max_total = rolling_window(3, 1, 1)

assert min_available == 2
assert max_total == 4

available_replicas = 3 - 1  # runningPods - unavailablePods 的一个时刻
assert available_replicas >= min_available
assert available_replicas <= max_total

print("rolling update window:", min_available, max_total)
```

这个例子不是真实控制器代码，但它把 Deployment 的核心约束翻译成了可以直接执行的逻辑。

---

## 工程权衡与常见坑

最常见的坑有两个：**资源不声明**，以及 **Service 暴露范围选错**。

先看资源问题。在很多团队里，namespace 会配置 ResourceQuota。它的白话含义是：“这个空间不是无限用的，你先把资源需求说清楚，系统才让你进来。” 如果 quota 要求每个 Pod 必须写 `requests/limits`，而你没写，Pod 在创建阶段就可能直接被拒绝，不是运行慢，而是根本创建不出来。

再看 Service 类型：

| Service 类型 | 访问范围 | 典型用途 | 常见坑 |
| --- | --- | --- | --- |
| ClusterIP | 仅集群内部 | 服务间调用 | 误以为外网能直接访问 |
| NodePort | 通过节点 IP:端口 访问 | 简单对外暴露、测试环境 | 端口范围有限，安全面更大 |
| LoadBalancer | 云厂商负载均衡入口 | 生产对外服务 | 依赖云环境，成本更高 |

滚动更新参数也有权衡：

| 参数 | 调大后的效果 | 风险 |
| --- | --- | --- |
| `maxSurge` | 更新更平滑，先起新 Pod | 短时资源占用增加，可能调度失败 |
| `maxUnavailable` | 释放旧 Pod 更快 | 可用副本下降，流量高峰时更危险 |

一个典型真实坑：某个 namespace 设置了 quota，要求所有容器必须声明 CPU 和内存请求。新手复制了一段最简 Pod YAML，没有写资源字段，结果 `kubectl apply` 成功提交后 Pod 一直起不来，事件里提示超出或不满足 quota。这个问题的根因不是镜像错，也不是节点坏，而是“你没有先声明资源”。

---

## 替代方案与适用边界

Deployment 不是唯一控制器，它只是无状态应用的默认解。什么时候该换？

| 对象 | 适合场景 | 为什么不是 Deployment |
| --- | --- | --- |
| Deployment | 无状态 Web、API、普通服务 | 需要副本和滚动更新 |
| StatefulSet | 有稳定身份的服务 | 需要固定网络标识或有序部署 |
| Job | 一次性任务 | 跑完就结束，不追求长期存活 |
| DaemonSet | 每节点一份的代理 | 目标是“每台机器都跑” |

多容器 Pod 也不是越多越好。它只在“必须共享网络/卷，且必须同节点”时才合理。比如主容器写本地日志文件，sidecar 容器实时采集并发送到日志系统，这两个容器放在同一个 Pod 是自然的，因为它们共享卷、生命周期强绑定。

反过来，如果日志处理本身可以独立扩缩容，就不该硬塞进同一个 Pod。把日志组件做成另一个 Deployment，能单独升级、单独限流、单独扩容，耦合更低。对新手来说，可以用一句规则判断：**只有存在强共享和强绑定，才考虑多容器 Pod；普通微服务默认单容器 Pod。**

这也能回到开头的 LLM 训练问题。训练 worker 之间需要的是跨节点通信和独立资源调度，不是“住在同一个 Pod 里”。所以它更接近“多个 Pod 组成一个分布式系统”，而不是“一个 Pod 装下整个系统”。

---

## 参考资料

| 来源 | 链接 | 用途 |
| --- | --- | --- |
| Kubernetes Pods 概览 | https://kubernetes.io/docs/concepts/workloads/pods/pod-overview/ | 理解 Pod 是最小部署与调度单位，共享网络与卷 |
| Deployment 文档 | https://v1-34.docs.kubernetes.io/docs/concepts/workloads/controllers/deployment/ | 理解副本控制、滚动更新、`maxUnavailable` 与 `maxSurge` |
| Service 文档 | https://kubernetes.io/docs/concepts/services-networking/service/ | 理解 ClusterIP、NodePort、LoadBalancer 的流量入口差异 |
| 资源管理文档 | https://v1-32.docs.kubernetes.io/docs/concepts/configuration/manage-resources-containers/ | 理解 `requests/limits`、调度与运行时约束 |
| ResourceQuota 文档 | https://kubernetes.io/docs/concepts/policy/resource-quotas/ | 理解 namespace 配额如何影响 Pod 创建 |

建议阅读顺序也应当是基础优先：先看 Pod 概览，理解“逻辑主机”和调度单位；再看资源管理，理解为什么必须声明 `requests/limits`；然后看 Deployment，理解副本和滚动更新；最后再看 Service，理解服务发现和访问路径。这样不会一开始就被控制器和网络概念混在一起压垮。
