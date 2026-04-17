## 核心结论

Kubernetes 调度策略可以先记成一句话：**先过滤，再打分**。过滤阶段回答“这个 Pod 能不能放到这个节点上”，打分阶段回答“如果能放，放到哪个节点更合适”。

调度器是 Kubernetes 中负责给 Pod 选节点的组件。它不会直接看容器当前瞬时使用了多少 CPU、内存，而是优先依据 Pod 声明的 `requests`、节点约束、污点容忍、亲和性等信息做决策。这里的 `requests` 可以理解为“调度时承诺要预留的资源量”。

一个最小玩具例子：

- Pod 请求 `1 CPU + 1Gi Memory`
- 节点 A 剩余 `0.5 CPU + 8Gi`
- 节点 B 剩余 `2 CPU + 2Gi`
- 节点 C 剩余 `4 CPU + 16Gi`，但带有 Pod 不容忍的污点

过滤后，A 因 CPU 不够被排除，C 因污点不匹配被排除，只剩 B。此时即使 C 更“空”，也不会进入打分阶段。

Filter 与 Score 的职责可以先用表格固定下来：

| 阶段 | 主要输入 | 关注问题 | 输出 |
|---|---|---|---|
| Filter | Pod 的 requests、端口、亲和规则、容忍、卷约束 | 这个节点是否满足硬约束 | feasible nodes，可行节点集合 |
| Score | 通过过滤的节点、资源占比、拓扑分布、镜像本地性 | 多个可行节点里谁更优 | 每个节点的分数 |
| Bind | 最高分节点 | 把 Pod 绑定到最终节点 | Pod 与 Node 的绑定结果 |

结论层面最容易被忽略的点有两个：

1. **调度依据是 `requests`，不是 `limits`。**
2. **`required` 型亲和/反亲和会显著压缩可行节点集合，资源紧张时容易让 Pod 长期 Pending。**

---

## 问题定义与边界

调度器要解决的问题不是“哪台机器现在最空”，而是：

> 在当前所有节点中，找出同时满足 Pod 硬约束的节点集合，再从中选出综合策略下最优的一个。

硬约束就是“不满足就绝不能放”的条件，例如：

- CPU、内存、扩展资源的 `requests`
- `nodeSelector` 与 `nodeAffinity.required...`
- `taints` 与 `tolerations`
- 容器端口冲突
- 卷挂载能力与拓扑限制

软约束就是“尽量满足，但不保证”的条件，例如：

- `preferredDuringScheduling...`
- `topologySpreadConstraints` 的倾向性分布
- 资源均衡或资源打包
- 镜像本地性

形式化地写，可行节点集合是：

$$
feasible\_nodes=\{n \mid \forall f \in Filters,\; f(Pod,n)=true\}
$$

这条式子的意思很直接：只有当一个节点通过全部过滤插件检查时，它才会进入候选集合。

常见过滤项可以先按类别理解：

| 过滤插件/规则 | 检查项 | 失败结果 |
|---|---|---|
| `NodeResourcesFit` / `PodFitsResources` 语义 | 节点剩余资源是否覆盖 Pod 的 `requests` | 节点直接淘汰 |
| `TaintToleration` | Pod 是否容忍节点污点 | 节点直接淘汰 |
| `NodeAffinity` | 节点标签是否满足 required 亲和规则 | 节点直接淘汰 |
| `NodePorts` | 宿主机端口是否冲突 | 节点直接淘汰 |
| `VolumeBinding` / 卷拓扑 | 存储卷能否在该节点或该拓扑域挂载 | 节点直接淘汰 |
| `InterPodAffinity` | 是否满足 Pod 间亲和/反亲和硬约束 | 节点直接淘汰 |

很多初学者把调度理解成“按 CPU 空闲排序”，这是不完整的。更准确的理解是：

- Filter 决定“能不能跑”
- Score 决定“跑在哪里更划算”

如果一个节点在 Filter 已经失败，它再空也没用。

---

## 核心机制与推导

打分阶段不是单一算法，而是多个评分插件共同工作。插件可以理解为“不同角度的评分器”。常见角度包括：

- 资源利用率
- Pod 拓扑分布
- 节点亲和偏好
- Pod 间亲和/反亲和偏好
- 镜像本地性

以 `NodeResourcesFit` 为例，它会按资源维度计算利用率，再映射为得分。核心形式可以写成：

$$
utilization_i=\frac{used_i+requested_i}{allocatable_i}
$$

$$
score_i = resourceScoringFunction(utilization_i)
$$

如果同一个节点要综合 CPU、内存、扩展资源三类得分，则总分可近似表示为：

$$
node\_score=\frac{\sum_i weight_i \times score_i}{\sum_i weight_i}
$$

这里的 `weight` 就是权重，意思是“这个维度的重要性有多高”。

看一个官方文档常见的玩具例子。某节点资源如下：

- CPU 可分配 8，当前已用 1，Pod 请求 2
- Memory 可分配 1Gi，当前已用 256Mi，Pod 请求 256Mi
- `intel.com/foo` 可分配 4，当前已用 1，Pod 请求 2

假设打分后得到：

| 资源 | 单项得分 | 权重 |
|---|---:|---:|
| CPU | 3 | 3 |
| Memory | 5 | 1 |
| `intel.com/foo` | 7 | 5 |

则：

$$
node\_score=\frac{3\times3+5\times1+7\times5}{3+1+5}=\frac{49}{9}\approx 5
$$

这说明节点最终分数不是某一个维度单独决定，而是加权综合的结果。

再看一个真实工程例子。某服务要求高可用，希望 3 个副本尽量分散到不同可用区：

- 集群只有 2 个 AZ
- Deployment 副本数为 3
- 配置了 `requiredDuringSchedulingIgnoredDuringExecution` 的 zone 级反亲和

结果会是：

- 前 2 个 Pod 可以分别落到两个 AZ
- 第 3 个 Pod 因为找不到满足“不能与前两个副本同 AZ”的节点而 Pending

这不是资源不足，而是**硬约束把 feasible nodes 压成了空集**。所以高可用策略不是“约束越严格越好”，而是“可靠性收益与可调度性损失之间的平衡”。

---

## 代码实现

kube-scheduler 基于调度框架工作，不同阶段挂不同插件。可以把主流程抽象成下面的伪代码：

```go
feasible := filterPlugins.Run(pod, allNodes)
if len(feasible) == 0 {
    return Unschedulable
}

scored := scorePlugins.Run(pod, feasible)
target := selectMax(scored)

reservePlugins.Run(pod, target)
permitPlugins.Run(pod, target)
bind(pod, target)
```

其中：

- `Filter` 负责删掉不满足硬约束的节点
- `Score` 负责给剩余节点排名
- `Reserve` 负责在绑定前做预留
- `Permit` 允许插件延迟或拒绝最终绑定
- `Bind` 负责把 Pod 和目标节点写回 API Server

下面用一个可运行的 Python 小程序模拟“过滤 + 加权打分”：

```python
from dataclasses import dataclass

@dataclass
class Node:
    name: str
    cpu_free: float
    mem_free: float
    tainted: bool
    affinity_match: bool
    cpu_score: int
    mem_score: int

def filter_nodes(nodes, req_cpu, req_mem, tolerate_taint=False):
    feasible = []
    for n in nodes:
        if n.cpu_free < req_cpu:
            continue
        if n.mem_free < req_mem:
            continue
        if n.tainted and not tolerate_taint:
            continue
        if not n.affinity_match:
            continue
        feasible.append(n)
    return feasible

def final_score(node, cpu_weight=3, mem_weight=1):
    total = node.cpu_score * cpu_weight + node.mem_score * mem_weight
    return total / (cpu_weight + mem_weight)

nodes = [
    Node("node-a", cpu_free=0.5, mem_free=8, tainted=False, affinity_match=True, cpu_score=9, mem_score=9),
    Node("node-b", cpu_free=2, mem_free=2, tainted=False, affinity_match=True, cpu_score=7, mem_score=6),
    Node("node-c", cpu_free=4, mem_free=16, tainted=True, affinity_match=True, cpu_score=10, mem_score=10),
]

feasible = filter_nodes(nodes, req_cpu=1, req_mem=1, tolerate_taint=False)
assert [n.name for n in feasible] == ["node-b"]

best = max(feasible, key=final_score)
assert best.name == "node-b"
assert final_score(best) == (7 * 3 + 6 * 1) / 4
```

这段代码虽然简化了真实实现，但保留了调度本质：**先做布尔筛选，再做数值排序**。

`scheduler-config` 中常见的默认链路可概括为：

| 阶段 | 常见默认插件 | 作用 |
|---|---|---|
| `Filter` | `NodeResourcesFit`、`NodePorts`、`TaintToleration`、`NodeAffinity`、`InterPodAffinity`、`PodTopologySpread`、`VolumeBinding` | 检查资源、端口、污点、亲和、拓扑、存储等硬约束 |
| `Score` | `NodeResourcesFit`、`NodeAffinity`、`TaintToleration`、`InterPodAffinity`、`PodTopologySpread`、`ImageLocality`、`VolumeBinding` | 对可行节点按资源、拓扑、镜像等维度排序 |
| `Reserve` | `VolumeBinding` 等 | 在绑定前预留相关资源 |
| `Permit` | 默认通常为空或按插件需要启用 | 控制是否允许继续绑定 |

这里要注意：默认插件集会随 Kubernetes 版本变化，上表适合用来理解链路，不应当当成“所有版本一字不差的清单”。

---

## 工程权衡与常见坑

最常见的误区是把 `limits` 当成调度依据。实际上，调度器主要看 `requests`。这会带来两个方向完全相反的问题：

| 常见坑 | 结果 | 防御措施 |
|---|---|---|
| `requests` 配得太低 | 调度器误以为空闲，节点被过量放 Pod，运行期争抢严重 | 用压测或历史监控校准 `requests` |
| `requests` 配得太高 | Pod 调不进去，节点明明还有实时余量却被判定不足 | 按 P50/P95 负载分层配置，不要机械放大 |
| 只写 `limits` 不写 `requests` | 行为依赖默认推导，容易让调度与预期不一致 | 显式同时定义 `requests` 与 `limits` |
| 过度使用 `required` 反亲和 | 可调度节点骤减，Pod 长期 Pending | 先尝试 `preferred` 或 `topologySpreadConstraints` |
| 亲和表达式过于复杂 | 大集群中调度延迟上升 | 缩小 selector 范围，减少高复杂度规则 |
| 把高可用完全交给硬反亲和 | 在 AZ 少、节点少时容易失效 | 用“柔性分散 + 副本规划 + 自动扩容”组合 |

真实工程里最典型的坑，就是“能跑”和“能稳”混在一起。举例：

- 某在线服务平时只吃 200m CPU，但高峰能冲到 1 核
- 团队把 `requests` 长期写成 `100m`
- 调度器会认为很多节点都能再塞进去
- 结果不是 Pending，而是节点上几十个 Pod 在高峰同时抢 CPU，延迟暴涨

这类问题本质上不是 kube-scheduler 算错了，而是输入给调度器的资源画像不真实。

下面是一个 Deployment 片段，同时展示 `requests`/`limits`、required 反亲和和 preferred 反亲和：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout
spec:
  replicas: 3
  selector:
    matchLabels:
      app: checkout
  template:
    metadata:
      labels:
        app: checkout
    spec:
      containers:
      - name: app
        image: nginx:1.27
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "1Gi"
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app: checkout
            topologyKey: topology.kubernetes.io/zone
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 80
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: checkout
              topologyKey: kubernetes.io/hostname
```

这段配置的含义是：

- 强制不同副本尽量不落在同一可用区
- 在满足上面前提时，再尽量不要落到同一台主机
- 如果集群只有 2 个 AZ，却要起 3 个副本，第 3 个副本可能永远 Pending

---

## 替代方案与适用边界

并不是所有高可用需求都应该直接上硬反亲和。更常见、更稳妥的选择是按场景分层。

常见评分或放置策略可以这样理解：

| 策略 | 核心倾向 | 推荐场景 |
|---|---|---|
| `LeastAllocated` | 优先放到更空闲的节点 | 追求负载分散、降低单机热点 |
| `MostAllocated` | 优先把 Pod 往已使用率高的节点集中 | 希望做资源打包，给集群自动扩容和缩容创造空间 |
| `BalancedAllocation` | 让 CPU、内存利用率更均衡 | 避免只打满单一资源造成碎片化 |
| `RequestedToCapacityRatio` | 按自定义利用率曲线评分 | 资源形态复杂、需要细调偏好的场景 |

如果目标是“尽量分布，但不要因为约束太硬而调不进去”，优先顺序一般是：

| 方案 | 适合什么问题 | 边界 |
|---|---|---|
| `preferred` 亲和/反亲和 | 希望有放置倾向，但允许退化 | 不能保证绝对隔离 |
| `topologySpreadConstraints` | 希望副本在 zone / node 间更均匀 | 表达“分散”强，表达“必须不共存”弱 |
| `required` 亲和/反亲和 | 强隔离、强合规、强依赖约束 | 容易导致不可调度 |

一个实际判断标准：

- 你要的是“硬隔离”，用 `required`
- 你要的是“尽量均匀”，优先 `topologySpreadConstraints`
- 你要的是“有偏好但不能阻塞上线”，用 `preferred`

例如只有 2 个 AZ，但服务需要 3 副本时，更稳妥的做法通常不是 zone 级 `required` 反亲和，而是：

- 用 `preferred` anti-affinity 表达“优先跨 AZ”
- 再配 `topologySpreadConstraints` 控制尽量均衡
- 结合 `Cluster Autoscaler` 在资源紧张时扩容节点池

这样做的好处是：当理想分布达不到时，系统仍然可以退化运行，而不是直接进入不可调度状态。

---

## 参考资料

下表列的是写这类文章时最应该反复核对的官方资料：

| 文档 | 作用 |
|---|---|
| [Kubernetes Scheduler](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/) | 核对 Filter/Score 两阶段调度流程 |
| [Assigning Pods to Nodes](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/) | 核对 nodeSelector、node affinity、pod affinity/anti-affinity 与性能边界 |
| [Resource Bin Packing](https://kubernetes.io/docs/concepts/scheduling-eviction/resource-bin-packing/) | 核对 `NodeResourcesFit` 的评分策略与权重机制 |
| [kube-scheduler Configuration (v1)](https://kubernetes.io/docs/reference/config-api/kube-scheduler-config.v1/) | 核对 `profiles`、插件阶段、`NodeResourcesFitArgs`、`PodTopologySpreadArgs` |
| [Kubernetes Best Practices: Resource Requests and Limits](https://cloud.google.com/blog/products/containers-kubernetes/kubernetes-best-practices-resource-requests-and-limits) | 校验 `requests` 与 `limits` 的工程实践差异 |

进一步阅读时，建议重点看三部分：

- `Kubernetes Scheduler`：理解“先过滤、后打分”的主流程
- `Resource Bin Packing`：理解资源评分不是简单看空闲量，而是看配置的打分曲线
- `Assigning Pods to Nodes`：理解亲和/反亲和与拓扑分布规则为什么会影响可调度性和性能
