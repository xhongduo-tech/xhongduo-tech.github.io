## 核心结论

Kubernetes 的 GPU 资源分配，核心不是“Pod 申请了几张卡”，而是“这张卡能不能和这个 Pod 的 CPU、内存、节点策略一起成立”。

先记两个最小公式：

- 资源是否够用：$R_g \le A_g$
- 拓扑是否可行：$H = H_{cpu} \cap H_{mem} \cap H_{gpu}$

这里的白话解释是：

- 扩展资源（extended resource）：不是 Kubernetes 内置的 CPU、内存，而是由外部组件注册进来的资源类型，比如 `nvidia.com/gpu`。
- 拓扑（topology）：硬件之间的“物理位置关系”，例如 CPU、内存、GPU 是否在同一个 NUMA 节点，是否需要跨 socket 通信。
- NUMA：非统一内存访问架构，意思是“同一台机器里，不同 CPU 插槽访问内存和设备的距离不同”。

结论可以压缩成一句话：`scheduler 负责选节点，kubelet 负责能不能落地`。

这也是很多初学者第一次遇到 GPU 调度问题时最容易误判的地方。Pod 显示已经被调度到某个节点，不代表它一定能启动；如果 CPU pinning、内存分配、GPU 设备拓扑无法对齐，kubelet 仍然可能拒绝它。

---

## 问题定义与边界

本文讨论的是 Kubernetes 中 GPU 资源如何被申请、调度、绑定和共享。重点是集群资源管理，不是 CUDA 编程，也不是单机驱动安装。

问题可以拆成三层：

| 问题层次 | 负责组件 | 主要关注点 |
|---|---|---|
| 资源层 | `scheduler` | 节点上有没有足够的 GPU 数量 |
| 拓扑层 | `kubelet` + `Topology Manager` | CPU、内存、GPU 是否能在 NUMA 上对齐 |
| 共享层 | `device plugin` / MIG / time-slicing | GPU 是独占还是共享，隔离和抖动如何处理 |

device plugin 的白话解释是：一个把硬件设备“上报给 Kubernetes”的插件。没有它，Kubernetes 不知道节点上有哪些 GPU，也不知道如何把 GPU 分配给容器。

要区分三类看起来相似、但本质不同的问题：

1. GPU 是否够
2. GPU 是否和 CPU/内存拓扑匹配
3. GPU 是独占还是共享

同样写成 `nvidia.com/gpu: 1`，实际含义可能完全不同。

一个玩具例子：

- 节点 A：单 NUMA，16 核 CPU + 1 张 GPU
- 节点 B：双 NUMA，CPU 在 NUMA0，空闲 GPU 在 NUMA1
- Pod 请求：`8 CPU + 1 GPU`

从“GPU 数量”看，A 和 B 都满足；从“真正运行效果”看，A 往往更稳定，B 可能因为跨 NUMA 访问导致吞吐下降、延迟升高。也就是说，申请成功不等于性能合理。

本文默认以下边界：

- 讨论 Linux 节点上的 NVIDIA GPU 场景
- 讨论 Kubernetes 原生资源模型和常见 NVIDIA 方案
- 不展开多机训练中的网络层细节，只在需要时提到 NCCL 通信影响

---

## 核心机制与推导

GPU 分配可以按三个判定层次理解：资源可用性、拓扑可行性、策略判定。

### 1. 资源可用性

最粗粒度的判断是：

$$
R_g \le A_g
$$

其中：

- $R_g$：Pod 请求的 GPU 数
- $A_g$：节点当前可分配的 GPU 数

如果 Pod 请求 `nvidia.com/gpu: 1`，而节点上还有 1 张可用卡，scheduler 会把这个节点视为候选节点。注意，这一步只回答“数量上是否可能”，不回答“拓扑上是否合理”。

### 2. 拓扑可行性

真正复杂的是第二层。Topology Manager 会收集多个 Hint Provider 给出的 NUMA 候选集合，再做交集：

$$
H = H_{cpu} \cap H_{mem} \cap H_{gpu}
$$

白话解释：

- `H_cpu`：CPU Manager 认为这个 Pod 的 CPU 可以落在哪些 NUMA 节点
- `H_mem`：内存分配偏好对应哪些 NUMA 节点
- `H_gpu`：设备插件报告这块 GPU 属于哪些 NUMA 节点

如果交集为空，说明“CPU、内存、GPU 没有共同的物理落点”，那就要看策略是否允许放行。

### 3. 策略判定

Topology Manager 常见策略可以简化成下面这张表：

| 策略 | 含义 | 交集为空时 |
|---|---|---|
| `best-effort` | 尽量对齐，但不强制 | 仍可能接纳 |
| `restricted` | 必须满足拓扑约束 | 拒绝 |
| `single-numa-node` | 必须由单个 NUMA 节点独立满足全部需求 | 拒绝，且要求更严格 |

这里的 single NUMA node 可以白话理解成：CPU、内存、GPU 都最好“挤在同一边”，不要跨插槽拼凑。

### 玩具例子：为什么 scheduler 成功，kubelet 仍然失败

假设一个双路服务器有两个 NUMA 节点：

- NUMA0：16 CPU，GPU0
- NUMA1：16 CPU，GPU1

此时：

- GPU0 已被占用
- GPU1 空闲
- NUMA1 上 CPU 被其他 Pod 吃掉大半，只剩 2 核
- NUMA0 上还有 12 核空闲

现在新 Pod 请求：

- `8 CPU`
- `1 GPU`

从 scheduler 视角：

- 节点还有 1 张 GPU
- 节点总 CPU 也够
- 所以节点是候选节点

从 kubelet + Topology Manager 视角：

- `H_cpu = {NUMA0}`，因为只有 NUMA0 能给出 8 个可 pin 的 CPU
- `H_gpu = {NUMA1}`，因为只剩 GPU1
- 则 $H = {NUMA0} \cap {NUMA1} = \varnothing$

结果如下：

| 请求 | `best-effort` | `restricted` | `single-numa-node` |
|---|---|---|---|
| `8 CPU + 1 GPU`，CPU 在 NUMA0，GPU 在 NUMA1 | 可能接纳，但跨 NUMA | 拒绝 | 拒绝 |

所以“调度成功”和“成功启动”是两个阶段，不是一个阶段。

### 真实工程例子：训练和高吞吐推理为什么对拓扑更敏感

真实生产里，模型服务很少只做纯 GPU 计算。常见链路包括：

- CPU 做 tokenizer、图像预处理、batch 拼接
- GPU 做前向计算
- 多卡之间做 NCCL 通信
- 容器还要读本地缓存、共享内存或页缓存

如果 Pod 只声明 `nvidia.com/gpu: 1`，却不关心 CPU 与 GPU 的 NUMA 对齐，就可能出现下面的现象：

- GPU 利用率不低，但整体 QPS 上不去
- P99 延迟显著升高
- 多卡训练时通信开销异常大
- 节点明明“还有卡”，但任务启动失败率很高

根因不是“GPU 不够”，而是“GPU 的位置不对”。

---

## 代码实现

代码层要覆盖两件事：Pod 如何申请 GPU，节点如何启用拓扑约束与共享策略。

### 1. 最小 Pod 示例

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-demo
spec:
  restartPolicy: Never
  containers:
    - name: app
      image: nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0
      resources:
        requests:
          cpu: "4"
          memory: "8Gi"
        limits:
          cpu: "4"
          memory: "8Gi"
          nvidia.com/gpu: "1"
```

这个配置只表达了一层意思：我要 1 张 GPU。它没有表达“我要哪种拓扑”“我要不要和某类节点绑定”。

### 2. 增强版 Pod：加入节点亲和

node affinity 的白话解释是：让 Pod 倾向或强制落到带某些标签的节点上。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-topology-aware
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: feature.node.kubernetes.io/pci-10de.present
                operator: In
                values: ["true"]
              - key: nvidia.com/gpu.product
                operator: In
                values: ["NVIDIA-A100-SXM4-40GB"]
              - key: topology.kubernetes.io/zone
                operator: In
                values: ["cn-shanghai-a"]
  containers:
    - name: infer
      image: your-registry/infer:latest
      resources:
        requests:
          cpu: "8"
          memory: "16Gi"
        limits:
          cpu: "8"
          memory: "16Gi"
          nvidia.com/gpu: "1"
```

这里通常会配合 NFD。NFD 的白话解释是：一个自动给节点打硬件标签的组件，Node Feature Discovery 的缩写。

### 3. kubelet 侧配置：CPU Manager 与 Topology Manager

如果节点要做更严格的 CPU/GPU 对齐，通常需要：

```yaml
cpuManagerPolicy: static
topologyManagerPolicy: restricted
topologyManagerScope: pod
```

核心含义：

- `cpuManagerPolicy: static`：给满足条件的 Pod 分配固定 CPU 核，而不是任意漂移
- `topologyManagerPolicy: restricted`：如果 CPU、内存、设备对不齐，就拒绝
- `topologyManagerScope: pod`：按 Pod 整体做决策，而不是单容器碎片化决策

可以把它理解成如下伪代码：

```text
if H_cpu ∩ H_mem ∩ H_gpu == ∅:
    reject
else:
    admit
```

### 4. 共享 GPU：time-slicing 的基本思路

如果要把 1 张物理 GPU 暴露成多个“共享配额”，常见做法是 time-slicing。它的白话解释是：不同进程轮流占用 GPU 时间片，而不是同时拥有独立硬件切片。

示意配置里的核心思想通常是：

- 1 张卡
- `replicas=4`
- 集群里就会出现 4 个可分配的共享 GPU 资源视图

这意味着最多 4 个 Pod 可以各自拿到“1 个共享 GPU”。但这不代表每个 Pod 固定拥有 25% 算力，也不代表显存和执行时间被严格隔离。

### 5. 一个可运行的 Python 玩具程序

下面用一个最小程序模拟“资源可用”和“拓扑交集”两个判断：

```python
from dataclasses import dataclass

@dataclass
class Request:
    cpu: int
    gpu: int

@dataclass
class NodeState:
    allocatable_gpu: int
    cpu_hints: set
    mem_hints: set
    gpu_hints: set

def can_schedule(req: Request, node: NodeState, policy: str) -> bool:
    if req.gpu > node.allocatable_gpu:
        return False

    h = node.cpu_hints & node.mem_hints & node.gpu_hints

    if policy == "best-effort":
        return True
    if policy in {"restricted", "single-numa-node"}:
        return len(h) > 0
    raise ValueError(f"unknown policy: {policy}")

req = Request(cpu=8, gpu=1)

node_ok = NodeState(
    allocatable_gpu=1,
    cpu_hints={0},
    mem_hints={0, 1},
    gpu_hints={0},
)

node_bad = NodeState(
    allocatable_gpu=1,
    cpu_hints={0},
    mem_hints={0, 1},
    gpu_hints={1},
)

assert can_schedule(req, node_ok, "restricted") is True
assert can_schedule(req, node_bad, "best-effort") is True
assert can_schedule(req, node_bad, "restricted") is False
assert can_schedule(req, node_bad, "single-numa-node") is False
print("all checks passed")
```

这个程序不是 Kubernetes 源码实现，只是把核心逻辑抽象成了最容易验证的形式：数量检查先过，再看拓扑交集，最后看策略是否允许。

---

## 工程权衡与常见坑

GPU 独占、共享、拓扑对齐三件事，不能同时无限制满足。工程里真正要做的是取舍，而不是追求“所有指标都满分”。

### 常见坑

| 常见坑 | 实际问题 | 结果 |
|---|---|---|
| 只看 `nvidia.com/gpu` 数量，不看 NUMA | scheduler 只做粗筛 | Pod 可能拿到最差拓扑 |
| 把共享 GPU 当独占 GPU | time-slicing 没有强隔离 | 尾延迟抖动明显 |
| 以为调度成功就一定能启动 | kubelet 还要做 admit | 节点上被拒绝 |
| 只做型号标签，不做拓扑标签 | 只知道“是什么卡”，不知道“卡在哪” | 性能不稳定 |
| 误以为共享算力线性叠加 | GPU 调度不是平均切蛋糕 | 吞吐和延迟都不可预测 |

### 工程里的三组典型权衡

1. 性能 vs 利用率  
独占 GPU 性能更稳定，但空闲时间可能浪费。共享 GPU 利用率更高，但抖动更大。

2. 隔离 vs 灵活性  
MIG 的硬件隔离更强，但切分方式固定、配置更复杂。time-slicing 更灵活，但隔离弱。

3. 调度简单性 vs 拓扑正确性  
只写 `nvidia.com/gpu: 1` 最简单，但最容易踩坑。加上 NFD、node affinity、CPU Manager、Topology Manager，复杂度会上升，但行为更可控。

### 两个常见故障现场

一个真实训练任务的典型现象：

- Pod 已经被调度到某个 A100 节点
- `kubectl describe pod` 里可以看到 Assigned
- 但容器迟迟起不来，事件里出现拓扑相关 admit 失败
- 排查后发现 CPU 只能从 NUMA0 分配，而剩余 GPU 全在 NUMA1

另一个在线推理任务的典型现象：

- 开启 time-slicing 后，卡利用率明显提升
- 平均延迟变化不大
- 但 P95/P99 延迟抖动明显增大
- 原因是多个 Pod 共享同一张物理 GPU，调度时序和显存竞争不可忽略

这两类问题都说明：GPU 资源管理不是单一维度问题。

---

## 替代方案与适用边界

不要把独占 GPU、time-slicing、MIG 混成一种方案。它们解决的问题不同。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 独占 GPU | 隔离强，性能稳定，问题最容易定位 | 利用率可能偏低 | 训练、关键在线推理 |
| time-slicing | 利用率高，部署简单，适合小负载混部 | 抖动大，隔离弱，性能不可预测 | 离线批处理、开发测试、对尾延迟不敏感的推理 |
| MIG | 硬件级切分，隔离强于 time-slicing | 配置复杂，切分粒度受硬件限制 | 多租户在线推理、稳定小实例服务 |

### 怎么选

如果业务更看重吞吐稳定和隔离，优先考虑：

- 独占 GPU
- 或支持时选择 MIG
- 同时配合严格拓扑对齐

如果业务更看重总体利用率，且能接受抖动，可以考虑：

- time-slicing
- 较弱的拓扑约束
- 更细的资源回收策略

按任务类型看，一般可以这样判断：

- 训练任务：优先独占 GPU，尤其是多卡训练；拓扑不对齐会直接伤害吞吐和通信效率。
- 在线推理任务：如果对延迟稳定性要求高，优先 MIG 或独占 GPU；如果是低优先级服务，才考虑共享。
- 批处理任务：最适合共享 GPU，因为任务通常对单次尾延迟不敏感，更关心总体成本。

适用边界也要说清楚：如果集群调度器本身不知道显存需求、PCIe 拓扑、NVLink 关系，仅靠 `nvidia.com/gpu` 这一级抽象，就无法自动做出最优决策。此时只能通过标签、亲和性、节点池拆分和更严格的准入策略去弥补。

---

## 参考资料

1. [Kubernetes Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
2. [Schedule GPUs in Kubernetes](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
3. [Kubernetes Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/)
4. [Kubernetes Resource Managers](https://kubernetes.io/docs/concepts/workloads/resource-managers/)
5. [NVIDIA GPU Operator - Time-Slicing GPUs](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.9/gpu-sharing.html)
