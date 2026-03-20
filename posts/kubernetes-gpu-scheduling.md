## 核心结论

Kubernetes 自身并不认识“GPU 是什么”，它只认识“节点上有没有一种可分配的扩展资源”。GPU 能被调度，前提是 **Device Plugin（设备插件，白话说就是把硬件能力上报给 kubelet 的组件）** 先把 `nvidia.com/gpu` 这类资源注册进去。注册完成后，kubelet 会把可用 GPU 数量写进节点状态，调度器再按 Pod 的资源请求做分配。

这解决的是“看见 GPU”问题，不解决“多 Pod 训练一起起”问题。多节点训练如 PyTorch DDP、Horovod，往往要求所有 worker 同时上线，否则通信初始化或 `allreduce` 会阻塞。**Volcano 的 gang scheduling（成组调度，白话说就是一组 Pod 要么一起放行，要么一起等待）** 用 `PodGroup.minMember` 或 `Job.minAvailable` 定义最小启动规模，只有资源一次性满足时才真正调度。

下面这张表先把两类能力分开：

| 组件 | 解决的问题 | 关注点 | 不解决什么 |
|---|---|---|---|
| Device Plugin | 把 GPU 暴露成 Kubernetes 可分配资源 | 资源注册、设备状态上报、故障后重注册 | 多 Pod 是否同时启动 |
| 原生 scheduler + affinity | 在满足约束的节点中选节点 | 资源匹配、节点标签、Pod 拓扑关系 | 分布式训练的“整组放行” |
| Volcano gang scheduling | 多个 Pod 作为一个整体等待和启动 | `minMember`、`minAvailable`、批量放行 | GPU 设备本身如何被发现 |

玩具例子可以这样理解：某 GPU 节点上跑了 NVIDIA Device Plugin，它告诉 kubelet“我这里有 2 张 GPU”；一个 Pod 写了 `limits.nvidia.com/gpu: 1`，调度器就能把它放到该节点。真实工程例子是：一个 4 worker 的 DDP 训练任务，如果集群此刻只有 3 张空闲 GPU，Volcano 不会先启动 3 个 worker，而是让 4 个都保持 Pending，直到第 4 张 GPU 出现。

---

## 问题定义与边界

先把问题拆成两层。

第一层是 **设备可见性**。kubelet 默认只知道 CPU、内存、临时存储等内建资源，不知道 NVIDIA GPU、FPGA 或其他加速卡。要让调度器识别 `nvidia.com/gpu`，必须先有设备插件向 kubelet 注册。没有这个步骤，Pod 即使写了 GPU 请求，也只会因为“资源不存在”而无法正确调度。

第二层是 **组任务一致性**。单个 Pod 调度成功，不代表整个训练任务能跑起来。分布式训练通常需要固定数量的 worker 全部就绪后才能进入训练阶段。可以用一个简化公式描述：

$$
\text{Gang Ready} \Leftrightarrow \text{scheduled} \ge \text{minMember}
$$

也就是，只有已经调度成功的 Pod 数量达到最小成员数，才允许整组进入运行态。否则继续等待。

Device Plugin 的基本注册流程如下：

| 阶段 | 发生什么 | 结果 |
|---|---|---|
| 启动 | 插件进程通常以 DaemonSet 方式在每个 GPU 节点运行 | 每个节点各自探测本机 GPU |
| gRPC 注册 | 插件向 kubelet 的设备插件接口注册资源名，如 `nvidia.com/gpu` | kubelet 知道有一种新资源 |
| 状态上报 | 插件持续通过 `ListAndWatch` 上报设备列表与健康状态 | kubelet 持续维护可用数量 |
| 节点状态更新 | kubelet 将资源容量写入 Node status | scheduler 可见 |
| Pod 请求 | Pod 写 `limits.nvidia.com/gpu` | scheduler 才能按数量匹配 |

新手容易忽略一个边界：**`nodeSelector` 和 affinity 不是二选一，而是叠加约束**。如果你写了 `nodeSelector: gpu=true`，又写了 `nodeAffinity` 要求 `region=us-1`，那么节点必须同时满足这两个条件。资源够但标签不匹配，Pod 仍然 Pending。

再看一个真实工程边界：4 个 GPU worker 的训练任务，在集群只剩 3 张空闲 GPU 时，原生调度器可能先让 3 个 Pod 跑起来、1 个继续等；Volcano 的 gang scheduling 则会选择“一个都不跑”，因为部分启动对训练并没有实际价值。

---

## 核心机制与推导

### 1. Device Plugin 如何让 kubelet 认识 GPU

Device Plugin 本质上是一个遵循 Kubernetes 设备插件协议的 gRPC 服务。它至少要做两件事：

1. 向 kubelet 注册资源名，比如 `nvidia.com/gpu`
2. 持续汇报设备列表与设备健康状态

这里的 `vendor-domain/resource` 是固定格式，白话说就是“厂商域名 + 资源类型”。例如 NVIDIA 用 `nvidia.com/gpu`，这样可以避免不同厂商资源名冲突。

一个关键机制是 `ListAndWatch`。kubelet 不是只在启动时读一次 GPU 数量，而是长期订阅设备状态。设备损坏、驱动异常、插件重启，都会通过这个通道反映到节点资源视图。一个简化伪代码如下：

```python
from dataclasses import dataclass

@dataclass
class Device:
    id: str
    health: str  # "Healthy" or "Unhealthy"

class FakeKubelet:
    def __init__(self):
        self.resources = {}

    def register(self, resource_name: str):
        self.resources[resource_name] = []

    def update_devices(self, resource_name: str, devices: list[Device]):
        healthy = [d for d in devices if d.health == "Healthy"]
        self.resources[resource_name] = healthy

def list_and_watch(kubelet: FakeKubelet, resource_name: str, snapshots: list[list[Device]]):
    kubelet.register(resource_name)
    for devices in snapshots:
        kubelet.update_devices(resource_name, devices)

k = FakeKubelet()
snapshots = [
    [Device("gpu0", "Healthy"), Device("gpu1", "Healthy")],
    [Device("gpu0", "Healthy"), Device("gpu1", "Unhealthy")],
]
list_and_watch(k, "nvidia.com/gpu", snapshots)

assert len(k.resources["nvidia.com/gpu"]) == 1
assert k.resources["nvidia.com/gpu"][0].id == "gpu0"
```

这个玩具例子表达的结论很直接：最终可分配 GPU 数量不是“机器上插了几张卡”，而是“插件持续上报后仍然健康的设备数”。

更深入一点，设备插件通常要盯住 `/var/lib/kubelet/device-plugins/`。原因是 kubelet 重启时，会清理并重建这里的 socket。插件如果没有感知到 socket 消失并重新注册，kubelet 会认为这个插件不存在，节点上的 GPU 容量可能瞬间变成 0。这也是为什么 NVIDIA Device Plugin 一般以 DaemonSet 常驻运行，而不是“一次启动就算完成”。

### 2. 亲和性如何约束 GPU Pod 放到哪里

Kubernetes 调度不是只看 GPU 数量，还要看约束条件。几个常见术语：

| 机制 | 白话解释 | 常见用途 |
|---|---|---|
| `nodeSelector` | 最简单的节点标签精确匹配 | 只放到 GPU 节点 |
| `nodeAffinity` | 更灵活的节点匹配规则，可表达“必须”或“尽量” | 指定机型、可用区、GPU 代际 |
| `podAffinity` | 希望和某些 Pod 靠近 | worker 靠近参数服务或缓存服务 |
| `podAntiAffinity` | 希望和某些 Pod 分开 | 避免多个训练副本抢同一节点 |

新手例子：把所有 GPU 节点打上 `accelerator=nvidia`，然后 Pod 要求这个标签。这样 CPU-only 节点不会被考虑。真实工程例子：A100 训练任务要求 `gpu.arch=ampere` 且 `topology.kubernetes.io/zone=cn-shanghai-b`，同时还希望多个 worker 分散到不同主机，避免单机故障拖垮整个作业。

### 3. Volcano 为什么能解决“部分启动”问题

Volcano 在调度层引入了 PodGroup 或 Job 概念。它不再把每个 Pod 当成完全独立的个体，而是先看“这一组至少要几个成员才能有意义”。

常见字段如下：

| 字段 | 含义 | 行为影响 |
|---|---|---|
| `minMember` | PodGroup 最少成员数 | 不够就不放行 |
| `minAvailable` | Job 至少可运行的 Pod 数 | 常用于批任务 |
| `schedulingPolicy` | 调度策略 | 影响队列、公平性、预占 |
| `preemptable` | 是否允许被更高优先级作业抢占 | 影响资源回收 |

它的核心判断也可以写成：

$$
\text{Pod Pending} \Leftrightarrow \text{scheduled} < \text{minMember}
$$

这意味着 Pending 不一定是坏事。在分布式训练里，Pending 反而是正确保护机制，因为“先跑一半”通常比“全部等待”更糟。

真实工程例子：一个 8 卡 DDP 作业，要求 8 个 worker 都拿到 GPU 并完成 rendezvous（集合初始化，白话说就是所有训练进程先互相找到彼此）。若只启动 6 个，程序常见结果不是“先训练起来”，而是卡在通信初始化阶段。Volcano 用 `group-min-member: "8"` 可以直接把这种无效启动挡在调度层。

---

## 代码实现

先看最基础的 GPU Pod 写法。对扩展设备资源，实际工程里通常只写 `limits`，Kubernetes 会把它视为请求值。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-demo
spec:
  nodeSelector:
    accelerator: nvidia
  containers:
    - name: trainer
      image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
      command: ["python", "-c", "print('train')"]
      resources:
        limits:
          nvidia.com/gpu: "1"
```

`requests` 与 `limits` 在 GPU 场景里的常见写法对比如下：

| 写法 | 是否常见 | 说明 |
|---|---|---|
| 只写 `limits.nvidia.com/gpu: 1` | 是 | 对扩展资源最常见，调度器按 1 张 GPU 计算 |
| 同时写 `requests` 和 `limits` 且值相同 | 可行 | 表达更显式，但通常不是必须 |
| `requests=1, limits=2` | 不推荐 | 扩展资源通常不支持这种弹性语义 |
| 不写 GPU 资源，只写标签亲和性 | 错误设计 | 只能放到 GPU 节点，不代表真正占到 GPU |

下面是一个带 Volcano 的最小示例。它表达的是“4 个成员必须同时满足，才启动这组训练 Pod”。

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: ddp-workers
spec:
  minMember: 4
---
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: ddp-job
spec:
  minAvailable: 4
  schedulerName: volcano
  tasks:
    - name: worker
      replicas: 4
      template:
        metadata:
          annotations:
            scheduling.volcano.sh/group-min-member: "4"
        spec:
          restartPolicy: Never
          nodeSelector:
            accelerator: nvidia
          containers:
            - name: trainer
              image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
              command: ["torchrun", "--nproc_per_node=1", "train.py"]
              resources:
                limits:
                  nvidia.com/gpu: "1"
```

玩具例子：4 个副本，每个副本要 1 张 GPU，总共需要 4 张。若集群现有 2 张空闲卡，这 4 个 Pod 会一起等待。真实工程例子：多机多卡训练里把 master 和 worker 都纳入同一组，确保所有通信参与者一起被放行，而不是 master 先启动后一直超时等待 worker。

---

## 工程权衡与常见坑

最常见的误区不是“不会写 YAML”，而是对系统边界理解错了。

| 常见坑 | 现象 | 根因 | 规避方式 |
|---|---|---|---|
| kubelet 重启后 GPU 变成 0 | 节点突然不再可调度 GPU Pod | 设备插件未重新注册 | DaemonSet 常驻运行，监听 socket 删除并重连 |
| 只写 `nodeSelector` 不写 GPU 资源 | Pod 跑在 GPU 节点但没真正占卡 | 标签约束不等于资源申请 | 必须写 `limits.nvidia.com/gpu` |
| `nodeSelector` 与 `nodeAffinity` 冲突 | Pod 永远 Pending | 两类规则都必须满足 | 统一标签体系，先验证节点标签 |
| 不用 gang scheduling 跑 DDP | 训练任务部分副本启动后卡死 | 分布式任务需要整组就绪 | 用 Volcano `minMember/minAvailable` |
| gang 数值设置过大 | 作业长期 Pending | 最小成员数超出当前集群承载 | 结合队列容量和弹性策略设置 |
| 忽略 GPU 异构 | 任务性能不稳定甚至报错 | 不同节点 GPU 型号、显存不一致 | 用 `nodeAffinity` 限定 GPU 代际和规格 |

这里有一个重要权衡：Volcano 提高了分布式训练的一致性，但会降低短期资源利用率。因为它宁愿等待整组资源齐全，也不愿部分启动。对于在线推理或单卡离线任务，这种等待可能没有必要；对于需要强同步的训练任务，这种等待往往是必须的。

另一个真实工程坑是 **资源碎片化**。假设 8 张 GPU 分散在多个节点上，而任务要求 4 个 Pod 且每个 Pod 还要求特定机型、可用区或反亲和规则，那么“理论总量够”不等于“实际能调度”。这时排查不能只看 `nvidia.com/gpu` 总数，还要同时看标签、拓扑、亲和性和组调度条件。

---

## 替代方案与适用边界

如果不使用 Volcano，也不是完全不能做 GPU 调度。原生 Kubernetes 已经能完成“设备发现 + 资源匹配 + 节点选择”这三件事。对于单 Pod 推理服务、单机训练、异步批处理任务，这通常已经够用。

但原生调度的边界也很清楚：它擅长给“单个 Pod”找节点，不擅长保证“一组 Pod 同时启动”。下面做一个对比：

| 方案 | GPU 资源发现 | 节点约束 | 多 Pod 同期启动 | 预占/队列能力 | 适用场景 |
|---|---|---|---|---|---|
| 原生 Kubernetes + Device Plugin | 支持 | 支持 | 不保证 | 基础能力 | 单 Pod、弱同步任务 |
| Kubernetes + Volcano | 支持 | 支持 | 强支持 | 更强 | DDP、Horovod、批训练 |
| 其他 batch scheduler / 自研控制器 | 取决于实现 | 通常支持 | 可支持 | 取决于实现 | 特殊调度规则、已有平台体系 |
| 应用层 barrier（MPI/torchrun 自己等） | 依赖底层已有 GPU 发现 | 不解决 | 只能“程序等”，不能“调度等” | 无 | 临时方案，不适合长期治理 |

新手例子：只用 `nodeAffinity` 把 4 个训练 Pod 限定在 GPU 节点，确实能避免它们跑到 CPU 节点上，但不能避免其中 3 个先启动、1 个迟迟等不到资源。程序最终可能仍然 hang。

真实工程例子：有些团队会在 `torchrun` 或 MPI 层自己做 barrier，试图让先起来的 Pod 等后面的 Pod。但这只能解决“程序什么时候开始训练”，解决不了“调度器已经把 3 张 GPU 锁住却仍然无法成组启动”的问题，反而会制造资源长期占用和队列阻塞。

所以适用边界可以简单记成一句话：**只关心单 Pod 能不能拿到 GPU，用 Device Plugin + 原生调度即可；关心一组 Pod 是否必须一起启动，就需要 Volcano 这类批调度能力。**

---

## 参考资料

| 资料 | 链接 | 简要说明 | 适用章节 |
|---|---|---|---|
| Kubernetes Device Plugins | https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/ | 官方说明设备插件注册、`ListAndWatch`、扩展资源暴露 | 问题定义、核心机制 |
| Assign Pods to Nodes | https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/ | 官方说明 `nodeSelector`、节点亲和性、Pod 亲和性/反亲和性 | 问题定义、工程权衡 |
| Volcano Plugins | https://volcano.sh/en/docs/v1-10-0/plugins/ | 官方说明 gang scheduling 插件和组调度语义 | 核心机制、替代方案 |
| Volcano Tutorials | https://volcano.sh/en/docs/v1-12-0/tutorials/ | 官方示例，包含 `PodGroup`、`Job`、`minAvailable` 用法 | 代码实现 |
| Batch Scheduling on Kubernetes | https://www.infracloud.io/blogs/batch-scheduling-on-kubernetes/ | 用案例解释批调度与资源等待 | 问题定义、替代方案 |
| AI Training Pipeline on K8s | https://www.youngju.dev/blog/kubernetes/ai_training_pipeline_k8s.en | 从训练任务角度解释 gang scheduling 对 DDP/Horovod 的价值 | 核心结论、工程例子 |
| Feisky Device Plugin Notes | https://kubernetes.feisky.xyz/en/extension/device | 补充说明设备插件与 kubelet socket 交互、重注册细节 | 核心机制、常见坑 |
