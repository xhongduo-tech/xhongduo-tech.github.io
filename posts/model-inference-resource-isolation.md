## 核心结论

`模型推理资源隔离` 是把推理所需的 GPU 算力、显存、请求队列和租户配额切成清晰边界，让一个租户的峰值流量主要伤害自己，而不是把整台服务一起拖垮。

这件事的重点不是“平均分资源”，而是控制最容易失控的两个对象：`GPU 时间` 和 `KV cache`。`KV cache` 可以理解为大模型在生成时为后续 token 保留的中间记忆，占用的主要是显存。长 prompt、长输出、突发高并发，都会把这两类资源迅速放大。

一个新手最容易误解的点是：只要 Kubernetes 给 Pod 配了 `cpu` 和 `memory` 限额，推理服务就算安全。这个结论通常不成立。推理阶段真正的瓶颈常常在 GPU 显存、模型实例并发、批处理队列，而不是 CPU。

先看一个反例。两支团队共用一张 GPU，团队 A 在下午 3 点突然打来大量长上下文请求，团队 B 仍然只有少量普通请求。如果没有隔离，A 会先把显存和队列占满，B 虽然流量不大，也会一起超时。

| 场景 | 没有隔离时发生什么 | 结果 |
| --- | --- | --- |
| 某租户发来大量长 prompt | KV cache 快速膨胀，显存被挤满 | 其他租户 OOM 或排队超时 |
| 某租户短时间突发高并发 | batching 队列被单租户占住 | 其他租户尾延迟上升 |
| 某租户坏请求反复重试 | GPU 时间被无效请求消耗 | 正常请求吞吐下降 |
| 某租户上线更大模型 | 权重与运行时开销增大 | 公共池剩余容量不足 |

所以，资源隔离不是“可选优化”，而是多租户推理进入生产环境后的基础约束。

一个常见的资源边界示意可以写成：

`租户 -> 配额 -> GPU slice -> 队列 -> 模型实例`

这里的 `GPU slice` 是 GPU 切片，也就是一块独立可分配的 GPU 资源；`模型实例` 是某个推理引擎实际跑起来的一份模型进程或执行单元。

---

## 问题定义与边界

先定义“隔离”到底在隔什么。

在模型推理里，需要被隔离的通常不是抽象的“机器资源”，而是以下几类具体对象：

| 资源类型 | 是否必须隔离 | 失控后果 | 典型控制手段 |
| --- | --- | --- | --- |
| GPU 设备或 MIG slice | 是 | 算力被抢占，吞吐抖动 | 独占 GPU、MIG、节点池隔离 |
| 显存预算 | 是 | OOM、频繁驱逐、实例重启 | KV 预算、最大上下文、headroom |
| 请求队列 | 是 | 单租户占满排队窗口 | 按租户拆队列、优先级队列 |
| 并发上限 | 是 | prefill 高峰压垮实例 | 每租户并发限额、速率限制 |
| batching 窗口 | 通常是 | 大请求拖慢小请求 | 最大队列延迟、分类 batching |
| CPU / 内存 | 需要，但不充分 | 控制面抖动、日志阻塞 | Pod request/limit |
| 网络带宽 | 视场景而定 | 流式输出阻塞 | 网关限流、带宽整形 |

这里要区分两个概念。

`调度隔离`：指资源在进入运行前就被分配好边界，比如某个 Namespace 最多只能申请 1 个 MIG slice。白话说，就是“先拿到准入证，再允许上车”。

`运行时隔离`：指请求已经进入模型服务后，还要继续控制显存、队列、批处理和并发。白话说，就是“上车以后也不能随便挤别人座位”。

这两个层面缺一不可。只有调度隔离，没有运行时隔离，结果通常是“Pod 能启动，但业务仍然互相拖垮”。只有运行时隔离，没有调度隔离，则会出现“服务自己很克制，但上层集群把资源超卖了”。

问题边界也要讲清楚。下面这些问题属于资源隔离范畴：

- 单租户长上下文把显存吃满。
- 单租户高并发占满批处理窗口。
- 不同租户请求混队导致尾延迟相互传染。
- 多模型混部时，一个模型实例抢走过多 GPU 时间。

下面这些问题不完全属于资源隔离：

- 模型回答质量差。
- 检索召回不好，导致 prompt 本来就过长。
- 路由策略错误，把简单请求送到大模型。
- 缓存命中率低，导致重复推理过多。
- 模型权重量化方案不合适，影响精度。

也就是说，资源隔离回答的是“同一批硬件怎么不互相踩踏”，不直接回答“模型本身对不对”。

---

## 核心机制与推导

推理隔离为什么成立，核心在于显存和时间都可以近似预算化。

一个隔离单元能否承载某个请求，先看下面这个近似关系：

$$
M_{total} = M_{weight} + M_{kv}(T) + M_{act} + M_{overhead} \le M_{slice}
$$

含义如下：

- $M_{weight}$：模型权重占用。
- $M_{kv}(T)$：KV cache 占用，随 token 数 $T$ 增长。
- $M_{act}$：激活值开销，也就是前向计算中临时产生的数据。
- $M_{overhead}$：运行时额外开销，比如 allocator、workspace、通信 buffer。
- $M_{slice}$：隔离单元可用显存，比如一块 MIG slice 或一张独占 GPU 的可用显存上限。

其中最容易被请求模式直接放大的，是 $M_{kv}(T)$。一个常见近似式是：

$$
M_{kv}(T) \approx 2 \cdot L \cdot T \cdot H_{kv} \cdot D \cdot b
$$

各参数含义：

- $L$：Transformer 层数。
- $T$：总 token 数，通常近似为输入 token 加已生成 token。
- $H_{kv}$：KV heads 数。
- $D$：每个 head 的维度。
- $b$：每个元素字节数，FP16 常取 2。

这个式子的直觉很简单：KV cache 要为每层保存 `K` 和 `V` 两份数据，所以有前面的 2；token 越长，要保存的“记忆”就越多，因此近似线性增长。

看一个玩具例子。

假设有一个 32 层模型，$L=32$，$H_{kv}=32$，$D=128$，用 FP16，所以 $b=2$。当总 token 数 $T=2048$ 时：

$$
M_{kv} \approx 2 \cdot 32 \cdot 2048 \cdot 32 \cdot 128 \cdot 2
$$

算出来大约是：

$$
1{,}073{,}741{,}824 \text{ bytes} \approx 1 \text{ GiB}
$$

如果总 token 数翻到 $4096$，那么：

$$
M_{kv} \approx 2 \text{ GiB}
$$

这说明长上下文不是“慢一点”这么简单，而是直接把一个请求的显存账单几乎翻倍。隔离设计如果只看平均请求，而不看最大 token 长度，就会在高峰期失效。

再把它写成更工程化的表：

| token 长度 T | 近似 KV cache | 如果 slice 可用 8 GiB，理论可承载并发数上限 | OOM 风险 |
| --- | --- | --- | --- |
| 1024 | 约 0.5 GiB | 更高 | 低 |
| 2048 | 约 1 GiB | 中等 | 中 |
| 4096 | 约 2 GiB | 明显下降 | 高 |
| 8192 | 约 4 GiB | 很低 | 很高 |

注意，这里的“理论可承载并发数上限”还没扣掉权重、激活值和运行时预留，只是用来说明趋势。真实系统可承载数会更低。

机制链路可以概括成：

`请求进入 -> 预算检查 -> 队列 -> batching -> KV cache 分配 -> 推理输出`

其中最关键的两个控制点是：

1. 预算检查  
先判断这个请求是否在租户边界内，例如最大 prompt 长度、最大输出长度、最大并发预算。

2. 队列与 batching  
即使请求合法，也不能让所有租户共用一个无限制公共队列，否则一个大租户会通过排队窗口继续伤害别人。

真实工程例子通常发生在多租户 Kubernetes 集群。假设团队 A 和团队 B 共用一组 A100。团队 A 运行长文档问答，请求普遍在 4k 到 8k token；团队 B 运行简单分类，请求通常只有几百 token。如果不隔离，A 的 prefill 阶段会大量占用显存和 GPU 时间，B 的小请求虽然很快，但会被卡在公共队列里。结果是平均吞吐看似还行，P99 延迟却突然恶化。这个现象就是典型的“平均值正常，尾部失控”。

所以隔离真正控制的是尾部行为，而不是把所有请求都变得绝对公平。

---

## 代码实现

代码实现通常分三层：资源层做硬隔离，编排层做配额控制，服务层做并发和 batching 控制。

第一层是 Kubernetes 或节点资源层。最小示例是给租户所在 Namespace 设置 GPU 配额：

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-a-quota
spec:
  hard:
    nvidia.com/gpu: "1"
```

如果底层使用 MIG，也可能暴露成特定的扩展资源名，但核心思想一样：租户先被限制“最多拿多少 GPU 资源”。

Pod 级别还要把 `requests` 和 `limits` 写清楚：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: llm-infer-tenant-a
spec:
  containers:
    - name: server
      image: my-infer:latest
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          nvidia.com/gpu: 1
```

这里的重点不是语法，而是“不能超卖”。很多 GPU 扩展资源天然要求 `requests == limits`，因为它不是像 CPU 那样可被时间片细粒度共享的普通资源。

第二层是推理引擎层。以 Triton 为例，模型实例和 batching 策略可以这样写：

```protobuf
instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 2000
}
```

`instance_group` 表示给这个模型配置多少实例；`dynamic_batching` 表示在多长时间窗口内攒请求做批处理。白话说，前者决定“开几条车道”，后者决定“是否等几辆车一起发车”。

但只到这里还不够。服务层还要知道“每个租户自己的预算是多少”，并在请求进入时拒绝越界请求。下面给一个可运行的 Python 玩具实现，它不依赖具体框架，但把关键机制表达出来了。

```python
from dataclasses import dataclass
from collections import defaultdict

GiB = 1024 ** 3

@dataclass
class TenantBudget:
    name: str
    max_prompt_tokens: int
    max_total_tokens: int
    kv_budget_bytes: int
    max_inflight: int

def kv_cache_needed_bytes(layers: int, total_tokens: int, kv_heads: int, head_dim: int, bytes_per_elem: int) -> int:
    return 2 * layers * total_tokens * kv_heads * head_dim * bytes_per_elem

class AdmissionController:
    def __init__(self, tenants):
        self.tenants = {t.name: t for t in tenants}
        self.inflight = defaultdict(int)

    def admit(self, tenant_name: str, prompt_tokens: int, gen_tokens: int) -> str:
        t = self.tenants[tenant_name]
        total_tokens = prompt_tokens + gen_tokens

        if prompt_tokens > t.max_prompt_tokens:
            return "reject: prompt too long"

        if total_tokens > t.max_total_tokens:
            return "reject: total tokens too large"

        kv_need = kv_cache_needed_bytes(
            layers=32,
            total_tokens=total_tokens,
            kv_heads=32,
            head_dim=128,
            bytes_per_elem=2,
        )

        if kv_need > t.kv_budget_bytes:
            return "reject: insufficient kv budget"

        if self.inflight[tenant_name] >= t.max_inflight:
            return "reject: too many inflight requests"

        self.inflight[tenant_name] += 1
        return "accept"

    def finish(self, tenant_name: str) -> None:
        assert self.inflight[tenant_name] > 0
        self.inflight[tenant_name] -= 1


tenant_a = TenantBudget(
    name="tenant-a",
    max_prompt_tokens=4096,
    max_total_tokens=4096,
    kv_budget_bytes=2 * GiB,
    max_inflight=2,
)

tenant_b = TenantBudget(
    name="tenant-b",
    max_prompt_tokens=1024,
    max_total_tokens=2048,
    kv_budget_bytes=1 * GiB,
    max_inflight=1,
)

ctrl = AdmissionController([tenant_a, tenant_b])

assert ctrl.admit("tenant-a", prompt_tokens=1024, gen_tokens=512) == "accept"
assert ctrl.admit("tenant-a", prompt_tokens=1024, gen_tokens=512) == "accept"
assert ctrl.admit("tenant-a", prompt_tokens=1024, gen_tokens=512) == "reject: too many inflight requests"

ctrl.finish("tenant-a")
assert ctrl.admit("tenant-b", prompt_tokens=1500, gen_tokens=100) == "reject: prompt too long"
assert ctrl.admit("tenant-b", prompt_tokens=1024, gen_tokens=2000) == "reject: total tokens too large"
assert ctrl.admit("tenant-b", prompt_tokens=1024, gen_tokens=1024) == "reject: insufficient kv budget"
assert ctrl.admit("tenant-b", prompt_tokens=512, gen_tokens=256) == "accept"
```

这个例子故意很小，但它说明三件事：

- 租户预算不是“建议值”，而是准入条件。
- token 长度应直接参与 KV cache 预算判断。
- 并发上限必须按租户统计，而不是只看全局总并发。

在真实系统里，还会继续往下拆：

`K8s quota -> GPU/MIG slice -> model instance -> request queue`

如果要落到更完整的工程方案，一个较常见的分层是：

- 资源层：独占 GPU、MIG、专用节点池。
- 编排层：Namespace 配额、Deployment 副本数、调度约束。
- 服务层：按租户队列、并发门限、最大上下文、batching 延迟上限。
- 业务层：API 网关鉴权、租户标签透传、计费与审计。

---

## 工程权衡与常见坑

隔离不是切得越细越好。切片越细，边界越强，但 GPU 利用率往往越差。因为模型推理喜欢足够大的并行空间，而过碎的 slice 会带来 kernel 效率损失、上下文切换开销和容量浪费。

这就是一个基本冲突：

- `高利用率` 倾向于做更大的共享池，便于动态批处理和弹性调度。
- `强隔离` 倾向于做更小更硬的边界，避免相互干扰。

生产环境里通常只能折中，而不是两者同时极致。

下面是几个常见坑：

| 常见坑 | 失败模式 | 规避方法 |
| --- | --- | --- |
| 只配 CPU / memory，不算 GPU / KV cache | Pod 能调度，推理时 OOM | 按峰值 token 长度做显存预算 |
| 只切 MIG，不切队列 | 公共队列仍被大租户占住 | 按租户拆队列，设置并发上限 |
| 切得太碎 | 吞吐下降，碎片化严重 | 先测模型最小可运行 footprint |
| 不留 headroom | 临时 buffer 或 prefill 峰值触发 OOM | 预留固定比例余量 |
| 忽视 MIG 重配成本 | 线上变更中断服务 | 低峰期 drain 后再调整 |
| 只看平均延迟 | P99 持续失控但报警不明显 | 单独监控尾延迟和拒绝率 |

这里的 `headroom` 指预留余量，也就是你明知一块 slice 理论上能塞满，但故意不塞满，用来吸收临时波峰和碎片开销。

一个常见误区是把 MIG 当成隔离的全部答案。MIG 的价值在于硬件层切片，它很适合做“这个租户拿不到别人的显存和算力”这件事。但它并不自动解决请求队列污染、长短请求混跑、动态 batching 偏斜这些问题。换句话说，MIG 解决的是“物理边界”，不是“服务行为”。

容量规划时，至少要检查以下项：

| 检查项 | 为什么必须看 |
| --- | --- |
| 峰值 prompt 长度 | 决定 KV cache 上界 |
| 最大生成长度 | 决定请求生命周期和总 token |
| 最大并发 | 决定同时占用的显存与 GPU 时间 |
| KV cache 预算 | 决定实例是否能稳定承载 |
| headroom 比例 | 决定是否能抗瞬时波峰 |
| batching 延迟上限 | 决定吞吐与尾延迟折中 |
| 变更是否需要 GPU reset | 决定运维窗口成本 |

真实工程里，最危险的不是“系统马上崩”，而是“系统还能跑，但用户体感突然变差”。例如平均延迟只涨了 10%，P99 却从 2 秒跳到 20 秒。这类问题如果没有按租户、按模型、按 token 长度分桶监控，很容易长期被忽略。

---

## 替代方案与适用边界

不是所有场景都需要复杂的资源隔离。

如果你只有一个团队、一个模型、请求长度也稳定，那么最简单的办法往往是给它独占一张 GPU 或一个模型副本。维护成本最低，故障面也最小。复杂隔离通常是多租户、多模型、请求波动大以后才值得上的。

下面做一个对比：

| 方案 | 优点 | 缺点 | 适用边界 |
| --- | --- | --- | --- |
| 独占 GPU | 实现最简单，隔离最强 | 资源利用率可能低 | 单团队、稳定负载 |
| MIG 切片 | 硬件级边界清晰 | 切太碎会损失吞吐，重配有成本 | 多团队共享高端 GPU |
| 只做限流 | 简单，接入成本低 | 无法控制显存级冲突 | 请求短、模型小、冲突轻 |
| 只做 batching | 吞吐高 | 容易放大排队延迟 | 单租户或请求模式相近 |
| 多副本按租户隔离 | 行为边界明确，便于计费 | 副本数多，成本高 | 中等规模多租户 |

还要注意，资源隔离只能解决“争抢”问题，不能解决所有性能问题。

例如：

- 如果主要问题是模型路由错了，小请求被送去超大模型，那么隔离无法替你省钱。
- 如果主要问题是缓存命中率太差，重复 prompt 太多，那么隔离只会让坏模式局部化，不会让它消失。
- 如果主要问题是多模型混部策略差，一个 embedding 模型和一个大语言模型互相抢设备，那么更重要的可能是重新布局服务拓扑，而不是继续加细粒度限制。

一个实用的判断标准是：

- 当你开始出现“一个租户的峰值明显拖累别的租户”时，应该升级到资源隔离。
- 当你只是“单业务负载波动不大，且资源足够”，不一定需要复杂隔离，简单副本独占往往更稳。

换句话说，资源隔离不是默认答案，而是多租户推理进入复杂阶段后的必要工具。

---

## 参考资料

按层次看，建议先理解“GPU 怎么切”，再理解“集群怎么限额”，最后理解“推理引擎怎样使用这些边界”。

| 层次 | 资料类型 |
| --- | --- |
| 硬件层 | GPU / MIG 切片与隔离能力 |
| 编排层 | Kubernetes 资源请求与配额 |
| 推理引擎层 | batching、实例组、KV cache 管理 |
| 研究与实现 | PagedAttention、vLLM 等推理内存优化 |

1. [NVIDIA Multi-Instance GPU User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html)
2. [NVIDIA DGX A100 User Guide: Using MIG](https://docs.nvidia.com/dgx/dgxa100-user-guide/using-mig.html)
3. [Kubernetes Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
4. [Kubernetes Resource Quotas](https://kubernetes.io/docs/concepts/policy/resource-quotas/)
5. [NVIDIA Triton Inference Server Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
6. [PagedAttention: Efficient Memory Management for LLM Serving with PagedAttention](https://huggingface.co/papers/2309.06180)
7. [vLLM Project](https://github.com/vllm-project/vllm)
8. [vAttention Project](https://github.com/microsoft/vattention)
