## 核心结论

GPU 资源调度的核心，不是“把卡塞满”，而是把三类资源做成可预测对象：算力 $C$、显存 $M$、尾延迟 $L_{tail}$。这里的“尾延迟”是指最慢那部分请求的延迟，常看 `p95` 或 `p99`，因为线上服务通常不是被平均值打垮，而是被最差那一小部分请求打垮。

`MIG`、`time-slicing`、`MPS` 不是同一种工具的不同参数，而是三种不同目标：

| 策略 | 隔离性 | 吞吐潜力 | 尾延迟稳定性 | 灵活性 | 典型场景 |
|---|---|---:|---:|---:|---|
| 独占 GPU | 很强 | 中到高 | 很好 | 很低 | 核心在线服务、关键训练任务 |
| MIG | 强 | 中 | 好 | 中 | 多租户在线推理、强 SLA 服务 |
| time-slicing | 弱 | 高 | 差到中 | 高 | 实验流量、低优先级 batch |
| MPS | 弱到中 | 高 | 依赖负载 | 中 | 协作型多进程、同团队共享任务 |

如果同一张卡上跑高优先级在线推理和低优先级批处理，优先目标应是“低尾延迟 + 显存隔离”，而不是“平均吞吐更高”。原因很直接：批任务多吃一点显存，或者长 kernel 多占一点执行时间，都会把短请求的 `p95/p99` 拉高，甚至直接触发 OOM。OOM 是显存溢出，意思是显存申请失败，进程可能报错退出；这类故障比吞吐下降更致命，因为它会让服务直接不可用。

---

## 问题定义与边界

本文讨论的问题，是多任务共卡时的资源分配，而不是单任务性能优化。更具体地说，是“单机单卡”或“单节点多卡共享”里的 GPU 调度：多个任务、多个租户、多个服务，如何在一块或几块 GPU 上共存，同时满足显存限制和时延要求。

不展开的边界有三类：

1. 不讨论集群级弹性调度，比如跨节点扩缩容、全局 bin packing。
2. 不讨论模型并行、张量并行、流水线并行。
3. 不讨论训练框架内部的算子级优化，比如 kernel fusion、FlashAttention。

统一记号如下：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $n$ | 同卡任务数 | 同时共享这张 GPU 的任务个数 |
| $q$ | 时间片长度 | 每轮轮到一个任务运行的时间窗口 |
| $m_i$ | 任务 $i$ 的显存需求 | 任务至少要占多少显存 |
| $M_{free}$ | 当前可用显存 | GPU 还能分配出去的显存 |
| $L_i$ | 任务 $i$ 的端到端时延 | 用户看到的总耗时 |
| $S_i$ | 服务时间 | 真正执行计算花的时间 |
| $W_i$ | 等待时间 | 排队、让路、竞争造成的时间 |

最基本的两个关系式是：

$$
L_i = S_i + W_i
$$

$$
\sum_{i=1}^{n} m_i \le M_{free}
$$

第一条说，请求延迟不是只由“算得多快”决定，还由“等了多久”决定。第二条说，只要显存总需求超过可用显存，系统就不再是“慢一点”，而是进入“不确定失败”状态：可能有人成功，有人失败，也可能反复重试后整体更慢。

一个玩具例子：

一张 40 GB GPU，上面同时来了 3 个任务：

- A：需要 16 GB
- B：需要 14 GB
- C：需要 12 GB

则总需求为：

$$
16 + 14 + 12 = 42 \text{ GB} > 40 \text{ GB}
$$

这时如果没有显存上限和准入控制，结果不会是“大家平分一点变慢”，而是某个进程先申请到显存，另一个进程后申请时直接 OOM，或者框架触发频繁回收、重试、排队，最后把整体延迟进一步放大。

所以，GPU 调度的第一原则不是“先调度算力”，而是“先做显存准入”。

---

## 核心机制与推导

先区分三种共享机制的本质。

`MIG` 是硬切分。硬切分的意思是，GPU 的部分计算资源和内存资源被切成多个彼此隔离的实例，每个实例像一张更小的独立 GPU。它的价值不在“更快”，而在“更稳”：别人抖动、泄漏、爆显存，不会直接污染你的实例。

`time-slicing` 是时间复用。时间复用的意思是，多个任务轮流占用 GPU，每次运行一个时间片 $q$。它提高的是空闲资源利用率，但代价是等待时间上升，尤其是短任务会被长任务拖住。

`MPS` 是协作并发。协作并发的意思是，多个 CUDA 进程共享一个服务端上下文，减少进程级上下文切换，让 kernel 更容易并发提交。它适合互相信任、协同工作的负载，不适合拿来做严格多租户隔离。

下面看一个最小推导。设有 $n$ 个任务共享 GPU，采用简单轮转时间片，每个任务一轮最多运行 $q$。对一个新来的短请求，如果它刚好错过本轮调度，最粗略等待时间可近似为：

$$
W_i \approx (n - 1) q
$$

这不是严格排队论结论，而是帮助理解的最小近似：你前面有多少个竞争者，就大概要等多少个时间片。

如果两路时间分片，即 $n=2$，每片 $q = 20ms$，那么一个短请求可能额外等待约：

$$
W_i \approx (2 - 1)\times 20 = 20ms
$$

如果考虑请求到达时机不理想、长 kernel 不可中断、框架额外同步开销，最坏常会接近 `20 ms 到 40 ms` 的额外等待。对一个原本 `30 ms` 就能完成的在线推理来说，这意味着尾延迟可能直接翻倍。

可以用一条简单时间线表示：

| 时间段 | 运行者 | 短请求状态 |
|---|---|---|
| 0-20 ms | 长任务 A | 等待 |
| 20-40 ms | 短任务 B | 执行 |
| 最坏情况下 0-40 ms | A 的长 kernel + 调度切换 | 持续等待 |

如果是三路共享，$n=3$，$q=20ms$，近似等待会变成：

$$
W_i \approx 40ms
$$

这就是为什么 `time-slicing` 常常“平均吞吐看起来不错，但 p95/p99 明显恶化”。

再看 `MPS`。MPS 的收益主要来自两点：

1. 多进程共享上下文，减少独立进程反复切换的开销。
2. 当多个 workload 的 kernel 能并发时，可以更充分利用 SM。

但它有两个前提：

1. 负载是协作型的，彼此互信。
2. 任务的执行特征兼容，不会因为一个超长 kernel 把别人都卡住。

这也解释了一个常见误区：`CUDA stream priority` 不是硬抢占。所谓 stream priority，是给 CUDA stream 设优先级，让调度器优先考虑高优先级 stream 中“尚未开始”的工作，但它不能像 CPU 抢占那样，把已经在 GPU 上运行的长 kernel 直接打断并替换掉。换句话说，它更像“优先排队”，不是“强制插队”。因此它可以改善一部分调度顺序，但不能替代 MIG 或真正的准入控制。

把三种机制放在一起看：

| 机制 | 主要原理 | 优点 | 代价 | 关键风险 |
|---|---|---|---|---|
| MIG | 硬件隔离实例 | 稳定、强隔离 | 碎片化、灵活性下降 | 小实例切错后利用率下降 |
| time-slicing | 轮流用 GPU | 容易提高利用率 | 等待时间增大 | 尾延迟抖动、无显存强隔离 |
| MPS | 多进程并发提交 | 吞吐提升、切换成本低 | 依赖负载兼容 | 不适合不互信多租户 |

真实工程例子通常是这样的：一张 A100 上有一个主推理服务、一个离线 embedding 生成任务、一个临时压测任务。若三者直接共卡，embedding 任务常有大 batch、大显存占用，压测任务又会制造大量短时突发，这时主推理服务的 `p99` 会先恶化，然后因为显存挤压出现 OOM 或频繁重试。很多团队一开始只看 GPU 利用率，以为“85% 很好”，上线后才发现用户请求变慢，原因就在于“利用率高”不等于“资源行为可预测”。

---

## 代码实现

实现上最重要的原则，是把“资源准入”和“调度策略”分开。

资源准入回答的是：这个任务现在能不能进？  
调度策略回答的是：能进以后，应该走哪种共享模式？

先看一个简化的可运行 Python 例子。它不是生产系统代码，但把核心判断编码清楚了：先检查显存，再看 SLA，再决定走 `MIG`、`time-slicing`、`MPS` 还是拒绝。

```python
from dataclasses import dataclass

@dataclass
class Task:
    name: str
    mem_gb: int
    sla_ms: int
    trusted: bool
    collaborative: bool
    priority: str  # "high", "medium", "low"

def choose_policy(task: Task, free_mem_gb: int, mig_enabled: bool) -> str:
    # 先做显存准入
    if task.mem_gb > free_mem_gb:
        return "reject_oom_risk"

    # 高 SLA 任务优先强隔离
    if task.priority == "high" or task.sla_ms <= 80:
        if mig_enabled:
            return "mig"
        return "dedicated"

    # 协作型、互信任务适合 MPS
    if task.trusted and task.collaborative and task.sla_ms <= 200:
        return "mps"

    # 低优先级且可容忍抖动的任务，才考虑时间分片
    if task.priority == "low" and task.sla_ms >= 300:
        return "time_slicing"

    # 其余情况保守处理
    return "dedicated"

online = Task("online_infer", mem_gb=10, sla_ms=60, trusted=True, collaborative=False, priority="high")
batch = Task("nightly_batch", mem_gb=12, sla_ms=1000, trusted=True, collaborative=False, priority="low")
coop = Task("shared_embedding", mem_gb=8, sla_ms=150, trusted=True, collaborative=True, priority="medium")
overflow = Task("too_big", mem_gb=48, sla_ms=100, trusted=True, collaborative=False, priority="high")

assert choose_policy(online, free_mem_gb=20, mig_enabled=True) == "mig"
assert choose_policy(batch, free_mem_gb=20, mig_enabled=True) == "time_slicing"
assert choose_policy(coop, free_mem_gb=20, mig_enabled=True) == "mps"
assert choose_policy(overflow, free_mem_gb=40, mig_enabled=True) == "reject_oom_risk"

print("scheduler policy checks passed")
```

这个例子故意把逻辑写得很直白，因为真正重要的不是“算法炫不炫”，而是以下工程约束必须显式存在：

1. 显存超限直接拒绝，不做乐观尝试。
2. 高 SLA 任务优先强隔离。
3. `MPS` 只给协作型、互信负载。
4. `time-slicing` 只给低优先级、可容忍抖动的任务。

伪代码可以再抽象成：

```text
if mem_demand > free_mem:
    reject
elif high_sla:
    use MIG if available else dedicated GPU
elif trusted and collaborative:
    use MPS
elif low_priority and latency_tolerant:
    use time-slicing
else:
    use dedicated GPU
```

如果在 Kubernetes 中表达这种策略，通常会拆成“节点标签 + 资源池”：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: online-infer
spec:
  nodeSelector:
    gpu.policy: mig
    gpu.sla: high
  containers:
  - name: server
    image: your-infer-image
    resources:
      limits:
        nvidia.com/mig-1g.10gb: 1
---
apiVersion: v1
kind: Pod
metadata:
  name: low-priority-batch
spec:
  nodeSelector:
    gpu.policy: time-slicing
    gpu.sla: low
  containers:
  - name: worker
    image: your-batch-image
    resources:
      limits:
        nvidia.com/gpu: 1
```

可以把策略配置写成一张表，便于运维和审计：

| 流量类型 | SLA | 优先级 | 推荐策略 | 显存上限 | 告警重点 |
|---|---:|---|---|---:|---|
| 在线推理主服务 | 50-100 ms | 高 | MIG / 独占 | 严格 | p99、OOM、排队时间 |
| 内部检索/embedding | 100-300 ms | 中 | MPS | 中等 | 吞吐、显存抖动 |
| 离线 batch | 秒级 | 低 | time-slicing | 可放宽但必须有限额 | 队列积压、平均耗时 |
| 降级路径 | 秒级以上 | 低 | CPU fallback | N/A | 成功率、降级比例 |

告警规则也应直接围绕风险展开，而不是只看 GPU 利用率：

| 指标 | 建议阈值 | 目的 |
|---|---:|---|
| `p95 latency` | 超过基线 1.5 倍 | 发现中度抖动 |
| `p99 latency` | 超过 SLA 上限 | 发现用户可感知故障 |
| `gpu memory used / limit` | > 90% 持续 5 分钟 | 发现 OOM 前兆 |
| `OOM count` | > 0 即报警 | 显存边界失控 |
| `queue wait time` | 持续升高 | 识别 time-slicing 排队放大 |

---

## 工程权衡与常见坑

第一类常见坑，是只看平均吞吐，不看尾延迟。平均吞吐表示整体干活能力，但线上体验经常由最慢那部分请求决定。一个系统即使平均延迟下降了，只要 `p99` 明显上升，用户仍然会觉得它更差。

第二类坑，是把 `time-slicing` 当隔离方案。它不是。它的本质是让多个任务轮流用 GPU，不提供 MIG 那样的硬件级资源边界。只要其中一个任务突然变长、变大、变得更吃显存，其他任务都会受到影响。

第三类坑，是共卡不设显存硬上限。显存和 CPU 内存不一样，GPU 框架一旦踩到边界，常见结果不是“慢一点”而是“直接失败”。因此要做的是 admission control，也就是准入控制：放不下就不进，而不是先进去再赌运气。

第四类坑，是误以为 `stream priority` 能解决高优先级在线请求抢不过 batch 的问题。前面已经解释，它不是硬抢占，只能改善调度顺序，不能把已经运行的长 kernel 立刻打断。

第五类坑，是把 `MPS` 用在互不信任的多租户环境。MPS 更像“合作共用一个厨房”，适合同团队、同服务链路、可控负载；不适合“彼此独立还要求强隔离”的环境。

一个真实工程故障模式是：夜间 batch 为了追求吞吐，增大 batch size，显存从 10 GB 涨到 18 GB；白天在线推理主服务原本在 12 GB 左右稳定运行，晚上和 batch 短时重叠时，显存峰值被挤到 30 GB 以上，再叠加缓存和中间张量，就可能直接撞上边界。表面上看是“利用率提高”，实际结果是短请求排队变长、显存波动变大、偶发 OOM 增多。对用户侧服务来说，这是典型的“用吞吐换不可控故障”，通常不值得。

建议持续监控的指标如下：

| 指标 | 为什么要看 |
|---|---|
| `p95/p99 latency` | 判断尾延迟是否被共享干扰拉高 |
| `OOM 次数` | 判断显存边界是否失控 |
| `GPU memory usage` | 判断是否接近危险上限 |
| `queue wait time` | 判断排队是否在放大 |
| `GPU utilization` | 只能做辅助指标，不能单独决策 |
| `request rejection rate` | 判断准入控制是否过严或容量不足 |

一个实用判断标准是：如果系统需要对外承诺 SLA，就必须优先保证隔离边界；如果系统只是内部可延迟任务，才可以优先考虑利用率。

---

## 替代方案与适用边界

没有一种 GPU 调度策略适合所有场景。选择方法不是问“哪个最好”，而是问“我要解决的主要矛盾是什么”。

如果主要矛盾是强隔离和稳定 SLA，优先选 `MIG` 或独占 GPU。两者区别在于：独占最简单，隔离也最彻底，但资源利用率可能低；`MIG` 在强隔离前提下提高了一些复用能力，但会带来实例规格碎片化。

如果主要矛盾是提高空闲期利用率，且任务能容忍抖动，可以考虑 `time-slicing`。但它适合实验流量、低优先级推理、离线 batch，不适合对尾延迟敏感的主服务。

如果主要矛盾是多个协作进程共同完成一件事，比如同一服务拆成多个 worker，或多个兼容 workload 共用同一模型资源，那么可以考虑 `MPS`。前提仍然是互信、兼容、可控。

如果 GPU 紧张但业务必须有兜底路径，还可以设置 `CPU fallback`。兜底路径的意思是，当 GPU 容量不足或调度拒绝时，把请求降级到 CPU 执行。它通常更慢，但比直接失败更好，尤其适合低频管理接口、非实时分析任务。

可以用一个决策矩阵快速判断：

| 方案 | 隔离性 | 利用率 | 尾延迟 | 运维复杂度 | 适用租户类型 |
|---|---|---:|---:|---:|---|
| 独占 GPU | 很强 | 低到中 | 最稳 | 低 | 核心单租户、高价值服务 |
| MIG | 强 | 中 | 稳 | 中 | 多租户、高 SLA |
| MPS | 弱到中 | 高 | 视负载而定 | 中 | 协作型、互信负载 |
| time-slicing | 弱 | 高 | 易抖动 | 中 | 低优先级、可延迟任务 |
| CPU fallback | 强 | 低 | 慢但可控 | 低 | 降级兜底、非实时任务 |

最实用的结论可以压缩成一句话：面向用户请求的主服务，不要把 `time-slicing` 当隔离方案；面向实验和离线任务，可以用它换利用率。需要强隔离时，用 `MIG` 或独占；需要协作并发时，用 `MPS`；需要保底可用时，加 `CPU fallback`。

---

## 参考资料

1. [NVIDIA MIG User Guide: Introduction](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html)  
支持本文关于 `MIG` 属于硬件级隔离、适合稳定多租户共享的结论。

2. [NVIDIA MPS Documentation: When to Use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)  
支持本文关于 `MPS` 适合协作型负载、目标是提高并发与吞吐而非强隔离的结论。

3. [NVIDIA GPU Operator: Time-Slicing GPUs in Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/23.9.1/gpu-sharing.html)  
支持本文关于 `time-slicing` 适合共享与利用率提升、但不等于强隔离的工程实现结论。

4. [CUDA Driver API: Stream Priority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)  
支持本文关于 `CUDA stream priority` 不是硬抢占，只能影响调度优先级的结论。

5. [Gandiva: Introspective Cluster Scheduling for Deep Learning](https://www.usenix.org/system/files/osdi18-xiao.pdf)  
支持本文关于深度学习任务调度需要围绕资源特征与作业行为进行匹配，而非只追求静态平均利用率的更广义思路。
