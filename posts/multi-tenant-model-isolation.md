## 核心结论

多租户模型隔离，指的是把不同客户的推理请求在 `CPU / GPU 计算 / 显存 / 队列 / 缓存` 五层分别控制共享边界，目标不是绝对独占，而是让一个租户的抖动、拥塞和故障不要扩散到其他租户。

在线大模型服务里，最先失控的通常不是算子算不过来，而是队列排队、`KV cache` 膨胀和缓存污染。`KV cache` 可以理解为模型为“继续生成下一个 token”暂存的上下文状态；它本质上吃显存，而且会随着上下文和并发增长。只隔离计算、不隔离显存，`OOM` 和尾延迟仍然会传播。只隔离显存、不隔离队列，短请求仍然会被长请求拖慢。隔离过重则会降低卡利用率，抬高单位成本。

端到端延迟可以写成：

$$
L_i = W_{q,i} + S_{cpu,i} + S_{gpu,i} + S_{mem,i} + S_{cache,i}
$$

其中 $L_i$ 是租户 $i$ 的总延迟，$W_{q,i}$ 是排队时间，$S_{cpu,i}$ 是前后处理时间，$S_{gpu,i}$ 是 GPU 计算时间，$S_{mem,i}$ 是显存争抢带来的额外代价，$S_{cache,i}$ 是缓存命中或回收带来的代价。

| 层 | 主要风险 | 常用隔离手段 |
|---|---|---|
| CPU | 前后处理线程互相抢核 | `cpuset`、Kubernetes `CPU Manager static` |
| GPU 计算 | 算子执行互相干扰 | `MIG`、固定实例绑定 |
| 显存 | `KV cache` 撑爆、回收、`OOM` | tenant quota、显存分片 |
| 队列 | 长请求拖慢短请求、尾延迟放大 | tenant admission、并发上限、iteration-level scheduling |
| 缓存 | 前缀缓存污染、边界不清 | namespace、按租户池化、`resctrl/CAT` |

---

## 问题定义与边界

本文讨论的是在线推理服务中的多租户隔离，不讨论模型训练，也不讨论离线批处理。这里的“租户”就是不同客户、业务线或组织单元；“隔离”不是要求每个租户都独占整机，而是要求共享发生在可控范围内。

问题的核心不是“有没有共享”，而是“共享是否会失控”。如果多个租户共用一张卡，输入请求大致会经过：接入、排队、前处理、GPU 执行、`KV cache` 占用、输出返回。只要其中一层没有边界，传播就会发生。比如只给每个租户固定 CPU 核心，但所有租户共享同一个 `KV cache` 上限，那么某个长上下文请求仍可能把整张卡的剩余显存吃满，其他租户一起变慢。

范围内与范围外可以明确成表：

| 类型 | 是否在本文范围内 | 说明 |
|---|---|---|
| 在线 LLM Serving | 是 | 请求实时到达，关注吞吐与延迟 |
| 多租户共享 GPU 节点 | 是 | 重点讨论共享边界控制 |
| 训练任务 | 否 | 训练更关注梯度、批量和长作业调度 |
| 纯 CPU 服务 | 否 | 不涉及 GPU/显存/KV cache 问题 |
| 单租户独占整机 | 否 | 隔离问题被硬件独占直接简化 |

一个重要边界是：本文默认租户之间没有数据共享要求。也就是说，缓存复用如果跨租户发生，必须先经过安全与策略判断，而不是默认开放。

---

## 核心机制与推导

多租户抖动通常先从排队和显存开始放大。排队的直觉是：请求来得比系统处理得快，等待就会突然变长。可以用近似式表示：

$$
W_{q,i} \approx \frac{1}{\mu_i - \lambda_i}
$$

这里 $\lambda_i$ 是租户 $i$ 的到达率，$\mu_i$ 是服务率。这个式子不是精确调度公式，但足够说明一个事实：当到达率逼近服务率时，等待时间会陡增。所以没有租户级 admission control 时，某个租户的突发流量会先把自己的队列打满，再挤占共享执行机会。

显存的边界更硬，因为它不是“慢一点”，而是可能直接失败。安全条件可以写成：

$$
\sum_i M_i \le M_{free}
$$

其中 $M_i$ 是各租户实际占用的动态显存，$M_{free}$ 是权重之外可供 `KV cache` 和中间张量使用的剩余显存。超过这个边界，系统就会发生回收、驱逐、重算，最坏情况下直接 `OOM`。

玩具例子如下：一张 GPU 可用于 `KV cache` 的空间是 `8 GB`。租户 A 和 B 平时各用 `3 GB`，总计 `6 GB`，系统正常。某一时刻 A 收到长上下文请求，`KV cache` 变成 `7 GB`，此时总需求变成 `10 GB > 8 GB`。如果没有租户配额，系统只能回收旧块、拒绝新请求、触发重算，或者直接 `OOM`。结果不会只影响 A，B 的 `p99` 也会恶化，因为 B 的缓存命中被打断，甚至排到本不该出现的重试路径上。

这就是“只隔离计算、不隔离显存”为什么不够。GPU 算子也许还排得进，但显存先爆了，延迟传播照样发生。实践里的优先顺序通常是：先保证显存安全，再做队列公平，再做 `CPU/GPU` 绑定，最后才谈缓存复用收益。

机制到风险的映射如下：

| 机制 | 能防什么 | 不能防什么 |
|---|---|---|
| `cpuset` / `CPU Manager` | CPU 抢占、线程漂移 | 显存爆掉、队列不公平 |
| `MIG` | GPU 计算和部分硬件资源隔离 | 业务层排队失衡、跨租户缓存策略问题 |
| 显存 quota | `KV cache` 失控、`OOM` 扩散 | CPU 抢核、短请求被长请求拖慢 |
| 租户队列 + admission | 流量突发传播、尾延迟放大 | 单个请求显存占用过大 |
| cache namespace | 缓存污染、边界模糊 | 物理资源本身不足 |

真实工程例子是：在一张 A100 或 H100 上承载多个企业客户的在线问答服务。平台给每个租户单独的请求队列和并发上限，用 `MIG` 切出多个实例，用 Kubernetes `static CPU Manager` 固定 tokenize 与后处理线程，再对 `KV cache` 设置租户配额，只允许在明确策略下做 `prefix cache` 复用。这个方案的关键不是“把一切都切碎”，而是把最危险的传播链路先堵住。

---

## 代码实现

实现顺序建议是：先定显存配额和队列，再定 `CPU/GPU` 绑定，最后做缓存复用。因为前两者决定系统是否稳定，后两者更多决定性能上限。

一个最小的接入层逻辑可以写成：

```python
from dataclasses import dataclass

@dataclass
class TenantState:
    queue_size: int
    queue_limit: int
    kv_usage_gb: float
    kv_quota_gb: float
    inflight: int
    inflight_limit: int

def admit_request(state: TenantState, est_kv_gb: float) -> str:
    if state.queue_size >= state.queue_limit:
        return "reject_queue_full"
    if state.inflight >= state.inflight_limit:
        return "delay_concurrency_limit"
    if state.kv_usage_gb + est_kv_gb > state.kv_quota_gb:
        return "throttle_kv_quota"
    return "accept"

tenant_a = TenantState(queue_size=2, queue_limit=8, kv_usage_gb=3.0, kv_quota_gb=4.0, inflight=1, inflight_limit=2)
tenant_b = TenantState(queue_size=8, queue_limit=8, kv_usage_gb=1.0, kv_quota_gb=4.0, inflight=0, inflight_limit=2)

assert admit_request(tenant_a, 0.5) == "accept"
assert admit_request(tenant_a, 1.5) == "throttle_kv_quota"
assert admit_request(tenant_b, 0.2) == "reject_queue_full"
```

这个代码块不负责真正调度 GPU，但它表达了多租户隔离的最小原则：先看队列是否还能接，再看并发是否越界，再看显存预算是否安全。只有这三层都满足，请求才进入执行域。

工程上常见配置关系如下：

| 配置项 | 作用 |
|---|---|
| Pod `cpu requests/limits` | 给前后处理线程预留稳定 CPU |
| `CPU Manager static` | 把 CPU 绑定到固定核心，减少抖动 |
| `MIG profile` | 把 GPU 切成硬件实例 |
| `resctrl` schemata | 限制 LLC 争抢，减少缓存互相污染 |
| `KV cache quota` | 给每个租户设置显存上限 |

最小实现思路可以概括成伪代码：

```text
on_request(tenant_id, req):
    if tenant_queue[tenant_id].full():
        reject_or_delay(req)
    if kv_cache_usage[tenant_id] + estimate_kv(req) > kv_quota[tenant_id]:
        throttle_or_evict(req)
    if inflight[tenant_id] >= inflight_limit[tenant_id]:
        delay(req)
    enqueue(tenant_id, req)
```

关键点是“同一租户的请求绑定到固定资源域”。这不一定意味着一租户一整卡，而是意味着它进入固定队列、固定显存预算、固定 CPU 核心集合，以及必要时固定的 GPU 实例。

---

## 工程权衡与常见坑

最常见的误判，是把多租户问题理解成单一的“算力切分”问题。实际上它更像一条链路上的多点失控。

| 误区 | 后果 |
|---|---|
| 只隔离 CPU | `KV cache` 仍可打爆显存，尾延迟照样传播 |
| 只做 `MIG` | 队列仍可能不公平，短请求被长请求挤压 |
| 共享 `prefix cache` 无边界 | 命中率提高，但隔离边界变软 |
| 一租户一整卡 | 利用率下降，成本显著上升 |

一个常见现象是：监控上看 GPU 利用率并不高，但 `p99` 已经恶化。原因往往不是算子跑满，而是队列里混进了长上下文请求，或者 `KV cache` 在高峰期反复回收。对策不是盲目扩卡，而是先检查租户级队列、并发上限和显存配额是否存在。

实践上的规避顺序通常是：

1. 先定显存上限，确保 $\sum_i M_i \le M_{free}$ 有工程可执行的控制面。
2. 再做租户队列与 admission，避免突发流量直接扩散。
3. 再做 `CPU/GPU` 绑定，降低共享资源上的随机抖动。
4. 最后做缓存复用，只在边界明确时追求命中率收益。

一句话总结这一节：共享不是问题，失控的共享才是问题。

---

## 替代方案与适用边界

不存在对所有业务都最优的单一方案，只有在 `SLA`、成本和利用率之间更合适的折中。`SLA` 可以理解为服务等级目标，比如延迟上限和可用性承诺。

| 方案 | 适用场景 | 优点 | 缺点 | 不适用边界 |
|---|---|---|---|---|
| 一租户一整卡 | 低并发、高价值客户 | 隔离最强，问题最容易定位 | 利用率低，成本高 | 标准化高并发服务 |
| 只做 `MIG` | 中等隔离要求 | 配置直观，硬件边界清晰 | 队列和缓存问题仍在 | 延迟严格场景 |
| `MIG + 队列 + 显存配额` | 大多数在线企业服务 | 隔离与利用率较均衡 | 实现复杂度更高 | 极低运维能力团队 |
| 软隔离 + 高复用 | 高并发、成本敏感 | 吞吐高，缓存收益大 | 边界更软，抖动更难控 | 强隔离合规场景 |

如果业务是低并发但单次请求价值高，比如法务、金融、医疗类专属客户，通常应偏向更强隔离。如果业务是高并发、标准化问答或代码补全，适度共享 `prefix cache`、接受有限抖动，往往更划算。

最终判断标准不是“是否共享”，而是“共享后的风险是否可观测、可限制、可回退”。做不到这三点，就不应把共享推到生产默认路径。

---

## 参考资料

1. [NVIDIA Multi-Instance GPU User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html)
2. [Kubernetes Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/workloads/resource-managers/)
3. [Linux Kernel Documentation: cgroup v2](https://docs.kernel.org/6.10/admin-guide/cgroup-v2.html)
4. [Linux Kernel Documentation: resctrl](https://docs.kernel.org/filesystems/resctrl.html)
5. [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/v0.18.0/features/automatic_prefix_caching/)
6. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
