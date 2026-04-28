## 核心结论

推理服务资源配额的本质，不是“少用资源”，而是把 `GPU 内存 / 活跃序列数 / 单批 token 数 / 队列长度` 变成可计算、可拒绝、可回退的硬边界。这样做的目标不是追求瞬时吞吐最大，而是避免 GPU OOM、KV cache 失控和排队抖动，把运行时事故提前改写成接单前的调度判断。

这里的 **KV cache**，可以先理解成“模型为了继续生成，必须暂存的历史上下文记忆”。生成越长、同时活跃的请求越多，这部分内存越容易先顶满。很多新手会直觉认为“模型慢是算力不够”，但在线推理里更常见的先触顶资源其实是显存和等待时间，而不是纯计算单元。

一个可直接记住的统一框架是：

| 控制项 | 控制对象 | 主要目标 |
|---|---|---|
| `gpu_memory_utilization` | 单实例 GPU 可用预算 | 防 OOM |
| `max_num_seqs` | 活跃序列数 | 限并发 |
| `max_num_batched_tokens` | 单轮批处理 token 数 | 控 prefill/decode 竞争 |
| 队列长度 | 进入服务前的等待请求数 | 控排队和 429 策略 |

总预算可以写成：

$M_{total} = M_w + M_{act} + M_{kv}$

其中，$M_w$ 是模型权重占用，$M_{act}$ 是运行时固定开销，$M_{kv}$ 是 KV cache 占用。真正的工程结论是：**配额不是额外负担，而是把不稳定的运行时风险提前变成可计算的调度问题。**

先看一个玩具例子。把单个推理实例想成一辆货车：`GPU 内存` 是货箱体积，`KV cache` 是已经装上车、途中不能随便扔掉的货，`max_num_seqs` 是这趟允许同时接多少单，`max_num_batched_tokens` 是单趟允许装多少件货。配额不是“让车空着跑”，而是装货前先判断“这批货能不能安全送到”，装不下就等下一趟，而不是开到半路爆胎。

再看一个真实工程例子。客服 LLM 白天多是几十字到几百字的短问答，晚上多是长文摘要。如果你只想着把 GPU 跑满，不限制 `max_num_seqs` 和 token 预算，白天短请求会被长 prompt 的 prefill 挤压，结果是吞吐看起来不低，但首 token 延迟和 P99 延迟明显变差，用户主观感受就是“简单问题也要等很久”。

---

## 问题定义与边界

资源配额解决的是“多租户推理服务如何稳定接单”，不是“模型怎么更准”，也不是“单机压测怎么跑出最高 token/s”。这里的 **多租户**，可以先理解成“多个业务线或多个用户群共享同一批 GPU 资源”。

这个问题至少有三层边界，不能混为一谈：

| 层级 | 解决的问题 | 常见工具 |
|---|---|---|
| 集群层 | 谁能占多少 GPU | Kubernetes `ResourceQuota` |
| Pod / 实例层 | 单实例最多装多少请求和 token | vLLM 配置项 |
| 调度层 | 请求如何排序、等待、拒绝 | priority、队列、429 |

集群层配额决定“谁能拿到几张卡”，但它不知道某个实例内部已经积累了多少 KV cache。服务层配额决定“这张卡上这个实例还敢不敢继续接单”，但它通常不负责跨业务线的资源公平。调度层再往前一步，决定“超限以后是排队、降级还是直接拒绝”。

这也是为什么只做一层控制通常不够。一个典型反例是：Kubernetes 的 namespace 没超 `requests.nvidia.com/gpu`，看起来资源合规，但某个推理实例内部没有限制 token 预算，结果单卡照样 OOM。换句话说，**集群层看到的是 GPU 数量，服务层面对的是 GPU 内容量。**

本文只讨论以下边界：

- 不讨论训练，只讨论在线或准在线推理服务。
- 不讨论模型效果，只讨论容量、稳定性和可预测性。
- 不讨论单机极限压榨，只讨论真实生产中可复用的安全边界。

再给一个场景例子。A 业务线做客服问答，短输入、低容忍延迟；B 业务线做长摘要，输入长、能容忍几秒等待。即使集群层已经把 GPU 平均分给 A 和 B，如果 B 的服务实例内部没有对 `max_num_batched_tokens` 做约束，长摘要任务仍可能把同实例的短请求拖慢。这说明“分到 GPU”不等于“服务稳定”。

---

## 核心机制与推导

推理服务的容量约束，核心可以拆成三类：内存约束、并发约束、批处理约束。

先解释两个常见阶段。**prefill** 可以理解成“模型先把整段输入读完并建立上下文状态”，更偏计算密集；**decode** 可以理解成“模型每次只生成少量新 token 并持续复用历史状态”，更偏内存密集。它们都吃资源，但吃资源的方式不同。

统一写法如下：

```text
M_total = M_w + M_act + M_kv
M_kv ≈ B × L × c_kv
M_total <= Q_gpu
B <= max_num_seqs
T <= max_num_batched_tokens
```

其中：

- $Q_{gpu}$：单实例可安全使用的 GPU 预算。
- $B$：活跃序列数，也就是同时在跑的请求条数。
- $L$：平均上下文长度。
- $c_{kv}$：单个 token 的 KV 成本。
- $T$：单轮批处理里的 token 总量。

这里最重要的近似关系是：

$M_{kv} \approx B \times L \times c_{kv}$

它不追求精确到字节，但非常适合做容量反推。逻辑是：活跃请求越多，每个请求上下文越长，KV 占用越高；KV 占用越高，可继续接单的余量越少。

下面给一个最小数值推导。假设单卡是 `24GB`，服务配置 `gpu_memory_utilization=0.8`，那么可安全使用预算近似是：

$Q_{gpu} = 24 \times 0.8 = 19.2GB$

再假设模型权重加运行时固定开销一共 `15GB`，则可留给 KV 的预算约为：

$M_{kv}^{budget} = 19.2 - 15 = 4.2GB$

如果当前模型和精度设置下，每条活跃序列平均需要 `128MB` KV，那么理论最大并发近似是：

$B_{max} = \lfloor 4.2 / 0.128 \rfloor = 32$

这就是为什么很多服务层参数不是拍脑袋填数字，而是从显存预算反推出来。

机制层再往前一步，还要理解 `max_num_batched_tokens`。它不是“总上下文长度上限”，而更像“本轮调度愿意塞进 batch 的 token 预算”。如果它过大，长 prompt 的 prefill 会长期占住批处理窗口，短请求被迫等；如果它过小，长上下文请求可能根本进不来，或者在没开 `chunked prefill` 时直接暴露配置冲突。

**chunked prefill** 可以先理解成“把长输入切片，不让它一次性占满整轮预算”。它的工程意义不是神奇加速，而是更偏向保护 decode 的连续性：先保证正在生成中的请求别被饿死，再用剩余预算分段处理新来的长 prompt。这能显著降低“短请求被长输入卡住”的概率，但如果切得太碎，也会引入额外调度开销。

机制可以总结成下面这张表：

| 阶段 | 资源特征 | 风险 |
|---|---|---|
| prefill | 计算密集，token 突增 | 抢占 batch 预算 |
| decode | 内存密集，持续占 KV | 尾延迟上升 |
| chunked prefill | 分段切分 prompt | 配置不当会影响吞吐 |

因此，正确的推导顺序通常是：

1. 从单卡预算 `Q_gpu` 出发。
2. 扣除权重和运行时固定开销。
3. 把剩余显存视作 KV cache 上限。
4. 从 KV 上限反推 `max_num_seqs`。
5. 再结合请求长度分布，约束 `max_num_batched_tokens`。
6. 最后用队列长度和拒绝策略兜住峰值流量。

---

## 代码实现

工程实现要同时覆盖集群层和服务层。只写 Kubernetes YAML，不够；只调推理引擎参数，也不够。一个实用组合是：Kubernetes 限总 GPU，vLLM 限单实例并发和 token 预算，网关或服务进程限队列和 429 策略。

先看集群层最小配置：

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: llm-gpu-quota
spec:
  hard:
    requests.nvidia.com/gpu: "4"
    limits.nvidia.com/gpu: "4"
```

这段配置的意义很明确：某个 namespace 最多声明 4 张 GPU。它解决“谁能拿多少卡”，但不解决“单卡上这个实例内部还能不能继续接请求”。

再看服务层启动参数示意：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /models/your-model \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill
```

这几个参数分别控制显存安全边界、活跃并发、单轮 token 预算，以及长 prompt 的切分策略。实践里不要先问“最大能配多少”，而要先问“平均请求分布是什么、P95 长度是什么、允许的 P99 延迟是多少”。

请求进入前的容量判断，可以抽象成下面的伪代码：

```text
if pending_requests > queue_limit:
    return 429

if estimated_kv + current_kv > kv_budget:
    enqueue_or_reject()

if active_seqs + incoming_seqs > max_num_seqs:
    enqueue_or_reject()

if incoming_batched_tokens + current_batched_tokens > max_num_batched_tokens:
    split_enqueue_or_reject()
```

如果想把这个逻辑变成一个可运行的玩具程序，可以写成：

```python
from math import floor

def can_admit_request(
    gpu_total_gb: float,
    gpu_utilization: float,
    weight_and_runtime_gb: float,
    kv_per_seq_gb: float,
    active_seqs: int,
    incoming_seqs: int,
    current_batched_tokens: int,
    incoming_batched_tokens: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    pending_requests: int,
    queue_limit: int,
) -> bool:
    q_gpu = gpu_total_gb * gpu_utilization
    kv_budget = q_gpu - weight_and_runtime_gb
    current_kv = active_seqs * kv_per_seq_gb
    estimated_kv = incoming_seqs * kv_per_seq_gb

    if pending_requests > queue_limit:
        return False
    if active_seqs + incoming_seqs > max_num_seqs:
        return False
    if current_batched_tokens + incoming_batched_tokens > max_num_batched_tokens:
        return False
    if current_kv + estimated_kv > kv_budget:
        return False
    return True

# 24GB GPU, utilization 0.8 => 19.2GB budget
# weight + runtime = 15GB, KV budget = 4.2GB
# each active seq costs 0.128GB => theoretical max about 32 seqs
assert can_admit_request(
    gpu_total_gb=24,
    gpu_utilization=0.8,
    weight_and_runtime_gb=15,
    kv_per_seq_gb=0.128,
    active_seqs=30,
    incoming_seqs=2,
    current_batched_tokens=6000,
    incoming_batched_tokens=1000,
    max_num_seqs=32,
    max_num_batched_tokens=8192,
    pending_requests=10,
    queue_limit=100,
) is True

assert can_admit_request(
    gpu_total_gb=24,
    gpu_utilization=0.8,
    weight_and_runtime_gb=15,
    kv_per_seq_gb=0.128,
    active_seqs=31,
    incoming_seqs=2,
    current_batched_tokens=6000,
    incoming_batched_tokens=1000,
    max_num_seqs=32,
    max_num_batched_tokens=8192,
    pending_requests=10,
    queue_limit=100,
) is False

assert floor((24 * 0.8 - 15) / 0.128) == 32
```

这段代码当然不是完整生产实现，但它表达了最关键的思想：**是否接单，先看预算，再看并发，再看批处理，再决定排队还是拒绝。**

配置项和行为的映射关系可以再压缩成一张表：

| 配置项 | 作用 | 超限后的行为 |
|---|---|---|
| `gpu_memory_utilization` | 限单实例可用显存 | 预留安全边界 |
| `max_num_seqs` | 限活跃序列数 | 新请求等待或拒绝 |
| `max_num_batched_tokens` | 限单批 token 总量 | 长请求拆分或延后 |
| 队列长度 | 限等待请求数 | 返回 429 或降级 |

一个真实工程例子是：客服模型在线上白天承接大量短问答，请求入口按租户分优先级。高优业务允许更高队列占比，低优业务超队列直接 429；实例内部统一使用 `chunked prefill`，防止长 prompt 抢满 batch。到了夜间离峰时，再通过变更参数或切换部署池，把 `max_num_batched_tokens` 调高，用更大批处理换吞吐。这说明配额不是固定常数，而是随业务时段和 SLA 目标变化的控制面。

---

## 工程权衡与常见坑

资源配额的目标，不是把 GPU 填到 100%，而是把 P99 延迟、OOM 风险和业务争用压进可接受范围。工程上最常见的错误，是把某一个参数单独拉满，指望它带来“纯收益”。实际上这些参数高度联动。

常见坑可以先看表：

| 坑 | 表现 | 规避方式 |
|---|---|---|
| 只配集群层，不配服务层 | GPU OOM | 同时设置 token 和序列预算 |
| token 预算过小 | 长请求启动失败或被拒绝 | 先确认 `max_model_len` |
| token 预算过大 | 短请求尾延迟升高 | 分开调 prefill 与 decode |
| 只看吞吐不看延迟 | 高峰期体验差 | 加入 P95 / P99 指标 |
| 过度依赖 FCFS | 长请求头阻塞 | priority 或分池 |

第一个坑最普遍。很多团队已经在 Kubernetes 上配置了 `requests.nvidia.com/gpu`，于是误以为资源治理已经完成。实际上这只限制了“可以起几个 GPU Pod”，没有限制“每个 Pod 如何用光自己的卡”。单实例内部如果放任长上下文和高并发同时增长，依然会把显存顶爆。

第二个坑是配置矛盾。比如 `max_num_batched_tokens < max_model_len`，同时又没开 `chunked prefill`，那么长上下文请求可能根本无法进入执行，严重时甚至在启动阶段就会暴露不一致。这里的 **max_model_len**，可以理解成“模型或服务允许处理的最长上下文长度”。

第三个坑是只看平均吞吐。你把 `max_num_batched_tokens` 设大，压测报告里的总 token/s 可能变好看，但生产里短请求会被长 prefill 挤压，P99 延迟显著上升。用户不关心你的平均值，他只关心“我这一问为什么等了 8 秒”。

第四个坑是忽略请求分布。假设你根据平均输入长度调出了一个“很稳”的配额，但业务突然切到长文总结，P95 输入长度翻倍，原来的 `max_num_seqs` 立刻不再安全。这也是为什么容量参数必须和流量画像绑定，而不是写死在模板里永久不动。

第五个坑是过度依赖 FCFS。**FCFS** 就是“先来先服务”。它实现最简单，但在长短请求混合场景下，头阻塞很明显。一个超长 prompt 进入 prefill 后，后面一串短请求都得等。对有 SLA 的业务，更现实的做法通常是优先级调度，或者把长短请求拆到不同实例池中。

---

## 替代方案与适用边界

资源配额不是唯一方案，它适合的是“多租户、共享 GPU、要求稳定 SLA”的在线推理场景。如果你的任务是单租户、低并发、离线批处理，那么很多精细配额可以简化，因为你追求的是吞吐而不是交互时延。

不同方案可以这样对比：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 只限 GPU 数量 | 简单 | 无法防单卡内部超载 | 小规模、低风险 |
| 服务层 token 配额 | 可控 | 参数需要调优 | 在线推理 |
| priority 调度 | 保 SLA | 复杂度高 | 多业务混部 |
| 分池部署 | 隔离强 | 资源利用率可能下降 | 长短请求差异大 |
| FCFS | 实现简单 | 头阻塞明显 | 低并发或容忍等待 |

如果你只有一个离线摘要任务，输入长度稳定，处理窗口也宽松，完全可以把 GPU 利用率打高一些，不一定需要复杂队列和租户优先级。因为这里最重要的是总处理量，而不是每个请求的响应时间。

但如果你做的是在线客服、多团队共用 GPU 节点，情况就反过来了。此时不能依赖“大家自觉”，必须有显式配额、队列和拒绝策略，否则一个业务线的长请求尖峰就可能拖垮整个共享实例。

再把几种替代策略说得更直白一些：

- `FCFS` 适合简单低负载场景，成本最低，但对混合长度流量不友好。
- `priority` 适合有明确 SLA 的混部场景，本质是在资源不足时明确“谁先活下来”。
- 分池适合长短请求差异极大的业务，比如客服问答和长文总结分开部署。
- 纯吞吐优先适合离线任务，此时可以牺牲单请求等待时间来换总处理量。

因此，是否上资源配额，不应从“这套机制复杂不复杂”判断，而应从“你的服务是否需要稳定边界”判断。共享资源越强、SLA 越硬、流量越不均匀，配额就越接近必需品。

---

## 参考资料

下表给出资料和对应用途，方便继续查证：

| 资料 | 用途 |
|---|---|
| Kubernetes Resource Quotas | 集群层配额 |
| vLLM Optimization and Tuning | 服务参数调优 |
| vLLM Cache Config | KV cache 与显存预算 |
| NVIDIA Triton Dynamic Batching | 批处理与资源利用率 |
| Orca | 推理调度与共享执行 |

1. [Kubernetes Resource Quotas](https://kubernetes.io/docs/concepts/policy/resource-quotas/)
2. [vLLM Optimization and Tuning](https://docs.vllm.ai/en/v0.18.0/configuration/optimization/)
3. [vLLM Cache Config](https://docs.vllm.ai/en/v0.18.2/api/vllm/config/cache/)
4. [NVIDIA Triton Dynamic Batching & Concurrent Model Execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html)
5. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
