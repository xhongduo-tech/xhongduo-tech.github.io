## 核心结论

`Prefill` 是“把整段输入一次性读完并算出中间状态”的阶段，白话说就是先把题干全部过一遍；`Decode` 是“基于已有状态逐 token 往后写”的阶段，白话说就是一个字一个字继续写答案。两者虽然都属于推理，但资源瓶颈不同：`Prefill` 通常更像 `compute-bound`，意思是主要被矩阵乘等算力吃满；`Decode` 通常更像 `memory-bound`，意思是主要被反复读取 `KV cache` 的带宽和显存访问拖住。

把两者强行放在同一批 GPU 上做 `colocated serving`，问题不在于“能不能跑”，而在于“不同形态的负载互相卡住”。长 prompt 的 `Prefill` 会占住 decode 槽位，直接把首 token 延迟抬高；而 decode 阶段为了维持低 `TPOT`，又会反过来压缩 prefill 的并发空间。`Disaggregated Prefill/Decode` 的核心就是把这两段拆到不同 GPU 实例，让前者按算力扩容，后者按显存带宽和 KV 容量扩容。

一个最小直观例子：某请求有 512 个输入 token。若采用 P/D 分离，`Prefill` 节点先完成 prompt 的前向计算，产出 `KV cache`，再把它传给 `Decode` 节点继续生成。这样，长 prompt 的一次性重计算不会堵住其他正在逐 token 输出的请求。对系统指标来说，可以把首 token 延迟近似写成：

$$
TTFT \approx T_{prefill} + T_{transfer} + T_{queue}
$$

而每个输出 token 的平均开销近似写成：

$$
TPOT \approx T_{decode}(token)
$$

工程上最重要的结论不是“分离一定更快”，而是“分离后能分别优化各自瓶颈”。DistServe 论文把这种收益定义到 `goodput` 上，即“在满足延迟约束前提下真正能接住的吞吐”。公开论文与官方系统文档都表明，在长 prompt、高并发、KV 足够大的条件下，P/D 分离常能带来约 `1.5x-3x` 量级的系统收益，某些受严格 SLO 约束的场景甚至更高。

| 方案 | 吞吐特征 | 延迟特征 | 资源利用 | 典型问题 |
| --- | --- | --- | --- | --- |
| `colocated` | 实现简单，但 prefill 和 decode 互相抢资源 | 长 prompt 容易拉高 `TTFT` | GPU 角色固定，难按瓶颈扩容 | 同时兼顾 `TTFT` 与 `TPOT` 很难 |
| `P/D 分离` | 更容易提升 `goodput` | 能隔离长 prefill 对 decode 的干扰 | 可分别为 P 与 D 配置不同规模 | 需要解决 KV 传输与调度成本 |

---

## 问题定义与边界

这里的问题不是“模型推理慢”这么笼统，而是“同一个实例同时承担两种不同形态的工作”。如果一个系统既要处理 8K prompt 的长上下文摘要，又要处理大量短对话续写，那么同一批 GPU 会在两种压力之间来回切换：一会儿被大矩阵乘打满，一会儿被 KV 读写和显存碎片卡住。

为什么长 prompt 会放大这个问题？因为 `KV cache` 会随着输入长度线性增长。`KV cache` 就是注意力层为后续生成保留下来的键值状态，白话说是“后面继续写时要反复翻看的草稿”。其规模可近似写成：

$$
S_{kv} = 2 \cdot L \cdot P \cdot H_{kv} \cdot b
$$

其中，`L` 是层数，`P` 是 prompt token 数，`H_kv` 是每层 KV 宽度，`b` 是每个元素的字节数，前面的 `2` 表示同时存 `K` 和 `V` 两份。`P` 越长，`S_kv` 越大，prefill 结束后要保留和搬运的状态也越大。

玩具例子可以说明边界。假设一个聊天请求只有 16 个输入 token，输出 32 个 token。此时 `Prefill` 很短，本地算完几乎不构成瓶颈。如果还要额外做远程调度、跨节点登记、传输 KV，那么新增的 `T_transfer + T_queue` 可能比 `T_prefill` 本身还大，整体反而变慢。这说明 P/D 分离不是默认更优，而是条件性优化。

| prompt 长度 | 并发量 | KV cache 大小 | 是否推荐分离 | 原因 |
| --- | --- | --- | --- | --- |
| 很短，如 `16-64` | 低到中 | 小 | 否 | 传输与调度开销可能大于收益 |
| 中等，如 `512-2K` | 中到高 | 中 | 视情况 | 看 `TTFT` SLO、prefix cache 命中和带宽 |
| 很长，如 `4K-32K+` | 高 | 大 | 是 | prefill 干扰 decode 更明显，拆分更容易回本 |
| 很长但 `prefix cache hit` 高 | 中 | 有效新增 KV 小 | 未必 | 命中后 prefill 可能转向 memory-bound，本地更划算 |

还有一个边界经常被忽略：如果 `prefix cache` 命中率很高，新增 prefill 的真实长度会明显变短。`prefix cache` 是“重复前缀直接复用历史 KV”的缓存，白话说是“开头相同的部分不必重算”。这时远程 prefill 的意义会下降，甚至应该改成条件分离，只把真正长、真正重的 prefill 送出去。

---

## 核心机制与推导

P/D 分离的依赖关系其实很简单：`Prefill` 先写出 `KV cache`，`Decode` 再读取它继续生成。这是一个典型的“先写后读”链路。难点不在概念，而在工程细节：怎么把 KV 高效传过去，怎么让传输不阻塞推理线程，怎么避免远端队列把收益吞掉。

对新手可以用一个不失真的玩具例子理解。把一次请求想成两个人接力做题。甲擅长快速把整道题的已知条件整理成草稿，这对应 `Prefill`；乙擅长根据草稿连续补出后续答案，这对应 `Decode`。如果两个人抢同一张桌子和同一支笔，就会互相打断；如果甲整理完草稿后直接塞进传递箱，乙拿到就继续写，整体节奏会更稳。这里的“草稿”就是 `KV cache`。

从时间构成上看，首 token 延迟不是只看 prefill 算多久，而是三段之和：

$$
TTFT \approx T_{prefill} + T_{transfer} + T_{queue}
$$

其中：
- `T_prefill`：prompt 前向计算时间。
- `T_transfer`：KV 传输时间，粗略可估为

$$
T_{transfer} \approx \frac{S_{kv}}{BW}
$$

这里 `BW` 是有效带宽。
- `T_queue`：在 prefill 队列、decode 队列、路由层等待的时间。

如果 `T_transfer + T_queue` 远小于“共置时 prefill 对 decode 的阻塞代价”，分离就值得；反过来，如果网络一般、请求又短，收益就可能被吃掉。

| 组件 | 职责 | 主要瓶颈 | 关键输出 |
| --- | --- | --- | --- |
| `Prefill worker` | 计算 prompt，生成 KV | 算力、批处理效率 | `KV cache` 与传输元数据 |
| `Transfer layer` | 传 KV 到 decode 端 | 带宽、零拷贝能力 | 远端可读的 KV |
| `Decode worker` | 逐 token 生成 | 显存带宽、KV 访问 | 输出 token 流 |
| `Scheduler/Router` | 选路、准入、回退 | 队列控制、负载均衡 | 决定本地算还是远程算 |

真实工程例子更能说明这套机制为什么不是纸上谈兵。NVIDIA Dynamo 的官方设计文档明确把分离流程拆成三步：prefill 生成 KV，KV 传给 decode，decode 继续计算；并强调高性能的关键是非阻塞 KV 传输。Mooncake 的公开资料则展示了以 `KV cache` 为中心的分离架构，并给出生产级吞吐数据。这说明业界不是在争论“能否分离”，而是在优化“怎样把分离做得足够便宜”。

---

## 代码实现

实现时的核心不是把一个服务拆成两个进程，而是把“路由、缓存、传输、回收”四件事讲清楚。下面这段 Python 是一个可运行的玩具实现，它不依赖 GPU，但能把请求如何决定是否分离、如何估算传输成本、以及什么时候回退到本地计算表达出来。

```python
from dataclasses import dataclass

@dataclass
class Request:
    prompt_tokens: int
    output_tokens: int
    prefix_cache_hit_tokens: int = 0

@dataclass
class Cluster:
    prefill_queue: int
    decode_queue: int
    kv_bandwidth_bytes_per_ms: float
    prefill_threshold_tokens: int
    queue_threshold: int

def kv_size_bytes(layers: int, prompt_tokens: int, h_kv: int, bytes_per_elem: int) -> int:
    # S_kv = 2 * L * P * H_kv * b
    return 2 * layers * prompt_tokens * h_kv * bytes_per_elem

def should_disaggregate(req: Request, cluster: Cluster) -> bool:
    effective_prefill = max(0, req.prompt_tokens - req.prefix_cache_hit_tokens)
    return (
        effective_prefill >= cluster.prefill_threshold_tokens
        and cluster.prefill_queue < cluster.queue_threshold
    )

def estimate_ttft_ms(req: Request, cluster: Cluster, layers=32, h_kv=4096, bytes_per_elem=2) -> float:
    effective_prefill = max(0, req.prompt_tokens - req.prefix_cache_hit_tokens)

    # 玩具模型：prefill 近似线性，短请求本地更省事
    t_prefill = effective_prefill * 0.08
    t_queue = cluster.prefill_queue * 2.0 + cluster.decode_queue * 1.0

    if should_disaggregate(req, cluster):
        s_kv = kv_size_bytes(layers, effective_prefill, h_kv, bytes_per_elem)
        t_transfer = s_kv / cluster.kv_bandwidth_bytes_per_ms
        return t_prefill + t_transfer + t_queue
    else:
        # 本地无远程传输，但 colocated 会被 decode 干扰
        interference = max(0, cluster.decode_queue - 2) * 5.0
        return t_prefill + t_queue + interference

# 长 prompt、高并发：更适合分离
cluster = Cluster(
    prefill_queue=1,
    decode_queue=8,
    kv_bandwidth_bytes_per_ms=5_000_000,
    prefill_threshold_tokens=512,
    queue_threshold=4,
)
long_req = Request(prompt_tokens=2048, output_tokens=256, prefix_cache_hit_tokens=0)
assert should_disaggregate(long_req, cluster) is True

# 短 prompt：不该为了分离额外付传输代价
short_req = Request(prompt_tokens=32, output_tokens=64, prefix_cache_hit_tokens=0)
assert should_disaggregate(short_req, cluster) is False

# 高前缀命中：虽然原始 prompt 长，但真实 prefill 不长，也可能不分离
cached_req = Request(prompt_tokens=2048, output_tokens=128, prefix_cache_hit_tokens=1800)
assert should_disaggregate(cached_req, cluster) is False

# 基本数值应为正
assert estimate_ttft_ms(long_req, cluster) > 0
assert estimate_ttft_ms(short_req, cluster) > 0
```

这段代码故意简化了大量细节，但保留了真正重要的判断逻辑：

1. `router` 先看“有效 prefill 长度”而不是原始 prompt 长度。
2. 再看队列状态，避免 prefill 端已经堵住时还继续远程拆分。
3. 只有在长 prompt 且队列健康时，才让 KV 走远程传输。
4. `decode` 端拿到 KV 后继续流式生成，并在请求结束后回收块。

| 模块 | 最小职责 | 生产实现会补充什么 |
| --- | --- | --- |
| `router` | 判断本地还是远程 prefill | SLO 感知、prefix 命中感知、回退策略 |
| `prefill worker` | 计算 KV | 专用批处理、不同 TP 配置 |
| `transfer layer` | 搬运 KV | RDMA、NIXL、NCCL、零拷贝 |
| `decode worker` | 连续生成 | continuous batching、流式输出 |
| `memory manager` | 分配与回收 KV | 块管理、碎片治理、超额保护 |

真实工程里常见的链路是：`PrefillRouter` 收到请求，判断是否走分离路径；`Prefill worker` 计算 prompt 并产出传输元数据；`Transfer layer` 用 RDMA/NIXL/NCCL 之类的机制把 KV 送到 decode 端；`Decode worker` 继续生成；`Memory manager` 在完成后释放中间状态。重点是“传输不能卡住推理线程”，否则你只是把阻塞从本地搬到了网络上。

---

## 工程权衡与常见坑

第一类坑是带宽不够。很多人只看到“prefill 是算力密集，decode 是内存密集”，于是立刻想拆开，但忽略了中间那个最贵的对象恰恰是 `KV cache`。如果 `S_kv` 很大，而有效带宽 `BW` 不够，那么 `T_transfer = S_kv / BW` 会迅速放大，尤其在长上下文和大模型下更明显。此时同机 NVLink、同节点放置，通常比跨机随意搬运更稳。

第二类坑是短请求误拆分。一个 32 token 的聊天请求，本地 prefill 也许几十毫秒就结束了，但如果被错误路由到远端，额外的排队、注册、传输、回写都会变成纯损耗。Dynamo 官方文档专门强调 `conditional disaggregation`，本质上就是别把所有请求一刀切地远程化。

第三类坑是 decode 端内存被写满。因为 decode 节点一边服务正在生成的请求，一边接收新到的 KV，若准入控制做得差，突发流量会把可用块瞬间打空。更稳的做法通常是 `pull` 式队列和容量感知调度，让 prefill 端不要无限制地主动往 decode 端灌状态。

| 常见坑 | 表现 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| 带宽不足 | `TTFT` 反而升高 | `KV cache` 太大，`T_transfer` 吞掉收益 | 优先 NVLink/IB，同节点优先，压缩传输路径 |
| 短请求误拆分 | 小请求变慢 | 调度不看有效 prefill 长度 | 设置长度阈值与本地回退 |
| 内存被写满 | decode 抖动、OOM、拒绝服务 | KV 写入速度超过消费速度 | 容量感知准入、`pull` 队列、分级回收 |
| `chunked prefill` 过度切分 | 重复算、吞吐变差 | 小块过多导致历史状态反复装载 | 只对超长请求切块，限制切分粒度 |
| 只看总吞吐 | 指标“好看”但体验差 | 忽略 `TTFT` 与 `TPOT` 的分离 | 按 SLO 分别监控 `TTFT`、`TPOT`、`goodput` |

这里要特别区分三个指标。`TTFT` 是首 token 时间，白话说是“用户多久看到第一口响应”；`TPOT` 是每个输出 token 的平均时间，白话说是“后续写字快不快”；`goodput` 则是在满足前两者约束后真正能接住的吞吐。只盯总吞吐，可能会把系统调成“机器很忙、用户很慢”的状态。

---

## 替代方案与适用边界

P/D 分离不是唯一答案。很多系统在落地时，真正用的是几种方法组合，而不是纯分离或纯共置。`continuous batching` 是“持续把新请求并进批次”，白话说是让 GPU 尽量不停工；`chunked prefill` 是“把超长输入拆段计算”；`prefix cache` 是“复用重复前缀的历史 KV”；`colocated serving` 则是“prefill 和 decode 仍在同一实例里处理”。

一个常见真实判断式可以写成：

$$
T_{transfer} + T_{queue} < T_{interference\_saved}
$$

左边是分离新增的成本，右边是分离后省掉的共置干扰。如果左边不小于右边，就不值得拆。对大量短对话、前缀高度复用、或者机房网络一般的场景，`continuous batching + prefix cache` 往往比强行做 P/D 分离更稳、更便宜。

| 方案 | 适合请求形态 | 优点 | 缺点 | 是否需要跨节点 KV 传输 |
| --- | --- | --- | --- | --- |
| `colocated serving` | 短请求、低到中并发 | 简单、稳定、链路短 | 长 prompt 易干扰 decode | 否 |
| `P/D 分离` | 长 prompt、高并发、严格 SLO | 易隔离瓶颈、易按角色扩容 | 调度和传输复杂 | 是 |
| `chunked prefill` | 超长上下文 | 降低单次长 prefill 峰值 | 过度切分会有重复成本 | 不一定 |
| `prefix cache` | 高重复前缀、多轮对话 | 直接减少 prefill 重算 | 命中依赖业务形态 | 否 |
| `continuous batching` | 大量中短请求 | 提升整体利用率 | 难彻底消除长 prefill 干扰 | 否 |

所以更准确的结论是：P/D 分离适合“长 prompt 明显拖累 decode”的系统，不适合“prefill 本来就不重”或“远程传输明显更贵”的系统。它是一个强力但有门槛的架构武器，不是推理服务的默认模板。

---

## 参考资料

资料用途表：

| 来源 | 内容类型 | 适合阅读阶段 | 能解决的问题 |
| --- | --- | --- | --- |
| DistServe | 论文 | 先读 | 为什么要分离，如何用 `goodput` 衡量收益 |
| NVIDIA Dynamo 文档 | 官方设计文档 | 第二步 | 路由、条件分离、非阻塞 KV 传输怎么做 |
| Mooncake 官方站点 | 官方系统背景 | 第三步 | 生产系统为何围绕 KV 设计 |
| Mooncake 性能页 | 官方性能结果 | 第三步 | P/D 分离对 `TTFT/ITL/吞吐` 的实际影响 |

1. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
2. [DistServe PDF](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)
3. [NVIDIA Dynamo: Disaggregated Serving](https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving)
4. [NVIDIA Dynamo Router: Disaggregated Serving](https://docs.nvidia.com/dynamo/components/router/disaggregated-serving)
5. [Mooncake 官方站点](https://kvcache-ai.github.io/Mooncake/)
6. [Mooncake PD Disaggregation Performance](https://kvcache-ai.github.io/Mooncake/performance/sglang-benchmark-results-v1.html)
