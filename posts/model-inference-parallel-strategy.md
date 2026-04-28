## 核心结论

模型推理并行的目标不是“把 GPU 数量堆上去”，而是在 `显存`、`KV cache`、`通信开销`、`尾延迟` 之间找到可运行且可盈利的平衡点。`KV cache` 是推理时为历史 token 保存的键值张量，可以理解为“模型为了继续往后生成，必须留在显存里的上下文记忆”。很多系统单卡能跑通压测样例，但一上长上下文和高并发就崩，不是算力先不够，而是显存先被吃满。

四类常见策略可以先这样记：

| 策略 | 主要解决什么问题 | 主要代价 |
|---|---|---|
| 副本并行 Replica Parallel | 请求太多，单实例吞吐不够 | 总显存线性重复，调度和负载均衡变复杂 |
| 张量并行 Tensor Parallel, TP | 单层太宽，单卡放不下或算不过来 | 层内同步频繁，强依赖高速互联 |
| 流水线并行 Pipeline Parallel, PP | 模型太深，整网放不下 | 流水线泡泡导致空转，调度复杂 |
| 连续批处理 Continuous Batching | 解码阶段 GPU 空转多、批次利用率差 | 吞吐升高时尾延迟可能变差 |

真实部署里，通常不是“只选一个”，而是组合使用。最常见的组合是：

$$
\text{GPU 总数} = R \times TP \times PP
$$

其中 $R$ 是副本数。工程上可以把它理解成：先决定单个模型副本如何切分，再决定要复制多少份去接流量。

能不能跑起来，先看每卡显存约束：

$$
W / TP + K(B, L) + O \le M
$$

这里 $W$ 是总权重占用，$K(B,L)$ 是 KV cache 占用，$O$ 是运行时额外开销，$M$ 是单卡可用显存。结论很直接：请求太多时先看副本，模型太宽先看 TP，模型太深再看 PP，空转太多再上连续批处理。

---

## 问题定义与边界

这里讨论的是**在线推理服务**，不是训练。在线推理的核心指标是三类：

| 维度 | 推理关注点 | 训练关注点 |
|---|---|---|
| 性能目标 | 吞吐、首 token 延迟、p95/p99 尾延迟 | 每步吞吐、总训练时长 |
| 资源瓶颈 | 显存、KV cache、通信、请求波动 | 显存、梯度同步、优化器状态 |
| 稳定性 | 抖动、OOM、队列堆积、超时 | 收敛、数值稳定性 |

并行策略的边界，不由“你有几张卡”单独决定，而由下面这些量共同决定：

| 符号 | 含义 |
|---|---|
| $R$ | 副本数 |
| $TP$ | 张量并行度 |
| $PP$ | 流水线并行度 |
| $B$ | 并发序列数，正在占用 KV cache 的请求数 |
| $L$ | 每条序列当前有效长度 |
| $N$ | Transformer 层数 |
| $H_{kv}$ | KV 头数 |
| $d$ | 每个 KV 头的维度 |
| $q$ | 每个元素字节数，如 FP16 约为 2 |
| $M$ | 单卡可用显存 |
| $W$ | 模型总权重占用 |
| $O$ | 运行时其他占用，如激活、框架缓存、碎片等 |

玩具例子：一个 7B 模型在单卡 24GB 上，短上下文、低并发时能正常跑。很多人因此以为“这张卡够了”。但如果把 `max_model_len` 提到 32k，再把并发序列数提到 32，权重虽然没变，KV cache 却会随 $B \times L$ 线性增长，显存马上爆掉。问题不是“算不动”，而是“记不下”。

真实工程例子：在线客服系统白天大部分请求只有 1k 到 2k token，晚上批量分析日志时会突然出现 16k 到 64k token 长上下文。平均吞吐看起来没问题，但只要调度器没限制长请求并发，几条长请求就可能把同一副本的 KV cache 打满，导致该副本 OOM 或者进入严重排队，最终把 p99 拉穿。

---

## 核心机制与推导

先看最关键的一条：推理阶段显存主要由两部分构成，权重和 KV cache。权重基本相对稳定，KV cache 则随在线请求实时变化。

KV cache 的简化估算可以写成：

$$
K(B, L) \approx 2 \times B \times L \times N \times H_{kv} \times d \times q
$$

式子里的 `2` 表示每个 token 都要保存 `K` 和 `V` 两类张量。这个式子不是精确到实现细节的字节账单，但足够指导部署。

为什么它会随并发序列数和上下文长度线性增长？因为每增加一条活跃序列，就多一份历史记忆；每多一个 token，就要在每一层再存一组 K/V。白话说，模型越“要记的东西多”，显存压力越大。

做一个玩具例子。假设：

- $B = 8$
- $L = 4096$
- $N = 32$
- $H_{kv} = 8$
- $d = 128$
- $q = 2$ bytes

则

$$
K \approx 2 \times 8 \times 4096 \times 32 \times 8 \times 128 \times 2
$$

结果约为 $4{,}294{,}967{,}296$ bytes，也就是约 4GB。注意这只是 KV cache，还没算权重和其他开销。所以“模型权重能放下”从来不等于“服务能稳定跑”。

再看四种策略的作用机制。

副本并行：直接复制多份完整模型，每个副本独立接请求。它最适合扩容量，因为副本之间几乎不做层内同步，工程复杂度最低。但它不解决单副本放不下的问题，显存成本也最高。

张量并行：把同一层内部的矩阵切到多卡上。可以理解为“一层太宽，一张桌子摆不下，就把同一层拆到多张桌子同时算”。它能降低单卡权重占用并加速大矩阵计算，但每层都要做通信，比如 all-reduce 或 all-gather，所以非常依赖 NVLink 这类高速互联。跨慢网络做大 TP，常见结果是卡没闲着，但通信把收益吃光。

流水线并行：按层切模型，把连续层段分给不同 GPU。它解决的是“模型整体太深，必须按层拆开”。缺点是有流水线泡泡。泡泡可以理解为“某些阶段在等前后工位传数据，自己暂时没活干”。

常见近似式：

$$
bubble \approx \frac{PP - 1}{m + PP - 1}
$$

其中 $m$ 是微批次数。这个式子的意思是：$PP$ 越大、$m$ 越少，空转比例越明显。比如 $PP=4, m=1$ 时，泡泡约为 $3/4$，非常差；如果 $m=16$，泡泡就降到 $3/19$，好很多。

连续批处理：不是等整批结束再上下一批，而是哪个请求有 token 要算就动态插入。它解决的是解码阶段每步算子很小、固定批次经常装不满的问题。真实服务里吞吐提升常常很明显，但代价是调度更复杂，而且长短请求混跑时，短请求的尾延迟可能被拉高。

一个简单的计算步骤可以写成：

| 步骤 | 问题 |
|---|---|
| 1 | 单卡是否能放下权重和运行时开销 |
| 2 | 若放不下，先增大 TP 或引入 PP |
| 3 | 放得下后，再估算目标并发下的 KV cache |
| 4 | 若 KV cache 不够，降低 $B/L$、增加副本，或扩更多 GPU |
| 5 | 资源足够后，再用连续批处理优化吞吐 |

真实工程例子：70B 在线服务常见做法是先在单节点内做 `TP=8`，因为层内通信走 NVLink；如果仍放不下，再考虑多节点 `PP=2`；当模型能稳定装下后，再通过连续批处理提高 token/s；只有当单副本容量和吞吐都不够时，才继续加 `R` 做横向扩容。这比一开始就盲目上 `TP+PP+R` 更稳。

---

## 代码实现

工程实现通常先解决“能放下”，再解决“跑得快”。下面给一个可运行的 Python 玩具实现，用来做显存可行性判断。它不是精确模拟框架，但足够表达部署决策逻辑。

```python
from dataclasses import dataclass

GiB = 1024 ** 3

@dataclass
class InferConfig:
    W_gib: float          # 模型总权重占用
    M_gib: float          # 单卡可用显存
    O_gib: float          # 其他开销
    B: int                # 并发序列数
    L: int                # 平均有效长度
    N: int                # 层数
    Hkv: int              # KV 头数
    d: int                # 每个头维度
    q_bytes: int          # 元素字节数
    TP: int               # 张量并行度

def kv_cache_gib(B: int, L: int, N: int, Hkv: int, d: int, q_bytes: int) -> float:
    bytes_used = 2 * B * L * N * Hkv * d * q_bytes
    return bytes_used / GiB

def fits_on_each_gpu(cfg: InferConfig) -> bool:
    per_gpu_weight = cfg.W_gib / cfg.TP
    kv_gib = kv_cache_gib(cfg.B, cfg.L, cfg.N, cfg.Hkv, cfg.d, cfg.q_bytes)
    total = per_gpu_weight + kv_gib + cfg.O_gib
    return total <= cfg.M_gib

cfg_small = InferConfig(
    W_gib=14.0, M_gib=24.0, O_gib=2.0,
    B=8, L=2048, N=32, Hkv=8, d=128, q_bytes=2,
    TP=1
)
assert fits_on_each_gpu(cfg_small) is True

cfg_oom = InferConfig(
    W_gib=14.0, M_gib=24.0, O_gib=2.0,
    B=32, L=8192, N=32, Hkv=8, d=128, q_bytes=2,
    TP=1
)
assert fits_on_each_gpu(cfg_oom) is False

cfg_tp = InferConfig(
    W_gib=14.0, M_gib=24.0, O_gib=2.0,
    B=8, L=2048, N=32, Hkv=8, d=128, q_bytes=2,
    TP=2
)
assert fits_on_each_gpu(cfg_tp) is True
```

对应到部署配置，核心参数通常是：

| 参数 | 作用 |
|---|---|
| `tensor_parallel_size` | 控制 TP |
| `pipeline_parallel_size` | 控制 PP |
| `max_model_len` | 限制单请求最大上下文长度 |
| `max_num_seqs` | 限制活跃序列数 |
| `gpu_memory_utilization` 或等价参数 | 控制显存水位 |
| `replicas` | 控制副本数 |

最小决策逻辑可以写成：

```text
if W / TP + K(B, L) + O <= M:
    run
else:
    increase TP or reduce B / L
```

但真实系统通常再加两条规则：

1. 如果 `TP` 要跨节点，先怀疑通信是否会成为瓶颈。
2. 如果 `PP` 增大后微批次太少，先怀疑流水线泡泡是否会吃掉收益。

---

## 工程权衡与常见坑

并行不是越多越好，而是越贴近瓶颈越好。下面是线上最常见的失败模式。

| 常见坑 | 为什么失败 | 规避方式 |
|---|---|---|
| 只算权重，不算 KV cache | 短上下文样例能跑，长上下文高并发时直接 OOM | 显存预算必须单独列出 $K(B,L)$ |
| TP 拉太大 | 层内同步频繁，通信时间盖过计算收益 | TP 尽量限制在同节点高速互联内 |
| PP 拉太大但微批次太少 | 流水线泡泡严重，GPU 大量空转 | 增加微批次，或降低 PP |
| 只看平均吞吐 | 平均值掩盖长请求冲击，p99 很差 | 同时监控首 token、p95、p99 |
| 长短请求混跑无隔离 | 长请求占满 KV cache，拖慢短请求 | 做队列分层或给长请求单独池化 |

还有一个容易被忽略的权衡：连续批处理通常能提升总吞吐，但它会让调度器始终追求“把卡塞满”。如果业务更看重交互体验，比如问答助手而不是离线批处理，就不能只盯 `tokens/s`，还要约束单请求等待时间。

---

## 替代方案与适用边界

不是所有场景都值得上 `TP + PP`。如果模型较小、上下文不长、并发也有限，最简单稳定的方案往往是单副本或多副本加连续批处理。少切分，就少通信，也更容易定位故障。

可以按场景粗分：

| 场景 | 推荐方案 | 不推荐原因 |
|---|---|---|
| 7B/13B，小上下文，中低并发 | 单卡或多副本 + 连续批处理 | 上 TP/PP 会引入不必要复杂度 |
| 30B/70B，单节点多 GPU，有 NVLink | 先 TP，再调连续批处理 | 直接上跨节点 PP 成本更高 |
| 超大模型，单节点放不下 | 节点内 TP，节点间 PP | 跨节点大 TP 通常通信太重 |
| 高并发 API 服务 | 多副本 + 连续批处理 + 限流 | 单纯加 TP 不会线性扩容量 |

70B 在线服务的一个典型决策过程是：先检查单卡显存，发现放不下，于是在单节点内做 TP；如果仍然不够，再把层按节点切成 PP；模型放下后，用连续批处理提高吞吐；最后若流量继续上涨，再增加副本数 $R$。顺序很重要，因为副本扩容解决的是流量问题，不解决单副本显存问题。

反例也很重要。如果你只是部署一个 8B 模型，平均上下文 2k，并发几十，直接做 4 个副本往往比复杂的 `TP=2, PP=2` 更划算。原因不是后者不能跑，而是通信、运维、故障域和调试成本都更高，而收益未必成立。

---

## 参考资料

本文的 KV cache 估算、并行切分原则和工程边界，主要参考了 vLLM、TensorRT-LLM、Megatron-LM 以及 PagedAttention 公开资料。

1. [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
2. [vLLM Paged Attention Design](https://docs.vllm.ai/en/v0.18.0/design/paged_attention/)
3. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/vllm)
4. [TensorRT-LLM Overview](https://docs.nvidia.com/tensorrt-llm/index.html)
5. [TensorRT-LLM: Deciding Model Sharding Strategy](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/deciding-model-sharding-strategy.html)
6. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
7. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
