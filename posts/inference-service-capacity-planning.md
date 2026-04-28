## 核心结论

推理服务容量规划，解决的不是“单卡最高跑分”，而是“在给定 SLO、显存预算、算力预算和真实请求分布下，系统最多能稳定接住多少流量”。

对在线 LLM 服务，容量上限通常同时受三类约束控制：

$$
capacity = \min(\text{算力上限}, \text{显存上限}, \text{排队延迟上限})
$$

这里的“算力上限”可以理解为 GPU 每秒最多处理多少 token；“显存上限”是模型权重、KV cache 和运行时开销把显存吃满前能承受的并发；“排队延迟上限”是即使机器还没满，只要请求排队时间把 `p95/p99` 顶穿，容量也已经失效。

对大语言模型，最常见的误判是：看到 `tokens/s` 还能涨，就以为还能继续加并发。真实情况往往是长上下文请求一多，KV cache 先成为瓶颈，结果不是直接 OOM，就是 `TTFT` 和 `queue time` 暴涨。

一个最小玩具例子：某服务平均响应时间 `W = 80 ms`，系统里平均同时有 `L = 10` 个请求在处理。根据

$$
L = \lambda W
$$

可得吞吐约为 $\lambda = 10 / 0.08 = 125$ req/s。表面看吞吐不错，但如果每个活跃请求平均要 `40 MB` 的 KV cache，那么 10 个请求只占 `400 MB`，继续放大并发后，先撞墙的可能是显存，不是算力。

| 指标 | 白话解释 | 用来判断什么 |
| --- | --- | --- |
| `tokens/s` | 每秒生成多少 token | 算力利用率 |
| `TTFT` | 首字时间，用户多久看到第一个 token | 交互体验 |
| `ITL` | 相邻两个输出 token 的间隔 | 生成流畅度 |
| `p95/p99` | 95%/99% 请求不超过的延迟 | 尾延迟风险 |
| `显存峰值` | GPU 显存最高占用 | OOM 风险 |
| `queue time` | 请求在真正执行前等了多久 | 排队压力 |

---

## 问题定义与边界

容量规划的输入不是单一 QPS，而是四类信息一起决定：

| 维度 | 内容 |
| --- | --- |
| 输入 | `SLO / 请求分布 / GPU 数 / 显存预算` |
| 输出 | `最大稳定负载 / 实例数 / batch / KV cache / 安全余量` |
| 约束 | `延迟 / OOM / 吞吐 / 成本` |
| 适用对象 | `LLM 在线推理` |

这里的 `SLO` 是服务等级目标，白话说就是“系统必须稳定达到的延迟和可用性要求”。本文只讨论在线 LLM 推理，不讨论训练、离线批处理，也不讨论纯 CPU 检索服务。

为什么不能只按“请求数”算容量？因为两个都是 `100 req/s` 的系统，资源消耗可能完全不同。短问答请求可能只有 `200` 个输入 token、`80` 个输出 token；长上下文 RAG 请求可能有 `8000` 个输入 token、`300` 个输出 token。前者更像短平快，后者会把 prefill 和 KV cache 压得很重。

`prefill` 是把输入上下文一次性编码进模型的阶段，白话说就是“先把题目读完”；`decode` 是逐 token 生成输出的阶段，白话说就是“再一个字一个字往外说”。二者对资源的要求不同：

- `prefill` 更偏算力密集，对长输入特别敏感。
- `decode` 更偏持续吞吐和 KV cache 占用，输出越长、并发越高，压力越大。

首次出现的关键术语如下：

| 术语 | 一句话解释 |
| --- | --- |
| `TTFT` | Time To First Token，用户等到第一个字出现的时间 |
| `ITL` | Inter-Token Latency，相邻输出 token 之间的时间 |
| `tokens/s` | 每秒处理或生成多少 token |
| `KV cache` | 模型为后续生成保留的上下文状态缓存，避免每步重算全部历史 |

---

## 核心机制与推导

容量直觉先从排队关系建立。`L = \lambda W` 是 Little's Law，白话说就是“系统里平均堆着多少活，就等于到达速度乘以每个活停留多久”。

$$
L = \lambda W
$$

当请求到达率 $\lambda$ 上升时，只要平均响应时间 $W$ 因排队或资源竞争变长，系统中的在途请求数 $L$ 就会继续膨胀，于是又进一步加重排队。这就是为什么很多服务不是“突然死掉”，而是先出现 `p95/p99` 明显恶化。

LLM 的显存可以粗分为三块：

$$
M_{total} \approx M_{weights} + M_{KV} + M_{runtime}
$$

- `M_weights`：模型权重，占一块比较固定的显存。
- `M_KV`：KV cache，占用会随着在途 token 增长。
- `M_runtime`：运行时临时张量、调度开销、碎片等。

其中最需要盯住的是：

$$
M_{KV} \propto \text{层数} \times \text{在途 token 数} \times \text{head\_dim} \times \text{bytes}
$$

这不是精确公式，但工程上足够说明核心结论：在模型结构和精度固定后，KV cache 基本随着“同时挂在系统里的 token 总量”线性变大。也就是说，并发越高、上下文越长、输出越长，显存压力越大。

阶段差异可以这样看：

| 阶段 | 主要工作 | 典型瓶颈 | 更敏感的指标 |
| --- | --- | --- | --- |
| `prefill` | 编码整段输入上下文 | 算力、带宽 | `TTFT` |
| `decode` | 逐 token 生成输出 | 吞吐、KV cache | `ITL`、`tokens/s` |

请求路径也可以抽象成一个简单流程：

`到达请求 -> 排队 -> prefill -> decode -> 输出`

这条链路里，长请求混入短请求时最容易出问题。因为长请求的 `prefill` 很重，如果调度不当，会阻塞本来很快就能首字返回的短请求，导致客服、Copilot、搜索问答这类对 `TTFT` 敏感的业务体验明显变差。

真实工程例子：一个线上客服机器人同时处理“订单查询”和“知识库问答”。前者输入短，用户最在意首字响应；后者是 RAG，请求里带 6k 到 8k token 的检索上下文。若两者直接混跑，一个长 RAG prefill 就可能把一批短请求的 `TTFT` 一起拉高。常见处理方式是按长度分流，或者启用 `chunked prefill`，即把超长 prefill 切成块，减少对 decode 的独占。

几种常见机制的作用边界如下：

| 机制 | 主要解决什么问题 | 代价 |
| --- | --- | --- |
| `dynamic batching` | 把短时间内到达的请求合并，提高利用率 | 等待时间增加 |
| `continuous batching` | 让不同请求在解码阶段持续拼批，提高吞吐 | 调度更复杂 |
| `chunked prefill` | 避免超长输入一次性堵住 GPU | 调参复杂，收益依赖请求分布 |
| `paged KV cache` | 降低 KV cache 碎片和浪费 | 实现复杂度更高 |

所以容量规划不是找一个数字，而是在三条线之间找平衡：显存余量、GPU 利用率、尾延迟预算。

---

## 代码实现

下面给一个最小可运行的 Python 估算器。它不是精确模拟器，而是一个“先粗估、再压测修正”的工程工具。

```python
from math import ceil

def plan_capacity(
    traffic_mix,
    req_per_sec,
    avg_latency_s,
    gpu_mem_gb,
    model_weight_gb,
    runtime_overhead_gb,
    kv_mb_per_token,
    target_p95_s,
    safety_margin=0.15,
):
    # traffic_mix: [{"ratio": 0.7, "input_tokens": 300, "output_tokens": 120}, ...]
    assert abs(sum(x["ratio"] for x in traffic_mix) - 1.0) < 1e-9
    assert gpu_mem_gb > model_weight_gb + runtime_overhead_gb
    assert 0 < safety_margin < 0.5

    avg_input_tokens = sum(x["ratio"] * x["input_tokens"] for x in traffic_mix)
    avg_output_tokens = sum(x["ratio"] * x["output_tokens"] for x in traffic_mix)
    avg_total_tokens = avg_input_tokens + avg_output_tokens

    # Little's Law: L = lambda * W
    concurrency_by_latency = req_per_sec * avg_latency_s

    usable_mem_gb = gpu_mem_gb * (1 - safety_margin)
    kv_budget_gb = usable_mem_gb - model_weight_gb - runtime_overhead_gb
    assert kv_budget_gb > 0

    kv_per_request_gb = avg_total_tokens * kv_mb_per_token / 1024
    concurrency_by_memory = int(kv_budget_gb // kv_per_request_gb)
    assert concurrency_by_memory > 0

    # 用较保守的方式取容量上限
    recommended_concurrency = min(int(concurrency_by_latency), concurrency_by_memory)

    # 简化 batch 建议：短请求更适合大些，长请求更适合小些
    if avg_input_tokens < 512:
        batch_size = min(16, max(4, recommended_concurrency))
    elif avg_input_tokens < 2048:
        batch_size = min(8, max(2, recommended_concurrency // 2))
    else:
        batch_size = min(4, max(1, recommended_concurrency // 2))

    # 如果平均延迟已经接近目标 p95，实例数按保护性方式放大
    pressure = avg_latency_s / target_p95_s
    scale_factor = 1.0 if pressure < 0.6 else 1.3 if pressure < 0.8 else 1.6
    instance_count = max(1, ceil(req_per_sec / max(1, recommended_concurrency) * scale_factor))

    return {
        "avg_input_tokens": round(avg_input_tokens, 1),
        "avg_output_tokens": round(avg_output_tokens, 1),
        "recommended_concurrency": recommended_concurrency,
        "recommended_batch_size": batch_size,
        "estimated_instance_count": instance_count,
        "concurrency_by_latency": round(concurrency_by_latency, 2),
        "concurrency_by_memory": concurrency_by_memory,
    }

# 玩具例子：短问答 70%，长 RAG 30%
result = plan_capacity(
    traffic_mix=[
        {"ratio": 0.7, "input_tokens": 300, "output_tokens": 120},
        {"ratio": 0.3, "input_tokens": 6000, "output_tokens": 220},
    ],
    req_per_sec=20,
    avg_latency_s=0.8,
    gpu_mem_gb=80,
    model_weight_gb=28,
    runtime_overhead_gb=8,
    kv_mb_per_token=0.0025,
    target_p95_s=1.5,
)

assert result["recommended_concurrency"] > 0
assert result["recommended_batch_size"] >= 1
assert result["estimated_instance_count"] >= 1
print(result)
```

这段代码把理论公式映射成了工程变量：

| 参数 | 含义 |
| --- | --- |
| `avg_input_tokens` | 平均输入长度 |
| `avg_output_tokens` | 平均输出长度 |
| `gpu_mem_gb` | 单卡显存 |
| `target_p95` | 目标尾延迟 |
| `kv_mb_per_token` | 每个 token 预计消耗多少 KV cache |
| `safety_margin` | 安全余量，防止估算贴边 |

它的思路就是三步：

1. 用 `L = λW` 估一个延迟允许下的并发。
2. 用 `M_total` 和 `KV budget` 估一个显存允许下的并发。
3. 取较小值作为保守建议，再据此反推 batch 和实例数。

真实工程里，建议把短问答和长 RAG 分开计算。因为同一个 batch 策略，不可能同时对二者都最优。短请求往往更看重 `TTFT`，长请求更看重显存预算和持续吞吐。

---

## 工程权衡与常见坑

容量规划里最危险的误区，是把“高吞吐”误当成“高容量”。batch 变大，GPU 利用率通常会上升，但等待合批的时间也会上升，于是 `TTFT`、`queue time` 和 `p95/p99` 可能先坏掉。

| 常见坑 | 后果 |
| --- | --- |
| 只看平均吞吐 | 尾延迟失控 |
| 混压短长请求 | 容量高估 |
| 忽略 KV cache 碎片 | 提前 OOM |
| 盲目加 batch | 排队恶化，首字变慢 |
| 只扩实例不看显存 | 单卡并发反而下降 |

一个典型反例：压测时全部用 `128` token 的短 prompt，结果 `tokens/s` 很高、显存也稳定，于是得出“每卡可承载 40 req/s”的结论。上线后，30% 流量变成 `8k` 上下文 RAG，请求一进来，prefill 变重、KV cache 变大，原结论立刻失效。

另一个常见坑，是把 prefill 和 decode 混成一个总时延指标。这样会掩盖真实瓶颈。比如某次优化后，总吞吐提升了，但其实是 decode 更快了；与此同时，长上下文 prefill 反而更容易阻塞，导致 `TTFT` 变差。对用户来说，这种“看起来 TPS 变高，实际首字更慢”的优化可能是负收益。

不同配置的权衡可以简化成下表：

| 方案 | 优点 | 风险 |
| --- | --- | --- |
| `batch 大` | 利用率高、吞吐好 | `TTFT` 和排队变差 |
| `batch 小` | 首字更快 | 吞吐低，成本高 |
| `按请求长度分流` | 减少长请求拖慢短请求 | 实现和运维复杂 |
| `chunked prefill` | 降低长上下文独占 | 调优成本高 |

工程上更可靠的做法不是“找一个万能 batch”，而是持续观测这些指标：

- `TTFT`
- `ITL`
- `tokens/s`
- `显存峰值`
- `queue time`

如果其中任何一项在高峰期开始异常，你的容量模型就需要回填修正，而不是继续沿用旧压测结果。

---

## 替代方案与适用边界

不是所有系统都值得做复杂容量模型。方法选择要看业务规模、请求分布复杂度和 SLO 严格程度。

| 方法 | 适合什么场景 | 局限 |
| --- | --- | --- |
| 纯经验法 | 小流量、内部环境 | 可迁移性差 |
| 公式估算法 | 方案初选、预算评估 | 精度有限 |
| 压测驱动法 | 上线前定标 | 依赖压测样本真实性 |
| 排队论法 | 在线服务、要看尾延迟 | 建模成本较高 |
| 生产观测回填法 | 稳定迭代优化 | 需要完整监控体系 |

适用边界也很明确：

| 场景 | 推荐做法 |
| --- | --- |
| 小流量 / 单一请求形态 / 成本优先 | 用简化规则和少量压测即可 |
| 高并发 / 长短混合 / 严格 SLO | 必须看真实分布、拆分阶段、纳入 KV cache |

可以用一个简单决策准则：

- 如果请求形态单一、上下文短、SLO 宽松，可以先用“单卡吞吐 + 显存余量”的简化模型。
- 如果存在长短混合、明显峰谷、强交互要求，就必须按 `prefill/decode` 分阶段建模，并用真实流量分布压测。
- 如果线上已经出现 `p95` 抖动、显存水位贴边、偶发 OOM，就不能只看 QPS，必须把 KV cache 和队列时间纳入容量规划。

一个真实工程判断例子：内部测试环境每天只有几千次请求，主要是短问答，这时用“每卡保守承载 5 到 10 并发”的粗估可能就够了。对外客服系统则不同，它要面对活动峰值、长知识库上下文和严格首字 SLA，这时如果还用粗估法，容量结论通常会偏乐观。

---

## 参考资料

| 资料 | 支撑哪部分结论 | 适合看什么 |
| --- | --- | --- |
| Triton Batchers | 动态批处理机制 | 看 batch 如何提升利用率 |
| Triton Model Analyzer | 压测与参数搜索 | 看实例数、batch、显存联合调优 |
| vLLM 论文 | PagedAttention、内存管理 | 看为什么 KV cache 会成为关键瓶颈 |
| vLLM Optimization | 连续批处理与调优参数 | 看线上吞吐和延迟怎么平衡 |
| TensorRT-LLM KV Cache | KV cache 机制说明 | 看缓存管理和显存行为 |

1. [NVIDIA Triton Inference Server: Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
2. [NVIDIA Triton Inference Server: Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html)
3. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
4. [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html)
5. [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html)
