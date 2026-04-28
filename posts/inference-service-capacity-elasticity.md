## 核心结论

推理服务容量弹性，本质上不是“机器不够就加机器”，而是**在延迟 SLO 约束下动态调整可服务容量**。SLO 是服务等级目标，白话说就是“系统答应用户的延迟上限，例如 p95 小于 300ms”。这里的容量，不只由副本数决定，还由单副本并发、批大小、单批耗时、显存占用和排队策略共同决定。

用统一记号表示：

| 记号 | 含义 | 白话解释 |
|---|---|---|
| `r` | 副本数 | 同一个模型服务开了几个实例 |
| `b` | 批大小 | 一次合并多少个请求一起算 |
| `s(b)` | 批处理耗时 | 处理大小为 `b` 的批次需要多久 |
| `λ` | 到达率 | 每秒进来多少个请求 |
| `μ(b)=b/s(b)` | 单副本等效吞吐 | 一个副本每秒大约能处理多少请求 |
| `C≈r*μ(b)` | 系统容量 | 全部副本合起来的近似吞吐上限 |
| `L≈W_q+s(b)` | 总延迟 | 排队时间加上真正计算时间 |

核心判断只有一句：如果 `λ < C`，系统通常可稳定；如果 `λ` 长时间接近或超过 `C`，队列会堆积，尾延迟会上升。

一个玩具例子很直观。假设单个 GPU 副本在 `b=8` 时有 `s(8)=0.24s`，则：

$$
\mu(8)=\frac{8}{0.24}\approx 33.3\ \text{req/s}
$$

如果高峰流量 `λ=80 req/s`，则至少需要：

$$
r \ge \lceil 80 / 33.3 \rceil = 3
$$

工程上通常不会只配 3，因为还要留抖动余量，常见做法是配 4。反过来，如果只把副本数从 1 加到 4，但完全不做 batching，吞吐会涨，但 GPU 利用率可能仍不理想；如果只做 batching 不扩副本，高峰时排队仍会失控。结论是：**容量弹性一定是多变量联动，不是单点优化。**

---

## 问题定义与边界

这个问题要解决的是：**流量会波动，用户又要求稳定延迟，系统如何在成本和性能之间自动找平衡。** 对初级工程师来说，可以把它理解成“高峰别炸，低谷别浪费”。

这里有一个容易混淆的点：容量弹性不是任何推理任务都适合。它成立依赖几个前提。

| 适用场景 | 原因 | 不适用或效果差的场景 | 原因 |
|---|---|---|---|
| 短请求分类、Embedding、重排序 | 请求形态接近，容易合批 | 极长上下文、长度差异极大 | 批内拖尾严重，排队不可预测 |
| GPU 推理 | 批处理常能显著提高吞吐 | 强状态依赖请求 | 难以自由并行或迁移 |
| 可控在线服务 | 能定义延迟 SLO 和扩缩容阈值 | 离线批处理 | 目标不是在线延迟 |
| LLM 连续解码服务 | continuous batching 有明显收益 | 显存已长期逼近上限 | 再加并发只会恶化延迟 |

边界最重要的两条：

第一，**不能只按 QPS 看容量**。QPS 是每秒请求数，白话说就是“单位时间来了多少单”。对传统短请求，QPS 还算有代表性；但在 LLM 场景，请求成本更接近“上下文 token 数 + 生成 token 数”。两个都是 1 req/s，一个可能只生成 20 个 token，另一个可能带 16k 上下文并生成 1k token，资源消耗完全不同。

第二，**显存常常比计算更早成为瓶颈**。KV cache 是 LLM 解码阶段保存历史上下文状态的缓存，白话说就是“模型为了继续往后生成，必须记住前面内容而占用的显存”。因此 LLM 服务往往要满足：

$$
N_{run}\cdot m_{kv}(context, decode) + M_{model} \le M_{gpu}
$$

其中 `N_run` 是当前并行请求数，`m_kv` 是单请求 KV cache 显存开销，`M_model` 是模型本体显存，`M_gpu` 是总显存。这个约束一旦被打满，继续加并发不会提升有效吞吐，只会增加排队、抢占和 OOM 风险。

所以容量弹性的边界不是“能不能自动扩容”，而是“你的请求是否足够可并行、可合批，并且显存与排队是否还留有操作空间”。

---

## 核心机制与推导

这个问题可以分三层机制理解。

第一层是**扩副本**。横向扩容最直接：`r` 增大，总容量 `C≈r*μ(b)` 近似线性增加。这是最容易解释、也最容易做的手段。

第二层是**batching**。batching 就是把多个请求合在一次前向计算里做。它之所以成立，是因为 `s(b)` 往往不会随着 `b` 线性增长。只要：

$$
s(b) < b \cdot s(1)
$$

就说明合批后的单位请求成本下降了，因此：

$$
\mu(b)=\frac{b}{s(b)}
$$

会随着 `b` 增大而提高。

第三层是**队列与显存驱动的扩缩容**。队列表示“已经来了但还没处理”的请求集合，白话说就是“门口排队的人”。当 `λ` 接近 `C` 时，等待时间 `W_q` 会快速上升，因此总延迟：

$$
L \approx W_q + s(b)
$$

会先从排队项开始恶化，而不是先从计算项恶化。这就是为什么真实系统里不能只盯 GPU 利用率，还要看 waiting requests、queue depth、KV cache 使用率等指标。

看一个玩具推导。

假设：
- 方案 A：`b=2, s(2)=0.11s`
- 方案 B：`b=8, s(8)=0.24s`

则：
- `μ_A = 2 / 0.11 ≈ 18.2 req/s`
- `μ_B = 8 / 0.24 ≈ 33.3 req/s`

同样面对 `λ=80 req/s`：
- 方案 A 需要 `ceil(80/18.2)=5` 个副本
- 方案 B 需要 `ceil(80/33.3)=3` 个副本，工程上常配 4 个

这说明**批处理策略本身就在决定容量**。很多人把 batching 当成执行层细节，这是不准确的。它直接改变了 `μ(b)`，也就直接改变了扩容所需的副本数。

再看真实工程例子。在线问答服务部署在 KServe + vLLM 上，白天高峰用户同时提问，夜间流量大幅回落。系统通常这样做：

1. 低峰保留少量副本，避免冷启动。
2. 高峰先看 `num_requests_waiting` 是否持续上升。
3. 如果等待请求上升且 `kv_cache_usage_perc` 还不高，优先扩副本。
4. 如果 GPU 计算负载高但显存仍有空间，continuous batching 继续吞并发。
5. 如果 KV cache 已接近上限，再继续放大并发通常只会把 p95/p99 拉坏。

这里的关键不是“谁更先进”，而是每个手段分别控制不同变量：扩副本控制 `r`，batching 控制 `μ(b)`，队列阈值控制 `W_q`，显存阈值控制可运行请求数上限。

---

## 代码实现

落地时至少要有三类配置：**服务端 batching、平台并发限制、自动扩缩容信号**。只写一个 HPA 或只调一个 `max_batch_size`，都不算完整实现。

先看一个最小容量估算脚本。它不替代压测，但足够帮助你在上线前排除明显不合理的参数。

```python
from math import ceil

def per_replica_throughput(batch_size: int, batch_latency_s: float) -> float:
    assert batch_size > 0
    assert batch_latency_s > 0
    return batch_size / batch_latency_s

def required_replicas(arrival_rate: float, batch_size: int, batch_latency_s: float, headroom: float = 0.2) -> int:
    assert arrival_rate >= 0
    assert 0 <= headroom < 1
    mu = per_replica_throughput(batch_size, batch_latency_s)
    effective_mu = mu * (1 - headroom)
    return ceil(arrival_rate / effective_mu)

mu = per_replica_throughput(batch_size=8, batch_latency_s=0.24)
assert round(mu, 1) == 33.3

r = required_replicas(arrival_rate=80, batch_size=8, batch_latency_s=0.24, headroom=0.2)
assert r == 4

r2 = required_replicas(arrival_rate=80, batch_size=2, batch_latency_s=0.11, headroom=0.2)
assert r2 == 6

print(mu, r, r2)
```

这个例子里，加入 20% 余量后，`b=8` 需要 4 个副本，而 `b=2` 需要 6 个副本。它说明工程配置里不能只写“最大副本数”，还要先量化不同批策略下的单副本能力。

下面是一个偏 KServe/Knative 风格的并发配置示意。`containerConcurrency` 是单副本硬并发上限，白话说就是“这个实例最多同时接几个请求”。`scaleTarget` 则更像软目标，用来告诉 autoscaler 希望单副本平均承受多少并发。

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llm-chat
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 8
    containerConcurrency: 16
    scaleTarget: 4
    model:
      modelFormat:
        name: huggingface
```

服务端 batching 参数通常类似这样：

```yaml
batching:
  max_batch_size: 8
  max_wait_time_ms: 20
```

这两个参数一起定义批策略：
- `max_batch_size` 控制一次最多攒多少请求
- `max_wait_time_ms` 控制最多等多久再发车

如果 `max_wait_time_ms` 过大，吞吐可能上去，但 `W_q` 会上升，尾延迟变差。也就是：

$$
L \approx W_q + s(b)
$$

里的 `W_q` 被人为放大了。

再看一个基于外部指标的扩缩容思路，适合 KEDA 一类系统：

```yaml
triggers:
  - type: prometheus
    metadata:
      metricName: num_requests_waiting
      threshold: "8"
  - type: prometheus
    metadata:
      metricName: kv_cache_usage_perc
      threshold: "75"
```

真实逻辑通常不是“任一指标高就无脑扩容”，而是像下面这样做联合判断：

```python
def decide_action(waiting, kv_cache_usage_perc, gpu_busy, p95_ms):
    assert 0 <= kv_cache_usage_perc <= 100
    if kv_cache_usage_perc >= 90:
        return "limit_concurrency_or_throttle"
    if waiting >= 8 and kv_cache_usage_perc < 75:
        return "scale_out"
    if gpu_busy and waiting < 4 and p95_ms < 300:
        return "keep_and_batch"
    return "hold"
```

这段逻辑的意思是：
- 显存快满了，先保护系统，不要盲目再塞请求
- 等待队列明显增长且显存还有余量，扩副本
- GPU 忙但系统还稳，可以继续依赖 batching 吃流量

指标层可以统一整理成下面这张表：

| 指标 | 作用 | 扩缩容含义 |
|---|---|---|
| `num_requests_waiting` | 看是否开始排队 | 持续升高说明容量不足 |
| `kv_cache_usage_perc` | 看显存状态 | 高位说明并发空间接近耗尽 |
| `containerConcurrency` | 单副本硬上限 | 防止实例被压垮 |
| `scaleTarget` | 单副本软目标 | 决定何时认为需要扩容 |
| `max_batch_size` | 控制吞吐上限 | 大了有利吞吐，不利尾延迟 |
| `max_wait_time` | 控制攒批等待 | 大了会推高排队时间 |

---

## 工程权衡与常见坑

容量弹性最容易失败，不是因为没有自动扩容，而是因为指标和约束选错了。

| 常见坑 | 为什么会出问题 | 规避方式 |
|---|---|---|
| 只看 CPU 做 HPA | GPU 推理瓶颈常在算力、显存、KV cache | 优先看 waiting、VRAM、KV cache、token/s |
| 把 batching 当纯收益 | 批越大，等待越久，尾延迟可能变坏 | 同时限制 `max_batch_size` 和 `max_wait_time` |
| 忽略请求长度分布 | 长短请求混跑会导致批内拖尾 | 分桶、分队列、按 token 成本估算 |
| 忽略 KV cache 上限 | 显存打满后吞吐不再线性增长 | 把 `kv_cache_usage_perc` 纳入决策 |
| 把软目标当硬限制 | `scaleTarget` 不等于绝对并发上限 | 用 `containerConcurrency` 兜底 |
| 低谷直接缩到 0 | 冷启动会把首请求延迟拉高 | 保留 warm pool 或设置 `minReplicas > 0` |

这里最容易误解的是 batching。很多新手会觉得“批越大越省”，这只对吞吐近似成立，对延迟不成立。因为总延迟里有：

$$
L \approx W_q + s(b)
$$

增大批大小时，`s(b)` 可能增长不快，所以吞吐变好；但 `W_q` 通常会增长，因为系统要等更多请求凑成批次。结果就是平均吞吐变好，但 p95/p99 变差。在线服务最终看的是用户体验，不是单张 GPU 跑出了多高 utilization。

另一个真实工程坑是**缩容振荡**。假设系统看到流量下降就立刻缩容，几秒后流量回弹又扩回来，结果副本反复重建，缓存失效，延迟和成本都不好。解决方法通常是增加观察窗口、设置稳定时间、将扩容和缩容阈值分开，而不是一个阈值来回跳。

过载保护逻辑也必须有。一个简单示例：

```python
def admission_control(waiting, kv_cache_usage_perc):
    assert waiting >= 0
    assert 0 <= kv_cache_usage_perc <= 100
    if kv_cache_usage_perc >= 92:
        return "reject_or_degrade"
    if waiting >= 20:
        return "shed_low_priority_traffic"
    return "accept"
```

这段代码体现的原则很直接：**系统已经接近不可恢复拥塞时，限流比继续排队更安全。** 因为无限排队只会制造更差的失败。

---

## 替代方案与适用边界

容量弹性不是唯一方案，不同系统可以只取其中一部分。

| 方案 | 能解决什么 | 解决不了什么 | 适用场景 |
|---|---|---|---|
| 只扩副本 | 提高总容量 `r` | 单副本效率低、GPU 利用率低 | 简单短请求服务 |
| 只做 batching | 提高单副本 `μ(b)` | 峰值排队、总容量不足 | 中低波动负载 |
| 只按 CPU 扩缩容 | 实现简单 | GPU/显存瓶颈被忽略 | CPU 推理或极简单服务 |
| KPA/KEDA + batching 联动 | 同时管副本、并发、排队、显存 | 实现复杂、调参成本高 | LLM 和高并发 GPU 推理 |

可以用一个公式理解为什么替代方案只能覆盖问题的一部分：

$$
C \approx r \cdot \mu(b)
$$

只扩副本是在调 `r`，只做 batching 是在调 `μ(b)`。如果系统问题同时来自“副本太少”和“单副本效率太低”，那你只动一个变量，就只能解决一半。

对初级工程师，一个实用判断标准是：

- 如果是图片分类、文本分类、Embedding 这类短而均匀的请求，按并发或 QPS 扩缩容通常就够用。
- 如果是 LLM 对话、RAG 问答、长文本生成，只看 QPS 会严重失真，必须把 token、上下文长度、KV cache 和 continuous batching 一起考虑。
- 如果你的平台已经有 Knative/KPA，先把并发和最小副本调顺，比一开始就上复杂自定义指标更稳。
- 如果已经进入大模型在线服务阶段，仅靠平台层扩容不够，必须让服务引擎本身支持 batching、分页缓存或更细的调度。

因此，容量弹性的适用边界并不是“用了 GPU 就必须上”，而是：**当请求成本不均、显存成为稀缺资源、且用户又对尾延迟敏感时，它才从可选项变成必选项。**

---

## 参考资料

| 资料来源 | 用途 | 对应章节 |
|---|---|---|
| KServe KPA 文档 | 理解软目标、并发扩缩容 | 代码实现、替代方案 |
| KServe LLM + KEDA 文档 | 理解外部指标驱动扩容 | 代码实现、工程权衡 |
| vLLM Metrics 文档 | 理解等待队列与 KV cache 指标 | 核心机制、代码实现 |
| Triton Dynamic Batching 文档 | 理解 batching 提升吞吐的原因 | 核心机制、替代方案 |
| GPU 动态批处理排队分析论文 | 理解排队与尾延迟关系 | 核心机制、工程权衡 |

1. [KServe KPA 自动扩缩容](https://kserve.github.io/website/docs/model-serving/predictive-inference/autoscaling/kpa-autoscaler)
2. [KServe LLM 指标与 KEDA 自动扩缩容](https://kserve.github.io/website/docs/model-serving/generative-inference/autoscaling)
3. [vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics.html)
4. [NVIDIA Triton Dynamic Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html)
5. [NVIDIA Triton Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
6. [Queueing analysis of GPU-based inference servers with dynamic batching](https://doi.org/10.1016/j.peva.2020.102183)
7. [BATCH: Machine Learning Inference Serving on Serverless Platforms with Adaptive Batching](https://hdl.handle.net/20.500.12571/27484)
