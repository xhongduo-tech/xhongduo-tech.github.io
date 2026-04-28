## 核心结论

推理服务性能基准，不是为了得到一个“这套服务很快”的绝对分数，而是为了在固定条件下，稳定比较不同版本、不同参数或不同部署方案的性能差异。这里的固定条件，至少包括模型版本、硬件、软件栈、输入长度分布、输出长度分布、并发模式、采样参数、预热规则和统计窗口。

对生成式推理服务，最重要的结论通常不是“平均延迟下降了”，而是下面三件事是否被分清了：

| 指标 | 白话解释 | 它回答的问题 |
| --- | --- | --- |
| `TTFT` | 从请求发出到第一个 token 回来的时间 | 用户多久看到“服务开始回答” |
| `TPOT` | 相邻两个输出 token 的平均间隔 | 模型后续吐字到底快不快 |
| token 吞吐 | 单位时间内总共生成多少 token | 机器总体产能有多高 |
| `P95/P99` | 最慢那一批请求的大致表现 | 高峰时用户最容易抱怨的体验 |
| 错误率 | 超时、失败、中断的比例 | 服务是否稳定可用 |

新手可以把它理解成“同一辆车、同一条路、同一批乘客，先把路线和人数固定，再比较谁跑得快”。如果今天拿早高峰、明天拿午夜空路，哪怕车没变，结果也会完全不同。推理服务同理：输入长短、并发多少、是否流式输出，都会直接改变结论。

真正有用的基准，必须能回答这个问题：变慢的根因到底是 `prefill` 慢了、`decode` 慢了、排队变长了，还是压根流量结构变了。如果不能拆出这几个来源，基准结果就很难指导工程决策。

---

## 问题定义与边界

“推理服务性能基准”测的是服务端的性能行为，不测模型质量本身。也就是说，它不回答“模型是否更聪明”，只回答“同样的任务，它多快、多稳、多能扛”。

这里最容易混淆的，是把性能测试和效果评测混在一起。比如两个模型回答内容长度不同、采样温度不同、是否开启流式不同，这时直接比较延迟和吞吐，结论没有意义，因为比较对象已经不是同一个任务了。

因此，测试边界必须提前写死。

| 维度 | 典型字段 | 是否必须固定 | 说明 |
| --- | --- | --- | --- |
| 模型 | 模型名、版本、量化方式 | 是 | 不同模型不能直接共用一个基线 |
| 硬件 | GPU 型号、张数、显存、CPU | 是 | 硬件变了，结果就不是同一基线 |
| 软件栈 | serving 框架、CUDA、驱动、内核参数 | 是 | 软件升级也可能改性能 |
| 输入分布 | prompt 长度、模板、语言 | 建议固定或分桶 | 长短 prompt 对 `prefill` 影响极大 |
| 输出分布 | `max_tokens`、停止条件 | 建议固定或分桶 | 输出越长，`decode` 影响越大 |
| 采样参数 | `temperature`、`top_p`、`seed` | 是 | 会影响生成长度和稳定性 |
| 并发模型 | 固定并发、固定到达率 | 是 | 两种负载模型结论不同 |
| 是否流式 | streaming on/off | 是 | 会改变 `TTFT` 的业务含义 |
| 预热规则 | warmup 次数、是否剔除首轮 | 是 | 冷启动常显著污染结果 |
| 统计窗口 | 采样时长、是否丢弃异常点 | 是 | 否则前后批次不可比 |

一个常见反例是：有人拿“离线批处理吞吐”去解释“在线聊天用户体验”。这是错误外推。离线任务更关心总产能，在线聊天更关心首 token 和尾延迟。即使同一套服务，在这两个场景下也可能需要完全相反的调优方向。

所以，基准测试的第一步不是跑脚本，而是定义边界。边界不清，所有数字都容易失真。

---

## 核心机制与推导

生成式推理服务的端到端时间，通常可以拆成四段：

$$
T_{e2e} = T_{queue} + T_{prefill} + T_{decode} + T_{post}
$$

这里每一项都对应不同工程含义。

- `queue`：排队时间。白话说，就是请求到了，但机器还没轮到它。
- `prefill`：首轮计算输入上下文。白话说，就是模型先把整段输入“读完并建立状态”。
- `decode`：逐 token 生成输出。白话说，就是模型一个字一个字往外吐。
- `post`：收尾处理。比如网络封包、日志落盘、响应格式化。

对用户来说，最先感知的是 `TTFT`：

$$
TTFT = t_{first\ token} - t_{request\ start}
$$

它包含的不只是模型首轮计算，通常还夹带排队、请求解析和部分网络开销。所以 `TTFT` 高，不一定是模型算得慢，也可能是队列堵了。

而 `TPOT` 用来描述后续生成速度：

$$
TPOT = \frac{t_{last\ token} - t_{first\ token}}{\max(n_{out}-1,1)}
$$

其中 $n_{out}$ 是输出 token 数。之所以减 1，是因为从首 token 到最后 token，中间真正经历的是“相邻 token 之间的间隔”。

总 token 吞吐通常写成：

$$
X_{tok} = \frac{\sum_i n_{out,i}}{T}
$$

它代表统计窗口 $T$ 内，所有请求一共生成了多少 token。吞吐高，说明总体产能强；但吞吐高不自动代表用户体验好，因为它可能是靠更长的排队时间换来的。

再看一个玩具例子。假设某次测试中：

- `TTFT = 190ms`
- `TPOT = 13ms/token`
- 输出长度为 80 token

那么端到端时间可近似写成：

$$
T_{e2e} \approx 190 + 79 \times 13 = 1217ms
$$

这个公式的意义很直接：即使首 token 只慢了 10ms，只要后续每个 token 都慢一点，总时延就会被持续放大。对长输出任务尤其明显。

再看排队为什么会突然失控。可以借助 Little 定律：

$$
N \approx X \times R
$$

- $N$：系统内的在途请求数
- $X$：吞吐
- $R$：平均响应时间

白话说，如果响应时间 $R$ 变长，而吞吐 $X$ 又没有同步提升，那么系统里堆着的请求数 $N$ 就会上升。请求一多，排队更长，尾延迟进一步恶化，形成正反馈。这就是很多服务“平时正常，一到高峰突然雪崩”的机制基础。

所以，基准不能只看一个数。`TTFT` 更像交互体验指标，`TPOT` 更像持续生成效率指标，吞吐更像产能指标，`P95/P99` 则是系统拥堵时的真实风险指标。把它们混成一个平均值，会掩盖瓶颈位置。

---

## 代码实现

一个可复现的基准脚本，核心不是“怎么发请求”，而是“怎么保证前后两轮可比”。最小流程通常是：

`load prompts -> warmup -> run benchmark -> collect timestamps -> aggregate p50/p95/p99 -> export csv/json`

其中最关键的是时间戳采集。至少要记录四个点：

| 时间点 | 含义 |
| --- | --- |
| `request_start` | 客户端发起请求的时刻 |
| `first_token_time` | 首个 token 到达的时刻 |
| `last_token_time` | 最后一个 token 到达的时刻 |
| `request_end` | 请求彻底完成的时刻 |

下面给一个可运行的 Python 玩具脚本。它不依赖真实模型服务，但完整展示了 `TTFT`、`TPOT`、端到端延迟和近似 `P95` 的计算方式。

```python
from math import ceil

records = [
    {
        "request_start_ms": 0,
        "first_token_ms": 180,
        "last_token_ms": 960,
        "request_end_ms": 970,
        "output_tokens": 61,
        "ok": True,
    },
    {
        "request_start_ms": 10,
        "first_token_ms": 205,
        "last_token_ms": 1235,
        "request_end_ms": 1245,
        "output_tokens": 81,
        "ok": True,
    },
    {
        "request_start_ms": 20,
        "first_token_ms": 190,
        "last_token_ms": 580,
        "request_end_ms": 590,
        "output_tokens": 31,
        "ok": True,
    },
]

def calc_metrics(record):
    ttft = record["first_token_ms"] - record["request_start_ms"]
    e2e = record["request_end_ms"] - record["request_start_ms"]
    n = record["output_tokens"]
    tpot = 0.0 if n <= 1 else (record["last_token_ms"] - record["first_token_ms"]) / (n - 1)
    return {"ttft_ms": ttft, "e2e_ms": e2e, "tpot_ms": tpot}

def percentile(values, p):
    values = sorted(values)
    idx = max(0, ceil(len(values) * p) - 1)
    return values[idx]

metrics = [calc_metrics(r) for r in records]
ttfts = [m["ttft_ms"] for m in metrics]
e2es = [m["e2e_ms"] for m in metrics]
tpots = [round(m["tpot_ms"], 2) for m in metrics]

assert ttfts == [180, 195, 170]
assert round(tpots[0], 2) == 13.0
assert percentile(e2es, 0.95) == 1235
assert all(r["ok"] for r in records)

summary = {
    "avg_ttft_ms": sum(ttfts) / len(ttfts),
    "p95_e2e_ms": percentile(e2es, 0.95),
    "avg_tpot_ms": sum(tpots) / len(tpots),
}

print(summary)
```

这个脚本体现了三条工程原则。

第一，输入样本必须固定。真实基准里，通常会先准备一组固定 prompt 集合，并把 prompt 长度分桶，比如 `0-256`、`256-1k`、`1k-4k`。否则今天测短请求、明天测长请求，指标变化不再能归因到系统本身。

第二，必须做预热。推理服务第一次运行时，常见额外开销包括权重映射、CUDA graph 初始化、内存池建立、JIT 编译和缓存填充。如果把这部分算进正式结果，得到的是“冷启动性能”，而不是“稳定运行性能”。

第三，结果要落盘。最少应该输出每个请求的原始记录和聚合结果两层数据。前者方便复盘离群点，后者方便做版本对比。只保留终端上的一行平均值，后续几乎无法排查。

一个真实工程例子是：你维护一个在线问答服务，准备把 serving 框架从版本 A 升到版本 B。正确做法不是只跑一次总吞吐，而是把真实 prompt 集按长度分桶，固定 `max_tokens`、固定并发 32、固定 warmup 200 请求、正式采样 10 分钟，然后分别比较每个桶的 `TTFT`、`TPOT`、`P95/P99` 和错误率。这样你才能知道，新版本究竟是短请求更快、长请求更慢，还是整体只是排队策略改变了。

---

## 工程权衡与常见坑

推理服务调优，本质上是在多个目标之间做交换。最典型的交换是：吞吐、首 token 延迟、尾延迟、GPU 利用率，往往不能同时最优。

比如 continuous batching，白话说就是“服务不断把新请求塞进正在运行的批次里”，它通常能提升 GPU 利用率和总吞吐，因为机器更少空转。但代价是，某些短请求可能要为批次协调等待更久，`TTFT` 或 `P95` 变差。对交互问答业务，这种代价可能不可接受；对离线摘要任务，则往往值得。

常见坑可以直接列出来看：

| 常见坑 | 为什么会错 | 规避方式 |
| --- | --- | --- |
| 只看平均值 | 平均值会掩盖最慢那批请求 | 强制同时看 `P50/P95/P99` |
| 只跑单一长度 | 模型对短输入和长输入行为不同 | 按输入、输出长度分桶 |
| 不做 warmup | 冷启动把结果拉差 | 剔除预热阶段 |
| 不固定并发或到达率 | 前后测试负载形状变了 | 明确使用 fixed concurrency 或 fixed arrival rate |
| 把离线结果当线上结论 | 场景目标不同 | 离线、在线分别建基线 |
| 只盯 GPU 利用率 | GPU 忙不代表用户快 | 同看排队、TTFT、错误率 |
| 混测流式和非流式 | 指标语义已经变了 | 分开测，单独解释 |
| 忽略错误率 | 吞吐高可能是因为超时提前断了 | 所有聚合指标都要带成功率 |

还有一个非常常见的误判：短请求和长请求混在同一个桶里。这样总平均吞吐可能看起来不错，但短请求的 `P95` 已经被长请求拖坏了。线上用户抱怨“回复慢”，而报表还显示“平均性能稳定”，就是这种混桶的典型后果。

再举一个真实工程例子。某团队把批大小调大后，token 吞吐从 `180 tok/s` 提升到 `240 tok/s`，GPU 利用率也显著上升，于是以为优化成功。但拆桶后发现，交互问答的 `P95 TTFT` 从 `320ms` 升到 `780ms`。原因是短请求虽然算得不慢，但排队更久了。最后他们的方案不是回退全部优化，而是按业务队列分层：交互流量走小批次低等待策略，离线任务走大批次高吞吐策略。

这说明一个关键点：性能优化没有统一最优解，只有和业务目标一致的最优解。

---

## 替代方案与适用边界

基准测试不是只有一种做法。不同方法解决的问题不同，不能混成一个“唯一正确”的数字。

| 方法 | 是否可复现 | 是否贴近生产 | 适合回答什么问题 | 主要局限 |
| --- | --- | --- | --- | --- |
| 离线 benchmark | 高 | 中 | 新旧版本谁更快 | 不完全代表真实用户流量 |
| 在线监控 | 低到中 | 高 | 用户当前真实体验如何 | 很难做严格变量控制 |
| A/B 测试 | 中 | 很高 | 某个版本上线后是否整体更优 | 成本高，实验周期长 |
| 厂商 perf 工具 | 高 | 中 | 特定栈下的性能上限和瓶颈 | 工具口径受厂商定义约束 |
| 自建脚本 | 中到高 | 中到高 | 最贴合自家业务的问题 | 维护成本高，容易口径漂移 |

如果你关心的是“某个框架升级后，纯性能有没有退化”，固定负载的离线 benchmark 最合适。因为它变量最少，可重复性最高。

如果你关心的是“上线后用户体感是不是更好”，只做离线压测不够，必须结合在线监控，至少持续观察真实流量下的 `P95/P99`、超时率和取消率。因为生产环境里还有网络抖动、上下游限流、混合租户和突发流量，这些都不在理想实验室里。

如果你关心的是“两个方案在真实业务价值上谁更优”，A/B 测试更可靠。它不只看服务快慢，还能看用户停留时长、转化率、会话完成率等业务指标。但它不是纯性能工具，因为业务噪声会混进来。

厂商 perf 工具也很有价值。它们通常提供标准化的打点和成熟的指标口径，适合先快速找上限或定位显著瓶颈。但边界是：它们更擅长回答“在这套栈里，理论和工程上能跑多快”，未必完全覆盖你的业务流量结构。

因此，比较稳妥的工程组合通常是：

1. 用离线 benchmark 建立版本基线。
2. 用在线监控观察真实生产行为。
3. 对高风险改动再补 A/B 测试。

这个组合的核心不是“多测几次”，而是让每种方法只回答它擅长回答的问题。

---

## 参考资料

下面这张表说明每条资料主要支撑哪一部分结论，避免只堆链接不说明用途。

| 来源 | 主要支撑内容 |
| --- | --- |
| NVIDIA TensorRT 性能测量文档 | 性能测量方法、吞吐与延迟口径、实验设计思路 |
| Triton / GenAI-Perf 文档 | 生成式推理场景下的负载生成与基准方式 |
| TensorRT-LLM Server Metrics 文档 | 服务端指标定义与观测口径 |
| vLLM Benchmarks API | 自建或复用 benchmark 脚本时的实现参考 |
| PagedAttention 论文 | LLM serving 中内存管理与吞吐/并发机制背景 |

1. [NVIDIA TensorRT: Advanced Performance Measurement Techniques](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/measurement-techniques.html)
2. [NVIDIA Triton Inference Server: GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)
3. [NVIDIA AI Perf Server Metrics Reference](https://docs.nvidia.com/aiperf/server-metrics/ai-perf-server-metrics-reference)
4. [vLLM Benchmarks Documentation](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/)
5. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
