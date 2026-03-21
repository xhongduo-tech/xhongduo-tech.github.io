## 核心结论

LLM 推理服务的监控，核心不是“平均延迟”，而是同时盯住三类量：

| 指标 | 白话解释 | 更接近用户感知 | 更接近资源占用 | 常见单位 |
| --- | --- | --- | --- | --- |
| TTFT | 从发出请求到看到第一个字的时间 | 高 | 中 | ms / s |
| TPOT | 第一个字出来后，后续每个 token 的平均节奏 | 高 | 高 | ms/token |
| 吞吐量 | 系统单位时间总共产出多少 token | 低 | 很高 | token/s 或 token/s/GPU |

新手可以先这样理解：TTFT 像“你发出指令后，第一句话多久回来”；TPOT 像“第一句话出来后，后面每句话说得有多快”；吞吐量像“这台服务同时能喂饱多少人”。

只看一个指标会误判系统状态。TTFT 很低但吞吐量很差，说明单用户体验不错但系统扛不住并发；吞吐量很高但 P99 TTFT 爆炸，说明机器很忙，但尾部用户已经开始明显卡顿。

SLO，服务级别目标，就是对用户体验的量化承诺。写法必须是“指标 + 分位 + 时间窗口”。例如：

$$
\text{SLO}_{TTFT}:\ P99(TTFT,\ 5m) < 500ms
$$

这句话的含义不是“平均 500ms”，而是“最近 5 分钟里，99% 的请求首 token 延迟小于 500ms”。这类写法同时能服务三件事：对外承诺、内部告警、容量规划。

---

## 问题定义与边界

先把边界说清楚，否则监控会出现“同一个名字，不同算法”的问题。

**TTFT，Time To First Token**，白话解释是“从请求提交到第一个有效输出 token 到达的总时间”。如果你从客户端测，它通常包含网络传输；如果你从服务端测，它通常不含公网网络，但会包含内部排队和推理阶段。对推理服务本身，常见拆分是：

$$
TTFT = T_{queue} + T_{prefill} + T_{first\ decode} (+ T_{network})
$$

这里：

- `queue` 是排队时间，意思是请求已经到了，但还没轮到 GPU 处理。
- `prefill` 是预填充阶段，意思是模型先把输入 prompt 编码进 KV cache。
- `first decode` 是第一次解码，意思是开始生成第一个输出 token。

一个初学者最容易混淆的是：TTFT 不是“纯模型算力时间”，它经常已经混入调度、排队、网络和流式协议开销。

下面这个时序图可以把边界看清楚：

```text
客户端提交请求
    |
    |---- 网络传输 ----|---- 排队 ----|---- prefill ----|-- 首次 decode --|  第一个 token 到达
    |<--------------------------- TTFT ------------------------------->|
```

**TPOT，Time Per Output Token**，白话解释是“首 token 之后，每个输出 token 平均隔多久”。它主要描述解码阶段的节奏。常见定义是：

$$
TPOT = \frac{E2E - TTFT}{OSL - 1}
$$

其中：

- $E2E$ 是端到端总时延，意思是从提交请求到最后一个 token 完成。
- $OSL$ 是输出 token 数，意思是这次回答总共生成了多少 token。
- 分母减 1，是因为首 token 已经被 TTFT 单独计算了。

如果输出只有 1 个 token，这个公式就不稳定，因此短输出场景要单独处理，不能机械套公式。

**吞吐量**有两个口径：

1. 系统吞吐量：整个服务总共每秒生成多少 token。
2. 单用户吞吐量：一个请求流式输出时，单个用户感受到的速度。

监控平台更常用系统吞吐量，因为它直接对应容量和成本。一个常见定义是：

$$
TPS = \frac{\#Output\ Tokens}{T_{last} - T_{first}}
$$

这里 $T_{first}$ 是压测开始后第一条请求发出的时间，$T_{last}$ 是最后一条请求最后一个 token 返回的时间。

玩具例子：如果 10 秒里总共生成了 3000 个 token，那么系统吞吐量就是 $3000 / 10 = 300$ token/s。  
真实工程例子：如果你有 4 张 GPU，总吞吐量 1400 token/s，那么更可比较的指标是 $1400 / 4 = 350$ token/s/GPU，这样才方便做不同机器型的横向比较。

---

## 核心机制与推导

这三个指标之所以必须联合看，是因为它们分别映射到不同阶段。

- TTFT 更像“起步成本”，通常受排队、长 prompt prefill、批处理策略影响。
- TPOT 更像“稳定巡航速度”，通常受 decode 算力、KV cache 命中率、并发竞争影响。
- 吞吐量更像“总产能”，通常受 GPU 利用率、批大小、调度策略影响。

一个更完整的关系可以写成：

$$
E2E = TTFT + (OSL - 1) \cdot TPOT
$$

如果请求输出长度分布变化很大，只看 E2E 几乎没法定位问题。因为 E2E 变大，可能是 TTFT 变差，也可能只是输出更长。

### 玩具例子

假设一个请求：

- 总时延 $E2E = 2.3s$
- 首 token 时间 $TTFT = 0.5s$
- 输出 token 数 $OSL = 10$

那么：

$$
TPOT = \frac{2.3 - 0.5}{10 - 1} = 0.2s = 200ms/token
$$

这个结果说明：用户先等 500ms 看到第一个字，之后每 200ms 再来一个 token。  
如果把 TTFT 优化到 250ms，但 TPOT 不变，用户会明显感觉“开始更快”；如果 TTFT 不变但 TPOT 从 200ms 降到 80ms，用户会感觉“回答展开更顺”。

### 真实工程例子

假设一次 10k 请求压测的结果如下：

| 指标 | 数值 |
| --- | --- |
| TTFT P50 | 127 ms |
| TTFT P95 | 312 ms |
| TTFT P99 | 524 ms |
| token/s/GPU | 356 |

如果你的 SLO 是“5 分钟窗口内，TTFT P99 < 500ms”，那么这个结果是**略微不达标**，因为 P99 已经到 524ms。这里最关键的不是“平均还不错”，而是尾部已经跨线。

再往下看，要判断是“体验不够”还是“机器不够”，可以用一个简单的判断表：

| 现象 | 可能原因 | 优先动作 |
| --- | --- | --- |
| TTFT P99 升高，吞吐量没满 | 排队策略差、长 prompt 拖慢 prefill | 调批策略、拆分大请求 |
| TTFT P99 升高，吞吐量已满 | 系统接近饱和 | 扩容、限流、降低 max tokens |
| TPOT 升高，TTFT 正常 | decode 竞争、KV cache 压力 | 看 KV cache、并发上限 |
| 吞吐量下降，TTFT 和 TPOT 都变差 | GPU 或压测端都可能成瓶颈 | 先排除压测工具瓶颈 |

SLO 的写法也应该标准化。推荐至少写三层：

| 层级 | 示例 | 用途 |
| --- | --- | --- |
| 用户体验 SLO | `TTFT P99 < 500ms / 5m` | 对体验负责 |
| 交互节奏 SLO | `TPOT P95 < 80ms / 5m` | 对流式感知负责 |
| 容量 SLO | `token/s/GPU > 300 / 15m` | 对成本和产能负责 |

分位和窗口一起写，是因为它们决定告警含义。P50 适合看主流体验，P95/P99 适合看尾部伤害；5 分钟窗口适合实时告警，15 分钟窗口更适合容量和趋势判断。

---

## 代码实现

如果你用 vLLM，Prometheus 兼容的 `/metrics` 已经暴露了关键指标。当前文档里推荐的核心指标包括：

- `vllm:time_to_first_token_seconds`
- `vllm:inter_token_latency_seconds`
- `vllm:e2e_request_latency_seconds`

在一些旧版本或兼容写法里，你还会看到：

- `vllm:time_per_output_token_seconds`

它本质上也是 TPOT/ITL 口径，但新版本更推荐 `inter_token_latency_seconds`。

下面是一个新手可读的 Prometheus 告警示例：

```yaml
groups:
- name: llm-slo
  rules:
  - alert: LLMHighTTFTP99
    expr: |
      histogram_quantile(
        0.99,
        sum by (le) (
          rate(vllm:time_to_first_token_seconds_bucket[5m])
        )
      ) > 0.5
    for: 10m
    labels:
      severity: page
    annotations:
      summary: "TTFT P99 超过 500ms"

  - alert: LLMHighTPOTP95
    expr: |
      histogram_quantile(
        0.95,
        sum by (le) (
          rate(vllm:inter_token_latency_seconds_bucket[5m])
        )
      ) > 0.08
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "TPOT P95 超过 80ms"

  - alert: LLMThroughputDrop
    expr: |
      (
        sum(rate(vllm:generation_tokens_total[5m]))
        /
        4
      ) < 300
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "token/s/GPU 低于 300"
```

如果你还在兼容旧版 vLLM，也可以把第二条里的指标替换成 `vllm:time_per_output_token_seconds_bucket`。

下面用一个可运行的 Python 例子演示公式和 SLO 判断：

```python
from statistics import mean

requests = [
    {"e2e": 2.3, "ttft": 0.5, "osl": 10},
    {"e2e": 1.5, "ttft": 0.3, "osl": 7},
    {"e2e": 3.1, "ttft": 0.7, "osl": 13},
]

def tpot(req):
    if req["osl"] <= 1:
        return 0.0
    return (req["e2e"] - req["ttft"]) / (req["osl"] - 1)

tpots = [tpot(r) for r in requests]
avg_tpot = mean(tpots)

total_output_tokens = sum(r["osl"] for r in requests)
t_first = 0.0
t_last = 3.5
system_tps = total_output_tokens / (t_last - t_first)

assert round(tpot(requests[0]), 3) == 0.2
assert avg_tpot > 0
assert system_tps > 0

ttft_p99 = 0.524  # 524ms
slo_target = 0.500
assert (ttft_p99 > slo_target) is True
```

压测侧，Locust 和 wrk2 都常用。Locust 适合做带业务逻辑的 HTTP 压测，重点是**复用 client**，不要每次请求都新建连接；wrk2 适合做更稳定的恒定速率压测，重点是带 `--latency` 输出分位。

Locust 示例：

```python
from locust import HttpUser, task, between

class LLMUser(HttpUser):
    wait_time = between(0.1, 0.3)

    @task
    def chat_completion(self):
        payload = {
            "model": "my-llm",
            "messages": [{"role": "user", "content": "解释 TTFT 和 TPOT 的区别"}],
            "stream": False,
            "max_tokens": 128,
        }
        with self.client.post("/v1/chat/completions", json=payload, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"bad status: {resp.status_code}")
```

wrk2 示例：

```bash
wrk -t4 -c128 -d2m -R300 --latency http://127.0.0.1:8000/health
```

这里 `-R300` 表示恒定 300 RPS，`--latency` 会输出分位。它的价值在于更容易看出系统在峰值负载下的尾部退化。

---

## 工程权衡与常见坑

最常见的错误，是只记平均值，不记 histogram。平均值会把尾部冲淡。一个系统可能平均 TTFT 只有 300ms，但 P99 已经到 5s，真实用户会明确感知“偶发卡死”。

| 常见坑 | 后果 | 缓解方式 |
| --- | --- | --- |
| 只记录平均值 | 看不到尾部 | 必须记录 histogram，至少看 P95/P99 |
| 只看 E2E | 无法区分 prefill 和 decode 问题 | 分开看 TTFT、TPOT、E2E |
| 压测工具每次新建 client | 压测端先成为瓶颈 | 复用连接、复用 session |
| wrk2 不带 `--latency` | 只有均值，缺少分位 | 固定使用 `--latency` |
| 只设一个告警阈值 | 告警要么过晚，要么过多 | 设 warning/page 两级 |
| 不记录样本量 | P99 不可信 | 至少 1000 样本看 P99，1 万样本看 P99.9 |

一个实用流程可以写成：

```text
TTFT P99 上升
  -> 看 waiting queue 是否增加
  -> 看 prefill 时间是否增加
  -> 看 token/s/GPU 是否已接近饱和
  -> 决定是调度优化、限流，还是扩容
```

真实工程里还要警惕“压测工具撒谎”。例如压测机 CPU 打满、连接数不够、TLS 或 JSON 编码过重，都会让你误以为模型服务变慢。判断方法很简单：服务端 GPU 利用率不高，但客户端延迟很差，这时先查压测端。

另一个坑是把 TTFT 的问题全部归因于模型。实际上长 prompt 的 prefill 成本非常高，同一模型、同一并发下，512 token 输入和 8k token 输入的 TTFT 不是一个量级。监控里最好按输入长度分桶，否则你会把“业务请求变长”误判成“服务退化”。

---

## 替代方案与适用边界

不是所有团队都需要一上来就做完整的 Prometheus histogram + 多窗口 SLO。

低负载、单机、内部工具型服务，可以先用简化方案：

| 场景 | 推荐方案 | 适用条件 | 可接受误差 |
| --- | --- | --- | --- |
| 低并发内部服务 | 平均值 + 最大值 + 单阈值告警 | 请求量小，用户少 | 较高 |
| 中等生产服务 | P50/P95/P99 + 5m 窗口 | 已有 Prometheus | 中 |
| 高负载在线服务 | histogram + 多级告警 + token/s/GPU | 需要容量管理和尾部优化 | 低 |
| 无 histogram 环境 | 应用侧采样上报分位 | 平台受限，先求可用 | 中到高 |

为什么 histogram 更适合尾部？因为它不是只给你一个平均数，而是保留“延迟分布”的形状。白话说，平均值只告诉你“总体大概多慢”，histogram 才告诉你“最倒霉的那 1% 到底慢成什么样”。

如果环境不支持 histogram，也不是完全不能做。你可以在应用层自己记录 TTFT 和 TPOT 样本，然后定期计算近 5 分钟的分位。但这个方案有三个问题：

1. 采样不全时，P99 容易失真。
2. 多实例合并分位很麻烦。
3. 告警和可视化不如 Prometheus 原生顺滑。

所以，替代方案可以用，但它更适合过渡，不适合作为成熟生产方案的终点。

---

## 参考资料

1. [NVIDIA NIM《Metrics》](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)：给出 TTFT、ITL/TPOT、E2E 的定义与关系，适合统一指标口径。
2. [Anyscale《Understand LLM latency and throughput metrics》](https://docs.anyscale.com/llm/serving/benchmarking/metrics)：清楚说明 TPOT、ITL、TPS 的公式，以及系统吞吐和用户吞吐的区别。
3. [vLLM《Metrics》](https://docs.vllm.ai/en/stable/design/metrics/)：说明 `/metrics` 暴露的 Prometheus 指标集合，适合落地监控面板。
4. [vLLM《Production Metrics》](https://docs.vllm.ai/en/stable/usage/metrics.html)：列出生产环境常用指标，包括 `vllm:time_to_first_token_seconds` 与 `vllm:inter_token_latency_seconds`。
5. [IETF Internet-Draft《LLM Benchmarking Methodology》](https://datatracker.ietf.org/doc/html/draft-gaikwad-llm-benchmarking-methodology-00)：给出 TTFT 测试方法、样本量要求、报告模板，适合写规范化压测报告。
6. [IETF Internet-Draft《LLM Benchmarking Terminology》](https://datatracker.ietf.org/doc/draft-gaikwad-llm-benchmarking-terminology/)：补充 TTFT、E2E 等术语边界，适合避免跨团队口径不一致。
7. [wrk2 README](https://github.com/giltene/wrk2)：解释恒定吞吐压测和 `--latency` 的意义，适合做尾延迟测试。
8. [Locust 官方网站](https://locust.io/)：适合需要 Python 逻辑、登录态和业务流程的压测场景。
