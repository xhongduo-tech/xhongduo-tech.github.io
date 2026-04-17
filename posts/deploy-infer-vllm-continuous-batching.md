## 核心结论

vLLM 的连续批处理，本质上是**按解码迭代调度**，不是按“整批请求”调度。解码迭代可以理解为“模型再生成一个 token 的这一步”。每一步结束后，调度器都会立刻检查三件事：谁完成了、谁还在跑、等待队列里谁可以补进来。于是批次里的空槽不会等到“整批结束”才释放，而是**每轮都回收、每轮都补位**。

这件事解决的是吞吐问题，不是单请求最短延迟问题。吞吐，白话说就是“单位时间总共吐出多少 token”。静态批处理里，短请求经常要陪长请求一起等，或者批次里已经有人结束但槽位要空到整批收尾，GPU 会周期性空转。连续批处理把这种空转压到更低，因此在**中高并发、请求长度差异大**的场景里，通常比静态 batch 更能把 GPU 吃满。

可以先记住一句最实用的判断：如果你的服务端经常同时处理很多请求，而且 prompt 长短、输出长度差异明显，那么连续批处理通常比静态批更合适；如果只有零星请求，或者请求长度几乎一致，它的优势会明显缩小。

一个新手版玩具例子：

假设 GPU 同时只能跑 4 个请求。A、B、C、D 先入场，分别还需要生成 2、8、3、9 个 token。静态批处理会让这 4 个请求绑定成一组，哪怕 A 在第 2 轮就结束了，它的位置也要等到 D 结束后才能接新请求。连续批处理不是这样，A 第 2 轮结束后，第 3 轮就能把排队中的 E 塞进来，于是这 4 个位置更接近“永远有人”。

| 对比项 | 静态批处理 | 连续批处理 |
|---|---|---|
| 调度粒度 | 整批请求 | 每次 decode 迭代 |
| 空槽处理 | 等整批完成后复用 | 每轮立刻复用 |
| 对长短混合请求 | 容易出现短请求陪跑 | 更容易拉平尾部等待 |
| 主要优化目标 | 实现简单、易控 | 吞吐最大化 |

把它想成旋转门最直观。门一共有 4 扇，每扇对应一个活跃请求。连续批处理不是“等 4 个人全出去再放下一组”，而是**谁出去就立刻补谁**。

---

## 问题定义与边界

要讨论连续批处理，先把 LLM 推理拆成两个阶段：

1. **Prefill**：把输入 prompt 整段喂进去，建立 KV Cache。KV Cache 可以理解为“模型已经看过的上下文记忆”。
2. **Decode**：在已有上下文上一步一步生成新 token，每轮通常只新增极少量 token，常见是一轮 1 个。

连续批处理主要发生在 decode 阶段，但它真正难的地方是：**decode 不能被 prefill 粗暴打断**。因为大 prompt 的 prefill 很吃计算，如果调度不当，长 prompt 会把正在持续输出的 decode 拖慢，p95 延迟会明显恶化。

所以问题定义不是“怎么把请求塞进 batch”这么简单，而是：

- 如何让活跃槽位尽量接近上限
- 如何在 prefill 和 decode 之间分配 token 预算
- 如何避免长 prompt 抢走整轮资源
- 如何在吞吐提升时，不把交互式延迟打坏

这里的边界也要讲清楚。连续批处理不是万能加速器，它更适合：

- 在线推理服务
- 中高并发
- 请求长度不均匀
- 目标偏向总吞吐和成本效率

它不一定适合：

- 低并发
- 几乎所有请求都很短
- 追求极致 TTFT（time to first token，首 token 延迟）
- 请求长度高度一致且容易静态打包

再看一个更具体的场景。假设 GPU 最多同时处理 4 个活跃序列：

| 时刻 | 活跃请求 | 等待队列 | 静态批行为 | 连续批行为 |
|---|---|---|---|---|
| 第 1 轮 | A B C D | E F | 开始跑 | 开始跑 |
| 第 2 轮后 | A 完成 | E F | A 槽位继续空着 | 下一轮把 E 补进来 |
| 第 3 轮后 | C 完成 | F | 仍然空着 | 下一轮把 F 补进来 |

静态批像“4 人满员出发的电梯，一趟里不上新人”；连续批像“扶梯，人流不断就一直补位”。

---

## 核心机制与推导

连续批处理背后的核心指标，可以用一个简化吞吐效率公式表示：

$$
\rho = \frac{N_{\text{active}}}{T_{\text{iter}}}
$$

其中：

- $N_{\text{active}}$：当前迭代里实际参与生成的活跃请求数
- $T_{\text{iter}}$：一轮 decode 的平均耗时
- $\rho$：这里用来表示“每单位时间内有效参与生成的并发强度”，它不是论文里的标准符号，但足够帮助理解调度收益

如果 `batch_limit = 4`，理想状态是每一轮都有 4 个活跃请求，那么：

$$
\rho \approx \frac{4}{T_{\text{iter}}}
$$

如果静态批里已经有 2 个请求提前结束，但新请求还不能补进来，那么同样的 $T_{\text{iter}}$ 下就只剩：

$$
\rho \approx \frac{2}{T_{\text{iter}}}
$$

这就是连续批处理提升吞吐的直觉来源：**尽量让 $N_{\text{active}} \approx \text{batch\_limit}$**。

再看一个玩具数值例子。假设：

- `batch_limit = 4`
- 每轮 decode 平均耗时 `5ms`
- 每轮每个活跃请求生成 1 个 token

如果连续批处理能长期保持 4 个活跃槽位，那么理想吞吐约为：

$$
\frac{4\ \text{tokens}}{5\ \text{ms}} = 800\ \text{tokens/s}
$$

如果静态批因为有人提前结束，长期平均只有 2.5 个活跃请求在跑，那么吞吐近似是：

$$
\frac{2.5\ \text{tokens}}{5\ \text{ms}} = 500\ \text{tokens/s}
$$

这不是精确硬件模型，但足够说明问题：**同样的单轮耗时，活跃槽位越满，总吞吐越高**。

真实机制比这个公式复杂，因为 prefill 和 decode 的开销类型不同。一般来说：

- prefill 更偏计算密集
- decode 更偏内存带宽和 KV 访问

vLLM 官方文档对 chunked prefill 的解释很关键：默认策略更偏向 prefill，会优化 TTFT，但可能导致 decode 的 inter-token latency 变差、GPU 利用率不够理想；启用 chunked prefill 后，调度会先优先 decode，再把剩余 token budget 分给 prefill，这样更容易把两类负载混在同一轮里执行。

也就是说，连续批处理的“连续”并不只是“有人走了补新人”，还包含另一层含义：**prefill 不要一次吃光整轮预算，而要被切成块，穿插进 decode 的节奏里**。否则只解决了槽位回收，没有解决长 prompt 阻塞。

一个真实工程例子是：在 H100 这类高端 GPU 上，团队通常不会只看“单请求快不快”，而是会用 vLLM 的连续批处理、异步调度、KV 管理和 FlashInfer/FlashAttention 等内核优化一起调，目标是把整机的 token/s 推高，并尽量让 GPU 长时间保持高利用率。这里要注意，案例里的“100% 利用率”是具体负载和具体配置下的观测结果，不是任何模型、任何流量形态都能复制的保证值。

---

## 代码实现

下面用一个最小可运行的 Python 模型说明调度循环。它不模拟 GPU，只模拟“每轮解码后，完成的请求被移出，等待中的请求立刻补进来”。

```python
from collections import deque

class Request:
    def __init__(self, req_id, remain_tokens, prompt_tokens=0):
        self.req_id = req_id
        self.remain_tokens = remain_tokens
        self.prompt_tokens = prompt_tokens
        self.prefilled = prompt_tokens == 0

    def do_prefill_chunk(self, chunk_size):
        if self.prefilled:
            return 0
        used = min(self.prompt_tokens, chunk_size)
        self.prompt_tokens -= used
        if self.prompt_tokens == 0:
            self.prefilled = True
        return used

    def decode_one_token(self):
        if not self.prefilled or self.remain_tokens <= 0:
            return 0
        self.remain_tokens -= 1
        return 1

    @property
    def finished(self):
        return self.prefilled and self.remain_tokens == 0

def simulate_continuous_batch(requests, batch_limit=4, max_num_batched_tokens=8, prefill_chunk=4):
    waiting = deque(requests)
    active = []
    emitted = 0
    steps = 0

    while waiting or active:
        while len(active) < batch_limit and waiting:
            active.append(waiting.popleft())

        steps += 1

        # 1. 先做 decode，优先保证在线生成不断流
        token_budget = max_num_batched_tokens
        for req in active:
            if token_budget <= 0:
                break
            emitted += req.decode_one_token()
            token_budget -= 1

        # 2. 再用剩余预算做 prefill chunk
        for req in active:
            if token_budget <= 0:
                break
            if not req.prefilled:
                used = req.do_prefill_chunk(min(prefill_chunk, token_budget))
                token_budget -= used

        # 3. 回收完成请求
        active = [req for req in active if not req.finished]

    return emitted, steps

reqs = [
    Request("A", remain_tokens=2),
    Request("B", remain_tokens=8),
    Request("C", remain_tokens=3),
    Request("D", remain_tokens=9),
    Request("E", remain_tokens=4),
    Request("F", remain_tokens=1),
]

emitted, steps = simulate_continuous_batch(reqs, batch_limit=4)
assert emitted == 27
assert steps >= 9
print(emitted, steps)
```

上面这段代码对应的角色关系是：

- `waiting`：等待队列，表示还没拿到活跃槽位的请求
- `active`：活跃槽位，表示本轮真正参与执行的请求
- `batch_limit`：活跃序列上限
- `max_num_batched_tokens`：单轮最多处理多少 token 的预算
- `prefill_chunk`：每次最多切多少 prompt token 进入 prefill

如果把逻辑写成更接近 vLLM 的伪代码，可以是：

```python
while has_waiting_or_active():
    remove_finished_requests()
    fill_empty_slots_from_waiting_queue()

    schedule_all_pending_decodes_first()

    if token_budget_remains():
        schedule_prefills()
        if prefill_too_long():
            chunk_prefill_by(max_num_batched_tokens)

    run_forward()
    sample_next_tokens()
    postprocess_and_release_kv_cache()
```

这里有三个实现点最关键：

1. **调度粒度是 step 级别**  
   每一步都能重排活跃集合，而不是等整批结束。

2. **decode 优先**  
   因为在线服务里，已经在输出的请求最怕被长 prompt 的 prefill 卡住。

3. **prefill 要可切块**  
   官方文档里 `enable_chunked_prefill` 和 `max_num_batched_tokens` 就是为这个服务的。`max_num_batched_tokens` 的含义很直接：单次迭代最多处理多少 token。它不是“总上下文长度”，而是“这一轮调度预算”。

---

## 工程权衡与常见坑

连续批处理提升吞吐的代价，是调度系统更复杂，而且不一定总赢。

最常见的误区，是把“吞吐提升”误读成“每个请求都更快”。实际并非如此。连续批处理优先优化的是整体 token/s，低并发时它的收益很有限，有时还不如简单静态批更省心。

| 常见坑 | 现象 | 原因 | 对策 |
|---|---|---|---|
| 低并发收益不明显 | GPU 没吃满，调度开销相对显眼 | 没有足够请求补位 | 降低 batch 上限，别为低流量过度调度 |
| 长 prompt 拉高 p95 | 已在 decode 的请求突然变慢 | prefill 一次吃掉太多预算 | 开启 `enable_chunked_prefill` |
| 吞吐高了但 TTFT 变差 | 首 token 更慢 | decode 优先挤压了 prefill | 区分交互流量和离线流量 |
| KV Cache 抖动 | 吞吐不稳定甚至频繁回收 | 过度接纳长上下文请求 | 控制 admission，预留足够 KV 空间 |
| 参数一味调大 | 结果反而退化 | `max_num_batched_tokens` 太大，prefill 干扰 decode | 用压测找拐点，不要只看理论上限 |

新手最容易忽略的是 prefill 的破坏性。想象你在接电话，decode 是“每秒都要回复一句”；这时突然来了一个超长 prompt，要先把几千个 token 全过一遍。如果不切 chunk，这个大请求就像一个人突然霸占前台，所有正在持续说话的人都被迫等它。

因此工程上通常至少会做三件事：

- 开启 `chunked prefill`
- 控制 `max_num_batched_tokens`
- 按请求长度或优先级做 admission control

真实工程例子里，常见做法不是“把 batch 调到最大”，而是用压测分别看：

- TTFT
- ITL
- p50 / p95 延迟
- 总 token/s
- GPU 利用率
- OOM 与 KV cache 回收频率

如果只盯吞吐，很容易把交互式体验调坏。

---

## 替代方案与适用边界

静态批处理并没有过时，它只是适用面更窄。

如果你的请求长度接近，且并发不高，静态批处理依然合理。比如离线批量摘要、固定长度分类、统一模板生成，这些任务里大家几乎同时起跑、同时结束，连续批处理的补位价值就没那么大。

还有一种常见折中是混合策略：系统先用较保守的 batch 配置预热，等流量稳定后再切到更激进的连续批参数；或者对超长 prompt 走单独通道，把普通交互请求和长上下文请求拆开服务。

| 策略 | 适用条件 | 优点 | 缺点 |
|---|---|---|---|
| 静态批 | 长度接近、并发低 | 简单、可预测 | 容易空槽，长短混跑差 |
| 连续批 | 中高并发、长度差异大 | 吞吐高、补位及时 | 调度更复杂，需要调参 |
| 混合策略 | 流量形态分层明显 | 能兼顾 TTFT 与吞吐 | 系统设计更复杂 |

可以用一个电梯和扶梯的比喻收尾，但要落回定义：静态批像“满 4 人启动一次的电梯”，适合一组人节奏接近；连续批像“持续补人的扶梯”，适合请求持续流入且有人快有人慢。这个比喻只帮助记忆，真正决定策略的，仍然是你的并发分布、请求长度分布和延迟目标。

---

## 参考资料

- vLLM 官方文档：`Welcome to vLLM`。用于确认 vLLM 把 continuous batching、chunked prefill、FlashInfer 集成作为核心能力。https://docs.vllm.ai/
- vLLM 官方文档：`Performance and Tuning / Chunked Prefill`。用于确认默认调度更偏 prefill，开启 chunked prefill 后会优先 decode，并通过 `max_num_batched_tokens` 控制单轮预算。https://docs.vllm.ai/en/v0.4.2/models/performance.html
- vLLM 官方 CLI 文档：`run-batch`。用于确认 `--max-num-batched-tokens`、`--max-num-seqs`、`--enable-chunked-prefill`、`--long-prefill-token-threshold` 等参数语义。https://docs.vllm.ai/en/stable/cli/run-batch/
- vLLM 官方博客：`Inside vLLM: Anatomy of a High-Throughput LLM Inference System`。用于理解 step 级调度、请求完成后的清理、连续批与 chunked prefill 在 V1 引擎中的位置。https://vllm.ai/blog/anatomy-of-vllm
- buildmvpfast.com 2026：连续批核心机制与吞吐导向的案例总结。可作为二手综述阅读，但工程数字应视为案例，不应直接当成普遍结论。https://www.buildmvpfast.com/blog/batching-strategies-llm-inference-continuous-batching-2026
- systemoverflow.com：面向初学者的连续批处理解释，适合建立“每轮检查完成与补位”的直觉。https://www.systemoverflow.com/learn/ml-nlp-systems/llm-serving/how-does-continuous-batching-work-in-llm-serving
- trackai.dev：带简单数值例子的连续批处理说明，适合快速理解为什么同样 batch 上限下，连续批的平均活跃槽位更高。https://trackai.dev/tracks/performance/latency-ttft/continuous-batching/
