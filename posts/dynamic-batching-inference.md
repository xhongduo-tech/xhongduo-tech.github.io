## 核心结论

动态批处理是指：服务端不再死等一个固定批大小，而是同时设置“最大批量”和“最长等待时间”，满足任一条件就立刻执行。白话说，它不是“攒满再走”，而是“满员先走，等太久也走”。

它解决的是推理服务里最常见的一组矛盾：单请求执行延迟希望低，GPU 利用率希望高，这两个目标通常相互拉扯。动态批处理的核心触发逻辑可以写成：

$$
trigger = (queue\_size \ge max\_batch\_size)\ \lor\ (wait\_time \ge batch\_timeout)
$$

这条规则的意义很直接：

- 负载高时，请求很快凑满批次，系统主要受 `max_batch_size` 控制，吞吐更高。
- 负载低时，请求凑不满批次，系统主要受 `batch_timeout` 控制，延迟有上界。

一个玩具例子：设 `max_batch_size=4`、`batch_timeout=20ms`。如果前 3 个请求在 12ms 内到达，此时队列没满，系统最多再等 8ms 就必须发车；如果第 4 个请求在第 17ms 到达，则立刻触发，不再继续等待。

真实工程里，这套机制通常不是终点。大模型服务如 vLLM、TGI 常进一步使用连续批处理。连续批处理是指：每轮 token 生成结束后，立刻把空出来的位置补上新请求。白话说，它不是“整批做完再换下一批”，而是“谁结束谁出队，空位马上补人”。动态批处理主要决定“何时发起一批”，连续批处理主要决定“运行中的批如何持续补齐”。两者结合，通常比静态批处理更接近延迟与吞吐的 Pareto 平衡点。

| 策略 | 触发条件 | 典型延迟特征 | 典型吞吐特征 | 适合场景 |
|---|---|---|---|---|
| 静态批处理 | 固定 batch size 才执行 | 低负载下等待明显 | 高负载下稳定 | 离线任务、负载稳定 |
| 动态批处理 | 达到最大批量或等待超时 | 延迟可控，随参数变化 | 比静态更稳健 | 在线推理、QPS 波动 |
| 连续批处理 | 每轮迭代补空位 | 首 token 更低，尾部更平滑 | 通常最高 | LLM 生成式推理 |

---

## 问题定义与边界

问题定义可以写得非常明确：在突发流量和日常流量都存在的情况下，如何让端到端延迟不过分上升，同时尽量提高单卡吞吐。

端到端延迟通常拆成两段：

$$
latency \approx queue\_wait + service\_time(batch\_size)
$$

这里的 `queue_wait` 是排队等待时间，白话说就是请求在真正上 GPU 前卡在队列里的时间；`service_time(batch_size)` 是执行时间，白话说就是这一批实际跑模型花的时间。吞吐近似写成：

$$
throughput \approx \frac{batch\_size}{batch\_service\_time}
$$

这两个式子说明了一件事：批变大，单批执行时间通常会上升，但每批处理的请求数也上升，所以总吞吐往往变好；代价是排队时间可能增加。

边界主要由两个参数定义：

- `max_batch_size`：一批最多收多少个请求，决定并发上限。
- `batch_timeout`：队列头请求最多允许等多久，决定等待上限。

动态批处理不解决所有问题。它默认有几个前提：

- 模型能接受 batch 输入，且批内请求形状差异不能过大。
- 服务层可以把多个独立请求合并后再拆回结果。
- 业务允许引入一个可控但非零的排队等待。

下面这个表把问题边界具体化：

| 指标 | 含义 | 直接影响因素 | 过大时的后果 |
|---|---|---|---|
| `queue_wait` | 入队到发批的等待 | `batch_timeout`、到达速率 | 用户感知延迟上升 |
| `service_time` | 模型真正执行时间 | `batch_size`、序列长度、硬件 | P99 延迟尖峰 |
| `throughput` | 单位时间处理请求数 | `batch_size`、执行效率 | 资源浪费或成本升高 |
| `GPU utilization` | GPU 忙碌比例 | 批聚合效果、调度策略 | 卡贵但没跑满 |

再看一个题目里要求的数值例子。设 `max_batch_size=4`，`batch_timeout=20ms`。

- 若 3 个请求在 12ms 内到达，则系统不会立即执行，因为还没满 4 个。
- 此时最晚在第 20ms 触发，所以还会继续等 8ms。
- 如果第 4 个请求在接下来的 5ms 内到达，也就是总等待到第 17ms 达到满批，则立即执行。
- 因此早到请求的额外等待被限制在 20ms 以内，而高负载时又可能更早触发。

这就是它与静态批处理的根本差别。静态批处理的规则是“只看数量”；动态批处理的规则是“数量和时间都看”。

---

## 核心机制与推导

动态批处理的机制可以看成一个“时间窗 + 容量”联合调度器。

当队列从空变为非空时，系统记录首个请求到达时刻，并开启一个截止时间：

$$
deadline = first\_arrival\_time + batch\_timeout
$$

之后每来一个请求，系统都检查两件事：

1. 当前队列大小是否已达到 `max_batch_size`
2. 当前时间是否已达到 `deadline`

任一满足，就触发 `flush_batch()`。

这个设计为什么合理，可以分别看低负载和高负载两种极端。

高负载时，请求到达间隔很小。设平均到达率为 $\lambda$，单批最大容量为 $B$，那么凑满一批所需时间近似为：

$$
t_{fill} \approx \frac{B}{\lambda}
$$

如果 $t_{fill} < batch\_timeout$，系统通常会在超时前满批触发。此时策略更像“近似静态批处理”，吞吐优先。

低负载时，请求稀疏，$t_{fill}$ 可能远大于 `batch_timeout`。如果还坚持等满，就会出现请求长时间悬挂。因此引入超时分支后，系统退化成“小批量快速执行”，延迟优先。

所以这两个条件分别覆盖两种负载区间：

- `queue_size ≥ max_batch_size` 覆盖高负载区间。
- `wait_time ≥ batch_timeout` 覆盖低负载区间。

这也是为什么它能构成一条实际可用的 Pareto 前沿。Pareto 前沿的白话解释是：在一组配置里，想让延迟继续更低，通常必须牺牲一部分吞吐；想让吞吐继续更高，通常必须容忍一部分延迟。不存在一个参数同时把两边都做到极致。

可以用一个 ASCII 图直观看：

```text
吞吐
^
|                            连续批处理
|                        ******
|                  *****      
|            ******           动态批处理可调区间
|       *****                 
|  *****                      
|**  静态批处理在低负载下容易掉队
+---------------------------------> 延迟
   低                            高
```

再给一个新手版解释。可以把“等待超时”理解成“发车倒计时”，把“最大批量”理解成“满员即发”。只靠满员规则，淡时会把人晾着；只靠倒计时规则，忙时又会错失合批收益。两条规则一起用，才覆盖完整工作区间。

如果想用监控数据估算参数，可以从历史数据里统计：

- 请求到达间隔分布
- 不同 batch size 下的 `service_time`
- P50/P95/P99 `queue_wait`

然后枚举参数组合，计算：

$$
latency_{p99} \approx queue\_wait_{p99}(B, T) + service\_time_{p99}(B)
$$

其中 $B$ 表示 `max_batch_size`，$T$ 表示 `batch_timeout`。业务若有 SLA，例如 P99 不超过 300ms，就选满足约束且吞吐最高的那组参数。

真实工程例子是 LLM 生成服务。假设一个批里有 8 个请求，它们的输出长度不同。静态或普通动态批处理常常要等这一整批都结束，期间短请求已经完成，但占位还在。连续批处理会在每轮 token 迭代后把已完成的序列移出，并把新请求补进来。这不是替代动态批处理，而是把批次利用率从“请求级”推进到“迭代级”。

---

## 代码实现

下面先给一个命令式的玩具实现，用纯 Python 模拟动态批处理调度。它不依赖异步框架，但能完整表达关键变量和触发逻辑。

```python
from dataclasses import dataclass

@dataclass
class Request:
    req_id: str
    arrival_ms: int

class DynamicBatcher:
    def __init__(self, max_batch_size: int, batch_timeout_ms: int):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.queue = []
        self.next_flush_deadline = None

    def enqueue(self, req: Request):
        if not self.queue:
            # 队列从空变非空时，启动本批次的超时截止时间
            self.next_flush_deadline = req.arrival_ms + self.batch_timeout_ms
        self.queue.append(req)

        if len(self.queue) >= self.max_batch_size:
            return self.flush(req.arrival_ms)
        return None

    def tick(self, now_ms: int):
        if self.queue and now_ms >= self.next_flush_deadline:
            return self.flush(now_ms)
        return None

    def flush(self, now_ms: int):
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]

        # 如果还有残留请求，新的 deadline 从残留队列头开始计算
        if self.queue:
            first_arrival = self.queue[0].arrival_ms
            self.next_flush_deadline = max(now_ms, first_arrival + self.batch_timeout_ms)
        else:
            self.next_flush_deadline = None

        return {
            "flush_at_ms": now_ms,
            "batch_ids": [r.req_id for r in batch],
            "batch_size": len(batch),
        }

# 玩具例子：3 个请求在 12ms 内到达，第 4 个在 17ms 到达，达到 max_batch_size 立即触发
batcher = DynamicBatcher(max_batch_size=4, batch_timeout_ms=20)
assert batcher.enqueue(Request("r1", 0)) is None
assert batcher.enqueue(Request("r2", 5)) is None
assert batcher.enqueue(Request("r3", 12)) is None
result = batcher.enqueue(Request("r4", 17))
assert result["batch_size"] == 4
assert result["flush_at_ms"] == 17

# 另一个例子：一直没等到第 4 个请求，则在 20ms 超时触发
batcher = DynamicBatcher(max_batch_size=4, batch_timeout_ms=20)
batcher.enqueue(Request("a", 0))
batcher.enqueue(Request("b", 7))
batcher.enqueue(Request("c", 12))
timeout_result = batcher.tick(20)
assert timeout_result["batch_size"] == 3
assert timeout_result["batch_ids"] == ["a", "b", "c"]
```

这段代码里最关键的变量有两个：

- `queue`：等待合批的请求队列。
- `next_flush_deadline`：当前批次最晚必须触发的时间点。

如果换成事件驱动风格，逻辑通常写成“入队事件 + 定时器回调”两部分。伪代码如下：

```python
class AsyncDynamicBatcher:
    def __init__(self, max_batch_size, batch_timeout_ms):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.queue = []
        self.batch_timer = None

    async def enqueue(self, request, now_ms):
        self.queue.append(request)

        if len(self.queue) == 1:
            self.batch_timer = schedule_after(
                self.batch_timeout_ms,
                self.on_timeout
            )

        if len(self.queue) >= self.max_batch_size:
            cancel(self.batch_timer)
            await self.flush_batch(now_ms)

    async def on_timeout(self, now_ms):
        if self.queue:
            await self.flush_batch(now_ms)

    async def flush_batch(self, now_ms):
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        results = await run_model(batch)
        reply(results)

        if self.queue:
            self.batch_timer = schedule_after(
                self.batch_timeout_ms,
                self.on_timeout
            )
```

真实工程例子可以再具体一点。假设你在做一个文本重写 API，业务高峰时 QPS 达到 200，低峰时只有 10。单请求跑一次模型需要 18ms，batch=8 时单批耗时 42ms。此时：

- 低峰如果强行固定 batch=8，用户可能要等很久凑批。
- 动态批处理设 `max_batch_size=8`、`batch_timeout=15ms` 后，低峰能把排队时间限制在 15ms 内。
- 高峰下又经常能在 15ms 之前凑满 8 个请求，从而获得更高吞吐。

---

## 工程权衡与常见坑

动态批处理的难点不在“会不会写”，而在“参数怎么定”。

第一个权衡是 `batch_timeout`。它太大时，短请求会被长时间压在队列里，首包延迟明显变差；它太小时，又聚不起像样的批，GPU 利用率低。第二个权衡是 `max_batch_size`。它太小会浪费卡；太大则可能导致单批执行时间拉长，甚至显存不稳定。

下面这个表总结常见坑：

| 配置错误 | 典型症状 | 根因 | 改进措施 |
|---|---|---|---|
| `batch_timeout` 过大 | P95/P99 排队延迟高 | 低负载下仍强行等人 | 下调超时，按流量自适应 |
| `batch_timeout` 过小 | GPU 利用率低 | 批次长期凑不起来 | 适当增大超时 |
| `max_batch_size` 过大 | 单批耗时抖动、OOM 风险 | 批内 token 差异大，显存吃紧 | 按显存和长度分桶 |
| `max_batch_size` 过小 | 吞吐上不去 | 并发上限被过早截断 | 结合压测提高上限 |
| 不做长度分桶 | 同批长短样本互相拖累 | padding 或解码尾部长 | 按输入长度/输出长度分组 |
| 只看平均值 | 线上体验仍差 | P99 被掩盖 | 监控 tail latency |

一个典型错误场景是：业务低峰 QPS 只有 10，但系统还保持 `batch_timeout=100ms`。这意味着很多请求会白白多等接近 100ms，而这一等待未必换来明显的吞吐收益。在线服务里，这通常是不划算的。

因此工程上常做运行时自适应。白话说，就是根据最近一小段时间的负载，动态调整参数，而不是全时段固定一个值。下面给一个简单示意：

```python
def tune_batch_params(qps, avg_queue_len):
    # 低负载：优先保延迟
    if qps < 20:
        return {"max_batch_size": 2, "batch_timeout_ms": 5}

    # 中等负载：折中
    if qps < 100:
        return {"max_batch_size": 4, "batch_timeout_ms": 10}

    # 高负载：优先保吞吐
    if avg_queue_len > 8:
        return {"max_batch_size": 8, "batch_timeout_ms": 15}

    return {"max_batch_size": 6, "batch_timeout_ms": 12}

assert tune_batch_params(10, 1)["batch_timeout_ms"] == 5
assert tune_batch_params(60, 3)["max_batch_size"] == 4
assert tune_batch_params(200, 12)["max_batch_size"] == 8
```

但这里也有一个坑：参数切换不能太频繁，否则系统会抖。常见做法是：

- 用滑动窗口计算 QPS 和队列长度。
- 设置滞回区间，避免刚到阈值就来回切换。
- 参数变更按秒级或分钟级生效，而不是每个请求都改。

另一个经常被忽略的问题是批内异构。异构是指同一批里请求长度差异很大。对白话解释就是：有人只生成 20 个 token，有人生成 1000 个 token，放一锅里跑，短请求会被长请求拖尾。LLM 场景下，这比单纯的 `batch_size` 更影响实际收益，所以动态批处理通常要和长度分桶、最大 token 限制、连续批处理联用。

---

## 替代方案与适用边界

动态批处理不是唯一方案。至少有三种常见策略：静态批处理、动态批处理、连续批处理。

可以把它们类比成交通控制，但类比只辅助理解，不替代定义：

- 静态批处理像固定绿灯时长，时间到了或数量够了之外不灵活。
- 动态批处理像“车满或等太久就放行”。
- 连续批处理像“只要前面有空位，后车立刻补上”。

核心差别如下：

| 策略 | 触发机制 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|---|
| 静态批处理 | 固定 batch size | 实现最简单，离线吞吐稳定 | 低负载等待严重 | 训练、批量离线推理 |
| 动态批处理 | `size` 或 `timeout` 触发 | 在线服务里最常用 | 参数调优复杂 | 通用在线推理 |
| 连续批处理 | 迭代级补位 | LLM 吞吐最高，空转少 | 实现复杂，对调度器要求高 | 自回归生成、长输出 |

什么时候选哪种策略，判断标准通常不是“哪种最先进”，而是“业务约束是什么”。

如果要求极低延迟，尤其是首 token 延迟非常敏感，连续批处理通常更有优势，因为它能减少批次切换空档。如果流量波动很大，但实现成本又要可控，动态批处理通常是最合适的中间解。如果是离线任务，静态批处理仍然非常有效，因为离线场景通常不在乎单请求等待。

真实工程里，经常采用混合方案：

- 请求入队阶段使用动态批处理决定何时发起。
- 发起后在 LLM 解码阶段使用连续批处理维持高利用率。
- 低负载时进一步减小 `batch_timeout`，甚至直接退化为近似逐请求处理。

一个简化的切换逻辑可以写成：

```text
if qps < low_load_threshold:
    use_continuous_mode_with_small_timeout()
elif model_is_decoder_only_llm:
    use_dynamic_batching + continuous_slot_refill()
else:
    use_dynamic_batching_only()
```

这里的边界也要说清楚。连续批处理并不天然适合所有模型。它最适合自回归生成模型，因为生成过程天然按 token 迭代；而对很多一次性编码模型，动态批处理已经足够，继续上复杂调度器未必划算。

---

## 参考资料

1. BentoML, *Static, Dynamic and Continuous Batching*，2024 左右版本。核心用途：给出静态、动态、连续三种批处理的定义和触发方式，适合作为概念框架。
2. DevTechTools, *LLM Inference Optimization: Dynamic Batching*，2025。核心用途：给出动态批处理的触发公式、延迟与吞吐关系，以及 `max_batch_size=4`、`batch_timeout=20ms` 的数值例子。
3. Iterathon, *LLM Batch Inference Cost Optimization*，2026。核心用途：说明连续批处理在真实 LLM 服务中的迭代级调度方式，以及 4 到 8 倍量级的吞吐收益背景。
4. Inference Academy, *Adaptive LLM Inference Pipelines*，2025。核心用途：帮助理解动态参数调节、自适应推理流水线和负载变化下的调度边界。
