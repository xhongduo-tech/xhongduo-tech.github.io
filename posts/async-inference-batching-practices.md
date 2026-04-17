## 核心结论

异步推理解决的是“调用方是否要原地等待”的问题，批处理解决的是“模型是否要一条一条执行”的问题。前者把请求接收和模型执行解耦，后者把多个请求合并后统一送入 GPU/TPU 执行。两者组合后，系统不再按“请求到一条，模型跑一条”的方式工作，而是按“请求先进入缓冲区，再由 worker 选择合适时机发批”的方式工作。

对大多数在线推理服务，这种改造的直接收益不是“单次请求更快”，而是“单位时间能处理更多请求”。原因很简单：模型服务存在固定开销，例如框架调度、kernel 发射、张量拼接、内存搬运、结果回写。这些开销如果每个请求都重复付一次，GPU 很容易看起来在忙，实际有效计算却不高。把多个请求合并后，这部分固定成本就能被更多样本摊薄。

因此，工程判断的重点不是“理论上能不能批处理”，而是“业务能接受多少额外排队时间”。如果业务可以接受额外 10 到 20 毫秒的排队窗口，那么异步队列加动态批处理通常是吞吐优化的默认优先方案；如果接口 SLA 要求几乎实时返回，那么你能使用的队列窗口会很短，批处理收益也会被明显压缩。

下面先看总体对比：

| 方案 | 响应路径 | 延迟特征 | 吞吐率 | 资源利用 |
|---|---|---|---|---|
| 同步直连模型 | 请求到达后立刻执行 | 单次延迟低，但高峰时抖动大 | 低到中 | GPU 容易有空转 |
| 异步队列 | 先入队，再由后台执行 | 增加队列等待 | 中 | 接收层更稳 |
| 异步队列 + 微批 | 入队后由 worker 聚合成小批次执行 | 平均延迟略增，尾延迟可控 | 高 | GPU 利用率通常最高 |

可以先用一个新手友好的例子建立直觉。前端把推理请求写入 Kafka 后，API 立即返回 `202 Accepted` 和任务 ID；后台 worker 每 20ms 最多取 8 个请求组成一个微批，一起调用模型；结果计算完成后，再通过 webhook、轮询接口或结果表返回。这就是“异步接收 + 微批执行”的典型模式。调用方先拿到“已受理”的确认，模型稍后再真正执行。

---

## 问题定义与边界

这类优化要解决的核心矛盾只有一个：单请求推理更接近“随到随做”，路径短、语义直观，但固定开销无法摊薄；批处理更接近“攒一批再发车”，吞吐更高，但会引入额外等待时间。

可以用一个近似公式描述吞吐量：

$$
Throughput \approx \frac{BatchSize}{LaunchLatency + ComputeTime + QueueDelay}
$$

其中：

- $BatchSize$：单批请求数，也就是一次送给模型执行的样本数量。
- $LaunchLatency$：固定启动开销，例如调度、kernel 发射、张量组织、运行时切换。
- $ComputeTime$：真正做矩阵计算、编码、解码的时间。
- $QueueDelay$：请求为了等更多同伴凑成一批而额外等待的时间。

这个公式不是严格的性能定律，但足够指导工程决策。它表达了一个很实用的事实：如果固定开销不小，而单请求计算量又不足以把硬件打满，那么适度增大批大小通常能显著提高吞吐；但如果为了凑批让请求在队列里等太久，用户看到的总延迟就会上升。

为了避免把问题说得过于抽象，可以把一次请求的总延迟拆成：

$$
Latency = QueueDelay + ServiceTime
$$

其中：

$$
ServiceTime = LaunchLatency + ComputeTime + PostProcessTime
$$

这里的 `PostProcessTime` 可以理解成结果解码、序列化、写回存储、通知下游等尾部处理时间。异步和批处理主要优化的是 `ServiceTime` 里的固定开销摊销，同时用一个可控的 `QueueDelay` 换更高吞吐。

系统边界通常由下面四类约束决定：

| 维度 | 约束内容 | 超限后问题 |
|---|---|---|
| 批容量 | `max_batch_size` 不能无限增大 | 显存溢出、单批过慢 |
| 最大等待 | `max_delay_ms` 必须受控 | p95/p99 延迟上升 |
| 序列长度差异 | 长短文本混批会产生 padding | 无效计算增加 |
| 显存限制 | KV cache、激活值随批量增长 | OOM 或频繁回收 |

这里先解释两个新手常见术语。

`padding` 指的是把不同长度的输入补成同样长度时填进去的空白 token。它只是为了让张量形状一致，不提供业务价值，但会真实消耗算力和显存。

`KV cache` 指的是生成式模型在解码阶段保存的历史中间状态，后续生成新 token 时可以复用。它能减少重复计算，但会随着并发数、上下文长度和批大小上升而快速吃掉显存。

下面用一个简单场景说明边界如何起作用。假设团队发现高峰期吞吐不足，于是把 `max_batch_size` 从 16 提到 64，目标是让 GPU 更满。但实际流量并不稳定，很多时间段请求并不能快速凑满 64 条，于是大量请求要在队列里等待几十毫秒，最终 p95 延迟明显上升。这个时候，问题不一定出在“批还不够大”，更可能出在“等待窗口已经过长”。更合理的做法通常是把 `max_delay_ms` 控制在 10 到 20ms 之间，再通过压测寻找一个批大小和等待时间的折中点。

可以用一句话概括这里的边界判断：

- 吞吐优化的上限通常由显存和长度分布决定。
- 线上可接受的排队窗口通常由 SLA 决定。

---

## 核心机制与推导

异步推理系统通常可以拆成四个阶段：请求入队、worker 聚合、模型执行、结果回传。这里的 worker 可以理解成“后台处理进程”或“消费进程”，负责不断从队列里取任务，并决定什么时候发出一个批次。

一个简化时序如下：

1. 客户端发起推理请求。
2. API 服务校验参数、记录任务，并把请求写入队列。
3. API 立即返回任务 ID 或受理状态。
4. worker 从队列取出请求，直到“批满”或“等待超时”。
5. worker 把微批送给模型执行。
6. 结果写入结果存储，或者通过 webhook/事件总线通知调用方。
7. 调用方通过回调、轮询或订阅方式取得结果。

这里的“微批”不是大离线训练中的超大 batch，而是在线系统中为控制延迟而使用的小批次，常见大小是 4、8、16、32。它的目标不是追求数学上的最大吞吐，而是在可接受延迟内尽量提高资源利用率。

为什么微批通常有效？因为很多推理栈都存在固定成本。设单请求执行时间约为：

$$
T_1 = L + C
$$

其中：

- $L$：固定启动开销。
- $C$：单请求实际计算时间。

如果把 $B$ 个请求合成一批，批次执行时间更接近：

$$
T_B = L + B \cdot C^\prime + Q
$$

其中：

- $C^\prime$：批处理模式下单样本的平均计算成本。
- $Q$：为凑批产生的排队时间。

于是单位时间的处理能力近似为：

$$
\text{吞吐} \approx \frac{B}{L + B \cdot C^\prime + Q}
$$

如果 $L$ 原本不小，那么批处理的核心收益就是把“每个请求都支付一次的固定成本”变成“每一批只支付一次”。这就是常说的“固定启动开销被摊薄”。

再进一步，可以把吞吐提升的直觉写得更清楚一些。同步模式下，理论吞吐近似为：

$$
TPS_{sync} \approx \frac{1}{L + C}
$$

批处理模式下，理论吞吐近似为：

$$
TPS_{batch} \approx \frac{B}{L + B \cdot C^\prime + Q}
$$

两者的提升倍数约为：

$$
Gain \approx \frac{B(L+C)}{L + B \cdot C^\prime + Q}
$$

这个式子揭示了三个关键事实：

- 如果 $L$ 越大，批处理收益通常越明显。
- 如果 $Q$ 过大，收益会被排队时间抵消。
- 如果 $C^\prime$ 随批大小上升得很快，说明硬件已经接近饱和，再继续增大 batch 价值会下降。

看一个可计算的玩具例子。假设：

- 固定启动开销 $L = 6ms$
- 单请求实际计算 $C = 4ms$
- 批处理后单样本平均计算成本 $C^\prime = 2.5ms$
- `max_batch_size = 8`
- 平均排队时间 $Q = 12ms$

那么：

- 同步单请求延迟约为 $6 + 4 = 10ms$
- 同步吞吐约为 $1 / 10ms = 100 req/s$
- 8 个请求合批后，总时间约为 $6 + 8 \times 2.5 + 12 = 38ms$
- 批处理吞吐约为 $8 / 38ms \approx 210 req/s$

即使采用偏保守的参数，吞吐也已经接近同步模式的两倍。如果请求更平稳、长度更接近、排队时间更短，收益还会继续提高。

再看一个真实工程更接近的例子。某个摘要生成服务在白天高峰有大量 256 到 512 token 的相似长度请求。团队做了三件事：

| 调整项 | 改动 | 目的 |
|---|---|---|
| 队列窗口 | `max_delay_ms=10` | 让 worker 有机会攒批，但不让等待过长 |
| 微批大小 | `max_batch_size=16` | 摊薄启动开销 |
| 长度分桶 | 256/512/1024 三档 | 降低 padding 浪费 |

结果通常会比“只改一个参数”稳定得多。GPU 利用率提升、tokens/sec 提升、尾延迟仍然可控，这才是线上系统真正需要的状态。

---

## 代码实现

下面给出一个可以直接运行的 Python 示例，用最小实现说明“异步入队 + 动态批处理”的结构。代码只依赖 Python 3.10+ 标准库，不依赖真实消息队列，但流程和线上服务是一致的。

```python
import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Request:
    req_id: int
    payload: str
    future: asyncio.Future
    enqueue_ts: float


class BatchWorker:
    def __init__(self, max_batch_size: int = 4, max_delay_ms: int = 20) -> None:
        self.queue: asyncio.Queue[Request] = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.max_delay_ms = max_delay_ms
        self._running = True

    async def enqueue_request(self, req_id: int, payload: str) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        req = Request(
            req_id=req_id,
            payload=payload,
            future=future,
            enqueue_ts=time.perf_counter(),
        )
        await self.queue.put(req)
        return await future

    async def fake_model_infer(self, batch: list[Request]) -> list[str]:
        # 模拟模型推理时间。真实场景中这里会调用 Triton、vLLM
        # 或其他推理服务，并返回与 batch 顺序一致的结果。
        await asyncio.sleep(0.01)
        return [item.payload.upper() for item in batch]

    async def worker_loop(self) -> None:
        while self._running:
            first = await self.queue.get()
            batch = [first]
            batch_start = time.perf_counter()

            while len(batch) < self.max_batch_size:
                elapsed_ms = (time.perf_counter() - batch_start) * 1000
                remaining_ms = self.max_delay_ms - elapsed_ms
                if remaining_ms <= 0:
                    break

                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=remaining_ms / 1000,
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            infer_start = time.perf_counter()
            results = await self.fake_model_infer(batch)
            infer_ms = (time.perf_counter() - infer_start) * 1000

            now = time.perf_counter()
            for req, result in zip(batch, results):
                total_ms = (now - req.enqueue_ts) * 1000
                queue_ms = total_ms - infer_ms
                if not req.future.done():
                    req.future.set_result(
                        {
                            "req_id": req.req_id,
                            "output": result,
                            "batch_size": len(batch),
                            "queue_delay_ms": round(max(queue_ms, 0), 2),
                            "service_time_ms": round(infer_ms, 2),
                            "total_latency_ms": round(total_ms, 2),
                        }
                    )

            for _ in batch:
                self.queue.task_done()

    async def shutdown(self) -> None:
        self._running = False


async def demo() -> None:
    worker = BatchWorker(max_batch_size=4, max_delay_ms=20)
    worker_task = asyncio.create_task(worker.worker_loop())

    # 模拟请求在短时间内陆续到达，而不是同一时刻全部出现。
    async def submit(req_id: int, payload: str, delay_ms: int) -> dict[str, Any]:
        await asyncio.sleep(delay_ms / 1000)
        return await worker.enqueue_request(req_id, payload)

    results = await asyncio.gather(
        submit(1, "hello", 0),
        submit(2, "batch", 5),
        submit(3, "world", 8),
        submit(4, "async", 12),
        submit(5, "later", 50),
    )

    for item in results:
        print(item)

    assert [item["req_id"] for item in results] == [1, 2, 3, 4, 5]
    assert [item["output"] for item in results] == [
        "HELLO",
        "BATCH",
        "WORLD",
        "ASYNC",
        "LATER",
    ]
    assert results[0]["batch_size"] == 4
    assert results[4]["batch_size"] == 1

    await worker.shutdown()
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task


if __name__ == "__main__":
    asyncio.run(demo())
```

这段代码体现了动态批处理最重要的两个触发条件：

- 达到 `max_batch_size`，立即发射。
- 未达到批大小时，最多只等 `max_delay_ms`。

如果你直接运行，会看到前 4 个请求被合成一个批，第 5 个请求因为到达时间较晚，单独形成下一批。这个行为正好对应线上系统最常见的两种情况：高峰期快速成批，低流量时超时发射。

可以把这段代码映射到真实工程组件中理解：

| 示例组件 | 线上对应物 | 作用 |
|---|---|---|
| `enqueue_request` | HTTP API / gRPC 接口 | 接收请求，生成任务上下文 |
| `asyncio.Queue` | Kafka / Redis Stream / SQS | 暂存请求，解耦接收与执行 |
| `worker_loop` | 后台消费者 / 推理 worker | 拉取请求，组织微批 |
| `fake_model_infer` | Triton / vLLM / 自研服务 | 执行真正的模型推理 |
| `future.set_result` | DB 写回 / webhook / 结果缓存 | 返回或持久化结果 |

参数含义可以先记住下面几个：

| 参数 | 作用 | 常见取值思路 |
|---|---|---|
| `max_batch_size` | 单批最多收多少请求 | 先从 4/8/16 做压测 |
| `max_delay_ms` | 为了凑批最多等待多久 | 常见 5 到 20ms |
| `queue_name` | 队列标识 | 按模型、租户、优先级拆分 |
| `retry_limit` | 失败重试次数 | 常见 2 到 5 次 |
| `result_ttl` | 结果保留时长 | 取决于轮询窗口与补偿需求 |

如果你使用 NVIDIA Triton，动态批处理配置通常类似：

```txt
max_batch_size: 16

dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 20000
}
```

这里的 `max_queue_delay_microseconds: 20000` 表示“最多等待 20ms”，不是“必须等满 20ms”。如果更早凑到合适批次，Triton 会提前发射。

再补一个更完整的工程落地示例。假设你在做图片审核服务：

1. 用户上传图片，API 返回任务 ID。
2. 请求元数据进入 Redis Stream。
3. worker 每次抓取最多 16 张图片组成一个批。
4. 批量送入 GPU 分类器。
5. 结果写回 MySQL 或对象存储。
6. 回调服务通知业务系统更新审核状态。

这种设计的关键价值不是“前端更快拿到分类结果”，而是“接入层不再被模型执行时间直接卡住”，因此高峰期更稳。

---

## 工程权衡与常见坑

异步和批处理不是“打开开关就自动提速”。真正困难的部分在于：你在提升吞吐的同时，也引入了队列、回调、重试、排序、一致性和观测性问题。

第一个常见坑是长度差异过大。对大语言模型来说，如果一个批里既有 128 token 的短请求，也有 2048 token 的长请求，那么短请求往往要被补齐到接近长请求长度。结果是 GPU 做了很多无效计算，吞吐看起来上去了，但有效 token 吞吐并不高。

这时通常会引入 `length bucket`，也就是“长度分桶”。做法不是把所有请求混在一起批处理，而是先按长度范围分组，再在组内形成批次。一个简单分桶方式如下：

| 桶 | 适合的输入长度 | 目的 |
|---|---|---|
| Bucket A | 1 到 256 tokens | 让短请求优先和短请求组批 |
| Bucket B | 257 到 512 tokens | 降低中等长度请求的 padding |
| Bucket C | 513 到 1024 tokens | 防止长请求拖慢短请求 |
| Bucket D | 1025+ tokens | 单独控制显存和尾延迟 |

分桶的收益通常体现在三个方面：

- padding 比例下降。
- 显存占用更稳定。
- 批执行时间波动变小。

第二个常见坑是显存估计过于乐观。批大小不是唯一决定显存使用的因素，上下文长度、KV cache、模型类型、精度、并发生成步数都会影响实际占用。很多线上 OOM 不是因为“平均情况超了”，而是因为“极端输入长度 + 批量峰值 + 缓存峰值”同时出现。

实际工程里，显存控制通常要同时看下面几项：

| 观测项 | 为什么要看 | 典型风险 |
|---|---|---|
| Batch size | 批越大，并发样本越多 | 单批时间过长 |
| Prompt length | 上下文越长，占用越高 | KV cache 激增 |
| Decode length | 生成越长，占用持续时间越久 | 长尾请求拖住 worker |
| GPU memory watermark | 接近上限时容易抖动 | OOM、频繁回收 |

第三个常见坑是队列堆积。异步化以后，API 层可能很快，但这不代表系统整体健康。真正的风险是“入口很稳，出口已经堵住”。如果只监控 HTTP 成功率，而不监控队列长度、消费速率和等待时间，你会误以为服务可用，实际上用户结果已经大面积延迟。

建议至少监控下面这些指标：

| 指标 | 说明 | 作用 |
|---|---|---|
| queue depth | 当前排队任务数 | 看是否积压 |
| queue wait p50/p95/p99 | 请求入队到出队的等待时间 | 看排队是否失控 |
| batch size distribution | 实际批大小分布 | 看批处理是否真的生效 |
| tokens/sec 或 samples/sec | 有效吞吐 | 看优化是否带来实际收益 |
| GPU utilization | GPU 利用率 | 看硬件是否被打满 |
| OOM count | 显存溢出次数 | 看批大小是否激进 |
| retry count | 重试次数 | 看执行链路是否稳定 |

第四个常见坑是结果回传不可靠。异步系统里，结果常通过 webhook、轮询结果表、消息总线等方式返回。此时最容易被忽视的是幂等性和重复投递问题。网络抖动、下游超时、回调失败都可能导致同一结果被重复发送。

因此，结果回传链路通常至少要有：

| 问题 | 现象 | 规避措施 |
|---|---|---|
| Padding 过多 | GPU 很忙，但有效 token 吞吐不高 | 按长度分桶、sequence grouping |
| 显存爆满 | OOM、进程重启、吞吐反降 | 限制批大小，预留 10% 到 20% 显存 |
| 队列堆积 | p95/p99 延迟明显上升 | admission control、限流、扩容 |
| Webhook 失败 | 结果丢失或重复通知 | 幂等键、重试、dead-letter |
| 批太大 | 单批时间过长，尾延迟上升 | 用压测选折中点，不盲目追大批 |

这里再解释两个术语。

`admission control` 可以理解成“入场控制”。当队列已经过长、显存接近上限、下游结果系统不可用时，系统不应该继续无限接收请求，而应该限流、降级或明确拒绝一部分流量。

`dead-letter queue` 可以理解成“死信队列”。那些重试多次仍然失败、暂时不适合继续自动处理的任务，需要被单独保存，等待人工分析或离线补偿。没有这一步，失败任务很容易在系统里悄悄消失。

一条相对完整的失败处理链路通常是：

1. 执行失败后按指数退避重试。
2. 超过重试上限后写入 dead-letter queue。
3. 记录失败原因、输入摘要、模型版本、批次信息。
4. 由补偿任务或人工流程重新处理。

如果你的系统要面向新手同学交接，最好把“什么是成功”定义得非常具体。对于异步推理来说，真正的成功不是“请求成功入队”，而是“结果被正确执行、正确落盘、正确回传，并且能被调用方可靠取回”。

---

## 替代方案与适用边界

异步加批处理不是唯一方案，它适合的是“并发较高，而且允许小幅排队”的推理服务。如果业务本身 QPS 不高，或者极端强调单请求响应时间，那么同步执行往往更简单，也更容易维护。

常见方案可以放在一起比较：

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 异步 + 动态批处理 | 高 QPS，可接受 10 到 20ms 排队 | 吞吐高，GPU 利用率高 | 系统复杂度更高 |
| 同步直连模型 | 低 QPS，极低延迟要求 | 路径短，调试简单 | 算力利用率低 |
| Edge cache / 结果缓存 | 请求高度重复 | 成本最低，响应最快 | 只对可缓存请求有效 |
| 模型预热 + 常驻实例 | 启动抖动明显 | 降低冷启动影响 | 不解决高并发调度问题 |
| 多副本扩容 | 流量波动大 | 实现直接 | 成本上升快 |

为了方便决策，可以用一个简单规则：

- 高 QPS，允许额外 20ms 左右延迟：优先异步 + 批处理。
- 低 QPS，或接口 SLA 极严：优先同步直连。
- 请求重复度高：先做缓存，再谈批处理。
- 单机显存接近上限：先做长度分桶和 admission control，再考虑放大批。
- 下游回调链路不稳定：先解决结果可靠性，再引入异步队列。

下面给出一个更直观的决策表：

| 业务特征 | 更合适的方向 | 原因 |
|---|---|---|
| 在线聊天首 token 极敏感 | 同步直连或很小微批 | 用户直接感知排队时间 |
| 批量文档摘要 | 异步 + 动态批处理 | 容易接受几十毫秒等待 |
| 图片离线审核 | 异步 + 大一点的批 | 吞吐优先于单次响应 |
| 高重复问答接口 | 结果缓存优先 | 避免重复推理最划算 |
| 极端低延迟风控打分 | 同步小模型优先 | 排队窗口通常不可接受 |

真实工程里，金融实时评分就是一个边界明确的反例。若风险评估接口必须在极短时间内返回，额外排队几毫秒都可能影响整体链路，那么即使 GPU 理论上能靠批处理提高吞吐，也不一定应该采用异步队列。更现实的做法可能是同步调用更小的模型，或者把重模型放到离线补充流程中。

同样，聊天生成服务也不是一刀切都适合大批处理。原因在于用户对首 token 延迟非常敏感。即使总吞吐提升了，如果首 token 时间显著变差，用户体验仍然会下降。因此在生成式场景中，经常需要把“prefill 阶段”和“decode 阶段”分开看，甚至针对不同阶段采用不同的调度策略。

---

## 参考资料

| 来源名称 | 核心内容摘要 | 学习重点 |
|---|---|---|
| AWS SageMaker 异步推理文档 | 介绍异步请求提交、结果存储、回调和任务生命周期 | 理解异步 API 形态、结果落地方式 |
| Baseten 异步推理文档 | 展示 fire-and-forget 式推理接口和异步结果查询模式 | 理解异步调用对接方式 |
| NVIDIA Triton 动态批处理指南 | 说明 `max_batch_size`、`preferred_batch_size`、队列延迟窗口等配置 | 学会服务端动态批处理落地 |
| vLLM 调度与连续批处理资料 | 展示在线生成任务如何通过更细粒度调度提高吞吐 | 理解生成式场景下的批处理差异 |
| 长度分桶与 bucket batching 资料 | 说明按序列长度分桶以降低 padding 浪费 | 理解长度分布对吞吐的直接影响 |
| 队列系统文档（Kafka / Redis Streams / SQS） | 介绍消息投递、消费、重试、死信等机制 | 理解异步链路的可靠性基础 |
| 可观测性资料（延迟分位数、队列指标） | 说明如何监控 p95/p99、队列深度、消费速率 | 学会判断系统是否真的优化成功 |

阅读这些资料时，建议按下面三个问题去抓重点：

1. 请求如何进入队列，调用方如何拿到任务状态。
2. worker 何时触发批次发射，批大小和等待时间如何一起控制。
3. 执行失败、回调失败、结果重复时，系统如何重试、保底和补偿。

如果只看配置项，很容易把问题误解为“把 `max_batch_size` 调大就行”。真正稳定的异步推理系统，核心不是某一个参数，而是四件事一起成立：接收层稳、批处理有效、失败可恢复、指标可观测。
