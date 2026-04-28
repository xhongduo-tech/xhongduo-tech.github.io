## 核心结论

Triton 推理服务器的动态批处理，本质是在服务器调度器里把短时间内到达的多个请求合并成一个 batch 后再送给模型执行。batch 可以理解为“一次并行处理的多条样本”。它解决的不是“模型能不能算”，而是“GPU 能不能被更充分地喂满”。

它最核心的收益是提升吞吐。吞吐指单位时间内系统能处理多少请求，常用 QPS 表示。原因很直接：对 GPU 而言，很多模型在 batch 从 1 增长到 4、8、16 时，单次执行时间不会线性增长，但一次能处理的请求数会增加，因此单位时间的有效产出更高。

代价也同样明确：动态批处理会引入排队等待。最早进入队列的请求必须等后续请求到来，才有机会被拼进更大的 batch。因此平均延迟可能小幅变化，但 `P95/P99` 这类尾延迟指标往往更敏感，配置不当时很容易先恶化。

下面这个对比可以先把三种常见方式分开：

| 方式 | 谁来拼 batch | 主要收益 | 主要代价 |
|---|---|---|---|
| 不 batching | 无 | 延迟路径最短、行为稳定 | GPU 利用率低 |
| 客户端 batching | 调用方 | 控制强、策略可定制 | 客户端复杂，跨业务难统一 |
| Triton 动态批处理 | Triton 服务器 | 提升吞吐、接入简单 | 增加排队延迟 |

玩具例子：一个图片分类模型，单张图片推理耗时 2ms，batch=4 时总耗时 4ms。此时单请求执行成本从 2ms 降到约 1ms，GPU 利用率明显提升，但最早到达的那张图可能要额外等待几十到几百微秒。

真实工程例子：在线 OCR 服务里，用户上传图片的时间点非常零散。若每张图都单独推理，GPU 经常处于“启动了计算，但没吃满”的状态。启用动态批处理后，Triton 会在一个很短的窗口内把多张图凑成 batch 交给 TensorRT 或 ONNX Runtime 后端，整体 QPS 提升，但如果 `max_queue_delay_microseconds` 设得过大，业务的 `P95` 延迟可能直接越过 SLA。

---

## 问题定义与边界

理解 Triton 动态批处理，先抓住三个旋钮：

| 符号 | Triton 配置项 | 含义 |
|---|---|---|
| $D$ | `max_queue_delay_microseconds` | 最老请求最多允许等多久 |
| $P$ | `preferred_batch_size` | 调度器优先想凑成的 batch 尺寸集合 |
| $N$ | `instance_group.count` | 该模型同时可运行的实例副本数 |

`max_queue_delay_microseconds` 是最大排队等待时间，单位是微秒。它不是“固定等这么久”，而是“最多可以等这么久”。  
`preferred_batch_size` 是优选 batch 大小，意思不是只允许这些大小，而是如果能凑到这些大小，调度器会优先发。  
`instance_group.count` 是模型实例数，可以理解为同一个模型在 GPU 上开的并行工作副本。

边界也要先说清楚。动态批处理不是所有模型都适合，前提通常有三个：

1. 模型输入支持 batch 维。
2. 请求彼此独立，不依赖前后顺序。
3. 模型最好是无状态。

无状态的意思是本次推理不依赖上一次请求留下的内部上下文。典型的分类、检测、OCR、排序模型通常满足；强状态对话、会话记忆模型、严格顺序处理任务通常不满足。

| 场景 | 是否适合动态批处理 | 原因 |
|---|---|---|
| 图像分类、OCR、向量召回 | 适合 | 请求独立，batch 收益明显 |
| 搜索排序、推荐粗排 | 适合 | 小请求密集，GPU 易被喂满 |
| 有状态会话模型 | 不适合直接套用 | 请求间存在顺序和状态依赖 |
| 严格超低延迟接口 | 视情况 | 任何排队都可能触发 SLA 风险 |

一个常见误区是把“高并发”直接等同于“适合动态批处理”。这不准确。高并发只是提供了“有机会凑 batch”的流量条件，最终还要看模型是否支持 batch、batch 后是否真的更快，以及业务是否允许增加一点排队时间。

---

## 核心机制与推导

调度器的行为可以简化成一句话：先尽量凑成更有价值的 batch，凑不到时再看是否值得继续等。

设当前模型配置为：

- `max_batch_size=8`
- `preferred_batch_size=[4, 8]`
- `max_queue_delay_microseconds=100`

调度流程大致如下：

1. 新请求进入队列。
2. 调度器查看当前队列是否能组成 `preferred_batch_size` 中的某个值。
3. 如果能组成 4 或 8，就优先立刻发射。
4. 如果凑不到 preferred size，再检查最老请求已经等了多久。
5. 若最老请求等待时间 $w_{oldest} < D$，可以继续等新请求。
6. 若 $w_{oldest} \ge D$，立即发出当前可形成的 batch，不再等待。

可以用一个近似规则描述：

$$
\text{batch} =
\begin{cases}
\text{最大的可形成 preferred batch}, & \text{如果存在} \\
\text{当前可用的最大 batch}, & \text{否则}
\end{cases}
$$

以及等待条件：

$$
w_{oldest} < D \Rightarrow \text{允许继续等}
$$

$$
w_{oldest} \ge D \Rightarrow \text{必须立即发射}
$$

玩具例子：

- `t=0us` 到达 3 个请求
- `t=60us` 再到 1 个请求

此时队列从 3 条变成 4 条，正好命中 `preferred_batch_size=4`，调度器会立刻发出 batch 4。

如果第 4 个请求没有来：

- `t=0us` 到达 3 个请求
- `t=100us` 时最老请求已等满 `D`

那么 Triton 会直接发出 batch 3，而不是继续等到 4。

吞吐可以粗略写成：

$$
QPS \approx \frac{N \times B}{S(B)}
$$

其中：

- $N$ 是实例副本数
- $B$ 是实际 batch 大小
- $S(B)$ 是该 batch 的服务时间，也就是执行一个 batch 需要多久

这个公式不是精确排队模型，但足够说明方向：如果 batch 变大后，$S(B)$ 增长慢于 $B$，吞吐就会上升。

延迟可以近似拆成：

$$
Latency \approx QueueWait + ExecutionTime
$$

因此 `P95 latency` 上升，往往不是模型本身变慢，而是 `QueueWait` 增长了。很多调优失败案例都不是算力不够，而是排队策略把尾延迟推高了。

真实工程例子：一个搜索排序模型单请求推理 3ms，batch=8 时总推理 8ms。若业务流量足够密集，单实例吞吐会显著提高；但如果 `D` 从 50us 调到 2000us，虽然 batch 更容易变大，最老请求却可能白白多等 2ms，这对 `P95 < 20ms` 的链路往往已经是危险值。

---

## 代码实现

从配置看，动态批处理最小可用形式通常就是 `config.pbtxt` 里的两块：`dynamic_batching` 和 `instance_group`。

```pbtxt
name: "ranker"
platform: "tensorrt_plan"
max_batch_size: 8

dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 100
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]
```

这个配置表达的意思是：

- 模型最大支持 batch 8。
- 调度器优先凑 batch 4 或 8。
- 最老请求最多等 100 微秒。
- 同时可有 2 个模型实例并行执行。

逻辑上可以用下面的伪代码理解：

```text
请求入队
  -> 查看当前是否能形成 preferred batch
  -> 如果能，立即发射
  -> 如果不能，检查最老请求等待时间
  -> 未超过 D，继续等待
  -> 超过 D，发射当前 batch
  -> 把 batch 交给空闲实例执行
```

下面给一个可运行的 Python 玩具模拟。它不是 Triton 源码复刻，只是把核心调度规则压成最小模型，便于理解“等多久”和“何时发射”的关系。

```python
from dataclasses import dataclass

@dataclass
class Request:
    id: int
    t_us: int

def simulate(arrivals, preferred=(4, 8), max_batch_size=8, max_delay_us=100):
    queue = []
    launched = []

    def flush(now_us):
        nonlocal queue, launched
        if not queue:
            return False

        size = min(len(queue), max_batch_size)

        # 优先找当前能组成的最大 preferred batch
        preferred_candidates = [p for p in preferred if p <= size]
        if preferred_candidates:
            b = max(preferred_candidates)
            launched.append((now_us, [r.id for r in queue[:b]]))
            queue = queue[b:]
            return True

        oldest_wait = now_us - queue[0].t_us
        if oldest_wait >= max_delay_us:
            launched.append((now_us, [r.id for r in queue[:size]]))
            queue = queue[size:]
            return True

        return False

    for i, t in enumerate(arrivals):
        queue.append(Request(i, t))
        while flush(t):
            pass

    # 收尾：把剩余请求推进到超时点
    while queue:
        now = queue[0].t_us + max_delay_us
        while flush(now):
            pass

    return launched

case1 = simulate([0, 0, 0, 60], preferred=(4, 8), max_delay_us=100)
assert case1 == [(60, [0, 1, 2, 3])]

case2 = simulate([0, 0, 0], preferred=(4, 8), max_delay_us=100)
assert case2 == [(100, [0, 1, 2])]

print(case1)
print(case2)
```

这个例子对应前面的机制：

- 第一个场景在 `60us` 凑到 batch 4，立即发射。
- 第二个场景一直凑不到 4，到 `100us` 触发超时，发射 batch 3。

如果把视角拉到 Triton 实现层，可以把它理解成两段职责：

| 位置 | 主要职责 |
|---|---|
| 入队逻辑 | 接收请求、放入调度队列 |
| 动态批调度逻辑 | 判断是否形成 preferred batch、是否超时、何时发射 |

因此配置只是表层，真正的运行结果还受后端执行时间、模型实例空闲情况、不同大小 batch 的实际性能曲线共同影响。

---

## 工程权衡与常见坑

第一个权衡是 `D`。它决定你愿意拿多少延迟去换 batch 大小。

- `D` 太小：请求刚进来就发走，batch 变碎，GPU 利用率低。
- `D` 太大：batch 变大更容易，但排队时间增加，`P95/P99` 容易恶化。

第二个权衡是 `preferred_batch_size`。很多人会把它配成一长串，例如 `[2,4,8,16,32]`，这通常不是好习惯。原因是 preferred size 应该反映“这些尺寸在硬件或后端上真的更值”，而不是“理论上都能跑”。如果 TensorRT profile 只对 8 和 16 明显更优，那就只保留 8 和 16。

第三个权衡是 `instance_group.count`。实例数增加不一定带来线性吞吐提升。因为请求会被多个实例分流，单个实例前的队列会变短，反而更难凑出大 batch。于是你可能看到一种现象：并发看起来更高了，但平均 batch 变小，单位实例效率下降，最终总体收益不如预期。

真实工程例子：一个在线排序服务原本 `P95=17ms`。团队把 `max_queue_delay_microseconds` 从 100us 提到 3000us，希望把 batch 从 4 提升到 16，同时把 `instance_group.count` 从 1 提到 4。结果 QPS 上升了 20%，但 `P95` 升到 29ms，`P99` 抖动更明显。根因不是单一参数错误，而是三个旋钮一起推高了排队和分流效应，导致吞吐增长与 SLA 目标冲突。

下面这张排查表比较实用：

| 参数 | 典型风险 | 观察指标 |
|---|---|---|
| `max_queue_delay_microseconds` | 尾延迟升高 | `P95/P99`、队列等待时间 |
| `preferred_batch_size` | 命中率低、收益不明显 | 实际 batch 分布、QPS |
| `instance_group.count` | 请求被分散，batch 变小 | 单实例平均 batch 大小 |
| `max_batch_size` | 配置与模型能力不匹配 | 是否真的形成 batch、后端报错 |

几个常见坑：

1. `max_queue_delay_microseconds=0` 不等于“尽量凑满再发”，它更接近“不主动等待”。
2. `max_batch_size` 配大了但模型输入维度或后端不支持，结果是根本批不起来。
3. 只看平均延迟，不看 `P95/P99`，很容易误判调优效果。
4. 忽略 batch 大小时的性能曲线，盲目追求更大 batch，最终可能因为显存、profile 或 kernel 行为导致收益变差。
5. 在低并发时启用动态批处理，调度器长期凑不满，只留下额外等待，没有吞吐红利。

---

## 替代方案与适用边界

动态批处理适合的是“无状态、独立、小请求、高并发、batch 收益明显”的组合。如果这几个条件不成立，就该考虑替代方案。

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| 动态批处理 | 服务端统一调度，吞吐高 | 增加排队延迟 | 无状态、高并发 |
| 客户端 batching | 调用方控制强 | 客户端改造复杂 | 上游可控、调用集中 |
| 静态 batching | 行为稳定、便于评估 | 灵活性差 | 请求模式稳定 |
| Sequence batcher | 适合状态/序列模型 | 配置和理解更复杂 | 有状态、顺序相关任务 |

客户端 batching 适合调用方本来就能聚合请求的场景。比如离线特征计算、批量文档分类，客户端天然一次拿到很多样本，这时直接在客户端拼 batch 更可控。

静态 batching 适合请求模式稳定的系统，例如固定每 10ms 收一批日志、每秒处理一轮任务。它的优点是行为可预期，延迟和资源曲线更容易算。

Sequence batcher 适合有状态模型。状态模型指推理过程依赖会话上下文或前一时刻状态，例如多轮会话、流式序列任务。对这种场景，直接使用动态批处理很容易把语义顺序打乱，正确方案通常是 sequence 级别的调度，而不是普通 request 级别的合并。

所以一个实用判断可以压成三句话：

1. 先问模型是否支持 batch。
2. 再问请求是否彼此独立。
3. 最后问业务是否能接受额外排队时间。

三者都满足，动态批处理通常值得试；缺一个，就要谨慎。

---

## 参考资料

1. [NVIDIA Triton 文档: Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
2. [NVIDIA Triton 文档: Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2530/user-guide/docs/user_guide/model_configuration.html)
3. [NVIDIA Triton 文档: Concurrent Model Execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_execution.html)
4. [NVIDIA Triton 教程: Dynamic Batching & Concurrent Model Execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html)
5. [Triton 源码: dynamic_batch_scheduler.cc](https://raw.githubusercontent.com/triton-inference-server/core/main/src/dynamic_batch_scheduler.cc)
