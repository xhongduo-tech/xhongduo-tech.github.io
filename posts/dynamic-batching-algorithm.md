## 核心结论

动态批处理不是“把 batch 尽量做大”，而是在请求持续到达时，在线决定“什么时候发一批、这一批发多大”。它解决的是 GPU 推理里的并行度碎片化问题，也就是算力被很多很小的任务切碎，导致卡很忙但没吃满。

对工程最重要的三个变量是 `B_max`、`D_max` 和队列策略。`B_max` 是最大批大小，白话讲就是“一车最多装多少件”；`D_max` 是最大等待时间，白话讲就是“最早到的请求最多能等多久”；队列策略决定谁先上车、会不会插队、会不会让少数请求长期排不到。最基本的发批规则可以写成：

$$
b_t = \min(B_{max}, q_t)
$$

这里 `q_t` 表示时刻 $t$ 队列里当前可参与组批的请求数，`b_t` 表示这次真正发出去的批大小。

新手可以先用“收快递”理解。快递员不是每来一个包裹就单独开车送，而是会等一小会儿，把同方向的包裹凑成一车再发。这样单次运输更划算，但最早来的包裹一定会多等几分钟。动态批处理就是把这个思路放到 GPU 推理里。

它最适合无状态、单次推理、批内样本互不依赖的在线服务，比如图像分类、embedding、ranking、重排序。它不适合直接照搬到 LLM 逐 token 生成，因为生成式推理每一步都要带着历史状态继续算，批的组成会不断变化。

下表先给出最重要的直观对比：

| 方案 | 吞吐 | GPU 利用率 | 平均延迟 | 尾延迟 `p95/p99` |
|---|---:|---:|---:|---:|
| 单条推理 | 低 | 低 | 低到中 | 稳定但上限不高 |
| 动态批处理 | 中到高 | 高 | 中 | 容易升高 |
| 大批激进合批 | 很高 | 很高 | 高 | 高，且更容易抖动 |

判断一个动态批处理方案是否可用，不是看 batch 是否够大，而是看它能不能在 SLA 约束下稳定满足：

$$
R_i = W_i + S(b_t)
$$

这里 `R_i` 是第 $i$ 个请求总响应时间，`W_i` 是等待时间，`S(b_t)` 是该批的服务时间。工程上的核心平衡点，就是让吞吐收益大于等待成本。

---

## 问题定义与边界

动态批处理要解决的问题很具体：单请求 GPU 推理并行度低。模型一次只处理一个样本时，很多算子无法把硬件吃满，出现明显的算力浪费。于是系统会在请求进入服务后，先放进队列，短暂等待更多请求到来，再拼成一批送进模型。

一个常见场景是图像分类在线接口。用户上传图片后，请求先入队；如果在 `5ms` 内来了更多请求，系统把它们一起推给模型；如果 `5ms` 到了仍没凑满，也必须发出去，避免某个请求无限等待。这个“最多只能等多久”的边界可以写成：

$$
t - a_i \le D_{max}
$$

其中 `a_i` 是请求到达时刻，$t$ 是当前决策时刻。超过这个阈值，就该强制放行。

先把术语固定下来：

| 术语 | 含义 | 白话解释 |
|---|---|---|
| `B_max` | 最大批大小 | 一次最多装多少请求 |
| `D_max` | 最大等待时间 | 最早那个请求最多等多久 |
| `q_t` | 当前可组批请求数 | 队列里现在有多少件能一起发 |
| `W_i` | 第 `i` 个请求等待时间 | 从到达到真正开始算之间等了多久 |
| `S(b_t)` | 批大小为 `b_t` 的服务时间 | GPU 真正执行这一批用了多久 |

动态批处理和普通批处理的区别，不在于“有没有 batch”，而在于 batch 是离线固定的，还是在线决定的。普通批处理常见于训练或离线推理，数据先攒好再整批处理；动态批处理则是请求已经开始来了，系统边接边调度。

| 类型 | 数据何时凑齐 | 调度决策是否在线 | 典型场景 |
|---|---|---|---|
| 普通批处理 | 处理前已凑齐 | 否 | 训练、离线打分 |
| 动态批处理 | 请求到达过程中凑齐 | 是 | 在线分类、embedding |
| 不适用场景 | 无法一次性完成输出 | 通常需要别的调度方式 | LLM 逐 token 生成 |

边界判断可以用一段极小伪代码看清：

```python
def choose_mode(requests_are_streaming, output_is_one_shot, samples_independent):
    if not requests_are_streaming:
        return "普通批处理"
    if requests_are_streaming and output_is_one_shot and samples_independent:
        return "动态批处理"
    return "不适合直接用普通动态批处理"

assert choose_mode(False, True, True) == "普通批处理"
assert choose_mode(True, True, True) == "动态批处理"
assert choose_mode(True, False, False) == "不适合直接用普通动态批处理"
```

这里的 `output_is_one_shot` 可以理解为“模型一次前向就把结果吐完”，比如分类分数、向量、排序分值。如果结果要分多步生成，就已经越过了本文讨论边界。

---

## 核心机制与推导

动态批处理的核心机制只有一句话：请求先排队，再按规则放行。这个规则通常同时受三件事约束。

1. 达到最大批大小就放行。
2. 最老请求达到最大等待时间就放行。
3. 队列策略决定具体挑哪些请求进入这一批。

最简单的批大小选择就是：

$$
b_t = \min(B_{max}, q_t)
$$

如果没有超时压力，`q_t` 越大，通常越容易获得更高吞吐。工程上常用一个近似来判断趋势：

$$
throughput \approx \frac{b_t}{S(b_t)}
$$

这不是严格排队论定理，而是非常常见的工程近似。含义很直接：一次处理更多样本，如果服务时间增长得没有批大小快，单位时间完成的请求数就会上升。

但单个请求的体验不能只看吞吐，而要拆成等待和执行两段：

$$
R_i = W_i + S(b_t)
$$

这条式子解释了为什么“批越大越好”是错的。因为增大 `b_t` 往往同时带来两件事：`S(b_t)` 可能上升，`W_i` 也会因为更久的聚批而上升。尤其在低流量或流量抖动时，`W_i` 常常比 `S(b_t)` 更快恶化。

看一个玩具例子。设：

- `B_max = 4`
- `D_max = 5ms`
- `S(b) = 2 + 0.5b ms`

3 个请求分别在 `0ms / 2ms / 4ms` 到达。到 `4ms` 时，队列里有 3 个请求，系统形成 `b=3` 的批并立即发出，此时：

$$
S(3) = 2 + 0.5 \times 3 = 3.5ms
$$

于是时序如下：

| 请求 | 到达时刻 | 发批时刻 | 批大小 | 等待时间 `W_i` | 总时延 `R_i` |
|---|---:|---:|---:|---:|---:|
| r1 | 0.0ms | 4.0ms | 3 | 4.0ms | 7.5ms |
| r2 | 2.0ms | 4.0ms | 3 | 2.0ms | 5.5ms |
| r3 | 4.0ms | 4.0ms | 3 | 0.0ms | 3.5ms |

这个例子说明两点。

第一，最早到达的请求会承担最多等待成本。第二，虽然 3 次单条推理看起来总服务时间也可能接近 `3 × 2.5 = 7.5ms`，但 GPU 会经历 3 次独立启动、调度和低并行度执行，硬件利用率通常更差。动态批处理的价值，往往体现在吞吐和设备利用率，而不是每个单请求都更快。

再看一个真实工程例子。假设你在 Triton 上部署图像 embedding 服务，请求量在高峰时每秒几百到几千。若不开动态批处理，GPU 会频繁处理很多 `batch=1` 的前向；若开了 `B_max=8`、`D_max=2ms`，高峰期大部分请求会自动凑成 `4~8` 的批，吞吐明显提升。但如果你把 `D_max` 拉到 `20ms`，虽然批更大，`p99` 很可能先炸，因为最早那批请求都在等人凑车。

调度伪代码可以写成下面这样：

```python
def maybe_dispatch(queue, now_ms, B_max, D_max):
    if not queue:
        return None

    oldest_arrival = queue[0]["arrival_ms"]
    wait_ms = now_ms - oldest_arrival
    batch_size = min(B_max, len(queue))

    # 达到上限或最老请求已超时，都必须发批
    if batch_size == B_max or wait_ms >= D_max:
        batch = queue[:batch_size]
        del queue[:batch_size]
        return batch

    return None
```

这段逻辑短，但已经包含动态批处理的基本骨架：队列检查、批上限、超时强制放行。

---

## 代码实现

工程里通常把实现拆成三层：请求入队、后台调度、模型执行与结果回填。这样代码职责清楚，也方便替换底层推理引擎。

先给出一个最小可运行版本。它不是高性能生产代码，但能完整展示“到达、排队、超时、发批”四件事。

```python
from dataclasses import dataclass

@dataclass
class Request:
    req_id: str
    arrival_ms: float

class DynamicBatcher:
    def __init__(self, B_max: int, D_max: float):
        self.B_max = B_max
        self.D_max = D_max
        self.queue = []

    def on_request(self, req: Request, now_ms: float):
        assert now_ms >= req.arrival_ms
        self.queue.append(req)
        return self.maybe_dispatch(now_ms)

    def maybe_dispatch(self, now_ms: float):
        if not self.queue:
            return []

        oldest_arrival = self.queue[0].arrival_ms
        wait_ms = now_ms - oldest_arrival
        batch_size = min(self.B_max, len(self.queue))

        # 为什么超时也要发批：
        # 因为否则低流量时队列可能一直攒不满，最早请求会被长期阻塞。
        if batch_size == self.B_max or wait_ms >= self.D_max:
            batch = self.queue[:batch_size]
            self.queue = self.queue[batch_size:]
            return batch

        return []

# 玩具测试
scheduler = DynamicBatcher(B_max=4, D_max=5.0)
assert scheduler.on_request(Request("r1", 0.0), 0.0) == []
assert scheduler.on_request(Request("r2", 2.0), 2.0) == []
assert scheduler.on_request(Request("r3", 4.0), 4.0) == []
forced = scheduler.maybe_dispatch(5.0)
assert [r.req_id for r in forced] == ["r1", "r2", "r3"]

# 满批立即发
scheduler = DynamicBatcher(B_max=2, D_max=10.0)
assert scheduler.on_request(Request("a", 0.0), 0.0) == []
batch = scheduler.on_request(Request("b", 1.0), 1.0)
assert [r.req_id for r in batch] == ["a", "b"]
```

如果再往工程化走一步，配置层通常至少要有下面几个参数：

| 配置项 | 作用 | 常见理解 |
|---|---|---|
| `B_max` | 批大小硬上限 | 防止无限扩批 |
| `D_max` | 最长排队时间 | 保护 SLA |
| `preferred_batch_size` | 偏好批大小 | 某些引擎对特定批更快时使用 |

在 Triton 里，这些逻辑大多由服务端完成。典型配置思路如下：

```text
dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 5000
}
```

这里的 `max_queue_delay_microseconds` 就是在配置 `D_max`。`preferred_batch_size` 的意思不是“只发这几种大小”，而是“如果条件合适，优先凑成这些大小”。只有当某些 batch size 对 TensorRT profile 或 kernel 选择明显更友好时，这个参数才有必要精调；否则先把 `B_max` 和等待上限调顺更重要。

---

## 工程权衡与常见坑

动态批处理的核心矛盾始终没变：更大的批通常带来更高吞吐，但也会放大等待、显存压力和尾延迟。真实系统不是比谁吞吐峰值高，而是比谁在 SLA 下更稳。

先看指标对比：

| 配置变化 | 平均延迟 | `p95/p99` | 吞吐 | 风险 |
|---|---:|---:|---:|---|
| 增大 `B_max` | 可能升 | 常明显升 | 常升 | OOM、尾延迟拉长 |
| 增大 `D_max` | 升 | 明显升 | 可能升 | 长尾请求被拖慢 |
| 降低 `D_max` | 降 | 降 | 可能降 | GPU 利用率变差 |

一个真实工程例子是图像分类服务压测。假设你把最大批从 `4` 提到 `16`，在高流量时吞吐可能上涨 20% 到 60%，但 `p99` 延迟也可能翻倍，因为有一部分请求在持续等待更大的批形成。如果输入尺寸波动再大一点，某些 `batch=16` 还可能顶到显存边缘，导致 OOM 或频繁触发更保守的执行路径。

常见坑可以直接列出来：

| 坑点 | 表现 | 原因 | 规避方式 |
|---|---|---|---|
| 只追求满批 | 长尾请求很慢 | 低流量时一直攒不满 | 设置 `D_max` 强制放行 |
| 只看平均延迟 | 线上用户投诉但均值正常 | `p95/p99` 被掩盖 | 同时看均值和尾延迟 |
| 批太大 | OOM 或抖动 | 显存占用随 batch 增长 | 联合输入长度分布调参 |
| 队列无优先级 | 关键请求被普通请求挤压 | 所有请求同权 | 按业务做优先级或独立队列 |
| 乱设偏好批 | 吞吐没升反而更抖 | 调度被限制在少数尺寸 | 仅在 profile 明显受益时启用 |

压测时至少要统计三件事：吞吐、尾延迟、实际 batch 分布。下面这段伪代码表达的是“不要只盯一个数字”：

```python
def collect_metrics(latencies_ms, batch_sizes):
    latencies_ms = sorted(latencies_ms)
    n = len(latencies_ms)

    def pct(p):
        idx = min(n - 1, max(0, int(n * p) - 1))
        return latencies_ms[idx]

    metrics = {
        "avg_latency_ms": sum(latencies_ms) / n,
        "p95_latency_ms": pct(0.95),
        "p99_latency_ms": pct(0.99),
        "avg_batch_size": sum(batch_sizes) / len(batch_sizes),
        "max_batch_size": max(batch_sizes),
    }
    return metrics

m = collect_metrics([4, 5, 6, 7, 20], [2, 2, 4])
assert round(m["avg_latency_ms"], 2) == 8.4
assert m["p95_latency_ms"] == 7
assert m["max_batch_size"] == 4
```

这里的 `p95/p99` 计算只是玩具实现，但它表达了正确方向：调参要看完整面板，而不是只看平均值。因为对用户体验真正致命的，常常是尾部少数慢请求，而不是中位数。

再强调一次，尾延迟本质仍然来自这条式子：

$$
R_i = W_i + S(b_t)
$$

很多线上问题最后都能追到这里：要么等太久，要么批太大导致执行太慢，要么两者一起变坏。

---

## 替代方案与适用边界

动态批处理不是通用答案。它适合“一次前向、一次返回”的无状态请求，不适合所有推理系统。

最容易被误用的场景就是 LLM 聊天接口。聊天模型不是一次前向就结束，而是生成一个 token 后，还要带着 KV cache 继续生成下一个 token。这里的“KV cache”可以白话理解为“模型为了继续接着写，必须记住前面已经算过的中间状态”。因此每一步都可能有新请求加入、旧请求退出、不同请求长度分化，批需要不断重组。

这时更常见的不是普通动态批处理，而是 continuous batching 或 inflight batching。它们的重点不是“等一批凑齐再发”，而是“每一步都重新安排当前活跃请求”。

| 方案 | 是否在线组批 | 是否适合一次性输出 | 是否适合逐 token 生成 | 典型场景 |
|---|---|---|---|---|
| 单请求执行 | 否 | 是 | 是，但吞吐低 | 低并发、极低延迟 |
| 固定批处理 | 否 | 是 | 否 | 训练、离线推理 |
| 动态批处理 | 是 | 是 | 通常否 | 分类、embedding、ranking |
| 连续批处理 | 是，且逐步重组 | 不一定 | 是 | LLM 在线生成 |

前面那条吞吐近似：

$$
throughput \approx \frac{b_t}{S(b_t)}
$$

在 token 级迭代场景里就不再够用了，因为这里的“服务时间”不再对应一次完整请求，而对应很多轮迭代步，每一轮活跃请求集合都在变。也就是说，`b_t` 不再是一个静态批，而是不断变化的“当前步活跃批”。

可以用一段流程式伪代码看差异：

```python
active = []

while True:
    active.extend(new_arrivals())      # 新请求随时加入
    step_batch = pick_runnable(active) # 选出当前还能继续生成的请求
    run_one_token(step_batch)          # 每次只推进一个生成步
    active = [r for r in active if not r.finished()]
```

这和普通动态批处理的差别很大。后者通常是“攒一批，跑完，返回”；前者是“不断重组活跃请求，每步推进一点”。

如果你的业务请求量不大，而且对单请求首包延迟极敏感，那么动态批处理的收益可能很有限。比如某些内部检索接口，一秒只来十几个请求，但要求 `p99 < 20ms`，这时直接单条执行或固定小批，可能比复杂调度更稳定。

---

## 参考资料

正文里的 `throughput ≈ b_t / S(b_t)`、`R_i = W_i + S(b_t)` 属于工程分析框架，不应当误解为所有场景下都能直接套用的严格闭式结果。Triton 文档主要支撑配置方法；vLLM 文档主要支撑生成式推理为何更常使用连续批处理；排队分析论文主要支撑动态批处理的性能建模视角。

| 来源 | 用途 | 支撑章节 |
|---|---|---|
| Triton 文档 | 说明 `dynamic_batching`、`max_queue_delay_microseconds`、`preferred_batch_size` | 代码实现、工程权衡 |
| vLLM 文档 | 说明生成式推理中的连续批处理思想 | 替代方案与适用边界 |
| 排队分析论文 | 提供动态批处理性能分析框架 | 核心机制与推导 |

1. [NVIDIA Triton Inference Server User Guide: Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
2. [vLLM Documentation](https://docs.vllm.ai/en/v0.9.1/)
3. [vLLM Engine Arguments and Scheduler Configuration](https://docs.vllm.ai/en/v0.10.1/configuration/engine_args.html)
4. [Queueing analysis of GPU-based inference servers with dynamic batching: A closed-form characterization](https://doi.org/10.1016/j.peva.2020.102183)
