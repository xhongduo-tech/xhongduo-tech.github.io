## 核心结论

推理请求队列管理，是对进入模型服务的流量做统一约束，核心问题不是“先来先服务”，而是“系统接近饱和时，哪些请求应该先执行，哪些请求应该等待，哪些请求应该被拒绝”。

高峰期能否平稳退化，通常不取决于单个模型跑得多快，而取决于队列策略是否完整。完整的意思是：`入队规则`、`排序规则`、`等待上限`、`超时`、`拒绝`、`背压` 必须一起设计。背压，白话说，就是把“我已经接不住了”的信号明确返回给上游，而不是继续吞请求把自己拖死。

一个最小玩具例子可以说明问题。假设同一秒来了 100 个请求，其中 10 个是用户在线对话，90 个是离线 embedding。如果系统只用无界 FIFO，也就是无上限的先进先出队列，那么前面一批长请求会先占住执行资源，后来的短请求即使只需要几十毫秒，也只能跟着长队慢慢等。结果不是“所有请求都慢一点”，而是高价值请求也被一起淹没。

| 控制项 | 作用 | 失败后果 | 适用场景 |
| --- | --- | --- | --- |
| 优先级 | 先保护高价值请求 | 在线请求被低价值任务淹没 | 混合业务流量 |
| 队列上限 `Qmax` | 限制积压规模 | 内存上涨、延迟失控 | 所有在线服务 |
| 等待上限 `D` | 防止排队过久 | 过期结果仍在执行 | 有响应时限的请求 |
| 超时 | 让失败尽早暴露 | 上游误以为系统仍可用 | 用户交互场景 |
| 拒绝 | 在入口止损 | 队列拖垮整个服务 | 峰值洪峰 |
| 背压 | 约束上游重试 | 重试风暴放大故障 | 多层服务调用 |

结论可以压缩成一句话：请求队列管理决定系统在高峰期是“有控制地丢一部分请求”，还是“让全部请求一起变差”。

---

## 问题定义与边界

本文讨论的是推理服务入口侧的请求管理，不讨论模型训练、参数量选择、提示词设计，也不讨论“单次推理为什么本来就很慢”。入口侧，白话说，就是请求刚进入服务、还没真正占用 GPU 计算之前的那一段控制逻辑。

它解决三个问题：

1. 谁先执行。
2. 最多等多久。
3. 满了怎么办。

它不直接解决一个根因：如果模型单次推理本身就需要 30 秒，队列再漂亮，也不能把它变成 300 毫秒。但队列策略可以避免这种慢请求把整台服务拖进雪崩。

一个真实工程例子是：同一套 LLM 网关同时承接“在线对话”和“离线 embedding”。两者都在调用模型，但业务价值不同。在线对话要低延迟，通常几百毫秒到几秒内必须响应；离线 embedding 可以容忍更长等待，甚至可以失败后稍后重试。如果把它们放进同一条无差别 FIFO 队列，那么系统最忙时，最需要保护的在线请求反而最容易受伤。

| 维度 | 适用对象 | 不适用对象 | 典型目标 | 常见误区 |
| --- | --- | --- | --- | --- |
| 本文范围 | 在线推理入口、模型网关、批推理接入层 | 训练集群调度、检索召回、模型压缩 | 稳定性、尾延迟、容量保护 | 把队列问题误当成纯算力问题 |
| 管理对象 | 请求流量 | 模型内部算子 | 控制拥塞 | 只看平均延迟 |
| 关键动作 | 排队、超时、拒绝、回压 | 提升单次推理速度 | 峰值可退化 | 只配一个超时参数 |

后文统一使用下面的符号：

| 符号 | 含义 |
| --- | --- |
| $\lambda$ | 到达率，单位时间进入系统的请求数 |
| $\mu$ | 单个副本服务率，单位时间能处理的请求数 |
| $c$ | 并行副本数 |
| $\rho$ | 系统利用率，$\rho=\lambda/(c\mu)$ |
| $Q$ | 当前队列长度 |
| $W_q$ | 平均排队等待时间 |
| $D$ | 允许的总等待上限或超时阈值 |
| $b$ | batch size，即一次合并处理的请求数 |

---

## 核心机制与推导

先看稳定性。一个排队系统能否稳住，最基本的条件是：

$$
\rho = \frac{\lambda}{c\mu}
$$

这里的 $\rho$ 叫利用率，白话说，就是“流入速度”相对于“处理能力”的占比。

- 当 $\rho < 1$ 时，系统理论上可以稳定。
- 当 $\rho \ge 1$ 时，请求进入速度不小于处理速度，队列会持续增长。

这是第一层判断。第二层是等待时间。即使 $\rho < 1$，只要它接近 1，排队时间也可能迅速变坏。工程里更常用 Little 定律把“有多少请求在排队”和“要等多久”连起来：

$$
L_q = \lambda W_q
$$

其中 $L_q$ 可以近似理解为平均排队请求数。这个式子的价值不在于学术推导，而在于它直接告诉你：如果到达率上升，而你观察到等待时间也在涨，那么队列长度一定会更早变差。

再把理论落到配置上，常见的约束会写成：

$$
Q \le Q_{max}
$$

以及：

$$
W_q + S(b) \le D
$$

这里 $S(b)$ 表示 batch 后的服务时间。batch，白话说，就是把几个请求拼成一批一起算，用吞吐换一点等待时间。它有用，但必须加等待上限；否则系统会为了“凑更大的批次”把请求一直扣在队列里。

一个最小数值例子如下：

- 有 2 个推理副本，即 $c=2$
- 每个副本平均 200ms 处理 1 个请求，所以 $\mu=5\ \text{req/s}$
- 总处理能力是 $c\mu=10\ \text{req/s}$

如果峰值到达率是 $\lambda=12\ \text{req/s}$，那么：

$$
\rho = \frac{12}{10}=1.2
$$

这表示系统每秒净积压约 2 个请求。持续 5 秒，就多积压 10 个请求。若 `Qmax=15`，再过约 2.5 秒就会开始拒绝。这时 CPU 甚至可能还没满，因为瓶颈先体现在排队，而不是先体现在机器利用率。

一个工程上更接近真实的流程是：

`请求到达 -> 判断是否允许入队 -> 写入优先级队列 -> 等待调度 -> 可能参与 batching -> 开始执行 -> 完成 / 超时 / 被拒绝`

不同策略的差异如下：

| 策略 | 核心规则 | 优点 | 风险 |
| --- | --- | --- | --- |
| FCFS | 先来先服务 | 简单、公平 | 长请求拖住短请求 |
| 优先级队列 | 先看优先级，再看到达时间 | 保护高价值请求 | 低优先级可能饥饿 |
| 动态 batching | 等待短时间凑批 | 提高吞吐 | 等待过长会伤延迟 |
| 背压 | 满载时拒绝或降级 | 防止整体雪崩 | 需要上游配合处理 |

这里有一个关键认识：CPU、GPU 使用率不是最早暴露问题的指标。更早的信号往往是 `Q`、`W_q`、`P95 等待时间` 和 `拒绝率`。因为系统开始拥堵时，先变差的是“排队”，不是“计算”。

---

## 代码实现

实现上最重要的不是语法，而是把规则显式化。至少要有 `enqueue()`、`dequeue()`、`should_reject()`、`should_timeout()` 这几个独立步骤。否则策略散在各个分支里，后面很难验证。

下面是一个可运行的 Python 玩具实现。它不依赖线程和网络，只演示队列控制逻辑。

```python
from dataclasses import dataclass, field
import heapq
import time

@dataclass(order=True)
class Request:
    sort_key: tuple = field(init=False, repr=False)
    priority: int
    arrival_ts: float
    request_id: str
    timeout_s: float

    def __post_init__(self):
        # heapq 是小根堆，所以优先级数字越小，优先级越高
        self.sort_key = (self.priority, self.arrival_ts)

class InferenceQueue:
    def __init__(self, qmax: int):
        self.qmax = qmax
        self.heap = []

    def should_timeout(self, req: Request, now: float) -> bool:
        return now - req.arrival_ts > req.timeout_s

    def should_reject(self, req: Request) -> bool:
        if len(self.heap) < self.qmax:
            return False
        # 队列满时，低优先级请求直接拒绝
        # 只有当新请求优先级更高时，才尝试挤掉一个更差的请求
        worst = max(self.heap, key=lambda x: (x.priority, x.arrival_ts))
        if req.priority < worst.priority:
            self.heap.remove(worst)
            heapq.heapify(self.heap)
            return False
        return True

    def enqueue(self, req: Request) -> str:
        if self.should_reject(req):
            return "rejected"
        heapq.heappush(self.heap, req)
        return "queued"

    def dequeue(self, now: float):
        while self.heap:
            req = heapq.heappop(self.heap)
            if self.should_timeout(req, now):
                continue
            return req
        return None

# 测试优先级与超时逻辑
q = InferenceQueue(qmax=2)
base = time.time()

r1 = Request(priority=5, arrival_ts=base, request_id="low-1", timeout_s=10)
r2 = Request(priority=5, arrival_ts=base + 0.01, request_id="low-2", timeout_s=10)
r3 = Request(priority=1, arrival_ts=base + 0.02, request_id="high-1", timeout_s=10)

assert q.enqueue(r1) == "queued"
assert q.enqueue(r2) == "queued"
assert q.enqueue(r3) == "queued"   # 高优请求顶掉一个低优请求

first = q.dequeue(base + 0.03)
assert first.request_id == "high-1"

expired = Request(priority=1, arrival_ts=base - 20, request_id="expired", timeout_s=1)
q2 = InferenceQueue(qmax=2)
assert q2.enqueue(expired) == "queued"
assert q2.dequeue(base) is None
```

这个例子做了三件事：

1. 入队时先判断是否拒绝。
2. 排序时按 `(priority, arrival_time)` 决定先后。
3. 出队时再检查有没有等太久。

如果要落到真实服务，配置通常至少包含这些字段：

| 字段 | 作用 |
| --- | --- |
| `priority` | 业务优先级 |
| `Qmax` | 最大队列长度 |
| `D` | 最大等待或总超时 |
| `max_queue_delay` | batching 最多等多久 |
| `max_ongoing_requests` | 同时执行中的请求上限 |
| `batch_size` | 批大小 |

日志字段也要提前设计好，否则问题来了你看不清是“算慢了”还是“排爆了”：

| 字段 | 含义 |
| --- | --- |
| `request_id` | 请求唯一标识 |
| `priority` | 请求优先级 |
| `enqueue_ts` | 入队时间 |
| `start_ts` | 开始执行时间 |
| `wait_ms` | 排队耗时 |
| `decision` | `queued / rejected / timeout / completed` |

真实工程例子里，LLM 网关通常会把在线对话设为高优队列，embedding 设为低优队列；当 `max_ongoing_requests` 或 `Qmax` 到线时，优先拒绝低优请求，并返回 `429` 或 `503`，要求上游退避重试。

---

## 工程权衡与常见坑

队列管理的核心不是“尽量不拒绝”，而是“把拒绝发生在正确的位置”。正确的位置通常是入口，而不是让请求在系统里排队几十秒后再失败。越晚失败，浪费的资源越多，对上游的误导也越强。

常见坑如下：

| 问题表现 | 根因 | 修复方式 |
| --- | --- | --- |
| CPU 不高但延迟暴涨 | 请求卡在队列而非算力 | 监控 `Q`、`W_q`、`P95 wait` |
| 高优请求也很慢 | 无差别 FIFO | 改成优先级队列或分队列 |
| 队列越堆越长 | $\rho \ge 1$ 且无拒绝 | 设 `Qmax`、限流、扩容 |
| 超时后流量更大 | 上游立即重试 | 指数退避 + 随机抖动 |
| 吞吐高但交互差 | batching 等待太久 | 设置 `max_queue_delay` |
| 低优请求永远不执行 | 优先级饥饿 | 做老化或保底配额 |

“老化”这个术语第一次出现时可以这样理解：一个低优先级请求等得太久后，系统逐步提高它的调度优先级，避免它永远排不到。

诊断指标不要只盯资源利用率，至少要看：

| 指标 | 说明 |
| --- | --- |
| 队列长度 | 是否在持续积压 |
| 平均等待时间 | 常态排队压力 |
| P95 等待时间 | 尾部体验是否恶化 |
| 拒绝率 | 背压是否开始触发 |
| 超时率 | 用户是否在等待中过期 |
| 重试率 | 上游是否在放大故障 |

一个很典型的症状反推是：`CPU/GPU 不高，但接口延迟升高`。这往往意味着瓶颈先出现在队列。原因可能是 `max_ongoing_requests` 设得太小、调度过于保守、batch 等待过长，或者低优请求把入口堵住了。反过来，如果 `CPU/GPU 很高，队列也持续增长`，那通常说明容量本身确实不够，单靠队列策略已经止不住了。

---

## 替代方案与适用边界

队列管理不是唯一解，它更像流量控制的基础层。很多情况下要和其他手段组合使用。

| 方案 | 解决什么 | 代价 | 适用场景 | 不适用场景 |
| --- | --- | --- | --- | --- |
| 优先级队列 | 保护高价值请求 | 可能饥饿 | 混合业务 | 单一低价值任务流 |
| 限流 | 阻止入口过载 | 会直接丢请求 | 明确容量边界 | 需要尽量吸收突发 |
| 扩缩容 | 提升总容量 | 成本、扩容滞后 | 可横向扩展服务 | 冷启动慢或 GPU 紧张 |
| 缓存 | 减少重复推理 | 命中率不稳定 | 重复请求多 | 个性化请求强 |
| 异步化 | 把长任务移出同步链路 | 复杂度增加 | 离线任务 | 实时交互 |
| 请求分流 | 隔离不同业务 | 运维复杂 | 在线 + 离线混合 | 流量太小不值得拆 |

简单的选择规则可以写成：

- 低延迟优先：先做优先级队列、短超时、严格 `Qmax`
- 高吞吐优先：先做 batching，但必须限制 `max_queue_delay`
- 突发流量明显：先做限流 + 背压，再谈扩容
- 在线与离线混跑：优先拆队列或拆服务，不要共享默认 FIFO

再给一个业务到参数的直观映射：

| 业务类型 | 推荐策略 |
| --- | --- |
| 用户实时聊天 | 高优先级、短超时、小队列、低 batch 等待 |
| 批量 embedding | 低优先级、可长等待、较大 batch、严格并发上限 |
| 内部评测任务 | 独立队列或离线执行，避免挤占在线入口 |

适用边界也要说清楚。如果单次请求服务时间方差极大，或者请求体积差异非常大，仅靠单一优先级队列可能不够，此时需要进一步按请求类型拆分，甚至对超长请求单独建通道。否则“少量特别慢的请求”仍然会污染整体尾延迟。

---

## 参考资料

1. [NVIDIA Triton Inference Server model_config.proto](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_220/user-guide/docs/protobuf_api/model_config.proto.html)
2. [Ray Serve Autoscaling Guide](https://docs.ray.io/en/latest/serve/autoscaling-guide.html)
3. [Ray Serve Advanced Autoscaling](https://docs.ray.io/en/latest/serve/advanced-guides/advanced-autoscaling.html)
4. [vLLM Request Queue Documentation](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/request_queue/)
5. [AWS Builders’ Library: Avoiding insurmountable queue backlogs](https://aws.amazon.com/builders-library/avoiding-insurmountable-queue-backlogs/)
6. [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)
