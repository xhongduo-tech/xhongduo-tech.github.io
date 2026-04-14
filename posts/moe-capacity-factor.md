## 核心结论

Capacity Factor 是 MoE 中控制“每个专家在一个批次里最多接收多少 token”的参数。MoE，Mixture of Experts，白话说就是“很多子网络里只激活少数几个来处理输入”的结构。它的核心公式是：

$$
\text{capacity} = \text{capacity\_factor} \times \frac{\text{batch\_tokens}}{\text{num\_experts}}
$$

工程里通常还会向下取整：

$$
\text{expert\_capacity} = \left\lfloor \text{capacity\_factor} \times \frac{\text{batch\_tokens}}{\text{num\_experts}} \right\rfloor
$$

这个值不是“建议负载”，而是“硬上限”。router，路由器，白话说就是“决定 token 该送到哪个专家的模块”，先给 token 选专家，再检查该专家是否还有容量。容量满了以后，超出的 token 要么被丢弃，要么进入某种补救机制，取决于具体实现。

新手可以先记住一句话：`capacity_factor = 1.25`，意思不是“每个专家一定多处理 25%”，而是“每个专家被允许最多比平均值多接 25% 的 token”。想象每个专家前面有一个盘子，平均应该装 100 个 token，`1.25` 就是把盘子扩大到最多装 125 个；`0.8` 则是盘子缩小到 80 个。

实践上，`capacity_factor` 不是越大越安全。过大时，padding，填充，白话说就是“为了让张量对齐而补进去的空位”，会增加，通信和显存开销上升；过小时，token drop，也就是“本该进入专家计算但被截断丢掉的 token”，会增加，训练质量可能直接受损。多数系统会从 `1.0` 或 `1.05` 起步，再结合 drop 率和专家负载分布调节。

---

## 问题定义与边界

MoE 的路由不是平均发牌，而是按分数分配。某些 token 会大量偏向少数专家，于是“平均每个专家该处理多少 token”和“某个专家实际收到多少 token”通常并不相等。如果不设置上限，少数热点专家会过载；如果上限太紧，很多 token 会被截断。

这里的边界要说清楚：

1. `capacity_factor = 1.0` 表示每个专家最多处理“理论平均负载”。
2. `capacity_factor > 1.0` 表示允许热点专家吃下比平均更多的 token，以降低丢弃概率。
3. `capacity_factor < 1.0` 表示主动缩容，节省计算，但会更容易丢 token。
4. 它只约束“单批次内每个专家的最大容量”，不直接保证长期均衡。长期均衡更多依赖 load balancing loss，负载均衡损失，白话说就是“额外加一个训练目标，惩罚总是挤向少数专家的路由行为”。

一个最小玩具例子：

- 批次里共有 400 个 token
- 4 个专家
- 平均每个专家应处理 100 个 token

如果 `capacity_factor = 1.0`，每个专家上限是 100。某轮路由后，专家负载变成 `[150, 120, 80, 50]`，那前两个专家的超额部分就必须处理：要么丢弃，要么走额外机制。若把因子调到 `1.25`，每个专家上限变成 125，那么第一个专家仍会溢出 25 个，第二个专家则不再溢出。

下表可以直接看出基本权衡：

| factor | token drop 风险 | padding/通信开销 | 适合场景 |
|---|---:|---:|---|
| 0.8 | 高 | 低 | 极端控算力、可接受部分丢弃 |
| 1.0 | 中 | 中 | 默认起步值，先看监控 |
| 1.1 | 低到中 | 中到偏高 | 轻度路由不均衡 |
| 1.25 | 低 | 高 | 专家负载波动明显、优先保留 token |
| 1.5 | 很低 | 很高 | 通常只用于实验，不宜直接长期训练 |

因此，Capacity Factor 的问题定义不是“让专家尽量多吃 token”，而是在“丢 token”和“浪费算力”之间设定一个明确边界。不同框架的推荐范围通常集中在 `1.0` 到 `1.25`，原因正是这个区间往往能用有限额外开销换来明显更低的 drop。

---

## 核心机制与推导

MoE 中一个 token 进入层时，通常经历三个步骤：

1. 路由：router 计算每个专家的分数，选出 top-k 专家。
2. 容量检查：统计每个专家当前已接收的 token 数，与 `expert_capacity` 比较。
3. 截断或补救：超过上限的 token 被丢弃、重路由，或进入 overflow 机制。

核心推导很直接。设批次 token 总数为 $T$，专家数为 $E$，平均负载就是 $\frac{T}{E}$。为了容忍不均衡，再乘一个放大系数 $c$，得到：

$$
\text{expert\_capacity}=\left\lfloor c\cdot\frac{T}{E}\right\rfloor
$$

这里的 $\lfloor \cdot \rfloor$ 是向下取整，因为张量维度必须是整数。

题目给的典型例子可以直接算：

- `batch_size = 32`
- `seq_len = 64`
- 所以 `batch_tokens = 32 × 64 = 2048`
- `num_experts = 4`
- `capacity_factor = 1.25`

那么：

$$
\text{expert\_capacity}=\left\lfloor1.25\times\frac{2048}{4}\right\rfloor=\lfloor640\rfloor=640
$$

也就是每个专家本轮最多接收 640 个 token。如果某个专家收到 700 个，就有 60 个 token 超出上限。

这里容易误解的一点是：即使总容量之和看起来足够，也不代表不会 drop。因为限制是“按专家分别截断”，不是“全局统一池子”。例如 4 个专家都能各接 640 个，总上限是 2560，大于 2048，但如果路由高度集中成 `[900, 800, 200, 148]`，前两个专家仍会溢出。

真实工程例子可以看 DeepSpeed 风格的 MoE 层。训练时，每个 GPU 上会先得到本地 token 的专家分配，再按专家打包并做 all-to-all 通信，把属于同一专家的 token 送到对应设备。此时 `capacity_factor` 直接决定每个专家打包缓冲区的大小。值偏大，缓冲区里空位多，通信包更大；值偏小，缓冲区更紧，但热点专家更容易丢 token。它影响的不只是数学定义，还影响内存布局、通信张量形状和吞吐。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现。它不依赖任何框架，只模拟“按指定专家分配 token，再按 capacity 截断”的过程。

```python
from math import floor
from collections import defaultdict

def route_with_capacity(assignments, num_experts, batch_tokens, capacity_factor):
    """
    assignments[i] 表示第 i 个 token 被路由到哪个专家
    返回：
    - accepted: 每个专家实际接收的 token 下标
    - dropped: 被截断的 token 下标
    - capacity: 每个专家上限
    """
    capacity = floor(capacity_factor * (batch_tokens / num_experts))
    accepted = defaultdict(list)
    dropped = []

    for token_id, expert_id in enumerate(assignments):
        if len(accepted[expert_id]) < capacity:
            accepted[expert_id].append(token_id)
        else:
            dropped.append(token_id)

    return accepted, dropped, capacity

# 玩具例子：12 个 token，3 个专家，平均每个专家 4 个 token
assignments = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

accepted, dropped, capacity = route_with_capacity(
    assignments=assignments,
    num_experts=3,
    batch_tokens=12,
    capacity_factor=1.0,
)

assert capacity == 4
assert len(accepted[0]) == 4
assert dropped == [4]  # 第 5 个发往 expert 0 的 token 被丢弃

accepted2, dropped2, capacity2 = route_with_capacity(
    assignments=assignments,
    num_experts=3,
    batch_tokens=12,
    capacity_factor=1.25,
)

assert capacity2 == 5
assert len(accepted2[0]) == 5
assert dropped2 == []

print("ok")
```

这个例子说明两点：

1. `capacity_factor` 决定的是真实截断阈值。
2. 同一批 token、同一路由结果，仅仅调整 factor，就会改变 drop 行为。

如果写成训练侧伪代码，核心逻辑通常是下面这样：

```python
batch_tokens = batch_size * seq_len
expert_capacity = floor(capacity_factor * (batch_tokens / num_experts))

router_scores = router(hidden_states)          # [T, E]
topk_experts = select_topk(router_scores, k=1) # 或 k=2

expert_buckets = [[] for _ in range(num_experts)]
dropped_tokens = []

for token_id, expert_id in enumerate(topk_experts):
    if len(expert_buckets[expert_id]) < expert_capacity:
        expert_buckets[expert_id].append(token_id)
    else:
        dropped_tokens.append(token_id)

drop_rate = len(dropped_tokens) / batch_tokens
log_metric("moe/token_drop_rate", drop_rate)
log_metric("moe/expert_capacity", expert_capacity)
log_metric("moe/max_expert_load", max(len(b) for b in expert_buckets))
```

真正工程实现还会更复杂，因为常见情况包括：

- top-2 路由，一个 token 可能同时发往两个专家
- 不同设备上的专家需要 all-to-all 重排
- drop 后可能保留 residual 路径，残差路径，白话说就是“即使专家没算到，也让原始表示沿主干网络继续走”

但无论细节如何变化，`capacity_factor -> expert_capacity -> 截断与日志监控` 这条链路是稳定的。

---

## 工程权衡与常见坑

最常见的误区是把 `capacity_factor` 当成“越大越稳”的保险丝。它确实能降低 drop，但代价不会消失，只会转移到 padding、显存和通信上。

| 调整方向 | 直接影响 | 典型监控指标 | 风险 |
|---|---|---|---|
| 调高 factor | expert_capacity 变大 | padding ratio 上升 | 吞吐下降、显存增大 |
| 调低 factor | expert_capacity 变小 | token drop rate 上升 | 训练质量下降 |
| 固定不调 | 配置稳定 | 指标平稳但可能不最优 | 数据分布变化时失配 |
| 动态调节 | 跟随负载变化 | 需同时看 drop 与吞吐 | 实现复杂，调试难 |

新手版可以这样理解：

- 调到 `1.5`，很多专家的容量框被开得很大，但实际没装满，于是产生很多“空槽位”，这就是 padding。空槽位也要占内存、走通信，推理和训练都会变慢。
- 调到 `0.8`，容量框明显变小，热门专家经常一满就截断，drop 率上升，模型会有一部分 token 没有走到原计划的专家计算路径。

一个实用调参流程是：

1. 从 `1.0` 或 `1.05` 起步。
2. 记录 `token_drop_rate`、`expert load histogram`、`padding ratio`。
3. 若 drop 持续偏高，逐步加到 `1.1`、`1.2`。
4. 若 drop 很低但 padding 明显偏高，尝试回调。
5. 在 warmup 和 full training 分阶段观察，不要只看单个批次。

还要注意三个坑：

第一，别只看平均值。平均每个专家 500 个 token，不代表没有专家瞬时冲到 900。应看分布尾部，例如 P95 或最大负载。

第二，别忽略层间差异。不同 MoE 层的路由偏好不一样，同一套 factor 未必对所有层都合适。

第三，别把低 drop 误当成好事。如果为追求零 drop 把 factor 开得过大，训练可能仍然变差，因为吞吐下降后有效训练步数减少，或者通信成为瓶颈。

---

## 替代方案与适用边界

固定 `capacity_factor` 是最常见方案，因为实现简单、张量形状稳定、易于并行。但它不是唯一选择。

| 方案 | 适用场景 | 复杂度 | 主要风险 |
|---|---|---:|---|
| Static capacity_factor | 批大小稳定、专家数固定 | 低 | 对突发不均衡不够灵活 |
| Dynamic cap | 批次波动大、负载分布变化明显 | 中 | 运行时形状变化，调试更难 |
| Overflow queue | 重要 token 不能轻易丢弃 | 高 | 实现复杂，延迟上升 |
| 更强负载均衡损失 | 路由长期偏斜明显 | 中 | 可能影响专家 specialization，专门化能力 |

Dynamic cap，动态容量，白话说就是“不是提前写死 1.1 或 1.25，而是根据当前批次负载情况临时决定容量”。它适合 batch size 波动较大、序列长度变化明显的系统，但代价是实现复杂，尤其在分布式场景下更难保持通信和张量布局稳定。

Overflow queue，溢出队列，白话说就是“专家满了先别丢，把超出的 token 暂存，再用备用路径处理”。这能减少重要 token 被直接舍弃的概率，但会增加调度逻辑和延迟，适合对质量更敏感、对系统复杂度容忍更高的场景。

因此，边界可以概括为：

- 批次稳定、实现优先简单：用 static factor。
- 轻微不均衡、追求吞吐：可以接受略激进的较低 factor。
- 不均衡严重、token 价值差异大：考虑 dynamic cap 或 overflow 机制。
- 如果问题根源是 router 总偏向少数专家，先检查负载均衡损失和路由训练，再调 factor；不要把它当成唯一补丁。

---

## 参考资料

- [EngineersOfAI: The Capacity Factor — Controlling Token Dropping](https://engineersofai.com/docs/llms/mixture-of-experts/training-moe-models?utm_source=openai)
- [AI Wiki: Mixture of Experts](https://aiwiki.ai/wiki/mixture_of_experts?utm_source=openai)
- [DeepSpeed Documentation: Mixture of Experts](https://deepspeed.readthedocs.io/en/stable/moe.html?utm_source=openai)

查最新默认值或推荐值时，优先看框架官方文档，因为不同版本可能修改默认参数、drop 策略或路由实现；直观解释可先读 EngineersOfAI 和 AI Wiki，再回到 DeepSpeed 文档确认具体接口和版本行为。
