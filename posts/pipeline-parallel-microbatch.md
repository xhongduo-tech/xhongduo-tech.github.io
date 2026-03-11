## 核心结论

流水线并行（Pipeline Parallelism）解决的不是“单卡算不动”这一类局部算力问题，而是“模型按深度堆得太长，参数和中间激活无法由单个设备独立承载”这个结构性问题。

它的核心收益不来自“把同一个 batch 同时丢给很多卡”，而来自微批次（micro-batch）调度。做法是先把一个训练批次切成 $m$ 个微批次，再把模型按层切成 $p$ 个阶段。这样，不同设备可以在同一时刻处理不同微批次的不同阶段，形成接力式并行。

在朴素流水线中，最关键的损耗叫 bubble。它表示流水线没有被完全填满或已经开始排空时，某些设备有算力却无事可做。对 GPipe 这种同步调度，bubble 比例常写成：

$$
\text{bubble ratio} \approx \frac{p-1}{m+p-1}
$$

这个公式说明两件事：

1. 阶段数 $p$ 越多，流水线越长，填充和排空的固定等待越明显。
2. 微批次数 $m$ 越多，同样的固定等待会被更多有效计算摊薄。

对初学者，最重要的直觉是：流水线并行加速的是“整体吞吐”，不是“单个样本延迟”。一个样本从第一个阶段走到最后一个阶段，路径没有缩短；真正变快的是单位时间内被处理完的微批次数。

下面先看一个最小表格。假设阶段数 $p=4$：

| $m$（微批次数） | 泡泡率 $\frac{p-1}{m+p-1}$ | 理想利用率 $1-\text{泡泡率}$ | 含义 |
|---|---:|---:|---|
| 4 | $3/7 \approx 42.9\%$ | $\approx 57.1\%$ | 流水线刚有重叠，但空闲仍很多 |
| 8 | $3/11 \approx 27.3\%$ | $\approx 72.7\%$ | 重叠变明显，吞吐改善较大 |
| 16 | $3/19 \approx 15.8\%$ | $\approx 84.2\%$ | 固定开销被摊薄，但继续增大收益开始递减 |
| 32 | $3/35 \approx 8.6\%$ | $\approx 91.4\%$ | 继续提升，但不再是成倍改善 |

这也是工程上常见经验的来源：$m$ 增大通常能降低 bubble，但收益不是线性增长，而是边际递减。原因很简单，bubble 是固定填充成本，不会随着微批次数一起变大。

还可以把这个结论写成更适合做估算的形式：

$$
\text{utilization} \approx \frac{m}{m+p-1}
$$

当 $m \gg p$ 时，利用率逼近 1；当 $m$ 只比 $p$ 略大时，bubble 仍然不可忽略。

---

## 问题定义与边界

问题定义可以压缩成一句话：给定一个按深度切成 $p$ 段的模型，如何调度 $m$ 个微批次的前向与反向，使设备空闲尽量少、显存仍可承受、梯度计算保持正确。

这里先把边界说清，否则后面的公式和调度表都容易被误读。

第一，流水线并行切的是模型深度，不是张量内部维度。  
例如一个 48 层 Transformer，可以切成 4 个阶段：

| 阶段 | 持有层 |
|---|---|
| $S_1$ | 第 1 到 12 层 |
| $S_2$ | 第 13 到 24 层 |
| $S_3$ | 第 25 到 36 层 |
| $S_4$ | 第 37 到 48 层 |

输入先经过 $S_1$，再传到 $S_2$，最后到 $S_4$。这是一条按深度串起来的计算链。

第二，前向和反向的依赖方向不能被随意打乱。  
前向必须满足：

$$
S_1 \rightarrow S_2 \rightarrow \cdots \rightarrow S_p
$$

反向必须满足：

$$
S_p \rightarrow S_{p-1} \rightarrow \cdots \rightarrow S_1
$$

这不是框架习惯，而是链式求导的数学要求。后一阶段还没完成前向，前一阶段就不可能拿到对应的反向梯度。

第三，真正限制吞吐的通常不只 bubble，还有阶段失衡（stage imbalance）。  
如果某一个阶段明显更慢，整条流水线的节拍就会被它决定。设第 $i$ 个阶段单个微批次前向时间为 $t_i^{(f)}$、反向时间为 $t_i^{(b)}$，那么稳态下的节拍近似由最慢阶段决定：

$$
T_{\text{cycle}} \approx \max_i \left(t_i^{(f)} + t_i^{(b)} + t_i^{(\text{comm})}\right)
$$

这里的 $t_i^{(\text{comm})}$ 表示该阶段与相邻阶段之间的通信时间。公式的含义很直接：哪一段最慢，哪一段就定义整条流水线的出件速度。

先看一个两阶段玩具例子。假设只有两张卡 A、B，模型切成两段，微批次数 $m=4$，只看前向：

| 时间步 | A 上的计算 | B 上的计算 |
|---|---|---|
| 1 | F1 | - |
| 2 | F2 | F1 |
| 3 | F3 | F2 |
| 4 | F4 | F3 |
| 5 | - | F4 |

这里 `F1` 表示“第 1 个微批次的前向”。  
时间步 1 中，B 必然空闲，因为 A 还没把任何激活传过来。  
时间步 5 中，A 已经完成全部前向，而 B 还在处理最后一个微批次。  
这两个端点上的空闲，就是 bubble 的最简单形态。

再看一个三阶段例子，$p=3,m=3$，采用 GPipe 的“先全前向，再全反向”：

| 时间步 | GPU1 | GPU2 | GPU3 | 说明 |
|---|---|---|---|---|
| 1 | F1 | - | - | 第 1 个微批次进入流水线 |
| 2 | F2 | F1 | - | 第 2 个微批次进入，GPU2 接到 F1 |
| 3 | F3 | F2 | F1 | 前向流水线第一次被填满 |
| 4 | - | F3 | F2 | 头部开始空闲，尾部继续前向 |
| 5 | - | - | F3 | 全部前向结束 |
| 6 | - | - | B3 | 开始统一反向 |
| 7 | - | B3 | B2 | 反向梯度向前传播 |
| 8 | B3 | B2 | B1 | 反向流水线第一次被填满 |
| 9 | B2 | B1 | - | 尾部开始空闲 |
| 10 | B1 | - | - | 反向结束 |

这个表暴露了 GPipe 的边界：逻辑最清楚，梯度语义也最稳定，但在“前向全做完”和“反向刚开始”之间，尾部阶段会出现明显等待。

对新手，一个常见误解是“设备都在同一时刻算，就一定快”。并不是。关键不在“是否同时算”，而在“同时算的是不是不同微批次、是不是保持持续出件”。流水线优化的是连续流，不是瞬时并发数。

---

## 核心机制与推导

先定义记号：

- $p$：流水线阶段数
- $m$：微批次数
- 假设每个阶段处理一个微批次的前向耗时相同，记为 1 个时间单位
- 先只分析前向，反向结构完全对称

在这个理想化模型下，GPipe 前向需要的总时间步为：

$$
T_{\text{forward}} = m + p - 1
$$

这个式子可以拆成两部分理解。

第一部分是“填满流水线”的 $p-1$ 步。  
第 1 个微批次必须先经过前面所有阶段，最后一个阶段在第 $p$ 个时间步才第一次拿到工作。

第二部分是“真正处理 $m$ 个微批次”的 $m$ 步。  
当流水线进入稳态后，每增加一个微批次，总长度只会多增加 1 步。

因此前向的总长度就是：

$$
\underbrace{p-1}_{\text{填充成本}} + \underbrace{m}_{\text{有效工作}}
$$

如果只看前向，总共存在 $m+p-1$ 个时间步，但只有其中的 $m$ 步对应“满负载出件节奏”。于是可把 bubble 比例写为：

$$
\text{bubble ratio} \approx \frac{p-1}{m+p-1}
$$

把它换个形式，就是更常见的利用率表达式：

$$
\text{utilization} \approx \frac{m}{m+p-1}
$$

再进一步：

$$
\text{waste} = 1 - \text{utilization} = \frac{p-1}{m+p-1}
$$

这组公式有三个直接的工程含义。

第一，$m$ 增大时，bubble 下降，但按倒数下降，不是线性清零。  
所以“把微批次数翻倍”不等于“空闲时间减半之后还能继续同样减半”。

第二，$p$ 增大时，填充成本变大。  
模型切得越碎，不一定越快。切分增加了可并行阶段数，也增加了流水线长度和通信次数。如果微批次数不够，更多阶段只会带来更多等待。

第三，上式描述的是理想调度上界，不是系统实测吞吐。  
只要阶段耗时不相等、通信不可忽略、内核启动有额外成本，实测结果就会偏离公式。

下面用具体数字算一次。若 $p=4,m=16$：

$$
\text{bubble ratio} = \frac{4-1}{16+4-1} = \frac{3}{19} \approx 15.8\%
$$

若把微批次数翻倍到 $m=32$：

$$
\text{bubble ratio} = \frac{3}{35} \approx 8.6\%
$$

bubble 的确下降了，但没有下降到原来的一半以下很多。原因是固定项 $p-1$ 没变，只是被更多微批次摊薄。

还可以反过来估算：如果希望 bubble 小于某个阈值 $\epsilon$，需要多少个微批次？

$$
\frac{p-1}{m+p-1} < \epsilon
$$

移项得到：

$$
p-1 < \epsilon(m+p-1)
$$

$$
(1-\epsilon)(p-1) < \epsilon m
$$

$$
m > \frac{(1-\epsilon)(p-1)}{\epsilon}
$$

例如希望 bubble 低于 6%，即 $\epsilon = 0.06$：

$$
m > \frac{0.94(p-1)}{0.06} \approx 15.67(p-1)
$$

这意味着若 $p=4$，则需要：

$$
m > 15.67 \times 3 \approx 47.0
$$

也就是至少接近 48 个微批次，bubble 才会低于 6%。  
这也解释了为什么“$m \ge 4p$ 就已经很好”只是一条很粗的经验，不代表 bubble 已经很小。对 $p=4$ 而言，$m=16$ 仍然有约 15.8% 的 bubble。

下面给一个更适合做规划的表格：

| 阶段数 $p$ | 目标 bubble | 微批次数下界 $m > \frac{(1-\epsilon)(p-1)}{\epsilon}$ | 近似结论 |
|---|---:|---:|---|
| 4 | 20% | $m > 12$ | 十几个微批次已能明显改善 |
| 4 | 10% | $m > 27$ | 需要接近 30 个微批次 |
| 4 | 6% | $m > 47$ | 需要非常多微批次 |
| 8 | 10% | $m > 63$ | 阶段越多，对 $m$ 要求越高 |

真实工程里，这个估算还必须结合显存。因为 $m$ 提高后，更多微批次会同时在流水线上“在途”，对应更多激活需要缓存。也就是说，降低 bubble 和控制显存是直接冲突的一组目标。

---

## 代码实现

下面先给一个可直接运行的 Python 程序。它做三件事：

1. 生成 GPipe 前向时间线
2. 计算理论 bubble 和利用率
3. 打印表格并验证一个具体例子

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PipelineStats:
    stages: int
    microbatches: int
    total_steps: int
    bubble_ratio: float
    utilization: float


def gpipe_forward_timeline(p: int, m: int) -> List[List[str]]:
    """
    Return a timeline for GPipe forward-only scheduling.

    timeline[t][s] is the work item on stage s at time step t.
    A cell is either 'F{k}' or '-'.
    """
    if p < 1 or m < 1:
        raise ValueError("p and m must both be >= 1")

    total_steps = m + p - 1
    timeline: List[List[str]] = []

    for t in range(total_steps):
        row: List[str] = []
        for stage in range(p):
            microbatch = t - stage
            if 0 <= microbatch < m:
                row.append(f"F{microbatch + 1}")
            else:
                row.append("-")
        timeline.append(row)

    return timeline


def bubble_ratio_formula(p: int, m: int) -> float:
    if p < 1 or m < 1:
        raise ValueError("p and m must both be >= 1")
    return (p - 1) / (m + p - 1)


def utilization_formula(p: int, m: int) -> float:
    if p < 1 or m < 1:
        raise ValueError("p and m must both be >= 1")
    return m / (m + p - 1)


def summarize_pipeline(p: int, m: int) -> PipelineStats:
    return PipelineStats(
        stages=p,
        microbatches=m,
        total_steps=m + p - 1,
        bubble_ratio=bubble_ratio_formula(p, m),
        utilization=utilization_formula(p, m),
    )


def format_timeline(timeline: List[List[str]]) -> str:
    if not timeline:
        return ""

    p = len(timeline[0])
    headers = ["step"] + [f"S{i + 1}" for i in range(p)]
    widths = [max(len(h), 4) for h in headers]

    for t, row in enumerate(timeline, start=1):
        widths[0] = max(widths[0], len(str(t)))
        for i, cell in enumerate(row, start=1):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt_row(headers)]
    lines.append("-+-".join("-" * w for w in widths))

    for t, row in enumerate(timeline, start=1):
        lines.append(fmt_row([str(t)] + row))

    return "\n".join(lines)


def main() -> None:
    # Example 1: verify the 2-stage, 4-microbatch timeline
    timeline = gpipe_forward_timeline(2, 4)
    expected = [
        ["F1", "-"],
        ["F2", "F1"],
        ["F3", "F2"],
        ["F4", "F3"],
        ["-", "F4"],
    ]
    assert timeline == expected

    # Example 2: verify formulas
    assert abs(bubble_ratio_formula(4, 16) - 3 / 19) < 1e-12
    assert abs(utilization_formula(4, 16) - 16 / 19) < 1e-12
    assert bubble_ratio_formula(4, 32) < bubble_ratio_formula(4, 16)

    stats = summarize_pipeline(4, 16)
    print("Pipeline stats:")
    print(stats)
    print()

    print("Forward timeline for p=4, m=6:")
    print(format_timeline(gpipe_forward_timeline(4, 6)))


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出的是一个离散时间视角下的“谁在什么时间步做什么工作”的表。它没有真的训练模型，但足以验证公式和调度直觉。

如果运行 `gpipe_forward_timeline(4, 6)`，得到的前向时间线大致如下：

```text
step | S1 | S2 | S3 | S4
-----+----+----+----+---
1    | F1 | -  | -  | -
2    | F2 | F1 | -  | -
3    | F3 | F2 | F1 | -
4    | F4 | F3 | F2 | F1
5    | F5 | F4 | F3 | F2
6    | F6 | F5 | F4 | F3
7    | -  | F6 | F5 | F4
8    | -  | -  | F6 | F5
9    | -  | -  | -  | F6
```

从这个表可以直接看出三段现象：

1. 前 3 步在填充流水线，右侧阶段逐步开始工作。
2. 中间几步出现稳态，多个阶段同时处理不同微批次。
3. 最后 3 步在排空流水线，左侧阶段逐步空闲。

这就是公式里那个 $p-1$ 的来源。它既对应填充，也对应排空，只是在比例推导里通常合并为同一类固定开销来理解。

如果把它写成更接近真实训练框架的伪代码，GPipe 的基本结构如下：

```python
# forward: 先把所有微批次送入流水线
for microbatch in range(m):
    x = load_microbatch(microbatch)
    for stage in range(p):
        x = stages[stage].forward(x)
        save_activation(stage, microbatch, x)
        if stage < p - 1:
            x = send_to_next_stage(x)

# backward: 等全部前向结束，再统一反向
for microbatch in reversed(range(m)):
    grad = load_loss_grad(microbatch)
    for stage in reversed(range(p)):
        act = load_activation(stage, microbatch)
        grad = stages[stage].backward(act, grad)
        if stage > 0:
            grad = send_to_prev_stage(grad)
```

这段伪代码里有四个实现点必须理解。

第一，必须缓存激活。  
反向传播不是只靠梯度就能算，它还需要前向阶段留下的中间结果。缓存的对象通常包括 layer input、某些归一化统计量、attention 中间值等。微批次越多，激活缓存压力越大。

第二，阶段之间的通信是主路径，不是旁路。  
每个微批次在阶段边界都要发送激活，反向时还要发送梯度。如果通信时间接近甚至超过单阶段计算时间，流水线吞吐会被直接拉低。

第三，调度和数值语义不能分开看。  
GPipe 的优点是所有微批次先完成前向，再统一反向，梯度累积后再更新参数，这与常规同步训练语义一致，比较容易保证收敛行为不变。

第四，真实工程一般不会停留在最朴素的 GPipe。  
很多系统会进一步实现 1F1B 或其变种，让前向和反向在稳态中交错执行，以减少尾部等待和激活驻留时间。

如果要把“代码为什么能对应数学公式”再说得更直白一点，可以记住一句话：  
在第 $t$ 个时间步，第 `stage` 个阶段处理的微批次编号近似是 `t - stage`。  
编号合法就说明该阶段有活干；编号越界就说明该阶段此时处在 bubble 中。

---

## 工程权衡与常见坑

流水线并行最常见的误区，是把“增大微批次数”当成免费优化。实际工程里，它几乎从来不是免费的。

先看一个总览表：

| 风险 | 直接成因 | 表现形式 | 常见缓解方法 |
|---|---|---|---|
| 显存爆表 | 在途微批次增多，激活缓存增加 | OOM、频繁重算、吞吐抖动 | 减小单个微批次大小、启用 activation checkpointing |
| 阶段失衡 | 切分按层数均分而非按耗时均分 | 某一阶段长期最忙，其余阶段等待 | 基于 profile 重切 stage |
| 通信放大 | 每个微批次都跨阶段传激活和梯度 | 算得不慢，但链路拥堵 | 减少跨机切分、提高通信粒度 |
| 尾部等待 | GPipe 先全前向再全反向 | 前向结束后尾部设备短时空闲明显 | 改用 1F1B 或交错调度 |
| 数值不稳定 | 单个微批次太小 | 统计波动大，loss 曲线更抖 | 保持全局 batch 不变，配合梯度累积 |
| 调度成本上升 | 微批次过多，队列和 buffer 更多 | 框架开销增加，理论收益无法兑现 | 只增加到“足够填满”为止 |

### 1. 显存不是只看参数量

很多初学者会把流水线并行理解成“每张卡只放一部分层，所以显存压力下降”。这只说对了一半。

参数显存确实下降了，因为每张卡只持有部分模型参数。  
但激活显存不一定下降，甚至可能上升。原因在于多个微批次会同时在流水线上不同位置“在途”，每个阶段都可能需要为多个尚未完成反向的微批次缓存激活。

一个粗略估算是：

$$
\text{activation memory} \propto \text{in-flight microbatches} \times \text{activation size per microbatch}
$$

在 GPipe 中，由于前向全做完才开始反向，激活驻留时间相对更长，显存压力通常比 1F1B 更高。

### 2. 阶段均衡比理论 bubble 更重要

公式假设每个阶段耗时相同，但真实模型往往不是这样。

例如在 Transformer 中，某些层可能包含更重的 attention、MoE 路由、长序列 KV 操作，或者某一阶段跨机器而不是同机 NVLink，相同层数并不代表相同耗时。

假设 4 个阶段单微批次总耗时如下：

| 阶段 | 计算+通信耗时 |
|---|---:|
| $S_1$ | 10 ms |
| $S_2$ | 11 ms |
| $S_3$ | 18 ms |
| $S_4$ | 10 ms |

这时理论上每步都能重叠，但流水线节拍其实被 $S_3$ 的 18 ms 锁死。其他阶段即使只需 10 ms，也只能等最慢阶段推进。所以真正要优化的不是“每段多少层”，而是“每段总耗时是否接近”。

### 3. 通信可能吃掉重叠收益

流水线并行的阶段边界上需要传两类东西：

1. 前向时传激活
2. 反向时传梯度

如果模型切得过碎，阶段边界变多；如果微批次太小，每次发送的数据包也更小，容易让链路利用率变差、启动开销占比升高。于是就会出现一种常见情况：理论上 bubble 降了，实测吞吐却没涨多少，因为收益被通信放大抵消了。

可以把单步时间粗略写成：

$$
T_{\text{step}} \approx \max_i(T_i^{\text{comp}} + T_i^{\text{comm}})
$$

只要 $T_i^{\text{comm}}$ 不小，优化就不能只盯着算力。

### 4. 微批次不是越小越好

设全局 batch 固定为 $B$，微批次数为 $m$，则单个微批次大小为：

$$
b_{\mu} = \frac{B}{m}
$$

当 $m$ 很大时，$b_{\mu}$ 会很小。  
这会带来两个问题：

1. 某些算子效率下降，GPU 不能被充分填满。
2. 统计性质变差，例如 batch norm 类操作或 loss 波动更明显。

所以工程上不是单纯追求更大的 $m$，而是在“bubble 可接受”“显存可承受”“单微批次算子仍有效率”之间取平衡。

### 5. 调度复杂度会快速上升

从论文图示看，流水线好像只是几条箭头。但到了工程实现，状态会显著增多：

- 每个阶段要跟踪哪些微批次已完成前向
- 哪些激活可释放、哪些还不能释放
- 哪些梯度已回传、哪些通信请求在飞
- 梯度累积何时归并
- 混合精度缩放、检查点重算何时插入

因此，对简单模型或浅层模型，调度和通信的额外开销可能超过理论收益。流水线并行最适合的是“模型足够深，切分后每段仍有足够计算量”的场景，而不是所有多卡训练场景。

如果把这些权衡压缩成一句工程判断，就是：  
先保证阶段均衡，再谈微批次数；先确认显存和通信能承受，再追求低 bubble。

---

## 替代方案与适用边界

GPipe 不是唯一方案，它只是最容易分析、也最容易保证同步训练语义的一种方案。

先看一个对比表：

| 策略 | 核心思路 | 优点 | 关注点 | 适用场景 |
|---|---|---|---|---|
| GPipe | 先全前向，再全反向 | 调度简单，梯度语义清晰 | 尾部等待明显，激活缓存较多 | 先追求正确性与实现稳定性 |
| 1F1B | 稳态中交替执行一个前向和一个反向 | 利用率更高，激活释放更早 | 调度状态更复杂 | 中大型模型训练主流方案 |
| PipeDream | 进一步引入异步和参数版本流水 | 吞吐更激进 | 梯度 staleness，训练语义更复杂 | 对吞吐极敏感、可接受参数陈旧 |
| 纯数据并行 | 每卡一份完整模型，分数据 | 实现最直接 | 模型必须单卡放得下 | 模型不深、样本量大 |
| 张量并行 | 单层内部张量切分 | 解决单层矩阵过大问题 | 通信强依赖单层算子结构 | 单层已大到单卡难容纳 |

### GPipe

GPipe 的优点是概念和语义都最直观：

- 所有微批次共享同一版参数做前向
- 所有反向完成后再统一做参数更新
- 梯度累积逻辑与常规大 batch 训练一致

因此它很适合作为学习流水线并行的起点，也适合优先要求数值行为清晰可控的系统。

### 1F1B

1F1B（One Forward, One Backward）是在流水线进入稳态后，尽量让某些阶段做前向的同时，另一些阶段做更早微批次的反向。它的两个直接收益是：

1. 尾部设备不必等所有前向结束才开始反向
2. 某些微批次的激活可以更早释放，显存压力更低

但它的实现明显更复杂。调度器必须同时维护两条依赖链：

- 当前哪些微批次在做前向
- 当前哪些微批次可以开始反向

换句话说，GPipe 更像“两段式批处理”，而 1F1B 更像“稳态中的连续流系统”。

### PipeDream

PipeDream 继续往前走，不再严格要求所有设备始终使用同一版参数。这样吞吐还能提高，但代价是梯度 staleness，即某些反向梯度对应的前向参数版本与当前更新版本不同。

这会改变训练语义。它不是简单的“更快版 GPipe”，而是“用更复杂的一致性换取吞吐”。因此它适用于对系统吞吐更敏感、对严格同步收敛要求没那么强的场景。

### 什么时候不该优先用流水线并行

下面这几个场景通常不应把流水线并行放在第一优先级：

1. 模型不够深。  
切完以后每个阶段只剩很少计算，通信反而占主导。

2. 单层太大而不是模型太深。  
如果瓶颈是单个矩阵乘法已经放不下，优先考虑张量并行，而不是按深度切层。

3. 模型完整放得进单卡，但样本很多。  
这种情况数据并行通常更直接，系统复杂度也更低。

4. 设备互联较差。  
如果跨阶段传激活和梯度本身就很慢，流水线重叠很难弥补通信损失。

可以把选择逻辑压缩成一句判断：

- 单层太大，优先看张量并行
- 模型很深，优先看流水线并行
- 模型放得下、只是数据多，优先看数据并行

所以，流水线并行的适用边界不是“卡多就上”，而是“模型足够深、阶段切分合理、通信可承受、显存还能容纳在途微批次”。这些条件同时成立时，微批次调度才真正有价值。

---

## 参考资料

1. Huang, Yanping, et al. “GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.” NeurIPS 2019.
2. Narayanan, Deepak, et al. “Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.” SC 2021.
3. Narayanan, Deepak, et al. “PipeDream: Generalized Pipeline Parallelism for DNN Training.” SOSP 2019.
4. Sebastian B oe hm, “Pipeline Parallelism: Distributed Training via Model Partitioning.”
5. System Overflow, “Pipeline Parallelism: Scaling Model Depth Across Devices.”
6. CSE 234 / Data Systems for Machine Learning 相关课程讲义中关于 pipeline scheduling、1F1B 与 bubble analysis 的部分。
