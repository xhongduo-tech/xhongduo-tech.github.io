## 核心结论

流水线并行的核心是把一个模型按层切成多个连续的 `stage`，每个 `stage` 放到不同设备上，让数据像装配线一样依次流过这些设备。这里的 `stage` 可以先理解为“模型的一段连续层”。

它解决的不是“把同一份参数复制到多张卡”，而是“让不同卡同时负责不同层”。因此，它主要优化的是**吞吐**，也就是单位时间能处理多少训练样本，而不是单条样本的最短延迟。

如果只把模型切开，但整个 mini-batch 仍然整批推进，那么前面的设备算完以后，后面的设备才开始工作，大量时间会空转。这种空转叫**气泡**，可以先理解为“流水线里没有任务经过的空白时间”。

GPipe 的关键改进是把一个 mini-batch 再切成 $m$ 个 micro-batch。`micro-batch` 就是“更小的一份训练数据切片”。这样第一个 micro-batch 刚流到下一个 stage，前一个 stage 就能立刻处理第二个 micro-batch，流水线开始滑动。常见近似效率公式是：

$$
\text{Efficiency} \approx 1 - \frac{p-1}{p-1+m}
= \frac{m}{m+p-1}
$$

其中 $p$ 是 stage 数，$m$ 是 micro-batch 数。于是气泡率近似为：

$$
\text{Bubble Rate} \approx \frac{p-1}{p-1+m}
$$

当 $m \gg p$ 时，气泡快速逼近 0，设备利用率接近满载。

1F1B 调度是另一个关键点。`1F1B` 可以先理解为“做一次前向，就尽快穿插一次反向”。它不像纯 GPipe 那样先堆很多前向再统一做反向，而是前后向交替，目的是更早释放激活，降低显存峰值。再配合激活检查点，流水线并行才真正适合大模型训练。

一个直观例子是 4 个 stage 的 Transformer。若把一个 mini-batch 切成 12 个 micro-batch，数据会持续在 4 张 GPU 上滚动推进，前向和反向不断交错，4 张卡大部分时间都在做有用计算，而不是等待。

---

## 问题定义与边界

问题定义很具体：在**同步训练**场景中，当单卡放不下完整模型，或者单卡虽然能放下但吞吐不足时，如何让多张 GPU 分工处理不同层，并尽量减少等待。

这里有三个概念必须分清：

| 概念 | 白话解释 | 解决什么问题 |
|---|---|---|
| 层划分 | 把模型按顺序切成几段，分别放到不同设备 | 单卡放不下整模 |
| 微批次滑动 | 把一个 batch 切成多个小块连续送入 | 减少流水线空转 |
| 气泡 | 某些 stage 没活干的时间 | 衡量利用率损失 |

边界也很明确。

第一，只讨论**顺序层结构可切分**的模型，例如 Transformer、深层 MLP、部分 CNN。若层之间存在大量跨 stage 的随机访问或复杂回边，流水线切分会非常难。

第二，只讨论**训练**，而且是前向和反向都参与的情形。推理也可做流水线，但优化目标不同。训练关注吞吐和显存，推理常关注单请求时延。

第三，流水线并行不是免费的。你必须同时考虑：

| 参数 | 含义 | 太小的问题 | 太大的问题 |
|---|---|---|---|
| $p$ | stage 数 | 并行度不够 | 通信边界变多，切分更难 |
| $m$ | micro-batch 数 | 气泡多，设备空闲 | 激活缓存增大，调度更复杂 |
| 激活保存量 | 为反向保留的中间结果 | 反向无法计算 | 显存峰值过高 |
| 通信带宽 | stage 间传输激活/梯度的能力 | GPU 等通信 | 算力被带宽拖慢 |

玩具例子最容易理解。假设有 4 张 GPU，每张负责模型的 1/4。

- 如果整个 mini-batch 不切分，GPU1 先做完全部前向，GPU2 才开始，GPU3 和 GPU4 前面一直闲着。
- 如果切成 4 个 micro-batch，GPU1 做完 micro-batch 1 就把结果送给 GPU2，然后自己立刻处理 micro-batch 2。此时 GPU2 处理 micro-batch 1，GPU1 处理 micro-batch 2，流水线开始形成。

看一个典型数值对比：

| $p$ | $m$ | 近似效率 $\frac{m}{m+p-1}$ | 气泡率 | 显存峰值趋势 |
|---|---:|---:|---:|---|
| 4 | 1 | 25.0% | 75.0% | 低 |
| 4 | 4 | 57.1% | 42.9% | 中 |
| 4 | 8 | 72.7% | 27.3% | 中高 |
| 4 | 12 | 80.0% | 20.0% | 更高 |
| 8 | 8 | 53.3% | 46.7% | 高 |

这张表说明一个常见事实：stage 数增加以后，如果 micro-batch 数不跟着增加，气泡会重新变严重。也就是说，流水线不是“切得越细越好”，而是“切分、微批次、显存”三者一起平衡。

---

## 核心机制与推导

先看朴素流水线。假设每个 stage 的前向耗时接近相同，一个 mini-batch 被切成 $m$ 个 micro-batch，stage 数为 $p$。理想情况下，如果没有启动和收尾损失，总工作槽位是 $m+p-1$ 个时间单位，其中只有 $m$ 个时间单位真正对应连续产出，前后的 $p-1$ 个单位是填充和排空。

因此近似有：

$$
\text{Efficiency} \approx \frac{m}{m+p-1}
$$

进一步可写成：

$$
\text{Efficiency} \approx 1 - \frac{p-1}{m+p-1}
$$

所以：

$$
\text{Bubble Rate} \approx \frac{p-1}{m+p-1}
$$

代入题目要求的例子，$p=4, m=12$：

$$
\text{Efficiency} = \frac{12}{12+4-1}=\frac{12}{15}=0.8
$$

也就是效率约 80%，气泡约 20%。

这能解释为什么“micro-batch 数远大于 stage 数”很重要。因为 $m$ 越大，分母中的固定损失 $p-1$ 占比越小。

下面用一个简化时间线表示。横轴是时间片，纵轴是 stage。

朴素前向堆叠可以写成：

```text
时间 ->   1    2    3    4    5    6
S1       F1   F2   F3   F4   .    .
S2       .    F1   F2   F3   F4   .
S3       .    .    F1   F2   F3   F4
S4       .    .    .    F1   F2   F3
```

这里 `.` 就是气泡。越靠前和越靠后，空白越多。

GPipe 的问题是，虽然前向被填满了，但它通常采用“先大量前向，再大量反向”的方式，因此每个 stage 需要保留更多尚未反向的激活。`激活` 可以先理解为“前向中间结果，反向求梯度时还要用”。

1F1B 的做法是，当流水线预热后，每个 stage 尽量按“一次前向、一次反向”交替。例如：

```text
时间 ->   1    2    3    4    5    6    7    8
S1       F1   F2   F3   F4   B1   F5   B2   F6
S2       .    F1   F2   F3   B1   F4   B2   F5
S3       .    .    F1   F2   B1   F3   B2   F4
S4       .    .    .    F1   B1   F2   B2   F3
```

这不是严格实现图，只是为了展示交替思想。要点是：某个 micro-batch 的前向走过后，不必等所有 micro-batch 的前向都结束，反向可以更早插入，于是激活保存时间缩短。

但这会引出一个新问题：**权重版本一致性**。因为前向时用的是某一版参数，若反向时参数已经被更新，就可能出现不一致。`weight stashing` 可以理解为“给不同 micro-batch 暂存它前向对应的参数版本”，这样反向时还能对上当时那一版。

激活检查点是另一个配套机制。它的逻辑是：

1. 前向时不保存所有中间激活，只保留少量检查点。
2. 反向时如果缺失某段中间结果，就重新跑一次那一段前向。
3. 用额外计算换显存。

因此它本质是在做下面这个交换：

| 资源 | 不做 checkpoint | 做 checkpoint |
|---|---|---|
| 显存 | 高 | 低 |
| 计算量 | 低 | 高 |
| 单步耗时 | 更短 | 更长 |
| 可支持的 $m$ | 更小 | 更大 |

真实工程里，经常不是单独使用其中一个，而是 `pipeline + 1F1B + activation checkpointing` 一起用。原因很直接：流水线负责提高吞吐，1F1B 降激活驻留时间，checkpoint 再压一次峰值显存，这样才能把 micro-batch 数推到足够大，让气泡真正降下去。

---

## 代码实现

先给一个可运行的玩具代码，用来计算不同 $m$ 下的效率和气泡率，并打印一条简单“曲线图”。它不依赖 GPU，但能把公式直观跑出来。

```python
def pipeline_efficiency(p: int, m: int) -> float:
    assert p >= 1 and m >= 1
    return m / (m + p - 1)

def bubble_rate(p: int, m: int) -> float:
    return 1.0 - pipeline_efficiency(p, m)

# 基本断言
assert abs(pipeline_efficiency(4, 12) - 0.8) < 1e-9
assert abs(bubble_rate(4, 12) - 0.2) < 1e-9
assert pipeline_efficiency(4, 8) > pipeline_efficiency(4, 4)
assert pipeline_efficiency(4, 32) > 0.9

def ascii_curve(p: int, max_m: int = 16) -> None:
    for m in range(1, max_m + 1):
        eff = pipeline_efficiency(p, m)
        bar = "#" * int(eff * 40)
        print(f"m={m:2d} eff={eff:.3f} {bar}")

if __name__ == "__main__":
    ascii_curve(4)
```

如果固定 $p=4$，随着 $m$ 增长，输出条形会越来越长，这就是“效率逼近 1”的直观图。

下面给一个更接近训练框架的伪代码。重点不是具体 API 名，而是理解流水线里的三类动作：`recv_tensor`、`forward`、`send_tensor`，以及反向里的 `recv_grad`、`backward`、`send_grad`。

```python
# 简化伪代码：每个 stage 只看自己这一段层
for step in range(num_steps):
    micro_batches = split(mini_batch, m)

    # warmup: 连续灌入前向
    for micro in warmup_window:
        x = recv_tensor(prev_stage) if not is_first_stage else micro.input
        y = local_forward(x)
        if not is_last_stage:
            send_tensor(next_stage, y)
        else:
            loss = compute_loss(y, micro.target)
        stash_activation(micro, y)

    # steady state: 1F1B
    for micro in steady_window:
        # 1F
        x = recv_tensor(prev_stage) if not is_first_stage else micro.input
        y = local_forward(x)
        if not is_last_stage:
            send_tensor(next_stage, y)

        # 1B
        if is_last_stage:
            loss = compute_loss(y, micro.target)
            grad = backward_from_loss(loss)
        else:
            grad = recv_grad(next_stage)

        grad_to_prev = local_backward(grad, micro)
        if not is_first_stage:
            send_grad(prev_stage, grad_to_prev)

        release_or_checkpoint(micro)

    optimizer_step()
```

如果用 DeepSpeed 这类框架，用户通常不会手写收发逻辑，而是声明模型切分和并行度，让引擎调度。例如概念上会有类似：

```python
engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config={
        "pipeline": {"seed_layers": True},
        "pipeline_parallel_degree": 4,
        "activation_checkpointing": {"partition_activations": True},
    },
)

for batch in dataloader:
    loss = engine.train_batch(data_iter=batch)
```

这里真正重要的不是背 API，而是知道框架帮你做了什么：

| 动作 | 含义 |
|---|---|
| 切 stage | 把连续层映射到不同 GPU |
| 切 micro-batch | 把 mini-batch 拆成流水线粒度 |
| 调度 1F1B | 交错前向和反向 |
| 处理权重版本 | 避免前后向参数不一致 |
| checkpoint | 以重算换显存 |

真实工程例子可以这样理解：假设一个 48 层 Transformer 被切成 4 个 stage，每个 stage 12 层。一个全局 mini-batch 再切成 12 个 micro-batch。GPU0 处理第 1 个 micro-batch 的第 1 到 12 层后，马上把激活传给 GPU1，同时开始处理第 2 个 micro-batch。等流水线进入稳定态后，4 张卡几乎都在持续执行前向或反向，而不是等整批结束。

---

## 工程权衡与常见坑

流水线并行最大的问题不是“能不能跑”，而是“跑起来以后是否真的更快、更省显存”。

先看最常见的坑：

| 问题 | 症状 | 缓解手段 |
|---|---|---|
| micro-batch 太少 | GPU 利用率低，阶段间等待明显 | 增加 $m$，同时用 checkpoint 控制显存 |
| micro-batch 太多 | 激活缓存上涨，调度开销增加 | 适度增大 gradient accumulation 或减少每卡 batch |
| stage 切分不均衡 | 某一张 GPU 总是最慢 | 重新分配层数，按实际耗时而不是按层数平均切 |
| 通信过慢 | GPU 算完后在等传输 | 减少跨 stage 张量尺寸，提升带宽，优化拓扑 |
| 1F1B 权重冲突 | loss 抖动、收敛异常 | 使用 weight stashing 或框架内建版本管理 |
| checkpoint 过重 | 步时增加过多 | 只对最耗显存的块做 checkpoint |

一个新手容易误解的点是：`m` 不是越大越好。因为每个 micro-batch 都会产生调度和缓存成本。尤其当单个 micro-batch 太小，矩阵乘法尺寸也会变差，GPU 核心未必能吃满。所以实际最优点通常不是理论上最大的 $m$，而是一个兼顾气泡、显存和 kernel 效率的区间。

再看显存权衡。纯 GPipe 前向全灌完再做反向时，需要保存大量尚未反向的激活。1F1B 可以更早回收一部分激活，checkpoint 又能进一步下降峰值。但 checkpoint 不是白送的，它通常意味着反向阶段要重算部分前向，工程里经常能看到大约 20% 到 30% 的额外计算开销，具体取决于切分粒度和重算范围。

真实工程例子里，大模型训练经常会把这些手段叠加使用。例如在千亿参数级 Transformer 训练中，单纯依靠数据并行已经不够，因为每卡放不下完整权重和激活；单纯做张量并行又会带来高频 All-Reduce。此时流水线并行把模型沿层切开，再用 1F1B 和多版本权重保持训练一致性，常见结果是显存压力显著下降，从而允许把 micro-batch 数继续做大，进一步压缩气泡。你在一些 DeepSpeed PipeDream 资料里会看到显存下降 37.5% 这一类案例，本质上就是这种组合策略带来的空间收益。

另一个常见坑是**以层数平均切分**。这通常不对。因为不同层的计算量、激活大小、注意力开销都可能不同。比如 Transformer 中靠近 embedding、输出头或者某些特殊 block 的耗时并不一定等同于普通 block。工程上真正要平衡的是每个 stage 的**实际步时**，否则最慢 stage 会成为整条流水线的瓶颈。

---

## 替代方案与适用边界

流水线并行不是唯一方案，它只是大模型并行家族中的一种。

先做一个简化对比：

| 方案 | 参数怎么放 | 主要通信 | 显存压力 | 延迟 | 代码复杂度 |
|---|---|---|---|---|---|
| 数据并行 | 每卡一份完整模型 | 梯度同步 | 高 | 低 | 低 |
| 张量并行 | 单层内部切分到多卡 | 层内高频通信 | 中 | 中高 | 高 |
| 流水线并行 | 按层切到不同卡 | stage 间激活/梯度传输 | 低到中 | 高 | 中高 |
| 专家并行 | 只激活部分专家 | 路由与专家通信 | 低到中 | 不稳定 | 高 |

适用边界可以概括成三条。

第一，模型必须能按层顺序切分，而且跨层依赖不能太复杂。Transformer 很适合，因为 block 结构重复、顺序明确。

第二，你追求的是**吞吐**，不是单样本最低时延。流水线需要预热和排空，对在线实时推理通常不友好。

第三，必须有足够的 micro-batch 可供摊销。如果 batch 非常小，流水线很难形成有效滑动。

看两个对比例子。

玩具例子：如果模型只有 2 层，batch 也只有 1，那么切成 2 个 stage 没太大意义。因为没有足够任务填满流水线，设备大多在等待。这时直接用数据并行或者单卡更简单。

真实工程例子：如果你训练的是多层 Transformer，单卡装不下，且训练目标是最大化 tokens/s，那么流水线并行通常是合理选择。它特别适合与数据并行、张量并行混合使用，例如“层间做流水线、层内做张量并行、数据侧再做复制”，这是很多大模型训练系统的标准组合。

因此，流水线并行最适合的不是“小模型提速”，而是“中大模型在显存受限条件下提高整体训练吞吐”。不适合的典型场景包括：

| 场景 | 原因 |
|---|---|
| 极小 batch 训练 | 无法摊销气泡 |
| 实时单请求推理 | 预热和传输增加尾延迟 |
| 层间依赖复杂模型 | 难以做干净切分 |
| 带宽很差的跨机环境 | 通信会吞掉理论收益 |

---

## 参考资料

1. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism  
说明了将 mini-batch 切为 micro-batch 的基本思想，以及流水线气泡与吞吐关系，支撑“核心机制与推导”和“替代方案”部分。

2. PipeDream: Generalized Pipeline Parallelism for DNN Training  
说明了 1F1B 调度、weight stashing 等机制，支撑“核心机制与推导”和“工程权衡与常见坑”部分。新手重点可看权重版本一致性问题。

3. DeepSpeed Pipeline Parallelism / PipeDream 相关文档  
提供工程化 API 和调度实现思路，支撑“代码实现”和“工程权衡与常见坑”部分。看官方文档可以直接对应训练循环和引擎调度。

4. PyTorch Activation Checkpointing 文档  
解释激活检查点如何以重算换显存，支撑“核心机制与推导”和“工程权衡与常见坑”部分。

5. 题目给定研究摘要中的两篇中文解析资料  
用于补充新手视角表述、p=4 且 m=12 的数值示例，以及 DeepSpeed PipeDream 工程案例的概念说明：
   - CSDN 文章：流水线并行、PipeDream、显存与微批次关系
   - TalksAI 文章：气泡率公式、1F1B 时间线理解
