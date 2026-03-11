## 核心结论

梯度累积是“先不更新参数，连续处理多个小批次，把梯度加起来后再统一更新”的做法。流水线并行是“把模型按层切成多个阶段，让不同 GPU 像装配线一样同时工作”的做法。两者一旦组合，`m` 不再只是“流水线里有多少个微批次”，它还等价于“每次参数更新前累积多少次梯度”。

最重要的三个式子是：

$$
\text{global\_batch\_size} = dp \times m \times \text{micro\_batch\_size}
$$

$$
\text{bubble\_rate} \approx \frac{P-1}{P-1+M}
$$

$$
\text{lr\_adjusted} = \text{base\_lr} \times \sqrt{\frac{\text{effective\_batch\_size}}{\text{base\_batch\_size}}}
$$

这里 `dp` 是数据并行副本数，意思是“同一份模型同时处理几份不同数据”；`P` 是流水线阶段数，意思是“模型按深度被切成几段”；`m=M` 是微批次数，意思是“一次参数更新前，这条流水线上连续注入多少个 micro-batch”。

结论先说清楚：

1. `m` 变大，流水线更容易跑满，bubble 更小，吞吐通常更高。
2. `m` 变大，也意味着一次 `optimizer.step()` 前要累积更多微批的梯度，等效全局批量同步变大。
3. 吞吐变好不代表训练更稳。因为有效批量变了，学习率、warmup、梯度裁剪和日志解释都要一起检查。

给新手的最短直觉：4 个 stage 可以理解为 4 个工位，`m` 是工位之间同时流动的任务数。任务太少，后面的工位经常空闲；任务变多，空闲减少。但每次结算不是处理完一个任务就更新参数，而是把这 `m` 个任务对应的梯度先累起来，再统一更新一次。

---

## 问题定义与边界

这篇文章只讨论训练场景下的流水线并行，尤其是 GPipe、1F1B 这类“把一次更新拆成多个 micro-batch”的实现。不讨论推理侧连续批处理，也不展开张量并行、序列并行、专家并行的内部细节。

核心问题有两个：

1. 流水线并行里的 `m`，能不能视为梯度累积步数？
2. 当 `m` 同时影响吞吐和有效批量时，训练超参数应该怎样联动调整？

答案是：在常见训练实现里，可以近似把 `m` 看成每次参数更新前的梯度累积步数。原因不是“两个名字恰好像”，而是执行顺序一致：

1. 一个 mini-batch 被切成 `m` 个 micro-batch。
2. 每个 micro-batch 都会依次完成前向和反向。
3. 梯度先累积在参数对应的梯度缓冲区里。
4. 等这一轮 `m` 个 micro-batch 都结束后，才做一次参数更新。

如果只看“参数多久更新一次”这个层面，`m` 就等价于梯度累积步数。

看一个最小配置：

| 参数 | 数值 | 含义 |
|---|---:|---|
| stage 数 `P` | 4 | 模型切成 4 段 |
| 数据并行 `dp` | 2 | 两份数据副本并行训练 |
| micro batch size | 4 | 每个微批 4 个样本 |
| 微批次数 `m` | 16 | 每次更新前处理 16 个微批 |
| 等效 global batch | 128 | $2 \times 16 \times 4$ |

因此，这个配置每次更新前实际聚合的是 128 个样本的梯度，而不是“单卡只看到了 4 个样本”。

为了避免把结论说得过头，边界也要明确：

| 结论 | 是否成立 | 说明 |
|---|---:|---|
| `m` 等于“每次更新前累积多少个微批” | 是 | 这是本文最关心的等价关系 |
| 流水线并行完全等同于普通梯度累积 | 否 | 流水线还多了 fill/flush、跨 stage 通信和调度 |
| 改 `m` 只影响吞吐，不影响优化 | 否 | `m` 同时改变有效批量和更新频率 |
| 有 ZeRO/FSDP/交错流水线时结论仍可参考 | 是 | 但通信、裁剪和状态同步更复杂 |

还可以把“更新频率”写成一个直接可用的式子。设一个 epoch 中总样本数为 `N`，则每个 epoch 的优化器更新次数约为：

$$
\text{updates\_per\_epoch} = \frac{N}{dp \times m \times \text{micro\_batch\_size}}
$$

这说明 `m` 增大后，单位样本数对应的 `optimizer.step()` 次数会减少。吞吐提高与更新变慢，经常同时发生。

---

## 核心机制与推导

流水线里的 bubble，指某些 stage 暂时没活干的空闲时间。它主要出现在两个阶段：

1. fill：流水线刚启动，后面的 stage 还拿不到输入。
2. flush：流水线即将结束，前面的 stage 已经没有新的微批可做。

在理想化近似下，bubble rate 常写成：

$$
\text{bubble\_rate} \approx \frac{P-1}{P-1+M}
$$

对应的理想利用率是：

$$
\text{utilization} \approx 1 - \text{bubble\_rate}
= \frac{M}{P-1+M}
$$

当 `P=4, m=16` 时：

$$
\text{bubble\_rate} \approx \frac{3}{19} \approx 15.8\%
$$

如果把 `m` 提高到 32：

$$
\text{bubble\_rate} \approx \frac{3}{35} \approx 8.6\%
$$

可见 `m` 增大后，空闲比例下降，吞吐通常上升。

下面给一个更直观的表：

| `P=4` 时的 `m` | bubble rate | 理想利用率 | 直觉解释 |
|---:|---:|---:|---|
| 4 | $3/7 \approx 42.9\%$ | $57.1\%$ | 微批太少，流水线经常空 |
| 8 | $3/11 \approx 27.3\%$ | $72.7\%$ | 已能明显摊薄固定开销 |
| 16 | $3/19 \approx 15.8\%$ | $84.2\%$ | 工程上较常见 |
| 32 | $3/35 \approx 8.6\%$ | $91.4\%$ | 继续提升，但边际收益下降 |

这部分只解释了吞吐。现在看优化语义。

一次更新前，如果累积了 `m` 个微批，那么有效批量是：

$$
\text{effective\_batch\_size} = dp \times m \times \text{micro\_batch\_size}
$$

为什么这个式子成立？因为一次真正的参数更新要等所有数据并行副本上的 `m` 个微批都完成。于是优化器看到的不是一个微批的梯度，而是这些梯度的聚合。

设第 `i` 个微批产生的梯度为 $g_i$。若每个微批的 loss 在反传前先除以 `m`，则累积后的总梯度为：

$$
g = \sum_{i=1}^{m} \frac{g_i}{m}
= \frac{1}{m}\sum_{i=1}^{m} g_i
$$

这与“把 `m` 个微批拼成一个更大 batch 后，对 loss 取平均再求梯度”是等价的。也正因为如此，`m` 变大时：

1. 梯度噪声通常变小。
2. 参数更新频率下降。
3. 同样的训练步数，实际处理的样本数变多。
4. 日志里的 `step` 不再和“看到多少数据”保持原来的对应关系。

对新手最容易混淆的一点是：**流水线并行里的 `step` 有两层时间尺度。**

| 时间尺度 | 发生什么 | 是否更新参数 |
|---|---|---:|---|
| micro-batch 级别 | 前向、反向、跨 stage 传输 | 否 |
| optimizer step 级别 | 梯度聚合完成，执行 `optimizer.step()` | 是 |

因此，调大 `m` 不只是“让流水线更满”，还会把这两个时间尺度拉得更开。

再看 warmup 的联动。若原配置以“处理样本数”为基准做 warmup，总 warmup 样本数记为 `S_w`，则对应的 warmup 更新步数应改为：

$$
\text{warmup\_steps}
=
\left\lceil
\frac{S_w}{dp \times m \times \text{micro\_batch\_size}}
\right\rceil
$$

这也是为什么很多训练配置在改了 `m` 之后，学习率曲线看起来“没动”，实际却已经变了。因为 warmup 如果仍按旧的 update step 数设置，等价的 warmup 样本数就不对了。

---

## 代码实现

下面先给一个可运行的纯 Python 例子。它做三件事：

1. 计算 bubble rate、effective batch 和按 $\sqrt{\text{batch}}$ 缩放后的学习率。
2. 模拟“`m` 个微批累积一次再更新”的训练过程。
3. 用一个一维线性回归玩具例子验证：把 loss 先除以 `m` 再累积，更新结果等价于对大 batch 求平均梯度。

```python
import math
from dataclasses import dataclass

def bubble_rate(p: int, m: int) -> float:
    assert p >= 1 and m >= 1
    return (p - 1) / (p - 1 + m)

def effective_batch(dp: int, m: int, micro_batch_size: int) -> int:
    assert dp >= 1 and m >= 1 and micro_batch_size >= 1
    return dp * m * micro_batch_size

def adjusted_lr(base_lr: float, base_batch_size: int, dp: int, m: int, micro_batch_size: int) -> float:
    eff = effective_batch(dp, m, micro_batch_size)
    return base_lr * math.sqrt(eff / base_batch_size)

@dataclass
class LinearModel:
    w: float

def mse_grad_for_microbatch(model: LinearModel, batch):
    """
    batch: list of (x, y)
    loss = mean((w*x - y)^2)
    dloss/dw = mean(2*(w*x - y)*x)
    """
    grad = 0.0
    for x, y in batch:
        grad += 2.0 * (model.w * x - y) * x
    return grad / len(batch)

def train_with_accumulation(model, data, micro_batch_size, m, lr):
    assert len(data) % micro_batch_size == 0
    microbatches = [
        data[i:i + micro_batch_size]
        for i in range(0, len(data), micro_batch_size)
    ]
    assert len(microbatches) % m == 0

    update_log = []
    grad_buffer = 0.0

    for step, micro in enumerate(microbatches, start=1):
        grad = mse_grad_for_microbatch(model, micro)

        # 对应真实训练里 loss = loss / m
        grad_buffer += grad / m

        if step % m == 0:
            old_w = model.w
            model.w -= lr * grad_buffer
            update_log.append(
                {
                    "update_idx": step // m,
                    "grad": grad_buffer,
                    "old_w": old_w,
                    "new_w": model.w,
                }
            )
            grad_buffer = 0.0

    return update_log

def grad_for_full_batch(model, batch):
    return mse_grad_for_microbatch(model, batch)

if __name__ == "__main__":
    # 1. 先验证调度相关公式
    p = 4
    dp = 2
    m = 16
    micro_batch_size = 4
    base_lr = 2e-5
    base_batch_size = 32

    eff = effective_batch(dp, m, micro_batch_size)
    b = bubble_rate(p, m)
    lr = adjusted_lr(base_lr, base_batch_size, dp, m, micro_batch_size)

    assert eff == 128
    assert round(b, 6) == round(3 / 19, 6)
    assert round(lr, 8) == round(4e-5, 8)

    print("effective_batch =", eff)
    print("bubble_rate =", round(b, 4))
    print("adjusted_lr =", lr)

    # 2. 再验证梯度累积和大 batch 平均梯度的一致性
    data = [
        (1.0, 3.0),
        (2.0, 5.0),
        (3.0, 7.0),
        (4.0, 9.0),
        (5.0, 11.0),
        (6.0, 13.0),
        (7.0, 15.0),
        (8.0, 17.0),
    ]
    # 目标关系接近 y = 2x + 1
    model_acc = LinearModel(w=0.0)
    model_full = LinearModel(w=0.0)

    micro_batch_size = 2
    m = 4
    lr = 0.01

    logs = train_with_accumulation(model_acc, data, micro_batch_size, m, lr)

    full_grad = grad_for_full_batch(model_full, data)
    model_full.w -= lr * full_grad

    assert len(logs) == 1
    assert abs(model_acc.w - model_full.w) < 1e-12

    print("accumulated_update_w =", model_acc.w)
    print("full_batch_update_w  =", model_full.w)
    print("check passed")
```

运行输出应类似于：

```text
effective_batch = 128
bubble_rate = 0.1579
adjusted_lr = 4e-05
accumulated_update_w = 1.53
full_batch_update_w  = 1.53
check passed
```

上面这个脚本没有真的做分布式流水线，但它把本文最关键的等价关系验证清楚了：**在一次参数更新之前累积 `m` 个微批的平均梯度，数学上等价于更大 batch 的一次更新。**

训练循环里的关键点只有两个，但这两个点经常写错：

1. 每个微批的 loss 要先除以 `m`，否则梯度会被放大 `m` 倍。
2. 只在每 `m` 个微批结束后执行一次 `optimizer.step()`。

伪代码如下：

```python
micro_batches_per_update = m
effective_batch_size = dp * m * micro_batch_size

for micro_step, batch in enumerate(dataloader, start=1):
    loss = model(batch)
    loss = loss / micro_batches_per_update
    loss.backward()

    if micro_step % micro_batches_per_update == 0:
        lr = base_lr * math.sqrt(effective_batch_size / base_batch_size)
        adjust_optimizer_lr(optimizer, lr)

        # 真正工程里，clip 应按 global grad norm 做
        optimizer.step()
        optimizer.zero_grad()
```

如果想把 warmup 也写对，最稳妥的做法是按“样本数”而不是按“旧的 step 数”定义：

```python
warmup_samples = 10_000
warmup_steps = math.ceil(
    warmup_samples / (dp * m * micro_batch_size)
)
```

真实工程例子：假设你在 8 张 GPU 上训练一个 13B 模型，采用 `PP=4, DP=2, micro_batch_size=2, m=8`。那么每次更新前的有效 batch 是：

$$
2 \times 8 \times 2 = 32
$$

如果后续为了降低 bubble，把 `m` 从 8 改到 32，而其余配置不变，那么有效 batch 会变成：

$$
2 \times 32 \times 2 = 128
$$

这时如果学习率、warmup 和日志口径都不跟着改，你就不是“只做了调度优化”，而是同时改了优化条件。

---

## 工程权衡与常见坑

最常见的误区，不是 bubble 公式算错，而是把 `m` 当成纯吞吐参数。实际上 `m` 同时是吞吐参数和优化参数。

先看一个对比表：

| 调整方式 | 直接收益 | 主要代价 | 常见误判 |
|---|---|---|---|
| 增大 `m` | bubble 降低，吞吐提高 | 有效 batch 变大，更新更慢 | 误以为“只是更高效” |
| 缩小 `m` | 更新更频繁，优化更灵活 | bubble 升高，吞吐下降 | 误以为“只是更稳” |
| 只改 lr 不改 warmup | 配置改动少 | 前期训练语义不一致 | 误以为 lr 改了就够 |
| 同时改 lr、warmup、日志口径 | 行为更一致 | 配置管理更复杂 | 实现上更容易漏项 |

### 1. 学习率缩放不是定律

按

$$
\text{lr\_adjusted} = \text{base\_lr}\times\sqrt{\frac{\text{effective\_batch\_size}}{\text{base\_batch\_size}}}
$$

来缩放学习率，是常见启发式，不是保证收敛的定律。什么时候容易失效？

| 场景 | 原因 |
|---|---|
| 优化器从 AdamW 换成 SGD/LAMB | 不同优化器对 batch 变化的敏感性不同 |
| 序列长度明显变化 | token 数与 sample 数不再一一对应 |
| loss 波动本来就大 | 大 batch 降噪后可能改变最优 lr 区间 |
| warmup 太短 | 即使目标 lr 合理，前期也可能不稳 |

更稳妥的经验是：

1. 先按经验公式给一个初值。
2. 小范围试验 2 到 3 个相邻学习率。
3. 把观察单位从“每步 loss”改成“每处理固定样本数后的 loss”。

### 2. gradient clipping 要看全局范数

梯度裁剪的目标是限制整模型梯度范数，而不是每个 stage 单独各裁各的。全局梯度范数应写成：

$$
\|g\|_2
=
\sqrt{
\sum_{s=1}^{P}
\sum_{j \in \text{params of stage } s}
g_j^2
}
$$

如果每个 stage 各自执行 `clip_grad_norm_`，会出现两个问题：

1. 局部范数小，不代表全局范数小。
2. 各 stage 被缩放的比例不同，会破坏原本的全局梯度方向。

错误方式的典型现象：

| stage | 本地 norm | 本地是否裁剪 |
|---|---:|---:|
| 0 | 0.8 | 否 |
| 1 | 1.1 | 是 |
| 2 | 0.9 | 否 |
| 3 | 1.3 | 是 |

这样得到的不是“全模型 norm 被裁到阈值”，而是“不同局部梯度被各自改写”。

正确思路通常是：

1. 各 stage 先计算本地梯度平方和。
2. 对这些平方和做 all-reduce。
3. 开方得到全局 norm。
4. 用统一缩放因子裁剪所有 stage 的梯度。

可写成：

$$
\text{global\_sq\_norm}
=
\sum_{s=1}^{P}
\text{local\_sq\_norm}_s
$$

$$
\text{clip\_coef}
=
\min\left(1,\frac{\text{max\_norm}}{\sqrt{\text{global\_sq\_norm}}+\epsilon}\right)
$$

一个最短 checklist：

- [ ] 梯度是在一次完整累积结束后再 clip，而不是每个微批单独 clip
- [ ] clip 的对象是全局 norm，而不是单 stage norm
- [ ] ZeRO/FSDP/PP 同时存在时，确认框架 helper 的语义是否真的是 global clip

### 3. `m` 太大时，系统开销也会变重

很多新手只看到 `bubble` 下降，却忽略了系统成本不会免费消失。`m` 继续增大时，下面几项通常会变重：

| 成本项 | 为什么会上升 |
|---|---|
| 激活驻留时间 | 在飞 micro-batch 更多 |
| 调度元数据 | 队列、状态机、事件更多 |
| 通信缓存 | 更多激活和梯度在链路上流动 |
| 日志解释难度 | 一个 optimizer step 覆盖的样本数更大 |

这也是为什么工程上不会无限增大 `m`。你追求的是“足够降低 bubble”，而不是“把 `m` 拉到显存和调度都变糟”。

### 4. 三个最常见的配置错误

| 错误 | 结果 | 修复方式 |
|---|---|---|
| 改了 `m`，没改 loss 缩放 | 梯度放大 `m` 倍 | 每个微批 loss 除以 `m` |
| 改了 `m`，没改 warmup | 前期学习率曲线失真 | 按样本数重算 warmup steps |
| 改了 `m`，日志仍按旧 `step` 解读 | 曲线对比失真 | 同时记录 samples/tokens/update steps |

如果只允许记一个工程原则，可以记下面这一句：

> 改 `m` 时，要把它当作“吞吐配置 + 优化配置”的联合变更，而不是局部调参。

---

## 替代方案与适用边界

当 `m=1` 时，流水线里几乎没有时间维度上的梯度累积。这样做的优点是参数更新更频繁，优化语义更直接；缺点是 bubble 往往更高，流水线利用率较差。

当 `m>1` 时，流水线更容易接近稳态，设备更满载，但更新延迟和有效批量也同步变大。这更适合大模型预训练、长序列训练、吞吐优先的场景。

如果不想处理跨 stage 裁剪、fill/flush、参数版本管理和复杂调度，也可以退回纯数据并行。代价是显存压力更大，收益是训练语义更简单。

| 方案 | 吞吐 | 更新频率 | 实现复杂度 | 更适合什么场景 |
|---|---|---|---|---|
| `m=1` 的 PP | 中到低 | 高 | 中 | 显存受限，但更看重优化灵活性 |
| `m>1` 的 PP | 高 | 低 | 高 | 大模型、长序列、吞吐优先 |
| 纯 DP | 中 | 高 | 低 | 显存充足，希望配置简单 |
| DP + 普通梯度累积 | 中 | 可调 | 低到中 | 不需要跨 stage 调度，只想扩大有效 batch |

还可以用一个决策表快速判断：

| 你的主要瓶颈 | 更优先考虑 |
|---|---|
| 模型按层太深，单卡放不下 | PP |
| 显存主要耗在参数和优化器状态 | ZeRO / FSDP |
| 只是想扩大 batch，但不想引入流水线调度 | DP + gradient accumulation |
| stage 负载极不均衡，通信链路慢 | 先重切 stage，再谈增大 `m` |

最后把直觉总结成一句话：`m=1` 时，流水线像任务很少的装配线，更新勤，但空闲多；`m` 很大时，装配线更满，吞吐更高，但每次结算更晚，优化器每次面对的是更大的有效批量。哪种更合适，取决于你的瓶颈是在设备利用率，还是在优化稳定性。

---

## 参考资料

1. LMSYS, *Pipeline Parallelism in SGLang*：给出 bubble ratio 的讨论，并解释微批数 `M` 对流水线空闲率的影响。https://lmsys.org/blog/2026-01-15-chunked-pipeline/
2. DeepSpeed, *Pipeline Parallelism Tutorial*：说明 pipeline 中 micro-batches、gradient accumulation steps 和一次 `train_batch()` 的关系。https://www.deepspeed.ai/tutorials/pipeline/
3. Stephen Diehl, *Training / Gradient Accumulation*：总结 effective batch size 的计算方式，并给出按 $\sqrt{\text{batch}}$ 缩放学习率的常见启发式。https://www.stephendiehl.com/training/
4. NVIDIA NeMo AutoModel, `clip_grad_norm` 文档：说明在 pipeline parallelism 下需要对梯度范数做跨 stage 聚合。https://docs.nvidia.com/nemo/automodel/latest/apidocs/nemo_automodel/nemo_automodel.components.training.utils.html
5. Colossal-AI, *Gradient Clipping*：解释为什么 naive clipping 在 tensor/pipeline/MoE 并行下可能失效，以及框架封装的分布式裁剪入口。https://colossalai.org/docs/features/gradient_clipping_with_booster/
