## 核心结论

One-Cycle 策略是一种学习率调度方法。学习率调度，指训练过程中不是一直使用同一个学习率，而是按规则改变参数更新步长。One-Cycle 的核心不是简单地“把学习率调一调”，而是把整段训练压缩成一个单周期：学习率先升后降，动量反向变化，让模型先快速探索，再稳定收敛。

新手版解释是：训练前段先把“步子”迈大一点，让模型尽快走出糟糕区域；训练后段再把“步子”放小，让模型稳住并收敛。学习率高时动量低，是为了更敢探索；学习率低时动量高，是为了更稳地落地。

动量是优化器里累积历史更新方向的机制。它可以理解为“参数更新的惯性”：动量越大，更新越依赖过去方向；动量越小，当前梯度对方向的影响越强。

One-Cycle 的典型调度关系可以写成：

$$
lr: lr_{init} \rightarrow lr_{max} \rightarrow lr_{min}
$$

$$
momentum: m_{max} \rightarrow m_{base} \rightarrow m_{max}
$$

也就是学习率升高时，动量降低；学习率降低时，动量升高。

| 训练方式 | 学习率变化 | 动量变化 | 训练节奏 | 主要目标 |
|---|---:|---:|---|---|
| 固定学习率 | 基本不变 | 基本不变 | 稳定但可能慢 | 平滑收敛 |
| 普通衰减学习率 | 先固定，后降低 | 通常不变 | 先训练，再细调 | 降低后期震荡 |
| One-Cycle | 先升后降 | 与学习率反向 | 先探索，后收敛 | 更少迭代达到可用精度 |

One-Cycle 追求的是更少迭代、更快达到可用精度。论文把这种现象称为 super-convergence，即“超收敛”：在一些任务上，用明显更少的 epoch 达到接近甚至更好的精度。例如论文报告在 CIFAR-10 上可以用约 1/10 的训练轮数达到相近效果。但这不是对所有任务都必然成立的定律。

PyTorch 中最核心的用法是：

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
)
```

其中 `max_lr` 是周期内达到的最高学习率，`epochs * steps_per_epoch` 决定总调度步数。

---

## 问题定义与边界

One-Cycle 解决的问题是：在训练预算有限时，如何用较少 epoch 或 iteration 拿到尽量好的模型效果。iteration 是一次参数更新，通常对应一个 batch。batch 是从训练集中取出的一小批样本。

如果你只有 5 个小时训练时间，One-Cycle 的目标就是让模型在这 5 个小时里尽快学到能用的东西，而不是按部就班慢慢收敛。它不是为了让曲线看起来最平滑，而是为了在有限计算资源下更快得到强基线。

设总训练步数为 $T$，上升阶段比例为 `pct_start = p`。那么：

$$
T_{up} = pT,\quad T_{down} = (1-p)T
$$

如果 `pct_start=0.3`，表示前 30% batch 用来把学习率从初始值升到最大值，后 70% batch 用来把学习率降到很小。

One-Cycle 的一个重要边界是：它主要是 batch 级调度，不是 epoch 级调度。也就是说，每处理完一个 batch，都应该更新一次 scheduler。

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        scheduler.step()  # 正确：每个 batch 后调用
```

| 场景类型 | 是否适合 One-Cycle | 原因 |
|---|---|---|
| CIFAR-10、Imagenette、小中型图像分类 | 适合 | 快速实验、训练成本低、指标反馈快 |
| 快速比较模型结构、数据增强、损失函数 | 适合 | 能较快得到可用基线 |
| 超长训练的大模型任务 | 需要谨慎 | 最优训练曲线可能需要更复杂 warmup 和衰减 |
| 强稳定性要求的任务 | 需要谨慎 | 学习率峰值过高可能造成震荡 |
| 复杂多目标优化任务 | 不宜默认套用 | 不同 loss 的尺度和收敛速度可能不一致 |

玩具例子：训练一个只有两层的 MLP 去分类二维点。如果固定学习率很小，模型可能慢慢挪动，几十个 epoch 才能把边界调好。One-Cycle 会先把学习率推高，让分类边界快速移动到大致正确的位置，再把学习率压低，让边界细调到更稳定的位置。

真实工程例子：在 CIFAR-10 或 Imagenette 上测试一个新的卷积网络结构时，团队可能不想先花 100 个 epoch 做完整训练。可以用 One-Cycle 跑 5 到 10 个 epoch，看模型是否具备基本潜力。如果短训效果明显差，通常没有必要立即投入长训资源。

---

## 核心机制与推导

One-Cycle 的关键机制是两阶段或三阶段的学习率调度。调度通常使用余弦退火。余弦退火是一种平滑变化曲线，用余弦函数让学习率从一个值平滑过渡到另一个值，避免突然跳变。

最常见的两阶段结构是：

| 阶段 | 学习率变化 | 动量变化 | 作用 |
|---|---|---|---|
| 上升阶段 | $lr_{init} \rightarrow lr_{max}$ | $m_{max} \rightarrow m_{base}$ | 提高探索能力，快速离开差区域 |
| 下降阶段 | $lr_{max} \rightarrow lr_{min}$ | $m_{base} \rightarrow m_{max}$ | 降低更新幅度，稳定收敛 |
| 可选第三阶段 | $lr_{init} \rightarrow lr_{min}$ | 通常维持或回升 | 更贴近原论文形式，做更深衰减 |

新手版解释：前 30% 步数像“加速跑”，让参数快速摆脱局部坏点；后 70% 步数像“刹车入库”，把更新幅度压小，让模型停在更好的位置。这里的“局部坏点”指当前参数附近 loss 不低、但普通小学习率很难快速离开的区域。

核心学习率公式是：

$$
lr_{init} = \frac{max\_lr}{div\_factor}
$$

$$
lr_{max} = max\_lr
$$

$$
lr_{min} = \frac{lr_{init}}{final\_div\_factor}
$$

动量调度是：

$$
mom: max\_momentum \rightarrow base\_momentum \rightarrow max\_momentum
$$

数值版解释：如果 `max_lr=1e-2`、`div_factor=25`、`final_div_factor=1e5`，那么：

$$
lr_{init} = \frac{10^{-2}}{25} = 4 \times 10^{-4}
$$

$$
lr_{min} = \frac{4 \times 10^{-4}}{10^5} = 4 \times 10^{-9}
$$

这说明训练开始和结束的步长差异很大。前段让模型大胆移动，后段让参数更新几乎停下来，进入细粒度收敛。

PyTorch 的 `three_phase` 控制调度形态：

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    epochs=10,
    steps_per_epoch=500,
    pct_start=0.3,
    div_factor=25,
    final_div_factor=1e5,
    base_momentum=0.85,
    max_momentum=0.95,
    three_phase=False,  # 默认更接近 fastai 的两阶段风格
)
```

`three_phase=False` 时，学习率通常从初始值升到最大值，再直接降到最低值。`three_phase=True` 时，会多一个中间阶段，更接近原始论文里的三段结构：先升到最大值，再降回初始值，最后进一步降到最低值。

---

## 代码实现

实现 One-Cycle 时，最重要的是把 scheduler 接到训练循环的 batch 位置，并且让 `optimizer.step()` 先于 `scheduler.step()`。优化器负责根据梯度更新参数，调度器负责更新下一步使用的学习率和动量。

新手版解释：每处理完一个 batch，就让调度器把“下一步该多大步子、该多稳”算出来；不要等一个 epoch 结束再调。

```python
import math

def one_cycle_values(step, total_steps, max_lr=1e-2, div_factor=25, final_div_factor=1e5, pct_start=0.3):
    lr_init = max_lr / div_factor
    lr_min = lr_init / final_div_factor
    up_steps = int(total_steps * pct_start)

    if step <= up_steps:
        t = step / up_steps
        lr = lr_init + (max_lr - lr_init) * (1 - math.cos(math.pi * t)) / 2
    else:
        t = (step - up_steps) / (total_steps - up_steps)
        lr = max_lr + (lr_min - max_lr) * (1 - math.cos(math.pi * t)) / 2

    return lr

total_steps = 100
lr0 = one_cycle_values(0, total_steps)
lr30 = one_cycle_values(30, total_steps)
lr100 = one_cycle_values(100, total_steps)

assert abs(lr0 - 4e-4) < 1e-12
assert abs(lr30 - 1e-2) < 1e-12
assert lr100 < 1e-8
assert lr30 > lr0 > lr100
```

上面是一个不依赖 PyTorch 的玩具实现，用来说明曲线方向。真实训练时应使用框架自带实现。

PyTorch 最小训练结构如下：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    div_factor=25,
    final_div_factor=1e5,
    base_momentum=0.85,
    max_momentum=0.95,
    three_phase=False,
)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

学习率方向可以概括为：

$$
early: lr_{init} \uparrow lr_{max},\quad late: lr_{max} \downarrow lr_{min}
$$

| 参数 | 含义 | 常见值或默认风格 | 常见误解 |
|---|---|---:|---|
| `max_lr` | 周期内最高学习率 | 需实验选择 | 不是随便设成常规 LR 的 10 倍 |
| `epochs` | 总 epoch 数 | 训练计划决定 | 与 `steps_per_epoch` 一起决定总步数 |
| `steps_per_epoch` | 每个 epoch 的 batch 数 | `len(train_loader)` | 不能漏填或填错 |
| `pct_start` | 学习率上升阶段比例 | `0.3` | 不是 warmup epoch 数 |
| `div_factor` | `max_lr / lr_init` | `25` | 越大初始 LR 越小 |
| `final_div_factor` | `lr_init / lr_min` | `1e4` 或 `1e5` | 越大最终 LR 越小 |
| `base_momentum` | 学习率最高时的较低动量 | `0.85` | 与最大动量相反使用 |
| `max_momentum` | 学习率较低时的较高动量 | `0.95` | 不是固定动量 |
| `three_phase` | 是否使用三阶段 | `False` | 默认不完全等同原论文 |

---

## 工程权衡与常见坑

One-Cycle 的优势是快，但快不等于无条件更好。它本质上把训练过程设计成“先探索、后收敛”的节奏，需要和任务、模型、正则化强度一起看。正则化是抑制过拟合的技术，例如 weight decay、dropout、数据增强。大学习率本身也有一定正则化效果，所以正则化强度不能盲目叠加。

如果你把 One-Cycle 当成普通学习率衰减器来用，结果可能不对。普通衰减器通常只负责“越训越小”，而 One-Cycle 要求训练曲线本身有明确的先升后降结构。

| 常见坑 | 错误做法 | 正确做法 |
|---|---|---|
| 调用频率错误 | 每个 epoch 调一次 `scheduler.step()` | 每个 batch 后调一次 |
| `max_lr` 过大 | 机械设成常规 LR 的 10 倍 | 先做 LR range test 或 `lr_find` |
| 忘记保存状态 | 断点续训只保存模型和优化器 | 同时保存 scheduler 状态 |
| 正则化过强 | 大 LR、强 weight decay、强 dropout 同时使用 | 观察验证集，逐项调参 |
| 忽略实现差异 | 默认 PyTorch 等同论文 | 根据需要检查 `three_phase` |
| batch 数变化 | 续训时改变 dataloader 长度 | 保持总步数定义一致 |

错误的 epoch 级调用：

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()

    scheduler.step()  # 错误：One-Cycle 的节奏被压成 epoch 级
```

正确的 batch 级调用：

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        scheduler.step()  # 正确：每次参数更新后调度一次
```

`max_lr` 的选择不应机械套公式。更稳妥的做法是做 LR range test，也就是从很小学习率开始逐步增大，观察 loss 从稳定下降到明显发散的区间，再选择合适的最高学习率。

可以把选择规则写成：

$$
max\_lr \lt lr_{diverge}
$$

其中 $lr_{diverge}$ 表示 loss 开始明显发散附近的学习率。实际选择通常要留出安全余量。

真实工程例子：一个团队在 Imagenette 上快速比较 ResNet-18 的数据增强方案。使用 One-Cycle 可以在 5 个 epoch 内得到比较有参考价值的排序。但如果目标是最终发布模型，仍然需要更长训练、更多随机种子、验证集稳定性检查，以及与 cosine、warmup cosine 等方案对比。

---

## 替代方案与适用边界

One-Cycle 不是唯一选择。对于稳定优先、长期训练、或需要更简单训练逻辑的任务，固定学习率、Cosine Annealing、Warmup + Cosine 都很常见。

Cosine Annealing 是余弦退火学习率，通常从较高学习率平滑降到较低学习率。Warmup 是预热阶段，指训练初期从小学习率逐步升高，避免一开始更新过猛。

新手版解释：One-Cycle 更像快速冲刺型训练计划，而余弦退火、更慢的 warmup 方案更像长跑型训练计划。前者适合快速得到强基线，后者适合更可控、更稳定的长时间训练。

One-Cycle 的轨迹可以简化表示为：

$$
lr_{onecycle}: low \rightarrow high \rightarrow very\ low
$$

Cosine Annealing 的轨迹可以简化表示为：

$$
lr_{cosine}: high \rightarrow low
$$

Warmup + Cosine 的轨迹可以简化表示为：

$$
lr_{warmup+cosine}: low \rightarrow high \rightarrow low
$$

它看起来和 One-Cycle 相似，但常见区别是：Warmup + Cosine 的上升阶段通常更短、更保守，最高学习率不一定追求超大，动量也不一定反向调度。

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 固定学习率 | 简单、容易排查问题 | 收敛可能慢，后期震荡 | 教学、玩具任务、小实验 |
| Step Decay | 实现简单，传统稳定 | 衰减点需要手动设 | 老代码库、经典 CNN 训练 |
| Cosine Annealing | 曲线平滑，长期训练常用 | 不强调快速探索 | 中长训练、稳定优先任务 |
| Warmup + Cosine | 初期稳定，后期平滑 | 参数更多 | Transformer、大模型、长训 |
| One-Cycle | 快速得到可用基线 | 对 `max_lr` 和 step 位置敏感 | 快速实验、有限算力、小中型分类 |

一个简版 Warmup + Cosine 替代方案如下：

```python
import math

def warmup_cosine_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine

assert warmup_cosine_lr(0, 100, 10, 1e-3, 1e-5) == 1e-4
assert warmup_cosine_lr(9, 100, 10, 1e-3, 1e-5) == 1e-3
assert warmup_cosine_lr(100, 100, 10, 1e-3, 1e-5) <= 1.1e-5
```

当你只想快速比较模型结构、数据增强、损失函数时，One-Cycle 很实用。当你需要超长训练和严格稳定的收敛轨迹时，可以优先考虑 Warmup + Cosine。当任务对指标波动非常敏感时，需要用验证集曲线、多个随机种子和断点续训测试来确认调度策略是否可靠。

---

## 参考资料

| 来源 | 类型 | 适合阅读目的 |
|---|---|---|
| Super-Convergence 论文 | 原始论文 | 理解 One-Cycle 与超收敛的提出背景 |
| PyTorch `OneCycleLR` 文档 | 框架实现 | 查参数含义、调用方式、实现差异 |
| fastai `fit_one_cycle` 文档 | 工程实践文档 | 理解 fastai 风格的训练接口 |

1. [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
2. [PyTorch OneCycleLR 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
3. [fastai fit_one_cycle 官方文档](https://docs.fast.ai/callback.schedule.html)
4. [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
