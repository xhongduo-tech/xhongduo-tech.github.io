## 核心结论

Cosine 学习率调度是一种按余弦曲线逐步降低学习率的方法：训练开始后，学习率从峰值 $\eta_{\max}$ 平滑下降到最小值 $\eta_{\min}$，常见组合是 `warmup + cosine decay`。

它的核心价值不是公式更复杂，而是训练过程更平滑。相比 `step decay` 这种突然降低学习率的方式，cosine decay 会让学习率连续变化，减少训练曲线因为学习率突变而出现的抖动。对于大模型预训练这类长周期训练，平滑衰减通常更符合工程需求。

结论型公式如下：

$$
\eta(t): \eta_{\max} \rightarrow \eta_{\min}
$$

其中 $t$ 表示当前优化步数，$\eta(t)$ 表示第 $t$ 步使用的学习率。

常见做法是：

1. 前若干步使用 `warmup`，把学习率从 0 或很小的值线性升到 $\eta_{\max}$。
2. 后续使用 cosine decay，把学习率从 $\eta_{\max}$ 平滑降到 $\eta_{\min}$。
3. $\eta_{\min}$ 通常设置为 $\eta_{\max}$ 的一个比例，例如 $0.1 \cdot \eta_{\max}$。

新手版理解：如果把训练比作开车，`warmup` 是先慢慢踩油门，cosine decay 是后面逐渐松油门，而不是突然一脚踩死。这里的比喻只帮助理解节奏，真正的定义仍然是“按优化步数改变学习率”。

余弦衰减曲线可以简化表示为：

```text
learning rate
^
|        /\
|       /  \
|      /    \
|     /      \__
|    /          \__
|___/              \____
|
+------------------------> step
    warmup      cosine decay
```

真实工程里，GPT-3、LLaMA、PaLM 等大模型训练都采用过 `warmup + decay` 形式的学习率调度，其中 cosine decay 是主流选择之一。原因很直接：预训练步数长、总 token budget 通常可预估、后期需要稳定收敛。

---

## 问题定义与边界

本文讨论的是**按优化步数变化的学习率调度**。学习率调度器是控制学习率随训练进度变化的规则，不是优化器本身，也不是损失函数设计。

学习率是优化器每次更新参数时的步长。学习率太大，模型参数可能在最优区域附近来回跳；学习率太小，训练会变慢，甚至很难从早期状态走出来。学习率调度器的作用，就是让学习率在不同训练阶段使用不同数值。

需要先分清几个术语：

| 术语 | 白话解释 | 本文中的边界 |
|---|---|---|
| 学习率调度器 | 决定每一步学习率是多少的规则 | 只控制学习率，不直接计算梯度 |
| 优化器 | 根据梯度更新模型参数的方法 | 例如 SGD、Adam、AdamW |
| warmup | 训练初期逐步升高学习率 | 通常按优化器更新步计数 |
| decay | 训练中后期降低学习率 | 本文重点讨论 cosine decay |
| step decay | 每隔一段时间突然降低学习率 | 学习率曲线不连续 |
| cosine decay | 按余弦曲线连续降低学习率 | 学习率曲线更平滑 |

符号说明如下：

| 符号 | 含义 | 示例 |
|---|---|---|
| $t$ | 当前优化步 | 第 1000 次 `optimizer.step()` |
| $T_w$ | warmup 步数 | 2000 |
| $T$ | 总优化步数 | 10000 |
| $\eta_{\max}$ | 峰值学习率 | $3 \times 10^{-4}$ |
| $\eta_{\min}$ | 最小学习率 | $3 \times 10^{-5}$ |
| $r$ | 最小学习率比例 | $0.1$，即 $\eta_{\min}=0.1\eta_{\max}$ |

这里的“步”必须特别小心。它通常指优化器更新步，而不是读入一个 micro-batch 的次数。micro-batch 是显存受限时拆出来的小批次；如果用了梯度累积，多个 micro-batch 才对应一次真正的参数更新。

Cosine 调度主要适合以下场景：

| 场景 | 是否适合 cosine decay | 原因 |
|---|---:|---|
| LLM 预训练 | 适合 | 总步数通常可根据 token budget 预估 |
| 长周期视觉模型训练 | 适合 | 后期需要平滑细调 |
| 快速小实验 | 不一定 | 调度器收益可能小于调参成本 |
| 总步数经常变动的训练 | 不太适合 | 曲线依赖总步数 $T$ |
| 在线训练或持续训练 | 需要改造 | 很难提前定义固定终点 |

新手版理解：你要先知道总共跑多少步，才能让学习率按计划平滑下降。如果训练步数经常变动，cosine schedule 就没那么好用。

---

## 核心机制与推导

标准 `warmup + cosine decay` 可以分成两段。

第一段是线性 warmup。线性是指每一步增加固定比例。训练刚开始时，模型参数通常是随机初始化的，梯度方向还不稳定。如果一开始就使用峰值学习率，参数更新可能过大，导致 loss 抖动甚至发散。因此 warmup 先把学习率从 0 平滑拉到 $\eta_{\max}$。

第二段是 cosine decay。余弦函数是一个平滑函数，本文使用它从 1 下降到 0，再映射到从 $\eta_{\max}$ 下降到 $\eta_{\min}$。

分段公式如下：

```text
t < T_w:
η(t) = η_max · t / T_w

t ≥ T_w:
η(t) = η_min + 0.5 (η_max - η_min) [1 + cos(π · (t - T_w) / (T - T_w))]
```

写成数学形式：

$$
\eta(t)=
\begin{cases}
\eta_{\max}\cdot \frac{t}{T_w}, & t < T_w \\\\
\eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left[1+\cos\left(\pi\cdot\frac{t-T_w}{T-T_w}\right)\right], & t \ge T_w
\end{cases}
$$

推导可以从边界条件看起。

当 $t=T_w$ 时：

$$
\frac{t-T_w}{T-T_w}=0
$$

所以：

$$
\cos(0)=1
$$

代入后：

$$
\eta(t)=\eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})(1+1)=\eta_{\max}
$$

也就是说，cosine decay 的起点正好接上 warmup 的终点。

当 $t=T$ 时：

$$
\frac{t-T_w}{T-T_w}=1
$$

所以：

$$
\cos(\pi)=-1
$$

代入后：

$$
\eta(t)=\eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})(1-1)=\eta_{\min}
$$

也就是说，训练结束时学习率正好降到最小值。

余弦函数还有一个关键性质：两端斜率接近 0。斜率是曲线在某一点的变化速度。对训练来说，这意味着刚进入衰减时不会突然掉学习率，接近结束时也不会急刹车。

曲线节奏可以理解为：

```text
learning rate
^
|             η_max
|              *
|            *   *
|          *       *
|        *           *
|      *               *
|    *                   *
|  *                       *__
|*                            ** η_min
+------------------------------------> step
0        T_w                 T
   warmup        cosine decay
```

玩具例子：设 $\eta_{\max}=3e-4$，$r=0.1$，所以 $\eta_{\min}=3e-5$，$T_w=2000$，$T=10000$。

| 步数 $t$ | 阶段 | 学习率 |
|---:|---|---:|
| 1000 | warmup | $1.5e-4$ |
| 2000 | warmup 结束 | $3.0e-4$ |
| 6000 | cosine 中点 | $1.65e-4$ |
| 10000 | 训练结束 | $3.0e-5$ |

在 $t=6000$ 时，cosine decay 的进度是：

$$
\frac{6000-2000}{10000-2000}=0.5
$$

此时：

$$
\cos(\pi \cdot 0.5)=0
$$

所以：

$$
\eta(6000)=3e-5+0.5\cdot(3e-4-3e-5)=1.65e-4
$$

真实工程例子：大模型预训练常用 `AdamW + warmup + cosine decay`。例如 LLaMA 使用 2000 步 warmup，并将最终学习率降到峰值的 10%。这类训练通常有明确的 token budget，例如计划训练 1T 或更多 token，因此可以提前把 token 数换算成优化步数，再设计 $T$ 和 $T_w$。

---

## 代码实现

代码实现的重点不是把公式写出来，而是正确接入训练循环。学习率调度器必须跟优化器更新步保持一致。

下面是一个可运行的 Python 实现，包含断言：

```python
import math

def cosine_lr(step, warmup_steps, total_steps, lr_max, lr_min_ratio=0.1):
    assert warmup_steps > 0
    assert total_steps > warmup_steps
    assert lr_max > 0
    assert 0 <= lr_min_ratio <= 1

    lr_min = lr_max * lr_min_ratio

    if step < warmup_steps:
        return lr_max * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)

    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * progress)
    )

lr_max = 3e-4
warmup_steps = 2000
total_steps = 10000

assert cosine_lr(0, warmup_steps, total_steps, lr_max) == 0
assert abs(cosine_lr(1000, warmup_steps, total_steps, lr_max) - 1.5e-4) < 1e-12
assert abs(cosine_lr(2000, warmup_steps, total_steps, lr_max) - 3e-4) < 1e-12
assert abs(cosine_lr(6000, warmup_steps, total_steps, lr_max) - 1.65e-4) < 1e-12
assert abs(cosine_lr(10000, warmup_steps, total_steps, lr_max) - 3e-5) < 1e-12
```

如果接入 PyTorch 训练循环，可以写成这种结构：

```python
for step, batch in enumerate(dataloader):
    loss = model(**batch).loss
    loss.backward()

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

在 PyTorch 中，一般应先执行：

```python
optimizer.step()
scheduler.step()
```

原因是调度器负责为下一次参数更新准备学习率。如果顺序写反，可能跳过初始学习率，尤其在使用内置 scheduler 时更容易踩坑。

如果不用 PyTorch 内置 scheduler，也可以手动设置学习率：

```python
for step, batch in enumerate(dataloader):
    loss = model(**batch).loss
    loss.backward()

    lr = cosine_lr(
        step=step,
        warmup_steps=2000,
        total_steps=10000,
        lr_max=3e-4,
        lr_min_ratio=0.1,
    )

    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.step()
    optimizer.zero_grad()
```

这段代码适合教学，但真实项目里更推荐使用框架内置 scheduler 或训练框架统一封装，避免多卡、梯度累积、恢复训练时出现状态不一致。

关键旁注：

| 字段 | 正确定义 | 常见错误 |
|---|---|---|
| `step` | 优化器更新次数 | 误用 micro-batch 次数 |
| `warmup_steps` | warmup 持续多少次参数更新 | 按 epoch 估算但未换算 |
| `total_steps` | 整个训练总优化步数 | 只写 dataloader 长度 |
| `lr_min_ratio` | 最小学习率占峰值学习率的比例 | 把 `0.1` 当成绝对学习率 |

如果使用梯度累积，例如每 8 个 micro-batch 更新一次参数，那么 scheduler 也应该每 8 个 micro-batch 调一次，而不是每个 micro-batch 都调一次。

---

## 工程权衡与常见坑

Cosine 调度依赖总步数 $T$，这是它的优势，也是它的限制。优势是训练计划明确时，曲线可以覆盖完整训练周期；限制是一旦 $T$ 填错，学习率曲线就会错位。

新手版理解：如果你把“总共要跑 10000 步”错写成 20000 步，那学习率曲线就会比训练进度慢一倍。训练结束时，模型可能还在较高学习率区，没有进入充分细调阶段。

常见坑如下：

| 坑 | 后果 | 规避方式 |
|---|---|---|
| $T$ 估小 | 学习率过早降到下限，后期训练变慢 | 用真实优化步数计算 `total_steps` |
| $T$ 估大 | 训练结束时学习率仍偏高，收敛不够稳定 | 先根据 token budget 或 epoch 数换算 |
| `scheduler.step()` 顺序错 | 初始学习率可能被跳过 | 通常先 `optimizer.step()` 再 `scheduler.step()` |
| warmup 计数混乱 | 实际 warmup 长度与预期不一致 | 统一使用优化器更新步 |
| 把 $\eta_{\min}$ 理解错 | 最小学习率大到离谱或小到接近 0 | 明确 `lr_min_ratio` 是比例还是绝对值 |
| 只抄超参 | 换任务后训练不稳定 | 重新搜索 $\eta_{\max}$、$T_w$ 和 batch size |

训练前必须确认 4 个字段：

| 检查项 | 必须回答的问题 |
|---|---|
| 总优化步数 $T$ | 一共会执行多少次 `optimizer.step()` |
| warmup 步数 $T_w$ | 前多少次更新用于线性升学习率 |
| 峰值学习率 $\eta_{\max}$ | warmup 后最高学习率是多少 |
| 最小学习率 $\eta_{\min}$ | 训练末尾降到多少，或占峰值多少比例 |

真实工程中还要注意恢复训练。假设训练在第 50000 步中断，恢复时 scheduler 也必须从第 50000 步继续。如果只恢复了模型权重和优化器状态，却没有恢复 scheduler 状态，学习率会回到错误位置，训练曲线会出现突变。

另一个常见问题是 batch size 变化。batch size 变大后，每一步看到的数据更多，梯度噪声通常更小，此时峰值学习率可能需要调整。模型规模变大、优化器从 SGD 换成 AdamW、数据分布变化，也都可能让原来的 $\eta_{\max}$ 和 $T_w$ 不再合适。

Cosine decay 不是自动提升效果的开关。它只能提供一个更平滑的学习率变化形状，不能替代学习率搜索、数据清洗、loss 检查和训练稳定性诊断。

---

## 替代方案与适用边界

Cosine decay 不是唯一选择。调度器选择要看训练目标、训练时长、总步数是否固定，以及你是否愿意为平滑收敛付出额外配置成本。

对比几种常见策略：

| 调度方式 | 机制 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| cosine decay | 按余弦曲线平滑下降 | 平滑，适合长训练 | 依赖总步数 $T$ | LLM 预训练、长周期训练 |
| step decay | 到指定步数突然降低 | 简单，易解释 | 学习率突变 | 传统 CNN 训练、固定 milestone 实验 |
| linear decay | 线性下降到最小值 | 实现简单，行为直观 | 中后期不如 cosine 柔和 | NLP 微调、短周期训练 |
| constant LR | 学习率不变 | 最简单 | 后期可能不够稳 | 快速实验、小模型 baseline |

适用边界可以进一步拆开看：

| 训练条件 | 推荐程度 | 原因 |
|---|---:|---|
| 长训练 | 高 | 平滑衰减能覆盖完整训练过程 |
| 短训练 | 中 | 收益不一定明显 |
| 步数可预估 | 高 | cosine 依赖总步数 |
| 步数变化大 | 低 | 曲线终点难定义 |
| 后期需要稳定细调 | 高 | 末尾学习率缓慢接近下限 |
| 只做快速验证 | 中低 | constant 或 linear 更省事 |

新手版理解：不是所有任务都需要“弧线下降”。有些训练像短跑，用简单的线性或阶梯式就够了。cosine 更适合长跑，尤其是你知道终点在哪里，并且希望接近终点时动作更小、更稳。

与 linear decay 相比，cosine decay 的中期下降更有弹性。linear decay 每一步下降速度固定，而 cosine decay 在起点和终点更慢，中间阶段下降更快。这种形状通常更符合长训练节奏：刚从 warmup 出来时不要立刻大幅降低学习率，中间逐渐降低探索幅度，最后用更小步长细调参数。

与 step decay 相比，cosine decay 最大的优势是连续。step decay 在某个 milestone 把学习率乘以一个系数，例如从 $3e-4$ 突然变成 $3e-5$。这种突变有时有效，但也可能让 loss 曲线出现明显拐点。cosine decay 不会在单个步数上突然改变学习率，因此训练日志通常更平滑。

工程选择可以用一句话概括：如果训练周期长、总步数明确、目标是稳定收敛，优先考虑 `warmup + cosine decay`；如果只是快速跑一个 baseline，constant LR 或 linear decay 可能更直接。

---

## 参考资料

论文和框架文档要分开看：论文说明方法来源和大模型训练配置，框架文档说明实现细节和公式定义。只想确认公式时，优先看 PyTorch 文档；想理解为什么大模型训练常用这种方法时，再看 SGDR、GPT-3、LLaMA 等资料。

| 来源类型 | 用途 |
|---|---|
| 理论来源 | 理解 cosine annealing 的思想 |
| 框架文档 | 确认 API 行为、参数含义和调用方式 |
| 工程案例 | 了解真实大模型训练如何设置 warmup 和最终学习率 |
| 训练论文 | 对比不同模型规模下的 schedule 选择 |

1. [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
2. [PyTorch CosineAnnealingLR 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
3. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
4. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
5. [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
