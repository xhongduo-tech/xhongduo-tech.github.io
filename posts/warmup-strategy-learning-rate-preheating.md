## 核心结论

Warmup 是训练初期的学习率预热策略：在前若干次真实参数更新中，把学习率 `LR` 从接近 0 平滑增加到目标学习率 $\eta_{\max}$，用来降低早期训练发散、loss spike 和梯度爆炸的概率。

它的目标不是“让模型更快收敛”，而是“让训练先稳定进入正常状态”。模型刚随机初始化时，参数还没有找到大致方向，梯度方向噪声较大，优化器内部统计量也不稳定。新手版理解是：刚上路就猛踩油门，车容易冲出轨道；warmup 是先低速起步，再正常加速。

Warmup 本质上是一个过渡策略。训练过程通常分为两段：

| 阶段 | 学习率行为 | 主要目标 |
|---|---:|---|
| warmup 阶段 | 从接近 0 增加到 $\eta_{\max}$ | 稳住早期参数更新 |
| 主训练阶段 | constant、linear decay、cosine decay 等 | 按主调度器完成训练 |

`warmup_ratio` 表示 warmup 步数占总训练更新步数的比例。例如总共 100000 次 `optimizer.step()`，`warmup_ratio = 0.03`，则 warmup 步数是 3000。直观上，前 3% 的训练不是直接使用完整学习率，而是逐步抬高学习率，让 loss、梯度范数和 Adam 的统计量先进入相对稳定的区间。

经验起点通常是总训练步数的 `1% - 5%`。这个范围不是定律，而是工程上常用的初始搜索区间。

---

## 问题定义与边界

Warmup 主要针对“优化过程早期不稳定”，不是模型结构缺陷的修复工具。

统一符号如下：

| 符号 | 含义 |
|---|---|
| $t$ | 全局 step，通常指第几次真实参数更新 |
| $T_w$ | warmup 步数 |
| $\eta_{\max}$ | warmup 结束后达到的目标学习率 |
| $\eta_t$ | 第 $t$ 步使用的学习率 |

训练初期不稳定不只是“学习率太大”一个原因。常见来源包括：

| 来源 | 白话解释 | warmup 是否直接缓解 |
|---|---|---|
| 梯度方向噪声大 | 模型还没形成有效表示，早期梯度容易抖 | 能缓解更新过猛 |
| Adam 动量估计不稳定 | 一阶矩 $m_t$ 还没有统计出稳定方向 | 能缓解 |
| Adam 二阶矩估计不稳定 | 二阶矩 $v_t$ 还没稳定估计梯度尺度 | 能缓解 |
| 混合精度数值脆弱 | FP16/BF16 下极端值更容易造成溢出或 loss spike | 能部分缓解 |
| 数据标签错误 | 训练信号本身有问题 | 不能解决 |
| 模型结构不匹配任务 | 模型容量、归一化、连接方式不合适 | 不能解决 |
| 初始化严重错误 | 参数初值范围明显不合理 | 只能减轻，不能根治 |

新手版理解：Adam 刚开始时，`m_t` 和 `v_t` 都还在“学习怎么统计梯度”。如果这时 `LR` 很大，参数更新会被不稳定估计放大，训练可能直接崩掉。Warmup 做的是把前几步的更新幅度压小，让优化器有时间获得更可靠的梯度统计。

问题边界要明确：如果训练一开始就出现 `NaN`，warmup 可能有帮助；但如果数据预处理错了、标签错位、loss 写反、模型输出维度不匹配，warmup 不会把错误训练成正确结果。

---

## 核心机制与推导

线性 warmup 是最常见形式。它把学习率从 0 平滑升到目标值：

$$
\eta_t = \eta_{\max}\cdot \frac{t}{T_w},\quad 0 < t \le T_w
$$

在代码里如果 step 从 0 开始计数，常写成：

$$
\eta_t = \eta_{\max}\cdot \frac{t+1}{T_w},\quad 0 \le t < T_w
$$

warmup 结束后，学习率交给主调度器。常数 warmup 变体可以写成：

$$
\eta_t =
\begin{cases}
\eta_{\max}\cdot \frac{t}{T_w}, & 0 < t \le T_w \\
\eta_{\max}, & t > T_w
\end{cases}
$$

玩具例子：设 $\eta_{\max}=3e-4$，$T_w=4$，使用线性 warmup。

| step | learning rate |
|---:|---:|
| 1 | 0.000075 |
| 2 | 0.000150 |
| 3 | 0.000225 |
| 4 | 0.000300 |
| 5 | 由主调度器决定 |

这相当于前 4 次参数更新逐步把学习率升满，而不是第一步就使用 `3e-4`。

Adam 类优化器的背景更关键。Adam 是一种自适应优化器，会维护梯度的一阶矩和二阶矩。一阶矩可以理解为“梯度方向的滑动平均”，二阶矩可以理解为“梯度平方大小的滑动平均”。

Adam 的典型更新公式是：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

其中 $g_t$ 是当前梯度，$m_t$ 是一阶矩，$v_t$ 是二阶矩。由于 $m_0=0, v_0=0$，早期估计会偏向 0，所以 Adam 使用 bias correction，也就是偏差修正：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

最终参数更新近似为：

$$
\theta_{t+1}=\theta_t-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

关键在于分母里的 $\sqrt{\hat v_t}$。如果早期 $\hat v_t$ 对梯度尺度估计不稳定，某些参数的实际步长可能异常大。Warmup 通过压低 $\eta_t$，把整体更新幅度先乘上一个较小系数，给 $\hat v_t$ 留出稳定窗口。它不改变 Adam 的数学定义，而是降低早期统计量尚未可靠时的破坏性。

真实工程例子：训练 Transformer 或 LLM 时，常见组合是 `AdamW + warmup + cosine decay` 或 `AdamW + warmup + linear decay`。在大 batch、长序列、混合精度训练中，前几百到几千步经常是最脆弱阶段。Warmup 能降低 loss spike 和梯度范数突然升高的风险，让训练更稳定地进入主阶段。

---

## 代码实现

实现 warmup 时，最重要的是以 `optimizer.step()` 的真实更新次数为准，而不是按 epoch 或 batch 数粗略估算。

最小伪代码如下：

```text
for each training batch:
    loss.backward()

    if should_update_parameters:
        lr = lr_at_step(global_update_step)
        set_optimizer_lr(lr)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_update_step += 1
```

可运行的 Python 版本：

```python
def lr_at_step(step, warmup_steps, max_lr):
    if warmup_steps <= 0:
        return max_lr
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    return max_lr


assert abs(lr_at_step(0, 4, 3e-4) - 7.5e-5) < 1e-12
assert abs(lr_at_step(1, 4, 3e-4) - 1.5e-4) < 1e-12
assert abs(lr_at_step(2, 4, 3e-4) - 2.25e-4) < 1e-12
assert abs(lr_at_step(3, 4, 3e-4) - 3e-4) < 1e-12
assert abs(lr_at_step(10, 4, 3e-4) - 3e-4) < 1e-12
```

新手版理解：如果设置 warmup 为 1000 steps，就代表前 1000 次参数更新都在逐步升高学习率；不是训练了 1000 个 batch，也不是跑了 1000 个 epoch。使用梯度累积时，多个 batch 才对应一次真实参数更新。

常见 scheduler 对比如下：

| scheduler | warmup 后行为 | 适用场景 |
|---|---|---|
| `linear` | 从 $\eta_{\max}$ 线性衰减到 0 | BERT 类微调、总步数明确的训练 |
| `constant_with_warmup` | 保持 $\eta_{\max}$ 不变 | 简单任务、想减少变量时 |
| `cosine_with_warmup` | 按余弦曲线衰减 | LLM、视觉模型、大规模预训练常见 |

PyTorch/Hugging Face 典型写法如下：

```python
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

num_training_steps = 100_000
num_warmup_steps = int(num_training_steps * 0.03)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

这里的 `scheduler.step()` 应该跟随真实参数更新。如果使用 gradient accumulation，需要只在执行 `optimizer.step()` 的时候推进 scheduler。

---

## 工程权衡与常见坑

Warmup 过短，可能压不住早期不稳定；warmup 过长，又会拖慢有效训练进度，尤其是总步数不多时影响更明显。一个实用起点是 `1% - 5%`，然后根据前期 loss 曲线、gradient norm 和是否出现 NaN 调整。

常见坑如下：

| 错误做法 | 后果 | 规避方式 |
|---|---|---|
| 把 `warmup_ratio` 当成 epoch 比例 | warmup 实际步数和预期不一致 | 按总 `optimizer.step()` 次数计算 |
| 梯度累积没算进去 | warmup 可能变长或变短 | 用真实参数更新次数定义 step |
| 分布式训练下各进程口径不一致 | scheduler 状态不同步，学习率错乱 | 每个进程使用相同 global update step |
| 断点续训时 scheduler 状态丢失 | warmup 重来或直接跳过 | 保存并恢复 optimizer/scheduler state |
| warmup 过长 | 有效高学习率训练时间减少，收敛变慢 | 从 `1% - 5%` 起步，看曲线调整 |
| warmup 过短 | 早期 loss spike、grad norm 激增 | 增加 warmup steps 或降低 $\eta_{\max}$ |
| 只看 batch 数不看更新数 | 使用 accumulation 后估算错误 | 统一以 `optimizer.step()` 为准 |

新手版理解：你以为 warmup 是 1000 步，但如果用了梯度累积，比如每 4 个 batch 才更新一次参数，那么 1000 个 batch 只对应 250 次真实更新。此时学习率可能还没升完，训练阶段已经被你误判了。

还有一个容易混淆的点：warmup 不是越长越安全。它确实降低了早期风险，但也降低了前期有效学习率。如果任务很短，例如只有几百个更新步，设置 10% 甚至 20% warmup 可能明显拖慢训练。

工程上更可靠的调参顺序是：先确定合理的目标学习率 $\eta_{\max}$，再设置 `warmup_ratio = 0.01 ~ 0.05`，观察前期 loss 是否平滑下降、grad norm 是否存在尖峰。如果仍然不稳定，再考虑增加 warmup、降低学习率、开启梯度裁剪或检查数据与 loss 实现。

---

## 替代方案与适用边界

Warmup 不是唯一选择。它解决的是“如何安全进入正常训练”，不是所有训练稳定性问题的总开关。

替代方案对比如下：

| 方案 | 适用条件 | 代价 |
|---|---|---|
| 降低初始学习率 | 小模型、短训练、简单微调 | 可能整体收敛变慢 |
| 梯度裁剪 | 偶发梯度爆炸、长序列训练 | 需要选择裁剪阈值 |
| 更保守初始化 | 自定义模型结构、训练一开始就不稳定 | 需要理解模型层尺度 |
| 只用 decay 不做 warmup | 训练本身很稳、学习率较小 | 大模型早期风险较高 |
| `constant_with_warmup` | 想保留简单主阶段学习率 | 后期可能不如 decay 精细 |
| 调整 AdamW 参数 | $\beta_1,\beta_2,\epsilon$ 不适配任务 | 搜索空间更大 |

新手版理解：如果训练本来就很稳、学习率很小、模型也不大，warmup 可能只是锦上添花；但如果是 Transformer、LLM、大 batch、长序列、混合精度训练，warmup 往往是基础配置。

不同规模下的经验判断：

| 场景 | warmup 必要性 |
|---|---|
| 小型 CNN、低学习率、短训练 | 可选，先看 loss 曲线 |
| BERT 类微调 | 常用，尤其是学习率偏大时 |
| Transformer 从头训练 | 通常需要 |
| LLM 预训练 | 基础配置 |
| 混合精度 + 大 batch | 强烈建议使用 |

需要明确的是，原始 BERT 和 Transformer 里的 warmup 后面通常接 decay，不是恒定不变。Transformer 原论文使用前期 warmup，之后按 step 衰减；BERT 官方实现使用线性 warmup 加线性 decay。如果想使用 `constant_with_warmup`，这是另一个调度器变体，不应误认为是 BERT 原始训练策略。

最终结论：warmup 是训练初期稳定性的工程工具。它适合放在 AdamW、Transformer、大 batch、混合精度等容易早期不稳定的训练中；如果训练问题来自数据、loss、模型结构或实现错误，应该先修正根因，再谈 warmup 调参。

---

## 参考资料

1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
2. [google-research/bert optimization.py](https://github.com/google-research/bert/blob/master/optimization.py)
3. [Hugging Face Transformers: Optimization and Schedules](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
4. [PyTorch AdamW Source](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py)
